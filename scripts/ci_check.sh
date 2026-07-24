#!/usr/bin/env bash
# CI 检查总入口(合入门槛的唯一入口)。
#
# 用途:按固定顺序执行全部机械检查与构建验证:
#   1. scripts/check_language_policy.py  语言策略检查;
#   2. scripts/check_iron_rules.sh       铁律机械检查;
#   3. tests/scripts + check_source_release.py  官方纯源码发布门禁;
#   4. clang-format 格式检查(工具缺失打印 SKIP,不判失败);
#   5. cmake --preset cpu-only           配置;
#   6. clang-tidy 静态检查(CPP-030;用第 5 步产出的 compile_commands.json,
#      工具或编译数据库缺失打印 SKIP,不判失败);
#   7. cmake --build --preset cpu-only   构建;
#   8. ctest --preset cpu-only           测试。
# 任一项 FAIL 则整体以 1 退出;末尾打印逐项 PASS/FAIL/SKIP 汇总表。
#
# 用法:bash scripts/ci_check.sh(任意工作目录下均可执行;
#      CI 与本地合入前自检共用本脚本,见 BUILD-030)。
#
# 规则出处:docs/standards/build-and-test.md(BUILD-030、BUILD-001)、
#          docs/standards/cpp-coding.md(CPP-030)。

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

overall=0
step_names=()
step_results=()

# 记录一项结果;$1=项名,$2=PASS|FAIL|SKIP
add_result() {
  step_names+=("$1")
  step_results+=("$2")
  [ "$2" = "FAIL" ] && overall=1
  return 0
}

echo "==== [1/8] check_language_policy ===="
if python3 scripts/check_language_policy.py; then
  add_result "check_language_policy" "PASS"
else
  add_result "check_language_policy" "FAIL"
fi

echo "==== [2/8] check_iron_rules ===="
if bash scripts/check_iron_rules.sh; then
  add_result "check_iron_rules" "PASS"
else
  add_result "check_iron_rules" "FAIL"
fi


echo "==== [3/8] check_source_release ===="
source_release_ok=1
if ! python3 -m unittest discover -s tests/scripts -p 'test_*.py' -v; then
  source_release_ok=0
fi
if ! python3 scripts/check_source_release.py --root "$ROOT" --mode working-tree; then
  source_release_ok=0
fi
if [ "$source_release_ok" -eq 1 ]; then
  add_result "check_source_release" "PASS"
else
  add_result "check_source_release" "FAIL"
fi

echo "==== [4/8] clang-format ===="
if ! command -v clang-format >/dev/null 2>&1; then
  echo "SKIP clang-format (not installed)"
  add_result "clang-format" "SKIP"
else
  # 优先用 git 列出 C/C++/CUDA 源文件;非 git 环境回退为 find。
  # 过滤磁盘不存在的条目:已删除但尚未提交的文件仍在索引(--cached)中,
  # 对其执行 clang-format 会报 No such file(M12 首次删文件时暴露)。
  fmt_files="$(git ls-files --cached --others --exclude-standard \
                 -- '*.h' '*.cpp' '*.cu' 2>/dev/null \
               | while IFS= read -r f; do [ -f "$f" ] && printf '%s\n' "$f"; done || true)"
  if [ -z "$fmt_files" ]; then
    fmt_dirs=()
    for d in include src tests examples python; do
      [ -d "$d" ] && fmt_dirs+=("$d")
    done
    if [ "${#fmt_dirs[@]}" -gt 0 ]; then
      fmt_files="$(find "${fmt_dirs[@]}" -type f \
                     \( -name '*.h' -o -name '*.cpp' -o -name '*.cu' \) \
                     2>/dev/null || true)"
    fi
  fi
  if [ -z "$fmt_files" ]; then
    echo "SKIP clang-format (no C/C++/CUDA sources found)"
    add_result "clang-format" "SKIP"
  elif printf '%s\n' "$fmt_files" | xargs -r clang-format --dry-run --Werror; then
    add_result "clang-format" "PASS"
  else
    add_result "clang-format" "FAIL"
  fi
fi

# clang-tidy 检查(CPP-030):依赖 configure 产出的 compile_commands.json,
# 因此实现为函数、在 configure 成功后调用;工具或编译数据库缺失打 SKIP。
# 受检文件 = 编译数据库中位于仓库内的条目(自动排除 FetchContent 的 _deps)。
run_clang_tidy() {
  local db="build/cpu-only/compile_commands.json"
  if ! command -v clang-tidy >/dev/null 2>&1; then
    echo "SKIP clang-tidy (not installed)"
    add_result "clang-tidy" "SKIP"
    return 0
  fi
  if [ ! -f "$db" ]; then
    echo "SKIP clang-tidy (compile_commands.json not found)"
    add_result "clang-tidy" "SKIP"
    return 0
  fi
  local tidy_files
  tidy_files="$(python3 - "$ROOT" "$db" <<'PYEOF'
import json
import sys

root, db = sys.argv[1], sys.argv[2]
files = sorted({entry["file"] for entry in json.load(open(db))})
for path in files:
    if path.startswith(root + "/") and "/_deps/" not in path:
        print(path)
PYEOF
)"
  if [ -z "$tidy_files" ]; then
    echo "SKIP clang-tidy (no in-repo sources in compile database)"
    add_result "clang-tidy" "SKIP"
    return 0
  fi
  if printf '%s\n' "$tidy_files" | xargs -r clang-tidy -p build/cpu-only --quiet; then
    add_result "clang-tidy" "PASS"
  else
    add_result "clang-tidy" "FAIL"
  fi
  return 0
}

echo "==== [5-8/8] cmake configure / clang-tidy / build / ctest (preset: cpu-only) ===="
if ! command -v cmake >/dev/null 2>&1; then
  echo "SKIP cmake (not installed)"
  add_result "cmake-configure" "SKIP"
  add_result "clang-tidy" "SKIP"
  add_result "cmake-build" "SKIP"
  add_result "ctest" "SKIP"
else
  if cmake --preset cpu-only; then
    add_result "cmake-configure" "PASS"
    run_clang_tidy
    if cmake --build --preset cpu-only; then
      add_result "cmake-build" "PASS"
      if ! command -v ctest >/dev/null 2>&1; then
        echo "SKIP ctest (not installed)"
        add_result "ctest" "SKIP"
      elif ctest --preset cpu-only; then
        add_result "ctest" "PASS"
      else
        add_result "ctest" "FAIL"
      fi
    else
      add_result "cmake-build" "FAIL"
      add_result "ctest" "SKIP"
    fi
  else
    add_result "cmake-configure" "FAIL"
    add_result "clang-tidy" "SKIP"
    add_result "cmake-build" "SKIP"
    add_result "ctest" "SKIP"
  fi
fi

# ---------------------------------------------------------------------------
# 汇总表
# ---------------------------------------------------------------------------
echo ""
echo "==== ci_check summary ===="
i=0
while [ "$i" -lt "${#step_names[@]}" ]; do
  printf '  %-24s %s\n' "${step_names[$i]}" "${step_results[$i]}"
  i=$((i + 1))
done
if [ "$overall" -eq 0 ]; then
  echo "OVERALL: PASS"
else
  echo "OVERALL: FAIL"
fi
exit "$overall"
