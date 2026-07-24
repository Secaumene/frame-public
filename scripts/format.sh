#!/usr/bin/env bash
# 代码格式化便利脚本(clang-format 就地应用 / 只读检查)。
#
# 用途:
#   收集仓内全部 *.h/*.cpp/*.cu 源文件(文件收集手法与 scripts/ci_check.sh
#   的 clang-format 步骤一致:优先 git ls-files,非 git 环境回退 find);
#   默认模式就地格式化(clang-format -i),--check 模式只读检查
#   (clang-format --dry-run --Werror,与 docs/standards/build-and-test.md 的格式检查口径同
#   语义)。
#
# 用法:bash scripts/format.sh [--check]
#
# 输出:
#   --check 模式:全部符合规范打印 "OK format --check" 并以 0 退出;发现
#   违例透传 clang-format 差异输出并以 1 退出;clang-format 未安装打印
#   "SKIP clang-format (not installed)" 并以 0 退出(与 ci_check.sh 口径
#   一致)。默认模式:成功打印 "OK formatted <N> files" 并以 0 退出;
#   clang-format 未安装打印 FAIL 并以 1 退出(无法完成任务,不可静默跳过)。
#
# 规则出处:docs/standards/build-and-test.md(第 10 节便利脚本表)、
#          docs/standards/build-and-test.md 的格式检查口径、scripts/ci_check.sh(同款文件
#          收集手法)。

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  echo "usage: $(basename "$0") [--check]" >&2
}

check_mode=0
if [ "$#" -gt 0 ]; then
  case "$1" in
    --check)
      check_mode=1
      shift
      ;;
    *)
      usage
      exit 1
      ;;
  esac
fi
if [ "$#" -gt 0 ]; then
  usage
  exit 1
fi

# 文件收集手法与 ci_check.sh 的 clang-format 步骤一致:优先 git ls-files,
# 非 git 环境回退为 find;过滤磁盘上已不存在的索引条目(已删除未提交文件)
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

if ! command -v clang-format >/dev/null 2>&1; then
  if [ "$check_mode" -eq 1 ]; then
    echo "SKIP clang-format (not installed)"
    exit 0
  else
    echo "FAIL clang-format not installed, cannot format in place" >&2
    exit 1
  fi
fi

if [ -z "$fmt_files" ]; then
  echo "OK format (no C/C++/CUDA sources found)"
  exit 0
fi

if [ "$check_mode" -eq 1 ]; then
  if printf '%s\n' "$fmt_files" | xargs -r clang-format --dry-run --Werror; then
    echo "OK format --check"
    exit 0
  else
    echo "FAIL format --check (see clang-format output above)" >&2
    exit 1
  fi
else
  file_count="$(printf '%s\n' "$fmt_files" | wc -l)"
  if printf '%s\n' "$fmt_files" | xargs -r clang-format -i; then
    echo "OK formatted $file_count files"
    exit 0
  else
    echo "FAIL clang-format -i failed" >&2
    exit 1
  fi
fi
