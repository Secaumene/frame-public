#!/usr/bin/env bash
# 铁律机械检查脚本(iron rules checker)。
#
# 用途:执行八项只匹配 #include 行与标识符 token 的机械检查
#      (避免误伤中文注释):
#   1. virtual 白名单(CPP-010):include/ 与 src/ 中命中 virtual token
#      (词边界,排除整行注释)的文件必须落在白名单
#      {include/frame/hal/*.h, include/frame/compiler/pass.h} 与
#      src/backends/** 之内;
#   2. dynamic_cast/typeid 禁用(CPP-011):include/ 与 src/ 全域禁止,
#      无目录白名单(唯一例外 python/ 绑定层不在本脚本扫描范围内);
#   3. 核心层后端隔离(BUILD-003、ARCH-001):src/{core,ir,ops,compiler,
#      runtime} 与 include/frame(含 hal——HAL 是纯接口,同样不得引入 SDK)
#      的 #include 行不得引入 cuda/sycl/level_zero/openvino/acl 系头文件
#      或 backends/ 路径头文件,也不得出现 FRAME_ENABLE_* 条件编译;
#   4. 待办标签格式(CPP-070):include/src/python/tests/cmake/scripts/
#      examples 中所有待办类关键词行必须符合
#      "FRAME-{IMPL|DESIGN|TEST|DOC|PERF|DEP}" 标签格式;
#   5. kernels/ 目录禁 dtype 运行时分支(CPP-012):
#      src/backends/*/kernels/ 内不得出现 switch/if 按 dtype 分支,
#      dispatch_dtype 实现文件白名单豁免(kernels/ 之外的 dtype 分支
#      归 code-reviewer 按 ARCH-042 人工判定,不在本脚本范围);
#   6. throw 禁用(CPP-020):include/ 与 src/ 中禁止 throw token
#      (唯一例外 python/ 绑定层不在本脚本扫描范围内);
#   7. Backend::launch 调用点白名单(ARCH-011)与 native_handle 调用
#      白名单(ARCH-030):eager 入口调用点仅允许 src/runtime/,
#      native_handle 仅允许 include/frame/hal/ 与 src/backends/
#      (tests/ 均不在扫描范围,单测天然放行);
#   8. 后端交叉 include(ARCH-002):src/backends/<X>/ 不得 include
#      其他后端目录或 src/{core,ir,compiler,runtime} 内部头。
#
# 用法:bash scripts/check_iron_rules.sh(任意工作目录下均可执行)。
# 输出:全绿打印 OK 并以 0 退出;否则逐条打印违例并以 1 退出;
#      被检目录不存在时视为该项通过。
#
# 规则出处:docs/standards/cpp-coding.md(CPP-010/011/012/020/070)、
#          docs/standards/build-and-test.md(BUILD-003)、
#          docs/architecture/overview.md(ARCH-002)、
#          docs/architecture/execution-model.md(ARCH-011)、
#          docs/architecture/backend-hal.md(ARCH-030)。

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail_count=0

# 打印一条违例并累加计数
report() {
  printf 'FAIL %s\n' "$1"
  fail_count=$((fail_count + 1))
}

# 将 grep -rn 的输出(file:line:content)逐条转为违例报告;$1=输出,$2=原因
report_grep() {
  local entry loc rest ln
  [ -z "$1" ] && return 0
  while IFS= read -r entry; do
    [ -z "$entry" ] && continue
    loc="${entry%%:*}"
    rest="${entry#*:}"
    ln="${rest%%:*}"
    report "${loc}:${ln} $2"
  done <<< "$1"
}

# 匹配「整行注释」的行首内容(//、/*、* 开头),用于排除注释行。
# 注意:本片段不含 ^ 锚点——调用处将其拼接在 "file:line:" 前缀正则之后,
# 内嵌 ^ 会使整条正则永不匹配(anchor 在中间不成立)。
COMMENT_LINE_RE='[[:space:]]*(//|/\*|\*)'

# ---------------------------------------------------------------------------
# 检查 1:virtual 白名单(CPP-010)
# ---------------------------------------------------------------------------
VIRTUAL_RE='\bvirtual\b'
scan_dirs=()
[ -d include ] && scan_dirs+=(include)
[ -d src ] && scan_dirs+=(src)
if [ "${#scan_dirs[@]}" -gt 0 ]; then
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    # 排除整行注释后确认仍有命中
    hits="$(grep -nE "$VIRTUAL_RE" "$f" | grep -vE "^[0-9]+:${COMMENT_LINE_RE}" || true)"
    [ -z "$hits" ] && continue
    case "$f" in
      include/frame/hal/*.h) : ;;
      include/frame/compiler/pass.h) : ;;
      src/backends/*) : ;;
      *)
        while IFS= read -r h; do
          [ -z "$h" ] && continue
          report "${f}:${h%%:*} virtual outside whitelist (CPP-010)"
        done <<< "$hits"
        ;;
    esac
  done < <(grep -rlIE "$VIRTUAL_RE" "${scan_dirs[@]}" \
             --include='*.h' --include='*.cpp' --include='*.cu' 2>/dev/null || true)
fi

# ---------------------------------------------------------------------------
# 检查 2:dynamic_cast/typeid 禁用(CPP-011)
# ---------------------------------------------------------------------------
# 与检查 1 不同,本项无目录白名单:CPP-011 的唯一例外是 python/ 绑定层,
# 而 python/ 不在扫描目录内;hal 头与后端实现文件不享有 RTTI 豁免。
RTTI_RE='\b(dynamic_cast|typeid)\b'
if [ "${#scan_dirs[@]}" -gt 0 ]; then
  out="$(grep -rnIE "$RTTI_RE" "${scan_dirs[@]}" \
           --include='*.h' --include='*.cpp' --include='*.cu' 2>/dev/null \
          | grep -vE "^[^:]+:[0-9]+:${COMMENT_LINE_RE}" || true)"
  report_grep "$out" "dynamic_cast/typeid forbidden outside python/ bindings (CPP-011)"
fi

# ---------------------------------------------------------------------------
# 检查 3:核心层后端隔离(BUILD-003)
# ---------------------------------------------------------------------------
# #include 行引入后端 SDK 头文件:token 须位于路径起始或 / _ . - 之后,
# 避免误伤形如 macl.h 的无关名字
BACKEND_INCLUDE_RE='^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"]([^">]*[/_.-])?(cuda|sycl|level_zero|ze_api|openvino|acl)'
# 核心层 include backends/ 路径头文件(ARCH-001:核心各层禁止依赖后端实现)
BACKENDS_INCLUDE_RE='^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"][^">]*backends/'
# FRAME_ENABLE_* 条件编译
ENABLE_IFDEF_RE='^[[:space:]]*#[[:space:]]*(if|ifdef|ifndef|elif)([[:space:]]|.*[^A-Za-z0-9_])FRAME_ENABLE_'

core_dirs=()
# src/interop 自 ADR-0013 新增(仅依赖 core 的宿主侧互操作层),src/frontend
# 自 ADR-0017 新增(链顶消费层),src/nn/src/data 自 ADR-0020 新增(nn 依赖集
# core+ir+ops 的构图组合子层旁支;data 目录本批未建,[ -d ] 守卫使其自动跳过、
# 不影响本检查),同受核心层后端隔离检查约束。
for d in src/core src/ir src/ops src/compiler src/runtime src/interop src/frontend src/nn src/data; do
  [ -d "$d" ] && core_dirs+=("$d")
done
if [ "${#core_dirs[@]}" -gt 0 ]; then
  out="$(grep -rnIE "$BACKEND_INCLUDE_RE" "${core_dirs[@]}" \
           --include='*.h' --include='*.cpp' --include='*.cu' 2>/dev/null || true)"
  report_grep "$out" "backend-specific header included in core layer (BUILD-003)"
  out="$(grep -rnIE "$BACKENDS_INCLUDE_RE" "${core_dirs[@]}" \
           --include='*.h' --include='*.cpp' --include='*.cu' 2>/dev/null || true)"
  report_grep "$out" "backends/ header included in core layer (ARCH-001)"
  out="$(grep -rnIE "$ENABLE_IFDEF_RE" "${core_dirs[@]}" \
           --include='*.h' --include='*.cpp' --include='*.cu' 2>/dev/null || true)"
  report_grep "$out" "FRAME_ENABLE_* conditional in core layer (BUILD-003)"
fi
# 公共头全域受检,含 hal/:HAL 是纯接口,同样禁止 SDK 头与后端条件编译
if [ -d include/frame ]; then
  out="$(grep -rnIE "$BACKEND_INCLUDE_RE" include/frame \
           --include='*.h' 2>/dev/null || true)"
  report_grep "$out" "backend-specific header included in public headers (BUILD-003)"
  out="$(grep -rnIE "$BACKENDS_INCLUDE_RE" include/frame \
           --include='*.h' 2>/dev/null || true)"
  report_grep "$out" "backends/ header included in public headers (ARCH-001)"
  out="$(grep -rnIE "$ENABLE_IFDEF_RE" include/frame \
           --include='*.h' 2>/dev/null || true)"
  report_grep "$out" "FRAME_ENABLE_* conditional in public headers (BUILD-003)"
fi

# ---------------------------------------------------------------------------
# 检查 4:待办标签格式(CPP-070)
# ---------------------------------------------------------------------------
# 关键词以拼接方式写出,避免本脚本自身被本检查命中
kw_a='TO''DO'
kw_b='FIX''ME'
kw_c='XX''X'
VALID_TAG_RE="${kw_a}\(FRAME-(IMPL|DESIGN|TEST|DOC|PERF|DEP)\):"
todo_dirs=()
# tools/ 自 ADR-0017 新增(frame_dslc 命令行工具);本阶段尚未落地,下方
# [ -d "$d" ] 存在性判定会将其自动跳过,不需要额外保护。
for d in include src python tests cmake scripts examples tools; do
  [ -d "$d" ] && todo_dirs+=("$d")
done
if [ "${#todo_dirs[@]}" -gt 0 ]; then
  out="$(grep -rnIE "(${kw_a}|${kw_b}|${kw_c})" "${todo_dirs[@]}" 2>/dev/null \
          | grep -vE "$VALID_TAG_RE" || true)"
  report_grep "$out" "bare or malformed to-do marker (CPP-070); required: FRAME-{IMPL|DESIGN|TEST|DOC|PERF|DEP} tag"
fi

# ---------------------------------------------------------------------------
# 检查 5:kernels/ 目录禁 dtype 运行时分支(CPP-012)
# ---------------------------------------------------------------------------
DTYPE_BRANCH_RE='switch *\(.*dtype|if *\(.*dtype'
for d in src/backends/*/kernels; do
  [ -d "$d" ] || continue
  out="$(grep -rnIE "$DTYPE_BRANCH_RE" "$d" 2>/dev/null \
          | grep -vE '(^|/)dispatch_dtype[^/:]*:' \
          | grep -vE "^[^:]+:[0-9]+:${COMMENT_LINE_RE}" || true)"
  report_grep "$out" "runtime dtype branch in kernels/ (CPP-012); use frame::dispatch_dtype"
done

# ---------------------------------------------------------------------------
# 检查 6:throw 禁用(CPP-020)
# ---------------------------------------------------------------------------
# 核心库以 Status/Result<T> 表达错误;throw 的唯一例外是 python/ 绑定层,
# 该目录不在扫描范围内。
THROW_RE='\bthrow\b'
if [ "${#scan_dirs[@]}" -gt 0 ]; then
  out="$(grep -rnIE "$THROW_RE" "${scan_dirs[@]}" \
           --include='*.h' --include='*.cpp' --include='*.cu' 2>/dev/null \
          | grep -vE "^[^:]+:[0-9]+:${COMMENT_LINE_RE}" || true)"
  report_grep "$out" "throw forbidden outside python/ bindings (CPP-020); use Status/Result<T>"
fi

# ---------------------------------------------------------------------------
# 检查 7:Backend::launch 调用点白名单(ARCH-011)与 native_handle 白名单(ARCH-030)
# ---------------------------------------------------------------------------
# eager 唯一入口 Backend::launch 的调用点(成员调用语法 ->launch( / .launch()
# 仅允许出现在下方白名单目录;新增调用点须扩充白名单并在 PR 描述声明属于
# ARCH-011 三类准入中的哪一类。tests/ 不在扫描范围(单算子单测天然放行)。
LAUNCH_CALL_RE='(->|\.)launch *\('
LAUNCH_WHITELIST_RE='^src/runtime/'
launch_dirs=()
[ -d include ] && launch_dirs+=(include)
[ -d src ] && launch_dirs+=(src)
[ -d python ] && launch_dirs+=(python)
if [ "${#launch_dirs[@]}" -gt 0 ]; then
  out="$(grep -rnIE "$LAUNCH_CALL_RE" "${launch_dirs[@]}" \
           --include='*.h' --include='*.cpp' --include='*.cu' 2>/dev/null \
          | grep -vE "^[^:]+:[0-9]+:${COMMENT_LINE_RE}" \
          | grep -vE "^${LAUNCH_WHITELIST_RE}" || true)"
  report_grep "$out" "Backend::launch call site outside whitelist (ARCH-011); eager entry allowed only in src/runtime/ or unit tests"
fi

# native_handle 仅允许 include/frame/hal/(接口声明)与 src/backends/(下沉实现)。
NATIVE_HANDLE_RE='\bnative_handle\b'
nh_dirs=()
for d in src/core src/ir src/ops src/compiler src/runtime src/frontend python; do
  [ -d "$d" ] && nh_dirs+=("$d")
done
if [ -d include/frame ]; then
  out="$(grep -rnIE "$NATIVE_HANDLE_RE" include/frame --exclude-dir=hal \
           --include='*.h' 2>/dev/null \
          | grep -vE "^[^:]+:[0-9]+:${COMMENT_LINE_RE}" || true)"
  report_grep "$out" "native_handle outside whitelist (ARCH-030); allowed only in include/frame/hal/ and src/backends/"
fi
if [ "${#nh_dirs[@]}" -gt 0 ]; then
  out="$(grep -rnIE "$NATIVE_HANDLE_RE" "${nh_dirs[@]}" \
           --include='*.h' --include='*.cpp' --include='*.cu' 2>/dev/null \
          | grep -vE "^[^:]+:[0-9]+:${COMMENT_LINE_RE}" || true)"
  report_grep "$out" "native_handle outside whitelist (ARCH-030); allowed only in include/frame/hal/ and src/backends/"
fi

# ---------------------------------------------------------------------------
# 检查 8:后端交叉 include(ARCH-002)
# ---------------------------------------------------------------------------
# src/backends/<X>/ 禁止 include 其他后端目录(backends/<Y>/,Y != X)
# 与 src/{core,ir,compiler,runtime} 内部头(公共能力一律经 include/frame/)。
for d in src/backends/*/; do
  [ -d "$d" ] || continue
  x="$(basename "$d")"
  out="$(grep -rnIE '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"][^">]*backends/' "$d" \
           --include='*.h' --include='*.cpp' --include='*.cu' 2>/dev/null \
          | grep -vE "backends/${x}/" || true)"
  report_grep "$out" "cross-backend include (ARCH-002); backend ${x} must not include other backends"
  out="$(grep -rnIE '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"]([^">]*/)?src/(core|ir|compiler|runtime)/' "$d" \
           --include='*.h' --include='*.cpp' --include='*.cu' 2>/dev/null || true)"
  report_grep "$out" "internal core-layer header included by backend (ARCH-002); use public headers under include/frame/"
done

# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
if [ "$fail_count" -eq 0 ]; then
  echo "OK"
  exit 0
fi
echo "check_iron_rules: ${fail_count} violation(s) found"
exit 1
