#!/usr/bin/env bash
# 测试便利脚本(ctest 封装,preset 白名单校验 + 参数透传)。
#
# 用途:
#   封装 `ctest --preset <preset> --output-on-failure`,preset 校验规则同
#   scripts/build.sh(BUILD-001);wheel/bench 两个 configure preset 没有同名
#   test preset(docs/standards/build-and-test.md 第 2 节 preset 清单),对这
#   两个名字提前 FAIL 并给出说明,而不是任由 ctest 报出难懂的错误。
#
# 用法:
#   bash scripts/test.sh <preset> [ctest 参数...]
#   额外参数原样透传给 ctest(如 `-R <regex>`、`-L <label>`)。
#
# 输出:
#   ctest 通过打印 "OK test (<preset>)" 并以 0 退出;preset 非法、preset 属于
#   wheel/bench、或 ctest 本身失败均打印 FAIL 并以 1 退出。
#
# 规则出处:docs/standards/build-and-test.md(BUILD-001、第 2 节 preset 清单、
#          第 10 节便利脚本表)。

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  echo "usage: $(basename "$0") <preset> [ctest args...]" >&2
}

if [ "$#" -lt 1 ]; then
  usage
  exit 1
fi

preset="$1"
shift

# preset 名单唯一权威来源是 CMakePresets.json,经 cmake --list-presets 解析
available="$(cmake --list-presets=configure 2>/dev/null | grep -o '"[^"]*"' | tr -d '"')"

if ! printf '%s\n' "$available" | grep -qx -- "$preset"; then
  echo "FAIL unknown preset: $preset" >&2
  echo "available presets:" >&2
  printf '%s\n' "$available" | sed 's/^/  /' >&2
  exit 1
fi

case "$preset" in
  wheel | bench)
    echo "FAIL preset '$preset' has no test preset (wheel/bench are build-only, see docs/standards/build-and-test.md #2)" >&2
    exit 1
    ;;
esac

if ctest --preset "$preset" --output-on-failure "$@"; then
  echo "OK test ($preset)"
else
  echo "FAIL test ($preset)" >&2
  exit 1
fi
