#!/usr/bin/env bash
# 系统安装便利脚本(cmake --install 封装,未构建先构建)。
#
# 用途:
#   封装 `cmake --install build/<preset> [--prefix <dir>]`;若目标 preset
#   尚未 configure(build/<preset> 下无 CMakeCache.txt),先调用
#   scripts/build.sh <preset> 补齐构建,再执行安装。
#
# 用法:
#   bash scripts/install.sh [--preset <name>] [--prefix <dir>]
#   --preset 缺省为 release;--prefix 透传给 cmake --install --prefix。
#
# 输出:
#   安装成功打印 "OK install (<preset>)" 并以 0 退出;安装失败(如无写权限)
#   打印 "FAIL install (<preset>), try --prefix pointing at a writable
#   directory" 并以 1 退出;补齐构建失败打印 FAIL 并以 1 退出。
#
# 规则出处:docs/standards/build-and-test.md(第 10 节安装与导出、BUILD-040)。

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  echo "usage: $(basename "$0") [--preset <name>] [--prefix <dir>]" >&2
}

preset="release"
prefix=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --preset)
      if [ "$#" -lt 2 ]; then
        usage
        exit 1
      fi
      preset="$2"
      shift 2
      ;;
    --prefix)
      if [ "$#" -lt 2 ]; then
        usage
        exit 1
      fi
      prefix="$2"
      shift 2
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

build_dir="$ROOT/build/$preset"
if [ ! -f "$build_dir/CMakeCache.txt" ]; then
  echo "preset '$preset' not configured, building first"
  if ! "$ROOT/scripts/build.sh" "$preset"; then
    echo "FAIL install ($preset), build step failed" >&2
    exit 1
  fi
fi

install_args=(--install "$build_dir")
if [ -n "$prefix" ]; then
  install_args+=(--prefix "$prefix")
fi

if cmake "${install_args[@]}"; then
  echo "OK install ($preset)"
else
  echo "FAIL install ($preset), try --prefix pointing at a writable directory" >&2
  exit 1
fi
