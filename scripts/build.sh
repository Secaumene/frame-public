#!/usr/bin/env bash
# 构建便利脚本(configure + build 一键封装,preset 白名单校验)。
#
# 用途:
#   日常开发循环封装 `cmake --preset <preset> && cmake --build --preset <preset>`;
#   动手前校验 preset 名是否落在 `cmake --list-presets=configure` 的合法名单内
#   (BUILD-001:一律经 preset 表达,禁止手敲 -D 长命令)。
#
# 用法:
#   bash scripts/build.sh <preset> [--fresh]
#   <preset>  必须是合法 configure preset 名(见下方名单来源);
#   --fresh   先删除 build/<preset>(若存在)再重新 configure,用于干净重建。
#
# 输出:
#   成功打印 "OK build (<preset>)" 并以 0 退出;preset 不在名单打印
#   "FAIL unknown preset: <name>" 与可用名单并以 1 退出;configure 或 build
#   失败打印 "FAIL build (<preset>)" 并以 1 退出。
#
# 规则出处:docs/standards/build-and-test.md(BUILD-001、第 10 节便利脚本表)。

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  echo "usage: $(basename "$0") <preset> [--fresh]" >&2
}

if [ "$#" -lt 1 ]; then
  usage
  exit 1
fi

preset="$1"
shift

fresh=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --fresh)
      fresh=1
      shift
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

# preset 名单唯一权威来源是 CMakePresets.json,经 cmake --list-presets 解析,
# 本脚本不手工维护副本清单(避免与 CMakePresets.json 走漂)
available="$(cmake --list-presets=configure 2>/dev/null | grep -o '"[^"]*"' | tr -d '"')"

if ! printf '%s\n' "$available" | grep -qx -- "$preset"; then
  echo "FAIL unknown preset: $preset" >&2
  echo "available presets:" >&2
  printf '%s\n' "$available" | sed 's/^/  /' >&2
  exit 1
fi

if [ "$fresh" -eq 1 ]; then
  echo "removing build/$preset (--fresh)"
  rm -rf "${ROOT:?}/build/$preset"
fi

if cmake --preset "$preset" && cmake --build --preset "$preset"; then
  echo "OK build ($preset)"
else
  echo "FAIL build ($preset)" >&2
  exit 1
fi
