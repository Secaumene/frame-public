#!/usr/bin/env bash
# 清理便利脚本(删除构建目录,交互确认防误删)。
#
# 用途:
#   删除 build/<preset>(单个 preset 的构建目录)或 build/ 下全部子目录
#   (<preset> 传 all);删除前列出将删目录与占用体积,交互确认后才真正执行。
#
# 用法:
#   bash scripts/clean.sh <preset>|all [--yes]
#   --yes 跳过交互确认(用于非交互调用);非交互终端(stdin 非 tty)且未传
#   --yes 时直接 FAIL,不允许静默确认导致误删。
#
# 输出:
#   目标目录不存在(或 all 模式下 build/ 无子目录)打印 "OK nothing to clean"
#   并以 0 退出;确认后删除成功打印 "OK clean (<preset>|all)";用户在交互
#   确认中拒绝打印 "SKIP clean aborted by user" 并以 0 退出;非交互无 --yes
#   打印 FAIL 并以 1 退出。
#
# 规则出处:docs/standards/build-and-test.md(第 10 节便利脚本表)。

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="$ROOT/build"

usage() {
  echo "usage: $(basename "$0") <preset>|all [--yes]" >&2
}

if [ "$#" -lt 1 ]; then
  usage
  exit 1
fi

target="$1"
shift

# 目标名收紧为 preset 名字符集(小写字母/数字/连字符)或 all:从源头拒绝
# 含 "/" 与 ".." 的输入,防止前缀校验被路径遍历绕过(code-reviewer 建议 1)
if [ "$target" != "all" ] && ! printf '%s' "$target" | grep -Eq '^[a-z0-9-]+$'; then
  echo "FAIL invalid preset name: $target (allowed: lowercase letters, digits, hyphen, or 'all')" >&2
  exit 1
fi

assume_yes=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --yes)
      assume_yes=1
      shift
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

# 待删目录清单
dirs=()
if [ "$target" = "all" ]; then
  if [ -d "$BUILD_ROOT" ]; then
    for d in "$BUILD_ROOT"/*/; do
      [ -d "$d" ] && dirs+=("${d%/}")
    done
  fi
else
  candidate="$BUILD_ROOT/$target"
  [ -d "$candidate" ] && dirs+=("$candidate")
fi

if [ "${#dirs[@]}" -eq 0 ]; then
  echo "OK nothing to clean"
  exit 0
fi

# 安全校验:只允许删除 build/ 之下的路径(防误删,组装路径后再校验前缀)
for d in "${dirs[@]}"; do
  case "$d" in
    "$BUILD_ROOT"/*) ;;
    *)
      echo "FAIL refusing to delete path outside build/: $d" >&2
      exit 1
      ;;
  esac
done

echo "about to remove:"
for d in "${dirs[@]}"; do
  size="$(du -sh "$d" 2>/dev/null | cut -f1)"
  printf '  %-8s %s\n' "$size" "$d"
done

if [ "$assume_yes" -ne 1 ]; then
  if [ ! -t 0 ]; then
    echo "FAIL non-interactive shell without --yes, refusing to delete" >&2
    exit 1
  fi
  read -r -p "delete the above? [y/N] " reply
  case "$reply" in
    y | Y | yes | YES) ;;
    *)
      echo "SKIP clean aborted by user"
      exit 0
      ;;
  esac
fi

for d in "${dirs[@]}"; do
  rm -rf "$d"
done

echo "OK clean ($target)"
