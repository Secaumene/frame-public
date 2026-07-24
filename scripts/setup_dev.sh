#!/usr/bin/env bash
# 开发环境一键初始化脚本(git 钩子 + Python 绑定可编辑安装)。
#
# 用途:
#   新克隆仓库后一次性执行,完成两件事:
#   1. 设置 git commit-msg 钩子路径(scripts/git-hooks/commit-msg,LANG-010);
#   2. 以可编辑模式安装 Python 绑定(pip install -e .),供本地 import frame
#      与 pytest 使用。
#
# 用法:bash scripts/setup_dev.sh(无参数)。
#
# 输出:
#   两项各自打印 OK/FAIL/SKIP;python3 或 pip 缺失时打印
#   "SKIP: python3/pip not found, install python bindings manually" 且不判
#   失败(铁律 2 仅要求 Python 是薄绑定层,尽力语义);git 配置失败或 pip
#   安装命令本身失败判 FAIL。末尾打印两项状态汇总。
#
# 规则出处:docs/standards/language-policy.md(LANG-010,commit-msg 钩子)、
#          docs/architecture/overview.md(ARCH-002,C++ 为核心,Python 只做绑定)。

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

overall=0

echo "==== [1/2] git commit-msg hook ===="
if git -C "$ROOT" config core.hooksPath scripts/git-hooks; then
  hook_status="OK"
  echo "OK git hooksPath = scripts/git-hooks"
else
  hook_status="FAIL"
  echo "FAIL git config core.hooksPath failed" >&2
  overall=1
fi

echo "==== [2/2] python bindings (editable install) ===="
if ! command -v python3 >/dev/null 2>&1 || ! python3 -m pip --version >/dev/null 2>&1; then
  pip_status="SKIP"
  echo "SKIP: python3/pip not found, install python bindings manually"
else
  if python3 -m pip install -e "$ROOT"; then
    pip_status="OK"
    echo "OK pip install -e ."
  else
    pip_status="FAIL"
    echo "FAIL pip install -e . failed" >&2
    overall=1
  fi
fi

echo ""
echo "==== setup_dev summary ===="
printf '  %-24s %s\n' "git-hooks" "$hook_status"
printf '  %-24s %s\n' "python-editable-install" "$pip_status"
if [ "$overall" -eq 0 ]; then
  echo "OK setup_dev"
else
  echo "FAIL setup_dev"
fi
exit "$overall"
