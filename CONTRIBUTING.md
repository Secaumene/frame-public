# 贡献指南

- 一切规则的入口是 `docs/README.md`:先通读,再按其中的路由表找到对应的 `docs/` 规范。
- 规则正文只存在于 `docs/`;动手写代码前先按 `docs/standards/reuse-policy.md` 完成复用搜索,提交前执行本地自检。
- commit 规范(格式、type/scope 枚举、中文摘要)见 `docs/standards/language-policy.md`,由 commit-msg 钩子执法。
- 首次克隆后安装钩子:`git config core.hooksPath scripts/git-hooks`,或运行 `bash scripts/setup_dev.sh`(一并执行 `pip install -e .`)。
- 本地自检:`bash scripts/ci_check.sh`(与 CI 完全一致;工具缺失打印 SKIP 不失败)。
- 发起 PR 前按 `.github/PULL_REQUEST_TEMPLATE.md` 填全三个必填段。
- 除非文件另有声明，提交的贡献按 `GPL-3.0-or-later` 许可，完整条款见根 `LICENSE`。
- Frame 官方只发布纯源码，不上传编译产物；发布边界见 BUILD-050～052。
