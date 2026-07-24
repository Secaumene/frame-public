# ADR-0023:采用 GPLv3+ 与官方纯源码发布

- 状态:已接受
- 日期:2026-07-24
- 关联铁律:#5
- 关联规则:BUILD-050、BUILD-051、BUILD-052、REUSE-014

## 背景

ADR-0006 约定在决定对外发布时重新裁决许可证。Frame 已进入公开源码阶段，同时
保留由用户自行安装的 CUDA、cuDNN、oneAPI、OpenVINO、CANN 等可选 SDK 接口。
GPLv3 第 10 节禁止对下游权利附加额外限制，而专有 SDK 另有独立分发条款，因此
项目许可证与 Frame 官方发布范围必须分开表达。

## 决策

Frame 源码与文档（另有声明者除外）采用 `GPL-3.0-or-later`：

- 根 `LICENSE` 保存 FSF 发布的 GPLv3 完整原文，`pyproject.toml` 使用同一 SPDX；
- 公开源码仓库固定为 `https://github.com/Secaumene/frame-public`；
- Frame 官方只发布净化后的纯源码树与托管平台自动生成的源码归档；
- CI 可以临时编译和测试，但不得上传编译产物、包、容器或厂商 SDK；
- 不授予 linking exception，也不官方发布与可选专有 SDK 链接或组合的产物；
- 纯源码规则只约束 Frame 官方发布流程，不缩减任何下游 GPL 权利；下游分发者
  自行确认 GPL 与第三方条款的兼容性；
- 启用或变更任何官方预编译分发渠道时，必须先立新 ADR 并完成许可证兼容审查。

判定方法：运行 `python3 scripts/check_source_release.py --mode working-tree` 与
`bash scripts/ci_check.sh`；公开副本另执行零协作痕迹净化和完整纯源码复验。

## 备选方案

- Apache-2.0：依赖兼容且宽松，但不满足本次强 copyleft 选择，否决。
- GPLv3 加自定义 linking exception：可扩大二进制分发范围，但会改变授权边界且
  本次不发布二进制，无必要，否决。
- 官方发布预编译包：安装便利，但会引入 GPL 与厂商条款的组合风险，否决。

## 后果

- 正面：公开源码保持自由软件属性，官方出口不携带厂商 SDK 或编译产物。
- 负面：用户必须本地编译；Frame 官方暂不提供 wheel、系统包、容器或预编译库。
- 跟进：ADR-0006 被本篇取代；同步 README、贡献指南、包元数据、发布规范、净化
  清单和机械门禁，并以全新单提交公开仓发布。
