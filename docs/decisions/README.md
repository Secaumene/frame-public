# 架构决策记录(ADR)说明与索引

> **强制等级**:规范(MUST)
> **相关铁律**:#1 编译优先 / #3 后端矩阵 / #5 两级复用
> **面向读者**:实现代码的工程师
> **最后更新**:2026-07-24(新增 ADR-0023:GPLv3+ 与官方纯源码发布)

ADR(Architecture Decision Record)记录一次不可逆或影响面大的技术决策:背景、选定方案、备选方案与否决理由、后果。单一事实来源原则:凡已被 ADR 裁决的事项,其他文档与代码注释只引用编号(如 ADR-0004),不复述论证。本目录的规则编号前缀为 `ADR-`。

## 1. 什么时候必须写 ADR(触发清单)

【ADR-010】【MUST】出现下表任一情形时,必须先提交 ADR 并在其进入「已接受」状态后才允许提交对应实现代码。判定方法:PR 涉及下表情形而 docs/decisions/ 中无对应「已接受」ADR 时,由 code-reviewer 检查表打回;第 1 项另由 pre-commit-check 对 CMake 中新增 `find_package`/`FetchContent` 与 docs/decisions/ 目录做机械比对。

| # | 触发情形 | 关联规则 / 文档 |
|---|---|---|
| 1 | 引入新第三方依赖(新增 `find_package` / `FetchContent`) | REUSE-010 |
| 2 | 新增或更改执行模式(编译路径 / eager 之外的第三种模式) | ARCH-012、ADR-0001 |
| 3 | 更改 IR 核心对象模型(Graph/Node/Value/Attribute 类型集合) | ARCH-020、ADR-0002 |
| 4 | 新增 src/ 一级子目录或新增架构层 | ARCH-003 |
| 5 | 更改 C++ 标准版本 | CPP-001、ADR-0004 |
| 6 | 突破虚函数 / RTTI 白名单 | CPP-010、CPP-011 |
| 7 | 更改标准 pass 管线顺序 | docs/architecture/compiler-passes.md |
| 8 | 选择或变更某后端的执行模式路线 | ADR-0005 |
| 9 | 支持动态 shape(v0 一律拒绝注册) | docs/architecture/operator-system.md |
| 10 | Intel NPU 绕过 OpenVINO 直连 Level Zero | docs/backends/intel-npu.md |
| 11 | 将第三方源码直接拷入仓库(vendoring) | docs/standards/reuse-policy.md |
| 12 | 启用或变更 Frame 官方预编译分发或发布渠道 | BUILD-052、ADR-0023 |

【ADR-011】【MUST】新增 ADR 或变更其状态时,必须同步更新本文件第 3 节索引表。判定方法:`ls docs/decisions/[0-9]*.md` 的文件数等于索引表数据行数,且每个文件头部「状态」字段与索引表一致。

## 2. 流程

1. 复制 docs/decisions/adr-template.md 中的模板全文;
2. 编号 = 现有最大编号 + 1(四位数字),文件名按 ADR-002 规则命名;
3. 以状态「提议」提交 PR,PR 描述中引用第 1 节触发清单的条目编号;
4. review 通过后将状态改为「已接受」;
5. 需要推迟裁决时,状态改「推迟」并写明可判定的重评条件(参照 ADR-0005);
6. 被新决策取代时,旧文件状态改「被 ADR-MMMM 取代」,不删除旧文件;
7. ADR 文件自身使用 adr-template.md 的固定格式,不使用 docs/ 通用文档头元数据块。

## 3. 索引表

| 编号 | 标题 | 状态 | 日期 |
|---|---|---|---|
| ADR-0001 | 编译优先执行模型 | 已接受 | 2026-07-04 |
| ADR-0002 | IR 双轨并行(自研主线 + MLIR 预研) | 已接受 | 2026-07-04 |
| ADR-0003 | 核心库 Status/Result&lt;T&gt; 错误模型 | 已接受 | 2026-07-04 |
| ADR-0004 | C++20 基线 | 已接受 | 2026-07-04 |
| ADR-0005 | 昇腾后端执行模式(aclnn vs GE) | 推迟(重评条件:v2.0 里程碑启动昇腾后端实现时) | 2026-07-04 |
| ADR-0006 | 暂不开源 | 被 ADR-0023 取代 | 2026-07-04 |
| ADR-0008 | 训练能力纳入产品范围(编译期反向图路线) | 已接受 | 2026-07-12 |
| ADR-0009 | 受限静态广播方向(实施另立里程碑) | 已接受 | 2026-07-13 |
| ADR-0010 | cuda 全归约采用 CUB DeviceReduce | 已接受 | 2026-07-13 |
| ADR-0011 | cuDNN 推迟至卷积类算子立项 | 被 ADR-0021 取代 | 2026-07-13 |
| ADR-0012 | spdlog 推迟,维持薄 warn_log | 推迟(重评条件:日志点 ≥3 或分级/落盘/异步需求) | 2026-07-13 |
| ADR-0013 | ONNX 权重交换采用自研最小 wire-format 编解码 | 已接受 | 2026-07-13 |
| ADR-0014 | 基准计时设施采用 Google Benchmark | 已接受 | 2026-07-13 |
| ADR-0015 | pybind11 升级至 v3.0.4 并收紧锁定语义 | 已接受 | 2026-07-13 |
| ADR-0016 | GoogleTest 升级至 v1.17.0 并移除 FIND_PACKAGE_ARGS | 已接受 | 2026-07-13 |
| ADR-0017 | 新增 frontend 前端层与 tools 工具目录(JSON DSL v0) | 已接受 | 2026-07-13 |
| ADR-0018 | 引入 nlohmann/json 作为工具层 JSON 解析库 | 已接受 | 2026-07-13 |
| ADR-0019 | 采纳 cublasLt 统一 matmul 路径并确立精度策略(allow_tf32) | 已接受 | 2026-07-18 |
| ADR-0020 | 新增 frame::nn 构图组合子层与 frame::data 数据加载层 | 已接受 | 2026-07-18 |
| ADR-0021 | 采纳 cuDNN 承载卷积/池化 CUDA 内核(销 ADR-0011 推迟) | 已接受 | 2026-07-18 |
| ADR-0022 | 采纳 cuFFT 承载 CUDA FFT 内核、pocketfft 承载 CPU 参考实现 | 已接受 | 2026-07-21 |
| ADR-0023 | 采用 GPLv3+ 与官方纯源码发布 | 已接受 | 2026-07-24 |
