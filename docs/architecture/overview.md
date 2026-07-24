# 分层架构总览

> **强制等级**:规范(MUST)
> **相关铁律**:#1 编译优先 / #2 语言支持 / #3 后端矩阵 / #4 语言策略 / #5 两级复用
> **面向读者**:实现代码的工程师
> **最后更新**:2026-07-23(M22/M23/M25/M27/M28 nn 工厂与 Python 公共面交付注记,
> ARCH-001 增两分支 + §2.11/§2.12)

## 1. 一张图看懂分层

自上而下为「从基础值类型到执行」的层次走读顺序;**依赖方向的唯一权威是第 3 章
ARCH-001**(每层只允许依赖排在它前面的层,反向禁止):

```
用户代码 (C++ / Python)
     │
┌────▼──────────────────────────────────────────────────────────┐
│ 核心层 core     Tensor / DType / Device / Shape / Storage /   │
│                 Status / Result<T>(基础值类型与错误模型)     │
├────────────────────────────────────────────────────────────────┤
│ 图 IR 层 ir     Graph / Node / Value / Attribute              │
│                 (静态计算图 + verify + 文本序列化)           │
├────────────────────────────────────────────────────────────────┤
│ 算子系统 ops    OpSchema / OpRegistry / KernelRegistry        │
│                 (算子定义与注册表,与后端无关)               │
├────────────────────────────────────────────────────────────────┤
│ 编译器层 compiler   PassManager:固定 pass 管线               │
│                     … → backend_lowering                      │
├────────────────────────────────────────────────────────────────┤
│ 运行时层 runtime    BackendRegistry 实现 / Executable 执行 /  │
│                     编译缓存 / 回退链 / eager 逃生舱入口      │
├────────────────────────────────────────────────────────────────┤
│ 前端层 frontend     ModelSpec 校验 / lower 为 ir::Graph /     │
│                     进程内训练执行 / C++ 源码生成(ADR-0017) │
├────────────────────────────────────────────────────────────────┤
│ 后端 HAL 层 hal     Backend / Stream / Event / Allocator /    │
│                     Executable(纯接口,无实现)              │
└────┬──────────────────────────────────────────────────────────┘
     │ 插件式注册 FRAME_REGISTER_BACKEND
 cpu │ cuda │ intel_gpu (SYCL/oneAPI) │ intel_npu (OpenVINO) │ ascend (CANN)
```

## 2. 各层职责与禁区

每层固定按「职责 / 不做什么 / 代码位置」三段描述。

### 2.1 核心层 core

- 职责:基础值类型(`Tensor`、`DType`、`Device`、`Shape`、`Storage`)、错误模型(`Status`/`Result<T>`,见 `include/frame/core/status.h`)、通用宏。`Device` 是值类型 `{std::string_view backend; int32_t index}`,不设 DeviceType 枚举(见 `architecture/backend-hal.md`)。
- 不做什么:不含任何图变换逻辑;设备差异全部下沉到 `hal::Allocator` 实现(`src/backends/`),core 只持纯接口指针。
- 代码位置:`include/frame/core/`、`src/core/`。

### 2.2 图 IR 层 ir

- 职责:图数据结构、验证器(`verify`)、确定性文本序列化(golden test 依据)、
  纯图**分析**工具(`memory_plan.h` 确定性内存规划分析,M9;落 ir 层的分层
  依据:compiler 的 pass 与 backends 的 Executable 须调同一份分析,而
  backends 不得依赖 compiler,ir 是二者公共可达层)。
- 不做什么:不含优化**变换**逻辑(分析工具只读图、产出计划、不改图;改图的
  优化一律在 compiler 层 pass)。
- 代码位置:`include/frame/ir/`、`src/ir/`。

### 2.3 算子系统 ops

- 职责:算子 schema 定义与注册(`OpSchema`/`OpRegistry`)、kernel 注册表(`KernelRegistry`)、编译期 dtype 分发工具(`dispatch_dtype`)。schema 以 `ir::AttrType` 描述属性(ops 依赖 ir,见 ARCH-001)。
- 不做什么:不含具体 kernel 实现(实现在各后端目录);不做分发之外的运行时逻辑。
- 代码位置:`include/frame/ops/`、`src/ops/`(内置算子 schema 桩在 `src/ops/schemas/`)。

### 2.4 编译器层 compiler

- 职责:全部图变换(标准 pass 管线 + 自定义 pass)、后端 lowering 决策。
- 不做什么:不直接调用设备 API;不管理执行期资源。
- 代码位置:`include/frame/compiler/`、`src/compiler/`(标准 pass 在 `src/compiler/passes/`)。

### 2.5 运行时层 runtime

- 职责:注册表的实现(`src/runtime/backend_registry.cpp`)与执行器(`src/runtime/executable.cpp`)、编译编排入口(`runtime::compile`,M7 已实化)与编译产物缓存(`src/runtime/compile.cpp`)、流/事件与内存生命周期管理、eager 逃生舱与回退链的入口(见 `architecture/execution-model.md`)。
- 不做什么:不做图优化。
- 代码位置:`include/frame/runtime/`、`src/runtime/`。

### 2.6 后端 HAL 层 hal

- 职责:纯接口(`Backend`/`Stream`/`Event`/`Allocator`/`Executable`)+ 注册宏。本层是虚函数白名单区域(CPP-010)。
- 不做什么:不含任何实现逻辑;实现全部在 `src/backends/`。
- 代码位置:`include/frame/hal/`。

### 2.7 后端实现层 backends

- 职责:HAL 接口的插件式实现;后端注册键为字符串:`"cpu"`、`"cuda"`、`"intel_gpu"`、`"intel_npu"`、`"ascend"`。cpu 后端永远启用,是数值对比与回退链终点的参考后端。
- 不做什么:不修改 IR(只读消费);不 include 其他后端。
- 代码位置:`src/backends/{cpu,cuda,intel_gpu,intel_npu,ascend}/`。

### 2.8 Python 绑定层

- 职责:pybind11 薄壳绑定(PY-001),`throw` 仅允许出现在此层(CPP-020)。
- 代码位置:`python/`。

### 2.9 互操作层 interop(ADR-0013 授权新增)

- 职责:与外部生态的宿主侧数据交换;首个能力 = ONNX 权重导入/导出
  (initializer TensorProto 子集,自研最小 wire-format 编解码,范围与
  判定见 ADR-0013)。
- 不做什么:不做算子图级互操作(另案);不参与执行/编译;仅依赖 core
  (host 内存 Tensor),不依赖 ir/ops/compiler/runtime/hal/backends——
  依赖口径与 core 相同:对 `hal::Allocator` 仅允许**前向声明的引用形参**
  (分配器由调用方注入,tensor.h/fused_elementwise_utils.h 同款先例),
  不 include hal 头、不派生 hal 接口、不产生链接依赖。
- 代码位置:`include/frame/interop/`、`src/interop/`。

### 2.10 前端层 frontend(ADR-0017 授权新增)

- 职责:「模型描述 → 可运行」的用户面入口,四能力全部只消费纯 C++
  `ModelSpec` 结构:校验(validate)、lower 为 `ir::Graph`(经
  `ops::create_node_with_inferred_types`)、进程内训练执行(runner,
  复用 `compiler::build_backward_graph` / `build_sgd_update_graph` /
  `runtime::compile` / `run_with_allocated_outputs`)、自包含 C++
  训练/推理源码生成(emitter)。v0 范围与 JSON schema 的唯一权威见
  `docs/architecture/frontend-dsl.md` 与 ADR-0017。
- 不做什么:不解析 JSON(解析仅存在于 `tools/` 工具层,nlohmann/json
  准入限定「工具与测试代码」,ADR-0018);不新增算子/pass/执行模式;
  不触碰 IR 对象模型;v0 无 Python 绑定。
- 代码位置:`include/frame/frontend/`、`src/frontend/`;命令行工具
  `tools/frame_dslc/`(工具目录非架构层,消费 frontend 库与公共 API,
  链接聚合库 `frame::frame`)。

### 2.11 神经网络模块层 nn(ADR-0020 授权新增)

- 职责:编译期构图组合子——Module 值语义树(参数声明 + 类型擦除构建器),
  产出前向 IR 子图与确定性有序参数清单;首批模块 Linear/Relu/Sequential/
  MseLoss。实现契约的唯一权威见 `docs/architecture/nn-design.md`
  (ARCH-070~076)。
- 不做什么:零 eager、不触数值;不依赖 compiler/runtime/hal/frontend
  (训练线组合归调用方);不做参数数值物化(初始化仅声明)。
- 代码位置:`include/frame/nn/`、`src/nn/`。
- 交付注记:除 M20 基础组合子外，M22/M23/M25/M27/M28 已在该层增加序列、频域、
  SSM、SNN 与图网络的静态构图工厂，并经 `frame.nn`/`frame._core` 薄绑定公开。
  M25–M28 已完成 CPU/CUDA/Python 最终验收；该交付不改变本层依赖边界或
  ARCH-001、ARCH-070~076 的既有要求。

### 2.12 数据加载层 data(ADR-0020 授权新增)

- 职责:内存数据集与批迭代——TensorDataset(cpu 列式)+ DataLoader
  (确定性洗牌/drop_last/逐样本行拷贝批组装)。契约见 nn-design.md
  ARCH-076。
- 不做什么:仅依赖 core;`hal::Allocator` 仅前向声明的引用形参注入
  (口径同 interop);不做多线程 prefetch/磁盘格式(v2.0)。
- 代码位置:`include/frame/data/`、`src/data/`。

## 3. 依赖方向铁则

- 【ARCH-001】【MUST】依赖方向单向:core ← ir ← ops ← compiler ← runtime ← frontend(箭头指向被依赖方,与代码一致:如 `include/frame/ops/op_schema.h` include `frame/ir/attribute.h`;frontend 为链顶消费层,ADR-0017);hal 仅依赖 core 与 ir;interop 仅依赖 core(ADR-0013);nn 依赖集 = core+ir+ops、不依赖 compiler/runtime/hal(ADR-0020,frontend→nn 单向);data 仅依赖 core、`hal::Allocator` 仅前向声明注入(ADR-0020,口径同 interop);核心各层(`src/{core,ir,ops,compiler,runtime,interop,frontend,nn,data}/` 与 `include/frame/`)禁止 include `src/backends/` 或任何后端 SDK 头文件。判定方法:运行 `scripts/check_iron_rules.sh`(核心层后端隔离检查);层内反向 include(如 `src/ir/` include ops/compiler 头)由 code-reviewer 按本条打回。
  core 公共头对 hal 类型仅允许前向声明;`src/core/` 实现单元允许 include `include/frame/hal/` 纯接口头(hal 头随 `frame_core` 发布,TOP-003,无链接依赖);hal 头不得 include 引用 hal 的 core 头,include 图必须无环。
- 【ARCH-002】【MUST】`src/backends/<X>/` 只允许 include:`include/frame/core/`、`include/frame/ir/`(只读消费 IR)、`include/frame/hal/`、`include/frame/ops/`(kernel 注册)、该后端自身头文件及其第三方 SDK 头文件;禁止 include 其他后端目录与 `src/{core,ir,compiler,runtime}/` 内部头。判定方法:运行 `scripts/check_iron_rules.sh`(后端交叉 include 检查)。
- 【ARCH-003】【MUST】新增 `src/` 一级子目录或新增架构层,必须先有已接受的 ADR。判定方法:PR 中出现新的 `src/` 一级子目录且 `docs/decisions/` 无对应 `NNNN-*.md` 即由 code-reviewer 打回;新增一级子目录同时是 design-reviewer 的强制触发条件。

## 4. 与五条铁律的映射

| 铁律 | 承载文档 | 代码落点 |
|---|---|---|
| #1 编译优先(图编译 / C++ 编译期机制) | `architecture/execution-model.md`、`architecture/compiler-passes.md`;CPP-010~CPP-014 | `src/compiler/`、`src/runtime/`;全部 C++ 代码 |
| #2 语言支持(C++ 核心 + Python 绑定) | `standards/python-binding.md` | `python/` |
| #3 后端矩阵(统一 HAL、插件式) | `architecture/backend-hal.md`、`backends/` 全部 | `include/frame/hal/`、`src/backends/` |
| #4 语言策略 | `standards/language-policy.md` | 全仓库 |
| #5 两级复用(第三方库 + 可扩展接口) | `standards/reuse-policy.md`、`architecture/operator-system.md`、`architecture/compiler-passes.md` | `third_party/`、`FRAME_REGISTER_*` 注册宏、`examples/03_custom_op/` |

## 5. 典型数据流走读:构建一个 MLP 并执行

伪代码级调用序列(默认编译路径,eager 准入见 ARCH-011):

| 步骤 | 动作 | 发生位置 |
|---|---|---|
| 1 | 用户以 C++ Tensor API 或 Python 绑定描述 MLP(matmul → add → relu → …) | `include/frame/core/`、`python/` |
| 2 | 构图/trace 产出 `ir::Graph`,每个 Node 指向已注册 OpSchema | `src/ir/`、`include/frame/ops/` |
| 3 | `graph.verify()` 校验 SSA/无环/schema 匹配等不变量 | `src/ir/` |
| 4 | PassManager 按固定顺序跑标准管线(canonicalize → … → memory_planning) | `src/compiler/passes/` |
| 5 | backend_lowering(标准管线第 9 段,M7 已实化):目标后端名取自图 device,`BackendRegistry::instance().get("cuda")` 取后端(返回 `Result<Backend*>`),逐非 graph_input 节点判定 `KernelRegistry::find(op, backend)` 支持性,遇不支持算子返回带算子名错误(ARCH-031)——本 pass 不调用 `Backend::compile`,支持性判定失败交由 runtime 决策回退链 | `src/compiler/passes/`、`include/frame/hal/`、`include/frame/ops/` |
| 6 | 标准管线全绿后,`runtime::compile`(`include/frame/runtime/compile.h`,M7 已实化)调用 `Backend::compile(graph, options)` 产出 `Executable` | `include/frame/runtime/`、`src/runtime/`、`src/backends/cuda/` |
| 7 | 以缓存键(图哈希, 后端名, dtype/shape 签名, 选项哈希)存入编译缓存 | `src/runtime/` |
| 8 | 重复执行:`Executable::run(inputs, outputs, stream)` | `src/runtime/` 调度,`src/backends/cuda/` 执行 |
| 9 | `stream.synchronize()` 后读取输出 | `src/backends/cuda/` |

每一步的接口契约分别见 `architecture/ir-design.md`、`architecture/compiler-passes.md`、`architecture/backend-hal.md`、`architecture/execution-model.md`。
