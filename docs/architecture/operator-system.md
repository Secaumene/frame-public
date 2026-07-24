# 算子注册、分发与自定义算子扩展

> **强制等级**:规范(MUST)
> **相关铁律**:#1 编译优先 / #5 两级复用
> **面向读者**:实现代码的工程师
> **最后更新**:2026-07-04

## 1. 三个概念的分离

| 概念 | 是什么 | 与后端的关系 | 注册表 |
|---|---|---|---|
| **OpSchema**(算子定义) | 名字、输入/输出数量与约束、属性 schema、trait 集合、shape 推断函数、可选 decomposition | 与后端无关,全局唯一 | `OpRegistry`(`include/frame/ops/op_registry.h`) |
| **Kernel**(算子实现) | 某后端上该算子的具体实现函数 | 每后端各自注册 | `KernelRegistry`(`include/frame/ops/kernel_registry.h`) |
| **Dispatch**(分发) | 从 `(op, backend)` 查到 Kernel 的过程;dtype 在 kernel 内部编译期展开 | 运行时仅在算子边界查一次表 | — |

不得混用:OpSchema 里不写实现,Kernel 里不改 schema,分发逻辑不散落在调用方。

## 2. OpTrait 封闭枚举

OpSchema 携带 trait 集合,声明于 `include/frame/ops/op_schema.h`:

```cpp
enum class OpTrait : uint8_t {
  kElementwise,      // 逐元素:输出 shape 等于输入 shape,无跨元素依赖(fusion 候选依据)
  kFusable,          // 允许被 operator_fusion pass 合并
  kHasSideEffect,    // 有副作用:禁止被 CSE/DCE 消除或重排
  kCommutative,      // 输入可交换(CSE 归一化依据)
};
```

- 【ARCH-043】【MUST】OpTrait 是封闭枚举,仅允许 {kElementwise, kFusable, kHasSideEffect, kCommutative} 四项(k 前缀与 `DTypeCode` 命名一致);新增 trait 必须先修订本文档与 `include/frame/ops/op_schema.h` 并通过 design-reviewer。判定方法:`op_schema.h` 中 `enum class OpTrait` 的枚举项与本节清单逐一相同,code-reviewer 对新增枚举项且未同步本文档的 diff 按本条打回。

## 3. 注册宏与注册表

- `FRAME_REGISTER_OP(name)`:builder 链式 API 声明输入/输出数量与约束、属性 schema、trait 标注、`shape_infer(fn)`(声明见 `include/frame/ops/op_schema.h`)、可选 `decomposition(fn)`(`DecomposeFn` 签名见下)。
- 变长输入(M9,design-reviewer 独立设计门通过):`variadic_input(name, doc,
  min_count)` 声明**尾随变长输入组**。硬约束:每 schema 至多一个变长组;必须
  位于全部定长 `input()` 声明之后;`min_count ≥ 0`;违例在 builder 期启动
  fail-fast(与 `register_op` 同口径,共用同一 fatal 出口)。输出侧无变长。
  查询面:`has_variadic_inputs()` / `min_input_count()`(= 定长数 + min_count)。
  v0 唯一使用者:`fused_elementwise_internal`(pass 产物算子,`_internal`
  后缀天然不落 PY-021「面向用户」判定)。
- `FRAME_REGISTER_KERNEL(op, backend_string, fn)`:注册键 = `(op, backend)` 二元组,**不含 dtype**——dtype 差异在 kernel 内部经 `dispatch_dtype` 编译期展开(第 4 章)。

`shape_infer(fn)` 与 `decomposition(fn)` 均接收 `const NodeContext&`(定义于 `include/frame/ops/op_schema.h`)——节点的只读视图,借用 `ir::Node` 的输入类型与属性表(借用契约:`attrs` 指针仅在调用期间有效):

```cpp
struct NodeContext {
  std::string_view op;
  std::vector<ir::TensorType> input_types;   // 按位输入类型(值持有)
  const std::unordered_map<std::string, ir::AttrValue>* attrs = nullptr;
  template <typename T> const T* attr(std::string_view name) const;  // 缺失/类型不符 → nullptr
};

using DecomposeFn = Result<ir::Graph> (*)(const NodeContext&);
```

`DecomposeFn` 产出语义等价微图:`graph_inputs` 按位对应本算子输入、图输出按位对应本算子输出;纯函数、不修改既有图(ARCH-021)。运行时回退经此分解执行微图(M10),编译期展开由 pass 负责。

- 【ARCH-040】【MUST】算子名全小写下划线英文(正则 `^[a-z][a-z0-9_]*$`,经 `ir::matches_op_name_charset` 校验——ops 注册与 ir 构图(`Graph::create_node`)共用 ir 层公开的这一份实现,依赖方向 ops→ir,ARCH-001 合法),全局唯一;重名注册与非法名在启动期报错(英文消息)而非静默覆盖。判定方法:`OpRegistry::register_op` 在注册时校验,非法名/重名/保留名(`ir::kGraphInputOp`/`ir::kGraphOutputMarker`)任一违例即在启动期 fatal(向 stderr 输出含算子名与违例原因的英文诊断后终止进程,而非返回可恢复的 `Status`——注册发生在静态初始化期,无人能处理 `Status` 返回值);`tests/cpp/ops/` 有「非法名被拒」「重名被拒」用例(经死亡测试断言进程终止)。
- `ir::OpQuery make_op_query()`(声明于 `include/frame/ops/op_registry.h`):把 `OpRegistry` 接线为 `Graph::verify` 所需的 `ir::OpQuery`(V3/V4,见 `architecture/ir-design.md` 第 4 章)——`op_registered` 查 `OpRegistry::find(name) != nullptr`;`check_schema` 校验输入/输出数量满足 schema 输入约束(定长 schema 恒等比较;带尾随变长输入组的 schema 按下限 `min_input_count()` 判定,M9)、必需属性存在、属性类型匹配(`ir::attr_type_of`)、拒绝未声明属性。`check_schema` 返回的错误消息**不带 `"V4: "` 前缀**——`Graph::verify`(`src/ir/graph.cpp`)统一加前缀后原样透传,避免双重前缀破坏 golden 对齐。
- 【ARCH-041】【MUST】每个新 OpSchema 必须同时提供:①shape 推断函数;②CPU 参考实现(`FRAME_REGISTER_KERNEL(op, "cpu", fn)`)。参考实现可以慢,用途是数值校验与回退链终点(见 `architecture/execution-model.md` 第 5 章)。判定方法:code-reviewer 按第 6 章 checklist 逐项核对;缺 shape 推断或缺 cpu kernel 即打回。

内置算子 schema 桩位于 `src/ops/schemas/`。

## 4. 分发机制与编译期优先(铁律 #1② 的落地)

- 运行时分发仅发生一次:在算子边界按 `(op, backend)` 查 KernelRegistry。
- kernel 内部对 dtype 的差异一律用模板参数化,经 `dispatch_dtype`(编译期展开工具,声明于 `include/frame/core/dtype.h`)把运行时 dtype 值映射为模板实例:

  ```cpp
  // 注册键不含 dtype;dtype 在 kernel 内一次性展开
  dispatch_dtype(dtype, [&]<typename T>() { relu_impl<T>(in, out, n); });
  ```

- 【ARCH-042】【MUST NOT】kernel 内层循环中禁止出现按 dtype 或设备的运行时分支(switch/if 逐元素判断)。判定方法:运行 `scripts/check_iron_rules.sh`(kernels/ 目录 dtype 分支检查);code-reviewer 复核 kernel 模板签名。

## 5. 动态 shape 边界(v0)

- 【ARCH-044】【MUST】shape 推断函数无法静态确定输出维度时,必须返回错误(拒绝),不得输出 unknown 维;禁止注册动态 shape 算子变体、禁止引入符号维度机制。动态 shape 支持是 ADR 议题(与 ARCH-013 同源),实现前必须有已接受的 ADR。判定方法:code-reviewer 对引入符号维度/动态 shape 注册的 diff 按本条打回;`tests/cpp/ops/` 有「shape 推断遇不可静态确定维度返回错误」用例。

## 6. 自定义算子扩展流程(7 步 checklist)

演示锚点见 `examples/03_custom_op/main.cpp`(schema + kernel 两步注册)。

1. [ ] 按 REUSE-001 五步搜索:既有算子、可组合的 decomposition、第三方库覆盖;搜索记录贴入 PR「复用检查」段。
2. [ ] `FRAME_REGISTER_OP` 写 OpSchema:输入/输出约束、属性、trait 标注(第 2 章)、shape 推断函数。
3. [ ] 写 CPU 参考实现并 `FRAME_REGISTER_KERNEL(op, "cpu", fn)`(ARCH-041)。
4. [ ] 按目标后端指南(`docs/backends/<X>.md` 第 6 章)写后端 kernel 并注册。
5. [ ] (可选)注册 `decomposition(fn)`(`DecomposeFn` 签名见第 3 章),让未实现该算子的后端可走回退链。
6. [ ] 补 GoogleTest:后端实现 vs CPU 参考实现数值对比,容差唯一来源为 BUILD-011(`standards/build-and-test.md`);测试放 `tests/cpp/ops/`。
7. [ ] (可选)pybind11 暴露 + 更新 `.pyi` 存根(PY-020)+ pytest 用例(`tests/python/`)。

## 7. 编译路径下算子如何被消费

backend_lowering pass(见 `architecture/compiler-passes.md`)判定 Node 到后端的支持性;`Executable` 由 runtime 编排入口在标准管线全绿后经 `Backend::compile` 产出(`include/frame/runtime/compile.h`,M7 已实化,见 `architecture/compiler-passes.md` 3.9)。后端可选两种模式,均必须实现 `Executable` 接口:

| 后端 | 执行模式 | 说明 |
|---|---|---|
| cpu | 逐 kernel 拼装 | 参考后端,永远启用 |
| cuda | 逐 kernel 拼装 + 自研图编译 | 详见 `docs/backends/cuda.md` |
| intel_gpu | 逐 kernel 拼装 + 自研图编译 | 详见 `docs/backends/intel-gpu.md` |
| intel_npu | 整图交厂商编译器(OpenVINO) | 无逐算子 eager kernel,详见 `docs/backends/intel-npu.md` |
| ascend | 待定(ADR-0005,推迟至 v2.0) | aclnn 逐算子拼装与 GE 整图编译均为候选,不做承诺;详见 `docs/backends/ascend.md` |

- 逐 kernel 模式:lowering 逐节点选 kernel,runtime 按拓扑序拼装执行计划。
- 整图模式:lowering 把子图整体转换为厂商 IR,交厂商编译器产出 `Executable`。

两种模式对上层(runtime、编译缓存、回退链)完全透明。
