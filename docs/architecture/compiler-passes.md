# 编译器 pass 管线与自定义 pass 接口

> **强制等级**:规范(MUST)
> **相关铁律**:#1 编译优先 / #5 两级复用
> **面向读者**:实现代码的工程师
> **最后更新**:2026-07-12

## 1. Pass 接口

`Pass` 属于虚函数白名单六类之一(CPP-010),声明于 `include/frame/compiler/pass.h`:

```cpp
class Pass {
 public:
  virtual ~Pass() = default;
  virtual std::string_view name() const = 0;
  virtual Status run(ir::Graph& graph) = 0;
};
```

- 【ARCH-050】【MUST】Pass 对象不得持有跨 `run` 调用的可变状态(必须可重入);配置一律经构造函数注入。判定方法:code-reviewer 检查 Pass 子类成员变量——`run` 中修改成员变量的 diff 按本条打回。

## 2. PassManager

声明于 `include/frame/compiler/pass_manager.h`,M6 已实化:

- `add_pass(std::unique_ptr<Pass>)`:实例直接加入,所有权转移;`add(std::string_view)`:按名经 `PassRegistry::create` 解析后加入。`add()` 签名不携带错误通道——解析失败时把错误暂存(保留首个错误,不被后续 `add`/`add_pass` 覆盖),在 `run()` 入口处原样返回、不执行任何 pass。两者均按加入顺序执行。
- 每个 pass 运行后自动调用 `graph.verify(ops::make_op_query())`(ARCH-022,见 `architecture/ir-design.md`),pass 自身不得跳过。`run()` 本身不做入图前的初始 verify——构图 API(`Graph::create_node` 等)已保证结构合法,这是调用方职责;若调用方传入的图本身非法,错误会挂在第一个 pass 名下返回(而非归因于图本身),这是刻意取舍,避免每次 `run()` 多付出一次全量 verify 的开销。
- 错误前缀纪律:pass `run()` 失败与 `graph.verify()` 失败,均以 `"pass '<name>': "` 为前缀包装该 `Status` 后返回;`verify()` 失败消息已自带 `"V<N>: "` 前缀(见 `architecture/ir-design.md` 第 4 章),这里只再加一层 pass 前缀,不重复加(单前缀纪律)。
- 调试开关 `set_dump_ir_after(std::string_view pass_name, std::ostream& os)`(v0 库内开关;命令行 `--dump-ir-after=<pass_name>` 接线归 M13 示例/工具,本章仅覆盖库内行为):命中 `pass_name` 的 pass 运行成功(`run()` 返回 OK)之后、`graph.verify()` 之前,把 `ir::dump_text(graph)`(IR 文本序列化格式,`architecture/ir-design.md` 第 3 章)写入 `os`。时机取舍:dump 选在 verify 之前,是因为 verify 失败时恰恰最需要看到当时的图内容用于诊断,若放在 verify 之后,verify 失败路径就永远看不到 dump。`pass_name` 以 `std::string` 拷贝存储、`os` 以指针存储(生命周期由调用方保证覆盖 `run()` 调用期间);未调用本方法 = 调试开关关闭。
- 只读观测面 `pass_names() const`:按装配序返回已加入各 pass 的 `name()`(`std::vector<std::string_view>`),供顺序断言与调试日志使用,不暴露 `Pass` 指针;`add()` 的延迟错误语义下,已成功加入的 pass 仍会出现在结果中(`pending_error_` 只影响 `run()` 是否执行,不影响此前已装配好的列表)。

## 3. 标准管线(固定顺序)

标准管线定义于 `include/frame/compiler/pipeline.h`,顺序如下(pass 名 = 全词文件名):

```
canonicalize → shape_inference → constant_folding
  → common_subexpression_elimination → dead_node_elimination
  → layout_assignment → operator_fusion → memory_planning → backend_lowering
```

- 【ARCH-053】【MUST】改动标准管线的顺序或增删标准 pass,必须先有已接受的 ADR。判定方法:`include/frame/compiler/pipeline.h` 中的管线定义与本节清单逐字一致;diff 触及该顺序且 `docs/decisions/` 无对应 ADR 即由 code-reviewer 打回。

每个标准 pass 一小节:一句话职责 + 前置条件 + 后置条件 + 桩状态。

### 3.1 canonicalize

- 职责:把等价图形归一化(常量输入位置、别名 op 重写),为后续 pass 减少模式数量。
- 前置:`verify()` 通过。后置:幂等——再跑一次图无变化(MUST,见第 6 章)。
- 状态:已实现(M8,src/compiler/passes/canonicalize.cpp);golden 与幂等测试见 tests/cpp/compiler/。

### 3.2 shape_inference

- 职责:自图输入向输出推断所有 Value 的 dtype/shape。
- 前置:图输入签名完整。后置:所有 Value 无 unknown 维;不可静态确定时返回错误(ARCH-013/ARCH-044)。
- **v0 实现口径 = 校验模式(m7-design-brief 决议点 1,已实化 M7)**:v0 构图 API
  (`create_node`)强制携带 `output_types`,图内永无 unknown 维(V5 已保证),故
  「推断」不写回类型,而是沿拓扑序对每个非 `graph_input` 节点重算——构造
  `NodeContext`(`op`/按位 `input_types`/`attrs` 借用指针)调
  `OpSchema::shape_infer()`,把重算结果与既有输出类型逐项比对:①shape 个数
  与节点输出个数比对;②逐位 shape 比对;③dtype 复核(输出 dtype == 第 0
  输入 dtype,0 输入节点豁免——`graph_input` 与 `constant`,后者自 M8 起
  实际使用该豁免;升宽/混合精度是未来 schema 扩展议题,v0 不支持);④全 Value(含 `graph_input`)的 shape 显式检查无动态维
  (ARCH-013/ARCH-044)——恒等透传的推断函数会原样复制输入侧已带的动态维,
  仅靠②的逐位比对拦不住"两侧同为动态维"的情形,故本步骤独立于②之外单独
  检查。任一环节不一致均返回英文错误(node op + 双方值)。无 schema 或 schema
  未设 `shape_infer` 同样报错(op 名)。写回模式(前端允许无类型构图后)是
  未来议题,届时再议 `Value` 写入口——v0 不新增 `Value::set_type`。
- 状态:已实化(M7,`src/compiler/passes/shape_inference.cpp`);golden 测试与
  错误路径用例见 `tests/cpp/compiler/test_shape_inference.cpp`。

### 3.3 constant_folding

- 职责:编译期求值全常量子图,替换为常量节点。
- 前置:shape_inference 完成。后置:图中无「输入全为常量且无 has_side_effect trait」的可折叠节点。
- 求值语义(M8 设计裁决,design-reviewer 通过):编译期求值恒用 cpu 参考
  kernel(ARCH-041 数值基准)——编译期数值 = cpu 参考数值,与目标后端无关;
  折叠属编译路径内部求值(非 eager 执行,不涉 ARCH-011 准入);折叠直调
  KernelFn 时 `KernelContext::stream = nullptr`,cpu kernel 不得解引用
  stream(自 M8 起为成文契约,此前仅为实现事实)。无 cpu kernel 的算子
  跳过折叠(不报错)。
- 状态:已实现(M8,src/compiler/passes/constant_folding.cpp);golden 与幂等测试见 tests/cpp/compiler/。

### 3.4 common_subexpression_elimination

- 职责:合并等价节点(同 op、同输入、同属性;commutative trait 参与输入归一化);带 has_side_effect trait 的节点除外。
- 前置:canonicalize 完成。后置:无两个等价的可合并节点。
- 状态:已实现(M8,src/compiler/passes/common_subexpression_elimination.cpp);golden 与幂等测试见 tests/cpp/compiler/。

### 3.5 dead_node_elimination

- 职责:删除对图输出不可达且无 has_side_effect trait 的节点(`graph_input`
  节点无条件豁免)。
- 前置:无。后置:所有存留节点均可达图输出、或带副作用、或为 `graph_input`
  节点(图签名不变式:v0 禁止 pass 改变图输入签名,未被使用的 graph_input
  一律保留;M8 设计裁决,design-reviewer 通过)。
- 状态:已实现(M8,src/compiler/passes/dead_node_elimination.cpp);golden 与幂等测试见 tests/cpp/compiler/。

### 3.6 layout_assignment

- 职责:为每个 Value 指定 layout,并在冲突处插入显式转换节点。
- 前置:shape_inference 完成。后置:所有 Value 的 layout 已确定且相邻节点 layout 兼容。
- v0 实现口径(M9 设计裁决,design-reviewer 通过):唯一具体 layout 为
  kRowMajor,经 `Graph::assign_layout` 窄写入口统一指派全部 Value(含
  `graph_input` 与图输出);单一 layout 下「冲突处插入转换节点」为空集,
  转换节点是未来多 layout 议题;后端 layout 偏好通道(CompileOptions 扩展
  或后端查询方法)留待 M11+/M14 多后端场景。
- 状态:已实现(M9,src/compiler/passes/layout_assignment.cpp);golden 与幂等测试见 tests/cpp/compiler/。

### 3.7 operator_fusion

- 职责:按 fusable/elementwise trait 把相邻节点合并为融合节点,减少 kernel 启动与内存往返。
- 前置:layout_assignment 完成。后置:融合前后数值等价(ARCH-052 强制测试)。
- v0 实现口径(M9 设计裁决,design-reviewer 通过;PLAN 第 7 节「融合节点的
  cpu 执行语义」决议点就此定案):融合节点 = 变长注册算子
  `fused_elementwise_internal`(线性链 attrs 编码,编码/解码 helper 单份共用;
  `_internal` 后缀不面向用户,不入绑定清单);cpu 执行语义 = **组合调用**
  ——逐 sub-op 调既有 cpu kernel,数值与未融合严格同源;否决解释执行
  (第二套数值实现引入等价性风险)。已知代价:cpu 融合执行仍产生中间临时
  张量、不减少内存往返——融合在 cpu 上是机制验证而非优化,真实收益属未来
  codegen 后端(M11+);融合 kernel 内部临时张量不在 memory_planning 计划内。
  融合候选:带 kElementwise+kFusable、无 attrs、单输出节点构成的线性链
  (长度 ≥2;链内中间输出单消费者且非图输出;融合时显式断言各 sub-op 具备
  cpu 参考 kernel,缺失则该链跳过);融合产物仅标 kElementwise、不标
  kFusable(v0 不做融合节点再融合)。
- 状态:已实现(M9,src/compiler/passes/operator_fusion.cpp);golden 与幂等/等价测试见 tests/cpp/compiler/。

### 3.8 memory_planning

- 职责:静态 shape 下 AOT 规划缓冲区复用与生命周期(铁律 #1① 的关键收益),产出供 runtime 落地的分配计划。
- 前置:shape/layout 全确定。后置:每个 Value 有确定的缓冲区偏移与生命周期区间。
- v0 实现口径(M9 设计裁决,design-reviewer 通过;照 §3.9 先例处理):
  `Pass::run(Graph&)` 无产物通道,pass 职责收窄为「计划可计算性验证」——
  调 `ir::compute_memory_plan`(分析函数与 MemoryPlan 结构位于 ir 层,分层
  依据见 `architecture/overview.md` §2.2),失败即报错,成功不改图放行;
  真正落地在 cpu `Executable`(compile 期同一函数复算存计划,run 期单块
  arena Storage + 字节偏移切片,替换 M7 朴素逐步分配)。arena 边界:仅规划
  **中间 Value**;`graph_input`(调用方内存)与凡在图输出列表中的 Value
  (独立缓冲,生命周期 = 整个 run,覆盖「既是图输出又被内部消费」情形)
  一律排除;中间 Value 的 last_use = 最大消费者拓扑下标;offset 按
  `frame::kDefaultAlignment`(64)向上对齐。成立依据:v0 无原位/别名语义
  (所有 kernel 分配新输出)。后置条件「每个 Value 有确定的缓冲区偏移与
  生命周期区间」在本口径下解释为:计划可由图**确定性推导**(同图两次推导
  逐字段相等)。
- 状态:已实现(M9,src/compiler/passes/memory_planning.cpp);golden 与幂等测试见 tests/cpp/compiler/。

### 3.9 backend_lowering

- 职责:后端支持性判定;`Executable` 由 runtime 编排经 `Backend::compile` 产出(`include/frame/runtime/compile.h`,见决议点 4/`architecture/execution-model.md` 第 2.1 节)——本 pass 自身不产出 `Executable`。后端遇不支持算子返回带算子名错误(ARCH-031),由 runtime 触发回退链决策(`architecture/execution-model.md` 第 5 章)。
- 前置:memory_planning 完成。后置:目标后端与全部节点的算子支持性已确认;不支持时 pass 本身即报错(回退改写是 runtime 层职责,不在本 pass)。
- **v0 实现口径(m7-design-brief 决议点 2,已实化 M7)**:pass 内不调用
  `Backend::compile`(`Pass::run(Graph&) → Status` 无产物通道,产物由 runtime
  编排入口在管线全绿后另行调用,见决议点 4)。职责收窄为两步:①目标后端名
  取自图 device(V6 单 device 保证唯一;空图/仅输入图——无非 `graph_input`
  节点——跳过整个判定,直接放行)且 `BackendRegistry::get` 可取,否则报错
  (消息含后端名);②逐非 `graph_input` 节点查
  `KernelRegistry::find(op, backend)`,缺失即返回带算子名与后端名的英文
  错误(ARCH-031 编译期报错落点)。**限定语**:上述 `KernelRegistry` 支持性
  判定仅适用于"逐 kernel 模式"后端(cpu/cuda/intel_gpu);"整图模式"后端
  (intel_npu,交厂商编译器整图处理)的支持性判定不经 `KernelRegistry`,而是
  内嵌在其自身 `Backend::compile` 内部按 ARCH-031 报错——按执行模式分流本
  pass 的判定逻辑是后续议题(前置登记见 `docs/plan/milestones.md` M15 节)。
  `standard_pipeline(backend)` 的 `backend` 参数 v0 继续忽略(目标后端名改
  自图 device 取得,参数留 M8+ 多后端分区场景)。
- 状态:已实化(M7,`src/compiler/passes/backend_lowering.cpp`);golden 测试
  与错误路径用例见 `tests/cpp/compiler/test_backend_lowering.cpp`。

## 4. 自定义 pass 注册

- 注册宏:`FRAME_REGISTER_PASS(PassClass)`(v0 为单参数;宏随 `include/frame/compiler/pass.h` 提供)。
- 注册类的硬性要求:必须满足 `pass.h` 的 `PassType` concept——派生自 `Pass`、默认可构造、提供 `static constexpr kName`(不满足即注册处编译错误)。惯用途径是继承 CRTP 基类 `PassBase<Derived>` 并实现 `run_impl(ir::Graph&)`,`name()`/`run()` 由基类编译期生成。
- v0 语义:注册使 pass 可按名查找;**不自动进入标准管线**,由调用方经 `PassManager::add_pass` 显式插入。stage/after 声明式排序机制(如 `stage=optimize, after={"common_subexpression_elimination"}`)为后续 ADR 议题,实现前禁止自行发明。
- 后端私有 pass:代码放 `src/backends/<X>/passes/`,仅由该后端在 lowering 阶段自行插入,不得进入标准管线(进入需 ADR,ARCH-053)。

## 5. pass 测试规范

- 【ARCH-051】【MUST】每个 pass 必须有 golden 测试:输入 IR 文本 → 运行 pass → 与期望 IR 文本逐字比对。测试放 `tests/cpp/compiler/`,数据放 `tests/cpp/compiler/testdata/`。判定方法:新增/修改 pass 的 PR 必须包含对应 golden 测试文件,缺失由 code-reviewer 打回;CI 运行 `tests/cpp/compiler/` 全绿。
- 【ARCH-052】【MUST】operator_fusion 等改变执行粒度的 pass,额外要求数值等价测试:融合前后执行结果在容差内一致,容差唯一来源为 BUILD-011(`standards/build-and-test.md`)。判定方法:`tests/cpp/compiler/` 存在对应数值等价用例且通过。

## 6. pass 编写 checklist

逐项勾选:

1. [ ] 不直接调用任何设备 API(ARCH-001:compiler 层禁区)。
2. [ ] 只经 `Graph` 公开 API 改图(ARCH-021)。
3. [ ] 在文档注释中声明前置依赖的 pass(如「须在 shape_inference 之后」)。
4. [ ] 幂等:连跑两次结果不变——canonicalize 类 MUST,优化类 SHOULD(偏离写入 PR 描述)。
5. [ ] golden 测试就位(ARCH-051);融合类补数值等价测试(ARCH-052)。
6. [ ] 无跨 `run` 可变状态(ARCH-050)。
7. [ ] **警告**:改动标准管线顺序需 ADR(ARCH-053),自定义 pass 不要求进入标准管线。
