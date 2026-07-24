# 图 IR 设计

> **强制等级**:规范(MUST)
> **相关铁律**:#1 编译优先 / #5 两级复用
> **面向读者**:实现代码的工程师
> **最后更新**:2026-07-12

## 1. 设计目标与非目标

**目标**:

1. 可验证:`verify()` 可机械检查全部不变量(第 4 章清单)。
2. 可序列化为确定性文本:同图同输出,作为 pass golden 测试的基础(ARCH-051)。
3. 显式携带 shape/dtype/layout/device:编译期可完成全部推断与规划,不留运行时悬念。

**非目标(v0)**:

- 不做多层 IR(与 MLIR 的关系见第 5 章)。
- 不做控制流(if/while):控制流支持是 ADR 议题,实现前必须有已接受的 ADR。
- 不做动态 shape(ARCH-013,见 `architecture/execution-model.md`)。

## 2. 核心对象模型

接口清单级描述,不含实现;头文件桩 `include/frame/ir/{attribute,node,graph,serialization}.h` 必须与本节逐一对应,不一致以本文档为准并同 PR 修正。

### 2.1 Graph

- 数据:节点有序列表(保持拓扑序)、图 inputs/outputs、名字;私有
  `node_set_`(`std::unordered_set<const Node*>`,决议点 5-③,M7 已实化)随
  `create_node`/`add_graph_input`/`erase_node` 同步维护,供图归属判定
  (`contains_node`)O(1) 查询——`topo_order_` 仍是拓扑序权威,`node_set_`
  只服务"是否属于本图"这一类查询,不对外暴露。
- 方法:`create_node`(构图标准入口)、`add_graph_input`、`mark_output(Value*)`、
  `mark_output(Node*, int32_t)`(M7 新增,决议点 5-②;`node` 须属于本图、
  `output_index` 须落在该节点输出个数区间内,合法时等价于
  `mark_output(node->output(output_index))`,面向 decomposition 等外部合法
  可变用途)、`erase_node`、`topological_order`、`verify(const OpQuery&)`、
  `verify_structure`、`replace_all_uses(Value* from, Value* to)`(M8/M9,全图 use 与图输出重定向,TensorType 四元组须相等;拓扑序不变式的重定位豁免(M9 收敛为单一严格条件,0 输入节点为其平凡实例):to 的 producer 的每个输入的 producer 拓扑下标**严格小于** from 的 producer 下标时,将其 rotate 至 from 的 producer 原位后再替换;不满足即报错)、`swap_node_inputs(Node*, i, j)`(M8,输入槽交换,语义责任在调用方)、`assign_layout(Value*, Layout)`(M9,layout 窄写入口:仅允许 kUnknown→具体或同值幂等重指派,具体→不同具体报错;不开放 dtype/shape/device 写口,维持 M7「不新增 `Value::set_type`」边界)。**`add_node` 已移除(M7,决议点 5-④)**:其"pass 变换
  场景"实为 `create_node` 已覆盖(pass 持 `Graph&` 可直接建节点接线);若 M8+
  变异 pass 出现真实需求,以受控形态回归,届时须过 design-reviewer。
- `kGraphInputOp`(图输入节点)与 `kGraphOutputMarker`(图输出标记行,序列化语法关键字,
  见第 3 章)均为 ir 层保留 op 名,`create_node` 拒绝以二者中任一名建节点,
  图输入只能经 `add_graph_input` 创建。此外 `create_node` 校验 op 名字符集
  `^[a-z][a-z0-9_]*$`,经 `frame::ir::matches_op_name_charset`(声明于
  `include/frame/ir/graph.h`)校验——ops 层 `OpRegistry::register_op`(ARCH-040)共用
  这一份实现(REUSE-002),依赖方向为 ops→ir(ARCH-001:ir 不依赖 ops,反向依赖不受限)。
- `OpQuery`:V3/V4 校验所需的算子注册信息经此结构体的两个回调
  (`op_registered`/`check_schema`)注入(ARCH-001:ir 不依赖 ops,回调由上层构造)。
- `verify(const OpQuery&)` 返回 `Status`,全量校验 V1—V7,必查项见第 4 章;
  `verify_structure()` 是不依赖 `OpQuery` 的结构子集(V1/V2/V5/V6/V7)。V1(SSA)
  校验除"producer 属于本图"外,另对每个被引用 Value 指针做相等判定
  `value == producer->outputs().data() + value->output_index()`(M7 已实化,
  决议点 5-①,design-reviewer 建议①:用指针相等而非区间比较,零 UB),拒绝
  悬挂的栈拷贝 Value 通过校验。

### 2.2 Node

- 数据:op 名(字符串,指向已注册 OpSchema)、输入 `Value*` 列表、输出 `Value` 列表、属性字典(`name -> Attribute`)。
- 方法:`set_attr`(构图期属性写入)、`find_attr`(按名取原始属性)、`attr<T>`
  (按名取强类型属性)、`attrs`(只读枚举全部属性;**枚举顺序不确定**,任何
  确定性输出——序列化文本、按字典序排列的错误消息列表等——必须由调用方
  自行按名排序;字典序是序列化层的契约,不是 `Node` 本身的契约)、`outputs()
  const`(只读整体访问,返回 `const std::vector<Value>&`)、`output(int32_t
  index)`(M7 新增,决议点 5-②;非 const,仅返回单元素指针 `Value*`,越界
  返回 `nullptr`)。**`outputs()` 的非 const 重载已移除(M7)**:收紧目标是
  防止 `push_back`/`resize` 等操作使已发出的 `Value*` 失效、破坏 SSA 指针
  稳定性;外部合法可变用途(如 decomposition 内标记新建节点的输出)改经
  `output(int32_t)` 或 `Graph::mark_output(Node*, int32_t)`。

### 2.3 Value

- SSA 风格:每个 Value 恰有一个 producer。
- 携带 `TensorType{dtype, shape, layout, device}`;其中 `device` 为核心层值类型 `Device{backend, index}`(见 `architecture/backend-hal.md` 第 1 章)。
- shape 数据结构保留动态维标记位以便未来扩展,但 v0 下 `verify()` 一律拒绝动态维(ARCH-013)。

### 2.4 Attribute

- 【ARCH-020】【MUST】属性类型限于封闭集合 = {int64, double, string, bool, int64 数组, double 数组, dtype, shape};不得超出。扩展需先修订本文档并通过 design-reviewer。判定方法:`include/frame/ir/attribute.h` 中的类型集合(`std::variant` 备选项)与本清单逐一相同,code-reviewer 对超出集合的 diff 按本条打回。

## 3. 文本序列化格式

单行一节点,格式:

```
%<id> = <op>(%<in0>, %<in1>, ...) {<attr>=<value>, ...} : <dtype>[<shape>]@<backend>:<index>
```

示例(MLP 片段):

```
%0 = graph_input() : f32[32,784]@cuda:0
%1 = graph_input() : f32[784,256]@cuda:0
%2 = matmul(%0, %1) : f32[32,256]@cuda:0
%3 = relu(%2) : f32[32,256]@cuda:0
graph_output(%3)
```

注:上例为简化插图,dtype 使用 `f32` 简写;`dump_text` 实际输出使用 `DType::name()`
全称(如 `float32`),权威格式以 `include/frame/ir/serialization.h` 头注释为准。double
属性/数组元素的文本格式(`std::to_chars` 最短往返表示)同样以该头注释为唯一权威。

用途与要求:

1. 调试 dump(PassManager 的 `--dump-ir-after=<pass_name>` 开关输出此格式)。
2. pass 的 golden 测试(输入文本 → pass → 与期望文本比对,见 ARCH-051)。
3. **确定性**:同一图任何两次序列化输出逐字节相同(id 按拓扑序分配、属性按名字典序输出)。判定方法:同图两次序列化对比的单元测试。
4. `parse_text` 提供 golden 测试的输入读入能力,仅接受 `dump_text` 产出的规范形态(非通用文本 IR 解析器);格式未尽细节以 `include/frame/ir/serialization.h` 头注释为唯一权威。
5. `format_attr_value`(M8 提升为公开确定性属性文本化函数,为 dump_text 与 CSE 等价键生成共用)。
6. layout 尾缀扩展(M9):`layout != kUnknown` 时类型后缀追加 layout 标记;
   `kUnknown` 不输出——与既有格式逐字节兼容,既有 golden 不受影响,第 3 条
   确定性契约保持。token 确切位置与 parse 规则以 `serialization.h` 头注释为
   唯一权威,本章仅登记扩展存在。编译缓存键取管线运行前输入图(恒 kUnknown),
   尾缀不进缓存键。

## 4. 验证器不变量清单

`verify(const OpQuery&)` 必查项,逐条编号(golden 测试与错误消息以编号对齐;错误消息英文,LANG-005):

| # | 不变量 |
|---|---|
| V1 | 每个 Value 恰有一个 producer(SSA) |
| V2 | 图无环 |
| V3 | 每个 Node 的 op 名已在 OpRegistry 注册(`graph_input` 节点豁免,改查:0 输入、恰 1 输出、该输出已登记于 `inputs()`,见下方注记) |
| V4 | 每个 Node 的输入/输出数量与属性满足其 OpSchema 约束(数量约束:定长恒等 / 变长下限,M9;`graph_input` 节点豁免,见下方注记) |
| V5 | 所有 Value 的 dtype/shape 已完成推断且无 unknown(v0 静态模式,ARCH-013) |
| V6 | device 一致性:单图单 device;跨 device 数据移动必须显式表示为 `copy` 节点 |
| V7 | 全部属性类型在 ARCH-020 封闭集合内 |

注记:V3/V4 经 `OpQuery` 回调注入执行(ARCH-001);`graph_input` 为 ir 层保留名,豁免
V3/V4 改做结构检查(V3 行注明);`verify_structure()` 覆盖结构子集 V1/V2/V5/V6/V7。

## 5. 与 MLIR 的关系(双轨并行,ADR-0002 已接受)

用户已裁决 IR 走双轨并行(ADR-0002,状态:已接受):

- **v0 主线**:自研轻量 in-memory 图 IR(即本文档),理由:四后端接入初期,MLIR 的学习与构建成本高,轻量 IR 足以支撑静态 shape 推理管线。
- **并行预研赛道**:MLIR 作为 lowering 基础设施或方言互转的候选,保留接入扩展点。
- 接入边界:
  - CMake 开关 `FRAME_ENABLE_MLIR`(默认 OFF)隔离全部 MLIR 相关构建;
  - MLIR 相关代码只允许出现在未来的 `src/compiler/mlir/` 目录(当前骨架不建该目录);
  - **当前不引入任何 MLIR 依赖**;实际引入依赖时仍受 REUSE-010 约束。

- 【ARCH-023】【MUST NOT】`src/compiler/mlir/` 之外的任何文件不得 include MLIR 头文件;CMake 中 MLIR 相关的 `find_package`/链接必须位于 `if(FRAME_ENABLE_MLIR)` 保护块内;`FRAME_ENABLE_MLIR=OFF` 的构建不得出现 MLIR 依赖。判定方法:code-reviewer 对出现 `mlir` include 或不受开关保护的 MLIR CMake 语句的 diff 按本条打回;`cpu-only` preset 构建成功即证明默认路径无 MLIR 依赖。

## 6. IR 修改纪律

- 【ARCH-021】【MUST】pass 之外的代码不得修改已构建的 `Graph`(构图阶段除外);全部图变换收敛在 `src/compiler/` 的 Pass 中。判定方法:code-reviewer 检查 `src/{runtime,backends}/` 中对 Graph 非 const 接口的调用,发现即打回。
- 【ARCH-022】【MUST】每个 pass 运行后自动调用 `verify()`,由 PassManager 统一保证,pass 自身不得跳过。判定方法:PassManager 的单元测试覆盖「pass 产出非法图时管线立即报错」;code-reviewer 对绕过 PassManager 直接跑 pass 的代码按本条打回。
