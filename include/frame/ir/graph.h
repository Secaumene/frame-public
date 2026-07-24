#pragma once
// Graph:静态计算图 —— 静态编译的中心数据结构(铁律 #1①)。
// eager 模式也先建微图再走编译路径(见 docs/architecture/execution-model.md)。
// 值语义,无虚函数。图变换一律收敛在 src/compiler/ 的 Pass 中(ARCH-021)。

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <frame/core/macros.h>
#include <frame/core/status.h>
#include <frame/ir/node.h>

namespace frame::ir {

// ir 层保留 op 名:图输入节点。所有权在 ir 层(序列化格式见
// docs/architecture/ir-design.md 第3章),不进入 OpRegistry;create_node
// 拒绝以此名建节点——图输入只能经 add_graph_input 创建。
inline constexpr std::string_view kGraphInputOp = "graph_input";

// ir 层保留名:纯序列化语法关键字(图输出标记行 "graph_output(%<id>)" 的前缀,
// 见 include/frame/ir/serialization.h 头注释),永不作为节点 op 名——否则
// 零输出节点行(`graph_output(...)`)与图输出标记行同形,parse_text 会误读。
// create_node 同样拒绝以此名建节点。
inline constexpr std::string_view kGraphOutputMarker = "graph_output";

// op 名字符集校验:^[a-z][a-z0-9_]*$(首字符小写字母,后续小写字母/数字/
// 下划线)。ir 层构图(create_node)与 ops 层注册(OpRegistry::register_op,
// ARCH-040)共用本函数这一份实现(REUSE-002),避免同签名同函数体的第二份复制;
// ops→ir 属合法依赖方向(ARCH-001:ir 不依赖 ops,ops 依赖 ir 不受限)。定义见
// src/ir/graph.cpp。
FRAME_API bool matches_op_name_charset(std::string_view name) noexcept;

// V3/V4 的算子注册信息经回调注入(ARCH-001:ir 不依赖 ops,回调由上层——通常是
// 持有 OpRegistry 的调用方——构造并传入)。两个回调均须非空,Graph::verify() 在
// 任一回调缺失时立即返回错误(fail-closed),不静默跳过 V3/V4。
// 取舍说明:verify() 非编译热路径(构图/pass 后一次性调用,不在算子执行内层
// 循环),std::function 的间接调用开销可忽略不计;这与 compiler/pass.h 中
// PassRegistry 工厂改用函数指针(零开销考量,面向可能高频调用的注册路径)是
// 不同场景,此处无需比照该先例改用函数指针。
struct OpQuery {
  std::function<bool(std::string_view op_name)> op_registered;  // V3:op 是否已注册
  std::function<Status(const Node& node)> check_schema;         // V4:node 是否满足 schema 约束
};

// Graph:节点有序列表(保持拓扑序)+ 图 inputs/outputs + 名字。
class FRAME_API Graph {
 public:
  Graph() = default;
  explicit Graph(std::string name) : name_(std::move(name)) {}

  std::string_view name() const { return name_; }

  // 创建节点并接线,所有权归 Graph,返回其裸指针。inputs 必须是本图内既有
  // Value(逐个校验 producer 已属于本图——构造性防环:新节点只能引用已存在
  // 节点的输出,因而始终能安全追加到拓扑序末尾);output_types 决定输出个数
  // 与各输出类型。op 等于 kGraphInputOp/kGraphOutputMarker 时返回错误(二者
  // 均为 ir 层保留名;图输入只能经 add_graph_input 创建)。op 还须匹配字符集
  // `^[a-z][a-z0-9_]*$`(经本文件的 matches_op_name_charset 校验,ops 层
  // OpRegistry::register_op 共用同一实现,ARCH-001:ops→ir 依赖方向合法),
  // 否则同样返回错误。构图标准入口(除 graph_input 外的全部节点均经本方法建立)。
  Result<Node*> create_node(std::string op, std::vector<Value*> inputs,
                            std::vector<TensorType> output_types);

  // 建立一个 kGraphInputOp 节点(0 输入、恰 1 输出),将其唯一输出登记进
  // inputs() 并返回该输出的裸指针。
  Result<Value*> add_graph_input(TensorType type);

  // 登记图输出;value 必须属于本图(其 producer 是本图内节点的输出),否则
  // 返回错误。同一 Value 可重复登记(每次调用各占序列化中的一行)。
  Status mark_output(Value* value);

  // 按 (节点, 输出序号) 登记图输出的重载(决议点 5-②):node 必须属于本图、
  // output_index 必须落在该节点输出个数区间内,否则返回错误;合法时等价于
  // mark_output(node->output(output_index))。面向外部合法可变用途(如
  // decomposition 内标记新建节点的输出,见
  // src/ops/schemas/elementwise.cpp::square_decompose)——非 const 的
  // Node::outputs() 已收紧为仅返回单元素指针的 output(int32_t),本重载是该
  // 用途在 Graph 侧的配套入口。
  Status mark_output(Node* node, int32_t output_index);

  // 删除节点:节点的全部输出若被其他节点的输入或图输出引用,返回错误
  // (拒绝制造悬挂引用);否则从节点列表/拓扑序/图输入列表(如适用)中一并
  // 移除。
  Status erase_node(Node* node);

  // 受控图变异 API①(M8/M9,pass 变换场景专用,ARCH-021):把全图内对 from 的
  // 引用整体改写为 to,含图输出列表(图输出也是一种 use)。校验(逐条,
  // 违例返回 InvalidArgument):from/to 均非空且其 producer 属本图;
  // from != to;二者 TensorType 四元组(dtype/shape/layout/device)完全相等。
  //
  // 拓扑序不变式与三种情形(裁决修订 2,M9,替换 M8 版"唯一豁免"表述):要求
  // to 的 producer 在 topological_order 中位于 from 的 producer 之前(消费者
  // 天然在拓扑序中靠后,替换后 to 仍需先于全部旧 from 消费者产出)。
  //   ①二者 producer 相同(同一节点的不同输出):天然同时产出,不存在先后
  //     顺序问题,无需处理。
  //   ②to 的 producer 天然位于 from 的 producer 之前(topo 下标更小):不变式
  //     已满足,无需处理。
  //   ③重定位豁免(裁决修订 2,严格版,取代 M8 的"0 输入节点"豁免——0 输入
  //     节点是本条件的平凡实例,vacuously true):当 to 的 producer 的**每一个
  //     输入**的 producer 拓扑下标均**严格小于** from 的 producer 下标时,
  //     允许经 std::rotate 把 to 的 producer 重定位到 from 的 producer 原
  //     位置(rotate(first=from_index, middle=to_index, last=to_index+1),
  //     等价于把 to 的 producer 从 to_index 挪到 from_index、原
  //     [from_index, to_index) 区间整体右移一位)。安全论证:该 rotate 只
  //     置换 [from_index, to_index] 区间内部的相对顺序(区间外元素位置不变;
  //     区间内除 to 的 producer 外,其余元素相对顺序保持不变、只整体右移一
  //     位,故它们各自与"位于区间外(下标 < from_index)的输入"或"区间内先于
  //     自己的输入"之间的先后关系不受影响);to 的 producer 自身的全部输入
  //     的 producer 下标均 < from_index,不在被移动的 [from_index, to_index]
  //     区间内,重定位后原样保持在 to 的 producer 的新位置(from_index)之前
  //     ——rotate 后 to 的所有依赖仍在其前。删除"或等于"支(等于支会把
  //     依赖 from 的消费者错误地排到 from 的 producer 之前,引入非法拓扑
  //     序,已废弃)。
  //   不满足①②③任一情形则返回错误,不做任何改动。
  //
  // use 扫描为全图节点输入 vector + 图输出列表的线性扫描(Value 无 uses 列表,
  // O(V·E),与 verify_structure() 内 check_ssa 同量级,v0 可接受)。调用后
  // from 可能 uses 清零,是否删除该节点由调用方决定,本方法不做任何删除。
  Status replace_all_uses(Value* from, Value* to);

  // 受控图变异 API②(M8,canonicalize pass 专用):交换 node 的第 i/j 个输入
  // 槽位,不触碰 Value/拓扑序/其余状态(同一组 SSA 使用仅换位)。node 须属
  // 本图,i/j 须落在 [0, node->inputs().size()) 区间内且互不相同。语义责任
  // 在调用方——本方法不校验该算子是否可交换(commutative trait),误用于
  // 不可交换算子会静默改变图语义,且 graph.verify() 不会拦截这一点。
  Status swap_node_inputs(Node* node, int32_t i, int32_t j);

  // 受控图变异 API③(M9,决议点 A,layout_assignment pass 专用窄写入口,与
  // 上方两个受控变异 API 同列):为 value 指派 layout。v0 未开放通用的
  // Value::set_type 写口(M7 裁决边界),本方法是唯一允许改写 TensorType 的
  // layout 字段的入口,不开放 dtype/shape/device 写口。
  // 校验(逐条,违例返回 InvalidArgument):value 非空且其 producer 属本图;
  // layout != Layout::kUnknown(不允许"指派回 unknown",与"指派"语义矛盾)。
  // 写入规则:value 当前 layout 为 kUnknown → 直接写入(首次指派);当前
  // layout 与待写入 layout 相同 → 幂等重指派,原样返回 Ok,不做任何改动;
  // 当前 layout 是其他具体值且与待写入 layout 不同 → 返回错误(消息含双方
  // layout,布局转换须经显式转换节点,v0 单一 layout 下该场景不存在)。
  Status assign_layout(Value* value, Layout layout);

  // 拓扑序节点视图。
  const std::vector<Node*>& topological_order() const { return topo_order_; }

  const std::vector<Value*>& inputs() const { return inputs_; }
  const std::vector<Value*>& outputs() const { return outputs_; }

  // 全量校验不变量 V1—V7(SSA/无环/op 已注册/schema 约束/无 unknown 维/单
  // device/属性类型封闭),错误消息英文并以 "V<N>: " 前缀对齐第4章编号
  // (LANG-005)。query 的 op_registered/check_schema 任一为空,立即返回
  // 错误(fail-closed,消息注明缺失的回调)。graph_input 节点豁免 V3/V4,
  // 改做结构检查(见 docs/architecture/ir-design.md 第4章注记)。
  Status verify(const OpQuery& query) const;

  // 结构子集校验:V1/V2/V5/V6/V7(不含依赖 OpQuery 的 V3/V4),用于构图期
  // 自检与 ir 层内部使用,不依赖 ops 层信息(ARCH-001)。
  Status verify_structure() const;

 private:
  std::string name_;
  std::vector<std::unique_ptr<Node>> nodes_;
  std::vector<Node*> topo_order_;
  std::vector<Value*> inputs_;
  std::vector<Value*> outputs_;
  // 节点归属的 O(1) 判定集合(决议点 5-③):随 create_node/add_graph_input/
  // erase_node 同步维护,内容恒等于 nodes_ 的裸指针集合;拓扑序 topo_order_
  // 仍是顺序权威(V2 校验一致性已有),本集合只服务"是否属于本图"这一类归属
  // 查询。保持仅供 ir 层内部使用(不新增公开访问器)——check_ssa 由
  // verify_structure() 以 const 引用传参读取,不对外暴露可变或直接访问口径。
  std::unordered_set<const Node*> node_set_;
};

// 按拓扑序、经全部既有公开构图 API(create_node/add_graph_input/mark_output/
// set_attr)重建一份与 source 等价的独立可变图——不经 dump_text/parse_text
// 往返,对任意 Device::backend 取值(含自定义/插件后端名)均成立(REUSE-002
// 单份实现,提升自 runtime::compile 原私有版,见 src/runtime/compile.cpp 头
// 注释与 docs/architecture/autograd.md 第2章生成算法①;runtime::compile 与
// compiler::build_backward_graph 共用这一份,禁止第二份复制)。value_map 非空
// 时,函数先清空该表、再写入 source 内每个 Value* 到其克隆 Value* 的完整映射
// (供调用方——如 compiler::build_backward_graph——据此把反向节点挂接到正确
// 的克隆 Value 上);调用方不需要该映射时可传 nullptr(默认值),不产生额外
// 开销。失败(理论上仅在 source 违反其自身构造期已强制的不变量时触发)经
// Result 透传。
FRAME_API Result<Graph> clone_graph(const Graph& source,
                                    std::unordered_map<const Value*, Value*>* value_map = nullptr);

}  // namespace frame::ir
