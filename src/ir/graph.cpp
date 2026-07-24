// Graph 静态计算图的实现单元。

#include <algorithm>
#include <cstdint>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <frame/ir/graph.h>

namespace frame::ir {

// op 名字符集校验:`^[a-z][a-z0-9_]*$`(首字符小写字母,后续小写字母/数字/
// 下划线),手写逐字符判定(不引入 <regex>)。声明见 include/frame/ir/graph.h;
// 公开(非匿名命名空间)是为了让 ops 层 OpRegistry::register_op(ARCH-040)
// 共用本函数这一份实现,避免同签名同函数体的第二份复制(REUSE-002)。
bool matches_op_name_charset(std::string_view name) noexcept {
  if (name.empty()) return false;
  const char first = name.front();
  if (first < 'a' || first > 'z') return false;
  for (size_t i = 1; i < name.size(); ++i) {
    const char c = name[i];
    const bool is_lower_alpha = c >= 'a' && c <= 'z';
    const bool is_digit = c >= '0' && c <= '9';
    if (!is_lower_alpha && !is_digit && c != '_') return false;
  }
  return true;
}

namespace {

// 判断 node 是否已属于图(按裸指针查 node_set,O(1),决议点 5-③)。
// 构造性防环校验(create_node)与图归属校验(mark_output)共用;node_set 由
// 调用方(Graph 自身成员函数,或经 verify_structure 转交的 check_ssa)以 const
// 引用传入——Graph::node_set_ 保持私有,不新增公开访问器。
bool contains_node(const std::unordered_set<const Node*>& node_set, const Node* node) {
  return node_set.find(node) != node_set.end();
}

// 在 topo_order 中定位 node 的下标。调用方须先确认 node 属于本图(node_set_
// 与 topo_order_ 恒同步维护),命中失败即违反内部不变量,走 FRAME_CHECK
// fatal(非用户输入错误)。replace_all_uses 的拓扑序不变式判定专用。
size_t topo_index_of(const std::vector<Node*>& topo_order, const Node* node) {
  const auto it = std::find(topo_order.begin(), topo_order.end(), node);
  FRAME_CHECK(it != topo_order.end());
  return static_cast<size_t>(std::distance(topo_order.begin(), it));
}

// 校验 value 指针确实落在其 producer 输出数组的对应槽位,而非一份地址不同的
// 悬挂拷贝(修订节 5-①,design-reviewer 建议①采纳):用相等判定
// `value == producer->outputs().data() + index` 而非指针区间比较,全程只做
// 指针相等比较(标准允许无关指针间的相等比较,零 UB);顺带复核
// output_index() 与实际数组下标一致。调用方须先确认 producer 非空。
bool value_matches_producer_slot(const Value* value) {
  const Node* producer = value->producer();
  const int32_t index = value->output_index();
  const std::vector<Value>& producer_outputs = producer->outputs();
  if (index < 0 || static_cast<size_t>(index) >= producer_outputs.size()) return false;
  return value == producer_outputs.data() + index;
}

// ---------------------------------------------------------------------------
// V1(SSA):每个被引用 Value 的 producer 非空且属于本图,且该 Value 指针确实
// 落在其 producer 输出数组的对应槽位(拒绝悬挂的栈拷贝 Value,修订节 5-①);
// graph inputs() 内每个 Value 的 producer 必须是 kGraphInputOp 节点。
// ---------------------------------------------------------------------------
Status check_ssa(const Graph& graph, const std::unordered_set<const Node*>& node_set) {
  for (const Node* node : graph.topological_order()) {
    for (const Value* input : node->inputs()) {
      if (input == nullptr || input->producer() == nullptr) {
        return Status::make(
            ErrorCode::kInvalidArgument,
            "V1: node '" + std::string(node->op()) + "' has an input value with no producer");
      }
      if (!contains_node(node_set, input->producer())) {
        return Status::make(ErrorCode::kInvalidArgument,
                            "V1: node '" + std::string(node->op()) +
                                "' has an input value produced outside this graph");
      }
      if (!value_matches_producer_slot(input)) {
        return Status::make(ErrorCode::kInvalidArgument,
                            "V1: node '" + std::string(node->op()) +
                                "' has an input value pointer that does not match its "
                                "producer's output slot");
      }
    }
  }
  for (const Value* output : graph.outputs()) {
    if (output == nullptr || output->producer() == nullptr ||
        !contains_node(node_set, output->producer())) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "V1: graph output value has no valid producer in this graph");
    }
    if (!value_matches_producer_slot(output)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "V1: graph output value pointer does not match its producer's "
                          "output slot");
    }
  }
  for (const Value* input : graph.inputs()) {
    if (input == nullptr || input->producer() == nullptr ||
        input->producer()->op() != kGraphInputOp) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "V1: graph input value's producer is not a graph_input node");
    }
  }
  return Status::ok();
}

// ---------------------------------------------------------------------------
// V2(无环):沿 inputs 边做 Kahn 拓扑排序,能排出的节点数须等于图节点总数;
// 另核对 topological_order() 内部无重复节点。
// ---------------------------------------------------------------------------
Status check_acyclic(const Graph& graph) {
  const std::vector<Node*>& topo = graph.topological_order();
  const std::unordered_set<const Node*> topo_set(topo.begin(), topo.end());
  if (topo_set.size() != topo.size()) {
    return Status::make(ErrorCode::kInternal, "V2: topological_order contains duplicate nodes");
  }

  // 入度表(全部节点先登记为 0)与邻接表(producer -> consumers)。
  std::unordered_map<const Node*, int64_t> indegree;
  std::unordered_map<const Node*, std::vector<const Node*>> consumers;
  indegree.reserve(topo.size());
  for (const Node* node : topo) {
    indegree.emplace(node, 0);
  }
  for (const Node* node : topo) {
    const auto node_degree_it = indegree.find(node);  // node 取自 topo,上面已登记,必存在
    for (const Value* input : node->inputs()) {
      if (input == nullptr || input->producer() == nullptr) continue;  // V1 负责报告
      const Node* producer = input->producer();
      if (indegree.find(producer) == indegree.end()) continue;  // producer 不属本图,V1 负责报告
      node_degree_it->second += 1;  // 入度记在消费者(node)一侧,而非生产者一侧
      consumers[producer].push_back(node);
    }
  }

  // 按 topo(确定性 vector 顺序)取零入度节点,不直接遍历 indegree
  // (unordered_map<const Node*, ...> 按指针遍历顺序不确定,
  // bugprone-nondeterministic-pointer-iteration-order)。
  std::queue<const Node*> ready;
  for (const Node* node : topo) {
    const auto degree_it = indegree.find(node);
    if (degree_it != indegree.end() && degree_it->second == 0) ready.push(node);
  }
  int64_t visited = 0;
  while (!ready.empty()) {
    const Node* current = ready.front();
    ready.pop();
    ++visited;
    const auto consumers_it = consumers.find(current);
    if (consumers_it == consumers.end()) continue;
    for (const Node* next : consumers_it->second) {
      const auto degree_it = indegree.find(next);
      if (degree_it == indegree.end()) continue;  // 不可达:next 取自 topo,已在上面登记
      degree_it->second -= 1;
      if (degree_it->second == 0) ready.push(next);
    }
  }
  if (visited != static_cast<int64_t>(topo.size())) {
    return Status::make(ErrorCode::kInvalidArgument, "V2: graph contains a cycle");
  }
  return Status::ok();
}

// dtype 合法性:code 落在 DTypeCode 枚举的定义范围内(不含哨兵 kCount 及其后)。
bool is_valid_dtype(DType dtype) {
  return static_cast<uint8_t>(dtype.code()) < static_cast<uint8_t>(DTypeCode::kCount);
}

// ---------------------------------------------------------------------------
// V5(无 unknown):全部 Value 的 shape.verify() 通过(拒 kDynamicDim)且 dtype 合法。
// ---------------------------------------------------------------------------
Status check_shapes_and_dtypes(const Graph& graph) {
  for (const Node* node : graph.topological_order()) {
    for (const Value& output : node->outputs()) {
      const TensorType& type = output.type();
      const Status shape_status = type.shape.verify();
      if (!shape_status.is_ok()) {
        return Status::make(shape_status.code(), "V5: node '" + std::string(node->op()) +
                                                     "' output has invalid shape: " +
                                                     std::string(shape_status.message()));
      }
      if (!is_valid_dtype(type.dtype)) {
        return Status::make(ErrorCode::kInvalidArgument, "V5: node '" + std::string(node->op()) +
                                                             "' output has invalid dtype code");
      }
    }
  }
  return Status::ok();
}

// ---------------------------------------------------------------------------
// V6(device 一致):全图所有 Value 的 device 相同(空图/仅输入图允许,因为
// 该场景下从未出现第二个不同的 device 值)。
// ---------------------------------------------------------------------------
Status check_device_consistency(const Graph& graph) {
  bool has_device = false;
  Device expected{};
  for (const Node* node : graph.topological_order()) {
    for (const Value& output : node->outputs()) {
      const Device& device = output.type().device;
      if (!has_device) {
        expected = device;
        has_device = true;
        continue;
      }
      if (!(device == expected)) {
        return Status::make(ErrorCode::kInvalidArgument,
                            "V6: graph contains values on inconsistent devices");
      }
    }
  }
  return Status::ok();
}

// ---------------------------------------------------------------------------
// V7(属性封闭,ARCH-020):机械遍历每个节点的属性,经 attr_type_of 映射到
// AttrType 后调用 attr_type_name;后者对全部合法 AttrType 均定义,调用本身
// 即完成"变体可映射"的核验(未来若扩型漏检,attr_type_name 内部的
// FRAME_CHECK 兜底会立即暴露,而非悄悄漏检)。attrs() 枚举顺序不确定,但本
// 检查只做机械遍历、不产出确定性输出,顺序不影响结果。
// ---------------------------------------------------------------------------
Status check_attrs_closed(const Graph& graph) {
  for (const Node* node : graph.topological_order()) {
    for (const auto& [attr_name, attr_value] : node->attrs()) {
      (void)attr_name;
      const AttrType attr_type = attr_type_of(attr_value);
      (void)attr_type_name(attr_type);
    }
  }
  return Status::ok();
}

}  // namespace

Result<Node*> Graph::create_node(std::string op, std::vector<Value*> inputs,
                                 std::vector<TensorType> output_types) {
  if (op == kGraphInputOp) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "Graph.create_node: op name 'graph_input' is reserved for graph "
                        "inputs (use add_graph_input)");
  }
  if (op == kGraphOutputMarker) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "Graph.create_node: op name 'graph_output' is reserved as the "
                        "serialization graph output marker");
  }
  if (!matches_op_name_charset(op)) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "Graph.create_node: op name '" + op + "' is invalid, must match ^[a-z][a-z0-9_]*$");
  }
  for (const Value* input : inputs) {
    if (input == nullptr || input->producer() == nullptr ||
        !contains_node(node_set_, input->producer())) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "Graph.create_node: input value does not belong to this graph");
    }
  }

  auto node = std::make_unique<Node>(std::move(op));
  node->inputs_ = std::move(inputs);
  node->outputs_.reserve(output_types.size());
  for (size_t i = 0; i < output_types.size(); ++i) {
    Value value;
    value.type_ = std::move(output_types[i]);
    value.producer_ = node.get();
    value.output_index_ = static_cast<int32_t>(i);
    node->outputs_.push_back(std::move(value));
  }

  Node* raw = node.get();
  nodes_.push_back(std::move(node));
  topo_order_.push_back(raw);
  node_set_.insert(raw);
  return raw;
}

Result<Value*> Graph::add_graph_input(TensorType type) {
  auto node = std::make_unique<Node>(std::string(kGraphInputOp));
  Value value;
  value.type_ = std::move(type);
  value.producer_ = node.get();
  value.output_index_ = 0;
  node->outputs_.push_back(std::move(value));

  Node* raw = node.get();
  nodes_.push_back(std::move(node));
  topo_order_.push_back(raw);
  node_set_.insert(raw);

  Value* value_ptr = raw->outputs_.data();
  inputs_.push_back(value_ptr);
  return value_ptr;
}

Status Graph::mark_output(Value* value) {
  if (value == nullptr || value->producer() == nullptr ||
      !contains_node(node_set_, value->producer())) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "Graph.mark_output: value does not belong to this graph");
  }
  outputs_.push_back(value);
  return Status::ok();
}

Status Graph::mark_output(Node* node, int32_t output_index) {
  if (node == nullptr || !contains_node(node_set_, node)) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "Graph.mark_output: node does not belong to this graph");
  }
  Value* value = node->output(output_index);
  if (value == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "Graph.mark_output: output index " + std::to_string(output_index) +
                            " is out of range for node '" + std::string(node->op()) + "'");
  }
  return mark_output(value);
}

Status Graph::erase_node(Node* node) {
  if (node == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument, "Graph.erase_node: node is null");
  }
  const auto node_it =
      std::find_if(nodes_.begin(), nodes_.end(),
                   [node](const std::unique_ptr<Node>& owned) { return owned.get() == node; });
  if (node_it == nodes_.end()) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "Graph.erase_node: node does not belong to this graph");
  }

  // 收集该节点全部输出 Value 的地址,用于后续引用检测。
  std::vector<const Value*> node_outputs;
  node_outputs.reserve(node->outputs().size());
  for (const Value& output : node->outputs()) {
    node_outputs.push_back(&output);
  }
  auto is_node_output = [&node_outputs](const Value* value) {
    return std::find(node_outputs.begin(), node_outputs.end(), value) != node_outputs.end();
  };

  for (const auto& other : nodes_) {
    if (other.get() == node) continue;
    for (const Value* input : other->inputs()) {
      if (is_node_output(input)) {
        return Status::make(ErrorCode::kInvalidArgument,
                            "Graph.erase_node: node output is referenced by another node's input");
      }
    }
  }
  for (const Value* output : outputs_) {
    if (is_node_output(output)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "Graph.erase_node: node output is referenced by a graph output");
    }
  }

  topo_order_.erase(std::remove(topo_order_.begin(), topo_order_.end(), node), topo_order_.end());
  inputs_.erase(std::remove_if(inputs_.begin(), inputs_.end(), is_node_output), inputs_.end());
  node_set_.erase(node);
  nodes_.erase(node_it);
  return Status::ok();
}

Status Graph::replace_all_uses(Value* from, Value* to) {
  if (from == nullptr || from->producer() == nullptr ||
      !contains_node(node_set_, from->producer())) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "Graph.replace_all_uses: from value does not belong to this graph");
  }
  if (to == nullptr || to->producer() == nullptr || !contains_node(node_set_, to->producer())) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "Graph.replace_all_uses: to value does not belong to this graph");
  }
  if (from == to) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "Graph.replace_all_uses: from and to must be different values");
  }

  // TensorType 四元组(dtype/shape/layout/device)完全相等校验;违例消息带
  // 双方全部字段实际值。
  const TensorType& from_type = from->type();
  const TensorType& to_type = to->type();
  const bool type_mismatch =
      !(from_type.dtype == to_type.dtype) || !(from_type.shape == to_type.shape) ||
      from_type.layout != to_type.layout || !(from_type.device == to_type.device);
  if (type_mismatch) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "Graph.replace_all_uses: from and to have different tensor types, from dtype '" +
            std::string(from_type.dtype.name()) + "' shape " + from_type.shape.to_string() +
            " layout " + std::to_string(static_cast<int>(from_type.layout)) + " device " +
            std::string(from_type.device.backend) + ":" + std::to_string(from_type.device.index) +
            ", to dtype '" + std::string(to_type.dtype.name()) + "' shape " +
            to_type.shape.to_string() + " layout " +
            std::to_string(static_cast<int>(to_type.layout)) + " device " +
            std::string(to_type.device.backend) + ":" + std::to_string(to_type.device.index));
  }

  Node* from_producer = from->producer();
  Node* to_producer = to->producer();

  if (from_producer != to_producer) {
    // 二者由不同节点产出:topo_order_ 内节点互不重复(V2 已保证),故二者
    // 下标必然不同,只需处理"to 在 from 之后或同位"(不可能同位)与
    // "to 在 from 之前"两种情形。
    const size_t from_index = topo_index_of(topo_order_, from_producer);
    const size_t to_index = topo_index_of(topo_order_, to_producer);
    if (to_index > from_index) {
      // 第二豁免(裁决修订 2,严格版,取代 M8 的"0 输入节点"豁免——0 输入
      // 节点是本条件的平凡实例):to 的 producer 的每一个输入的 producer
      // 拓扑下标均须严格 < from_index,方可重定位;安全论证见 graph.h 头注释。
      bool relocation_safe = true;
      for (const Value* input : to_producer->inputs()) {
        const size_t input_producer_index = topo_index_of(topo_order_, input->producer());
        if (input_producer_index >= from_index) {
          relocation_safe = false;
          break;
        }
      }
      if (!relocation_safe) {
        return Status::make(
            ErrorCode::kInvalidArgument,
            "Graph.replace_all_uses: topological order invariant violated, to's producer op '" +
                std::string(to_producer->op()) + "' must appear before from's producer op '" +
                std::string(from_producer->op()) +
                "' in topological order, or have every input's producer strictly before from's "
                "producer to qualify for relocation");
      }
      // 重定位:就地把 to 的 producer 挪到 from 的 producer 原位置,保持
      // topo_order_(= CpuExecutable 执行序)合法——
      // std::rotate(first=from_index, middle=to_index, last=to_index+1) 把
      // [from_index, to_index) 与 [to_index, to_index+1) 两段互换,结果是
      // to_producer 移到 from_index 位置,原 [from_index, to_index) 区间整体
      // 右移一位(from_producer 随之落到 from_index+1,即 to_producer 紧后)。
      std::rotate(topo_order_.begin() + static_cast<std::ptrdiff_t>(from_index),
                  topo_order_.begin() + static_cast<std::ptrdiff_t>(to_index),
                  topo_order_.begin() + static_cast<std::ptrdiff_t>(to_index) + 1);
    }
    // to_index < from_index:天然满足拓扑序不变式,无需处理。
  }
  // from_producer == to_producer:同一节点的两个不同输出,二者同时产出,不
  // 存在先后顺序问题,天然满足不变式,无需处理。

  // use 扫描:全图节点输入 vector(O(V·E),与 check_ssa 同量级)+ 图输出
  // 列表(图输出也是一种 use,遗漏会导致 folding/CSE 无法作用于输出节点)。
  for (const auto& node : nodes_) {
    for (Value*& input : node->inputs_) {
      if (input == from) input = to;
    }
  }
  for (Value*& output : outputs_) {
    if (output == from) output = to;
  }

  return Status::ok();
}

Status Graph::swap_node_inputs(Node* node, int32_t i, int32_t j) {
  if (node == nullptr || !contains_node(node_set_, node)) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "Graph.swap_node_inputs: node does not belong to this graph");
  }
  const int32_t input_count = static_cast<int32_t>(node->inputs_.size());
  if (i < 0 || i >= input_count || j < 0 || j >= input_count) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "Graph.swap_node_inputs: index out of range, i=" + std::to_string(i) +
                            ", j=" + std::to_string(j) +
                            ", input count=" + std::to_string(input_count));
  }
  if (i == j) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "Graph.swap_node_inputs: i and j must be different indices, got " + std::to_string(i));
  }
  std::swap(node->inputs_[static_cast<size_t>(i)], node->inputs_[static_cast<size_t>(j)]);
  return Status::ok();
}

namespace {

// Layout 枚举名(仅供本文件内 assign_layout 的错误消息使用,调试/诊断用途,
// 非序列化契约——序列化尾缀的权威落点见 include/frame/ir/serialization.h
// 头注释,二者不复用同一实现)。
std::string_view layout_debug_name(Layout layout) {
  switch (layout) {
    case Layout::kUnknown:
      return "kUnknown";
    case Layout::kRowMajor:
      return "kRowMajor";
  }
  FRAME_CHECK(false);  // Layout 是封闭枚举,switch 已穷举全部合法取值
  return {};
}

}  // namespace

Status Graph::assign_layout(Value* value, Layout layout) {
  if (value == nullptr || value->producer() == nullptr ||
      !contains_node(node_set_, value->producer())) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "Graph.assign_layout: value does not belong to this graph");
  }
  if (layout == Layout::kUnknown) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "Graph.assign_layout: layout must not be kUnknown");
  }

  const Layout current = value->type_.layout;
  if (current == layout) {
    return Status::ok();  // 幂等重指派
  }
  if (current != Layout::kUnknown) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "Graph.assign_layout: value already has layout '" +
                            std::string(layout_debug_name(current)) + "', cannot reassign to '" +
                            std::string(layout_debug_name(layout)) + "'");
  }

  value->type_.layout = layout;
  return Status::ok();
}

Status Graph::verify_structure() const {
  FRAME_RETURN_IF_ERROR(check_ssa(*this, node_set_));
  FRAME_RETURN_IF_ERROR(check_acyclic(*this));
  FRAME_RETURN_IF_ERROR(check_shapes_and_dtypes(*this));
  FRAME_RETURN_IF_ERROR(check_device_consistency(*this));
  FRAME_RETURN_IF_ERROR(check_attrs_closed(*this));
  return Status::ok();
}

Result<Graph> clone_graph(const Graph& source,
                          std::unordered_map<const Value*, Value*>* value_map) {
  Graph clone{std::string(source.name())};
  std::unordered_map<const Value*, Value*> local_value_map;
  std::unordered_map<const Value*, Value*>& mapping =
      value_map != nullptr ? *value_map : local_value_map;
  mapping.clear();
  mapping.reserve(source.inputs().size());

  for (const Node* node : source.topological_order()) {
    if (node->op() == kGraphInputOp) {
      const Result<Value*> new_input = clone.add_graph_input(node->outputs()[0].type());
      if (!new_input.is_ok()) return new_input.status();
      mapping.emplace(&node->outputs()[0], new_input.value());
      continue;
    }

    std::vector<Value*> new_inputs;
    new_inputs.reserve(node->inputs().size());
    for (const Value* input : node->inputs()) {
      const auto it = mapping.find(input);
      if (it == mapping.end()) {
        return Status::make(ErrorCode::kInternal,
                            "ir::clone_graph: input value has no clone mapping for node '" +
                                std::string(node->op()) +
                                "' (violates topological order invariant)");
      }
      new_inputs.push_back(it->second);
    }

    std::vector<TensorType> output_types;
    output_types.reserve(node->outputs().size());
    for (const Value& output : node->outputs()) {
      output_types.push_back(output.type());
    }

    Result<Node*> new_node =
        clone.create_node(std::string(node->op()), std::move(new_inputs), std::move(output_types));
    if (!new_node.is_ok()) return new_node.status();

    // node->attrs() 枚举顺序不确定,但 set_attr 按名覆盖,顺序不影响结果。
    for (const auto& [attr_name, attr_value] : node->attrs()) {
      new_node.value()->set_attr(attr_name, attr_value);
    }

    for (size_t i = 0; i < node->outputs().size(); ++i) {
      mapping.emplace(&node->outputs()[i], new_node.value()->output(static_cast<int32_t>(i)));
    }
  }

  for (const Value* output : source.outputs()) {
    const auto it = mapping.find(output);
    if (it == mapping.end()) {
      return Status::make(ErrorCode::kInternal,
                          "ir::clone_graph: graph output value has no clone mapping");
    }
    const Status mark_status = clone.mark_output(it->second);
    if (!mark_status.is_ok()) return mark_status;
  }

  return clone;
}

Status Graph::verify(const OpQuery& query) const {
  if (!query.op_registered) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "V3: OpQuery.op_registered callback is not set (fail-closed)");
  }
  if (!query.check_schema) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "V4: OpQuery.check_schema callback is not set (fail-closed)");
  }

  FRAME_RETURN_IF_ERROR(verify_structure());

  for (const Node* node : topological_order()) {
    if (node->op() == kGraphInputOp) {
      // graph_input 节点豁免 V3/V4,改做结构检查:0 输入、恰 1 输出、该输出
      // 已登记于 inputs()。
      if (!node->inputs().empty()) {
        return Status::make(ErrorCode::kInvalidArgument,
                            "V3: graph_input node must have zero inputs");
      }
      if (node->outputs().size() != 1) {
        return Status::make(ErrorCode::kInvalidArgument,
                            "V3: graph_input node must have exactly one output");
      }
      const Value* output = node->outputs().data();
      if (std::find(inputs_.begin(), inputs_.end(), output) == inputs_.end()) {
        return Status::make(ErrorCode::kInvalidArgument,
                            "V3: graph_input node output is not registered in graph inputs");
      }
      continue;
    }
    if (!query.op_registered(node->op())) {
      return Status::make(ErrorCode::kNotFound,
                          "V3: op '" + std::string(node->op()) + "' is not registered");
    }
    const Status schema_status = query.check_schema(*node);
    if (!schema_status.is_ok()) {
      return Status::make(schema_status.code(), "V4: " + std::string(schema_status.message()));
    }
  }
  return Status::ok();
}

}  // namespace frame::ir
