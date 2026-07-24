// build_backward_graph / build_sgd_update_graph 的实现单元(声明与契约见
// include/frame/compiler/autograd.h;权威设计见
// docs/architecture/autograd.md 第2/5/6章)。

#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <frame/compiler/autograd.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/constant_utils.h>
#include <frame/ops/graph_builder.h>
#include <frame/ops/op_registry.h>
#include <frame/ops/op_schema.h>

namespace frame::compiler {

namespace {

// 把 micro(某算子 GradientFn 产出的梯度微图)展开进 training:micro 的每个
// graph_input 按位绑定 bound_graph_inputs 中对应位置的训练图既有 Value(不
// 新建 graph_input 节点——训练图自身的 graph_input 集合在 clone_graph 阶段已
// 固定,梯度微图只是复用其中已存在的 Value 作为接线起点);micro 的其余节点
// 逐个经 training.create_node 复制(inputs 经 micro Value -> training Value
// 映射表改接线,output_types 直接取 micro 节点自身已定型的输出类型——梯度
// 微图是"独立合法可执行图"、类型已具体,op_schema.h GradientFn 文档,不需要
// 重新推断)、属性原样搬运。返回 micro 图输出(gx_0..gx_{n-1})对应的 training
// Value 列表,按位与 micro.outputs() 一一对应。
//
// 与 ir::clone_graph 的关系(REUSE-002 自查,非重复):clone_graph 对
// source 的每个 graph_input 节点都新建一份训练图自己的 graph_input(源→克隆
// 全新 1:1 映射);本函数恰相反——不新建任何 graph_input 节点,而是把 micro
// 的 graph_input 位绑定到调用方已给定的、训练图内既有的 Value(外部预置
// 映射)。二者目标结构不同(前者产出独立图,后者把子图"焊接"进既有图),与
// src/runtime/fallback_executable.cpp::FallbackExecutable::resolve_node 内
// DecomposeFn 微图内联("拓扑序遍历 + Value→槽位映射表 + create_node 复制")
// 是同一构图算法思路在不同目标表示(IR 图 vs 扁平执行槽位)上的独立实现,不
// 强行合并。
Result<std::vector<ir::Value*>> inline_gradient_micrograph(
    ir::Graph& training, const ir::Graph& micro,
    const std::vector<ir::Value*>& bound_graph_inputs) {
  if (micro.inputs().size() != bound_graph_inputs.size()) {
    return Status::make(ErrorCode::kInternal,
                        "compiler::build_backward_graph: gradient micrograph declared " +
                            std::to_string(micro.inputs().size()) + " graph input(s), expected " +
                            std::to_string(bound_graph_inputs.size()) +
                            " (violates positional gradient contract, autograd.md)");
  }

  std::unordered_map<const ir::Value*, ir::Value*> value_map;
  value_map.reserve(micro.inputs().size());
  for (size_t i = 0; i < micro.inputs().size(); ++i) {
    value_map.emplace(micro.inputs()[i], bound_graph_inputs[i]);
  }

  for (const ir::Node* micro_node : micro.topological_order()) {
    if (micro_node->op() == ir::kGraphInputOp) continue;

    std::vector<ir::Value*> new_inputs;
    new_inputs.reserve(micro_node->inputs().size());
    for (const ir::Value* micro_input : micro_node->inputs()) {
      const auto it = value_map.find(micro_input);
      if (it == value_map.end()) {
        // 理论不可达:拓扑序 + SSA 保证 producer 先于消费者处理。
        return Status::make(ErrorCode::kInternal,
                            "compiler::build_backward_graph: gradient micrograph sub-op '" +
                                std::string(micro_node->op()) +
                                "' input value has no mapping (violates topological order "
                                "invariant)");
      }
      new_inputs.push_back(it->second);
    }

    std::vector<ir::TensorType> output_types;
    output_types.reserve(micro_node->outputs().size());
    for (const ir::Value& output : micro_node->outputs()) {
      output_types.push_back(output.type());
    }

    Result<ir::Node*> new_node = training.create_node(
        std::string(micro_node->op()), std::move(new_inputs), std::move(output_types));
    if (!new_node.is_ok()) return new_node.status();

    for (const auto& [attr_name, attr_value] : micro_node->attrs()) {
      new_node.value()->set_attr(attr_name, attr_value);
    }

    for (size_t j = 0; j < micro_node->outputs().size(); ++j) {
      value_map.emplace(&micro_node->outputs()[j],
                        new_node.value()->output(static_cast<int32_t>(j)));
    }
  }

  std::vector<ir::Value*> results;
  results.reserve(micro.outputs().size());
  for (const ir::Value* output : micro.outputs()) {
    const auto it = value_map.find(output);
    if (it == value_map.end()) {
      return Status::make(ErrorCode::kInternal,
                          "compiler::build_backward_graph: gradient micrograph output value has "
                          "no mapping");
    }
    results.push_back(it->second);
  }
  return results;
}

// 累加同一 Value 的多路径梯度贡献(SSA 下的多路径梯度求和,autograd.md 第2
// 章④):grad_of[value] 已存在贡献时,经 add 节点合并(两个贡献均是 value
// 自身类型的梯度,与 value 同 shape/dtype,天然满足 add 的同 shape/dtype
// 契约);否则直接登记。
Status accumulate_gradient(ir::Graph& training,
                           std::unordered_map<const ir::Value*, ir::Value*>& grad_of,
                           const ir::Value* value, ir::Value* contribution) {
  const auto it = grad_of.find(value);
  if (it == grad_of.end()) {
    grad_of.emplace(value, contribution);
    return Status::ok();
  }
  const Result<ir::Node*> add_node =
      ops::create_node_with_inferred_types(training, "add", {it->second, contribution});
  if (!add_node.is_ok()) return add_node.status();
  it->second = add_node.value()->output(0);
  return Status::ok();
}

}  // namespace

Result<ir::Graph> build_backward_graph(const ir::Graph& forward, int32_t loss_output_index,
                                       std::span<const int32_t> wrt_input_indices) {
  // M26 向后兼容扩展:允许多输出 forward,loss_output_index 选择其中任一
  // 标量目标。clone_graph 保真保留全部原输出,本函数只在其后按 wrt 顺序
  // 追加梯度,无需 IR 输出裁剪/重排 API(ARCH-061/067)。

  // ①loss_output_index 界内校验。
  const int32_t output_count = static_cast<int32_t>(forward.outputs().size());
  if (loss_output_index < 0 || loss_output_index >= output_count) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "compiler::build_backward_graph: loss_output_index " +
                            std::to_string(loss_output_index) + " is out of range [0, " +
                            std::to_string(output_count) + ")");
  }
  const ir::Value* loss_forward_value = forward.outputs()[static_cast<size_t>(loss_output_index)];

  // loss 必须标量(ARCH-061:rank 0 或 numel==1)。
  const Shape& loss_shape = loss_forward_value->type().shape;
  const bool loss_is_scalar = loss_shape.rank() == 0 || loss_shape.numel() == 1;
  if (!loss_is_scalar) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "compiler::build_backward_graph: loss output must be scalar (rank 0 or "
                        "numel==1), got shape " +
                            loss_shape.to_string());
  }

  // ②wrt_input_indices 界内且去重校验。
  const int32_t input_count = static_cast<int32_t>(forward.inputs().size());
  std::unordered_set<int32_t> seen_wrt_indices;
  seen_wrt_indices.reserve(wrt_input_indices.size());
  for (const int32_t idx : wrt_input_indices) {
    if (idx < 0 || idx >= input_count) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "compiler::build_backward_graph: wrt_input_indices entry " +
                              std::to_string(idx) + " is out of range [0, " +
                              std::to_string(input_count) + ")");
    }
    const bool duplicate = !seen_wrt_indices.insert(idx).second;
    if (duplicate) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "compiler::build_backward_graph: wrt_input_indices entry " +
                              std::to_string(idx) + " is duplicated");
    }
  }

  // ③克隆 forward(ir::clone_graph,REUSE-002,autograd.md 第2章生成算法①)。
  std::unordered_map<const ir::Value*, ir::Value*> value_map;
  Result<ir::Graph> cloned = ir::clone_graph(forward, &value_map);
  if (!cloned.is_ok()) return cloned.status();
  ir::Graph training = std::move(cloned.value());

  const auto loss_clone_it = value_map.find(loss_forward_value);
  if (loss_clone_it == value_map.end()) {
    // 理论不可达:clone_graph 对 source 的每个 Value 均建立映射。
    return Status::make(ErrorCode::kInternal,
                        "compiler::build_backward_graph: loss output value has no clone mapping");
  }
  ir::Value* loss_value = loss_clone_it->second;

  // 标记 loss 依赖链(autograd.md 第2章"先正向标记依赖集"的逆拓扑序实现):
  // 从 loss 的 producer 出发,按逆拓扑序把"已标记节点"的全部输入 producer 也
  // 标记为在链上——逆序遍历保证某节点被标记时,其全部消费者均已处理完毕。
  std::unordered_set<const ir::Node*> in_chain;
  in_chain.insert(loss_value->producer());
  // 值拷贝而非引用:下方②③步会持续向 training 添加新节点
  // (create_node_with_inferred_types),Graph::topological_order() 底层是
  // vector<Node*> 成员,追加节点可能触发该 vector 扩容重新分配,若持有其引用
  // 会在扩容后悬空(UB/崩溃)。此处只需遍历"克隆自 forward 的原始节点集合"
  // (本就不应包含随后新增的梯度节点),故在任何新增节点之前拍下快照副本即
  // 完全正确、且规避该生命周期陷阱。
  const std::vector<ir::Node*> topo = training.topological_order();
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    ir::Node* node = *it;
    if (in_chain.find(node) == in_chain.end()) continue;
    for (const ir::Value* input : node->inputs()) {
      in_chain.insert(input->producer());
    }
  }

  // ②种子梯度:constant(1),同 loss dtype/shape。
  std::unordered_map<const ir::Value*, ir::Value*> grad_of;
  const std::vector<double> seed_values(static_cast<size_t>(loss_value->type().shape.numel()), 1.0);
  const ops::AttrMap seed_attrs{
      {"value", seed_values},
      {"shape", loss_value->type().shape},
      {"dtype", loss_value->type().dtype},
  };
  const Result<ir::Node*> seed_node = ops::create_node_with_inferred_types(
      training, ops::kConstantOpName, loss_value->type().device, seed_attrs);
  if (!seed_node.is_ok()) return seed_node.status();
  const Status seed_status =
      accumulate_gradient(training, grad_of, loss_value, seed_node.value()->output(0));
  if (!seed_status.is_ok()) return seed_status;

  // ③④逆拓扑序展开梯度微图 + 多路径累加。
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    ir::Node* node = *it;
    if (in_chain.find(node) == in_chain.end()) continue;
    // 0 输入节点(graph_input,或理论上的 0 输入常规算子如 constant)无预测
    // 对象需要回传梯度:ARCH-062 的"链上未注册 GradientFn 即报错"意在避免
    // 静默漏掉本应存在的梯度、产出错误结果;0 输入节点没有输入位可供漏掉,
    // 跳过不产生该风险,故不要求其注册 GradientFn(v0 七算子清单本身也不含
    // 0 输入算子)。
    if (node->inputs().empty()) continue;

    const ops::OpSchema* schema = ops::OpRegistry::instance().find(node->op());
    if (schema == nullptr) {
      // 理论不可达:训练图克隆自已通过 V3 校验的 forward(op 必然已注册)。
      return Status::make(
          ErrorCode::kInternal,
          "compiler::build_backward_graph: op '" + std::string(node->op()) + "' is not registered");
    }
    if (schema->has_trait(ops::OpTrait::kHasSideEffect)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "compiler::build_backward_graph: op '" + std::string(node->op()) +
                              "' has OpTrait::kHasSideEffect and cannot appear on the loss "
                              "dependency chain (ARCH-062)");
    }
    const ops::GradientFn grad_fn = schema->gradient();
    if (grad_fn == nullptr) {
      return Status::make(ErrorCode::kUnimplemented,
                          "compiler::build_backward_graph: op '" + std::string(node->op()) +
                              "' has no registered GradientFn (ARCH-062)");
    }

    // 组齐 gy 列表(逐输出查 grad_of;缺失属理论不可达——node 被标记 in_chain
    // 正是因为其某个输出被下游消费并已回传梯度)。
    std::vector<ir::Value*> gy_list;
    gy_list.reserve(node->outputs().size());
    for (const ir::Value& output : node->outputs()) {
      const auto grad_it = grad_of.find(&output);
      if (grad_it == grad_of.end()) {
        return Status::make(ErrorCode::kInternal,
                            "compiler::build_backward_graph: op '" + std::string(node->op()) +
                                "' output has no accumulated gradient despite being on the loss "
                                "dependency chain (violates in_chain invariant)");
      }
      gy_list.push_back(grad_it->second);
    }

    ops::NodeContext ctx;
    ctx.op = node->op();
    ctx.input_types.reserve(node->inputs().size());
    for (const ir::Value* input : node->inputs()) {
      ctx.input_types.push_back(input->type());
    }
    ctx.attrs = &node->attrs();

    const Result<ir::Graph> micro = grad_fn(ctx);
    if (!micro.is_ok()) {
      return Status::make(micro.status().code(),
                          "compiler::build_backward_graph: op '" + std::string(node->op()) +
                              "' GradientFn failed: " + std::string(micro.status().message()));
    }
    const ir::Graph& micro_graph = micro.value();

    const size_t n = node->inputs().size();
    const size_t m = node->outputs().size();
    if (micro_graph.inputs().size() != n + 2 * m) {
      return Status::make(
          ErrorCode::kInternal,
          "compiler::build_backward_graph: op '" + std::string(node->op()) +
              "' gradient micrograph declared " + std::to_string(micro_graph.inputs().size()) +
              " graph input(s), expected " + std::to_string(n + 2 * m) +
              " (violates positional gradient contract [x.., y.., gy..], autograd.md §3)");
    }
    if (micro_graph.outputs().size() != n) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "compiler::build_backward_graph: op '" + std::string(node->op()) +
              "' gradient micrograph produced " + std::to_string(micro_graph.outputs().size()) +
              " graph output(s), expected " + std::to_string(n) +
              " (violates positional gradient contract [gx..], autograd.md §3: v0 requires a "
              "gradient for every input position)");
    }

    std::vector<ir::Value*> bound_inputs;
    bound_inputs.reserve(n + 2 * m);
    for (ir::Value* input : node->inputs()) bound_inputs.push_back(input);
    for (size_t j = 0; j < m; ++j) bound_inputs.push_back(node->output(static_cast<int32_t>(j)));
    for (ir::Value* gy : gy_list) bound_inputs.push_back(gy);

    const Result<std::vector<ir::Value*>> gx_list =
        inline_gradient_micrograph(training, micro_graph, bound_inputs);
    if (!gx_list.is_ok()) return gx_list.status();

    for (size_t i = 0; i < n; ++i) {
      const Status acc_status =
          accumulate_gradient(training, grad_of, node->inputs()[i], gx_list.value()[i]);
      if (!acc_status.is_ok()) return acc_status;
    }
  }

  // ⑤mark_output:forward 的全部原输出已由 ir::clone_graph 按原序登记,
  // 不需要、也不应重复 mark_output。本步只在其后依序追加 wrt 梯度,因此
  // 输出布局恒为 [forward_outputs..., grad(wrt_0), ...](ARCH-061)。
  for (const int32_t idx : wrt_input_indices) {
    ir::Value* wrt_forward_value = forward.inputs()[static_cast<size_t>(idx)];
    const auto wrt_clone_it = value_map.find(wrt_forward_value);
    if (wrt_clone_it == value_map.end()) {
      // 理论不可达:clone_graph 对 source 的每个 graph_input 均建立映射。
      return Status::make(ErrorCode::kInternal,
                          "compiler::build_backward_graph: wrt input value has no clone mapping");
    }
    ir::Value* wrt_clone_value = wrt_clone_it->second;

    const auto grad_it = grad_of.find(wrt_clone_value);
    ir::Value* grad_value = nullptr;
    if (grad_it != grad_of.end()) {
      grad_value = grad_it->second;
    } else {
      // 不在 loss 依赖链上:数学上梯度恒为 0(标准自动微分惯例——未参与 loss
      // 计算的输入梯度即 0,而非错误;autograd.md 对该边角未显式约定,按主流
      // 自动微分框架的既定惯例补齐一份全零 constant)。
      const ir::TensorType& wrt_type = wrt_clone_value->type();
      const std::vector<double> zero_values(static_cast<size_t>(wrt_type.shape.numel()), 0.0);
      const ops::AttrMap zero_attrs{
          {"value", zero_values},
          {"shape", wrt_type.shape},
          {"dtype", wrt_type.dtype},
      };
      const Result<ir::Node*> zero_node = ops::create_node_with_inferred_types(
          training, ops::kConstantOpName, wrt_type.device, zero_attrs);
      if (!zero_node.is_ok()) return zero_node.status();
      grad_value = zero_node.value()->output(0);
    }

    const Status mark_wrt_status = training.mark_output(grad_value);
    if (!mark_wrt_status.is_ok()) return mark_wrt_status;
  }

  return training;
}

Result<ir::Graph> build_sgd_update_graph(std::span<const ir::TensorType> param_types,
                                         double learning_rate) {
  // 校验①:param_types 非空(空图无参数可更新,属调用方误用)。
  if (param_types.empty()) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "compiler::build_sgd_update_graph: param_types must not be empty");
  }

  // 校验②:learning_rate 须为有限值(拒 NaN/inf——二者传播进 constant 会产出
  // 静默错误的更新图,ARCH-031 口径不静默降级)。
  if (!std::isfinite(learning_rate)) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "compiler::build_sgd_update_graph: learning_rate must be finite, got " +
                            std::to_string(learning_rate));
  }

  // 校验③:逐个 param dtype 属 v0 constant 可编码白名单。
  for (size_t i = 0; i < param_types.size(); ++i) {
    if (!ops::is_constant_dtype_supported(param_types[i].dtype)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "compiler::build_sgd_update_graph: param_types[" + std::to_string(i) +
                              "] has unsupported dtype '" +
                              std::string(param_types[i].dtype.name()) +
                              "' (v0 supports float32/float16/bfloat16 only)");
    }
  }

  ir::Graph graph("sgd_update");

  // 图输入按位 = [param_0..param_{n-1}, grad_0..grad_{n-1}](本函数头注释;
  // grad_i 类型恒等于 param_i)。分两趟 add_graph_input 建立,天然满足该按位
  // 布局。
  const size_t n = param_types.size();
  std::vector<ir::Value*> param_values;
  param_values.reserve(n);
  for (const ir::TensorType& param_type : param_types) {
    const Result<ir::Value*> param_value = graph.add_graph_input(param_type);
    if (!param_value.is_ok()) return param_value.status();
    param_values.push_back(param_value.value());
  }
  std::vector<ir::Value*> grad_values;
  grad_values.reserve(n);
  for (const ir::TensorType& param_type : param_types) {
    const Result<ir::Value*> grad_value = graph.add_graph_input(param_type);
    if (!grad_value.is_ok()) return grad_value.status();
    grad_values.push_back(grad_value.value());
  }

  // 逐参数位:new_param_i = add(param_i, mul(constant(-learning_rate), grad_i))。
  // constant 按 param_types[i].shape 逐元素填充 -learning_rate(v0 mul 无广播,
  // 不能用 rank-0 标量再乘,与 square_gradient 的 constant(2) 同构造手法,
  // src/ops/schemas/elementwise.cpp)。
  for (size_t i = 0; i < n; ++i) {
    const ir::TensorType& param_type = param_types[i];
    const std::vector<double> neg_lr_values(static_cast<size_t>(param_type.shape.numel()),
                                            -learning_rate);
    const ops::AttrMap neg_lr_attrs{
        {"value", neg_lr_values},
        {"shape", param_type.shape},
        {"dtype", param_type.dtype},
    };
    const Result<ir::Node*> neg_lr_node = ops::create_node_with_inferred_types(
        graph, ops::kConstantOpName, param_type.device, neg_lr_attrs);
    if (!neg_lr_node.is_ok()) return neg_lr_node.status();

    const Result<ir::Node*> mul_node = ops::create_node_with_inferred_types(
        graph, "mul", {neg_lr_node.value()->output(0), grad_values[i]});
    if (!mul_node.is_ok()) return mul_node.status();

    const Result<ir::Node*> add_node = ops::create_node_with_inferred_types(
        graph, "add", {param_values[i], mul_node.value()->output(0)});
    if (!add_node.is_ok()) return add_node.status();

    const Status mark_status = graph.mark_output(add_node.value(), 0);
    if (!mark_status.is_ok()) return mark_status;
  }

  return graph;
}

}  // namespace frame::compiler
