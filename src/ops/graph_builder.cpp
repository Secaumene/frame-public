// create_node_with_inferred_types 实现单元(声明与契约见
// include/frame/ops/graph_builder.h)。
//
// 与 src/compiler/passes/shape_inference.cpp「解析 schema + 调 shape_infer」
// 一段的关系(REUSE-002 自查,否则注释论证分支):两处均含"OpRegistry::find
// 查 schema → 校验 shape_infer 已设置 → 构造 NodeContext → 调 infer_fn →
// 校验结果"的骨架,表面同构。但未跨文件抽取,理由三条:①用途不同 ——
// shape_inference pass 是对图中*既有*节点重算并逐位比对既有输出类型(不产出
// 新节点,`m7-design-brief` 决议点 1 的"校验模式"),本文件是依据推断结果
// *新建*节点(供构图期使用),二者对推断结果的消费方式完全不同,不是同一段
// 逻辑的两次调用而是两个不同调用形态;②pass 已随 M7 上线并被
// tests/cpp/compiler/test_shape_inference.cpp 的错误消息前缀("shape_inference:
// ..."）与子串断言覆盖,跨文件抽取需要把错误消息模板参数化或改变前缀语义,
// 收益(消灭约 15 行样板)不足以抵销改动一个已测试稳定核心 pass 文件的回归
// 风险,且改动该文件不在本里程碑 spec 列出的文件清单内;③两处唯一可抽取的
// 公共步骤(schema 查找 + 空指针判断 + 调用 infer_fn)本身很短,若强行跨文件
// 抽取需要额外传参区分错误消息前缀,复杂度增量抵消了去重收益。本文件内部
// 仍按 REUSE-002 精神把该步骤抽成一个私有函数 infer_output_shapes,供下方两个
// 重载共用一份实现(避免本文件内部出现第二份复制)。

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/ir/graph.h>
#include <frame/ops/graph_builder.h>
#include <frame/ops/op_registry.h>
#include <frame/ops/op_schema.h>

namespace frame::ops {

namespace {

// 查 schema + 调 shape_infer 的公共步骤,供下方两个重载共用一份实现
// (REUSE-002,文件头注释已就跨文件层面的非抽取给出论证)。
Result<std::vector<Shape>> infer_output_shapes(std::string_view op,
                                               const std::vector<ir::TensorType>& input_types,
                                               const AttrMap& attrs) {
  const OpSchema* schema = OpRegistry::instance().find(op);
  if (schema == nullptr) {
    return Status::make(ErrorCode::kNotFound, "create_node_with_inferred_types: op '" +
                                                  std::string(op) + "' is not registered");
  }
  const ShapeInferFn infer_fn = schema->shape_infer();
  if (infer_fn == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "create_node_with_inferred_types: op '" + std::string(op) +
                            "' has no shape_infer function registered (ARCH-041)");
  }

  NodeContext ctx;
  ctx.op = op;
  ctx.input_types = input_types;
  ctx.attrs = &attrs;

  // 非 const:允许 return 时自动移动(performance-no-automatic-move,与
  // src/runtime/compile.cpp 同款理由)。
  Result<std::vector<Shape>> inferred = infer_fn(ctx);
  if (!inferred.is_ok()) {
    return Status::make(inferred.status().code(),
                        "create_node_with_inferred_types: op '" + std::string(op) +
                            "' shape_infer failed: " + std::string(inferred.status().message()));
  }
  return inferred;
}

// create_node 成功后的公共收尾:逐 attrs 写入 node。
void apply_attrs(ir::Node& node, const AttrMap& attrs) {
  for (const auto& [name, value] : attrs) {
    node.set_attr(name, value);
  }
}

}  // namespace

Result<ir::Node*> create_node_with_inferred_types(ir::Graph& graph, std::string_view op,
                                                  std::vector<ir::Value*> inputs,
                                                  const AttrMap& attrs) {
  if (inputs.empty()) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "create_node_with_inferred_types: op '" + std::string(op) +
                            "' called with 0 inputs; use the Device-taking overload for "
                            "zero-input ops");
  }

  std::vector<ir::TensorType> input_types;
  input_types.reserve(inputs.size());
  for (const ir::Value* input : inputs) {
    input_types.push_back(input->type());
  }

  const Result<std::vector<Shape>> inferred = infer_output_shapes(op, input_types, attrs);
  if (!inferred.is_ok()) return inferred.status();

  // 有输入契约(裁决修订 5):输出 dtype/device 取 inputs[0],layout 固定
  // kUnknown(留给 layout_assignment pass 指派)。
  const ir::TensorType& first_input_type = input_types[0];
  std::vector<ir::TensorType> output_types;
  output_types.reserve(inferred.value().size());
  for (const Shape& shape : inferred.value()) {
    ir::TensorType output_type;
    output_type.dtype = first_input_type.dtype;
    output_type.shape = shape;
    output_type.layout = ir::Layout::kUnknown;
    output_type.device = first_input_type.device;
    output_types.push_back(output_type);
  }

  Result<ir::Node*> node =
      graph.create_node(std::string(op), std::move(inputs), std::move(output_types));
  if (!node.is_ok()) return node.status();

  apply_attrs(*node.value(), attrs);
  return node.value();
}

Result<ir::Node*> create_node_with_inferred_types(ir::Graph& graph, std::string_view op,
                                                  Device device, const AttrMap& attrs) {
  const Result<std::vector<Shape>> inferred = infer_output_shapes(op, {}, attrs);
  if (!inferred.is_ok()) return inferred.status();

  // 0 输入契约(裁决修订 5,constant 等):dtype 取 attrs 的 "dtype" 属性,
  // device 由调用方显式提供,layout 固定 kUnknown。
  const auto dtype_it = attrs.find("dtype");
  const DType* dtype_attr =
      dtype_it != attrs.end() ? std::get_if<DType>(&dtype_it->second) : nullptr;
  if (dtype_attr == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "create_node_with_inferred_types: op '" + std::string(op) +
                            "' requires a 'dtype' attribute (DType) to construct output "
                            "TensorType for a zero-input op");
  }

  std::vector<ir::TensorType> output_types;
  output_types.reserve(inferred.value().size());
  for (const Shape& shape : inferred.value()) {
    ir::TensorType output_type;
    output_type.dtype = *dtype_attr;
    output_type.shape = shape;
    output_type.layout = ir::Layout::kUnknown;
    output_type.device = device;
    output_types.push_back(output_type);
  }

  Result<ir::Node*> node = graph.create_node(std::string(op), {}, std::move(output_types));
  if (!node.is_ok()) return node.status();

  apply_attrs(*node.value(), attrs);
  return node.value();
}

}  // namespace frame::ops
