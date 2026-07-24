// 内置 pass:形状推断(shape_inference)——v0 为校验模式(m7-design-brief 决议点
// 1):沿拓扑序对每个非 graph_input 节点重新调用其 OpSchema::shape_infer(),把
// 推断结果与既有输出类型逐项比对,发现不一致即报错;不写回(v0 构图 API 强制
// 携带 output_types,图内不存在需要"补全"的 unknown 类型,写回模式是前端允许
// 无类型构图后的未来议题)。
// pass 名 = 全词文件名(见 include/frame/compiler/pipeline.h)。

#include <cstddef>
#include <string>
#include <vector>

#include <frame/compiler/pass.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/ir/graph.h>
#include <frame/ops/op_registry.h>

namespace frame::compiler {
namespace {

// CRTP 接入:只需提供 static constexpr kName 与 run_impl(ir::Graph&)。
class ShapeInferencePass final : public PassBase<ShapeInferencePass> {
 public:
  static constexpr std::string_view kName = "shape_inference";

  Status run_impl(ir::Graph& graph) {
    // 逐节点重算并比对(校验模式):graph_input 节点无 OpSchema,天然跳过。
    for (ir::Node* node : graph.topological_order()) {
      if (node->op() == ir::kGraphInputOp) continue;

      const ops::OpSchema* schema = ops::OpRegistry::instance().find(node->op());
      if (schema == nullptr) {
        return Status::make(
            ErrorCode::kNotFound,
            "shape_inference: op '" + std::string(node->op()) + "' is not registered");
      }
      const ops::ShapeInferFn infer_fn = schema->shape_infer();
      if (infer_fn == nullptr) {
        return Status::make(ErrorCode::kInvalidArgument,
                            "shape_inference: op '" + std::string(node->op()) +
                                "' has no shape_infer function registered (ARCH-041)");
      }

      ops::NodeContext ctx;
      ctx.op = node->op();
      ctx.input_types.reserve(node->inputs().size());
      for (const ir::Value* input : node->inputs()) {
        ctx.input_types.push_back(input->type());
      }
      ctx.attrs = &node->attrs();

      const Result<std::vector<Shape>> inferred = infer_fn(ctx);
      if (!inferred.is_ok()) {
        return Status::make(
            inferred.status().code(),
            "shape_inference: node '" + std::string(node->op()) +
                "' shape_infer failed: " + std::string(inferred.status().message()));
      }
      const std::vector<Shape>& inferred_shapes = inferred.value();
      const std::vector<ir::Value>& outputs = node->outputs();

      // ①个数比对(修订节 5-⑥:比对前先比对 shape 个数与输出个数)。
      if (inferred_shapes.size() != outputs.size()) {
        return Status::make(ErrorCode::kInvalidArgument,
                            "shape_inference: node '" + std::string(node->op()) +
                                "' shape_infer returned " + std::to_string(inferred_shapes.size()) +
                                " shape(s), node has " + std::to_string(outputs.size()) +
                                " output(s)");
      }

      // ②逐位比对。
      for (size_t i = 0; i < outputs.size(); ++i) {
        const Shape& inferred_shape = inferred_shapes[i];
        const Shape& existing_shape = outputs[i].type().shape;
        if (!(inferred_shape == existing_shape)) {
          return Status::make(ErrorCode::kInvalidArgument,
                              "shape_inference: node '" + std::string(node->op()) + "' output " +
                                  std::to_string(i) + " shape mismatch: inferred " +
                                  inferred_shape.to_string() + ", existing " +
                                  existing_shape.to_string());
        }
      }

      // ③dtype 复核 v0:输出 dtype == 第 0 输入 dtype(m7-design-brief 决议点
      // 1;graph_input 无输入,已在循环开头 continue 跳过豁免;0 输入豁免自
      // M8 起由 constant 算子(ops::kConstantOpName,见
      // include/frame/ops/constant_utils.h)实际使用,守卫 !inputs().empty()
      // 避免越界)。升宽/混合精度是未来 schema 扩展议题。
      if (!node->inputs().empty()) {
        const DType expected_dtype = ctx.input_types[0].dtype;
        for (size_t i = 0; i < outputs.size(); ++i) {
          if (!(outputs[i].type().dtype == expected_dtype)) {
            return Status::make(
                ErrorCode::kInvalidArgument,
                "shape_inference: node '" + std::string(node->op()) + "' output " +
                    std::to_string(i) + " dtype '" + std::string(outputs[i].type().dtype.name()) +
                    "' does not match input 0 dtype '" + std::string(expected_dtype.name()) + "'");
          }
        }
      }
    }

    // ④全 Value 无 unknown 维显式检查(修订节 5-②):恒等透传的推断函数会原样
    // 复制输入侧已带的动态维,仅靠上面的逐位比对拦不住"两侧同为动态维"这一
    // 情形;显式遍历全图(含 graph_input)各输出的 shape,拒绝任何含动态维的
    // Value(ARCH-013/ARCH-044)。graph.verify() 的 V5 之后同样会查一遍,但 pass
    // 自身在此提前给出定位到具体节点/输出序号的诊断,而非等到 verify() 报出
    // 通用 V5 消息。
    for (const ir::Node* node : graph.topological_order()) {
      const std::vector<ir::Value>& outputs = node->outputs();
      for (size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i].type().shape.has_dynamic_dim()) {
          return Status::make(ErrorCode::kInvalidArgument,
                              "shape_inference: node '" + std::string(node->op()) + "' output " +
                                  std::to_string(i) +
                                  " has a dynamic dimension, static shape required "
                                  "(ARCH-013/ARCH-044)");
        }
      }
    }

    return Status::ok();
  }
};

}  // namespace

FRAME_REGISTER_PASS(ShapeInferencePass);

}  // namespace frame::compiler
