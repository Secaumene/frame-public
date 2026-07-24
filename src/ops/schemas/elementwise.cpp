// 内置逐元素算子 schema 注册桩(add/sub/mul/div/relu/square 等)。
// 逐元素算子输出 shape 等于输入 shape,标注 OpTrait::kElementwise(fusion 候选依据)。
// add/mul/relu/square 已在本文件下方注册(M5 首批 elementwise 四算子齐);
// sub/div 见下方待办标注。

// TODO(FRAME-IMPL): sub/div 属未来批次,不在 M5 首批范围(elementwise v0 四
//   算子 add/mul/relu/square 已全部注册,见 PLAN.md「M5 内置算子 v0 批次」行)。
//   参考:docs/architecture/operator-system.md 第2/3章;include/frame/ops/op_schema.h。
//   完成判据:sub/div 各自分支落地后 OpRegistry::find("sub")/("div") 均可取到
//   schema(含 shape_infer),tests/cpp/ops/ 用例通过。

#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/constant_utils.h>
#include <frame/ops/graph_builder.h>
#include <frame/ops/op_registry.h>

#include "schema_math.h"

namespace {

// 构造 shape 全形常量节点:同目录共享工具(铁律 5,见 schema_math.h 头注释)。
using frame::ops::schemas::make_constant_splat;
using frame::ops::schemas::static_shape_numel_fits_int64;

// 二元逐元素算子(add/mul)共用的 shape 推断:v0 无广播 —— 恰 2 输入、两输入
// dtype 相同、shape 相同,输出复用该 shape(与任一输入一致);任一违例返回
// 英文错误(消息含双方实际值,ARCH-031 口径:不静默降级)。op_name 仅用于拼
// 错误消息,校验逻辑本身与具体算子无关(REUSE-002:add/mul 共用本函数,禁止
// 逐算子复制第二份同构校验)。
frame::Result<std::vector<frame::Shape>> infer_binary_elementwise_shape(
    std::string_view op_name, const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' expects 2 inputs, got " +
                                   std::to_string(ctx.input_types.size()));
  }
  const frame::ir::TensorType& lhs = ctx.input_types[0];
  const frame::ir::TensorType& rhs = ctx.input_types[1];
  if (!(lhs.dtype == rhs.dtype)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) + "' requires lhs and rhs of the same dtype, got '" +
            std::string(lhs.dtype.name()) + "' and '" + std::string(rhs.dtype.name()) + "'");
  }
  if (!(lhs.shape == rhs.shape)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) +
            "' requires lhs and rhs of the same shape (v0 has no broadcasting), got " +
            lhs.shape.to_string() + " and " + rhs.shape.to_string());
  }
  return std::vector<frame::Shape>{lhs.shape};
}

// ShapeInferFn 是零捕获函数指针(见 op_schema.h),不能直接绑定带额外形参的
// infer_binary_elementwise_shape;以下两个薄封装各自把 op 名字面量转发给共用
// 校验函数,本身不重复任何校验逻辑。
frame::Result<std::vector<frame::Shape>> infer_add_shape(const frame::ops::NodeContext& ctx) {
  return infer_binary_elementwise_shape("add", ctx);
}

frame::Result<std::vector<frame::Shape>> infer_mul_shape(const frame::ops::NodeContext& ctx) {
  return infer_binary_elementwise_shape("mul", ctx);
}

// 一元逐元素算子(relu 等)共用的 shape 推断:恰 1 输入,输出恒等于输入的
// shape/dtype;输入数不为 1 时返回英文错误(消息含实际值,ARCH-031 口径:不
// 静默降级)。与二元版 infer_binary_elementwise_shape 并列而非合并——二者
// 校验的输入个数不同(1 vs 2)、错误消息模板不同,强行合一需要额外分支判断,
// 反而掩盖各自语义,不属于 REUSE-002 意图消灭的"同构复制"。
frame::Result<std::vector<frame::Shape>> infer_unary_elementwise_shape(
    std::string_view op_name, const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' expects 1 input, got " +
                                   std::to_string(ctx.input_types.size()));
  }
  return std::vector<frame::Shape>{ctx.input_types[0].shape};
}

frame::Result<std::vector<frame::Shape>> infer_relu_shape(const frame::ops::NodeContext& ctx) {
  return infer_unary_elementwise_shape("relu", ctx);
}

frame::Result<std::vector<frame::Shape>> infer_square_shape(const frame::ops::NodeContext& ctx) {
  return infer_unary_elementwise_shape("square", ctx);
}

// sigmoid 同为一元逐元素算子(M21,批3 T4),复用 infer_unary_elementwise_shape
// 这一份骨架(REUSE-002,同 infer_relu_shape/infer_square_shape 先例)。
frame::Result<std::vector<frame::Shape>> infer_sigmoid_shape(const frame::ops::NodeContext& ctx) {
  return infer_unary_elementwise_shape("sigmoid", ctx);
}

// 代理阶跃同为一元逐元素算子,但额外限制三档浮点与必需正有限 alpha。
frame::Result<std::vector<frame::Shape>> infer_heaviside_surrogate_shape(
    const frame::ops::NodeContext& ctx) {
  frame::Result<std::vector<frame::Shape>> shapes =
      infer_unary_elementwise_shape("heaviside_surrogate", ctx);
  if (!shapes.is_ok()) return shapes.status();

  const frame::ir::TensorType& x = ctx.input_types[0];
  const frame::DTypeCode code = x.dtype.code();
  const bool supported = code == frame::DTypeCode::kFloat32 || code == frame::DTypeCode::kFloat16 ||
                         code == frame::DTypeCode::kBFloat16;
  if (!supported) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'heaviside_surrogate' does not support dtype '" +
                                   std::string(x.dtype.name()) +
                                   "' (supports float32/float16/bfloat16 only)");
  }
  for (int64_t dim : x.shape.dims()) {
    if (dim < 0) {
      return frame::Status::make(
          frame::ErrorCode::kInvalidArgument,
          "op 'heaviside_surrogate' requires a fully static input shape, got " +
              x.shape.to_string());
    }
  }
  if (!static_shape_numel_fits_int64(x.shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'heaviside_surrogate' input shape " + x.shape.to_string() +
                                   " element count overflows int64");
  }

  const double* alpha = ctx.attr<double>("alpha");
  if (alpha == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'heaviside_surrogate' is missing required attribute 'alpha'");
  }
  if (!std::isfinite(*alpha) || !(*alpha > 0.0)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'heaviside_surrogate' attribute 'alpha' must be finite and positive, got " +
            std::to_string(*alpha));
  }
  return shapes;
}

// relu_grad_internal(x, gy) 的 shape 推断(M17,relu 的梯度 internal 算子):
// 恰 2 输入(x/gy)、同 dtype/shape,输出 = 该 shape——与 add/mul 的二元校验
// 规则完全一致(x>0 处透传 gy、余 0 是 kernel 层语义,不影响 shape/dtype 约束
// 本身),直接复用 infer_binary_elementwise_shape 这一份实现(REUSE-002,同
// infer_add_shape/infer_mul_shape 先例)。
frame::Result<std::vector<frame::Shape>> infer_relu_grad_internal_shape(
    const frame::ops::NodeContext& ctx) {
  return infer_binary_elementwise_shape("relu_grad_internal", ctx);
}

// square 的 decomposition:square(x) = mul(x, x)——全仓首个 decomposition 注册
// (m5-design-brief 决议点 4)。用途是 M10 回退链②的测试素材(未实现 square
// kernel 的后端可执行本微图代替),**不替代** cpu kernel(ARCH-041 仍要求二者
// 成对注册,decomposition 是补充而非豁免)。纯函数、不修改既有图(ARCH-021);
// 按位对应契约:graph_inputs[0] 对应本算子唯一输入,图输出[0] 对应本算子
// 唯一输出。微图结构:1 个 graph_input(类型取 ctx.input_types[0])→ 1 个
// mul 节点(两输入引用同一个 Value,即 x*x)→ mark_output 该 mul 节点的唯一
// 输出。构图各步理论上不应失败(输入类型已由 mul 承诺的同 shape/dtype 契约
// 满足,恰是自身与自身相乘),但仍逐步透传 Status,不吞掉任何构图期错误。
frame::Result<frame::ir::Graph> square_decompose(const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'square' decomposition expects 1 input, got " + std::to_string(ctx.input_types.size()));
  }

  frame::ir::Graph graph("square_decompose");

  const frame::Result<frame::ir::Value*> input_result = graph.add_graph_input(ctx.input_types[0]);
  if (!input_result.is_ok()) {
    return input_result.status();
  }
  frame::ir::Value* x = input_result.value();

  // 两个输入均引用同一个 Value x(square(x) = mul(x, x)),输出类型与输入
  // 恒等(square 的 shape_infer 亦是恒等)。
  const frame::Result<frame::ir::Node*> mul_result =
      graph.create_node("mul", {x, x}, {ctx.input_types[0]});
  if (!mul_result.is_ok()) {
    return mul_result.status();
  }
  frame::ir::Node* mul_node = mul_result.value();

  const frame::Status mark_status = graph.mark_output(mul_node, 0);
  if (!mark_status.is_ok()) {
    return mark_status;
  }

  return graph;
}

// ---------------------------------------------------------------------------
// M17 GradientFn(docs/architecture/autograd.md 第3/4章,ARCH-063):按位契约
// graph_inputs=[x_0..x_{n-1}, y_0..y_{m-1}, gy_0..gy_{m-1}](不需要的位也须
// 占位声明),图输出=[gx_0..gx_{n-1}]。构图风格与 square_decompose 一致
// (add_graph_input/create_node/mark_output,纯函数、不修改既有图)。
// ---------------------------------------------------------------------------

// add(a,b) 的梯度:ga=gy、gb=gy(同一 Value 重复 mark_output,M2 既有能力)。
// a/b/y 三位占位声明但用不到(add 的梯度不依赖任何操作数的具体值)。
frame::Result<frame::ir::Graph> add_gradient(const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'add' gradient expects 2 inputs, got " + std::to_string(ctx.input_types.size()));
  }
  frame::ir::Graph graph("add_gradient");

  const frame::Result<frame::ir::Value*> a_in = graph.add_graph_input(ctx.input_types[0]);
  if (!a_in.is_ok()) return a_in.status();
  const frame::Result<frame::ir::Value*> b_in = graph.add_graph_input(ctx.input_types[1]);
  if (!b_in.is_ok()) return b_in.status();

  const frame::Result<std::vector<frame::Shape>> y_shapes = infer_add_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const frame::Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const frame::Result<frame::ir::Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();

  const frame::Status mark_ga = graph.mark_output(gy_in.value());
  if (!mark_ga.is_ok()) return mark_ga;
  const frame::Status mark_gb = graph.mark_output(gy_in.value());
  if (!mark_gb.is_ok()) return mark_gb;

  return graph;
}

// mul(a,b) 的梯度:ga=mul(gy,b)、gb=mul(gy,a)。
frame::Result<frame::ir::Graph> mul_gradient(const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'mul' gradient expects 2 inputs, got " + std::to_string(ctx.input_types.size()));
  }
  frame::ir::Graph graph("mul_gradient");

  const frame::Result<frame::ir::Value*> a_in = graph.add_graph_input(ctx.input_types[0]);
  if (!a_in.is_ok()) return a_in.status();
  const frame::Result<frame::ir::Value*> b_in = graph.add_graph_input(ctx.input_types[1]);
  if (!b_in.is_ok()) return b_in.status();

  const frame::Result<std::vector<frame::Shape>> y_shapes = infer_mul_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const frame::Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const frame::Result<frame::ir::Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();

  const frame::Result<frame::ir::Node*> ga_node =
      frame::ops::create_node_with_inferred_types(graph, "mul", {gy_in.value(), b_in.value()});
  if (!ga_node.is_ok()) return ga_node.status();
  const frame::Result<frame::ir::Node*> gb_node =
      frame::ops::create_node_with_inferred_types(graph, "mul", {gy_in.value(), a_in.value()});
  if (!gb_node.is_ok()) return gb_node.status();

  const frame::Status mark_ga = graph.mark_output(ga_node.value(), 0);
  if (!mark_ga.is_ok()) return mark_ga;
  const frame::Status mark_gb = graph.mark_output(gb_node.value(), 0);
  if (!mark_gb.is_ok()) return mark_gb;

  return graph;
}

// square(x) 的梯度:gx=mul(gy, mul(constant(2), x))(§4 清单原文)。
frame::Result<frame::ir::Graph> square_gradient(const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'square' gradient expects 1 input, got " + std::to_string(ctx.input_types.size()));
  }
  frame::ir::Graph graph("square_gradient");

  const frame::Result<frame::ir::Value*> x_in = graph.add_graph_input(ctx.input_types[0]);
  if (!x_in.is_ok()) return x_in.status();

  const frame::Result<std::vector<frame::Shape>> y_shapes = infer_square_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const frame::Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const frame::Result<frame::ir::Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();

  // constant(2):shape 与 x 完全一致(v0 mul 无广播,不能用 rank-0 标量再乘;
  // GradientFn 微图内类型必须具体,ctx.input_types[0].shape 在构图期已知静态
  // 值,§3 头注释"具体类型、可过 verify、非占位模板")。
  const int64_t x_numel = ctx.input_types[0].shape.numel();
  const std::vector<double> two_values(static_cast<size_t>(x_numel), 2.0);
  const frame::ops::AttrMap two_attrs{
      {"value", two_values},
      {"shape", ctx.input_types[0].shape},
      {"dtype", ctx.input_types[0].dtype},
  };
  const frame::Result<frame::ir::Node*> two_node = frame::ops::create_node_with_inferred_types(
      graph, frame::ops::kConstantOpName, ctx.input_types[0].device, two_attrs);
  if (!two_node.is_ok()) return two_node.status();

  const frame::Result<frame::ir::Node*> two_x_node = frame::ops::create_node_with_inferred_types(
      graph, "mul", {two_node.value()->output(0), x_in.value()});
  if (!two_x_node.is_ok()) return two_x_node.status();
  const frame::Result<frame::ir::Node*> gx_node = frame::ops::create_node_with_inferred_types(
      graph, "mul", {gy_in.value(), two_x_node.value()->output(0)});
  if (!gx_node.is_ok()) return gx_node.status();

  const frame::Status mark_gx = graph.mark_output(gx_node.value(), 0);
  if (!mark_gx.is_ok()) return mark_gx;

  return graph;
}

// relu(x) 的梯度:gx=relu_grad_internal(x, gy)(x>0 处透传 gy,余 0)。
frame::Result<frame::ir::Graph> relu_gradient(const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'relu' gradient expects 1 input, got " + std::to_string(ctx.input_types.size()));
  }
  frame::ir::Graph graph("relu_gradient");

  const frame::Result<frame::ir::Value*> x_in = graph.add_graph_input(ctx.input_types[0]);
  if (!x_in.is_ok()) return x_in.status();

  const frame::Result<std::vector<frame::Shape>> y_shapes = infer_relu_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const frame::Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const frame::Result<frame::ir::Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();

  const frame::Result<frame::ir::Node*> gx_node = frame::ops::create_node_with_inferred_types(
      graph, "relu_grad_internal", {x_in.value(), gy_in.value()});
  if (!gx_node.is_ok()) return gx_node.status();

  const frame::Status mark_gx = graph.mark_output(gx_node.value(), 0);
  if (!mark_gx.is_ok()) return mark_gx;

  return graph;
}

// relu_grad_internal(x,gy)=I(x>0)·gy 的梯度(M26,ARCH-068)。ReLU 二阶导
// 在 kink 外为零,kink 处沿既有 x>0 口径同样取零;对 gy 的梯度继续复用
// relu_grad_internal(x,ggx),保持任意有限次图变换的闭包。
frame::Result<frame::ir::Graph> relu_grad_internal_gradient(const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'relu_grad_internal' gradient expects 2 inputs, got " +
                                   std::to_string(ctx.input_types.size()));
  }
  const frame::Result<std::vector<frame::Shape>> output_shapes =
      infer_relu_grad_internal_shape(ctx);
  if (!output_shapes.is_ok()) return output_shapes.status();

  frame::ir::Graph graph("relu_grad_internal_gradient");
  const frame::Result<frame::ir::Value*> x_in = graph.add_graph_input(ctx.input_types[0]);
  if (!x_in.is_ok()) return x_in.status();
  const frame::Result<frame::ir::Value*> gy_in = graph.add_graph_input(ctx.input_types[1]);
  if (!gy_in.is_ok()) return gy_in.status();
  frame::ir::TensorType gx_type = ctx.input_types[0];
  gx_type.shape = output_shapes.value()[0];
  const frame::Result<frame::ir::Value*> gx_in = graph.add_graph_input(gx_type);
  if (!gx_in.is_ok()) return gx_in.status();
  const frame::Result<frame::ir::Value*> ggx_in = graph.add_graph_input(gx_type);
  if (!ggx_in.is_ok()) return ggx_in.status();

  const frame::Result<frame::ir::Node*> g_x_node = make_constant_splat(
      graph, ctx.input_types[0].shape, ctx.input_types[0].dtype, ctx.input_types[0].device, 0.0);
  if (!g_x_node.is_ok()) return g_x_node.status();
  const frame::Result<frame::ir::Node*> g_gy_node = frame::ops::create_node_with_inferred_types(
      graph, "relu_grad_internal", {x_in.value(), ggx_in.value()});
  if (!g_gy_node.is_ok()) return g_gy_node.status();

  const frame::Status mark_g_x = graph.mark_output(g_x_node.value(), 0);
  if (!mark_g_x.is_ok()) return mark_g_x;
  const frame::Status mark_g_gy = graph.mark_output(g_gy_node.value(), 0);
  if (!mark_g_gy.is_ok()) return mark_g_gy;
  return graph;
}

// sigmoid(x) 的梯度(M21,批3 T4):s=sigmoid(x),gx=gy*s*(1-s)。1-s 经
// add(constant(1), mul(constant(-1), s)) 组合(v0 无 sub 算子,mse_loss_gradient
// 的 constant(-1) 先例同机制,REUSE-002);常量经 constant 全形烘焙
// (square_gradient 的 constant(2) 先例同机制)。
frame::Result<frame::ir::Graph> sigmoid_gradient(const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sigmoid' gradient expects 1 input, got " + std::to_string(ctx.input_types.size()));
  }
  frame::ir::Graph graph("sigmoid_gradient");

  const frame::Result<frame::ir::Value*> x_in = graph.add_graph_input(ctx.input_types[0]);
  if (!x_in.is_ok()) return x_in.status();

  const frame::Result<std::vector<frame::Shape>> y_shapes = infer_sigmoid_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const frame::Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const frame::Result<frame::ir::Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();

  const frame::Result<frame::ir::Node*> s_node =
      frame::ops::create_node_with_inferred_types(graph, "sigmoid", {x_in.value()});
  if (!s_node.is_ok()) return s_node.status();
  frame::ir::Value* s = s_node.value()->output(0);

  const int64_t x_numel = ctx.input_types[0].shape.numel();
  const std::vector<double> one_values(static_cast<size_t>(x_numel), 1.0);
  const frame::ops::AttrMap one_attrs{
      {"value", one_values},
      {"shape", ctx.input_types[0].shape},
      {"dtype", ctx.input_types[0].dtype},
  };
  const frame::Result<frame::ir::Node*> one_node = frame::ops::create_node_with_inferred_types(
      graph, frame::ops::kConstantOpName, ctx.input_types[0].device, one_attrs);
  if (!one_node.is_ok()) return one_node.status();

  const std::vector<double> neg_one_values(static_cast<size_t>(x_numel), -1.0);
  const frame::ops::AttrMap neg_one_attrs{
      {"value", neg_one_values},
      {"shape", ctx.input_types[0].shape},
      {"dtype", ctx.input_types[0].dtype},
  };
  const frame::Result<frame::ir::Node*> neg_one_node = frame::ops::create_node_with_inferred_types(
      graph, frame::ops::kConstantOpName, ctx.input_types[0].device, neg_one_attrs);
  if (!neg_one_node.is_ok()) return neg_one_node.status();

  const frame::Result<frame::ir::Node*> neg_s_node = frame::ops::create_node_with_inferred_types(
      graph, "mul", {neg_one_node.value()->output(0), s});
  if (!neg_s_node.is_ok()) return neg_s_node.status();

  const frame::Result<frame::ir::Node*> one_minus_s_node =
      frame::ops::create_node_with_inferred_types(
          graph, "add", {one_node.value()->output(0), neg_s_node.value()->output(0)});
  if (!one_minus_s_node.is_ok()) return one_minus_s_node.status();

  const frame::Result<frame::ir::Node*> s_times_one_minus_s_node =
      frame::ops::create_node_with_inferred_types(graph, "mul",
                                                  {s, one_minus_s_node.value()->output(0)});
  if (!s_times_one_minus_s_node.is_ok()) return s_times_one_minus_s_node.status();

  const frame::Result<frame::ir::Node*> gx_node = frame::ops::create_node_with_inferred_types(
      graph, "mul", {gy_in.value(), s_times_one_minus_s_node.value()->output(0)});
  if (!gx_node.is_ok()) return gx_node.status();

  const frame::Status mark_gx = graph.mark_output(gx_node.value(), 0);
  if (!mark_gx.is_ok()) return mark_gx;

  return graph;
}

// heaviside_surrogate 的代理梯度固定为 gy*alpha*s*(1-s),其中
// s=sigmoid(alpha*x)。离散前向不参与该微图,高阶导数继续变换公开算子。
frame::Result<frame::ir::Graph> heaviside_surrogate_gradient(const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'heaviside_surrogate' gradient expects 1 input, got " +
                                   std::to_string(ctx.input_types.size()));
  }
  const frame::Result<std::vector<frame::Shape>> y_shapes = infer_heaviside_surrogate_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();

  frame::ir::Graph graph("heaviside_surrogate_gradient");
  const frame::Result<frame::ir::Value*> x_in = graph.add_graph_input(ctx.input_types[0]);
  if (!x_in.is_ok()) return x_in.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const frame::Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const frame::Result<frame::ir::Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();

  const frame::Shape& shape = ctx.input_types[0].shape;
  const frame::DType dtype = ctx.input_types[0].dtype;
  const frame::Device device = ctx.input_types[0].device;
  const double alpha = *ctx.attr<double>("alpha");
  const frame::Result<frame::ir::Node*> alpha_node =
      make_constant_splat(graph, shape, dtype, device, alpha);
  if (!alpha_node.is_ok()) return alpha_node.status();
  const frame::Result<frame::ir::Node*> scaled_node = frame::ops::create_node_with_inferred_types(
      graph, "mul", {alpha_node.value()->output(0), x_in.value()});
  if (!scaled_node.is_ok()) return scaled_node.status();
  const frame::Result<frame::ir::Node*> s_node = frame::ops::create_node_with_inferred_types(
      graph, "sigmoid", {scaled_node.value()->output(0)});
  if (!s_node.is_ok()) return s_node.status();

  const frame::Result<frame::ir::Node*> one_node =
      make_constant_splat(graph, shape, dtype, device, 1.0);
  if (!one_node.is_ok()) return one_node.status();
  const frame::Result<frame::ir::Node*> neg_one_node =
      make_constant_splat(graph, shape, dtype, device, -1.0);
  if (!neg_one_node.is_ok()) return neg_one_node.status();
  const frame::Result<frame::ir::Node*> neg_s_node = frame::ops::create_node_with_inferred_types(
      graph, "mul", {neg_one_node.value()->output(0), s_node.value()->output(0)});
  if (!neg_s_node.is_ok()) return neg_s_node.status();
  const frame::Result<frame::ir::Node*> one_minus_s_node =
      frame::ops::create_node_with_inferred_types(
          graph, "add", {one_node.value()->output(0), neg_s_node.value()->output(0)});
  if (!one_minus_s_node.is_ok()) return one_minus_s_node.status();
  const frame::Result<frame::ir::Node*> smooth_node = frame::ops::create_node_with_inferred_types(
      graph, "mul", {s_node.value()->output(0), one_minus_s_node.value()->output(0)});
  if (!smooth_node.is_ok()) return smooth_node.status();
  const frame::Result<frame::ir::Node*> scaled_smooth_node =
      frame::ops::create_node_with_inferred_types(
          graph, "mul", {alpha_node.value()->output(0), smooth_node.value()->output(0)});
  if (!scaled_smooth_node.is_ok()) return scaled_smooth_node.status();
  const frame::Result<frame::ir::Node*> gx_node = frame::ops::create_node_with_inferred_types(
      graph, "mul", {gy_in.value(), scaled_smooth_node.value()->output(0)});
  if (!gx_node.is_ok()) return gx_node.status();

  const frame::Status mark_gx = graph.mark_output(gx_node.value(), 0);
  if (!mark_gx.is_ok()) return mark_gx;
  return graph;
}

// tanh 同为一元逐元素算子(M22,批4 T3),复用 infer_unary_elementwise_shape
// 这一份骨架(REUSE-002,同 infer_relu_shape/infer_sigmoid_shape 先例)。
frame::Result<std::vector<frame::Shape>> infer_tanh_shape(const frame::ops::NodeContext& ctx) {
  return infer_unary_elementwise_shape("tanh", ctx);
}

// rsqrt(x) = x^(-1/2),同为一元逐元素算子,同上复用骨架。
frame::Result<std::vector<frame::Shape>> infer_rsqrt_shape(const frame::ops::NodeContext& ctx) {
  return infer_unary_elementwise_shape("rsqrt", ctx);
}

// tanh(x) 的梯度(M22,批4 T3,§1.2 表):t=tanh(x) 重算(sigmoid_gradient 从
// x 重算 s 的先例,可与前向节点 CSE);gx = gy·(1 + (−1)·t·t),经
// constant(±1) splat + mul/add 组合(v0 无 sub 算子,同 sigmoid_gradient
// 机制)。
frame::Result<frame::ir::Graph> tanh_gradient(const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'tanh' gradient expects 1 input, got " + std::to_string(ctx.input_types.size()));
  }
  frame::ir::Graph graph("tanh_gradient");

  const frame::Result<frame::ir::Value*> x_in = graph.add_graph_input(ctx.input_types[0]);
  if (!x_in.is_ok()) return x_in.status();

  const frame::Result<std::vector<frame::Shape>> y_shapes = infer_tanh_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const frame::Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const frame::Result<frame::ir::Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();

  const frame::Result<frame::ir::Node*> t_node =
      frame::ops::create_node_with_inferred_types(graph, "tanh", {x_in.value()});
  if (!t_node.is_ok()) return t_node.status();
  frame::ir::Value* t = t_node.value()->output(0);

  const frame::Result<frame::ir::Node*> t2_node =
      frame::ops::create_node_with_inferred_types(graph, "mul", {t, t});
  if (!t2_node.is_ok()) return t2_node.status();

  const frame::Shape& x_shape = ctx.input_types[0].shape;
  const frame::DType x_dtype = ctx.input_types[0].dtype;
  const frame::Device x_device = ctx.input_types[0].device;

  const frame::Result<frame::ir::Node*> neg_one_node =
      make_constant_splat(graph, x_shape, x_dtype, x_device, -1.0);
  if (!neg_one_node.is_ok()) return neg_one_node.status();
  const frame::Result<frame::ir::Node*> neg_t2_node = frame::ops::create_node_with_inferred_types(
      graph, "mul", {neg_one_node.value()->output(0), t2_node.value()->output(0)});
  if (!neg_t2_node.is_ok()) return neg_t2_node.status();

  const frame::Result<frame::ir::Node*> one_node =
      make_constant_splat(graph, x_shape, x_dtype, x_device, 1.0);
  if (!one_node.is_ok()) return one_node.status();
  const frame::Result<frame::ir::Node*> one_minus_t2_node =
      frame::ops::create_node_with_inferred_types(
          graph, "add", {one_node.value()->output(0), neg_t2_node.value()->output(0)});
  if (!one_minus_t2_node.is_ok()) return one_minus_t2_node.status();

  const frame::Result<frame::ir::Node*> gx_node = frame::ops::create_node_with_inferred_types(
      graph, "mul", {gy_in.value(), one_minus_t2_node.value()->output(0)});
  if (!gx_node.is_ok()) return gx_node.status();

  const frame::Status mark_gx = graph.mark_output(gx_node.value(), 0);
  if (!mark_gx.is_ok()) return mark_gx;

  return graph;
}

// rsqrt(x) 的梯度(M22,批4 T3,§1.2 表):r=rsqrt(x) 重算;
// gx = gy·(−0.5)·r·r·r,经 constant(−0.5) splat + mul 组合。r 是 M22
// 新增的 spec 外增项(layer_norm 梯度需要 1/√(σ²+ε),设计门批注见
// docs/plan/2026-07-19-batch4-m22-seq.md §1.2)。
frame::Result<frame::ir::Graph> rsqrt_gradient(const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'rsqrt' gradient expects 1 input, got " + std::to_string(ctx.input_types.size()));
  }
  frame::ir::Graph graph("rsqrt_gradient");

  const frame::Result<frame::ir::Value*> x_in = graph.add_graph_input(ctx.input_types[0]);
  if (!x_in.is_ok()) return x_in.status();

  const frame::Result<std::vector<frame::Shape>> y_shapes = infer_rsqrt_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const frame::Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const frame::Result<frame::ir::Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();

  const frame::Result<frame::ir::Node*> r_node =
      frame::ops::create_node_with_inferred_types(graph, "rsqrt", {x_in.value()});
  if (!r_node.is_ok()) return r_node.status();
  frame::ir::Value* r = r_node.value()->output(0);

  const frame::Result<frame::ir::Node*> r2_node =
      frame::ops::create_node_with_inferred_types(graph, "mul", {r, r});
  if (!r2_node.is_ok()) return r2_node.status();
  const frame::Result<frame::ir::Node*> r3_node =
      frame::ops::create_node_with_inferred_types(graph, "mul", {r2_node.value()->output(0), r});
  if (!r3_node.is_ok()) return r3_node.status();

  const frame::Shape& x_shape = ctx.input_types[0].shape;
  const frame::DType x_dtype = ctx.input_types[0].dtype;
  const frame::Device x_device = ctx.input_types[0].device;

  const frame::Result<frame::ir::Node*> neg_half_node =
      make_constant_splat(graph, x_shape, x_dtype, x_device, -0.5);
  if (!neg_half_node.is_ok()) return neg_half_node.status();
  const frame::Result<frame::ir::Node*> coeff_node = frame::ops::create_node_with_inferred_types(
      graph, "mul", {neg_half_node.value()->output(0), r3_node.value()->output(0)});
  if (!coeff_node.is_ok()) return coeff_node.status();

  const frame::Result<frame::ir::Node*> gx_node = frame::ops::create_node_with_inferred_types(
      graph, "mul", {gy_in.value(), coeff_node.value()->output(0)});
  if (!gx_node.is_ok()) return gx_node.status();

  const frame::Status mark_gx = graph.mark_output(gx_node.value(), 0);
  if (!mark_gx.is_ok()) return mark_gx;

  return graph;
}

}  // namespace

FRAME_REGISTER_OP("add")
    .input("lhs", "left operand")
    .input("rhs", "right operand")
    .output("out", "elementwise sum of lhs and rhs")
    .trait(frame::ops::OpTrait::kElementwise)
    .trait(frame::ops::OpTrait::kFusable)
    .trait(frame::ops::OpTrait::kCommutative)
    .shape_infer(&infer_add_shape)
    .gradient(&add_gradient);

FRAME_REGISTER_OP("mul")
    .input("lhs", "left operand")
    .input("rhs", "right operand")
    .output("out", "elementwise product of lhs and rhs")
    .trait(frame::ops::OpTrait::kElementwise)
    .trait(frame::ops::OpTrait::kFusable)
    .trait(frame::ops::OpTrait::kCommutative)
    .shape_infer(&infer_mul_shape)
    .gradient(&mul_gradient);

// relu 是一元算子(非可交换,不标 kCommutative——该 trait 语义仅适用于二元
// 可交换运算,见 op_schema.h OpTrait 定义)。
FRAME_REGISTER_OP("relu")
    .input("x", "input tensor")
    .output("out", "elementwise relu (max(x, 0)) of x")
    .trait(frame::ops::OpTrait::kElementwise)
    .trait(frame::ops::OpTrait::kFusable)
    .shape_infer(&infer_relu_shape)
    .gradient(&relu_gradient);

// square 同样是一元算子,不标 kCommutative;额外注册 decomposition(见
// square_decompose 头注释),cpu kernel 见
// src/backends/cpu/kernels/elementwise.cpp。
FRAME_REGISTER_OP("square")
    .input("x", "input tensor")
    .output("out", "elementwise square (x * x) of x")
    .trait(frame::ops::OpTrait::kElementwise)
    .trait(frame::ops::OpTrait::kFusable)
    .shape_infer(&infer_square_shape)
    .decomposition(&square_decompose)
    .gradient(&square_gradient);

// sigmoid(x) = 1/(1+e^-x)(M21,批3 T4)。仅标 kElementwise,不标 kFusable——
// fused_elementwise 内核(src/backends/cpu/kernels/fused_elementwise.cpp)当前
// 不认识 sigmoid 表达式,标了会被 operator_fusion pass 融合出内核不支持的坏图。
// TODO(FRAME-PERF): sigmoid 融合支持。参考:src/ops/schemas/fused_elementwise.cpp。
//   完成判据:fused kernel 表达式集含 sigmoid 且 kFusable 打开后 dev 全测过。
FRAME_REGISTER_OP("sigmoid")
    .input("x", "input tensor")
    .output("out", "elementwise sigmoid (1 / (1 + exp(-x))) of x")
    .trait(frame::ops::OpTrait::kElementwise)
    .shape_infer(&infer_sigmoid_shape)
    .gradient(&sigmoid_gradient);

// 代理阶跃前向为离散 x>=0,但 GradientFn 使用批准的平滑 sigmoid 代理。
// 不标 kFusable,避免融合后丢失 alpha 与专属代理梯度语义。
FRAME_REGISTER_OP("heaviside_surrogate")
    .input("x", "input tensor")
    .attr("alpha", frame::ir::AttrType::kDouble, /*required=*/true)
    .output("out", "elementwise step: one where x is non-negative, else zero")
    .trait(frame::ops::OpTrait::kElementwise)
    .shape_infer(&infer_heaviside_surrogate_shape)
    .gradient(&heaviside_surrogate_gradient);

// tanh(x)(M22,批4 T3)。仅标 kElementwise,不标 kFusable(同 sigmoid,
// fused_elementwise 内核暂不认识 tanh 表达式)。
FRAME_REGISTER_OP("tanh")
    .input("x", "input tensor")
    .output("out", "elementwise hyperbolic tangent of x")
    .trait(frame::ops::OpTrait::kElementwise)
    .shape_infer(&infer_tanh_shape)
    .gradient(&tanh_gradient);

// rsqrt(x) = x^(-1/2)(M22,批4 T3,spec 外增项——layer_norm 梯度需要
// 1/√(σ²+ε),设计门批注见 docs/plan/2026-07-19-batch4-m22-seq.md §1.2)。仅
// 标 kElementwise,不标 kFusable(同 tanh/sigmoid)。
FRAME_REGISTER_OP("rsqrt")
    .input("x", "input tensor")
    .output("out", "elementwise reciprocal square root (x^(-1/2)) of x")
    .trait(frame::ops::OpTrait::kElementwise)
    .shape_infer(&infer_rsqrt_shape)
    .gradient(&rsqrt_gradient);

// relu_grad_internal(x, gy)(M17,relu 的梯度 internal 算子):不面向用户
// (_internal 后缀,PY-021 天然豁免),仅供 relu_gradient 内联使用;cpu kernel
// 见 src/backends/cpu/kernels/elementwise.cpp。
FRAME_REGISTER_OP("relu_grad_internal")
    .input("x", "relu's original input")
    .input("gy", "upstream gradient of relu's output (same shape/dtype as x)")
    .output("gx", "gradient w.r.t. x: gy where x>0, else 0")
    .shape_infer(&infer_relu_grad_internal_shape)
    .gradient(&relu_grad_internal_gradient);
