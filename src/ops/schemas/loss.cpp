// 内置损失算子 schema 注册桩(M17,ADR-0008/autograd.md 第4章):mse_loss 是
// 面向用户算子(无 _internal 后缀,ARCH-064;已随注册同步完成 Python 绑定,
// 见 python/src/bind_ops.cpp::mse_loss,PY-021 自 M12 起全量执法);
// mse_loss_grad_internal 是其梯度专用 internal 算子(不面向用户,PY-021 天然
// 豁免)。二者不与 add/mul/matmul/reduction 共文件——损失函数是独立于
// elementwise/matmul/reduction 三个既有门类的第四个门类,ARCH-064 已明确
// "损失核心化实现更短更稳"、不以既有算子组合表达,新开文件对齐这一独立门类
// 定位。cpu kernel 见 src/backends/cpu/kernels/loss.cpp。

#include <cstdint>
#include <string>
#include <vector>

#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/constant_utils.h>
#include <frame/ops/graph_builder.h>
#include <frame/ops/op_registry.h>

#include "schema_math.h"

namespace {

using frame::ops::schemas::make_constant_splat;

// mse_loss(pred, target) 的 shape 推断:恰 2 输入、同 dtype/shape(v0 无
// 广播,与 add/mul 的二元约束同规则),输出恒为 rank-0 标量——与
// infer_binary_elementwise_shape(src/ops/schemas/elementwise.cpp,匿名命名
// 空间私有符号,不可跨翻译单元调用)结构相似但不合并:①该函数返回值是"与
// 输入相同的 shape",而 mse_loss 的合法输出是标量,直接复用其返回值需要
// 额外丢弃再另造 Shape() 的间接层,收益有限;②错误消息的操作数名称
// (pred/target)与该函数固定的 lhs/rhs 不同,不是同一份文本的重复。
frame::Result<std::vector<frame::Shape>> infer_mse_loss_shape(const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'mse_loss' expects 2 inputs, got " + std::to_string(ctx.input_types.size()));
  }
  const frame::ir::TensorType& pred = ctx.input_types[0];
  const frame::ir::TensorType& target = ctx.input_types[1];
  if (!(pred.dtype == target.dtype)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'mse_loss' requires pred and target of the same dtype, got '" +
                                   std::string(pred.dtype.name()) + "' and '" +
                                   std::string(target.dtype.name()) + "'");
  }
  if (!(pred.shape == target.shape)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'mse_loss' requires pred and target of the same shape (v0 has no broadcasting), got " +
            pred.shape.to_string() + " and " + target.shape.to_string());
  }
  return std::vector<frame::Shape>{frame::Shape()};  // rank-0 标量
}

// mse_loss_grad_internal(pred, target, gy) 的 shape 推断(M17,mse_loss 的
// 梯度 internal 算子):恰 3 输入,pred/target 同 dtype/shape、gy 标量(rank
// 0 或 numel==1)且 dtype 与 pred 一致;输出 shape = pred.shape(gpred)。
frame::Result<std::vector<frame::Shape>> infer_mse_loss_grad_internal_shape(
    const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 3) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'mse_loss_grad_internal' expects 3 inputs, got " +
                                   std::to_string(ctx.input_types.size()));
  }
  const frame::ir::TensorType& pred = ctx.input_types[0];
  const frame::ir::TensorType& target = ctx.input_types[1];
  const frame::ir::TensorType& gy = ctx.input_types[2];

  if (!(pred.dtype == target.dtype)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'mse_loss_grad_internal' requires pred and target of the same dtype, got '" +
            std::string(pred.dtype.name()) + "' and '" + std::string(target.dtype.name()) + "'");
  }
  if (!(pred.shape == target.shape)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'mse_loss_grad_internal' requires pred and target of the same shape (v0 has no "
        "broadcasting), got " +
            pred.shape.to_string() + " and " + target.shape.to_string());
  }
  if (gy.shape.rank() != 0 && gy.shape.numel() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'mse_loss_grad_internal' requires gy to be scalar (rank 0 or "
                               "numel==1), got shape " +
                                   gy.shape.to_string());
  }
  if (!(gy.dtype == pred.dtype)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'mse_loss_grad_internal' requires gy and pred of the same dtype, got '" +
            std::string(gy.dtype.name()) + "' and '" + std::string(pred.dtype.name()) + "'");
  }

  return std::vector<frame::Shape>{pred.shape};
}

// mse_loss(pred,target) 的梯度(M17,§4 清单):
// gpred=mse_loss_grad_internal(pred,target,gy)=2*(pred-target)/N*gy;gtarget
// 同式反号,微图内以 mul(constant(-1), gpred) 组合(定案:kernel 单输出
// gpred,少一个 kernel)。
frame::Result<frame::ir::Graph> mse_loss_gradient(const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'mse_loss' gradient expects 2 inputs, got " + std::to_string(ctx.input_types.size()));
  }
  frame::ir::Graph graph("mse_loss_gradient");

  const frame::Result<frame::ir::Value*> pred_in = graph.add_graph_input(ctx.input_types[0]);
  if (!pred_in.is_ok()) return pred_in.status();
  const frame::Result<frame::ir::Value*> target_in = graph.add_graph_input(ctx.input_types[1]);
  if (!target_in.is_ok()) return target_in.status();

  const frame::Result<std::vector<frame::Shape>> y_shapes = infer_mse_loss_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const frame::Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const frame::Result<frame::ir::Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();

  const frame::Result<frame::ir::Node*> gpred_node = frame::ops::create_node_with_inferred_types(
      graph, "mse_loss_grad_internal", {pred_in.value(), target_in.value(), gy_in.value()});
  if (!gpred_node.is_ok()) return gpred_node.status();
  frame::ir::Value* gpred = gpred_node.value()->output(0);

  // gtarget = -gpred = mul(constant(-1), gpred)。constant 的 shape 须与
  // gpred(=pred.shape)完全一致(v0 mul 无广播,不能用 rank-0 标量再乘;
  // ctx.input_types[0].shape 在构图期已知静态值)。
  const int64_t pred_numel = ctx.input_types[0].shape.numel();
  const std::vector<double> neg_one_values(static_cast<size_t>(pred_numel), -1.0);
  const frame::ops::AttrMap neg_one_attrs{
      {"value", neg_one_values},
      {"shape", ctx.input_types[0].shape},
      {"dtype", ctx.input_types[0].dtype},
  };
  const frame::Result<frame::ir::Node*> neg_one_node = frame::ops::create_node_with_inferred_types(
      graph, frame::ops::kConstantOpName, ctx.input_types[0].device, neg_one_attrs);
  if (!neg_one_node.is_ok()) return neg_one_node.status();
  const frame::Result<frame::ir::Node*> gtarget_node = frame::ops::create_node_with_inferred_types(
      graph, "mul", {neg_one_node.value()->output(0), gpred});
  if (!gtarget_node.is_ok()) return gtarget_node.status();

  const frame::Status mark_gpred = graph.mark_output(gpred_node.value(), 0);
  if (!mark_gpred.is_ok()) return mark_gpred;
  const frame::Status mark_gtarget = graph.mark_output(gtarget_node.value(), 0);
  if (!mark_gtarget.is_ok()) return mark_gtarget;

  return graph;
}

// mse_loss_grad_internal(pred,target,gy) 的梯度(M26,ARCH-068)。gy 按
// schema 可为 rank-0 或任意 numel==1 shape,先 reshape 成 rank-0 再复用
// sum_grad_internal 全轴展开;返回 g_gy 时再 reshape 回原 gy shape。
frame::Result<frame::ir::Graph> mse_loss_grad_internal_gradient(
    const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 3) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'mse_loss_grad_internal' gradient expects 3 inputs, got " +
                                   std::to_string(ctx.input_types.size()));
  }
  const frame::Result<std::vector<frame::Shape>> output_shapes =
      infer_mse_loss_grad_internal_shape(ctx);
  if (!output_shapes.is_ok()) return output_shapes.status();

  frame::ir::Graph graph("mse_loss_grad_internal_gradient");
  const frame::Result<frame::ir::Value*> pred_in = graph.add_graph_input(ctx.input_types[0]);
  if (!pred_in.is_ok()) return pred_in.status();
  const frame::Result<frame::ir::Value*> target_in = graph.add_graph_input(ctx.input_types[1]);
  if (!target_in.is_ok()) return target_in.status();
  const frame::Result<frame::ir::Value*> gy_in = graph.add_graph_input(ctx.input_types[2]);
  if (!gy_in.is_ok()) return gy_in.status();
  frame::ir::TensorType gpred_type = ctx.input_types[0];
  gpred_type.shape = output_shapes.value()[0];
  const frame::Result<frame::ir::Value*> gpred_in = graph.add_graph_input(gpred_type);
  if (!gpred_in.is_ok()) return gpred_in.status();
  const frame::Result<frame::ir::Value*> ggpred_in = graph.add_graph_input(gpred_type);
  if (!ggpred_in.is_ok()) return ggpred_in.status();

  frame::ir::Value* gy_scalar = gy_in.value();
  if (ctx.input_types[2].shape.rank() != 0) {
    const frame::ops::AttrMap scalar_attrs{{"target_shape", frame::Shape()}};
    const frame::Result<frame::ir::Node*> scalar_node = frame::ops::create_node_with_inferred_types(
        graph, "reshape", {gy_in.value()}, scalar_attrs);
    if (!scalar_node.is_ok()) return scalar_node.status();
    gy_scalar = scalar_node.value()->output(0);
  }

  const frame::ops::AttrMap expand_attrs{{"input_shape", ctx.input_types[0].shape},
                                         {"axes", std::vector<int64_t>{}}};
  const frame::Result<frame::ir::Node*> expanded_gy_node =
      frame::ops::create_node_with_inferred_types(graph, "sum_grad_internal", {gy_scalar},
                                                  expand_attrs);
  if (!expanded_gy_node.is_ok()) return expanded_gy_node.status();

  const double scale = 2.0 / static_cast<double>(ctx.input_types[0].shape.numel());
  const frame::Result<frame::ir::Node*> scale_node = make_constant_splat(
      graph, ctx.input_types[0].shape, ctx.input_types[0].dtype, ctx.input_types[0].device, scale);
  if (!scale_node.is_ok()) return scale_node.status();
  const frame::Result<frame::ir::Node*> neg_one_node = make_constant_splat(
      graph, ctx.input_types[0].shape, ctx.input_types[0].dtype, ctx.input_types[0].device, -1.0);
  if (!neg_one_node.is_ok()) return neg_one_node.status();

  const frame::Result<frame::ir::Node*> scaled_gy_node =
      frame::ops::create_node_with_inferred_types(
          graph, "mul", {scale_node.value()->output(0), expanded_gy_node.value()->output(0)});
  if (!scaled_gy_node.is_ok()) return scaled_gy_node.status();
  const frame::Result<frame::ir::Node*> g_pred_node = frame::ops::create_node_with_inferred_types(
      graph, "mul", {scaled_gy_node.value()->output(0), ggpred_in.value()});
  if (!g_pred_node.is_ok()) return g_pred_node.status();
  const frame::Result<frame::ir::Node*> g_target_node = frame::ops::create_node_with_inferred_types(
      graph, "mul", {neg_one_node.value()->output(0), g_pred_node.value()->output(0)});
  if (!g_target_node.is_ok()) return g_target_node.status();

  const frame::Result<frame::ir::Node*> neg_target_node =
      frame::ops::create_node_with_inferred_types(
          graph, "mul", {neg_one_node.value()->output(0), target_in.value()});
  if (!neg_target_node.is_ok()) return neg_target_node.status();
  const frame::Result<frame::ir::Node*> difference_node =
      frame::ops::create_node_with_inferred_types(
          graph, "add", {pred_in.value(), neg_target_node.value()->output(0)});
  if (!difference_node.is_ok()) return difference_node.status();
  const frame::Result<frame::ir::Node*> scaled_difference_node =
      frame::ops::create_node_with_inferred_types(
          graph, "mul", {scale_node.value()->output(0), difference_node.value()->output(0)});
  if (!scaled_difference_node.is_ok()) return scaled_difference_node.status();
  const frame::Result<frame::ir::Node*> weighted_node = frame::ops::create_node_with_inferred_types(
      graph, "mul", {scaled_difference_node.value()->output(0), ggpred_in.value()});
  if (!weighted_node.is_ok()) return weighted_node.status();
  const frame::ops::AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  const frame::Result<frame::ir::Node*> g_gy_scalar_node =
      frame::ops::create_node_with_inferred_types(graph, "sum", {weighted_node.value()->output(0)},
                                                  sum_attrs);
  if (!g_gy_scalar_node.is_ok()) return g_gy_scalar_node.status();

  frame::ir::Value* g_gy = g_gy_scalar_node.value()->output(0);
  if (ctx.input_types[2].shape.rank() != 0) {
    const frame::ops::AttrMap gy_shape_attrs{{"target_shape", ctx.input_types[2].shape}};
    const frame::Result<frame::ir::Node*> g_gy_node =
        frame::ops::create_node_with_inferred_types(graph, "reshape", {g_gy}, gy_shape_attrs);
    if (!g_gy_node.is_ok()) return g_gy_node.status();
    g_gy = g_gy_node.value()->output(0);
  }

  const frame::Status mark_g_pred = graph.mark_output(g_pred_node.value(), 0);
  if (!mark_g_pred.is_ok()) return mark_g_pred;
  const frame::Status mark_g_target = graph.mark_output(g_target_node.value(), 0);
  if (!mark_g_target.is_ok()) return mark_g_target;
  const frame::Status mark_g_gy = graph.mark_output(g_gy);
  if (!mark_g_gy.is_ok()) return mark_g_gy;
  return graph;
}

}  // namespace

// mse_loss 是面向用户算子(ARCH-064),已随注册同步完成 Python 绑定
// (python/src/bind_ops.cpp::mse_loss,PY-021 自 M12 起全量执法)。
FRAME_REGISTER_OP("mse_loss")
    .input("pred", "predicted tensor")
    .input("target", "target tensor")
    .output("out", "mean squared error mean((pred-target)^2), scalar (rank-0)")
    .shape_infer(&infer_mse_loss_shape)
    .gradient(&mse_loss_gradient);

// mse_loss_grad_internal(M17,mse_loss 的梯度 internal 算子):不面向用户
// (_internal 后缀,PY-021 天然豁免),仅供 mse_loss_gradient 内联使用;cpu
// kernel 见 src/backends/cpu/kernels/loss.cpp。
FRAME_REGISTER_OP("mse_loss_grad_internal")
    .input("pred", "mse_loss's first input")
    .input("target", "mse_loss's second input")
    .input("gy", "upstream gradient of mse_loss's scalar output")
    .output("gpred", "gradient w.r.t. pred: 2*(pred-target)/N*gy")
    .shape_infer(&infer_mse_loss_grad_internal_shape)
    .gradient(&mse_loss_grad_internal_gradient);
