// 内置二维/一维卷积算子 schema 注册桩(M21,批3 T4):conv2d(公开)+
// conv2d_grad_input_internal/conv2d_grad_filter_internal(内部,梯度专用)+
// conv1d(公开,经 decomposition 复用 conv2d)。设计依据:
// docs/plan/2026-07-18-batch3-m21-conv.md 第1节(design-reviewer 两轮
// APPROVE)。bias 经 `variadic_input("bias", /*min_count=*/0)` 表达可选第三
// 输入(裁决点①);shape 推断与 GradientFn 依 ctx.input_types.size()(2/3 两态)
// 分支,有 bias 时梯度追加 gbias 位(ARCH-063)。conv1d 的梯度不新开专属
// internal 算子,直接用 reshape + conv2d 的两个梯度 internal 算子组合(§1.2)。
// cpu kernel 见 src/backends/cpu/kernels/conv.cpp。

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/graph_builder.h>
#include <frame/ops/op_registry.h>

#include "schema_math.h"

namespace {

using frame::ErrorCode;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::ops::NodeContext;

// floor 除法:同目录共享工具(铁律 5 收敛,见 schema_math.h 头注释)。
using frame::ops::schemas::floor_div_positive_denominator;

// ---------------------------------------------------------------------------
// conv2d 及其两个梯度 internal 算子共用的二维卷积几何推导。
// ---------------------------------------------------------------------------

// 二维卷积的几何参数(shape 推断中间结果),供 conv2d 本身及
// conv2d_grad_input_internal/conv2d_grad_filter_internal 三处共用(REUSE-002,
// 三个调用点)。
struct Conv2dGeometry {
  int64_t n = 0;
  int64_t cin = 0;
  int64_t h = 0;
  int64_t w = 0;
  int64_t cout = 0;
  int64_t cin_per_group = 0;
  int64_t kh = 0;
  int64_t kw = 0;
  int64_t groups = 0;
  int64_t stride_h = 0;
  int64_t stride_w = 0;
  int64_t pad_h = 0;
  int64_t pad_w = 0;
  int64_t out_h = 0;
  int64_t out_w = 0;
};

// 校验 x_shape([N,Cin,H,W])/w_shape([Cout,Cin/groups,KH,KW])/stride/padding/
// groups 的自洽性并推出输出空间维(floor 口径 (H+2p-KH)/s+1)。op_name 仅用于
// 拼错误消息;三个调用点(conv2d 本身与两个梯度 internal 算子)传入不同的
// x_shape/w_shape 来源(前者直接取自输入类型,后两者之一取自 kShape 属性),
// 校验与推导逻辑本身完全相同,故收敛为一份实现(REUSE-002)。
// x_shape/w_shape 均为 [N,C,H,W] 形态、stride/padding 均为 2 元 [h,w] 形态,
// 语义上确有相邻同型形参,但均为本算子族(NCHW 卷积)固定契约形态,调用点
// 均以具名局部变量传入,误置换会在 shape 推断/数值校验阶段立即报错并非静默
// 产出错误结果(同 src/nn/layers.cpp::Sequential 的 inputs/params 先例)。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Result<Conv2dGeometry> compute_conv2d_geometry(std::string_view op_name, const Shape& x_shape,
                                               const Shape& w_shape,
                                               const std::vector<int64_t>& stride,
                                               const std::vector<int64_t>& padding,
                                               int64_t groups) {
  if (x_shape.rank() != 4) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) +
                            "' requires x to be rank-4 [N, Cin, H, W], got rank " +
                            std::to_string(x_shape.rank()));
  }
  if (w_shape.rank() != 4) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) +
                            "' requires w to be rank-4 [Cout, Cin/groups, KH, KW], got rank " +
                            std::to_string(w_shape.rank()));
  }
  if (stride.size() != 2) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) +
                            "' attribute 'stride' must have 2 elements, got " +
                            std::to_string(stride.size()));
  }
  if (padding.size() != 2) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) +
                            "' attribute 'padding' must have 2 elements, got " +
                            std::to_string(padding.size()));
  }
  if (groups < 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) +
                            "' attribute 'groups' must be positive, got " + std::to_string(groups));
  }

  Conv2dGeometry geo;
  geo.n = x_shape.dim(0);
  geo.cin = x_shape.dim(1);
  geo.h = x_shape.dim(2);
  geo.w = x_shape.dim(3);
  geo.cout = w_shape.dim(0);
  geo.cin_per_group = w_shape.dim(1);
  geo.kh = w_shape.dim(2);
  geo.kw = w_shape.dim(3);
  geo.groups = groups;
  geo.stride_h = stride[0];
  geo.stride_w = stride[1];
  geo.pad_h = padding[0];
  geo.pad_w = padding[1];

  if (geo.n <= 0 || geo.cin <= 0 || geo.h <= 0 || geo.w <= 0) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) +
                            "' requires x dims [N, Cin, H, W] to be positive, got " +
                            x_shape.to_string());
  }
  if (geo.cout <= 0 || geo.cin_per_group <= 0 || geo.kh <= 0 || geo.kw <= 0) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) +
                            "' requires w dims [Cout, Cin/groups, KH, KW] to be positive, got " +
                            w_shape.to_string());
  }
  if (geo.stride_h < 1 || geo.stride_w < 1) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) + "' attribute 'stride' entries must be >= 1, got [" +
            std::to_string(geo.stride_h) + ", " + std::to_string(geo.stride_w) + "]");
  }
  if (geo.pad_h < 0 || geo.pad_w < 0) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) +
                            "' attribute 'padding' entries must be >= 0, got [" +
                            std::to_string(geo.pad_h) + ", " + std::to_string(geo.pad_w) + "]");
  }
  if (geo.cin % geo.groups != 0) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) + "' Cin=" + std::to_string(geo.cin) +
                            " is not divisible by groups=" + std::to_string(geo.groups));
  }
  if (geo.cout % geo.groups != 0) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) + "' Cout=" + std::to_string(geo.cout) +
                            " is not divisible by groups=" + std::to_string(geo.groups));
  }
  const int64_t expected_cin_per_group = geo.cin / geo.groups;
  if (geo.cin_per_group != expected_cin_per_group) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) + "' w's second dimension (Cin/groups) is " +
                            std::to_string(geo.cin_per_group) + ", expected " +
                            std::to_string(expected_cin_per_group) +
                            " (Cin=" + std::to_string(geo.cin) +
                            ", groups=" + std::to_string(geo.groups) + ")");
  }

  geo.out_h = floor_div_positive_denominator(geo.h + 2 * geo.pad_h - geo.kh, geo.stride_h) + 1;
  geo.out_w = floor_div_positive_denominator(geo.w + 2 * geo.pad_w - geo.kw, geo.stride_w) + 1;
  if (geo.out_h < 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) + "' produces non-positive output height " +
                            std::to_string(geo.out_h) + " (H=" + std::to_string(geo.h) +
                            ", padding_h=" + std::to_string(geo.pad_h) +
                            ", KH=" + std::to_string(geo.kh) +
                            ", stride_h=" + std::to_string(geo.stride_h) + ")");
  }
  if (geo.out_w < 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) + "' produces non-positive output width " +
                            std::to_string(geo.out_w) + " (W=" + std::to_string(geo.w) +
                            ", padding_w=" + std::to_string(geo.pad_w) +
                            ", KW=" + std::to_string(geo.kw) +
                            ", stride_w=" + std::to_string(geo.stride_w) + ")");
  }

  return geo;
}

// conv2d 的 shape 推断:x[N,Cin,H,W] + w[Cout,Cin/g,KH,KW] + 可选 bias[Cout]
// (variadic 组元素数 0/1,>1 报错)。attrs:stride/padding(各 kInt64Array 2元)
// + groups(kInt64)。输出 [N,Cout,out_h,out_w]。
Result<std::vector<Shape>> infer_conv2d_shape(const NodeContext& ctx) {
  const size_t total = ctx.input_types.size();
  if (total < 2) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "op 'conv2d' expects at least 2 inputs (x, w), got " + std::to_string(total));
  }
  const size_t bias_count = total - 2;
  if (bias_count > 1) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "op 'conv2d' bias group expects at most 1 tensor, got " + std::to_string(bias_count));
  }
  const frame::ir::TensorType& x = ctx.input_types[0];
  const frame::ir::TensorType& w = ctx.input_types[1];
  if (!(x.dtype == w.dtype)) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d' requires x and w of the same dtype, got '" +
                            std::string(x.dtype.name()) + "' and '" + std::string(w.dtype.name()) +
                            "'");
  }

  const std::vector<int64_t>* stride = ctx.attr<std::vector<int64_t>>("stride");
  if (stride == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d' is missing required attribute 'stride' (int64 array)");
  }
  const std::vector<int64_t>* padding = ctx.attr<std::vector<int64_t>>("padding");
  if (padding == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d' is missing required attribute 'padding' (int64 array)");
  }
  const int64_t* groups = ctx.attr<int64_t>("groups");
  if (groups == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d' is missing required attribute 'groups'");
  }

  const Result<Conv2dGeometry> geo_result =
      compute_conv2d_geometry("conv2d", x.shape, w.shape, *stride, *padding, *groups);
  if (!geo_result.is_ok()) return geo_result.status();
  const Conv2dGeometry& geo = geo_result.value();

  if (bias_count == 1) {
    const frame::ir::TensorType& bias = ctx.input_types[2];
    if (!(bias.dtype == x.dtype)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "op 'conv2d' requires bias and x of the same dtype, got '" +
                              std::string(bias.dtype.name()) + "' and '" +
                              std::string(x.dtype.name()) + "'");
    }
    if (bias.shape.rank() != 1) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "op 'conv2d' requires bias to be rank-1, got rank " + std::to_string(bias.shape.rank()));
    }
    if (bias.shape.dim(0) != geo.cout) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "op 'conv2d' requires bias size to equal Cout=" + std::to_string(geo.cout) + ", got " +
              std::to_string(bias.shape.dim(0)));
    }
  }

  return std::vector<Shape>{Shape({geo.n, geo.cout, geo.out_h, geo.out_w})};
}

// conv2d_grad_input_internal(dy,w) 的 shape 推断(M21,conv2d 的梯度 internal
// 算子之一,BackwardData):attrs=input_shape(kShape,原 conv2d 的 x 静态
// shape)+stride/padding/groups(与原 conv2d 节点属性同值转发)。输出=
// input_shape;另据 input_shape/w/attrs 重算几何并核对 dy 实际 shape 与之一致
// (sum_grad_internal 先例:输出形状不能由输入反推,须携带 input_shape)。
Result<std::vector<Shape>> infer_conv2d_grad_input_internal_shape(const NodeContext& ctx) {
  if (ctx.input_types.size() != 2) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d_grad_input_internal' expects 2 inputs, got " +
                            std::to_string(ctx.input_types.size()));
  }
  const frame::ir::TensorType& dy = ctx.input_types[0];
  const frame::ir::TensorType& w = ctx.input_types[1];
  if (!(dy.dtype == w.dtype)) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d_grad_input_internal' requires dy and w of the same dtype, "
                        "got '" +
                            std::string(dy.dtype.name()) + "' and '" + std::string(w.dtype.name()) +
                            "'");
  }

  const Shape* input_shape = ctx.attr<Shape>("input_shape");
  if (input_shape == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d_grad_input_internal' is missing required attribute "
                        "'input_shape'");
  }
  if (input_shape->has_dynamic_dim()) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d_grad_input_internal' attribute 'input_shape' " +
                            input_shape->to_string() +
                            " has a dynamic dimension, static shape required (ARCH-013)");
  }
  const std::vector<int64_t>* stride = ctx.attr<std::vector<int64_t>>("stride");
  if (stride == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d_grad_input_internal' is missing required attribute 'stride' "
                        "(int64 array)");
  }
  const std::vector<int64_t>* padding = ctx.attr<std::vector<int64_t>>("padding");
  if (padding == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d_grad_input_internal' is missing required attribute "
                        "'padding' (int64 array)");
  }
  const int64_t* groups = ctx.attr<int64_t>("groups");
  if (groups == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d_grad_input_internal' is missing required attribute "
                        "'groups'");
  }

  const Result<Conv2dGeometry> geo_result = compute_conv2d_geometry(
      "conv2d_grad_input_internal", *input_shape, w.shape, *stride, *padding, *groups);
  if (!geo_result.is_ok()) return geo_result.status();
  const Conv2dGeometry& geo = geo_result.value();

  const Shape expected_dy_shape({geo.n, geo.cout, geo.out_h, geo.out_w});
  if (!(dy.shape == expected_dy_shape)) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d_grad_input_internal' requires dy shape to be consistent "
                        "with input_shape/w/stride/padding/groups, got " +
                            dy.shape.to_string() + ", expected " + expected_dy_shape.to_string());
  }

  return std::vector<Shape>{*input_shape};
}

// conv2d_grad_filter_internal(x,dy) 的 shape 推断(M21,conv2d 的梯度 internal
// 算子之二,BackwardFilter):attrs=filter_shape(kShape,原 conv2d 的 w 静态
// shape)+stride/padding/groups。输出=filter_shape。
Result<std::vector<Shape>> infer_conv2d_grad_filter_internal_shape(const NodeContext& ctx) {
  if (ctx.input_types.size() != 2) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d_grad_filter_internal' expects 2 inputs, got " +
                            std::to_string(ctx.input_types.size()));
  }
  const frame::ir::TensorType& x = ctx.input_types[0];
  const frame::ir::TensorType& dy = ctx.input_types[1];
  if (!(x.dtype == dy.dtype)) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d_grad_filter_internal' requires x and dy of the same dtype, "
                        "got '" +
                            std::string(x.dtype.name()) + "' and '" + std::string(dy.dtype.name()) +
                            "'");
  }

  const Shape* filter_shape = ctx.attr<Shape>("filter_shape");
  if (filter_shape == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d_grad_filter_internal' is missing required attribute "
                        "'filter_shape'");
  }
  if (filter_shape->has_dynamic_dim()) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d_grad_filter_internal' attribute 'filter_shape' " +
                            filter_shape->to_string() +
                            " has a dynamic dimension, static shape required (ARCH-013)");
  }
  const std::vector<int64_t>* stride = ctx.attr<std::vector<int64_t>>("stride");
  if (stride == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d_grad_filter_internal' is missing required attribute "
                        "'stride' (int64 array)");
  }
  const std::vector<int64_t>* padding = ctx.attr<std::vector<int64_t>>("padding");
  if (padding == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d_grad_filter_internal' is missing required attribute "
                        "'padding' (int64 array)");
  }
  const int64_t* groups = ctx.attr<int64_t>("groups");
  if (groups == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d_grad_filter_internal' is missing required attribute "
                        "'groups'");
  }

  const Result<Conv2dGeometry> geo_result = compute_conv2d_geometry(
      "conv2d_grad_filter_internal", x.shape, *filter_shape, *stride, *padding, *groups);
  if (!geo_result.is_ok()) return geo_result.status();
  const Conv2dGeometry& geo = geo_result.value();

  const Shape expected_dy_shape({geo.n, geo.cout, geo.out_h, geo.out_w});
  if (!(dy.shape == expected_dy_shape)) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d_grad_filter_internal' requires dy shape to be consistent "
                        "with x/filter_shape/stride/padding/groups, got " +
                            dy.shape.to_string() + ", expected " + expected_dy_shape.to_string());
  }

  return std::vector<Shape>{*filter_shape};
}

// conv2d(x,w[,bias]) 的梯度(§1.2 表):gx=conv2d_grad_input_internal(dy,w),
// gw=conv2d_grad_filter_internal(x,dy),有 bias 时 gbias=sum(dy,axes=[0,2,3])。
Result<frame::ir::Graph> conv2d_gradient(const NodeContext& ctx) {
  const size_t total = ctx.input_types.size();
  if (total < 2) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "op 'conv2d' gradient expects at least 2 inputs (x, w), got " + std::to_string(total));
  }
  const size_t bias_count = total - 2;
  if (bias_count > 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d' gradient bias group expects at most 1 tensor, got " +
                            std::to_string(bias_count));
  }

  frame::ir::Graph graph("conv2d_gradient");

  const Result<frame::ir::Value*> x_in = graph.add_graph_input(ctx.input_types[0]);
  if (!x_in.is_ok()) return x_in.status();
  const Result<frame::ir::Value*> w_in = graph.add_graph_input(ctx.input_types[1]);
  if (!w_in.is_ok()) return w_in.status();
  frame::ir::Value* bias_in = nullptr;
  if (bias_count == 1) {
    const Result<frame::ir::Value*> bias_result = graph.add_graph_input(ctx.input_types[2]);
    if (!bias_result.is_ok()) return bias_result.status();
    bias_in = bias_result.value();
  }

  const Result<std::vector<Shape>> y_shapes = infer_conv2d_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const Result<frame::ir::Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();

  const std::vector<int64_t>& stride = *ctx.attr<std::vector<int64_t>>("stride");
  const std::vector<int64_t>& padding = *ctx.attr<std::vector<int64_t>>("padding");
  const int64_t groups = *ctx.attr<int64_t>("groups");

  const frame::ops::AttrMap gx_attrs{
      {"input_shape", ctx.input_types[0].shape},
      {"stride", stride},
      {"padding", padding},
      {"groups", groups},
  };
  const Result<frame::ir::Node*> gx_node = frame::ops::create_node_with_inferred_types(
      graph, "conv2d_grad_input_internal", {gy_in.value(), w_in.value()}, gx_attrs);
  if (!gx_node.is_ok()) return gx_node.status();

  const frame::ops::AttrMap gw_attrs{
      {"filter_shape", ctx.input_types[1].shape},
      {"stride", stride},
      {"padding", padding},
      {"groups", groups},
  };
  const Result<frame::ir::Node*> gw_node = frame::ops::create_node_with_inferred_types(
      graph, "conv2d_grad_filter_internal", {x_in.value(), gy_in.value()}, gw_attrs);
  if (!gw_node.is_ok()) return gw_node.status();

  const Status mark_gx = graph.mark_output(gx_node.value(), 0);
  if (!mark_gx.is_ok()) return mark_gx;
  const Status mark_gw = graph.mark_output(gw_node.value(), 0);
  if (!mark_gw.is_ok()) return mark_gw;

  if (bias_count == 1) {
    (void)bias_in;  // bias 自身不参与前向反向计算路径,仅占位声明(ARCH-063)
    const frame::ops::AttrMap gbias_attrs{
        {"axes", std::vector<int64_t>{0, 2, 3}},
    };
    const Result<frame::ir::Node*> gbias_node =
        frame::ops::create_node_with_inferred_types(graph, "sum", {gy_in.value()}, gbias_attrs);
    if (!gbias_node.is_ok()) return gbias_node.status();
    const Status mark_gbias = graph.mark_output(gbias_node.value(), 0);
    if (!mark_gbias.is_ok()) return mark_gbias;
  }

  return graph;
}

// conv2d_grad_input_internal(dy,w) 自身的梯度(§1.2 表,R11 封闭性):
// wrt dy → conv2d(g,w,无bias);wrt w → conv2d_grad_filter_internal(g,dy)。
Result<frame::ir::Graph> conv2d_grad_input_internal_gradient(const NodeContext& ctx) {
  if (ctx.input_types.size() != 2) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d_grad_input_internal' gradient expects 2 inputs, got " +
                            std::to_string(ctx.input_types.size()));
  }
  frame::ir::Graph graph("conv2d_grad_input_internal_gradient");

  const Result<frame::ir::Value*> dy_in = graph.add_graph_input(ctx.input_types[0]);
  if (!dy_in.is_ok()) return dy_in.status();
  const Result<frame::ir::Value*> w_in = graph.add_graph_input(ctx.input_types[1]);
  if (!w_in.is_ok()) return w_in.status();

  const Result<std::vector<Shape>> y_shapes = infer_conv2d_grad_input_internal_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const Result<frame::ir::Value*> g_in = graph.add_graph_input(y_type);
  if (!g_in.is_ok()) return g_in.status();

  const std::vector<int64_t>& stride = *ctx.attr<std::vector<int64_t>>("stride");
  const std::vector<int64_t>& padding = *ctx.attr<std::vector<int64_t>>("padding");
  const int64_t groups = *ctx.attr<int64_t>("groups");

  const frame::ops::AttrMap conv2d_attrs{
      {"stride", stride},
      {"padding", padding},
      {"groups", groups},
  };
  const Result<frame::ir::Node*> wrt_dy_node = frame::ops::create_node_with_inferred_types(
      graph, "conv2d", {g_in.value(), w_in.value()}, conv2d_attrs);
  if (!wrt_dy_node.is_ok()) return wrt_dy_node.status();

  const frame::ops::AttrMap gfi_attrs{
      {"filter_shape", ctx.input_types[1].shape},
      {"stride", stride},
      {"padding", padding},
      {"groups", groups},
  };
  const Result<frame::ir::Node*> wrt_w_node = frame::ops::create_node_with_inferred_types(
      graph, "conv2d_grad_filter_internal", {g_in.value(), dy_in.value()}, gfi_attrs);
  if (!wrt_w_node.is_ok()) return wrt_w_node.status();

  const Status mark_wrt_dy = graph.mark_output(wrt_dy_node.value(), 0);
  if (!mark_wrt_dy.is_ok()) return mark_wrt_dy;
  const Status mark_wrt_w = graph.mark_output(wrt_w_node.value(), 0);
  if (!mark_wrt_w.is_ok()) return mark_wrt_w;

  return graph;
}

// conv2d_grad_filter_internal(x,dy) 自身的梯度(§1.2 表,R11 封闭性):
// wrt x → conv2d_grad_input_internal(dy,g)(dy 留 y 实参位、g 入滤波器实参位,
// 计划 1.2 表附推导);wrt dy → conv2d(x,g,无bias)。
Result<frame::ir::Graph> conv2d_grad_filter_internal_gradient(const NodeContext& ctx) {
  if (ctx.input_types.size() != 2) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv2d_grad_filter_internal' gradient expects 2 inputs, got " +
                            std::to_string(ctx.input_types.size()));
  }
  frame::ir::Graph graph("conv2d_grad_filter_internal_gradient");

  const Result<frame::ir::Value*> x_in = graph.add_graph_input(ctx.input_types[0]);
  if (!x_in.is_ok()) return x_in.status();
  const Result<frame::ir::Value*> dy_in = graph.add_graph_input(ctx.input_types[1]);
  if (!dy_in.is_ok()) return dy_in.status();

  const Result<std::vector<Shape>> y_shapes = infer_conv2d_grad_filter_internal_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const Result<frame::ir::Value*> g_in = graph.add_graph_input(y_type);
  if (!g_in.is_ok()) return g_in.status();

  const std::vector<int64_t>& stride = *ctx.attr<std::vector<int64_t>>("stride");
  const std::vector<int64_t>& padding = *ctx.attr<std::vector<int64_t>>("padding");
  const int64_t groups = *ctx.attr<int64_t>("groups");

  const frame::ops::AttrMap gii_attrs{
      {"input_shape", ctx.input_types[0].shape},
      {"stride", stride},
      {"padding", padding},
      {"groups", groups},
  };
  const Result<frame::ir::Node*> wrt_x_node = frame::ops::create_node_with_inferred_types(
      graph, "conv2d_grad_input_internal", {dy_in.value(), g_in.value()}, gii_attrs);
  if (!wrt_x_node.is_ok()) return wrt_x_node.status();

  const frame::ops::AttrMap conv2d_attrs{
      {"stride", stride},
      {"padding", padding},
      {"groups", groups},
  };
  const Result<frame::ir::Node*> wrt_dy_node = frame::ops::create_node_with_inferred_types(
      graph, "conv2d", {x_in.value(), g_in.value()}, conv2d_attrs);
  if (!wrt_dy_node.is_ok()) return wrt_dy_node.status();

  const Status mark_wrt_x = graph.mark_output(wrt_x_node.value(), 0);
  if (!mark_wrt_x.is_ok()) return mark_wrt_x;
  const Status mark_wrt_dy = graph.mark_output(wrt_dy_node.value(), 0);
  if (!mark_wrt_dy.is_ok()) return mark_wrt_dy;

  return graph;
}

// ---------------------------------------------------------------------------
// conv1d:一维卷积几何 + shape 推断 + decomposition(reshape→conv2d→reshape)
// + 梯度(reshape + conv2d 两个梯度 internal 算子 + reshape 组合,无 conv1d
// 专属 internal 算子,§1.2)。
// ---------------------------------------------------------------------------

// 一维卷积的几何参数,结构与 Conv2dGeometry 相似但秩/属性形态不同(rank-3
// 输入、标量 stride/padding,而非 rank-4/数组),各自独立实现(理由同
// src/ops/schemas/matmul.cpp 头注释:强行合并需要额外的维度参数化,增加的
// 间接层不足以抵销收益)。
struct Conv1dGeometry {
  int64_t n = 0;
  int64_t cin = 0;
  int64_t l = 0;
  int64_t cout = 0;
  int64_t cin_per_group = 0;
  int64_t k = 0;
  int64_t groups = 0;
  int64_t stride = 0;
  int64_t padding = 0;
  int64_t out_l = 0;
};

// 校验 x_shape([N,Cin,L])/w_shape([Cout,Cin/groups,K])/stride/padding/groups
// 的自洽性并推出输出长度(floor 口径),供 conv1d 的 shape 推断使用。
Result<Conv1dGeometry> compute_conv1d_geometry(std::string_view op_name, const Shape& x_shape,
                                               const Shape& w_shape, int64_t stride,
                                               int64_t padding, int64_t groups) {
  if (x_shape.rank() != 3) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) +
                            "' requires x to be rank-3 [N, Cin, L], got rank " +
                            std::to_string(x_shape.rank()));
  }
  if (w_shape.rank() != 3) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) +
                            "' requires w to be rank-3 [Cout, Cin/groups, K], got rank " +
                            std::to_string(w_shape.rank()));
  }
  if (groups < 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) +
                            "' attribute 'groups' must be positive, got " + std::to_string(groups));
  }
  if (stride < 1) {
    return Status::make(ErrorCode::kInvalidArgument, "op '" + std::string(op_name) +
                                                         "' attribute 'stride' must be >= 1, got " +
                                                         std::to_string(stride));
  }
  if (padding < 0) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) + "' attribute 'padding' must be >= 0, got " +
                            std::to_string(padding));
  }

  Conv1dGeometry geo;
  geo.n = x_shape.dim(0);
  geo.cin = x_shape.dim(1);
  geo.l = x_shape.dim(2);
  geo.cout = w_shape.dim(0);
  geo.cin_per_group = w_shape.dim(1);
  geo.k = w_shape.dim(2);
  geo.groups = groups;
  geo.stride = stride;
  geo.padding = padding;

  if (geo.n <= 0 || geo.cin <= 0 || geo.l <= 0) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) +
                            "' requires x dims [N, Cin, L] to be positive, got " +
                            x_shape.to_string());
  }
  if (geo.cout <= 0 || geo.cin_per_group <= 0 || geo.k <= 0) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) +
                            "' requires w dims [Cout, Cin/groups, K] to be positive, got " +
                            w_shape.to_string());
  }
  if (geo.cin % geo.groups != 0) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) + "' Cin=" + std::to_string(geo.cin) +
                            " is not divisible by groups=" + std::to_string(geo.groups));
  }
  if (geo.cout % geo.groups != 0) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) + "' Cout=" + std::to_string(geo.cout) +
                            " is not divisible by groups=" + std::to_string(geo.groups));
  }
  const int64_t expected_cin_per_group = geo.cin / geo.groups;
  if (geo.cin_per_group != expected_cin_per_group) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) + "' w's second dimension (Cin/groups) is " +
                            std::to_string(geo.cin_per_group) + ", expected " +
                            std::to_string(expected_cin_per_group) +
                            " (Cin=" + std::to_string(geo.cin) +
                            ", groups=" + std::to_string(geo.groups) + ")");
  }

  geo.out_l = floor_div_positive_denominator(geo.l + 2 * geo.padding - geo.k, geo.stride) + 1;
  if (geo.out_l < 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) + "' produces non-positive output length " +
                            std::to_string(geo.out_l) + " (L=" + std::to_string(geo.l) +
                            ", padding=" + std::to_string(geo.padding) + ", K=" +
                            std::to_string(geo.k) + ", stride=" + std::to_string(geo.stride) + ")");
  }

  return geo;
}

// conv1d 的 shape 推断:x[N,Cin,L] + w[Cout,Cin/g,K] + 可选 bias[Cout];attrs
// stride/padding/groups 均 kInt64(标量)。输出 [N,Cout,out_l]。
Result<std::vector<Shape>> infer_conv1d_shape(const NodeContext& ctx) {
  const size_t total = ctx.input_types.size();
  if (total < 2) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "op 'conv1d' expects at least 2 inputs (x, w), got " + std::to_string(total));
  }
  const size_t bias_count = total - 2;
  if (bias_count > 1) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "op 'conv1d' bias group expects at most 1 tensor, got " + std::to_string(bias_count));
  }
  const frame::ir::TensorType& x = ctx.input_types[0];
  const frame::ir::TensorType& w = ctx.input_types[1];
  if (!(x.dtype == w.dtype)) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv1d' requires x and w of the same dtype, got '" +
                            std::string(x.dtype.name()) + "' and '" + std::string(w.dtype.name()) +
                            "'");
  }

  const int64_t* stride = ctx.attr<int64_t>("stride");
  if (stride == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv1d' is missing required attribute 'stride'");
  }
  const int64_t* padding = ctx.attr<int64_t>("padding");
  if (padding == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv1d' is missing required attribute 'padding'");
  }
  const int64_t* groups = ctx.attr<int64_t>("groups");
  if (groups == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv1d' is missing required attribute 'groups'");
  }

  const Result<Conv1dGeometry> geo_result =
      compute_conv1d_geometry("conv1d", x.shape, w.shape, *stride, *padding, *groups);
  if (!geo_result.is_ok()) return geo_result.status();
  const Conv1dGeometry& geo = geo_result.value();

  if (bias_count == 1) {
    const frame::ir::TensorType& bias = ctx.input_types[2];
    if (!(bias.dtype == x.dtype)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "op 'conv1d' requires bias and x of the same dtype, got '" +
                              std::string(bias.dtype.name()) + "' and '" +
                              std::string(x.dtype.name()) + "'");
    }
    if (bias.shape.rank() != 1) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "op 'conv1d' requires bias to be rank-1, got rank " + std::to_string(bias.shape.rank()));
    }
    if (bias.shape.dim(0) != geo.cout) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "op 'conv1d' requires bias size to equal Cout=" + std::to_string(geo.cout) + ", got " +
              std::to_string(bias.shape.dim(0)));
    }
  }

  return std::vector<Shape>{Shape({geo.n, geo.cout, geo.out_l})};
}

// conv1d 的 decomposition(裁决点②):reshape x[N,Cin,L]→[N,Cin,1,L] +
// reshape w[Cout,Cin/g,K]→[Cout,Cin/g,1,K] → conv2d(stride=[1,s],
// padding=[0,p],groups=g) → reshape 回 [N,Cout,out_l];bias 原样透传给
// conv2d(裁决点②,不写 conv1d 专属 cuDNN kernel,CPU 侧仍有直循环参考
// kernel,ARCH-041)。
Result<frame::ir::Graph> conv1d_decompose(const NodeContext& ctx) {
  const size_t total = ctx.input_types.size();
  if (total < 2) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "op 'conv1d' decomposition expects at least 2 inputs (x, w), got " + std::to_string(total));
  }
  const size_t bias_count = total - 2;
  if (bias_count > 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv1d' decomposition bias group expects at most 1 tensor, got " +
                            std::to_string(bias_count));
  }

  frame::ir::Graph graph("conv1d_decompose");

  const Result<frame::ir::Value*> x_in = graph.add_graph_input(ctx.input_types[0]);
  if (!x_in.is_ok()) return x_in.status();
  const Result<frame::ir::Value*> w_in = graph.add_graph_input(ctx.input_types[1]);
  if (!w_in.is_ok()) return w_in.status();
  frame::ir::Value* bias_in = nullptr;
  if (bias_count == 1) {
    const Result<frame::ir::Value*> bias_result = graph.add_graph_input(ctx.input_types[2]);
    if (!bias_result.is_ok()) return bias_result.status();
    bias_in = bias_result.value();
  }

  const Shape& x_shape = ctx.input_types[0].shape;
  if (x_shape.rank() != 3) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv1d' decomposition requires x to be rank-3 [N, Cin, L], got "
                        "rank " +
                            std::to_string(x_shape.rank()));
  }
  const Shape& w_shape = ctx.input_types[1].shape;
  if (w_shape.rank() != 3) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv1d' decomposition requires w to be rank-3 [Cout, Cin/groups, "
                        "K], got rank " +
                            std::to_string(w_shape.rank()));
  }
  const Shape x4_shape({x_shape.dim(0), x_shape.dim(1), 1, x_shape.dim(2)});
  const Shape w4_shape({w_shape.dim(0), w_shape.dim(1), 1, w_shape.dim(2)});

  const frame::ops::AttrMap x4_attrs{{"target_shape", x4_shape}};
  const Result<frame::ir::Node*> x4_node =
      frame::ops::create_node_with_inferred_types(graph, "reshape", {x_in.value()}, x4_attrs);
  if (!x4_node.is_ok()) return x4_node.status();

  const frame::ops::AttrMap w4_attrs{{"target_shape", w4_shape}};
  const Result<frame::ir::Node*> w4_node =
      frame::ops::create_node_with_inferred_types(graph, "reshape", {w_in.value()}, w4_attrs);
  if (!w4_node.is_ok()) return w4_node.status();

  const int64_t* stride = ctx.attr<int64_t>("stride");
  if (stride == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv1d' decomposition is missing required attribute 'stride'");
  }
  const int64_t* padding = ctx.attr<int64_t>("padding");
  if (padding == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv1d' decomposition is missing required attribute 'padding'");
  }
  const int64_t* groups = ctx.attr<int64_t>("groups");
  if (groups == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv1d' decomposition is missing required attribute 'groups'");
  }

  std::vector<frame::ir::Value*> conv2d_inputs{x4_node.value()->output(0),
                                               w4_node.value()->output(0)};
  if (bias_in != nullptr) conv2d_inputs.push_back(bias_in);
  const frame::ops::AttrMap conv2d_attrs{
      {"stride", std::vector<int64_t>{1, *stride}},
      {"padding", std::vector<int64_t>{0, *padding}},
      {"groups", *groups},
  };
  const Result<frame::ir::Node*> y4_node =
      frame::ops::create_node_with_inferred_types(graph, "conv2d", conv2d_inputs, conv2d_attrs);
  if (!y4_node.is_ok()) return y4_node.status();

  const Shape& y4_shape = y4_node.value()->output(0)->type().shape;
  const Shape y_shape({y4_shape.dim(0), y4_shape.dim(1), y4_shape.dim(3)});
  const frame::ops::AttrMap y_attrs{{"target_shape", y_shape}};
  const Result<frame::ir::Node*> y_node = frame::ops::create_node_with_inferred_types(
      graph, "reshape", {y4_node.value()->output(0)}, y_attrs);
  if (!y_node.is_ok()) return y_node.status();

  const Status mark_y = graph.mark_output(y_node.value(), 0);
  if (!mark_y.is_ok()) return mark_y;

  return graph;
}

// conv1d(x,w[,bias]) 的梯度(§1.2 末段):微图 = reshape + conv2d 两个梯度
// internal 算子 + reshape(无 conv1d 专属 internal 算子);gbias=sum(dy,
// axes=[0,2])。
Result<frame::ir::Graph> conv1d_gradient(const NodeContext& ctx) {
  const size_t total = ctx.input_types.size();
  if (total < 2) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "op 'conv1d' gradient expects at least 2 inputs (x, w), got " + std::to_string(total));
  }
  const size_t bias_count = total - 2;
  if (bias_count > 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'conv1d' gradient bias group expects at most 1 tensor, got " +
                            std::to_string(bias_count));
  }

  frame::ir::Graph graph("conv1d_gradient");

  const Result<frame::ir::Value*> x_in = graph.add_graph_input(ctx.input_types[0]);
  if (!x_in.is_ok()) return x_in.status();
  const Result<frame::ir::Value*> w_in = graph.add_graph_input(ctx.input_types[1]);
  if (!w_in.is_ok()) return w_in.status();
  if (bias_count == 1) {
    const Result<frame::ir::Value*> bias_result = graph.add_graph_input(ctx.input_types[2]);
    if (!bias_result.is_ok()) return bias_result.status();
  }

  const Result<std::vector<Shape>> y_shapes = infer_conv1d_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const Result<frame::ir::Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();

  const Shape& x_shape = ctx.input_types[0].shape;
  const Shape& w_shape = ctx.input_types[1].shape;
  const Shape& dy_shape = y_type.shape;

  const Shape x4_shape({x_shape.dim(0), x_shape.dim(1), 1, x_shape.dim(2)});
  const Shape w4_shape({w_shape.dim(0), w_shape.dim(1), 1, w_shape.dim(2)});
  const Shape dy4_shape({dy_shape.dim(0), dy_shape.dim(1), 1, dy_shape.dim(2)});

  const frame::ops::AttrMap x4_attrs{{"target_shape", x4_shape}};
  const Result<frame::ir::Node*> x4_node =
      frame::ops::create_node_with_inferred_types(graph, "reshape", {x_in.value()}, x4_attrs);
  if (!x4_node.is_ok()) return x4_node.status();

  const frame::ops::AttrMap w4_attrs{{"target_shape", w4_shape}};
  const Result<frame::ir::Node*> w4_node =
      frame::ops::create_node_with_inferred_types(graph, "reshape", {w_in.value()}, w4_attrs);
  if (!w4_node.is_ok()) return w4_node.status();

  const frame::ops::AttrMap dy4_attrs{{"target_shape", dy4_shape}};
  const Result<frame::ir::Node*> dy4_node =
      frame::ops::create_node_with_inferred_types(graph, "reshape", {gy_in.value()}, dy4_attrs);
  if (!dy4_node.is_ok()) return dy4_node.status();

  const int64_t stride = *ctx.attr<int64_t>("stride");
  const int64_t padding = *ctx.attr<int64_t>("padding");
  const int64_t groups = *ctx.attr<int64_t>("groups");
  const std::vector<int64_t> stride2d{1, stride};
  const std::vector<int64_t> padding2d{0, padding};

  const frame::ops::AttrMap gx4_attrs{
      {"input_shape", x4_shape},
      {"stride", stride2d},
      {"padding", padding2d},
      {"groups", groups},
  };
  const Result<frame::ir::Node*> gx4_node = frame::ops::create_node_with_inferred_types(
      graph, "conv2d_grad_input_internal",
      {dy4_node.value()->output(0), w4_node.value()->output(0)}, gx4_attrs);
  if (!gx4_node.is_ok()) return gx4_node.status();
  const frame::ops::AttrMap gx_attrs{{"target_shape", x_shape}};
  const Result<frame::ir::Node*> gx_node = frame::ops::create_node_with_inferred_types(
      graph, "reshape", {gx4_node.value()->output(0)}, gx_attrs);
  if (!gx_node.is_ok()) return gx_node.status();

  const frame::ops::AttrMap gw4_attrs{
      {"filter_shape", w4_shape},
      {"stride", stride2d},
      {"padding", padding2d},
      {"groups", groups},
  };
  const Result<frame::ir::Node*> gw4_node = frame::ops::create_node_with_inferred_types(
      graph, "conv2d_grad_filter_internal",
      {x4_node.value()->output(0), dy4_node.value()->output(0)}, gw4_attrs);
  if (!gw4_node.is_ok()) return gw4_node.status();
  const frame::ops::AttrMap gw_attrs{{"target_shape", w_shape}};
  const Result<frame::ir::Node*> gw_node = frame::ops::create_node_with_inferred_types(
      graph, "reshape", {gw4_node.value()->output(0)}, gw_attrs);
  if (!gw_node.is_ok()) return gw_node.status();

  const Status mark_gx = graph.mark_output(gx_node.value(), 0);
  if (!mark_gx.is_ok()) return mark_gx;
  const Status mark_gw = graph.mark_output(gw_node.value(), 0);
  if (!mark_gw.is_ok()) return mark_gw;

  if (bias_count == 1) {
    const frame::ops::AttrMap gbias_attrs{{"axes", std::vector<int64_t>{0, 2}}};
    const Result<frame::ir::Node*> gbias_node =
        frame::ops::create_node_with_inferred_types(graph, "sum", {gy_in.value()}, gbias_attrs);
    if (!gbias_node.is_ok()) return gbias_node.status();
    const Status mark_gbias = graph.mark_output(gbias_node.value(), 0);
    if (!mark_gbias.is_ok()) return mark_gbias;
  }

  return graph;
}

}  // namespace

FRAME_REGISTER_OP("conv2d")
    .input("x", "input tensor, rank-4 [N, Cin, H, W]")
    .input("w", "filter tensor, rank-4 [Cout, Cin/groups, KH, KW]")
    .variadic_input("bias", "optional bias tensor, rank-1 [Cout] (0 or 1 tensor)",
                    /*min_count=*/0)
    .attr("stride", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .attr("padding", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .attr("groups", frame::ir::AttrType::kInt64, /*required=*/true)
    .output("out", "2d convolution of x by w (+ bias), rank-4 [N, Cout, out_h, out_w]")
    .shape_infer(&infer_conv2d_shape)
    .gradient(&conv2d_gradient);

// conv2d_grad_input_internal/conv2d_grad_filter_internal(M21,conv2d 的梯度
// internal 算子):不面向用户(_internal 后缀,PY-021 天然豁免),仅供
// conv2d_gradient/conv1d_gradient 内联使用;cpu kernel 见
// src/backends/cpu/kernels/conv.cpp。
FRAME_REGISTER_OP("conv2d_grad_input_internal")
    .input("dy", "upstream gradient of conv2d's output, rank-4 [N, Cout, out_h, out_w]")
    .input("w", "conv2d's original filter operand, rank-4 [Cout, Cin/groups, KH, KW]")
    .attr("input_shape", frame::ir::AttrType::kShape, /*required=*/true)
    .attr("stride", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .attr("padding", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .attr("groups", frame::ir::AttrType::kInt64, /*required=*/true)
    .output("gx", "gradient w.r.t. x (BackwardData), rank-4 [N, Cin, H, W]")
    .shape_infer(&infer_conv2d_grad_input_internal_shape)
    .gradient(&conv2d_grad_input_internal_gradient);

FRAME_REGISTER_OP("conv2d_grad_filter_internal")
    .input("x", "conv2d's original input operand, rank-4 [N, Cin, H, W]")
    .input("dy", "upstream gradient of conv2d's output, rank-4 [N, Cout, out_h, out_w]")
    .attr("filter_shape", frame::ir::AttrType::kShape, /*required=*/true)
    .attr("stride", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .attr("padding", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .attr("groups", frame::ir::AttrType::kInt64, /*required=*/true)
    .output("gw", "gradient w.r.t. w (BackwardFilter), rank-4 [Cout, Cin/groups, KH, KW]")
    .shape_infer(&infer_conv2d_grad_filter_internal_shape)
    .gradient(&conv2d_grad_filter_internal_gradient);

// conv1d:同构 x[N,Cin,L]/w[Cout,Cin/g,K]/可选 bias[Cout],attrs stride/
// padding/groups 均 kInt64(标量)。decomposition 落 cuDNN(裁决点②);梯度
// 直接组合 reshape + conv2d 两个梯度 internal 算子(无 conv1d 专属 internal
// 算子)。
FRAME_REGISTER_OP("conv1d")
    .input("x", "input tensor, rank-3 [N, Cin, L]")
    .input("w", "filter tensor, rank-3 [Cout, Cin/groups, K]")
    .variadic_input("bias", "optional bias tensor, rank-1 [Cout] (0 or 1 tensor)",
                    /*min_count=*/0)
    .attr("stride", frame::ir::AttrType::kInt64, /*required=*/true)
    .attr("padding", frame::ir::AttrType::kInt64, /*required=*/true)
    .attr("groups", frame::ir::AttrType::kInt64, /*required=*/true)
    .output("out", "1d convolution of x by w (+ bias), rank-3 [N, Cout, out_l]")
    .shape_infer(&infer_conv1d_shape)
    .decomposition(&conv1d_decompose)
    .gradient(&conv1d_gradient);
