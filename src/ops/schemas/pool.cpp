// 内置二维池化算子 schema 注册桩(M21,批3 T4):max_pool2d/avg_pool2d(公开)
// + max_pool2d_grad_internal/max_pool2d_select_internal/
// avg_pool2d_grad_internal(内部,梯度专用)。设计依据:
// docs/plan/2026-07-18-batch3-m21-conv.md 第1节(design-reviewer 两轮
// APPROVE)。max_pool2d 的 argmax 平局约定:取窗口内最低线性索引(与 cpu/cuda
// kernel 侧一致,严格伴随性依赖此约定,见
// src/backends/cpu/kernels/pool.cpp)。cpu kernel 见同一文件。

#include <cstdint>
#include <string>
#include <string_view>
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

using frame::ErrorCode;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::ops::NodeContext;

// floor 除法:同目录共享工具(铁律 5 收敛,见 schema_math.h 头注释)。
using frame::ops::schemas::floor_div_positive_denominator;

// 二维池化的几何参数,供 max_pool2d/avg_pool2d 及三个梯度 internal 算子共用
// (REUSE-002)。
struct Pool2dGeometry {
  int64_t n = 0;
  int64_t c = 0;
  int64_t h = 0;
  int64_t w = 0;
  int64_t kh = 0;
  int64_t kw = 0;
  int64_t stride_h = 0;
  int64_t stride_w = 0;
  int64_t pad_h = 0;
  int64_t pad_w = 0;
  int64_t out_h = 0;
  int64_t out_w = 0;
};

// 校验 x_shape([N,C,H,W])/kernel/stride/padding 的自洽性并推出输出空间维
// (floor 口径)。额外约束(池化专属,与 conv 不同):padding*2 <= kernel(逐维),
// 超出报错含实际值——池化窗口不允许被 padding 完全覆盖。op_name 仅用于拼
// 错误消息;kernel/stride/padding 三者均为 2 元 [h,w] 数组,是本算子族固定
// 契约形态(同 conv.cpp::compute_conv2d_geometry 头注释论证,调用点均以具名
// 局部变量传入,误置换在数值校验阶段立即报错)。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Result<Pool2dGeometry> compute_pool2d_geometry(std::string_view op_name, const Shape& x_shape,
                                               const std::vector<int64_t>& kernel,
                                               const std::vector<int64_t>& stride,
                                               const std::vector<int64_t>& padding) {
  if (x_shape.rank() != 4) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) +
                            "' requires x to be rank-4 [N, C, H, W], got rank " +
                            std::to_string(x_shape.rank()));
  }
  if (kernel.size() != 2) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) +
                            "' attribute 'kernel' must have 2 elements, got " +
                            std::to_string(kernel.size()));
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

  Pool2dGeometry geo;
  geo.n = x_shape.dim(0);
  geo.c = x_shape.dim(1);
  geo.h = x_shape.dim(2);
  geo.w = x_shape.dim(3);
  geo.kh = kernel[0];
  geo.kw = kernel[1];
  geo.stride_h = stride[0];
  geo.stride_w = stride[1];
  geo.pad_h = padding[0];
  geo.pad_w = padding[1];

  if (geo.n <= 0 || geo.c <= 0 || geo.h <= 0 || geo.w <= 0) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) +
                            "' requires x dims [N, C, H, W] to be positive, got " +
                            x_shape.to_string());
  }
  if (geo.kh < 1 || geo.kw < 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) +
                            "' attribute 'kernel' entries must be >= 1, got [" +
                            std::to_string(geo.kh) + ", " + std::to_string(geo.kw) + "]");
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
  if (geo.pad_h * 2 > geo.kh) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) +
                            "' padding_h*2=" + std::to_string(geo.pad_h * 2) +
                            " exceeds KH=" + std::to_string(geo.kh));
  }
  if (geo.pad_w * 2 > geo.kw) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) +
                            "' padding_w*2=" + std::to_string(geo.pad_w * 2) +
                            " exceeds KW=" + std::to_string(geo.kw));
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

// 读取 kernel/stride/padding 三个必需 kInt64Array 属性的公共步骤,供
// max_pool2d/avg_pool2d 两处 shape 推断共用(REUSE-002);op_name 仅用于拼
// 错误消息。返回三个属性指针(借用契约同 ctx.attrs,调用期间有效)。
struct PoolAttrs {
  const std::vector<int64_t>* kernel = nullptr;
  const std::vector<int64_t>* stride = nullptr;
  const std::vector<int64_t>* padding = nullptr;
};

Result<PoolAttrs> read_pool_attrs(std::string_view op_name, const NodeContext& ctx) {
  PoolAttrs attrs;
  attrs.kernel = ctx.attr<std::vector<int64_t>>("kernel");
  if (attrs.kernel == nullptr) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) + "' is missing required attribute 'kernel' (int64 array)");
  }
  attrs.stride = ctx.attr<std::vector<int64_t>>("stride");
  if (attrs.stride == nullptr) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) + "' is missing required attribute 'stride' (int64 array)");
  }
  attrs.padding = ctx.attr<std::vector<int64_t>>("padding");
  if (attrs.padding == nullptr) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) + "' is missing required attribute 'padding' (int64 array)");
  }
  return attrs;
}

// max_pool2d 的 shape 推断:x[N,C,H,W];attrs kernel/stride/padding(各
// kInt64Array 2元)。输出 [N,C,out_h,out_w]。
Result<std::vector<Shape>> infer_max_pool2d_shape(const NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument, "op 'max_pool2d' expects 1 input, got " +
                                                         std::to_string(ctx.input_types.size()));
  }
  const Result<PoolAttrs> attrs_result = read_pool_attrs("max_pool2d", ctx);
  if (!attrs_result.is_ok()) return attrs_result.status();
  const PoolAttrs& attrs = attrs_result.value();

  const Result<Pool2dGeometry> geo_result = compute_pool2d_geometry(
      "max_pool2d", ctx.input_types[0].shape, *attrs.kernel, *attrs.stride, *attrs.padding);
  if (!geo_result.is_ok()) return geo_result.status();
  const Pool2dGeometry& geo = geo_result.value();

  return std::vector<Shape>{Shape({geo.n, geo.c, geo.out_h, geo.out_w})};
}

// avg_pool2d 的 shape 推断:与 max_pool2d 共用几何校验(REUSE-002),仅算子名
// 不同(分母恒 KH·KW、padding 是否参与是 kernel 层语义,不影响 shape)。
Result<std::vector<Shape>> infer_avg_pool2d_shape(const NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument, "op 'avg_pool2d' expects 1 input, got " +
                                                         std::to_string(ctx.input_types.size()));
  }
  const Result<PoolAttrs> attrs_result = read_pool_attrs("avg_pool2d", ctx);
  if (!attrs_result.is_ok()) return attrs_result.status();
  const PoolAttrs& attrs = attrs_result.value();

  const Result<Pool2dGeometry> geo_result = compute_pool2d_geometry(
      "avg_pool2d", ctx.input_types[0].shape, *attrs.kernel, *attrs.stride, *attrs.padding);
  if (!geo_result.is_ok()) return geo_result.status();
  const Pool2dGeometry& geo = geo_result.value();

  return std::vector<Shape>{Shape({geo.n, geo.c, geo.out_h, geo.out_w})};
}

// max_pool2d_grad_internal(dy,x) 的 shape 推断(M21,max_pool2d 的梯度
// internal 算子之一):attrs=input_shape(kShape)+kernel/stride/padding。x
// 实际输入的 shape 须与 input_shape 一致(冗余但按计划显式携带,§1.2:
// "pool 反向系携带 input_shape + kernel/stride/padding");输出=input_shape。
Result<std::vector<Shape>> infer_max_pool2d_grad_internal_shape(const NodeContext& ctx) {
  if (ctx.input_types.size() != 2) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'max_pool2d_grad_internal' expects 2 inputs, got " +
                            std::to_string(ctx.input_types.size()));
  }
  const frame::ir::TensorType& dy = ctx.input_types[0];
  const frame::ir::TensorType& x = ctx.input_types[1];
  if (!(dy.dtype == x.dtype)) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'max_pool2d_grad_internal' requires dy and x of the same dtype, "
                        "got '" +
                            std::string(dy.dtype.name()) + "' and '" + std::string(x.dtype.name()) +
                            "'");
  }

  const Shape* input_shape = ctx.attr<Shape>("input_shape");
  if (input_shape == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'max_pool2d_grad_internal' is missing required attribute "
                        "'input_shape'");
  }
  if (input_shape->has_dynamic_dim()) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'max_pool2d_grad_internal' attribute 'input_shape' " +
                            input_shape->to_string() +
                            " has a dynamic dimension, static shape required (ARCH-013)");
  }
  if (!(x.shape == *input_shape)) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'max_pool2d_grad_internal' requires x shape to match attribute "
                        "'input_shape', got " +
                            x.shape.to_string() + ", expected " + input_shape->to_string());
  }

  const Result<PoolAttrs> attrs_result = read_pool_attrs("max_pool2d_grad_internal", ctx);
  if (!attrs_result.is_ok()) return attrs_result.status();
  const PoolAttrs& attrs = attrs_result.value();

  const Result<Pool2dGeometry> geo_result = compute_pool2d_geometry(
      "max_pool2d_grad_internal", *input_shape, *attrs.kernel, *attrs.stride, *attrs.padding);
  if (!geo_result.is_ok()) return geo_result.status();
  const Pool2dGeometry& geo = geo_result.value();

  const Shape expected_dy_shape({geo.n, geo.c, geo.out_h, geo.out_w});
  if (!(dy.shape == expected_dy_shape)) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'max_pool2d_grad_internal' requires dy shape to be consistent with "
                        "input_shape/kernel/stride/padding, got " +
                            dy.shape.to_string() + ", expected " + expected_dy_shape.to_string());
  }

  return std::vector<Shape>{*input_shape};
}

// max_pool2d_select_internal(g,x) 的 shape 推断(M21,max_pool2d 的梯度
// internal 算子之二):attrs=kernel/stride/padding(无 input_shape——x 本身
// 即为真实 shape 来源,与 grad_internal 不同)。g 须与 x 同 shape/dtype
// (g 是"流入 gx 的上游梯度",与 x/gx 同形);输出=池化输出形。
Result<std::vector<Shape>> infer_max_pool2d_select_internal_shape(const NodeContext& ctx) {
  if (ctx.input_types.size() != 2) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'max_pool2d_select_internal' expects 2 inputs, got " +
                            std::to_string(ctx.input_types.size()));
  }
  const frame::ir::TensorType& g = ctx.input_types[0];
  const frame::ir::TensorType& x = ctx.input_types[1];
  if (!(g.dtype == x.dtype)) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'max_pool2d_select_internal' requires g and x of the same dtype, "
                        "got '" +
                            std::string(g.dtype.name()) + "' and '" + std::string(x.dtype.name()) +
                            "'");
  }
  if (!(g.shape == x.shape)) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'max_pool2d_select_internal' requires g and x of the same shape, "
                        "got " +
                            g.shape.to_string() + " and " + x.shape.to_string());
  }

  const Result<PoolAttrs> attrs_result = read_pool_attrs("max_pool2d_select_internal", ctx);
  if (!attrs_result.is_ok()) return attrs_result.status();
  const PoolAttrs& attrs = attrs_result.value();

  const Result<Pool2dGeometry> geo_result = compute_pool2d_geometry(
      "max_pool2d_select_internal", x.shape, *attrs.kernel, *attrs.stride, *attrs.padding);
  if (!geo_result.is_ok()) return geo_result.status();
  const Pool2dGeometry& geo = geo_result.value();

  return std::vector<Shape>{Shape({geo.n, geo.c, geo.out_h, geo.out_w})};
}

// avg_pool2d_grad_internal(dy) 的 shape 推断(M21,avg_pool2d 的梯度 internal
// 算子):单输入 dy;attrs=input_shape(kShape)+kernel/stride/padding。输出=
// input_shape。
Result<std::vector<Shape>> infer_avg_pool2d_grad_internal_shape(const NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'avg_pool2d_grad_internal' expects 1 input, got " +
                            std::to_string(ctx.input_types.size()));
  }
  const frame::ir::TensorType& dy = ctx.input_types[0];

  const Shape* input_shape = ctx.attr<Shape>("input_shape");
  if (input_shape == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'avg_pool2d_grad_internal' is missing required attribute "
                        "'input_shape'");
  }
  if (input_shape->has_dynamic_dim()) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'avg_pool2d_grad_internal' attribute 'input_shape' " +
                            input_shape->to_string() +
                            " has a dynamic dimension, static shape required (ARCH-013)");
  }

  const Result<PoolAttrs> attrs_result = read_pool_attrs("avg_pool2d_grad_internal", ctx);
  if (!attrs_result.is_ok()) return attrs_result.status();
  const PoolAttrs& attrs = attrs_result.value();

  const Result<Pool2dGeometry> geo_result = compute_pool2d_geometry(
      "avg_pool2d_grad_internal", *input_shape, *attrs.kernel, *attrs.stride, *attrs.padding);
  if (!geo_result.is_ok()) return geo_result.status();
  const Pool2dGeometry& geo = geo_result.value();

  const Shape expected_dy_shape({geo.n, geo.c, geo.out_h, geo.out_w});
  if (!(dy.shape == expected_dy_shape)) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'avg_pool2d_grad_internal' requires dy shape to be consistent with "
                        "input_shape/kernel/stride/padding, got " +
                            dy.shape.to_string() + ", expected " + expected_dy_shape.to_string());
  }

  return std::vector<Shape>{*input_shape};
}

// 从 ctx.attrs 转发 kernel/stride/padding 三个 kInt64Array 属性,构造下游
// internal 节点所需的 AttrMap 片段(供多处梯度微图构建复用,REUSE-002)。
frame::ops::AttrMap forward_pool_attrs(const PoolAttrs& attrs) {
  return frame::ops::AttrMap{
      {"kernel", *attrs.kernel},
      {"stride", *attrs.stride},
      {"padding", *attrs.padding},
  };
}

// max_pool2d(x) 的梯度:gx=max_pool2d_grad_internal(dy,x)。
Result<frame::ir::Graph> max_pool2d_gradient(const NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "op 'max_pool2d' gradient expects 1 input, got " + std::to_string(ctx.input_types.size()));
  }
  frame::ir::Graph graph("max_pool2d_gradient");

  const Result<frame::ir::Value*> x_in = graph.add_graph_input(ctx.input_types[0]);
  if (!x_in.is_ok()) return x_in.status();

  const Result<std::vector<Shape>> y_shapes = infer_max_pool2d_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const Result<frame::ir::Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();

  const Result<PoolAttrs> attrs_result = read_pool_attrs("max_pool2d", ctx);
  if (!attrs_result.is_ok()) return attrs_result.status();
  frame::ops::AttrMap gx_attrs = forward_pool_attrs(attrs_result.value());
  gx_attrs.emplace("input_shape", ctx.input_types[0].shape);

  const Result<frame::ir::Node*> gx_node = frame::ops::create_node_with_inferred_types(
      graph, "max_pool2d_grad_internal", {gy_in.value(), x_in.value()}, gx_attrs);
  if (!gx_node.is_ok()) return gx_node.status();

  const Status mark_gx = graph.mark_output(gx_node.value(), 0);
  if (!mark_gx.is_ok()) return mark_gx;

  return graph;
}

// avg_pool2d(x) 的梯度:gx=avg_pool2d_grad_internal(dy)。
Result<frame::ir::Graph> avg_pool2d_gradient(const NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "op 'avg_pool2d' gradient expects 1 input, got " + std::to_string(ctx.input_types.size()));
  }
  frame::ir::Graph graph("avg_pool2d_gradient");

  const Result<frame::ir::Value*> x_in = graph.add_graph_input(ctx.input_types[0]);
  if (!x_in.is_ok()) return x_in.status();

  const Result<std::vector<Shape>> y_shapes = infer_avg_pool2d_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const Result<frame::ir::Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();

  const Result<PoolAttrs> attrs_result = read_pool_attrs("avg_pool2d", ctx);
  if (!attrs_result.is_ok()) return attrs_result.status();
  frame::ops::AttrMap gx_attrs = forward_pool_attrs(attrs_result.value());
  gx_attrs.emplace("input_shape", ctx.input_types[0].shape);

  const Result<frame::ir::Node*> gx_node = frame::ops::create_node_with_inferred_types(
      graph, "avg_pool2d_grad_internal", {gy_in.value()}, gx_attrs);
  if (!gx_node.is_ok()) return gx_node.status();

  const Status mark_gx = graph.mark_output(gx_node.value(), 0);
  if (!mark_gx.is_ok()) return mark_gx;

  return graph;
}

// 构造 shape 全形、值全为 fill_value 的 constant 节点(square_gradient 的
// constant(2) 先例同机制,REUSE-002:pool 系两个"wrt x 为零梯度"分支共用)。
Result<frame::ir::Node*> make_constant_splat(frame::ir::Graph& graph, const Shape& shape,
                                             frame::DType dtype, frame::Device device,
                                             double fill_value) {
  const int64_t numel = shape.numel();
  const std::vector<double> values(static_cast<size_t>(numel), fill_value);
  const frame::ops::AttrMap attrs{
      {"value", values},
      {"shape", shape},
      {"dtype", dtype},
  };
  return frame::ops::create_node_with_inferred_types(graph, frame::ops::kConstantOpName, device,
                                                     attrs);
}

// max_pool2d_grad_internal(dy,x) 自身的梯度(§1.2 表,R11 封闭性):
// wrt dy → max_pool2d_select_internal(g,x);wrt x → 零(分段常数,relu 拐点
// 同口径)。
Result<frame::ir::Graph> max_pool2d_grad_internal_gradient(const NodeContext& ctx) {
  if (ctx.input_types.size() != 2) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'max_pool2d_grad_internal' gradient expects 2 inputs, got " +
                            std::to_string(ctx.input_types.size()));
  }
  frame::ir::Graph graph("max_pool2d_grad_internal_gradient");

  const Result<frame::ir::Value*> dy_in = graph.add_graph_input(ctx.input_types[0]);
  if (!dy_in.is_ok()) return dy_in.status();
  const Result<frame::ir::Value*> x_in = graph.add_graph_input(ctx.input_types[1]);
  if (!x_in.is_ok()) return x_in.status();

  const Result<std::vector<Shape>> y_shapes = infer_max_pool2d_grad_internal_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const Result<frame::ir::Value*> g_in = graph.add_graph_input(y_type);
  if (!g_in.is_ok()) return g_in.status();

  const Result<PoolAttrs> attrs_result = read_pool_attrs("max_pool2d_grad_internal", ctx);
  if (!attrs_result.is_ok()) return attrs_result.status();
  const frame::ops::AttrMap select_attrs = forward_pool_attrs(attrs_result.value());

  const Result<frame::ir::Node*> wrt_dy_node = frame::ops::create_node_with_inferred_types(
      graph, "max_pool2d_select_internal", {g_in.value(), x_in.value()}, select_attrs);
  if (!wrt_dy_node.is_ok()) return wrt_dy_node.status();

  const Result<frame::ir::Node*> wrt_x_node = make_constant_splat(
      graph, ctx.input_types[1].shape, ctx.input_types[1].dtype, ctx.input_types[1].device, 0.0);
  if (!wrt_x_node.is_ok()) return wrt_x_node.status();

  const Status mark_wrt_dy = graph.mark_output(wrt_dy_node.value(), 0);
  if (!mark_wrt_dy.is_ok()) return mark_wrt_dy;
  const Status mark_wrt_x = graph.mark_output(wrt_x_node.value(), 0);
  if (!mark_wrt_x.is_ok()) return mark_wrt_x;

  return graph;
}

// max_pool2d_select_internal(g,x) 自身的梯度(§1.2 表,R11 封闭性):
// wrt g → max_pool2d_grad_internal(gg,x);wrt x → 零。
Result<frame::ir::Graph> max_pool2d_select_internal_gradient(const NodeContext& ctx) {
  if (ctx.input_types.size() != 2) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'max_pool2d_select_internal' gradient expects 2 inputs, got " +
                            std::to_string(ctx.input_types.size()));
  }
  frame::ir::Graph graph("max_pool2d_select_internal_gradient");

  const Result<frame::ir::Value*> g_in = graph.add_graph_input(ctx.input_types[0]);
  if (!g_in.is_ok()) return g_in.status();
  const Result<frame::ir::Value*> x_in = graph.add_graph_input(ctx.input_types[1]);
  if (!x_in.is_ok()) return x_in.status();

  const Result<std::vector<Shape>> y_shapes = infer_max_pool2d_select_internal_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const Result<frame::ir::Value*> gg_in = graph.add_graph_input(y_type);
  if (!gg_in.is_ok()) return gg_in.status();

  const Result<PoolAttrs> attrs_result = read_pool_attrs("max_pool2d_select_internal", ctx);
  if (!attrs_result.is_ok()) return attrs_result.status();
  frame::ops::AttrMap grad_attrs = forward_pool_attrs(attrs_result.value());
  grad_attrs.emplace("input_shape", ctx.input_types[1].shape);

  const Result<frame::ir::Node*> wrt_g_node = frame::ops::create_node_with_inferred_types(
      graph, "max_pool2d_grad_internal", {gg_in.value(), x_in.value()}, grad_attrs);
  if (!wrt_g_node.is_ok()) return wrt_g_node.status();

  const Result<frame::ir::Node*> wrt_x_node = make_constant_splat(
      graph, ctx.input_types[1].shape, ctx.input_types[1].dtype, ctx.input_types[1].device, 0.0);
  if (!wrt_x_node.is_ok()) return wrt_x_node.status();

  const Status mark_wrt_g = graph.mark_output(wrt_g_node.value(), 0);
  if (!mark_wrt_g.is_ok()) return mark_wrt_g;
  const Status mark_wrt_x = graph.mark_output(wrt_x_node.value(), 0);
  if (!mark_wrt_x.is_ok()) return mark_wrt_x;

  return graph;
}

// avg_pool2d_grad_internal(dy) 自身的梯度(§1.2 表,R11 封闭性):
// wrt dy → avg_pool2d(g)(线性伴随互逆)。
Result<frame::ir::Graph> avg_pool2d_grad_internal_gradient(const NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'avg_pool2d_grad_internal' gradient expects 1 input, got " +
                            std::to_string(ctx.input_types.size()));
  }
  frame::ir::Graph graph("avg_pool2d_grad_internal_gradient");

  const Result<frame::ir::Value*> dy_in = graph.add_graph_input(ctx.input_types[0]);
  if (!dy_in.is_ok()) return dy_in.status();

  const Result<std::vector<Shape>> y_shapes = infer_avg_pool2d_grad_internal_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const Result<frame::ir::Value*> g_in = graph.add_graph_input(y_type);
  if (!g_in.is_ok()) return g_in.status();

  const Result<PoolAttrs> attrs_result = read_pool_attrs("avg_pool2d_grad_internal", ctx);
  if (!attrs_result.is_ok()) return attrs_result.status();
  const frame::ops::AttrMap pool_attrs = forward_pool_attrs(attrs_result.value());

  const Result<frame::ir::Node*> wrt_dy_node =
      frame::ops::create_node_with_inferred_types(graph, "avg_pool2d", {g_in.value()}, pool_attrs);
  if (!wrt_dy_node.is_ok()) return wrt_dy_node.status();

  const Status mark_wrt_dy = graph.mark_output(wrt_dy_node.value(), 0);
  if (!mark_wrt_dy.is_ok()) return mark_wrt_dy;

  return graph;
}

}  // namespace

FRAME_REGISTER_OP("max_pool2d")
    .input("x", "input tensor, rank-4 [N, C, H, W]")
    .attr("kernel", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .attr("stride", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .attr("padding", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .output("out", "max pooling of x, rank-4 [N, C, out_h, out_w]; padding uses -inf semantics")
    .shape_infer(&infer_max_pool2d_shape)
    .gradient(&max_pool2d_gradient);

FRAME_REGISTER_OP("avg_pool2d")
    .input("x", "input tensor, rank-4 [N, C, H, W]")
    .attr("kernel", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .attr("stride", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .attr("padding", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .output("out",
            "average pooling of x, rank-4 [N, C, out_h, out_w]; denominator is always "
            "KH*KW (padding included)")
    .shape_infer(&infer_avg_pool2d_shape)
    .gradient(&avg_pool2d_gradient);

// max_pool2d_grad_internal/max_pool2d_select_internal/avg_pool2d_grad_internal
// (M21,pool 系梯度 internal 算子):不面向用户(_internal 后缀,PY-021 天然
// 豁免),仅供各自 gradient 内联使用;cpu kernel 见
// src/backends/cpu/kernels/pool.cpp。
FRAME_REGISTER_OP("max_pool2d_grad_internal")
    .input("dy", "upstream gradient of max_pool2d's output, rank-4 [N, C, out_h, out_w]")
    .input("x", "max_pool2d's original input, rank-4 [N, C, H, W]")
    .attr("input_shape", frame::ir::AttrType::kShape, /*required=*/true)
    .attr("kernel", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .attr("stride", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .attr("padding", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .output("gx",
            "gradient w.r.t. x: dy scattered back to x's argmax positions (ties broken by "
            "the lowest linear window index)")
    .shape_infer(&infer_max_pool2d_grad_internal_shape)
    .gradient(&max_pool2d_grad_internal_gradient);

FRAME_REGISTER_OP("max_pool2d_select_internal")
    .input("g", "tensor to gather from, rank-4 [N, C, H, W] (same shape as x)")
    .input("x", "reference tensor for recomputing each window's argmax, rank-4 [N, C, H, W]")
    .attr("kernel", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .attr("stride", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .attr("padding", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .output("out",
            "per window, the value of g at x's argmax position, rank-4 [N, C, out_h, "
            "out_w]")
    .shape_infer(&infer_max_pool2d_select_internal_shape)
    .gradient(&max_pool2d_select_internal_gradient);

FRAME_REGISTER_OP("avg_pool2d_grad_internal")
    .input("dy", "upstream gradient of avg_pool2d's output, rank-4 [N, C, out_h, out_w]")
    .attr("input_shape", frame::ir::AttrType::kShape, /*required=*/true)
    .attr("kernel", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .attr("stride", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .attr("padding", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .output("gx",
            "gradient w.r.t. x: dy spread evenly (/ (KH*KW)) back over each covering "
            "window, rank-4 [N, C, H, W]")
    .shape_infer(&infer_avg_pool2d_grad_internal_shape)
    .gradient(&avg_pool2d_grad_internal_gradient);
