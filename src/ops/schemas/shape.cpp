// 内置形状变换算子 schema 注册桩:reshape(M21,批3 T4,公开)+
// transpose/concat/slice(M22,批4 T3,公开,§1.4 决议点D)。设计依据:
// docs/plan/2026-07-18-batch3-m21-conv.md 第1.1/1.2 节、
// docs/plan/2026-07-19-batch4-m22-seq.md §1.4(design-reviewer 均已 APPROVE)。
// reshape:numel 守恒、无跨元素依赖但非逐元素(输出 shape 不等于输入 shape,
// 不标 kElementwise);梯度=reshape 回输入形状。transpose/concat/slice 三者
// 对 fp32/fp16/bf16 三档为纯搬运,无数值语义;concat↔slice 互为伴随
// (M21 max pool 互逆对同构),transpose 自伴(逆排列)。cpu kernel(逐字节/
// 逐元素搬运)见 src/backends/cpu/kernels/shape.cpp。

#include <cstdint>
#include <string>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/graph_builder.h>
#include <frame/ops/op_registry.h>

#include "schema_math.h"

namespace {

using frame::DType;
using frame::ErrorCode;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::ops::NodeContext;

// 构造 shape 全形常量节点:同目录共享工具(铁律 5,见 schema_math.h 头注释)。
using frame::ops::schemas::make_constant_splat;

// reshape 的 shape 推断:1 输入;attr target_shape(kShape)。numel 必须与输入
// 相等(错误消息含双方 numel);输出即 target_shape。
Result<std::vector<Shape>> infer_reshape_shape(const NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument, "op 'reshape' expects 1 input, got " +
                                                         std::to_string(ctx.input_types.size()));
  }
  const Shape& x_shape = ctx.input_types[0].shape;

  const Shape* target_shape = ctx.attr<Shape>("target_shape");
  if (target_shape == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'reshape' is missing required attribute 'target_shape'");
  }
  if (target_shape->has_dynamic_dim()) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'reshape' attribute 'target_shape' " + target_shape->to_string() +
                            " has a dynamic dimension, static shape required (ARCH-013)");
  }

  const int64_t x_numel = x_shape.numel();
  const int64_t target_numel = target_shape->numel();
  if (x_numel != target_numel) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'reshape' requires numel to be preserved: x " + x_shape.to_string() +
                            " has numel=" + std::to_string(x_numel) + ", target_shape " +
                            target_shape->to_string() +
                            " has numel=" + std::to_string(target_numel));
  }

  return std::vector<Shape>{*target_shape};
}

// reshape(x) 的梯度:gx=reshape(gy) 回 x 的原 shape(numel 恒相等,故 reshape
// 自身的 shape 推断在此必然成功)。
Result<frame::ir::Graph> reshape_gradient(const NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument, "op 'reshape' gradient expects 1 input, got " +
                                                         std::to_string(ctx.input_types.size()));
  }
  frame::ir::Graph graph("reshape_gradient");

  const Result<frame::ir::Value*> x_in = graph.add_graph_input(ctx.input_types[0]);
  if (!x_in.is_ok()) return x_in.status();

  const Result<std::vector<Shape>> y_shapes = infer_reshape_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const Result<frame::ir::Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();

  const frame::ops::AttrMap gx_attrs{{"target_shape", ctx.input_types[0].shape}};
  const Result<frame::ir::Node*> gx_node =
      frame::ops::create_node_with_inferred_types(graph, "reshape", {gy_in.value()}, gx_attrs);
  if (!gx_node.is_ok()) return gx_node.status();

  const Status mark_gx = graph.mark_output(gx_node.value(), 0);
  if (!mark_gx.is_ok()) return mark_gx;

  return graph;
}

// ---------------------------------------------------------------------------
// transpose(x; perm)(M22,批4 T3,§1.4):任意秩;perm 必须是 [0,rank) 的排列
// (长度=rank、逐元素在界、无重复)。输出 dims[i]=x.dims[perm[i]]。
// ---------------------------------------------------------------------------

Result<std::vector<Shape>> infer_transpose_shape(const NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument, "op 'transpose' expects 1 input, got " +
                                                         std::to_string(ctx.input_types.size()));
  }
  const Shape& x_shape = ctx.input_types[0].shape;
  const int64_t rank = x_shape.rank();

  const std::vector<int64_t>* perm = ctx.attr<std::vector<int64_t>>("perm");
  if (perm == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'transpose' is missing required attribute 'perm' (int64 array)");
  }
  if (static_cast<int64_t>(perm->size()) != rank) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'transpose' attribute 'perm' must have rank=" + std::to_string(rank) +
                            " element(s), got " + std::to_string(perm->size()));
  }
  std::vector<bool> seen(static_cast<size_t>(rank), false);
  for (int64_t p : *perm) {
    if (p < 0 || p >= rank) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "op 'transpose' attribute 'perm' entry " + std::to_string(p) +
                              " is out of range for rank " + std::to_string(rank) +
                              " (must satisfy 0 <= p < rank)");
    }
    if (seen[static_cast<size_t>(p)]) {
      return Status::make(ErrorCode::kInvalidArgument, "op 'transpose' attribute 'perm' entry " +
                                                           std::to_string(p) + " is duplicated");
    }
    seen[static_cast<size_t>(p)] = true;
  }

  std::vector<int64_t> out_dims(static_cast<size_t>(rank));
  for (int64_t i = 0; i < rank; ++i) {
    out_dims[static_cast<size_t>(i)] = x_shape.dim((*perm)[static_cast<size_t>(i)]);
  }
  return std::vector<Shape>{Shape(out_dims)};
}

// transpose(x) 的梯度:gx=transpose(gy, inverse_perm)(构图期求逆排列,自伴
// 闭包)。inverse_perm[perm[i]]=i。
Result<frame::ir::Graph> transpose_gradient(const NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "op 'transpose' gradient expects 1 input, got " + std::to_string(ctx.input_types.size()));
  }
  frame::ir::Graph graph("transpose_gradient");

  const Result<frame::ir::Value*> x_in = graph.add_graph_input(ctx.input_types[0]);
  if (!x_in.is_ok()) return x_in.status();

  const Result<std::vector<Shape>> y_shapes = infer_transpose_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const Result<frame::ir::Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();

  // perm 已由上面 infer_transpose_shape(ctx) 校验通过,此处直接取用(非空、
  // 合法排列)。
  const std::vector<int64_t>& perm = *ctx.attr<std::vector<int64_t>>("perm");
  const int64_t rank = static_cast<int64_t>(perm.size());
  std::vector<int64_t> inverse_perm(static_cast<size_t>(rank));
  for (int64_t i = 0; i < rank; ++i) {
    inverse_perm[static_cast<size_t>(perm[static_cast<size_t>(i)])] = i;
  }

  const frame::ops::AttrMap gx_attrs{{"perm", inverse_perm}};
  const Result<frame::ir::Node*> gx_node =
      frame::ops::create_node_with_inferred_types(graph, "transpose", {gy_in.value()}, gx_attrs);
  if (!gx_node.is_ok()) return gx_node.status();

  const Status mark_gx = graph.mark_output(gx_node.value(), 0);
  if (!mark_gx.is_ok()) return mark_gx;

  return graph;
}

// ---------------------------------------------------------------------------
// concat(xs...; axis)(M22,批4 T3,§1.4):variadic_input min_count=1(照抄
// conv.cpp bias 的 variadic 先例);同秩、除 axis 外逐维相等;
// 0<=axis<rank。min_count=1 的退化单输入=恒等拷贝,为 slice 满切片梯度所需
// (见下方 slice_gradient)。
// ---------------------------------------------------------------------------

Result<std::vector<Shape>> infer_concat_shape(const NodeContext& ctx) {
  const size_t total = ctx.input_types.size();
  if (total < 1) {
    return Status::make(ErrorCode::kInvalidArgument, "op 'concat' expects at least 1 input, got 0");
  }
  const frame::ir::TensorType& first = ctx.input_types[0];
  const int64_t rank = first.shape.rank();

  const int64_t* axis = ctx.attr<int64_t>("axis");
  if (axis == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'concat' is missing required attribute 'axis'");
  }
  if (*axis < 0 || *axis >= rank) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'concat' attribute 'axis' " + std::to_string(*axis) +
                            " is out of range for rank " + std::to_string(rank) +
                            " (must satisfy 0 <= axis < rank)");
  }

  int64_t axis_total = first.shape.dim(*axis);
  for (size_t i = 1; i < total; ++i) {
    const frame::ir::TensorType& current = ctx.input_types[i];
    if (!(current.dtype == first.dtype)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "op 'concat' requires all inputs of the same dtype, input " +
                              std::to_string(i) + " has '" + std::string(current.dtype.name()) +
                              "', input 0 has '" + std::string(first.dtype.name()) + "'");
    }
    if (current.shape.rank() != rank) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "op 'concat' requires all inputs of the same rank, input " +
                              std::to_string(i) + " has rank " +
                              std::to_string(current.shape.rank()) + ", input 0 has rank " +
                              std::to_string(rank));
    }
    for (int64_t d = 0; d < rank; ++d) {
      if (d == *axis) continue;
      if (current.shape.dim(d) != first.shape.dim(d)) {
        return Status::make(ErrorCode::kInvalidArgument,
                            "op 'concat' requires all inputs to match on non-axis dimension " +
                                std::to_string(d) + ", input " + std::to_string(i) + " has " +
                                std::to_string(current.shape.dim(d)) + ", input 0 has " +
                                std::to_string(first.shape.dim(d)));
      }
    }
    axis_total += current.shape.dim(*axis);
  }

  std::vector<int64_t> out_dims = first.shape.dims();
  out_dims[static_cast<size_t>(*axis)] = axis_total;
  return std::vector<Shape>{Shape(out_dims)};
}

// concat(xs...) 的梯度:gx_i=slice(gy, axis, start_i, stop_i)(各输入沿 axis
// 宽度的前缀和切回)。
Result<frame::ir::Graph> concat_gradient(const NodeContext& ctx) {
  const size_t total = ctx.input_types.size();
  if (total < 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'concat' gradient expects at least 1 input, got 0");
  }
  frame::ir::Graph graph("concat_gradient");

  std::vector<frame::ir::Value*> x_ins(total);
  for (size_t i = 0; i < total; ++i) {
    const Result<frame::ir::Value*> x_in = graph.add_graph_input(ctx.input_types[i]);
    if (!x_in.is_ok()) return x_in.status();
    x_ins[i] = x_in.value();
  }

  const Result<std::vector<Shape>> y_shapes = infer_concat_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const Result<frame::ir::Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();

  const int64_t axis = *ctx.attr<int64_t>("axis");

  int64_t start = 0;
  for (size_t i = 0; i < total; ++i) {
    const int64_t width = ctx.input_types[i].shape.dim(axis);
    const int64_t stop = start + width;
    const frame::ops::AttrMap slice_attrs{
        {"axis", axis},
        {"start", start},
        {"stop", stop},
    };
    const Result<frame::ir::Node*> gx_node =
        frame::ops::create_node_with_inferred_types(graph, "slice", {gy_in.value()}, slice_attrs);
    if (!gx_node.is_ok()) return gx_node.status();
    const Status mark_gx = graph.mark_output(gx_node.value(), 0);
    if (!mark_gx.is_ok()) return mark_gx;
    start = stop;
  }
  (void)x_ins;  // 各输入自身不参与梯度重算路径,仅占位声明(ARCH-063)

  return graph;
}

// ---------------------------------------------------------------------------
// slice(x; axis, start, stop)(M22,批4 T3,§1.4):单轴连续切片,
// 0<=start<stop<=dim。
// ---------------------------------------------------------------------------

Result<std::vector<Shape>> infer_slice_shape(const NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument, "op 'slice' expects 1 input, got " +
                                                         std::to_string(ctx.input_types.size()));
  }
  const Shape& x_shape = ctx.input_types[0].shape;
  const int64_t rank = x_shape.rank();

  const int64_t* axis = ctx.attr<int64_t>("axis");
  if (axis == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'slice' is missing required attribute 'axis'");
  }
  const int64_t* start = ctx.attr<int64_t>("start");
  if (start == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'slice' is missing required attribute 'start'");
  }
  const int64_t* stop = ctx.attr<int64_t>("stop");
  if (stop == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'slice' is missing required attribute 'stop'");
  }

  if (*axis < 0 || *axis >= rank) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'slice' attribute 'axis' " + std::to_string(*axis) +
                            " is out of range for rank " + std::to_string(rank) +
                            " (must satisfy 0 <= axis < rank)");
  }
  const int64_t dim = x_shape.dim(*axis);
  if (*start < 0 || *start >= dim) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'slice' attribute 'start' " + std::to_string(*start) +
                            " is out of range for dimension " + std::to_string(dim) +
                            " (must satisfy 0 <= start < dim)");
  }
  if (*stop <= *start || *stop > dim) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'slice' attribute 'stop' " + std::to_string(*stop) +
                            " is invalid for start=" + std::to_string(*start) + " and dimension " +
                            std::to_string(dim) + " (must satisfy start < stop <= dim)");
  }

  std::vector<int64_t> out_dims = x_shape.dims();
  out_dims[static_cast<size_t>(*axis)] = *stop - *start;
  return std::vector<Shape>{Shape(out_dims)};
}

// slice(x) 的梯度:gx=concat(前置 constant(0) splat, gy, 后置 constant(0)
// splat; axis),零段宽 0 时省略该输入;满切片时退化为 concat 单输入(min_count=1
// 由此必要,concat 单输入=恒等拷贝)。
Result<frame::ir::Graph> slice_gradient(const NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument, "op 'slice' gradient expects 1 input, got " +
                                                         std::to_string(ctx.input_types.size()));
  }
  frame::ir::Graph graph("slice_gradient");

  const Result<frame::ir::Value*> x_in = graph.add_graph_input(ctx.input_types[0]);
  if (!x_in.is_ok()) return x_in.status();

  const Result<std::vector<Shape>> y_shapes = infer_slice_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const Result<frame::ir::Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();

  const int64_t axis = *ctx.attr<int64_t>("axis");
  const int64_t start = *ctx.attr<int64_t>("start");
  const int64_t stop = *ctx.attr<int64_t>("stop");
  const Shape& x_shape = ctx.input_types[0].shape;
  const int64_t dim = x_shape.dim(axis);
  const DType x_dtype = ctx.input_types[0].dtype;
  const frame::Device x_device = ctx.input_types[0].device;

  std::vector<frame::ir::Value*> concat_inputs;
  if (start > 0) {
    std::vector<int64_t> pre_dims = x_shape.dims();
    pre_dims[static_cast<size_t>(axis)] = start;
    const Result<frame::ir::Node*> pre_node =
        make_constant_splat(graph, Shape(pre_dims), x_dtype, x_device, 0.0);
    if (!pre_node.is_ok()) return pre_node.status();
    concat_inputs.push_back(pre_node.value()->output(0));
  }
  concat_inputs.push_back(gy_in.value());
  if (stop < dim) {
    std::vector<int64_t> post_dims = x_shape.dims();
    post_dims[static_cast<size_t>(axis)] = dim - stop;
    const Result<frame::ir::Node*> post_node =
        make_constant_splat(graph, Shape(post_dims), x_dtype, x_device, 0.0);
    if (!post_node.is_ok()) return post_node.status();
    concat_inputs.push_back(post_node.value()->output(0));
  }

  const frame::ops::AttrMap concat_attrs{{"axis", axis}};
  const Result<frame::ir::Node*> gx_node =
      frame::ops::create_node_with_inferred_types(graph, "concat", concat_inputs, concat_attrs);
  if (!gx_node.is_ok()) return gx_node.status();

  const Status mark_gx = graph.mark_output(gx_node.value(), 0);
  if (!mark_gx.is_ok()) return mark_gx;

  return graph;
}

}  // namespace

FRAME_REGISTER_OP("reshape")
    .input("x", "input tensor")
    .attr("target_shape", frame::ir::AttrType::kShape, /*required=*/true)
    .output("out", "x reshaped to target_shape (numel preserved)")
    .shape_infer(&infer_reshape_shape)
    .gradient(&reshape_gradient);

// transpose(x; perm)(M22,批4 T3):任意秩,perm 必须是 [0,rank) 的排列。
FRAME_REGISTER_OP("transpose")
    .input("x", "input tensor")
    .attr("perm", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .output("out", "x with dimensions permuted: out.dims[i] = x.dims[perm[i]]")
    .shape_infer(&infer_transpose_shape)
    .gradient(&transpose_gradient);

// concat(xs...; axis)(M22,批4 T3):variadic_input min_count=1(单输入=恒等
// 拷贝,slice 满切片梯度所需)。
FRAME_REGISTER_OP("concat")
    .variadic_input("xs", "tensors to concatenate along axis (same rank/dtype, >= 1 tensor)",
                    /*min_count=*/1)
    .attr("axis", frame::ir::AttrType::kInt64, /*required=*/true)
    .output("out", "concatenation of xs along axis")
    .shape_infer(&infer_concat_shape)
    .gradient(&concat_gradient);

// slice(x; axis, start, stop)(M22,批4 T3):单轴连续切片,0<=start<stop<=dim。
FRAME_REGISTER_OP("slice")
    .input("x", "input tensor")
    .attr("axis", frame::ir::AttrType::kInt64, /*required=*/true)
    .attr("start", frame::ir::AttrType::kInt64, /*required=*/true)
    .attr("stop", frame::ir::AttrType::kInt64, /*required=*/true)
    .output("out", "x sliced along axis to the half-open range [start, stop)")
    .shape_infer(&infer_slice_shape)
    .gradient(&slice_gradient);
