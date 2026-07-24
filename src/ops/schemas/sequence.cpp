// 内置序列/归一化算子 schema 注册桩:M22 softmax/layer_norm 与 M25
// selective_scan(均为公开算子)。设计依据:
// docs/plan/2026-07-19-batch4-m22-seq.md §1.2/1.3(design-reviewer 两轮
// APPROVE)。两算子均限 rank-2 [rows, cols]、作用于末轴(高秩场景由调用方
// reshape 折叠 leading span,与 Linear 的 batch 显式口径同哲学);梯度微图
// 全部经已注册公开算子表达(R11,近零新增内部算子):行广播统一经
// "sum(axes,keepdims=false) → reshape[N,1] → matmul(ones[1,D])"(bcast_col,
// 广播逐行标量至各列)或"reshape[D]→[1,D] → matmul(ones[N,1], ·)"(bcast_row,
// 广播逐列向量至各行)两种三连表达;减法经 constant(−1) splat + mul + add
// 组合(v0 无 sub 算子)。cpu kernel 见 src/backends/cpu/kernels/sequence.cpp。

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

using frame::Device;
using frame::DType;
using frame::ErrorCode;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::Value;
using frame::ops::AttrMap;
using frame::ops::create_node_with_inferred_types;
using frame::ops::NodeContext;

// 构造 shape 全形常量节点:同目录共享工具(铁律 5,见 schema_math.h 头注释)。
using frame::ops::schemas::make_constant_splat;
using frame::ops::schemas::static_shape_numel_fits_int64;

// ---------------------------------------------------------------------------
// selective_scan(x,a,b,c,d):任意 rank>=1,最后一轴为时间步。五个输入
// shape/dtype 完全相同,逐条前导序列执行一阶状态递推。
// ---------------------------------------------------------------------------

Result<std::vector<Shape>> infer_selective_scan_shape(const NodeContext& ctx) {
  constexpr size_t kInputCount = 5;
  if (ctx.input_types.size() != kInputCount) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'selective_scan' expects 5 inputs (x, a, b, c, d), got " +
                            std::to_string(ctx.input_types.size()));
  }

  const frame::ir::TensorType& x = ctx.input_types[0];
  if (x.shape.rank() < 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'selective_scan' requires rank >= 1, got rank 0");
  }
  for (size_t i = 0; i < kInputCount; ++i) {
    const frame::ir::TensorType& input = ctx.input_types[i];
    const Status shape_status = input.shape.verify();
    if (!shape_status.is_ok()) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "op 'selective_scan' input " + std::to_string(i) + " shape " +
                              input.shape.to_string() +
                              " is invalid: " + std::string(shape_status.message()));
    }
    if (!static_shape_numel_fits_int64(input.shape)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "op 'selective_scan' input " + std::to_string(i) + " shape " +
                              input.shape.to_string() + " element count overflows int64");
    }
    if (!(input.shape == x.shape)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "op 'selective_scan' requires all input shapes to match x " +
                              x.shape.to_string() + ", input " + std::to_string(i) + " has " +
                              input.shape.to_string());
    }
    if (!(input.dtype == x.dtype)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "op 'selective_scan' requires all input dtypes to match x '" +
                              std::string(x.dtype.name()) + "', input " + std::to_string(i) +
                              " has '" + std::string(input.dtype.name()) + "'");
    }
  }

  const int64_t steps = x.shape.dim(x.shape.rank() - 1);
  if (steps < 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'selective_scan' requires the last dimension (steps) to be >= 1, got " +
                            std::to_string(steps));
  }
  const frame::DTypeCode code = x.dtype.code();
  const bool supported = (code == frame::DTypeCode::kFloat32) ||
                         (code == frame::DTypeCode::kFloat16) ||
                         (code == frame::DTypeCode::kBFloat16);
  if (!supported) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'selective_scan' does not support dtype '" +
                            std::string(x.dtype.name()) +
                            "' (v0 supports float32/float16/bfloat16 only)");
  }
  return std::vector<Shape>{x.shape};
}

// 以逐时间步 slice + concat 静态反转最后一轴。steps 已由 shape 推断保证 >=1;
// steps=1 仍走同一路径,退化为单输入 concat。
// axis/steps 均由同一个已校验 shape 派生,调用点始终以具名变量传入;若误置换,
// 非方形测试会由 slice axis 越界或时间范围校验立即拒绝。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Result<Node*> reverse_time(Graph& graph, Value* x, int64_t axis, int64_t steps) {
  std::vector<Value*> reversed_steps;
  reversed_steps.reserve(static_cast<size_t>(steps));
  for (int64_t stop = steps; stop > 0; --stop) {
    const AttrMap slice_attrs{{"axis", axis}, {"start", stop - 1}, {"stop", stop}};
    const Result<Node*> step_node =
        create_node_with_inferred_types(graph, "slice", {x}, slice_attrs);
    if (!step_node.is_ok()) return step_node.status();
    reversed_steps.push_back(step_node.value()->output(0));
  }
  const AttrMap concat_attrs{{"axis", axis}};
  return create_node_with_inferred_types(graph, "concat", reversed_steps, concat_attrs);
}

// selective_scan 的 GradientFn 输入按 [x,a,b,c,d,y,gy],输出按
// [gx,ga,gb,gc,gd]。先重算状态 h,再把伴随递推静态反转为第二次
// selective_scan,最终仅用 mul/add 组合五个梯度。
Result<Graph> selective_scan_gradient(const NodeContext& ctx) {
  if (ctx.input_types.size() != 5) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'selective_scan' gradient expects 5 inputs (x, a, b, c, d), got " +
                            std::to_string(ctx.input_types.size()));
  }
  const Result<std::vector<Shape>> y_shapes = infer_selective_scan_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();

  Graph graph("selective_scan_gradient");
  std::vector<Value*> inputs;
  inputs.reserve(5);
  for (const frame::ir::TensorType& input_type : ctx.input_types) {
    const Result<Value*> input = graph.add_graph_input(input_type);
    if (!input.is_ok()) return input.status();
    inputs.push_back(input.value());
  }
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const Result<Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const Result<Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();
  (void)y_in;  // 前向输出不参与解析式,仅占位满足 GradientFn 输入契约

  Value* x = inputs[0];
  Value* a = inputs[1];
  Value* b = inputs[2];
  Value* c = inputs[3];
  Value* d = inputs[4];
  const Shape& shape = ctx.input_types[0].shape;
  const int64_t axis = shape.rank() - 1;
  const int64_t steps = shape.dim(axis);
  const DType dtype = ctx.input_types[0].dtype;
  const Device device = ctx.input_types[0].device;

  const Result<Node*> ones_node = make_constant_splat(graph, shape, dtype, device, 1.0);
  if (!ones_node.is_ok()) return ones_node.status();
  const Result<Node*> zeros_node = make_constant_splat(graph, shape, dtype, device, 0.0);
  if (!zeros_node.is_ok()) return zeros_node.status();
  Value* ones = ones_node.value()->output(0);
  Value* zeros = zeros_node.value()->output(0);

  // h=selective_scan(x,a,b,ones,zeros),即重算完整状态序列。
  const Result<Node*> h_node =
      create_node_with_inferred_types(graph, "selective_scan", {x, a, b, ones, zeros});
  if (!h_node.is_ok()) return h_node.status();
  Value* h = h_node.value()->output(0);

  // h_prev=concat(zero_at_t0,h[:-1]);steps=1 时退化为单步零常量。
  std::vector<int64_t> one_step_dims = shape.dims();
  one_step_dims[static_cast<size_t>(axis)] = 1;
  const Result<Node*> zero_step_node =
      make_constant_splat(graph, Shape(one_step_dims), dtype, device, 0.0);
  if (!zero_step_node.is_ok()) return zero_step_node.status();
  Value* zero_step = zero_step_node.value()->output(0);

  Value* h_prev = zero_step;
  if (steps > 1) {
    const AttrMap h_prefix_attrs{{"axis", axis}, {"start", int64_t{0}}, {"stop", steps - 1}};
    const Result<Node*> h_prefix_node =
        create_node_with_inferred_types(graph, "slice", {h}, h_prefix_attrs);
    if (!h_prefix_node.is_ok()) return h_prefix_node.status();
    const AttrMap concat_attrs{{"axis", axis}};
    const Result<Node*> h_prev_node = create_node_with_inferred_types(
        graph, "concat", {zero_step, h_prefix_node.value()->output(0)}, concat_attrs);
    if (!h_prev_node.is_ok()) return h_prev_node.status();
    h_prev = h_prev_node.value()->output(0);
  }

  // q=gy*c;a_next=concat(a[1:],zero_at_last),使伴随递推写成正向扫描。
  const Result<Node*> q_node = create_node_with_inferred_types(graph, "mul", {gy_in.value(), c});
  if (!q_node.is_ok()) return q_node.status();
  Value* a_next = zero_step;
  if (steps > 1) {
    const AttrMap a_suffix_attrs{{"axis", axis}, {"start", int64_t{1}}, {"stop", steps}};
    const Result<Node*> a_suffix_node =
        create_node_with_inferred_types(graph, "slice", {a}, a_suffix_attrs);
    if (!a_suffix_node.is_ok()) return a_suffix_node.status();
    const AttrMap concat_attrs{{"axis", axis}};
    const Result<Node*> a_next_node = create_node_with_inferred_types(
        graph, "concat", {a_suffix_node.value()->output(0), zero_step}, concat_attrs);
    if (!a_next_node.is_ok()) return a_next_node.status();
    a_next = a_next_node.value()->output(0);
  }

  const Result<Node*> q_reversed_node = reverse_time(graph, q_node.value()->output(0), axis, steps);
  if (!q_reversed_node.is_ok()) return q_reversed_node.status();
  const Result<Node*> a_next_reversed_node = reverse_time(graph, a_next, axis, steps);
  if (!a_next_reversed_node.is_ok()) return a_next_reversed_node.status();
  const Result<Node*> lambda_reversed_node =
      create_node_with_inferred_types(graph, "selective_scan",
                                      {q_reversed_node.value()->output(0),
                                       a_next_reversed_node.value()->output(0), ones, ones, zeros});
  if (!lambda_reversed_node.is_ok()) return lambda_reversed_node.status();
  const Result<Node*> lambda_node =
      reverse_time(graph, lambda_reversed_node.value()->output(0), axis, steps);
  if (!lambda_node.is_ok()) return lambda_node.status();
  Value* lambda = lambda_node.value()->output(0);

  const Result<Node*> lambda_b_node = create_node_with_inferred_types(graph, "mul", {lambda, b});
  if (!lambda_b_node.is_ok()) return lambda_b_node.status();
  const Result<Node*> gy_d_node = create_node_with_inferred_types(graph, "mul", {gy_in.value(), d});
  if (!gy_d_node.is_ok()) return gy_d_node.status();
  const Result<Node*> gx_node = create_node_with_inferred_types(
      graph, "add", {lambda_b_node.value()->output(0), gy_d_node.value()->output(0)});
  if (!gx_node.is_ok()) return gx_node.status();
  const Result<Node*> ga_node = create_node_with_inferred_types(graph, "mul", {lambda, h_prev});
  if (!ga_node.is_ok()) return ga_node.status();
  const Result<Node*> gb_node = create_node_with_inferred_types(graph, "mul", {lambda, x});
  if (!gb_node.is_ok()) return gb_node.status();
  const Result<Node*> gc_node = create_node_with_inferred_types(graph, "mul", {gy_in.value(), h});
  if (!gc_node.is_ok()) return gc_node.status();
  const Result<Node*> gd_node = create_node_with_inferred_types(graph, "mul", {gy_in.value(), x});
  if (!gd_node.is_ok()) return gd_node.status();

  for (Node* gradient :
       {gx_node.value(), ga_node.value(), gb_node.value(), gc_node.value(), gd_node.value()}) {
    const Status mark_gradient = graph.mark_output(gradient, 0);
    if (!mark_gradient.is_ok()) return mark_gradient;
  }
  return graph;
}

// ---------------------------------------------------------------------------
// 梯度微图共用的小型构图 helper(softmax_gradient/layer_norm_gradient 共用,
// REUSE-002,本文件内部,不入公开 API)。
// ---------------------------------------------------------------------------

// 逐行均值:[N,D] -> [N](沿末轴 sum 再乘 1/D,keepdims 缺省 false)。
Result<Node*> reduce_mean_axis1(Graph& graph, Value* x, int64_t n, int64_t d, DType dtype,
                                Device device) {
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{1}}};
  const Result<Node*> sum_node = create_node_with_inferred_types(graph, "sum", {x}, sum_attrs);
  if (!sum_node.is_ok()) return sum_node.status();
  const Result<Node*> recip_node =
      make_constant_splat(graph, Shape({n}), dtype, device, 1.0 / static_cast<double>(d));
  if (!recip_node.is_ok()) return recip_node.status();
  return create_node_with_inferred_types(
      graph, "mul", {sum_node.value()->output(0), recip_node.value()->output(0)});
}

// bcast_col:逐行标量([N])沿列方向复制展开为 [N,D]
// (reshape[N]->[N,1] + matmul(_, ones[1,D]))。n(行数)/d(列数)相邻同型
// (int64_t),均取自调用方 x_shape.dim(0)/dim(1) 具名局部变量;误置换在
// 非方阵输入(N!=D,本文件全部梯度检查测试固定采用非方阵形状)下会被
// reshape 的 numel 守恒校验立即拦截,不会静默产出错误结果。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Result<Node*> broadcast_col(Graph& graph, Value* per_row, int64_t n, int64_t d, DType dtype,
                            Device device) {
  const AttrMap reshape_attrs{{"target_shape", Shape({n, 1})}};
  const Result<Node*> col_node =
      create_node_with_inferred_types(graph, "reshape", {per_row}, reshape_attrs);
  if (!col_node.is_ok()) return col_node.status();
  const Result<Node*> ones_node = make_constant_splat(graph, Shape({1, d}), dtype, device, 1.0);
  if (!ones_node.is_ok()) return ones_node.status();
  return create_node_with_inferred_types(
      graph, "matmul", {col_node.value()->output(0), ones_node.value()->output(0)});
}

// bcast_row:逐列向量([D])沿行方向复制展开为 [N,D]
// (reshape[D]->[1,D] + matmul(ones[N,1], _))。n/d 误置换风险论证同上方
// broadcast_col。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Result<Node*> broadcast_row(Graph& graph, Value* per_col, int64_t n, int64_t d, DType dtype,
                            Device device) {
  const AttrMap reshape_attrs{{"target_shape", Shape({1, d})}};
  const Result<Node*> row_node =
      create_node_with_inferred_types(graph, "reshape", {per_col}, reshape_attrs);
  if (!row_node.is_ok()) return row_node.status();
  const Result<Node*> ones_node = make_constant_splat(graph, Shape({n, 1}), dtype, device, 1.0);
  if (!ones_node.is_ok()) return ones_node.status();
  return create_node_with_inferred_types(
      graph, "matmul", {ones_node.value()->output(0), row_node.value()->output(0)});
}

// -v:constant(-1) splat + mul(v0 无 sub 算子,sigmoid_gradient 的
// constant(-1) 先例同机制)。
Result<Node*> negate(Graph& graph, Value* v, const Shape& shape, DType dtype, Device device) {
  const Result<Node*> neg_one_node = make_constant_splat(graph, shape, dtype, device, -1.0);
  if (!neg_one_node.is_ok()) return neg_one_node.status();
  return create_node_with_inferred_types(graph, "mul", {neg_one_node.value()->output(0), v});
}

// 减法:a-b = a + (-b)。a/b 是一对不可交换的减法操作数,相邻同型
// (Value*)存在误置换风险;调用点均以具名局部变量按数学式直接传入,误置换
// 不改变 shape/dtype(无法被 shape 校验拦截),但会使对应梯度符号反转,
// 在本文件配套的解析梯度 ≡ 数值微分测试(tests/cpp/ops/test_op_softmax.cpp、
// test_op_layer_norm.cpp)中会以数值不符立即失败,不会静默放行。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Result<Node*> subtract(Graph& graph, Value* a, Value* b, const Shape& shape, DType dtype,
                       Device device) {
  const Result<Node*> neg_b_node = negate(graph, b, shape, dtype, device);
  if (!neg_b_node.is_ok()) return neg_b_node.status();
  return create_node_with_inferred_types(graph, "add", {a, neg_b_node.value()->output(0)});
}

// ---------------------------------------------------------------------------
// softmax(x[N,D]):限 rank-2,末轴,无属性。kernel 内减行 max 数值稳定(kernel
// 层语义,不需要 max 归约算子)。
// ---------------------------------------------------------------------------

Result<std::vector<Shape>> infer_softmax_shape(const NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument, "op 'softmax' expects 1 input, got " +
                                                         std::to_string(ctx.input_types.size()));
  }
  const Shape& x_shape = ctx.input_types[0].shape;
  if (x_shape.rank() != 2) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "op 'softmax' requires x to be rank-2 [N, D], got rank " + std::to_string(x_shape.rank()));
  }
  return std::vector<Shape>{x_shape};
}

// softmax(x) 的梯度(§1.2 表):t=softmax(x) 重算;s=sum(t·gy, axes=[1]);
// gx = t·(gy − bcast_col(s))。
Result<Graph> softmax_gradient(const NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument, "op 'softmax' gradient expects 1 input, got " +
                                                         std::to_string(ctx.input_types.size()));
  }
  Graph graph("softmax_gradient");

  const Result<Value*> x_in = graph.add_graph_input(ctx.input_types[0]);
  if (!x_in.is_ok()) return x_in.status();

  const Result<std::vector<Shape>> y_shapes = infer_softmax_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const Result<Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const Result<Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();

  const Shape& x_shape = ctx.input_types[0].shape;
  const int64_t n = x_shape.dim(0);
  const int64_t d = x_shape.dim(1);
  const DType dtype = ctx.input_types[0].dtype;
  const Device device = ctx.input_types[0].device;

  // t = softmax(x) 重算(前向节点 CSE 候选,同 sigmoid_gradient 从 x 重算 s
  // 的先例)。
  const Result<Node*> t_node = create_node_with_inferred_types(graph, "softmax", {x_in.value()});
  if (!t_node.is_ok()) return t_node.status();
  Value* t = t_node.value()->output(0);

  // s = sum(t·gy, axes=[1]),shape [N](keepdims 缺省 false)。
  const Result<Node*> tg_node = create_node_with_inferred_types(graph, "mul", {t, gy_in.value()});
  if (!tg_node.is_ok()) return tg_node.status();
  const AttrMap s_attrs{{"axes", std::vector<int64_t>{1}}};
  const Result<Node*> s_node =
      create_node_with_inferred_types(graph, "sum", {tg_node.value()->output(0)}, s_attrs);
  if (!s_node.is_ok()) return s_node.status();

  // 行广播 bcast_col(s):[N] -> [N,D]。
  const Result<Node*> s_bcast_node =
      broadcast_col(graph, s_node.value()->output(0), n, d, dtype, device);
  if (!s_bcast_node.is_ok()) return s_bcast_node.status();

  // 差值:gy - bcast_col(s)。
  const Result<Node*> diff_node =
      subtract(graph, gy_in.value(), s_bcast_node.value()->output(0), x_shape, dtype, device);
  if (!diff_node.is_ok()) return diff_node.status();

  // 最终梯度:gx = t·(gy - bcast_col(s))。
  const Result<Node*> gx_node =
      create_node_with_inferred_types(graph, "mul", {t, diff_node.value()->output(0)});
  if (!gx_node.is_ok()) return gx_node.status();

  const Status mark_gx = graph.mark_output(gx_node.value(), 0);
  if (!mark_gx.is_ok()) return mark_gx;

  return graph;
}

// ---------------------------------------------------------------------------
// layer_norm(x[N,D], gamma[D], beta[D]; eps):行归一化 + 仿射,gamma/beta 在
// 算子内沿行广播(conv bias [Cout] 算子内广播的裁决点①先例)。
// ---------------------------------------------------------------------------

Result<std::vector<Shape>> infer_layer_norm_shape(const NodeContext& ctx) {
  if (ctx.input_types.size() != 3) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'layer_norm' expects 3 inputs (x, gamma, beta), got " +
                            std::to_string(ctx.input_types.size()));
  }
  const frame::ir::TensorType& x = ctx.input_types[0];
  const frame::ir::TensorType& gamma = ctx.input_types[1];
  const frame::ir::TensorType& beta = ctx.input_types[2];

  if (x.shape.rank() != 2) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'layer_norm' requires x to be rank-2 [N, D], got rank " +
                            std::to_string(x.shape.rank()));
  }
  const int64_t d = x.shape.dim(1);

  if (gamma.shape.rank() != 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'layer_norm' requires gamma to be rank-1 [D], got rank " +
                            std::to_string(gamma.shape.rank()));
  }
  if (gamma.shape.dim(0) != d) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'layer_norm' requires gamma size to equal D=" + std::to_string(d) +
                            ", got " + std::to_string(gamma.shape.dim(0)));
  }
  if (!(gamma.dtype == x.dtype)) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'layer_norm' requires gamma and x of the same dtype, got '" +
                            std::string(gamma.dtype.name()) + "' and '" +
                            std::string(x.dtype.name()) + "'");
  }

  if (beta.shape.rank() != 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'layer_norm' requires beta to be rank-1 [D], got rank " +
                            std::to_string(beta.shape.rank()));
  }
  if (beta.shape.dim(0) != d) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'layer_norm' requires beta size to equal D=" + std::to_string(d) +
                            ", got " + std::to_string(beta.shape.dim(0)));
  }
  if (!(beta.dtype == x.dtype)) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'layer_norm' requires beta and x of the same dtype, got '" +
                            std::string(beta.dtype.name()) + "' and '" +
                            std::string(x.dtype.name()) + "'");
  }

  const double* eps = ctx.attr<double>("eps");
  if (eps == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'layer_norm' is missing required attribute 'eps'");
  }
  if (!(*eps > 0.0)) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "op 'layer_norm' attribute 'eps' must be positive, got " + std::to_string(*eps));
  }

  return std::vector<Shape>{x.shape};
}

// layer_norm(x,gamma,beta) 的梯度(§1.2 表)。从 x 重算 μ、σ²、r=rsqrt(σ²+eps)、
// x̂;a=bcast_row(gamma)·gy;
// gx = bcast_col(r)·(a − bcast_col(mean_row(a)) − x̂·bcast_col(mean_row(a·x̂)));
// ggamma = sum(gy·x̂, axes=[0]);gbeta = sum(gy, axes=[0])。微图节点较多
// (约 48,含 broadcast/mean 三连展开),按块注释每个中间量对应的数学量。
Result<Graph> layer_norm_gradient(const NodeContext& ctx) {
  if (ctx.input_types.size() != 3) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'layer_norm' gradient expects 3 inputs (x, gamma, beta), got " +
                            std::to_string(ctx.input_types.size()));
  }
  Graph graph("layer_norm_gradient");

  const Result<Value*> x_in = graph.add_graph_input(ctx.input_types[0]);
  if (!x_in.is_ok()) return x_in.status();
  const Result<Value*> gamma_in = graph.add_graph_input(ctx.input_types[1]);
  if (!gamma_in.is_ok()) return gamma_in.status();
  const Result<Value*> beta_in = graph.add_graph_input(ctx.input_types[2]);
  if (!beta_in.is_ok()) return beta_in.status();

  const Result<std::vector<Shape>> y_shapes = infer_layer_norm_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const Result<Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const Result<Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();
  (void)beta_in;  // beta 自身不参与反向重算路径,仅占位声明(ARCH-063)

  const Shape& x_shape = ctx.input_types[0].shape;
  const int64_t n = x_shape.dim(0);
  const int64_t d = x_shape.dim(1);
  const DType dtype = ctx.input_types[0].dtype;
  const Device device = ctx.input_types[0].device;
  const double eps = *ctx.attr<double>("eps");

  // --- 行均值 μ = mean_row(x),[N];行广播 bcast_col(μ),[N,D];
  // 去均值 centered = x − bcast(μ) ---
  const Result<Node*> mu_node = reduce_mean_axis1(graph, x_in.value(), n, d, dtype, device);
  if (!mu_node.is_ok()) return mu_node.status();
  const Result<Node*> mu_bcast_node =
      broadcast_col(graph, mu_node.value()->output(0), n, d, dtype, device);
  if (!mu_bcast_node.is_ok()) return mu_bcast_node.status();
  const Result<Node*> centered_node =
      subtract(graph, x_in.value(), mu_bcast_node.value()->output(0), x_shape, dtype, device);
  if (!centered_node.is_ok()) return centered_node.status();

  // --- 行方差 σ² = mean_row(centered²),[N];r = rsqrt(σ²+eps),[N] ---
  const Result<Node*> sq_node = create_node_with_inferred_types(
      graph, "mul", {centered_node.value()->output(0), centered_node.value()->output(0)});
  if (!sq_node.is_ok()) return sq_node.status();
  const Result<Node*> var_node =
      reduce_mean_axis1(graph, sq_node.value()->output(0), n, d, dtype, device);
  if (!var_node.is_ok()) return var_node.status();
  const Result<Node*> eps_splat_node = make_constant_splat(graph, Shape({n}), dtype, device, eps);
  if (!eps_splat_node.is_ok()) return eps_splat_node.status();
  const Result<Node*> var_eps_node = create_node_with_inferred_types(
      graph, "add", {var_node.value()->output(0), eps_splat_node.value()->output(0)});
  if (!var_eps_node.is_ok()) return var_eps_node.status();
  const Result<Node*> r_node =
      create_node_with_inferred_types(graph, "rsqrt", {var_eps_node.value()->output(0)});
  if (!r_node.is_ok()) return r_node.status();

  // --- 行广播 bcast_col(r),[N,D];归一化 x̂ = centered·bcast_col(r),[N,D] ---
  const Result<Node*> r_bcast_node =
      broadcast_col(graph, r_node.value()->output(0), n, d, dtype, device);
  if (!r_bcast_node.is_ok()) return r_bcast_node.status();
  const Result<Node*> xhat_node = create_node_with_inferred_types(
      graph, "mul", {centered_node.value()->output(0), r_bcast_node.value()->output(0)});
  if (!xhat_node.is_ok()) return xhat_node.status();

  // --- 列广播 a = bcast_row(gamma)·gy,[N,D] ---
  const Result<Node*> gamma_bcast_node =
      broadcast_row(graph, gamma_in.value(), n, d, dtype, device);
  if (!gamma_bcast_node.is_ok()) return gamma_bcast_node.status();
  const Result<Node*> a_node = create_node_with_inferred_types(
      graph, "mul", {gamma_bcast_node.value()->output(0), gy_in.value()});
  if (!a_node.is_ok()) return a_node.status();

  // --- 行均值 mean_row(a),[N];行广播 bcast_col(mean_row(a)),[N,D] ---
  const Result<Node*> mean_a_node =
      reduce_mean_axis1(graph, a_node.value()->output(0), n, d, dtype, device);
  if (!mean_a_node.is_ok()) return mean_a_node.status();
  const Result<Node*> mean_a_bcast_node =
      broadcast_col(graph, mean_a_node.value()->output(0), n, d, dtype, device);
  if (!mean_a_bcast_node.is_ok()) return mean_a_bcast_node.status();

  // --- 乘积 a·x̂,[N,D];行均值 mean_row(a·x̂),[N];
  // 行广播 bcast_col(mean_row(a·x̂)),[N,D] ---
  const Result<Node*> a_xhat_node = create_node_with_inferred_types(
      graph, "mul", {a_node.value()->output(0), xhat_node.value()->output(0)});
  if (!a_xhat_node.is_ok()) return a_xhat_node.status();
  const Result<Node*> mean_a_xhat_node =
      reduce_mean_axis1(graph, a_xhat_node.value()->output(0), n, d, dtype, device);
  if (!mean_a_xhat_node.is_ok()) return mean_a_xhat_node.status();
  const Result<Node*> mean_a_xhat_bcast_node =
      broadcast_col(graph, mean_a_xhat_node.value()->output(0), n, d, dtype, device);
  if (!mean_a_xhat_bcast_node.is_ok()) return mean_a_xhat_bcast_node.status();

  // --- 最终梯度 gx = bcast_col(r)·(a − bcast(mean_row(a)) − x̂·bcast(mean_row(a·x̂))) ---
  const Result<Node*> term1_node =
      subtract(graph, a_node.value()->output(0), mean_a_bcast_node.value()->output(0), x_shape,
               dtype, device);
  if (!term1_node.is_ok()) return term1_node.status();
  const Result<Node*> xhat_mean_axhat_node = create_node_with_inferred_types(
      graph, "mul", {xhat_node.value()->output(0), mean_a_xhat_bcast_node.value()->output(0)});
  if (!xhat_mean_axhat_node.is_ok()) return xhat_mean_axhat_node.status();
  const Result<Node*> term_node =
      subtract(graph, term1_node.value()->output(0), xhat_mean_axhat_node.value()->output(0),
               x_shape, dtype, device);
  if (!term_node.is_ok()) return term_node.status();
  const Result<Node*> gx_node = create_node_with_inferred_types(
      graph, "mul", {r_bcast_node.value()->output(0), term_node.value()->output(0)});
  if (!gx_node.is_ok()) return gx_node.status();
  const Status mark_gx = graph.mark_output(gx_node.value(), 0);
  if (!mark_gx.is_ok()) return mark_gx;

  // --- gamma 梯度 ggamma = sum(gy·x̂, axes=[0]),[D] ---
  const Result<Node*> gy_xhat_node =
      create_node_with_inferred_types(graph, "mul", {gy_in.value(), xhat_node.value()->output(0)});
  if (!gy_xhat_node.is_ok()) return gy_xhat_node.status();
  const AttrMap ggamma_attrs{{"axes", std::vector<int64_t>{0}}};
  const Result<Node*> ggamma_node = create_node_with_inferred_types(
      graph, "sum", {gy_xhat_node.value()->output(0)}, ggamma_attrs);
  if (!ggamma_node.is_ok()) return ggamma_node.status();
  const Status mark_ggamma = graph.mark_output(ggamma_node.value(), 0);
  if (!mark_ggamma.is_ok()) return mark_ggamma;

  // --- beta 梯度 gbeta = sum(gy, axes=[0]),[D] ---
  const AttrMap gbeta_attrs{{"axes", std::vector<int64_t>{0}}};
  const Result<Node*> gbeta_node =
      create_node_with_inferred_types(graph, "sum", {gy_in.value()}, gbeta_attrs);
  if (!gbeta_node.is_ok()) return gbeta_node.status();
  const Status mark_gbeta = graph.mark_output(gbeta_node.value(), 0);
  if (!mark_gbeta.is_ok()) return mark_gbeta;

  return graph;
}

}  // namespace

// selective_scan(x,a,b,c,d):最后一轴状态递推,无属性、无 traits。
FRAME_REGISTER_OP("selective_scan")
    .input("x", "input sequence, rank >= 1 with time on the last axis")
    .input("a", "per-step recurrent multiplier, same shape and dtype as x")
    .input("b", "per-step input multiplier, same shape and dtype as x")
    .input("c", "per-step state output multiplier, same shape and dtype as x")
    .input("d", "per-step direct output multiplier, same shape and dtype as x")
    .output("y", "selective scan output, same shape and dtype as x")
    .shape_infer(&infer_selective_scan_shape)
    .gradient(&selective_scan_gradient);

// softmax(x[N,D]):末轴 softmax,无属性。
FRAME_REGISTER_OP("softmax")
    .input("x", "input tensor, rank-2 [N, D]")
    .output("out", "softmax of x along the last axis, rank-2 [N, D]")
    .shape_infer(&infer_softmax_shape)
    .gradient(&softmax_gradient);

// layer_norm(x[N,D], gamma[D], beta[D]; eps):行归一化 + 仿射。
FRAME_REGISTER_OP("layer_norm")
    .input("x", "input tensor, rank-2 [N, D]")
    .input("gamma", "per-feature scale, rank-1 [D], broadcast across rows")
    .input("beta", "per-feature shift, rank-1 [D], broadcast across rows")
    .attr("eps", frame::ir::AttrType::kDouble, /*required=*/true)
    .output("out", "row-wise layer normalization of x with affine (gamma, beta), rank-2 [N, D]")
    .shape_infer(&infer_layer_norm_shape)
    .gradient(&layer_norm_gradient);
