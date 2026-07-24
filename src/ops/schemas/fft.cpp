// 内置频域算子 schema 注册桩(M23,批5 T3,§1.1–1.3 决议点A–C):rfft/irfft。
// 设计依据:docs/plan/2026-07-21-batch5-m23-fft.md(design-reviewer APPROVE)。
// 复数以打包实数张量承载(决议点A):末轴 2 = (re, im) 交错,与各后端 FFT 库
// 原生复数类型逐字节一致(布局论证见 ADR-0022),不引入 complex dtype、不扩展
// 多输出机制(全仓算子维持恒单输出不变式)。
//
// 签名(决议点B):
//   rfft(x[...,n])       -> out[...,k,2]   k=n/2+1(整除),rank>=1,n>=2,无属性
//   irfft(z[...,k,2]; n) -> out[...,n]     校验末轴==2 且 k==n/2+1
// dtype 限 fp32(CPU 参考库无半精度支持,GPU 库半精度须另开专用 API 面,
// YAGNI,见 ADR-0022);shape_infer 本身不做 dtype 限定(与 layer_norm/
// softmax 同口径),fp32-only 校验下沉至 kernel 层 fail-loud(M22 白名单
// 先例),见 src/backends/cpu/kernels/fft.cpp。
//
// 梯度微图(决议点C,w 为长度 k 的 Hermitian 重数掩码:w_0=1;n 偶时
// w_{k-1}=1、其余=2;n 奇时除 w_0 外全=2):
//   rfft 梯度:gx = n · irfft(gy ⊙ (1/w), n)
//   irfft 梯度:gz = (1/n) · (w ⊙ rfft(gx)),gx 即 irfft 输出端入梯度(spec §1.3 记号)
// 掩码沿 (k,2) 末两轴同值展开、直接构造与 gy/z 同形的 constant 全形数组
// (kDoubleArray 编码;§1.3:谱网络小形状优先直接 constant 全形数组,微图不含
// matmul,二阶微图更干净,设计门建议 2)。标量缩放 n、1/n 经 constant splat +
// mul(schema_math.h::make_constant_splat)。R11 闭包:rfft 微图用
// {irfft, constant, mul},irfft 微图用 {rfft, constant, mul},互引用在注册层
// 合法,全部已注册且自身可微,无新增内部算子。cpu kernel 见
// src/backends/cpu/kernels/fft.cpp。

#include <cstdint>
#include <string>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/ir/attribute.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/constant_utils.h>
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
using frame::ops::kConstantOpName;
using frame::ops::NodeContext;

// 构造 shape 全形常量节点:同目录共享工具(铁律 5,见 schema_math.h 头注释)。
using frame::ops::schemas::make_constant_splat;

// ---------------------------------------------------------------------------
// rfft(x[...,n]) 的 shape 推断:rank>=1、末轴 n>=2;输出 [...,k,2],
// k=n/2+1(整除,决议点B)。无属性。
// ---------------------------------------------------------------------------

Result<std::vector<Shape>> infer_rfft_shape(const NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'rfft' expects 1 input, got " + std::to_string(ctx.input_types.size()));
  }
  const frame::ir::TensorType& x = ctx.input_types[0];
  const int64_t rank = x.shape.rank();
  if (rank < 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'rfft' requires x to have rank >= 1, got rank " + std::to_string(rank));
  }
  const int64_t n = x.shape.dim(rank - 1);
  if (n < 2) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "op 'rfft' requires the last dimension (n) to be >= 2, got n=" + std::to_string(n));
  }
  const int64_t k = n / 2 + 1;

  std::vector<int64_t> out_dims = x.shape.dims();
  out_dims[static_cast<size_t>(rank - 1)] = k;
  out_dims.push_back(2);
  return std::vector<Shape>{Shape(out_dims)};
}

// ---------------------------------------------------------------------------
// irfft(z[...,k,2]; n) 的 shape 推断:rank>=2、末轴==2、k==n/2+1;输出
// [...,n]。属性 n(kInt64,必需)。
// ---------------------------------------------------------------------------

Result<std::vector<Shape>> infer_irfft_shape(const NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument, "op 'irfft' expects 1 input, got " +
                                                         std::to_string(ctx.input_types.size()));
  }
  const frame::ir::TensorType& z = ctx.input_types[0];
  const int64_t rank = z.shape.rank();
  if (rank < 2) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "op 'irfft' requires z to have rank >= 2 (trailing axes are [k, 2]), got rank " +
            std::to_string(rank));
  }
  const int64_t last_dim = z.shape.dim(rank - 1);
  if (last_dim != 2) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'irfft' requires the last dimension to be 2 (interleaved re/im), "
                        "got " +
                            std::to_string(last_dim));
  }

  const int64_t* n_ptr = ctx.attr<int64_t>("n");
  if (n_ptr == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'irfft' is missing required attribute 'n' (int64)");
  }
  const int64_t n = *n_ptr;
  const int64_t k = z.shape.dim(rank - 2);
  const int64_t expected_k = n / 2 + 1;
  if (k != expected_k) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'irfft' requires k=n/2+1 for attribute n=" + std::to_string(n) +
                            ", expected k=" + std::to_string(expected_k) +
                            ", got k=" + std::to_string(k));
  }

  std::vector<int64_t> out_dims(z.shape.dims().begin(), z.shape.dims().end() - 1);
  out_dims.back() = n;
  return std::vector<Shape>{Shape(out_dims)};
}

// ---------------------------------------------------------------------------
// 梯度微图共用:Hermitian 重数掩码 w(长度 k,决议点C)。
// ---------------------------------------------------------------------------

// w_0=1;n 偶时 w_{k-1}=1、其余(含索引 0 已单独覆盖)=2;n 奇时除 w_0 外全=2。
// reciprocal=true 返回逐元素倒数 1/w(rfft 梯度用)。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::vector<double> hermitian_multiplicity_mask(int64_t n, int64_t k, bool reciprocal) {
  std::vector<double> w(static_cast<size_t>(k), 2.0);
  w[0] = 1.0;
  if (n % 2 == 0) {
    w[static_cast<size_t>(k - 1)] = 1.0;
  }
  if (reciprocal) {
    for (double& value : w) value = 1.0 / value;
  }
  return w;
}

// 把长度 k 的掩码 w 沿 (k,2) 末两轴同值展开、并复制到 full_shape 的全部
// leading 维,构造与 full_shape 同形的 constant 节点(§1.3:谱网络小形状优先
// 直接 constant 全形数组,微图不含 matmul,二阶微图更干净,设计门建议 2)。
// full_shape 末两轴须为 [k, 2](调用方保证,k 由 w.size() 给出)。
Result<Node*> make_hermitian_mask_constant(Graph& graph, const Shape& full_shape,
                                           const std::vector<double>& w, DType dtype,
                                           Device device) {
  const int64_t k = static_cast<int64_t>(w.size());
  const int64_t numel = full_shape.numel();
  const int64_t leading_count = numel / (k * 2);
  std::vector<double> values;
  values.reserve(static_cast<size_t>(numel));
  for (int64_t b = 0; b < leading_count; ++b) {
    for (int64_t idx = 0; idx < k; ++idx) {
      values.push_back(w[static_cast<size_t>(idx)]);
      values.push_back(w[static_cast<size_t>(idx)]);
    }
  }
  const AttrMap attrs{
      {"value", values},
      {"shape", full_shape},
      {"dtype", dtype},
  };
  return create_node_with_inferred_types(graph, kConstantOpName, device, attrs);
}

// ---------------------------------------------------------------------------
// rfft(x) 的梯度(决议点C):gx = n · irfft(gy ⊙ (1/w), n)。
// ---------------------------------------------------------------------------

Result<Graph> rfft_gradient(const NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument, "op 'rfft' gradient expects 1 input, got " +
                                                         std::to_string(ctx.input_types.size()));
  }
  Graph graph("rfft_gradient");

  const Result<Value*> x_in = graph.add_graph_input(ctx.input_types[0]);
  if (!x_in.is_ok()) return x_in.status();

  const Result<std::vector<Shape>> y_shapes = infer_rfft_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const Result<Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const Result<Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();
  (void)x_in;  // x 自身不参与反向重算路径,仅占位声明(ARCH-063)

  const Shape& x_shape = ctx.input_types[0].shape;
  const int64_t rank = x_shape.rank();
  const int64_t n = x_shape.dim(rank - 1);
  const int64_t k = n / 2 + 1;
  const DType dtype = ctx.input_types[0].dtype;
  const Device device = ctx.input_types[0].device;

  // gy ⊙ (1/w),shape 与 gy 相同 [...,k,2]。
  const std::vector<double> recip_w = hermitian_multiplicity_mask(n, k, /*reciprocal=*/true);
  const Result<Node*> recip_w_node =
      make_hermitian_mask_constant(graph, y_type.shape, recip_w, dtype, device);
  if (!recip_w_node.is_ok()) return recip_w_node.status();
  const Result<Node*> scaled_gy_node = create_node_with_inferred_types(
      graph, "mul", {gy_in.value(), recip_w_node.value()->output(0)});
  if (!scaled_gy_node.is_ok()) return scaled_gy_node.status();

  // 对缩放后的梯度做逆变换 irfft(gy ⊙ (1/w), n),shape 与 x_shape 相同 [...,n]。
  const AttrMap irfft_attrs{{"n", n}};
  const Result<Node*> irfft_node = create_node_with_inferred_types(
      graph, "irfft", {scaled_gy_node.value()->output(0)}, irfft_attrs);
  if (!irfft_node.is_ok()) return irfft_node.status();

  // 最终梯度:标量 n 乘以上一步的逆变换结果。
  const Result<Node*> n_splat_node =
      make_constant_splat(graph, x_shape, dtype, device, static_cast<double>(n));
  if (!n_splat_node.is_ok()) return n_splat_node.status();
  const Result<Node*> gx_node = create_node_with_inferred_types(
      graph, "mul", {n_splat_node.value()->output(0), irfft_node.value()->output(0)});
  if (!gx_node.is_ok()) return gx_node.status();

  const Status mark_gx = graph.mark_output(gx_node.value(), 0);
  if (!mark_gx.is_ok()) return mark_gx;

  return graph;
}

// ---------------------------------------------------------------------------
// irfft(z;n) 的梯度(决议点C):gz = (1/n) · (w ⊙ rfft(gy))。
// ---------------------------------------------------------------------------

Result<Graph> irfft_gradient(const NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument, "op 'irfft' gradient expects 1 input, got " +
                                                         std::to_string(ctx.input_types.size()));
  }
  Graph graph("irfft_gradient");

  const Result<Value*> z_in = graph.add_graph_input(ctx.input_types[0]);
  if (!z_in.is_ok()) return z_in.status();

  const Result<std::vector<Shape>> y_shapes = infer_irfft_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const Result<Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const Result<Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();
  (void)z_in;  // z 自身不参与反向重算路径,仅占位声明(ARCH-063)

  const Shape& z_shape = ctx.input_types[0].shape;
  const int64_t rank = z_shape.rank();
  const int64_t k = z_shape.dim(rank - 2);
  const int64_t n = *ctx.attr<int64_t>("n");
  const DType dtype = ctx.input_types[0].dtype;
  const Device device = ctx.input_types[0].device;

  // rfft(gy),gy shape [...,n],输出 shape 与 z 相同 [...,k,2]。
  const Result<Node*> rfft_node = create_node_with_inferred_types(graph, "rfft", {gy_in.value()});
  if (!rfft_node.is_ok()) return rfft_node.status();

  // 用掩码 w 逐元素加权 rfft(gy) 的结果。
  const std::vector<double> w = hermitian_multiplicity_mask(n, k, /*reciprocal=*/false);
  const Result<Node*> w_node = make_hermitian_mask_constant(graph, z_shape, w, dtype, device);
  if (!w_node.is_ok()) return w_node.status();
  const Result<Node*> weighted_node = create_node_with_inferred_types(
      graph, "mul", {rfft_node.value()->output(0), w_node.value()->output(0)});
  if (!weighted_node.is_ok()) return weighted_node.status();

  // 最终梯度:标量 1/n 乘以上一步的加权结果。
  const Result<Node*> recip_n_splat_node =
      make_constant_splat(graph, z_shape, dtype, device, 1.0 / static_cast<double>(n));
  if (!recip_n_splat_node.is_ok()) return recip_n_splat_node.status();
  const Result<Node*> gz_node = create_node_with_inferred_types(
      graph, "mul", {recip_n_splat_node.value()->output(0), weighted_node.value()->output(0)});
  if (!gz_node.is_ok()) return gz_node.status();

  const Status mark_gz = graph.mark_output(gz_node.value(), 0);
  if (!mark_gz.is_ok()) return mark_gz;

  return graph;
}

}  // namespace

// rfft(x[...,n]):末轴快速傅里叶变换(实数输入),打包复数输出。无属性,不
// 归一化(numpy 口径)。
FRAME_REGISTER_OP("rfft")
    .input("x", "real input tensor, rank >= 1, last dimension n >= 2")
    .output("out", "packed complex spectrum, shape [..., k, 2] (interleaved re/im), k=n/2+1")
    .shape_infer(&infer_rfft_shape)
    .gradient(&rfft_gradient);

// irfft(z[...,k,2]; n):rfft 的逆变换,归一化 1/n(numpy 口径,
// irfft(rfft(x), n)≡x)。
FRAME_REGISTER_OP("irfft")
    .input("z", "packed complex spectrum, shape [..., k, 2] (interleaved re/im)")
    .attr("n", frame::ir::AttrType::kInt64, /*required=*/true)
    .output("out", "real output tensor, shape [..., n]")
    .shape_infer(&infer_irfft_shape)
    .gradient(&irfft_gradient);
