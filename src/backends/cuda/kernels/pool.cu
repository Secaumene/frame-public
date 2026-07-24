// CUDA 池化梯度自写内核(M21,批3 T5,ADR-0021 决策 2/裁决点③):
// max_pool2d_select_internal(g,x)->out 与 avg_pool2d_grad_internal(dy)->dx
// 无 cuDNN legacy API 对应物(前者需按 argmax 取值,cuDNN 无此原语;后者单
// 输入 dy 无法喂给 cudnnPoolingBackward,该函数强制要求 y/x 实参),故自写
// __global__ kernel(铁律 5:优先复用 cuDNN 已在 cudnnPoolingForward/
// max_pool2d_grad_internal 处使用,仅这两个确无对应物的算子才自写,见
// pool.cpp 头注释与计划 1.3 节)。
//
// argmax 平局约定(与 src/backends/cpu/kernels/pool.cpp、
// src/ops/schemas/pool.cpp 头注释一致,严格伴随性依赖此约定):窗口内以 kh
// 外层、kw 内层的行优先顺序遍历,严格 `>` 比较——取窗口内最低线性索引
// kh*KW+kw。逐输出位(select)/逐输入位(avg grad)一线程一元素,无原子操作
// (select 为 gather 一对一;avg grad 按输入位聚合覆盖窗口,同一线程独占写
// 自己的输出位)。

#include <cstdint>
#include <string>
#include <string_view>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/stream.h>
#include <frame/ops/kernel_registry.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "../cuda_status.h"
#include "accum_load_store.cuh"
#include "launch_config.cuh"
#include "pool_geometry.h"

namespace {

using frame::backends::cuda::Pool2dRuntimeParams;
using frame::backends::cuda::read_pool2d_runtime_params;
using frame::backends::cuda::read_shape_attr;

// device 端位型 <-> float 转换:同目录共享工具(铁律 5 收敛,M22 批4 判重,见
// accum_load_store.cuh 头注释)。
using frame::backends::cuda::elementwise_load;
using frame::backends::cuda::elementwise_store;

// max_pool2d_select_internal(g,x)->out:每线程处理 1 个输出位置(n,c,oh,ow),
// 以 x 重算窗口 argmax(平局取最低线性索引),取 g 在该位置的值。
template <typename T>
__global__ void max_pool2d_select_kernel(const T* g, const T* x, T* out, Pool2dRuntimeParams p) {
  const int64_t total = p.n * p.c * p.out_h * p.out_w;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  // 线性 idx -> (n,c,oh,ow),对应 [N,C,out_h,out_w] 行优先布局。
  int64_t rem = idx;
  const int64_t ow = rem % p.out_w;
  rem /= p.out_w;
  const int64_t oh = rem % p.out_h;
  rem /= p.out_h;
  const int64_t c = rem % p.c;
  rem /= p.c;
  const int64_t n = rem;

  bool found = false;
  float best = 0.0F;
  int64_t best_ih = 0;
  int64_t best_iw = 0;
  for (int64_t kh_i = 0; kh_i < p.kh; ++kh_i) {
    const int64_t ih = oh * p.stride_h - p.pad_h + kh_i;
    if (ih < 0 || ih >= p.h) continue;
    for (int64_t kw_i = 0; kw_i < p.kw; ++kw_i) {
      const int64_t iw = ow * p.stride_w - p.pad_w + kw_i;
      if (iw < 0 || iw >= p.w) continue;
      const int64_t x_idx = (n * p.c + c) * p.h * p.w + ih * p.w + iw;
      const float value = elementwise_load(x, x_idx);
      if (!found || value > best) {
        found = true;
        best = value;
        best_ih = ih;
        best_iw = iw;
      }
    }
  }
  // found 恒为 true(schema 侧 padding*2<=kernel 校验保证每个窗口至少覆盖一个
  // 有效位置,src/ops/schemas/pool.cpp::compute_pool2d_geometry);device 端
  // kernel 无法像 cpu kernel 那样以 Status 报告该防御性分支,故越界即跳过写入
  // (理论不可达,不产出垃圾值)。
  if (!found) return;
  const int64_t g_idx = (n * p.c + c) * p.h * p.w + best_ih * p.w + best_iw;
  const int64_t out_idx = (n * p.c + c) * p.out_h * p.out_w + oh * p.out_w + ow;
  elementwise_store(out, out_idx, elementwise_load(g, g_idx));
}

// avg_pool2d_grad_internal(dy)->dx:每线程处理 1 个输入位置(n,c,ih,iw),枚举
// 全部满足 (ih+pad-kh)%stride==0 的覆盖窗口求和 dy/denom(gather 风格,无
// 原子操作)。
template <typename T>
__global__ void avg_pool2d_grad_kernel(const T* dy, T* dx, Pool2dRuntimeParams p) {
  const int64_t total = p.n * p.c * p.h * p.w;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  int64_t rem = idx;
  const int64_t iw = rem % p.w;
  rem /= p.w;
  const int64_t ih = rem % p.h;
  rem /= p.h;
  const int64_t c = rem % p.c;
  rem /= p.c;
  const int64_t n = rem;

  const float denom = static_cast<float>(p.kh * p.kw);
  float accum = 0.0F;
  for (int64_t kh_i = 0; kh_i < p.kh; ++kh_i) {
    const int64_t num_h = ih + p.pad_h - kh_i;
    if (num_h % p.stride_h != 0) continue;
    const int64_t oh = num_h / p.stride_h;
    if (oh < 0 || oh >= p.out_h) continue;
    for (int64_t kw_i = 0; kw_i < p.kw; ++kw_i) {
      const int64_t num_w = iw + p.pad_w - kw_i;
      if (num_w % p.stride_w != 0) continue;
      const int64_t ow = num_w / p.stride_w;
      if (ow < 0 || ow >= p.out_w) continue;
      const int64_t dy_idx = (n * p.c + c) * p.out_h * p.out_w + oh * p.out_w + ow;
      accum += elementwise_load(dy, dy_idx);
    }
  }
  const int64_t dx_idx = (n * p.c + c) * p.h * p.w + ih * p.w + iw;
  elementwise_store(dx, dx_idx, accum / denom);
}

// launch 配置计算:同目录共享工具(铁律 5 收敛,M22 批4 判重,见
// launch_config.cuh 头注释)。
using frame::backends::cuda::compute_launch_config;
using frame::backends::cuda::LaunchConfig;

cudaStream_t native_stream(frame::hal::Stream* stream) {
  return stream != nullptr ? static_cast<cudaStream_t>(stream->native_handle()) : nullptr;
}

bool is_supported_dtype(frame::DTypeCode code) {
  return code == frame::DTypeCode::kFloat32 || code == frame::DTypeCode::kFloat16 ||
         code == frame::DTypeCode::kBFloat16;
}

frame::Status require_rank(std::string_view op_name, std::string_view operand_label,
                           int64_t expected_rank, const frame::Tensor& tensor) {
  if (tensor.shape().rank() != expected_rank) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' cuda kernel requires " +
                                   std::string(operand_label) + " to be rank-" +
                                   std::to_string(expected_rank) + ", got rank " +
                                   std::to_string(tensor.shape().rank()));
  }
  return frame::Status::ok();
}

// max_pool2d_select_internal(g,x)->out:2 输入(g,x,同 shape),attrs=
// kernel/stride/padding(几何取自 x.shape,无 input_shape——x 本身即输入
// shape,同 cpu kernel 先例)。
frame::Status max_pool2d_select_internal_cuda_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'max_pool2d_select_internal' cuda kernel expects 2 inputs, got " +
            std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'max_pool2d_select_internal' cuda kernel expects 1 output, got " +
            std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& g = ctx.inputs[0];
  const frame::Tensor& x = ctx.inputs[1];
  frame::Tensor& out = ctx.outputs[0];

  frame::Status rank_status = require_rank("max_pool2d_select_internal", "g", 4, g);
  if (!rank_status.is_ok()) return rank_status;
  rank_status = require_rank("max_pool2d_select_internal", "x", 4, x);
  if (!rank_status.is_ok()) return rank_status;
  if (!(g.shape() == x.shape())) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'max_pool2d_select_internal' cuda kernel requires g and x of "
                               "the same shape, got " +
                                   g.shape().to_string() + " and " + x.shape().to_string());
  }
  const bool elem_type_mismatch = !(g.dtype() == x.dtype()) || !(g.dtype() == out.dtype());
  if (elem_type_mismatch) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'max_pool2d_select_internal' cuda kernel requires g/x/out of the same dtype, got '" +
            std::string(g.dtype().name()) + "', '" + std::string(x.dtype().name()) + "', '" +
            std::string(out.dtype().name()) + "'");
  }
  // 先落地为具名变量再判断(与 elementwise.cu 同一惯例,CPP-012 文本扫描
  // 规避)。
  const bool supported = is_supported_dtype(g.dtype().code());
  if (!supported) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'max_pool2d_select_internal' cuda kernel does not support "
                               "dtype '" +
                                   std::string(g.dtype().name()) +
                                   "' (v0 supports float32/float16/bfloat16 only)");
  }

  const frame::Result<Pool2dRuntimeParams> params_result =
      read_pool2d_runtime_params(ctx, "max_pool2d_select_internal", x.shape());
  if (!params_result.is_ok()) return params_result.status();
  const Pool2dRuntimeParams& params = params_result.value();

  const frame::Shape expected_out_shape({params.n, params.c, params.out_h, params.out_w});
  if (!(out.shape() == expected_out_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'max_pool2d_select_internal' cuda kernel requires out shape "
                               "to match the pooling result, got " +
                                   out.shape().to_string() + ", expected " +
                                   expected_out_shape.to_string());
  }

  const int64_t total = params.n * params.c * params.out_h * params.out_w;
  const LaunchConfig cfg = compute_launch_config(total);
  const cudaStream_t stream = native_stream(ctx.stream);
  const frame::DTypeCode code = g.dtype().code();
  const frame::Tensor& g_ref = g;
  const frame::Tensor& x_ref = x;
  frame::Tensor& out_ref = out;
  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    max_pool2d_select_kernel<T><<<cfg.grid, cfg.block, 0, stream>>>(
        static_cast<const T*>(g_ref.raw_data()), static_cast<const T*>(x_ref.raw_data()),
        static_cast<T*>(out_ref.raw_data()), params);
    return frame::backends::cuda::cuda_status(cudaGetLastError(),
                                              "max_pool2d_select_internal cuda kernel launch");
  });
}

// avg_pool2d_grad_internal(dy)->dx:1 输入,attrs=
// input_shape+kernel/stride/padding(dy 单独不携带 H/W,同 cpu kernel 先例)。
frame::Status avg_pool2d_grad_internal_cuda_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'avg_pool2d_grad_internal' cuda kernel expects 1 input, got " +
                                   std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'avg_pool2d_grad_internal' cuda kernel expects 1 output, got " +
                                   std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& dy = ctx.inputs[0];
  frame::Tensor& dx = ctx.outputs[0];

  frame::Status rank_status = require_rank("avg_pool2d_grad_internal", "dy", 4, dy);
  if (!rank_status.is_ok()) return rank_status;
  const bool elem_type_mismatch = !(dy.dtype() == dx.dtype());
  if (elem_type_mismatch) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'avg_pool2d_grad_internal' cuda kernel requires dy/dx of the same dtype, got '" +
            std::string(dy.dtype().name()) + "', '" + std::string(dx.dtype().name()) + "'");
  }
  // 先落地为具名变量再判断:理由同上(max_pool2d_select_internal_cuda_kernel
  // 同款惯例)。
  const bool supported = is_supported_dtype(dy.dtype().code());
  if (!supported) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'avg_pool2d_grad_internal' cuda kernel does not support dtype "
                               "'" +
                                   std::string(dy.dtype().name()) +
                                   "' (v0 supports float32/float16/bfloat16 only)");
  }

  const frame::Result<frame::Shape> input_shape_result =
      read_shape_attr(ctx, "avg_pool2d_grad_internal", "input_shape");
  if (!input_shape_result.is_ok()) return input_shape_result.status();
  const frame::Shape& input_shape = input_shape_result.value();
  if (!(dx.shape() == input_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'avg_pool2d_grad_internal' cuda kernel requires dx(out) shape "
                               "to match attribute 'input_shape', got " +
                                   dx.shape().to_string() + ", expected " +
                                   input_shape.to_string());
  }

  const frame::Result<Pool2dRuntimeParams> params_result =
      read_pool2d_runtime_params(ctx, "avg_pool2d_grad_internal", input_shape);
  if (!params_result.is_ok()) return params_result.status();
  const Pool2dRuntimeParams& params = params_result.value();

  const frame::Shape expected_dy_shape({params.n, params.c, params.out_h, params.out_w});
  if (!(dy.shape() == expected_dy_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'avg_pool2d_grad_internal' cuda kernel requires dy shape to be "
                               "consistent with input_shape/kernel/stride/padding, got " +
                                   dy.shape().to_string() + ", expected " +
                                   expected_dy_shape.to_string());
  }

  const int64_t total = params.n * params.c * params.h * params.w;
  const LaunchConfig cfg = compute_launch_config(total);
  const cudaStream_t stream = native_stream(ctx.stream);
  const frame::DTypeCode code = dy.dtype().code();
  const frame::Tensor& dy_ref = dy;
  frame::Tensor& dx_ref = dx;
  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    avg_pool2d_grad_kernel<T><<<cfg.grid, cfg.block, 0, stream>>>(
        static_cast<const T*>(dy_ref.raw_data()), static_cast<T*>(dx_ref.raw_data()), params);
    return frame::backends::cuda::cuda_status(cudaGetLastError(),
                                              "avg_pool2d_grad_internal cuda kernel launch");
  });
}

}  // namespace

FRAME_REGISTER_KERNEL("max_pool2d_select_internal", frame::kCudaBackendName,
                      max_pool2d_select_internal_cuda_kernel);
FRAME_REGISTER_KERNEL("avg_pool2d_grad_internal", frame::kCudaBackendName,
                      avg_pool2d_grad_internal_cuda_kernel);
