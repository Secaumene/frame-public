// CUDA gather/scatter-add 自写内核(M22 gather,M28 scatter_add):
// gather=行拷贝(逐输出元素一线程);scatter_add/gather_grad_internal=atomicAdd 散加
// (scatter-add,重复索引累加)。indices 有效性经 kernels/gather.h 声明、
// kernels/gather.cpp 实现的 D2H 预检(kernel 启动前完成,§1.5);预检通过后,
// 本文件两个 kernel 均信任 indices 已在界内,不再重复校验(与
// max_pool2d_select_kernel"理论不可达"惯例不同——这里是真正被前置同步预检
// 保证,非仅理论推导)。fp16/bf16 的 atomicAdd(__half*/__nv_bfloat16*)经
// cuda_fp16.h/cuda_bf16.h 提供:声明门槛 __CUDA_ARCH__>=700,内部对 SM>=90
// 用原生 tensor 指令、SM<90 用 atomicCAS 环路回退(该头文件既有实现自带
// 回退,不需要本文件重复;本机 CMakePresets.json 的
// CMAKE_CUDA_ARCHITECTURES=native 锁定编译目标为本机实测 CC 12.0,SM>=90
// 全部满足)。indices 分派 int32/int64 与值 dtype 三档嵌套
// dispatch_dtype(CPP-012 编译期展开)。

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>

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
#include "gather.h"
#include "launch_config.cuh"

namespace {

// device 端位型 <-> float 转换:同目录共享工具(铁律 5 收敛,M22 批4 判重,见
// accum_load_store.cuh 头注释)。
using frame::backends::cuda::elementwise_load;

// 原子累加 value 到 ptr[i](gather_grad_internal 的 scatter-add 核心)。float
// 原生 atomicAdd(float*, float)全架构支持;fp16/bf16 经 cuda_fp16.h/
// cuda_bf16.h 提供的 atomicAdd(__half*/__nv_bfloat16*, ...)(文件头注释已论证
// 本机编译目标满足其声明门槛,无需手写 CAS 环路兜底)。fp16/bf16 分支以半精度
// 原生 atomicAdd 直接原子累加(而非升 float 后累加、结束时一次写回),与 cpu
// 参考实现(gather_grad_internal 用 float 缓冲逐元素累加、最后一次性转换写回,
// 见 src/backends/cpu/kernels/gather.cpp)存在累加精度差异——重复 indices 累加
// 时,fp16/bf16 各次原子加之间会各自舍入到半精度,而 cpu 参考全程以 float
// 精度累加。此为批4 spec §1.5(docs/plan/2026-07-19-batch4-m22-seq.md)授权的
// 实现选择(CUDA 原子操作无法在半精度指针上原地做“先转 float 再原子加”),
// 数值差异由 BUILD-011 容差覆盖。
template <typename T>
__device__ __forceinline__ void atomic_add_element(T* ptr, int64_t i, float value) {
  if constexpr (std::is_same_v<T, float>) {
    atomicAdd(&ptr[i], value);
  } else if constexpr (std::is_same_v<T, frame::float16_t>) {
    atomicAdd(reinterpret_cast<__half*>(&ptr[i]), __float2half(value));
  } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
    atomicAdd(reinterpret_cast<__nv_bfloat16*>(&ptr[i]), __float2bfloat16(value));
  }
  // 其余 dtype:调用方已按白名单拒绝,运行时不可达,无操作。
}

// gather(x[N,F], indices[K]) -> out[K,F]:一线程一输出元素(线性下标
// linear=row*f+col),idx=indices[row],out[linear]=x[idx*f+col]。indices 已
// 经 D2H 预检确认 0<=idx<N,本 kernel 不再重复校验。
template <typename T, typename IdxT>
__global__ void gather_kernel(const T* x, const IdxT* indices, T* out, int64_t total, int64_t f) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) return;
  const int64_t row = linear / f;
  const int64_t col = linear % f;
  const int64_t idx = static_cast<int64_t>(indices[row]);
  out[linear] = x[idx * f + col];
}

// gather_grad_internal(gy[K,F], indices[K]) -> gx[N,F](scatter-add,重复
// indices 累加):一线程一 gy 元素(线性下标 linear=row*f+col),
// gx[idx*f+col] += gy[linear](atomicAdd)。调用方已 cudaMemsetAsync(0) 清零
// gx。
template <typename T, typename IdxT>
__global__ void gather_grad_kernel(const T* gy, const IdxT* indices, T* gx, int64_t total,
                                   int64_t f) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) return;
  const int64_t row = linear / f;
  const int64_t col = linear % f;
  const int64_t idx = static_cast<int64_t>(indices[row]);
  atomic_add_element(gx, idx * f + col, elementwise_load(gy, linear));
}

// launch 配置计算:同目录共享工具(铁律 5 收敛,M22 批4 判重,见
// launch_config.cuh 头注释)。
using frame::backends::cuda::compute_launch_config;
using frame::backends::cuda::LaunchConfig;

cudaStream_t native_stream(frame::hal::Stream* stream) {
  return stream != nullptr ? static_cast<cudaStream_t>(stream->native_handle()) : nullptr;
}

// value(x/gy)侧 dtype 白名单(v0 三档浮点,gather 族与其余 elementwise 系
// 算子同一白名单,§1.5)。
bool is_supported_gather_value_dtype(frame::DTypeCode code) {
  return code == frame::DTypeCode::kFloat32 || code == frame::DTypeCode::kFloat16 ||
         code == frame::DTypeCode::kBFloat16;
}

// gather(x[N,F], indices[K]) -> out[K,F]:host 包装——校验、indices D2H 预检
// (kernels/gather.h)、value/indices dtype 嵌套 dispatch_dtype 后 launch
// gather_kernel。
frame::Status gather_cuda_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'gather' cuda kernel expects 2 inputs, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'gather' cuda kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& x = ctx.inputs[0];
  const frame::Tensor& indices = ctx.inputs[1];
  frame::Tensor& out = ctx.outputs[0];

  if (x.shape().rank() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'gather' cuda kernel requires x to be rank-2 [N, F], got rank " +
                                   std::to_string(x.shape().rank()));
  }
  if (indices.shape().rank() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'gather' cuda kernel requires indices to be rank-1 [K], got rank " +
            std::to_string(indices.shape().rank()));
  }
  const bool value_type_mismatch = !(x.dtype() == out.dtype());
  if (value_type_mismatch) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'gather' cuda kernel requires x/out of the same dtype, got '" +
                                   std::string(x.dtype().name()) + "', '" +
                                   std::string(out.dtype().name()) + "'");
  }
  const bool value_supported = is_supported_gather_value_dtype(x.dtype().code());
  if (!value_supported) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'gather' cuda kernel does not support dtype '" +
                                   std::string(x.dtype().name()) +
                                   "' (v0 supports float32/float16/bfloat16 only)");
  }

  const int64_t n = x.shape().dim(0);
  const int64_t f = x.shape().dim(1);
  const int64_t k = indices.shape().dim(0);
  if (n < 0 || f < 0 || k < 0 || (f != 0 && k > std::numeric_limits<int64_t>::max() / f)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'gather' cuda kernel shape element count overflows int64 or "
                               "contains a negative dimension");
  }
  const frame::Shape expected_out_shape({k, f});
  if (!(out.shape() == expected_out_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'gather' cuda kernel requires out shape to match [K, F], got " +
                                   out.shape().to_string() + ", expected " +
                                   expected_out_shape.to_string());
  }

  const cudaStream_t stream = native_stream(ctx.stream);
  FRAME_RETURN_IF_ERROR(frame::backends::cuda::validate_gather_indices_range("gather", indices, n,
                                                                             "bound N", stream));

  const int64_t total = k * f;
  if (total == 0) return frame::Status::ok();
  const LaunchConfig cfg = compute_launch_config(total);
  const frame::DTypeCode value_code = x.dtype().code();
  const frame::DTypeCode index_code = indices.dtype().code();
  const frame::Tensor& x_ref = x;
  const frame::Tensor& indices_ref = indices;
  frame::Tensor& out_ref = out;
  return frame::dispatch_dtype(value_code, [&]<typename T>() -> frame::Status {
    return frame::dispatch_dtype(index_code, [&]<typename IdxT>() -> frame::Status {
      if constexpr (std::is_same_v<IdxT, std::int32_t> || std::is_same_v<IdxT, std::int64_t>) {
        gather_kernel<T, IdxT>
            <<<cfg.grid, cfg.block, 0, stream>>>(static_cast<const T*>(x_ref.raw_data()),
                                                 static_cast<const IdxT*>(indices_ref.raw_data()),
                                                 static_cast<T*>(out_ref.raw_data()), total, f);
        return frame::backends::cuda::cuda_status(cudaGetLastError(), "gather cuda kernel launch");
      }
      return frame::Status::ok();
    });
  });
}

// scatter-add 族 CUDA host 包装的共享核心:公共 scatter_add 与旧 internal
// 仅参数化名称、输入输出角色、shape 属性及首维符号;D2H 预检、清零、dtype
// 分派与 gather_grad_kernel atomic 路径保持单份。
struct ScatterAddCudaContract {
  std::string_view op_name;
  std::string_view values_role;
  std::string_view output_role;
  std::string_view dtype_role;
  std::string_view shape_attr_name;
  std::string_view consistency_role;
  std::string_view output_dim_name;
  std::string_view index_bound_description;
};

frame::Status scatter_add_cuda_kernel_impl(frame::ops::KernelContext& ctx,
                                           const ScatterAddCudaContract& contract) {
  const std::string_view op_name = contract.op_name;
  const std::string_view values_role = contract.values_role;
  const std::string_view output_role = contract.output_role;
  const std::string_view dtype_role = contract.dtype_role;
  const std::string_view shape_attr_name = contract.shape_attr_name;
  const std::string_view consistency_role = contract.consistency_role;
  const std::string_view output_dim_name = contract.output_dim_name;
  const std::string_view index_bound_description = contract.index_bound_description;
  if (ctx.inputs.size() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel expects 2 inputs, got " +
                                   std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel expects 1 output, got " +
                                   std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& values = ctx.inputs[0];
  const frame::Tensor& indices = ctx.inputs[1];
  frame::Tensor& output = ctx.outputs[0];

  if (values.shape().rank() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' cuda kernel requires " +
                                   std::string(values_role) + " to be rank-2 [K, F], got rank " +
                                   std::to_string(values.shape().rank()));
  }
  if (indices.shape().rank() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel requires indices to be rank-1 [K], got rank " +
                                   std::to_string(indices.shape().rank()));
  }
  if (output.shape().rank() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' cuda kernel requires " +
                                   std::string(output_role) + " to be rank-2 [" +
                                   std::string(output_dim_name) + ", F], got rank " +
                                   std::to_string(output.shape().rank()));
  }
  const bool value_type_mismatch = !(values.dtype() == output.dtype());
  if (value_type_mismatch) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' cuda kernel requires " +
                                   std::string(dtype_role) + " of the same dtype, got '" +
                                   std::string(values.dtype().name()) + "', '" +
                                   std::string(output.dtype().name()) + "'");
  }
  const bool value_supported = is_supported_gather_value_dtype(values.dtype().code());
  if (!value_supported) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) + "' cuda kernel does not support dtype '" +
            std::string(values.dtype().name()) + "' (v0 supports float32/float16/bfloat16 only)");
  }

  if (ctx.attrs == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel is missing required attribute '" +
                                   std::string(shape_attr_name) + "': no attrs provided");
  }
  const auto output_shape_it = ctx.attrs->find(std::string(shape_attr_name));
  if (output_shape_it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel is missing required attribute '" +
                                   std::string(shape_attr_name) + "'");
  }
  const frame::Shape* output_shape_ptr = std::get_if<frame::Shape>(&output_shape_it->second);
  if (output_shape_ptr == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' cuda kernel attribute '" +
                                   std::string(shape_attr_name) +
                                   "' has the wrong type, expected shape");
  }
  if (!(output.shape() == *output_shape_ptr)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) + "' cuda kernel requires " + std::string(output_role) +
            " shape to match attribute '" + std::string(shape_attr_name) + "', got " +
            output.shape().to_string() + ", expected " + output_shape_ptr->to_string());
  }

  const int64_t n = output.shape().dim(0);
  const int64_t f = output.shape().dim(1);
  const int64_t k = indices.shape().dim(0);
  if (n < 0 || f < 0 || k < 0 ||
      (f != 0 && (n > std::numeric_limits<int64_t>::max() / f ||
                  k > std::numeric_limits<int64_t>::max() / f))) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel shape element count overflows int64 or contains "
                                   "a negative dimension");
  }
  const frame::Shape expected_values_shape({k, f});
  if (!(values.shape() == expected_values_shape)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) + "' cuda kernel requires " + std::string(values_role) +
            " shape to be consistent with " + std::string(consistency_role) + ", got " +
            values.shape().to_string() + ", expected " + expected_values_shape.to_string());
  }

  const cudaStream_t stream = native_stream(ctx.stream);
  FRAME_RETURN_IF_ERROR(frame::backends::cuda::validate_gather_indices_range(
      op_name, indices, n, index_bound_description, stream));

  const size_t output_bytes = static_cast<size_t>(output.numel()) * output.dtype().itemsize();
  if (output_bytes > 0) {
    FRAME_RETURN_IF_ERROR(frame::backends::cuda::cuda_status(
        cudaMemsetAsync(output.raw_data(), 0, output_bytes, stream),
        std::string(op_name) + " cuda kernel: cudaMemsetAsync zero-init"));
  }

  const int64_t total = k * f;
  if (total == 0) return frame::Status::ok();
  const LaunchConfig cfg = compute_launch_config(total);
  const frame::DTypeCode value_code = values.dtype().code();
  const frame::DTypeCode index_code = indices.dtype().code();
  const frame::Tensor& values_ref = values;
  const frame::Tensor& indices_ref = indices;
  frame::Tensor& output_ref = output;
  return frame::dispatch_dtype(value_code, [&]<typename T>() -> frame::Status {
    return frame::dispatch_dtype(index_code, [&]<typename IdxT>() -> frame::Status {
      if constexpr (std::is_same_v<IdxT, std::int32_t> || std::is_same_v<IdxT, std::int64_t>) {
        gather_grad_kernel<T, IdxT>
            <<<cfg.grid, cfg.block, 0, stream>>>(static_cast<const T*>(values_ref.raw_data()),
                                                 static_cast<const IdxT*>(indices_ref.raw_data()),
                                                 static_cast<T*>(output_ref.raw_data()), total, f);
        return frame::backends::cuda::cuda_status(cudaGetLastError(),
                                                  std::string(op_name) + " cuda kernel launch");
      }
      return frame::Status::ok();
    });
  });
}

// 旧 internal CUDA kernel 的兼容薄 wrapper:input_shape、诊断和行为保持不变。
frame::Status gather_grad_internal_cuda_kernel(frame::ops::KernelContext& ctx) {
  constexpr ScatterAddCudaContract kContract{.op_name = "gather_grad_internal",
                                             .values_role = "gy",
                                             .output_role = "gx(out)",
                                             .dtype_role = "gy/gx(out)",
                                             .shape_attr_name = "input_shape",
                                             .consistency_role = "indices/gx(out)",
                                             .output_dim_name = "N",
                                             .index_bound_description = "bound N"};
  return scatter_add_cuda_kernel_impl(ctx, kContract);
}

// 公共 scatter_add CUDA kernel 的薄 wrapper。
frame::Status public_scatter_add_cuda_kernel(frame::ops::KernelContext& ctx) {
  constexpr ScatterAddCudaContract kContract{.op_name = "scatter_add",
                                             .values_role = "updates",
                                             .output_role = "out",
                                             .dtype_role = "updates/out",
                                             .shape_attr_name = "output_shape",
                                             .consistency_role = "indices/out",
                                             .output_dim_name = "V",
                                             .index_bound_description = "out's first dimension V"};
  return scatter_add_cuda_kernel_impl(ctx, kContract);
}

}  // namespace

FRAME_REGISTER_KERNEL("gather", frame::kCudaBackendName, gather_cuda_kernel);
FRAME_REGISTER_KERNEL("scatter_add", frame::kCudaBackendName, public_scatter_add_cuda_kernel);
FRAME_REGISTER_KERNEL("gather_grad_internal", frame::kCudaBackendName,
                      gather_grad_internal_cuda_kernel);
