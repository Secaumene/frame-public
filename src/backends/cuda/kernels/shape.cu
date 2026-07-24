// CUDA transpose 自写内核(M22,批4 T4,§1.4/1.6 决议点D/F):逐输出元素按
// perm 反查输入索引的 strided copy kernel,与 cpu 参考
// (src/backends/cpu/kernels/shape.cpp::transpose_cpu_kernel)同一算法——对每
// 个输出线性下标解码出多维下标(行优先),第 i 维对应输入的第 perm[i] 维,据此
// 按输入 strides 求出输入线性下标后逐元素(itemsize 宽)字节拷贝。不经
// dispatch_dtype(与 reshape/cpu 侧 transpose 同一理由:字节拷贝天然与 dtype
// 无关,ARCH-042 豁免),仅经本文件 is_supported_dtype 做白名单校验(取
// itemsize 供拷贝用)。host 包装(校验/attrs 读取/launch)与 __global__ kernel
// 同文件(镜像 pool.cu 的自写 kernel host/device 一体先例,不跨文件声明
// launcher)。concat/slice 为纯连续段拷贝,改用 cudaMemcpyAsync host 编排,见
// 同目录 shape.cpp,无需 __global__ kernel(理由见其头注释)。

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/stream.h>
#include <frame/ops/kernel_registry.h>

#include "../cuda_status.h"
#include "launch_config.cuh"

namespace {

// transpose 是本仓唯一秩不受限的算子(conv/pool 固定 rank-4、matmul/softmax/
// layer_norm 固定 rank-2);该上限覆盖本仓全部现实用例(含 spec"非自逆 rank-3
// perm"用例),超出上限时 host 端校验直接返回 kInvalidArgument(fail-loud,
// 不静默截断/不越界写栈上定长数组)。
constexpr int64_t kMaxTransposeRank = 8;

// device 端按值传入的几何参数:out_dims 供线性下标解码,permuted_x_strides[d]
// = 输入张量按 perm[d] 取的 stride(host 端已完成 perm 置换,设备侧无需再持有
// perm 本身,减少每线程的间接访问)。
struct TransposeGeometry {
  int64_t rank = 0;
  int64_t out_dims[kMaxTransposeRank] = {};
  int64_t permuted_x_strides[kMaxTransposeRank] = {};
};

// launch 配置计算:同目录共享工具(铁律 5 收敛,M22 批4 判重,见
// launch_config.cuh 头注释)。
using frame::backends::cuda::compute_launch_config;
using frame::backends::cuda::LaunchConfig;

cudaStream_t native_stream(frame::hal::Stream* stream) {
  return stream != nullptr ? static_cast<cudaStream_t>(stream->native_handle()) : nullptr;
}

// dtype 白名单(v0 三档浮点,与 cpu 参考一致)。
bool is_supported_dtype(frame::DTypeCode code) {
  return code == frame::DTypeCode::kFloat32 || code == frame::DTypeCode::kFloat16 ||
         code == frame::DTypeCode::kBFloat16;
}

// 逐输出元素按 perm 反查输入索引:字节拷贝,dtype 无关(ARCH-042 豁免,理由
// 见文件头注释)。geom 由 host 端预计算,均以元素计数而非字节计数——设备侧
// 再乘 itemsize。
__global__ void transpose_kernel(const char* x_bytes, char* out_bytes, int64_t numel,
                                 size_t itemsize, TransposeGeometry geom) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= numel) return;

  int64_t remaining = linear;
  int64_t x_linear = 0;
  for (int64_t d = geom.rank - 1; d >= 0; --d) {
    const int64_t dim_size = geom.out_dims[d];
    const int64_t safe_dim = dim_size > 0 ? dim_size : 1;
    const int64_t out_index_d = remaining % safe_dim;
    remaining /= safe_dim;
    x_linear += out_index_d * geom.permuted_x_strides[d];
  }

  const char* src = x_bytes + static_cast<size_t>(x_linear) * itemsize;
  char* dst = out_bytes + static_cast<size_t>(linear) * itemsize;
  for (size_t b = 0; b < itemsize; ++b) dst[b] = src[b];
}

frame::Status transpose_cuda_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'transpose' cuda kernel expects 1 input, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'transpose' cuda kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& x = ctx.inputs[0];
  frame::Tensor& out = ctx.outputs[0];

  const bool elem_type_mismatch = !(x.dtype() == out.dtype());
  if (elem_type_mismatch) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'transpose' cuda kernel requires x/out of the same dtype, got "
                               "'" +
                                   std::string(x.dtype().name()) + "', '" +
                                   std::string(out.dtype().name()) + "'");
  }
  const bool supported = is_supported_dtype(x.dtype().code());
  if (!supported) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'transpose' cuda kernel does not support dtype '" +
                                   std::string(x.dtype().name()) +
                                   "' (v0 supports float32/float16/bfloat16 only)");
  }

  if (ctx.attrs == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'transpose' cuda kernel is missing required attribute 'perm' "
                               "(int64 array): no attrs provided");
  }
  const auto perm_it = ctx.attrs->find("perm");
  if (perm_it == ctx.attrs->end()) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'transpose' cuda kernel is missing required attribute 'perm' (int64 array)");
  }
  const std::vector<int64_t>* perm = std::get_if<std::vector<int64_t>>(&perm_it->second);
  if (perm == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'transpose' cuda kernel attribute 'perm' has the wrong type, expected int64 array");
  }

  const int64_t rank = x.shape().rank();
  if (static_cast<int64_t>(perm->size()) != rank) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'transpose' cuda kernel attribute 'perm' must have rank=" + std::to_string(rank) +
            " element(s), got " + std::to_string(perm->size()));
  }
  if (rank > kMaxTransposeRank) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'transpose' cuda kernel requires rank <= " + std::to_string(kMaxTransposeRank) +
            ", got rank " + std::to_string(rank));
  }

  std::vector<int64_t> expected_out_dims(static_cast<size_t>(rank));
  for (int64_t i = 0; i < rank; ++i) {
    const int64_t p = (*perm)[static_cast<size_t>(i)];
    if (p < 0 || p >= rank) {
      return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                                 "op 'transpose' cuda kernel attribute 'perm' entry " +
                                     std::to_string(p) + " is out of range for rank " +
                                     std::to_string(rank));
    }
    expected_out_dims[static_cast<size_t>(i)] = x.shape().dim(p);
  }
  const frame::Shape expected_out_shape(expected_out_dims);
  if (!(out.shape() == expected_out_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'transpose' cuda kernel requires out shape to match the "
                               "permuted result, got " +
                                   out.shape().to_string() + ", expected " +
                                   expected_out_shape.to_string());
  }

  const int64_t numel = out.numel();
  if (numel == 0) return frame::Status::ok();

  const frame::Strides x_strides = frame::row_major_strides(x.shape());
  const std::vector<int64_t>& x_stride_values = x_strides.values();

  TransposeGeometry geom;
  geom.rank = rank;
  for (int64_t i = 0; i < rank; ++i) {
    geom.out_dims[i] = expected_out_dims[static_cast<size_t>(i)];
    geom.permuted_x_strides[i] =
        x_stride_values[static_cast<size_t>((*perm)[static_cast<size_t>(i)])];
  }

  const LaunchConfig cfg = compute_launch_config(numel);
  const cudaStream_t stream = native_stream(ctx.stream);
  transpose_kernel<<<cfg.grid, cfg.block, 0, stream>>>(static_cast<const char*>(x.raw_data()),
                                                       static_cast<char*>(out.raw_data()), numel,
                                                       x.dtype().itemsize(), geom);
  return frame::backends::cuda::cuda_status(cudaGetLastError(), "transpose cuda kernel launch");
}

}  // namespace

FRAME_REGISTER_KERNEL("transpose", frame::kCudaBackendName, transpose_cuda_kernel);
