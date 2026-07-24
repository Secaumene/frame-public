// CUDA sequence 自写内核:layer_norm(M22)与 selective_scan(M25)。
//
// selective_scan(x,a,b,c,d)->out 按最后一轴逐行执行状态递推。每个时间步
// 转成 float 精度仿射对(a,q=b*x),再以 CUB DeviceScan::InclusiveScan 计算
// 前缀复合;fp16/bf16 与 fp32 共用 float 仿射对,保证中间状态以 fp32 累计。
// 每个前导行独立排入当前 stream,单行 pair 缓冲按 stream 顺序复用,最终 kernel
// 计算 out=c*h+d*x。pair 与 CUB workspace 均经 Backend allocator/Storage RAII
// 分配,返回前同步当前 stream 后再释放,不调用裸 cudaMalloc。
//
// layer_norm 自写内核(M22,批4 T4,§1.2/1.6 决议点B/F):cuDNN legacy
// 无 LN 对应物(ADR-0021 决策 5「无现成即自写」同口径),故自写 __global__
// kernel——一行一 block、行均值/方差以 float 精度两遍归约(先求均值、再求
// centered 平方和,数值稳定式与 cpu 参考
// src/backends/cpu/kernels/sequence.cpp::layer_norm_cpu_kernel 同语义,不用
// E[x^2]-E[x]^2 单遍式以避免抵消误差),eps 从 attrs 取,gamma/beta 沿行广播。
// host 包装(校验/attrs 读取/launch)与 __global__ kernel 同文件(镜像
// pool.cu 的自写 kernel host/device 一体先例,不跨文件声明 launcher)。反向
// 无 kernel——梯度 = 公开算子微图(§1.2 表),由既有 mul/sum/reshape/matmul/
// add/rsqrt cuda kernel 承载,构图侧见
// src/ops/schemas/sequence.cpp::layer_norm_gradient。

#include <cstddef>
#include <cstdint>
#include <cub/device/device_scan.cuh>
#include <memory>
#include <string>
#include <variant>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/storage.h>
#include <frame/core/tensor.h>
#include <frame/hal/backend.h>
#include <frame/hal/stream.h>
#include <frame/ops/kernel_registry.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "../cuda_status.h"
#include "accum_load_store.cuh"
#include "launch_config.cuh"

namespace {

// device 端位型 <-> float 转换:同目录共享工具(铁律 5 收敛,M22 批4 判重,见
// accum_load_store.cuh 头注释)。
using frame::backends::cuda::compute_launch_config;
using frame::backends::cuda::elementwise_load;
using frame::backends::cuda::elementwise_store;
using frame::backends::cuda::LaunchConfig;

// 仿射状态转移 f(h)=a*h+q。CUB scan 全程使用该 float 载体,使三档输入
// dtype 的状态累计精度统一为 fp32。
struct AffinePair {
  float a;
  float q;
};

// CUB inclusive scan 按 scan_op(prefix,next) 调用;返回 next compose prefix,
// 即先应用前缀转移、再应用当前时间步转移。该运算满足结合律,但不交换。
struct AffineCompose {
  __host__ __device__ AffinePair operator()(const AffinePair& prefix,
                                            const AffinePair& next) const {
    return AffinePair{next.a * prefix.a, next.a * prefix.q + next.q};
  }
};

// 单行输入转为仿射对(a,b*x)。row_base 是该行在连续输入中的首元素下标。
template <typename T>
__global__ void selective_scan_prepare_kernel(const T* x, const T* a, const T* b, AffinePair* pairs,
                                              int64_t row_base, int64_t steps) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= steps) return;
  const int64_t index = row_base + t;
  pairs[t] = AffinePair{elementwise_load(a, index),
                        elementwise_load(b, index) * elementwise_load(x, index)};
}

// CUB scan 后 pairs[t].q 即 h[t],据此生成 out=c*h+d*x 并写回原 dtype。
template <typename T>
__global__ void selective_scan_finalize_kernel(const T* x, const T* c, const T* d,
                                               const AffinePair* pairs, T* out, int64_t row_base,
                                               int64_t steps) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= steps) return;
  const int64_t index = row_base + t;
  const float value = elementwise_load(c, index) * pairs[t].q +
                      elementwise_load(d, index) * elementwise_load(x, index);
  elementwise_store(out, index, value);
}

// 每 block 固定线程数(一行一 block),用于静态共享内存数组大小与树形归约
// 的 stride 起点(需为 2 的幂,树形归约正确性前提)。
constexpr int kLayerNormBlockSize = 256;

// layer_norm(x[N,D], gamma[D], beta[D]; eps) -> out[N,D]:blockIdx.x = 行号,
// 三趟遍历(均值、方差、输出),各趟内以 blockDim.x 为步长跨行遍历 D 个元素、
// 共享内存树形归约求 block 内总和。row_mean/row_rstd 存共享标量,经
// __syncthreads() 保证归约完成后对全 block 可见。
template <typename T>
__global__ void layer_norm_kernel(const T* x, const T* gamma, const T* beta, T* out, int64_t n,
                                  int64_t d, float eps) {
  const int64_t row = blockIdx.x;
  if (row >= n) return;
  const int64_t row_base = row * d;

  __shared__ float reduce_buffer[kLayerNormBlockSize];
  __shared__ float row_mean;
  __shared__ float row_rstd;

  // --- 第一趟:行均值 μ ---
  float local_sum = 0.0F;
  for (int64_t j = threadIdx.x; j < d; j += blockDim.x) {
    local_sum += elementwise_load(x, row_base + j);
  }
  reduce_buffer[threadIdx.x] = local_sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce_buffer[threadIdx.x] += reduce_buffer[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) row_mean = reduce_buffer[0] / static_cast<float>(d);
  __syncthreads();
  const float mean = row_mean;

  // --- 第二趟:行方差 σ² = mean((x-μ)²),r = 1/sqrt(σ²+eps) ---
  float local_var_sum = 0.0F;
  for (int64_t j = threadIdx.x; j < d; j += blockDim.x) {
    const float centered = elementwise_load(x, row_base + j) - mean;
    local_var_sum += centered * centered;
  }
  reduce_buffer[threadIdx.x] = local_var_sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce_buffer[threadIdx.x] += reduce_buffer[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float variance = reduce_buffer[0] / static_cast<float>(d);
    row_rstd = 1.0F / sqrtf(variance + eps);
  }
  __syncthreads();
  const float r = row_rstd;

  // --- 第三趟:输出 out = gamma*(x-μ)*r + beta(gamma/beta 沿行广播) ---
  for (int64_t j = threadIdx.x; j < d; j += blockDim.x) {
    const float xhat = (elementwise_load(x, row_base + j) - mean) * r;
    const float g = elementwise_load(gamma, j);
    const float b = elementwise_load(beta, j);
    elementwise_store(out, row_base + j, g * xhat + b);
  }
}

cudaStream_t native_stream(frame::hal::Stream* stream) {
  return stream != nullptr ? static_cast<cudaStream_t>(stream->native_handle()) : nullptr;
}

// dtype 白名单(v0 三档浮点,与 cpu 参考一致):fp32/fp16/bf16。
bool is_supported_dtype(frame::DTypeCode code) {
  return code == frame::DTypeCode::kFloat32 || code == frame::DTypeCode::kFloat16 ||
         code == frame::DTypeCode::kBFloat16;
}

// 在当前 stream 上逐行执行 prepare -> CUB scan -> finalize。单行 pair 缓冲
// 与 CUB workspace 由 Storage 托管;所有排队工作完成后才允许 RAII 析构释放。
template <typename T>
frame::Status run_selective_scan(const T* x, const T* a, const T* b, const T* c, const T* d, T* out,
                                 int64_t row_count, int64_t steps, frame::hal::Allocator& allocator,
                                 frame::Device device, cudaStream_t stream) {
  const size_t pair_bytes = static_cast<size_t>(steps) * sizeof(AffinePair);
  const frame::Result<std::shared_ptr<frame::Storage>> pair_storage =
      frame::Storage::allocate(allocator, pair_bytes, frame::kDefaultAlignment, device);
  if (!pair_storage.is_ok()) return pair_storage.status();
  auto* pairs = static_cast<AffinePair*>(pair_storage.value()->data());

  size_t temp_bytes = 0;
  FRAME_RETURN_IF_ERROR(frame::backends::cuda::cuda_status(
      cub::DeviceScan::InclusiveScan(nullptr, temp_bytes, pairs, AffineCompose{}, steps, stream),
      "selective_scan cuda kernel: cub::DeviceScan::InclusiveScan temp storage query"));

  const frame::Result<std::shared_ptr<frame::Storage>> temp_storage =
      frame::Storage::allocate(allocator, temp_bytes, frame::kDefaultAlignment, device);
  if (!temp_storage.is_ok()) return temp_storage.status();

  const LaunchConfig config = compute_launch_config(steps);
  frame::Status run_status = frame::Status::ok();
  for (int64_t row = 0; row < row_count; ++row) {
    const int64_t row_base = row * steps;
    selective_scan_prepare_kernel<T>
        <<<config.grid, config.block, 0, stream>>>(x, a, b, pairs, row_base, steps);
    run_status = frame::backends::cuda::cuda_status(
        cudaGetLastError(), "selective_scan cuda kernel: prepare kernel launch");
    if (!run_status.is_ok()) break;

    run_status = frame::backends::cuda::cuda_status(
        cub::DeviceScan::InclusiveScan(temp_storage.value()->data(), temp_bytes, pairs,
                                       AffineCompose{}, steps, stream),
        "selective_scan cuda kernel: cub::DeviceScan::InclusiveScan");
    if (!run_status.is_ok()) break;

    selective_scan_finalize_kernel<T>
        <<<config.grid, config.block, 0, stream>>>(x, c, d, pairs, out, row_base, steps);
    run_status = frame::backends::cuda::cuda_status(
        cudaGetLastError(), "selective_scan cuda kernel: finalize kernel launch");
    if (!run_status.is_ok()) break;
  }

  // pair_storage/temp_storage 即将析构;先等待当前 stream 上的 CUB 与自写
  // kernel 完成,避免未来 Allocator 改为流序释放后出现提前回收竞态。
  const frame::Status sync_status = frame::backends::cuda::cuda_status(
      cudaStreamSynchronize(stream),
      "selective_scan cuda kernel: cudaStreamSynchronize before temp storage release");
  return !run_status.is_ok() ? run_status : sync_status;
}

frame::Status selective_scan_cuda_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 5) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'selective_scan' cuda kernel expects 5 inputs (x, a, b, c, d), got " +
            std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'selective_scan' cuda kernel expects 1 output, got " +
                                   std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& x = ctx.inputs[0];
  const frame::Tensor& a = ctx.inputs[1];
  const frame::Tensor& b = ctx.inputs[2];
  const frame::Tensor& c = ctx.inputs[3];
  const frame::Tensor& d = ctx.inputs[4];
  frame::Tensor& out = ctx.outputs[0];

  if (x.shape().rank() < 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'selective_scan' cuda kernel requires rank >= 1");
  }
  if (x.shape().has_dynamic_dim()) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'selective_scan' cuda kernel requires a fully static shape, got " +
            x.shape().to_string());
  }

  const bool input_shape_mismatch = !(a.shape() == x.shape()) || !(b.shape() == x.shape()) ||
                                    !(c.shape() == x.shape()) || !(d.shape() == x.shape());
  if (input_shape_mismatch) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'selective_scan' cuda kernel requires x/a/b/c/d of the same shape, got " +
            x.shape().to_string() + ", " + a.shape().to_string() + ", " + b.shape().to_string() +
            ", " + c.shape().to_string() + ", " + d.shape().to_string());
  }
  if (!(out.shape() == x.shape())) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'selective_scan' cuda kernel requires out shape to match x, got " +
            out.shape().to_string() + ", expected " + x.shape().to_string());
  }

  const bool elem_type_mismatch = !(a.dtype() == x.dtype()) || !(b.dtype() == x.dtype()) ||
                                  !(c.dtype() == x.dtype()) || !(d.dtype() == x.dtype()) ||
                                  !(out.dtype() == x.dtype());
  if (elem_type_mismatch) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'selective_scan' cuda kernel requires x/a/b/c/d/out of the same dtype, got '" +
            std::string(x.dtype().name()) + "', '" + std::string(a.dtype().name()) + "', '" +
            std::string(b.dtype().name()) + "', '" + std::string(c.dtype().name()) + "', '" +
            std::string(d.dtype().name()) + "', '" + std::string(out.dtype().name()) + "'");
  }
  const frame::DTypeCode code = x.dtype().code();
  const bool supported = is_supported_dtype(code);
  if (!supported) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'selective_scan' cuda kernel does not support dtype '" +
                                   std::string(x.dtype().name()) +
                                   "' (v0 supports float32/float16/bfloat16 only)");
  }

  const int64_t steps = x.shape().dim(x.shape().rank() - 1);
  if (steps < 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'selective_scan' cuda kernel requires the last dimension "
                               "(steps) to be >= 1, got " +
                                   std::to_string(steps));
  }
  const int64_t numel = x.numel();
  if (numel == 0) return frame::Status::ok();
  const int64_t row_count = numel / steps;

  const frame::Result<frame::hal::Backend*> backend_lookup =
      frame::hal::BackendRegistry::instance().get(ctx.device.backend);
  if (!backend_lookup.is_ok()) {
    return frame::Status::make(
        backend_lookup.status().code(),
        "op 'selective_scan' cuda kernel: " + std::string(backend_lookup.status().message()));
  }
  frame::hal::Allocator* allocator = backend_lookup.value()->allocator(ctx.device);
  if (allocator == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInternal,
        "op 'selective_scan' cuda kernel: allocator unavailable for device '" +
            std::string(ctx.device.backend) + ":" + std::to_string(ctx.device.index) + "'");
  }

  const cudaStream_t stream = native_stream(ctx.stream);
  const void* x_data = x.raw_data();
  const void* a_data = a.raw_data();
  const void* b_data = b.raw_data();
  const void* c_data = c.raw_data();
  const void* d_data = d.raw_data();
  void* out_data = out.raw_data();
  const frame::Device device = ctx.device;
  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    return run_selective_scan(static_cast<const T*>(x_data), static_cast<const T*>(a_data),
                              static_cast<const T*>(b_data), static_cast<const T*>(c_data),
                              static_cast<const T*>(d_data), static_cast<T*>(out_data), row_count,
                              steps, *allocator, device, stream);
  });
}

frame::Status layer_norm_cuda_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 3) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'layer_norm' cuda kernel expects 3 inputs (x, gamma, beta), "
                               "got " +
                                   std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'layer_norm' cuda kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& x = ctx.inputs[0];
  const frame::Tensor& gamma = ctx.inputs[1];
  const frame::Tensor& beta = ctx.inputs[2];
  frame::Tensor& out = ctx.outputs[0];

  if (x.shape().rank() != 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'layer_norm' cuda kernel requires x to be rank-2 [N, D], got rank " +
            std::to_string(x.shape().rank()));
  }
  const int64_t n = x.shape().dim(0);
  const int64_t d = x.shape().dim(1);
  if (gamma.shape().rank() != 1 || gamma.shape().dim(0) != d) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'layer_norm' cuda kernel requires gamma to be rank-1 [D=" +
                                   std::to_string(d) + "], got " + gamma.shape().to_string());
  }
  if (beta.shape().rank() != 1 || beta.shape().dim(0) != d) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'layer_norm' cuda kernel requires beta to be rank-1 [D=" +
                                   std::to_string(d) + "], got " + beta.shape().to_string());
  }
  if (!(out.shape() == x.shape())) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'layer_norm' cuda kernel requires out shape to match x, got " +
                                   out.shape().to_string() + ", expected " + x.shape().to_string());
  }

  const bool elem_type_mismatch =
      !(x.dtype() == gamma.dtype()) || !(x.dtype() == beta.dtype()) || !(x.dtype() == out.dtype());
  if (elem_type_mismatch) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'layer_norm' cuda kernel requires x/gamma/beta/out of the same dtype, got '" +
            std::string(x.dtype().name()) + "', '" + std::string(gamma.dtype().name()) + "', '" +
            std::string(beta.dtype().name()) + "', '" + std::string(out.dtype().name()) + "'");
  }
  const bool supported = is_supported_dtype(x.dtype().code());
  if (!supported) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'layer_norm' cuda kernel does not support dtype '" +
                                   std::string(x.dtype().name()) +
                                   "' (v0 supports float32/float16/bfloat16 only)");
  }

  if (ctx.attrs == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'layer_norm' cuda kernel is missing required attribute 'eps': no attrs provided");
  }
  const auto eps_it = ctx.attrs->find("eps");
  if (eps_it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'layer_norm' cuda kernel is missing required attribute 'eps'");
  }
  const double* eps_ptr = std::get_if<double>(&eps_it->second);
  if (eps_ptr == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'layer_norm' cuda kernel attribute 'eps' has the wrong type, expected double");
  }
  const float eps = static_cast<float>(*eps_ptr);

  if (n == 0) return frame::Status::ok();

  const cudaStream_t stream = native_stream(ctx.stream);
  const frame::DTypeCode code = x.dtype().code();
  const frame::Tensor& x_ref = x;
  const frame::Tensor& gamma_ref = gamma;
  const frame::Tensor& beta_ref = beta;
  frame::Tensor& out_ref = out;
  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    layer_norm_kernel<T><<<static_cast<unsigned int>(n), kLayerNormBlockSize, 0, stream>>>(
        static_cast<const T*>(x_ref.raw_data()), static_cast<const T*>(gamma_ref.raw_data()),
        static_cast<const T*>(beta_ref.raw_data()), static_cast<T*>(out_ref.raw_data()), n, d, eps);
    return frame::backends::cuda::cuda_status(cudaGetLastError(), "layer_norm cuda kernel launch");
  });
}

}  // namespace

FRAME_REGISTER_KERNEL("selective_scan", frame::kCudaBackendName, selective_scan_cuda_kernel);
FRAME_REGISTER_KERNEL("layer_norm", frame::kCudaBackendName, layer_norm_cuda_kernel);
