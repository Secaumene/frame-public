// CUDA 频域内核(M23,批5 T4,§1.2/1.6 决议点B/F,ADR-0022):rfft/irfft 经
// cuFFT cufftPlanMany(rank=1)R2C/C2R 实现(末轴变换)。cuFFT 调用面圈禁:仅
// 本文件与 cufft_utils.h(ADR-0022 决策 2:`grep -rln "cufft" src/ include/`
// 命中须 ⊆ 这两个文件 + CMakeLists.txt 链接行)。
//
// 布局约定同 src/backends/cpu/kernels/fft.cpp:rfft 输出/irfft 输入按
// [...,k,2] 交错存放,与 cufftComplex(交错 re/im)逐字节一致,故直接
// reinterpret_cast 复用宿主浮点缓冲、零布局搬运。batch(前导维乘积)经
// cufftPlanMany 的 batch 形参一次性表达,inembed/onembed 传 nullptr 取默认
// 连续布局(决议点F:istride/ostride=1,idist/odist 由变换方向的复数/实数
// 长度给出)。
//
// plan 现建现毁(ADR-0022 决策 6,v0 不缓存):每次 kernel 调用按 (n, batch)
// 现建 cufftPlanMany + cufftSetStream 绑执行流,退出前经 CufftPlanGuard
// (cufft_utils.h)析构销毁。销毁前必须显式 cudaStreamSynchronize——
// cufftDestroy 是否具备 stream 顺序语义未见 cuFFT 官方文档明文承诺,保守
// 起见与 conv.cpp::allocate_workspace_if_needed 头注释同一纪律(cudaFree 类
// 释放不假设与异步在途工作天然有序)。
//
// C2R 会破坏输入缓冲(cuFFT 文档行为,即使输入输出为不同缓冲区亦然):irfft
// 执行前把输入先 cudaMemcpyAsync(D2D)拷到临时 workspace(既有
// Storage::allocate + Allocator 先例,同 conv.cpp),cufftExecC2R 对该临时
// 副本操作,原始输入张量保持不变。
//
// irfft 的 1/n 归一化(numpy 口径)在 cufftExecC2R 执行后经既有 cuBLAS
// cublasSscal 完成——CUDA::cublas 已随 matmul.cpp 链接,复用 CudaBackend::
// acquire_cublas_handle 惰性 handle,避免为一次标量缩放新写 __global__
// kernel(本文件与其余 cuFFT/cuDNN/cuBLAS host 编排文件同款,不含 __global__
// 代码,故为 .cpp 而非 .cu)。
//
// dtype 限 fp32(ADR-0022 决策 5):cuFFT 半精度须另开 cufftXt API 面,v0 不
// 支持,fail-loud 拒绝其余 dtype(M22 白名单先例)。

#include <cstddef>
#include <cstdint>
#include <cublas_v2.h>
#include <cufft.h>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/storage.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/stream.h>
#include <frame/ops/kernel_registry.h>

#include <cuda_runtime.h>

#include "../cuda_backend.h"
#include "../cuda_status.h"
#include "cufft_utils.h"

namespace {

using frame::backends::cuda::CublasHandleGuard;
using frame::backends::cuda::CudaBackend;
using frame::backends::cuda::cufft_status;
using frame::backends::cuda::CufftPlanGuard;

// (op, backend) 查找 + static_cast 到 CudaBackend*:同目录各文件独立持有一份
// 实现(REUSE-002,理由同 conv.cpp/sequence.cpp::lookup_cuda_backend 头注释:
// 不跨文件借用匿名命名空间符号)。
frame::Result<CudaBackend*> lookup_cuda_backend(std::string_view op_name, frame::Device device) {
  const frame::Result<frame::hal::Backend*> backend_lookup =
      frame::hal::BackendRegistry::instance().get(device.backend);
  if (!backend_lookup.is_ok()) {
    return frame::Status::make(backend_lookup.status().code(),
                               "op '" + std::string(op_name) + "' cuda kernel: " +
                                   std::string(backend_lookup.status().message()));
  }
  // static_cast 而非 dynamic_cast(CPP-011):理由同 conv.cpp/sequence.cpp
  // ::lookup_cuda_backend。
  return static_cast<CudaBackend*>(backend_lookup.value());
}

cudaStream_t native_stream(frame::hal::Stream* stream) {
  return stream != nullptr ? static_cast<cudaStream_t>(stream->native_handle()) : nullptr;
}

// 按 bytes 经既有分配器分配临时 device 缓冲(REUSE-002:与
// conv.cpp::allocate_workspace_if_needed 同一动机,cuda 侧各文件独立持有一份
// 实现)。仅供 irfft 的"输入先拷贝到临时缓冲再执行 C2R"步骤使用。
frame::Result<std::shared_ptr<frame::Storage>> allocate_device_buffer(
    frame::hal::Allocator& allocator, frame::Device device, size_t bytes) {
  return frame::Storage::allocate(allocator, bytes, frame::kDefaultAlignment, device);
}

// rfft(x[...,n]) -> out[...,k,2](k=n/2+1):cufftPlanMany(CUFFT_R2C,rank=1)。
// 无属性,不归一化(numpy 口径,决议点B)。
frame::Status rfft_cuda_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'rfft' cuda kernel expects 1 input, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'rfft' cuda kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& x = ctx.inputs[0];
  frame::Tensor& out = ctx.outputs[0];

  const frame::DType x_type = x.dtype();
  const frame::DType out_type = out.dtype();
  const bool fp32_only_violation =
      x_type.code() != frame::DTypeCode::kFloat32 || out_type.code() != frame::DTypeCode::kFloat32;
  if (fp32_only_violation) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'rfft' cuda kernel requires x/out to be float32 (cuFFT has no half precision support "
        "in v0), got x='" +
            std::string(x_type.name()) + "', out='" + std::string(out_type.name()) + "'");
  }

  const int64_t rank = x.shape().rank();
  if (rank < 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'rfft' cuda kernel requires x to have rank >= 1, got rank " + std::to_string(rank));
  }
  const int64_t n = x.shape().dim(rank - 1);
  if (n < 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'rfft' cuda kernel requires the last dimension (n) to be >= 2, got n=" +
            std::to_string(n));
  }
  const int64_t k = n / 2 + 1;

  std::vector<int64_t> expected_out_dims = x.shape().dims();
  expected_out_dims[static_cast<size_t>(rank - 1)] = k;
  expected_out_dims.push_back(2);
  const frame::Shape expected_out_shape(expected_out_dims);
  if (!(out.shape() == expected_out_shape)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'rfft' cuda kernel requires out shape to match the rfft result, got " +
            out.shape().to_string() + ", expected " + expected_out_shape.to_string());
  }

  int64_t batch = 1;
  for (int64_t i = 0; i < rank - 1; ++i) batch *= x.shape().dim(i);

  // rfft 不需要 CudaBackend 惰性 handle(无 cudnn/cublas 调用),仅需原生
  // stream(直接取自 ctx.stream,不经 lookup_cuda_backend——那是
  // acquire_cudnn_handle/acquire_cublas_handle 场景专用,irfft 侧因 cublasSscal
  // 缩放步骤才需要,见 irfft_cuda_kernel)。
  const cudaStream_t stream = native_stream(ctx.stream);

  CufftPlanGuard plan_guard;
  int transform_size[1] = {static_cast<int>(n)};
  const cufftResult plan_result = cufftPlanMany(
      &plan_guard.plan, /*rank=*/1, transform_size, /*inembed=*/nullptr,
      /*istride=*/1, /*idist=*/static_cast<int>(n), /*onembed=*/nullptr,
      /*ostride=*/1, /*odist=*/static_cast<int>(k), CUFFT_R2C, static_cast<int>(batch));
  FRAME_RETURN_IF_ERROR(cufft_status(plan_result, "rfft cuda kernel: cufftPlanMany"));
  plan_guard.created = true;

  FRAME_RETURN_IF_ERROR(
      cufft_status(cufftSetStream(plan_guard.plan, stream), "rfft cuda kernel: cufftSetStream"));

  const float* x_data = static_cast<const float*>(x.raw_data());
  auto* out_data = reinterpret_cast<cufftComplex*>(out.raw_data());
  const cufftResult exec_result =
      cufftExecR2C(plan_guard.plan, const_cast<float*>(x_data), out_data);
  FRAME_RETURN_IF_ERROR(cufft_status(exec_result, "rfft cuda kernel: cufftExecR2C"));

  // CufftPlanGuard 析构(cufftDestroy)前必须确保上面的异步 cufftExecR2C 已在
  // stream 上完成,见本文件头注释与 cufft_utils.h::CufftPlanGuard 头注释。
  return frame::backends::cuda::cuda_status(
      cudaStreamSynchronize(stream),
      "rfft cuda kernel: cudaStreamSynchronize before plan teardown");
}

// irfft(z[...,k,2]; n) -> out[...,n]:cufftPlanMany(CUFFT_C2R,rank=1) +
// cufftExecC2R 后接 cublasSscal 做 1/n 归一化(numpy 口径,决议点B)。属性
// n(kInt64,必需)。
frame::Status irfft_cuda_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'irfft' cuda kernel expects 1 input, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'irfft' cuda kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& z = ctx.inputs[0];
  frame::Tensor& out = ctx.outputs[0];

  const frame::DType z_type = z.dtype();
  const frame::DType out_type = out.dtype();
  const bool fp32_only_violation =
      z_type.code() != frame::DTypeCode::kFloat32 || out_type.code() != frame::DTypeCode::kFloat32;
  if (fp32_only_violation) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'irfft' cuda kernel requires z/out to be float32 (cuFFT has no half precision "
        "support in v0), got z='" +
            std::string(z_type.name()) + "', out='" + std::string(out_type.name()) + "'");
  }

  const int64_t rank = z.shape().rank();
  if (rank < 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'irfft' cuda kernel requires z to have rank >= 2 (trailing axes are [k, 2]), got "
        "rank " +
            std::to_string(rank));
  }
  const int64_t last_dim = z.shape().dim(rank - 1);
  if (last_dim != 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'irfft' cuda kernel requires the last dimension to be 2 (interleaved re/im), got " +
            std::to_string(last_dim));
  }

  if (ctx.attrs == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'irfft' cuda kernel is missing required attribute 'n' (int64): "
                               "no attrs provided");
  }
  const auto n_it = ctx.attrs->find("n");
  if (n_it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'irfft' cuda kernel is missing required attribute 'n' (int64)");
  }
  const int64_t* n_ptr = std::get_if<int64_t>(&n_it->second);
  if (n_ptr == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'irfft' cuda kernel attribute 'n' has the wrong type, expected int64");
  }
  const int64_t n = *n_ptr;

  const int64_t k = z.shape().dim(rank - 2);
  const int64_t expected_k = n / 2 + 1;
  if (k != expected_k) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'irfft' cuda kernel requires k=n/2+1 for attribute n=" + std::to_string(n) +
            ", expected k=" + std::to_string(expected_k) + ", got k=" + std::to_string(k));
  }

  std::vector<int64_t> expected_out_dims(z.shape().dims().begin(), z.shape().dims().end() - 1);
  expected_out_dims.back() = n;
  const frame::Shape expected_out_shape(expected_out_dims);
  if (!(out.shape() == expected_out_shape)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'irfft' cuda kernel requires out shape to match the irfft result, got " +
            out.shape().to_string() + ", expected " + expected_out_shape.to_string());
  }

  int64_t batch = 1;
  for (int64_t i = 0; i < rank - 2; ++i) batch *= z.shape().dim(i);

  const frame::Result<CudaBackend*> backend_result = lookup_cuda_backend("irfft", ctx.device);
  if (!backend_result.is_ok()) return backend_result.status();
  CudaBackend* cuda_backend = backend_result.value();
  const cudaStream_t stream = native_stream(ctx.stream);

  // C2R 会破坏输入缓冲(即使输入输出为不同缓冲区亦然,cuFFT 文档行为):把 z
  // 先 D2D 拷到临时缓冲,cufftExecC2R 对该副本操作,原始 z 保持不变(见本文件
  // 头注释)。
  frame::hal::Allocator* allocator = cuda_backend->allocator(ctx.device);
  if (allocator == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInternal,
                               "op 'irfft' cuda kernel: allocator unavailable for device '" +
                                   std::string(ctx.device.backend) + "'");
  }
  const size_t input_bytes = static_cast<size_t>(z.numel()) * sizeof(float);
  const frame::Result<std::shared_ptr<frame::Storage>> scratch_result =
      allocate_device_buffer(*allocator, ctx.device, input_bytes);
  if (!scratch_result.is_ok()) return scratch_result.status();
  const std::shared_ptr<frame::Storage>& scratch_storage = scratch_result.value();
  FRAME_RETURN_IF_ERROR(frame::backends::cuda::cuda_status(
      cudaMemcpyAsync(scratch_storage->data(), z.raw_data(), input_bytes, cudaMemcpyDeviceToDevice,
                      stream),
      "irfft cuda kernel: cudaMemcpyAsync(z -> scratch, D2D pre-copy before C2R)"));

  CufftPlanGuard plan_guard;
  int transform_size[1] = {static_cast<int>(n)};
  const cufftResult plan_result = cufftPlanMany(
      &plan_guard.plan, /*rank=*/1, transform_size, /*inembed=*/nullptr,
      /*istride=*/1, /*idist=*/static_cast<int>(k), /*onembed=*/nullptr,
      /*ostride=*/1, /*odist=*/static_cast<int>(n), CUFFT_C2R, static_cast<int>(batch));
  FRAME_RETURN_IF_ERROR(cufft_status(plan_result, "irfft cuda kernel: cufftPlanMany"));
  plan_guard.created = true;

  FRAME_RETURN_IF_ERROR(
      cufft_status(cufftSetStream(plan_guard.plan, stream), "irfft cuda kernel: cufftSetStream"));

  auto* scratch_data = reinterpret_cast<cufftComplex*>(scratch_storage->data());
  float* out_data = out.data<float>();
  const cufftResult exec_result = cufftExecC2R(plan_guard.plan, scratch_data, out_data);
  FRAME_RETURN_IF_ERROR(cufft_status(exec_result, "irfft cuda kernel: cufftExecC2R"));

  // 1/n 归一化(numpy 口径):复用既有 cuBLAS cublasSscal(CUDA::cublas 已随
  // matmul.cpp 链接),不为一次标量缩放新写 __global__ kernel(本文件头注释)。
  const frame::Result<CublasHandleGuard> cublas_guard = cuda_backend->acquire_cublas_handle(stream);
  if (!cublas_guard.is_ok()) return cublas_guard.status();
  const cublasHandle_t cublas_handle = cublas_guard.value().handle;
  const float scale = 1.0F / static_cast<float>(n);
  const cublasStatus_t scale_status =
      cublasSscal(cublas_handle, static_cast<int>(out.numel()), &scale, out_data, /*incx=*/1);
  if (scale_status != CUBLAS_STATUS_SUCCESS) {
    return frame::Status::make(frame::ErrorCode::kInternal,
                               "op 'irfft' cuda kernel: cublasSscal failed with status " +
                                   std::to_string(static_cast<int>(scale_status)));
  }

  // CufftPlanGuard/scratch_storage 析构(cufftDestroy/deallocate)前必须确保
  // 上面全部异步工作已在 stream 上完成,理由同 rfft_cuda_kernel 尾注释。
  return frame::backends::cuda::cuda_status(
      cudaStreamSynchronize(stream),
      "irfft cuda kernel: cudaStreamSynchronize before plan/scratch teardown");
}

}  // namespace

FRAME_REGISTER_KERNEL("rfft", frame::kCudaBackendName, rfft_cuda_kernel);
FRAME_REGISTER_KERNEL("irfft", frame::kCudaBackendName, irfft_cuda_kernel);
