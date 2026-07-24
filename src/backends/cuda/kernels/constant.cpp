// CUDA 参考 kernel:constant(阶段 C-14)。0 输入 1 输出,attrs=value/shape/
// dtype。物化逻辑单份复用(REUSE-002):host 端经既有
// ops::fill_tensor_from_constant_attrs(见 include/frame/ops/constant_utils.h,
// cpu kernel/constant_folding pass 共用同一份)填充 host staging 张量,再经
// cudaMemcpy H2D 搬到调用方预分配的设备输出张量。不含 __global__ 代码,故为
// .cpp 而非 .cu。

#include <cstddef>
#include <string>

#include <frame/core/device.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/ops/constant_utils.h>
#include <frame/ops/kernel_registry.h>

#include "../cuda_status.h"

namespace {

frame::Status constant_cuda_kernel(frame::ops::KernelContext& ctx) {
  if (!ctx.inputs.empty()) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'constant' cuda kernel expects 0 inputs, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'constant' cuda kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }
  if (ctx.attrs == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'constant' cuda kernel requires non-null attrs");
  }

  frame::Tensor& out = ctx.outputs[0];

  const frame::Result<frame::hal::Backend*> cpu_backend =
      frame::hal::BackendRegistry::instance().get(frame::kCpuBackendName);
  if (!cpu_backend.is_ok()) {
    return frame::Status::make(
        cpu_backend.status().code(),
        "op 'constant' cuda kernel: " + std::string(cpu_backend.status().message()));
  }
  frame::hal::Allocator* cpu_allocator = cpu_backend.value()->allocator(frame::cpu_device());
  if (cpu_allocator == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInternal,
                               "op 'constant' cuda kernel: cpu allocator unavailable for host "
                               "staging");
  }

  const frame::Result<frame::Tensor> staging =
      frame::Tensor::empty(out.shape(), out.dtype(), frame::cpu_device(), *cpu_allocator);
  if (!staging.is_ok()) return staging.status();
  frame::Tensor staging_tensor = staging.value();

  const frame::Status fill_status =
      frame::ops::fill_tensor_from_constant_attrs(*ctx.attrs, staging_tensor);
  if (!fill_status.is_ok()) return fill_status;

  const size_t bytes = static_cast<size_t>(out.numel()) * out.dtype().itemsize();
  if (bytes == 0) return frame::Status::ok();
  const cudaError_t error =
      cudaMemcpy(out.raw_data(), staging_tensor.raw_data(), bytes, cudaMemcpyHostToDevice);
  return frame::backends::cuda::cuda_status(error, "constant cuda kernel: cudaMemcpy H2D");
}

}  // namespace

FRAME_REGISTER_KERNEL(frame::ops::kConstantOpName, frame::kCudaBackendName, constant_cuda_kernel);
