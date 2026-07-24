// CUDA 参考 kernel:fused_elementwise_internal(阶段 C-15)。后端无关的组合
// 调用执行体已上提到 ops::execute_fused_chain(见 include/frame/ops/
// fused_elementwise_utils.h,M11 决议点 C 建议⑥);本文件仅剩薄 wrapper:自取
// cuda 后端 allocator 后转发,与 src/backends/cpu/kernels/fused_elementwise.cpp
// 同构(REUSE-002 单份执行体、各后端各自一行转发)。不含 __global__ 代码,故为
// .cpp 而非 .cu。

#include <string>

#include <frame/core/device.h>
#include <frame/core/status.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/ops/fused_elementwise_utils.h>
#include <frame/ops/kernel_registry.h>

namespace {

frame::Status fused_elementwise_cuda_kernel(frame::ops::KernelContext& ctx) {
  const frame::Result<frame::hal::Backend*> backend =
      frame::hal::BackendRegistry::instance().get(ctx.device.backend);
  if (!backend.is_ok()) {
    return frame::Status::make(
        backend.status().code(),
        "op 'fused_elementwise_internal' cuda kernel: " + std::string(backend.status().message()));
  }
  frame::hal::Allocator* allocator = backend.value()->allocator(ctx.device);
  if (allocator == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInternal,
                               "op 'fused_elementwise_internal' cuda kernel: allocator "
                               "unavailable for device '" +
                                   std::string(ctx.device.backend) + "'");
  }
  return frame::ops::execute_fused_chain(ctx, *allocator);
}

}  // namespace

FRAME_REGISTER_KERNEL(frame::ops::kFusedElementwiseOpName, frame::kCudaBackendName,
                      fused_elementwise_cuda_kernel);
