// CPU 参考 kernel:fused_elementwise_internal(M11,决议点 C 建议⑥上提)——
// 后端无关的组合调用执行体已上提到 ops::execute_fused_chain(见
// include/frame/ops/fused_elementwise_utils.h),本文件仅剩薄 wrapper:自取
// cpu 后端 allocator 后转发。cuda 侧留一份同构 wrapper(见
// src/backends/cuda/kernels/fused_elementwise.cpp),REUSE-002 单份执行体、
// 各后端各自一行转发。

#include <string>

#include <frame/core/device.h>
#include <frame/core/status.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/ops/fused_elementwise_utils.h>
#include <frame/ops/kernel_registry.h>

namespace {

frame::Status fused_elementwise_cpu_kernel(frame::ops::KernelContext& ctx) {
  // 中间临时张量经 ctx.device.backend 取该后端 allocator 分配(决议点 B 采纳
  // 建议①:不硬编码 "cpu" 字面量;kernel 位于 src/backends/cpu/ 内,后端自取
  // 自身 allocator 合法,无分层问题)。
  const frame::Result<frame::hal::Backend*> backend =
      frame::hal::BackendRegistry::instance().get(ctx.device.backend);
  if (!backend.is_ok()) {
    return frame::Status::make(
        backend.status().code(),
        "op 'fused_elementwise_internal' cpu kernel: " + std::string(backend.status().message()));
  }
  frame::hal::Allocator* allocator = backend.value()->allocator(ctx.device);
  if (allocator == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInternal,
                               "op 'fused_elementwise_internal' cpu kernel: allocator "
                               "unavailable for device '" +
                                   std::string(ctx.device.backend) + "'");
  }
  return frame::ops::execute_fused_chain(ctx, *allocator);
}

}  // namespace

FRAME_REGISTER_KERNEL(frame::ops::kFusedElementwiseOpName, frame::kCpuBackendName,
                      fused_elementwise_cpu_kernel);
