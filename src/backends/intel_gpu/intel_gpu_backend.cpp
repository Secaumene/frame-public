// Intel GPU(oneAPI/SYCL)后端实现单元(骨架;cpu-only 下不参与构建)。
// TODO(FRAME-IMPL): 依 docs/backends/intel-gpu.md 落地 SYCL 队列/USM 分配/整图 codegen,
//   实现各 HAL 方法。参考:docs/backends/intel-gpu.md。完成判据:intel-gpu preset(icpx)
//   构建成功且 tests/cpp/backends/ 在 Intel GPU 上冒烟用例通过。

#include "intel_gpu_backend.h"

namespace frame::backends::intel_gpu {

std::string_view IntelGpuBackend::name() const { return kIntelGpuBackendName; }

Result<int32_t> IntelGpuBackend::device_count() const { return FRAME_UNIMPLEMENTED(); }

Result<std::unique_ptr<hal::Stream>> IntelGpuBackend::create_stream(Device /*device*/) {
  return FRAME_UNIMPLEMENTED();
}

hal::Allocator* IntelGpuBackend::allocator(Device /*device*/) { return nullptr; }

Status IntelGpuBackend::copy(void* /*dst*/, Device /*dst_device*/, const void* /*src*/,
                             Device /*src_device*/, size_t /*bytes*/, hal::Stream* /*stream*/) {
  return FRAME_UNIMPLEMENTED();
}

Result<std::unique_ptr<hal::Executable>> IntelGpuBackend::compile(
    const ir::Graph& /*graph*/, const hal::CompileOptions& /*options*/) {
  return FRAME_UNIMPLEMENTED();
}

Status IntelGpuBackend::launch(const hal::KernelInvocation& /*invocation*/,
                               hal::Stream* /*stream*/) {
  return FRAME_UNIMPLEMENTED();
}

}  // namespace frame::backends::intel_gpu

// TODO(FRAME-IMPL): 宏实现落地后在此静态注册。参考:src/runtime/backend_registry.cpp。
//   完成判据:BackendRegistry::get(kIntelGpuBackendName) 可取到实例。
FRAME_REGISTER_BACKEND(frame::kIntelGpuBackendName, frame::backends::intel_gpu::IntelGpuBackend)
