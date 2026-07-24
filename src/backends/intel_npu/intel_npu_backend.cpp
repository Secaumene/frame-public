// Intel NPU(OpenVINO Runtime)后端实现单元(骨架;cpu-only 下不参与构建)。
// TODO(FRAME-IMPL): 依 docs/backends/intel-npu.md 落地 OpenVINO 模型编译/推理请求,
//   把整图转为 ov::Model 并经 compile_model 产出 Executable。参考:docs/backends/intel-npu.md。
//   完成判据:intel-npu preset 构建成功且 tests/cpp/backends/ 在 Intel NPU 上冒烟用例通过。

#include "intel_npu_backend.h"

namespace frame::backends::intel_npu {

std::string_view IntelNpuBackend::name() const { return kIntelNpuBackendName; }

Result<int32_t> IntelNpuBackend::device_count() const { return FRAME_UNIMPLEMENTED(); }

Result<std::unique_ptr<hal::Stream>> IntelNpuBackend::create_stream(Device /*device*/) {
  return FRAME_UNIMPLEMENTED();
}

hal::Allocator* IntelNpuBackend::allocator(Device /*device*/) { return nullptr; }

Status IntelNpuBackend::copy(void* /*dst*/, Device /*dst_device*/, const void* /*src*/,
                             Device /*src_device*/, size_t /*bytes*/, hal::Stream* /*stream*/) {
  return FRAME_UNIMPLEMENTED();
}

Result<std::unique_ptr<hal::Executable>> IntelNpuBackend::compile(
    const ir::Graph& /*graph*/, const hal::CompileOptions& /*options*/) {
  return FRAME_UNIMPLEMENTED();
}

Status IntelNpuBackend::launch(const hal::KernelInvocation& /*invocation*/,
                               hal::Stream* /*stream*/) {
  return FRAME_UNIMPLEMENTED();
}

}  // namespace frame::backends::intel_npu

// TODO(FRAME-IMPL): 宏实现落地后在此静态注册。参考:src/runtime/backend_registry.cpp。
//   完成判据:BackendRegistry::get(kIntelNpuBackendName) 可取到实例。
FRAME_REGISTER_BACKEND(frame::kIntelNpuBackendName, frame::backends::intel_npu::IntelNpuBackend)
