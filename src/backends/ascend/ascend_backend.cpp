// 昇腾 NPU(CANN/AscendCL)后端实现单元(骨架;cpu-only 下不参与构建)。
// TODO(FRAME-IMPL): 依 docs/backends/ascend.md 落地 AscendCL 流/内存/整图执行,
//   注意昇腾执行模式裁决(ADR-0005)。参考:docs/backends/ascend.md。完成判据:
//   ascend preset 构建成功且 tests/cpp/backends/ 在昇腾设备上冒烟用例通过。

#include "ascend_backend.h"

namespace frame::backends::ascend {

std::string_view AscendBackend::name() const { return kAscendBackendName; }

Result<int32_t> AscendBackend::device_count() const { return FRAME_UNIMPLEMENTED(); }

Result<std::unique_ptr<hal::Stream>> AscendBackend::create_stream(Device /*device*/) {
  return FRAME_UNIMPLEMENTED();
}

hal::Allocator* AscendBackend::allocator(Device /*device*/) { return nullptr; }

Status AscendBackend::copy(void* /*dst*/, Device /*dst_device*/, const void* /*src*/,
                           Device /*src_device*/, size_t /*bytes*/, hal::Stream* /*stream*/) {
  return FRAME_UNIMPLEMENTED();
}

Result<std::unique_ptr<hal::Executable>> AscendBackend::compile(
    const ir::Graph& /*graph*/, const hal::CompileOptions& /*options*/) {
  return FRAME_UNIMPLEMENTED();
}

Status AscendBackend::launch(const hal::KernelInvocation& /*invocation*/, hal::Stream* /*stream*/) {
  return FRAME_UNIMPLEMENTED();
}

}  // namespace frame::backends::ascend

// TODO(FRAME-IMPL): 宏实现落地后在此静态注册。参考:src/runtime/backend_registry.cpp。
//   完成判据:BackendRegistry::get(kAscendBackendName) 可取到实例。
FRAME_REGISTER_BACKEND(frame::kAscendBackendName, frame::backends::ascend::AscendBackend)
