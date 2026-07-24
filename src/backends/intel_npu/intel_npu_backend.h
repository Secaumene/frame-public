#pragma once
// Intel NPU(OpenVINO Runtime)后端内部头(不进 include/;仅供 src/backends/intel_npu/ 使用)。
// 骨架期不引 SDK 头、不写 SDK 调用,仅留 HAL 接口骨架。虚函数依据见
// include/frame/hal/backend.h 头部。
// 后端隔离要求见 docs/architecture/overview.md 的 ARCH-001。

#include <cstddef>
#include <memory>
#include <string_view>

#include <frame/core/device.h>
#include <frame/core/status.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/executable.h>
#include <frame/hal/stream.h>

namespace frame::backends::intel_npu {

// Intel NPU 设备内存分配器(骨架)。
class IntelNpuAllocator final : public hal::Allocator {
 public:
  Result<void*> allocate(size_t bytes, size_t alignment) override;
  void deallocate(void* ptr) override;
};

// Intel NPU 后端:实现 Backend HAL 全部接口(骨架)。
// 注:一律经 OpenVINO Runtime 接入(Level Zero 直连路线已废弃,见 cmake/frame_dependencies.cmake)。
class IntelNpuBackend final : public hal::Backend {
 public:
  std::string_view name() const override;
  Result<int32_t> device_count() const override;
  Result<std::unique_ptr<hal::Stream>> create_stream(Device device) override;
  hal::Allocator* allocator(Device device) override;
  Status copy(void* dst, Device dst_device, const void* src, Device src_device, size_t bytes,
              hal::Stream* stream) override;
  Result<std::unique_ptr<hal::Executable>> compile(const ir::Graph& graph,
                                                   const hal::CompileOptions& options) override;
  Status launch(const hal::KernelInvocation& invocation, hal::Stream* stream) override;
};

}  // namespace frame::backends::intel_npu
