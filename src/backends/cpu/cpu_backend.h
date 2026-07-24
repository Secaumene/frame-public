#pragma once
// CPU 参考后端内部头(不进 include/;仅供 src/backends/cpu/ 使用)。
// CPU 后端永远启用,是其余后端实现的样板。虚函数依据见 include/frame/hal/backend.h 头部。

#include <cstddef>
#include <memory>
#include <string_view>

#include <frame/core/device.h>
#include <frame/core/status.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/event.h>
#include <frame/hal/executable.h>
#include <frame/hal/stream.h>

namespace frame::backends::cpu {

// CPU 内存分配器:直接走主机内存(骨架桩)。
class CpuAllocator final : public hal::Allocator {
 public:
  Result<void*> allocate(size_t bytes, size_t alignment) override;
  void deallocate(void* ptr) override;
};

// CPU 执行流:同步后端,全部操作立即完成(无需真实排队)。
class CpuStream final : public hal::Stream {
 public:
  Status synchronize() override;
  Status record(hal::Event& event) override;
  Status wait(const hal::Event& event) override;
  void* native_handle() override;
};

// CPU 事件:同步后端,query()/synchronize() 恒表示「已完成」(见 event.h 头注释)。
class CpuEvent final : public hal::Event {
 public:
  bool query() const override;
  Status synchronize() override;
};

// CPU 参考后端:实现 Backend HAL 全部接口。
class CpuBackend final : public hal::Backend {
 public:
  std::string_view name() const override;
  Result<int32_t> device_count() const override;
  Result<std::unique_ptr<hal::Stream>> create_stream(Device device) override;
  Result<std::unique_ptr<hal::Event>> create_event(Device device) override;
  hal::Allocator* allocator(Device device) override;
  Status copy(void* dst, Device dst_device, const void* src, Device src_device, size_t bytes,
              hal::Stream* stream) override;
  Result<std::unique_ptr<hal::Executable>> compile(const ir::Graph& graph,
                                                   const hal::CompileOptions& options) override;
  Status launch(const hal::KernelInvocation& invocation, hal::Stream* stream) override;
};

}  // namespace frame::backends::cpu
