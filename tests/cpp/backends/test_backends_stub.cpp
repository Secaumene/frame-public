// backends 子系统"转正"测试:cpu 参考后端经 FRAME_REGISTER_BACKEND 静态注册后
// BackendRegistry::get(kCpuBackendName) 可取到实例;cpu 分配器 allocate/
// deallocate 配对往返(64 对齐)。BackendRegistry 更全面的用例(get/available/
// register_backend 重名/register_backend_or_die fatal/FRAME_REGISTER_BACKEND
// 宏)见 tests/cpp/backends/test_backend_registry.cpp;CpuBackend 其余接口
// (device_count/流/事件/拷贝/launch)用例见 tests/cpp/backends/test_cpu_backend.cpp。
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>

TEST(BackendsStub, CpuBackendRegistered) {
  const frame::Result<frame::hal::Backend*> found =
      frame::hal::BackendRegistry::instance().get(frame::kCpuBackendName);
  ASSERT_TRUE(found.is_ok());
  EXPECT_EQ(found.value()->name(), frame::kCpuBackendName);
}

TEST(BackendsStub, AllocatorAllocateDeallocateRoundTrip) {
  const frame::Result<frame::hal::Backend*> backend_result =
      frame::hal::BackendRegistry::instance().get(frame::kCpuBackendName);
  ASSERT_TRUE(backend_result.is_ok());
  frame::hal::Backend* backend = backend_result.value();

  frame::hal::Allocator* allocator = backend->allocator(frame::cpu_device());
  ASSERT_NE(allocator, nullptr);

  constexpr size_t kAlignment = 64;
  constexpr size_t kBytes = 256;
  const frame::Result<void*> allocated = allocator->allocate(kBytes, kAlignment);
  ASSERT_TRUE(allocated.is_ok());
  void* ptr = allocated.value();
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % kAlignment, 0u);

  // 写读校验:分配到的内存必须真实可写可读(而非仅返回一个未回填的指针)。
  std::memset(ptr, 0x7E, kBytes);
  std::vector<unsigned char> readback(kBytes);
  std::memcpy(readback.data(), ptr, kBytes);
  for (unsigned char byte : readback) {
    EXPECT_EQ(byte, 0x7E);
  }

  allocator->deallocate(ptr);  // 配对释放;deallocate 无返回值,未崩溃即通过。
}
