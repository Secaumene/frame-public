// HAL 一致性"转正"测试:以 BackendRegistry::available() 参数化,对每个已注册
// 后端跑一遍完整的 HAL 契约流程(get → device_count → stream → event →
// record/wait/synchronize → allocator 往返 → copy 往返),验证"任意后端满足同一
// 套 HAL 契约"这一端到端场景(铁律 #3:后端只经统一抽象接入)。逐项行为的细粒度、
// 可独立定位失败原因的用例见 tests/cpp/hal_conformance/test_hal_conformance.cpp。
// 当前 BackendRegistry::available() 仅 "cpu",但本文件禁止对后端名做任何特判,
// 新后端注册后自动纳入本套件。
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/event.h>
#include <frame/hal/stream.h>

namespace {

// BackendRegistry::available() 返回 std::string_view 列表(借用注册表内部
// std::string 的存储);此处拷贝为 std::string 供 gtest 参数化生成器长期持有
// (gtest 在 RegisterTests() 阶段才求值该生成器,借用的 string_view 到那时仍安全,
// 但拷贝为 std::string 更稳妥、也符合 TestWithParam<std::string> 的常见用法)。
std::vector<std::string> AvailableBackendNames() {
  std::vector<std::string> names;
  for (std::string_view name : frame::hal::BackendRegistry::instance().available()) {
    names.emplace_back(name);
  }
  return names;
}

class HalConformanceStub : public ::testing::TestWithParam<std::string> {};

TEST_P(HalConformanceStub, AllBackendsSatisfyHalContract) {
  // GetParam() 返回 const std::string&,取引用避免多余一次拷贝
  // (performance-unnecessary-copy-initialization)。
  const std::string& backend_name = GetParam();

  const frame::Result<frame::hal::Backend*> backend_result =
      frame::hal::BackendRegistry::instance().get(backend_name);
  ASSERT_TRUE(backend_result.is_ok());
  frame::hal::Backend* backend = backend_result.value();
  ASSERT_EQ(backend->name(), std::string_view(backend_name));

  const frame::Result<int32_t> count = backend->device_count();
  ASSERT_TRUE(count.is_ok());
  ASSERT_GE(count.value(), 1);

  const frame::Device device{backend->name(), 0};

  frame::Result<std::unique_ptr<frame::hal::Stream>> stream_result = backend->create_stream(device);
  ASSERT_TRUE(stream_result.is_ok());
  std::unique_ptr<frame::hal::Stream> stream = std::move(stream_result.value());
  ASSERT_NE(stream, nullptr);

  frame::Result<std::unique_ptr<frame::hal::Event>> event_result = backend->create_event(device);
  ASSERT_TRUE(event_result.is_ok());
  std::unique_ptr<frame::hal::Event> event = std::move(event_result.value());
  ASSERT_NE(event, nullptr);
  // 未 record 语义定案(backend-hal.md 2.3):query()==true、synchronize()==Ok。
  EXPECT_TRUE(event->query());
  EXPECT_TRUE(event->synchronize().is_ok());

  ASSERT_TRUE(stream->record(*event).is_ok());
  ASSERT_TRUE(stream->wait(*event).is_ok());
  ASSERT_TRUE(stream->synchronize().is_ok());

  frame::hal::Allocator* allocator = backend->allocator(device);
  ASSERT_NE(allocator, nullptr);
  constexpr size_t kAlignment = 64;
  constexpr size_t kBytes = 256;
  const frame::Result<void*> allocated = allocator->allocate(kBytes, kAlignment);
  ASSERT_TRUE(allocated.is_ok());
  void* ptr = allocated.value();
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % kAlignment, 0u);
  // 分配指针不假设可 host 端直接解引用(完成判据见 docs/backends/cuda.md
  // 第 4 章 Allocator 行待办标注):往返经 backend->copy(H2D 写入分配指针、
  // D2H 读回 host 缓冲)中转,不再对 ptr 做 host 端 memset/memcmp。
  std::vector<unsigned char> expected(kBytes, 0x3C);
  ASSERT_TRUE(backend->copy(ptr, device, expected.data(), frame::cpu_device(), kBytes, stream.get())
                  .is_ok());
  std::vector<unsigned char> readback(kBytes, 0);
  ASSERT_TRUE(backend->copy(readback.data(), frame::cpu_device(), ptr, device, kBytes, stream.get())
                  .is_ok());
  ASSERT_TRUE(stream->synchronize().is_ok());
  EXPECT_EQ(std::memcmp(readback.data(), expected.data(), kBytes), 0);
  allocator->deallocate(ptr);

  // H2H 拷贝往返:src -> dst -> back,验证经两次 backend->copy 后数据保持不变
  // (同上,src/back 的读写也全程经 backend->copy 中转,不直接解引用)。
  const frame::Result<void*> src_result = allocator->allocate(kBytes, kAlignment);
  const frame::Result<void*> dst_result = allocator->allocate(kBytes, kAlignment);
  const frame::Result<void*> back_result = allocator->allocate(kBytes, kAlignment);
  ASSERT_TRUE(src_result.is_ok());
  ASSERT_TRUE(dst_result.is_ok());
  ASSERT_TRUE(back_result.is_ok());
  void* src = src_result.value();
  void* dst = dst_result.value();
  void* back = back_result.value();

  std::vector<unsigned char> pattern(kBytes);
  for (size_t i = 0; i < kBytes; ++i) pattern[i] = static_cast<unsigned char>(i * 3 + 5);

  ASSERT_TRUE(backend->copy(src, device, pattern.data(), frame::cpu_device(), kBytes, stream.get())
                  .is_ok());
  ASSERT_TRUE(backend->copy(dst, device, src, device, kBytes, stream.get()).is_ok());
  ASSERT_TRUE(backend->copy(back, device, dst, device, kBytes, stream.get()).is_ok());
  std::vector<unsigned char> copy_readback(kBytes, 0);
  ASSERT_TRUE(
      backend->copy(copy_readback.data(), frame::cpu_device(), back, device, kBytes, stream.get())
          .is_ok());
  ASSERT_TRUE(stream->synchronize().is_ok());
  EXPECT_EQ(std::memcmp(copy_readback.data(), pattern.data(), kBytes), 0);

  allocator->deallocate(src);
  allocator->deallocate(dst);
  allocator->deallocate(back);
}

INSTANTIATE_TEST_SUITE_P(RegisteredBackends, HalConformanceStub,
                         ::testing::ValuesIn(AvailableBackendNames()),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                           return info.param;
                         });

}  // namespace
