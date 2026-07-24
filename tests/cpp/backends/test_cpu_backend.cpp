// CpuBackend 单测(src/backends/cpu/cpu_backend.cpp):device_count==1;
// create_stream/create_event/allocator/copy/launch 对非法设备(backend!="cpu"
// 或 index!=0)的拒绝路径;allocator 分配非法 alignment/bytes==0 的错误路径
// (纯 round-trip 见 tests/cpp/backends/test_backends_stub.cpp:
// BackendsStub.AllocatorAllocateDeallocateRoundTrip);copy 字节往返一致 +
// 重叠区域 memmove 语义 + 空指针报错;launch 执行已注册 kernel 且效果可观察、
// 未注册 op 返回含算子名与后端名的错误(ARCH-031)。
//
// 本文件不直接 include src/backends/cpu/cpu_backend.h(该头仅供
// src/backends/cpu/ 内部使用,不进 include/);一律经
// BackendRegistry::instance().get(kCpuBackendName) 取 Backend* 公开接口测试。
// 本文件经 FRAME_REGISTER_KERNEL 注册的 kernel 以 "test_cpu_backend_" 前缀
// op 名,跨全体测试文件保持进程级唯一(同 tests/cpp/ops/test_kernel_registry.cpp
// 头注释纪律)。
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/event.h>
#include <frame/hal/stream.h>
#include <frame/ops/kernel_registry.h>

namespace {

// launch 冒烟用的 kernel:把 attrs["fill_value"](int64)写入 outputs[0] 的每个
// int32 元素,提供可观察副作用(整数精确相等,无需 BUILD-011 容差工具)。
frame::Status FillKernel(frame::ops::KernelContext& ctx) {
  if (ctx.outputs.empty()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "test_cpu_backend_fill_op: missing output");
  }
  if (ctx.attrs == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "test_cpu_backend_fill_op: missing attrs");
  }
  const auto it = ctx.attrs->find("fill_value");
  if (it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "test_cpu_backend_fill_op: missing 'fill_value' attr");
  }
  const int64_t fill_value = std::get<int64_t>(it->second);

  frame::Tensor& out = ctx.outputs[0];
  int32_t* data = out.data<int32_t>();
  for (int64_t i = 0; i < out.numel(); ++i) {
    data[i] = static_cast<int32_t>(fill_value);
  }
  return frame::Status::ok();
}

}  // namespace

FRAME_REGISTER_KERNEL("test_cpu_backend_fill_op", frame::kCpuBackendName, FillKernel);

namespace {

using frame::Device;
using frame::ErrorCode;
using frame::hal::Backend;
using frame::hal::BackendRegistry;

class CpuBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const frame::Result<Backend*> result = BackendRegistry::instance().get(frame::kCpuBackendName);
    ASSERT_TRUE(result.is_ok());
    backend_ = result.value();
    device_ = frame::cpu_device();
  }

  Backend* backend_ = nullptr;
  Device device_{};
};

// ---- device_count 语义 -------------------------------------------------------

TEST_F(CpuBackendTest, DeviceCountIsOne) {
  const frame::Result<int32_t> count = backend_->device_count();
  ASSERT_TRUE(count.is_ok());
  EXPECT_EQ(count.value(), 1);
}

// ---- 非法设备拒绝路径:backend 字段错误 -------------------------------------

TEST_F(CpuBackendTest, CreateStreamRejectsWrongDeviceBackend) {
  const frame::Result<std::unique_ptr<frame::hal::Stream>> result =
      backend_->create_stream(Device{"not_cpu", 0});
  EXPECT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("not_cpu"), std::string_view::npos);
}

TEST_F(CpuBackendTest, CreateEventRejectsWrongDeviceBackend) {
  const frame::Result<std::unique_ptr<frame::hal::Event>> result =
      backend_->create_event(Device{"not_cpu", 0});
  EXPECT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
}

TEST_F(CpuBackendTest, AllocatorRejectsWrongDeviceBackendReturnsNullptr) {
  // allocator() 签名不带 Status/Result(见 include/frame/hal/allocator.h),
  // 违例经 stderr 输出诊断后返回 nullptr,是该接口下"无可用分配器"的唯一表达。
  EXPECT_EQ(backend_->allocator(Device{"not_cpu", 0}), nullptr);
}

TEST_F(CpuBackendTest, CopyRejectsWrongDstDeviceBackend) {
  int32_t dst = 0;
  int32_t src = 0;
  const frame::Status status =
      backend_->copy(&dst, Device{"not_cpu", 0}, &src, device_, sizeof(int32_t), nullptr);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
}

TEST_F(CpuBackendTest, LaunchRejectsWrongDeviceBackend) {
  frame::hal::KernelInvocation invocation;
  invocation.op = "test_cpu_backend_fill_op";
  invocation.device = Device{"not_cpu", 0};
  const frame::Status status = backend_->launch(invocation, nullptr);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
}

// ---- 非法设备拒绝路径:index 字段错误 ---------------------------------------

TEST_F(CpuBackendTest, CreateStreamRejectsNonZeroDeviceIndex) {
  const frame::Result<std::unique_ptr<frame::hal::Stream>> result =
      backend_->create_stream(Device{frame::kCpuBackendName, 1});
  EXPECT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
}

TEST_F(CpuBackendTest, CreateEventRejectsNonZeroDeviceIndex) {
  const frame::Result<std::unique_ptr<frame::hal::Event>> result =
      backend_->create_event(Device{frame::kCpuBackendName, 1});
  EXPECT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
}

TEST_F(CpuBackendTest, AllocatorRejectsNonZeroDeviceIndexReturnsNullptr) {
  EXPECT_EQ(backend_->allocator(Device{frame::kCpuBackendName, 1}), nullptr);
}

TEST_F(CpuBackendTest, CopyRejectsNonZeroSrcDeviceIndex) {
  int32_t dst = 0;
  int32_t src = 0;
  const frame::Status status = backend_->copy(
      &dst, device_, &src, Device{frame::kCpuBackendName, 1}, sizeof(int32_t), nullptr);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
}

TEST_F(CpuBackendTest, LaunchRejectsNonZeroDeviceIndex) {
  frame::hal::KernelInvocation invocation;
  invocation.op = "test_cpu_backend_fill_op";
  invocation.device = Device{frame::kCpuBackendName, 1};
  const frame::Status status = backend_->launch(invocation, nullptr);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
}

// ---- allocator:非法参数 ----------------------------------------------------

TEST_F(CpuBackendTest, AllocatorAllocateRejectsNonPowerOfTwoAlignment) {
  frame::hal::Allocator* allocator = backend_->allocator(device_);
  ASSERT_NE(allocator, nullptr);

  const frame::Result<void*> result = allocator->allocate(/*bytes=*/64, /*alignment=*/3);
  EXPECT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
}

TEST_F(CpuBackendTest, AllocatorAllocateRejectsZeroBytes) {
  frame::hal::Allocator* allocator = backend_->allocator(device_);
  ASSERT_NE(allocator, nullptr);

  const frame::Result<void*> result = allocator->allocate(/*bytes=*/0, /*alignment=*/64);
  EXPECT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
}

// ---- copy:字节往返一致 / 重叠区域 memmove 语义 / 空指针报错 ----------------

TEST_F(CpuBackendTest, CopyRoundTripPreservesBytes) {
  frame::hal::Allocator* allocator = backend_->allocator(device_);
  ASSERT_NE(allocator, nullptr);

  constexpr size_t kBytes = 128;
  const frame::Result<void*> src_result = allocator->allocate(kBytes, 64);
  const frame::Result<void*> dst_result = allocator->allocate(kBytes, 64);
  ASSERT_TRUE(src_result.is_ok());
  ASSERT_TRUE(dst_result.is_ok());
  void* src = src_result.value();
  void* dst = dst_result.value();

  std::vector<unsigned char> pattern(kBytes);
  for (size_t i = 0; i < kBytes; ++i) pattern[i] = static_cast<unsigned char>(i * 7 + 1);
  std::memcpy(src, pattern.data(), kBytes);

  const frame::Status status = backend_->copy(dst, device_, src, device_, kBytes, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_EQ(std::memcmp(dst, pattern.data(), kBytes), 0);

  allocator->deallocate(src);
  allocator->deallocate(dst);
}

TEST_F(CpuBackendTest, CopyToleratesOverlappingRegionsWithMemmoveSemantics) {
  frame::hal::Allocator* allocator = backend_->allocator(device_);
  ASSERT_NE(allocator, nullptr);

  constexpr size_t kBytes = 128;
  constexpr size_t kOverlapBytes = 64;
  constexpr size_t kShift = 16;  // dst = buffer + kShift,与 src = buffer 重叠。
  const frame::Result<void*> buffer_result = allocator->allocate(kBytes, 64);
  ASSERT_TRUE(buffer_result.is_ok());
  auto* buffer = static_cast<unsigned char*>(buffer_result.value());

  std::vector<unsigned char> pattern(kBytes);
  for (size_t i = 0; i < kBytes; ++i) pattern[i] = static_cast<unsigned char>(i);
  std::memcpy(buffer, pattern.data(), kBytes);

  // 期望值:用独立缓冲区跑一次标准库 memmove 计算(验证的是"dst=buffer+kShift、
  // src=buffer 的重叠拷贝结果与 memmove 语义一致"这一可观察契约,而非绑定具体
  // 实现细节)。
  std::vector<unsigned char> expected = pattern;
  std::memmove(expected.data() + kShift, expected.data(), kOverlapBytes);

  const frame::Status status =
      backend_->copy(buffer + kShift, device_, buffer, device_, kOverlapBytes, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_EQ(std::memcmp(buffer, expected.data(), kBytes), 0);

  allocator->deallocate(buffer);
}

TEST_F(CpuBackendTest, CopyRejectsNullPointersWhenBytesPositive) {
  const frame::Status status = backend_->copy(nullptr, device_, nullptr, device_,
                                              /*bytes=*/16, nullptr);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
}

// ---- launch:已注册 kernel 执行成功且效果可观察 / 未注册 op 报错 -----------

TEST_F(CpuBackendTest, LaunchExecutesRegisteredKernelWithObservableSideEffect) {
  frame::hal::Allocator* allocator = backend_->allocator(device_);
  ASSERT_NE(allocator, nullptr);

  const frame::Result<frame::Tensor> output_result =
      frame::Tensor::empty(frame::Shape({4}), frame::DType::of<int32_t>(), device_, *allocator);
  ASSERT_TRUE(output_result.is_ok());
  std::vector<frame::Tensor> outputs{output_result.value()};

  const std::unordered_map<std::string, frame::ir::AttrValue> attrs{
      {"fill_value", int64_t{42}},
  };

  frame::hal::KernelInvocation invocation;
  invocation.op = "test_cpu_backend_fill_op";
  invocation.outputs = outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;

  const frame::Result<std::unique_ptr<frame::hal::Stream>> stream_result =
      backend_->create_stream(device_);
  ASSERT_TRUE(stream_result.is_ok());

  const frame::Status status = backend_->launch(invocation, stream_result.value().get());
  ASSERT_TRUE(status.is_ok());

  const int32_t* data = outputs[0].data<int32_t>();
  for (int64_t i = 0; i < outputs[0].numel(); ++i) {
    EXPECT_EQ(data[i], 42);
  }
}

TEST_F(CpuBackendTest, LaunchUnregisteredOpReturnsErrorWithOpAndBackendNameInMessage) {
  frame::hal::KernelInvocation invocation;
  invocation.op = "test_cpu_backend_never_registered_op";
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kNotFound);
  // ARCH-031:未注册 op 的错误消息须含算子名与后端名(禁静默降级)。
  EXPECT_NE(status.message().find("test_cpu_backend_never_registered_op"), std::string_view::npos);
  EXPECT_NE(status.message().find("cpu"), std::string_view::npos);
}

}  // namespace
