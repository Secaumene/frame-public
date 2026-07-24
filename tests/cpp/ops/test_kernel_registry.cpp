// KernelRegistry 单测:register_kernel 成功注册后 find 返回同一 KernelFn;重复
// (op, backend) 返回 kAlreadyExists 且消息含键;find 缺失返回 kNotFound;
// FRAME_REGISTER_KERNEL 宏静态注册生效。KernelContext 骨架期只有前向声明(见
// include/frame/ops/kernel_registry.h),桩内核函数只需匹配 KernelFn 签名、不
// 触碰 KernelContext 任何成员即可编译(引用不完整类型作函数形参在 C++ 中合法,
// 只要函数体不要求其完整性,如成员访问/sizeof)。ops 层注册键跨全体
// tests/cpp/ops/ 测试文件保持进程级唯一(见 test_ops_stub.cpp 顶部注释同一
// 纪律),本文件以 "test_kernel_" 前缀 op 名 + "test_backend_" 前缀后端名双重
// 唯一化。
#include <gtest/gtest.h>
#include <string_view>

#include <frame/ops/kernel_registry.h>

namespace {

frame::Status DummyKernelA(frame::ops::KernelContext&) { return frame::Status::ok(); }
frame::Status DummyKernelB(frame::ops::KernelContext&) { return frame::Status::ok(); }
frame::Status DummyKernelMacroTarget(frame::ops::KernelContext&) { return frame::Status::ok(); }

}  // namespace

FRAME_REGISTER_KERNEL("test_kernel_macro_op", "test_backend_macro", DummyKernelMacroTarget);

namespace {

using frame::ErrorCode;
using frame::ops::KernelFn;
using frame::ops::KernelRegistry;

TEST(KernelRegistryTest, RegisterThenFindReturnsSameKernelFn) {
  const frame::Status status = KernelRegistry::instance().register_kernel(
      "test_kernel_register_find_op", "test_backend_a", DummyKernelA);
  ASSERT_TRUE(status.is_ok());

  const frame::Result<KernelFn> found =
      KernelRegistry::instance().find("test_kernel_register_find_op", "test_backend_a");
  ASSERT_TRUE(found.is_ok());
  EXPECT_EQ(found.value(), DummyKernelA);
}

TEST(KernelRegistryTest, DuplicateOpBackendPairIsRejectedWithKeyInMessage) {
  KernelRegistry& registry = KernelRegistry::instance();
  ASSERT_TRUE(
      registry.register_kernel("test_kernel_duplicate_op", "test_backend_b", DummyKernelA).is_ok());

  const frame::Status status =
      registry.register_kernel("test_kernel_duplicate_op", "test_backend_b", DummyKernelB);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kAlreadyExists);
  EXPECT_NE(status.message().find("test_kernel_duplicate_op"), std::string_view::npos);
  EXPECT_NE(status.message().find("test_backend_b"), std::string_view::npos);
}

TEST(KernelRegistryTest, FindReturnsNotFoundForUnregisteredKey) {
  const frame::Result<KernelFn> found =
      KernelRegistry::instance().find("test_kernel_never_registered", "test_backend_a");
  EXPECT_FALSE(found.is_ok());
  EXPECT_EQ(found.status().code(), ErrorCode::kNotFound);
}

TEST(KernelRegistryTest, DifferentBackendSameOpAreDistinctKeys) {
  KernelRegistry& registry = KernelRegistry::instance();
  ASSERT_TRUE(
      registry.register_kernel("test_kernel_multi_backend_op", "test_backend_a", DummyKernelA)
          .is_ok());
  ASSERT_TRUE(
      registry.register_kernel("test_kernel_multi_backend_op", "test_backend_b", DummyKernelB)
          .is_ok());

  EXPECT_EQ(registry.find("test_kernel_multi_backend_op", "test_backend_a").value(), DummyKernelA);
  EXPECT_EQ(registry.find("test_kernel_multi_backend_op", "test_backend_b").value(), DummyKernelB);
}

TEST(KernelRegistryTest, FrameRegisterKernelMacroRegistersFindableKernel) {
  const frame::Result<KernelFn> found =
      KernelRegistry::instance().find("test_kernel_macro_op", "test_backend_macro");
  ASSERT_TRUE(found.is_ok());
  EXPECT_EQ(found.value(), DummyKernelMacroTarget);
}

}  // namespace
