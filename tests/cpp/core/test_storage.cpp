// storage 模块单测:fake allocator 验证"分配即持有、析构即释放"、移动语义下无
// 双重释放、0 字节请求合法。cpu 后端真实 Allocator 要到 M4 才落地,本文件全程
// 只用 host 内存的 FakeAllocator,不依赖任何已注册后端。
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <utility>

#include <frame/core/status.h>
#include <frame/core/storage.h>

#include "fake_allocator.h"

namespace {

using frame::cpu_device;
using frame::Result;
using frame::Storage;

TEST(StorageTest, AllocateHoldsMemoryAndDestructorReleasesExactlyOnce) {
  frame::testing::FakeAllocator allocator;
  {
    Result<std::shared_ptr<Storage>> result =
        Storage::allocate(allocator, /*bytes=*/256, /*alignment=*/64, cpu_device());
    ASSERT_TRUE(result.is_ok());
    // 仅作只读访问(data()/nbytes()/device()),未依赖独立于 result 的所有权
    // 语义,故取 const 引用而非拷贝(performance-unnecessary-copy-initialization)。
    const std::shared_ptr<Storage>& storage = result.value();
    EXPECT_NE(storage->data(), nullptr);
    EXPECT_EQ(storage->nbytes(), 256u);
    EXPECT_EQ(storage->device(), cpu_device());
    EXPECT_EQ(allocator.allocate_count(), 1);
    EXPECT_EQ(allocator.deallocate_count(), 0);
  }
  // 离开作用域,shared_ptr 引用计数归零,Storage 析构应触发唯一一次 deallocate。
  EXPECT_EQ(allocator.deallocate_count(), 1);
}

TEST(StorageTest, ZeroByteAllocationIsLegalAndSkipsAllocator) {
  frame::testing::FakeAllocator allocator;
  Result<std::shared_ptr<Storage>> result =
      Storage::allocate(allocator, /*bytes=*/0, /*alignment=*/64, cpu_device());
  ASSERT_TRUE(result.is_ok());
  std::shared_ptr<Storage> storage = result.value();
  EXPECT_EQ(storage->data(), nullptr);
  EXPECT_EQ(storage->nbytes(), 0u);
  // 0 字节请求不下沉到 allocator(storage.cpp 注释:避免依赖具体后端对 0 字节
  // 请求的行为),故 allocate_count 应保持为 0。
  EXPECT_EQ(allocator.allocate_count(), 0);
  storage.reset();
  EXPECT_EQ(allocator.deallocate_count(), 0);
}

TEST(StorageTest, MoveConstructionTransfersOwnershipWithoutDoubleFree) {
  frame::testing::FakeAllocator allocator;
  void* original_data = nullptr;
  {
    Result<std::shared_ptr<Storage>> result =
        Storage::allocate(allocator, /*bytes=*/128, /*alignment=*/32, cpu_device());
    ASSERT_TRUE(result.is_ok());
    Storage moved_from = std::move(*result.value());
    original_data = moved_from.data();

    Storage moved_to(std::move(moved_from));
    EXPECT_EQ(moved_to.data(), original_data);
    // 故意访问移后源以验证其文档化状态契约(data()==nullptr,避免其析构重复
    // 释放);非误用,标记 NOLINT 抑制 bugprone-use-after-move。
    EXPECT_EQ(moved_from.data(), nullptr);  // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(allocator.allocate_count(), 1);
    EXPECT_EQ(allocator.deallocate_count(), 0);
    // moved_to / moved_from / result 内被移空的 Storage 在此作用域结束时依次析构。
  }
  // 三个 Storage 对象析构完毕:仅 moved_to 持有真实内存,应恰好释放一次
  // (既不能因移后源重复释放而变成 2,也不能因资源丢失而停留在 0)。
  EXPECT_EQ(allocator.deallocate_count(), 1);
}

TEST(StorageTest, MoveAssignmentReleasesPreviousMemoryExactlyOnce) {
  frame::testing::FakeAllocator allocator;
  {
    Result<std::shared_ptr<Storage>> first =
        Storage::allocate(allocator, /*bytes=*/64, /*alignment=*/16, cpu_device());
    Result<std::shared_ptr<Storage>> second =
        Storage::allocate(allocator, /*bytes=*/64, /*alignment=*/16, cpu_device());
    ASSERT_TRUE(first.is_ok());
    ASSERT_TRUE(second.is_ok());

    Storage target = std::move(*first.value());
    Storage source = std::move(*second.value());
    ASSERT_EQ(allocator.allocate_count(), 2);
    ASSERT_EQ(allocator.deallocate_count(), 0);

    // 覆盖 target 原持有的内存:operator= 应先释放自身旧内存,再窃取 source 的资源。
    target = std::move(source);
    EXPECT_EQ(allocator.deallocate_count(), 1);
    // 故意访问移后源以验证其文档化状态契约(data()==nullptr);非误用,
    // 标记 NOLINT 抑制 bugprone-use-after-move。
    EXPECT_EQ(source.data(), nullptr);  // NOLINT(bugprone-use-after-move)
    // target / source / first 与 second 内被移空的 Storage 在此作用域结束时依次析构。
  }
  // 两块内存(target 的旧内存 + target 最终持有的内存)各释放一次,合计 2 次,
  // 与 allocate_count 相等,证明既无泄漏也无双重释放。
  EXPECT_EQ(allocator.deallocate_count(), 2);
}

TEST(StorageTest, SelfMoveAssignmentIsNoOp) {
  frame::testing::FakeAllocator allocator;
  Result<std::shared_ptr<Storage>> result =
      Storage::allocate(allocator, /*bytes=*/32, /*alignment=*/16, cpu_device());
  ASSERT_TRUE(result.is_ok());
  Storage storage = std::move(*result.value());
  void* original_data = storage.data();

  // 经引用间接触发自赋值,规避编译器对字面自赋值的告警;验证 operator= 的
  // this == &other 保护分支不会误触发释放。
  Storage& self_reference = storage;
  storage = std::move(self_reference);

  EXPECT_EQ(storage.data(), original_data);
  EXPECT_EQ(allocator.deallocate_count(), 0);
}

TEST(StorageTest, AllocateReturnsOutOfMemoryStatusWhenAllocatorFails) {
  // 请求 SIZE_MAX 字节:必然分配失败,驱动 FakeAllocator::allocate 的 nothrow
  // 失败分支(fake_allocator.h),验证 hal/allocator.h「失败返回 Status,不抛
  // 异常」契约经 Storage::allocate 逐层透传给调用方。
  frame::testing::FakeAllocator allocator;
  Result<std::shared_ptr<Storage>> result = Storage::allocate(
      allocator, /*bytes=*/std::numeric_limits<size_t>::max(), /*alignment=*/64, cpu_device());
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), frame::ErrorCode::kOutOfMemory);
  EXPECT_EQ(allocator.allocate_count(), 1);
  EXPECT_EQ(allocator.deallocate_count(), 0);
}

}  // namespace
