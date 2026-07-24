// core 子系统"转正"测试:Tensor::empty 分配张量元数据(shape/dtype/device)
// 与底层内存。cpu 后端真实 Allocator 要到 M4 才落地,这里用 FakeAllocator
// (host 内存 + 分配计数)自建,不依赖任何已注册后端。
#include <gtest/gtest.h>

#include <frame/core/tensor.h>

#include "fake_allocator.h"

TEST(CoreStub, TensorEmptyAllocatesMetadata) {
  frame::testing::FakeAllocator allocator;
  const frame::Shape shape({2, 3, 4});
  const frame::DType dtype = frame::DType::of<float>();
  const frame::Device device = frame::cpu_device();

  frame::Result<frame::Tensor> result = frame::Tensor::empty(shape, dtype, device, allocator);
  ASSERT_TRUE(result.is_ok());
  const frame::Tensor& tensor = result.value();

  EXPECT_EQ(tensor.shape(), shape);
  EXPECT_EQ(tensor.dtype(), dtype);
  EXPECT_EQ(tensor.device(), device);
  EXPECT_EQ(tensor.numel(), 24);
  EXPECT_EQ(tensor.strides(), frame::row_major_strides(shape));
  ASSERT_NE(tensor.raw_data(), nullptr);
  EXPECT_EQ(tensor.raw_data(), allocator.last_allocated());
  EXPECT_EQ(allocator.allocate_count(), 1);
}
