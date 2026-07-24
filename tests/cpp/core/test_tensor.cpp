// tensor 模块单测:empty 元数据 + 底层内存指针、data<T>() dtype 不符触发
// FRAME_CHECK fatal、view 共享底层 storage 且 numel 不符触发 FRAME_CHECK、
// offset=0 语义。全程使用 host 内存的 FakeAllocator,不依赖任何已注册后端
// (cpu 后端真实 Allocator 要到 M4 才落地)。
#include <gtest/gtest.h>

#include <frame/core/tensor.h>

#include "fake_allocator.h"

namespace {

using frame::cpu_device;
using frame::DType;
using frame::Shape;
using frame::Tensor;

TEST(TensorTest, EmptyAllocatesMetadataAndBackingMemory) {
  frame::testing::FakeAllocator allocator;
  const Shape shape({2, 3, 4});
  const DType dtype = DType::of<float>();
  const frame::Device device = cpu_device();

  frame::Result<Tensor> result = Tensor::empty(shape, dtype, device, allocator);
  ASSERT_TRUE(result.is_ok());
  const Tensor& tensor = result.value();

  EXPECT_EQ(tensor.shape(), shape);
  EXPECT_EQ(tensor.dtype(), dtype);
  EXPECT_EQ(tensor.device(), device);
  EXPECT_EQ(tensor.numel(), 24);
  EXPECT_EQ(tensor.strides(), frame::row_major_strides(shape));
  ASSERT_NE(tensor.raw_data(), nullptr);
  // 未分片(offset=0)张量的 raw_data 必须与 allocator 返回的原始指针完全一致。
  EXPECT_EQ(tensor.raw_data(), allocator.last_allocated());
  EXPECT_EQ(allocator.allocate_count(), 1);
}

TEST(TensorTest, DataTemplateReturnsTypedPointerForMatchingDtype) {
  frame::testing::FakeAllocator allocator;
  frame::Result<Tensor> result =
      Tensor::empty(Shape({2, 3}), DType::of<float>(), cpu_device(), allocator);
  ASSERT_TRUE(result.is_ok());
  Tensor tensor = result.value();

  float* typed = tensor.data<float>();
  EXPECT_EQ(static_cast<void*>(typed), tensor.raw_data());
}

TEST(TensorTest, DataTemplateAbortsOnDtypeMismatch) {
  frame::testing::FakeAllocator allocator;
  frame::Result<Tensor> result =
      Tensor::empty(Shape({2, 3}), DType::of<float>(), cpu_device(), allocator);
  ASSERT_TRUE(result.is_ok());
  Tensor tensor = result.value();

  // dtype 实为 float32,以 double(float64)取强类型指针须触发 FRAME_CHECK fatal
  // (include/frame/core/macros.h:FRAME_CHECK 失败即 fprintf + abort)。
  EXPECT_DEATH({ tensor.data<double>(); }, "FRAME_CHECK failed");
}

TEST(TensorTest, ViewSharesUnderlyingStorage) {
  frame::testing::FakeAllocator allocator;
  frame::Result<Tensor> result =
      Tensor::empty(Shape({2, 3, 4}), DType::of<float>(), cpu_device(), allocator);  // 24 个元素
  ASSERT_TRUE(result.is_ok());
  // 仅经 view()/raw_data() 等 const 方法只读访问,不依赖独立于 result 的拷贝
  // 语义,故取 const 引用而非拷贝(performance-unnecessary-copy-initialization)。
  const Tensor& tensor = result.value();

  Tensor view = tensor.view(Shape({4, 6}));  // 同为 24 个元素,合法 view

  EXPECT_EQ(view.raw_data(), tensor.raw_data());  // 共享底层 storage,不拷贝数据
  EXPECT_EQ(view.numel(), tensor.numel());
  EXPECT_EQ(view.dtype(), tensor.dtype());
  EXPECT_EQ(view.shape(), Shape({4, 6}));
  EXPECT_EQ(view.strides(), frame::row_major_strides(Shape({4, 6})));
}

TEST(TensorTest, ViewAbortsOnNumelMismatch) {
  frame::testing::FakeAllocator allocator;
  frame::Result<Tensor> result =
      Tensor::empty(Shape({2, 3}), DType::of<float>(), cpu_device(), allocator);  // 6 个元素
  ASSERT_TRUE(result.is_ok());
  // 仅经 view() 这一 const 方法只读访问,不依赖独立于 result 的拷贝语义,
  // 故取 const 引用而非拷贝(performance-unnecessary-copy-initialization)。
  const Tensor& tensor = result.value();

  // 5 != 6,view() 内部 FRAME_CHECK(new_shape.numel() == shape_.numel()) 须 fatal。
  EXPECT_DEATH({ tensor.view(Shape({5})); }, "FRAME_CHECK failed");
}

// offset_ 语义(单位:元素个数,非字节,见 tensor.h 成员注释)。v0 的 Tensor 只有
// empty()/view() 两条公开构造路径:empty() 固定 offset_=0,view() 直接沿用父
// 张量的 offset_(而父张量的 offset_ 恒为 0),因此当前无法经公开 API 构造出
// 非零 offset 的 Tensor。本用例覆盖"offset=0 时 raw_data() 等于 storage 首地址"
// 这一可观察语义;非零 offset 分支(raw_data() 按 offset*itemsize() 字节偏移)
// 要等切片类 API(如 narrow/slice)落地后才可测,已记入测试报告。
TEST(TensorTest, ZeroOffsetRawDataMatchesStorageOrigin) {
  frame::testing::FakeAllocator allocator;
  frame::Result<Tensor> result =
      Tensor::empty(Shape({3, 3}), DType::of<float>(), cpu_device(), allocator);
  ASSERT_TRUE(result.is_ok());
  const Tensor& tensor = result.value();

  EXPECT_EQ(tensor.raw_data(), allocator.last_allocated());
}

}  // namespace
