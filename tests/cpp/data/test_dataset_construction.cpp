// frame::data 骨架编译冒烟占位测试(M20 批2 Task 3)。洗牌确定性/批切分
// 边界(numel 不整除 + drop_last 两态)/批组装数值正确三组完整测试见
// docs/architecture/nn-design.md §8,由 test-writer 后续交付;本文件仅验证
// TensorDataset::create 可用且 size() 符合样本数预期,防止本骨架 PR 是空测。

#include <gtest/gtest.h>
#include <utility>
#include <vector>

#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/tensor.h>
#include <frame/data/dataset.h>

#include "../core/fake_allocator.h"

namespace {

using frame::cpu_device;
using frame::DType;
using frame::Shape;
using frame::Tensor;
using frame::data::TensorDataset;

TEST(TensorDatasetSmoke, CreateWithFourSamplesTwoColumnsReportsCorrectSize) {
  frame::testing::FakeAllocator allocator;

  frame::Result<Tensor> features_result =
      Tensor::empty(Shape({4, 3}), DType::of<float>(), cpu_device(), allocator);
  ASSERT_TRUE(features_result.is_ok());
  frame::Result<Tensor> labels_result =
      Tensor::empty(Shape({4, 1}), DType::of<float>(), cpu_device(), allocator);
  ASSERT_TRUE(labels_result.is_ok());

  std::vector<Tensor> columns{features_result.value(), labels_result.value()};
  frame::Result<TensorDataset> dataset_result = TensorDataset::create(std::move(columns));
  ASSERT_TRUE(dataset_result.is_ok());

  const TensorDataset& dataset = dataset_result.value();
  EXPECT_EQ(dataset.size(), 4);
  EXPECT_EQ(dataset.columns().size(), static_cast<size_t>(2));
}

}  // namespace
