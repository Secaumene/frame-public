// 批边界两态测试组(M20 批2 Task 3,docs/architecture/nn-design.md ARCH-076 /
// 判定方法段"批边界(整除/不整除 × drop_last 两态)"一项;逐字验证
// include/frame/data/dataloader.h 头注释①③契约)。覆盖:
//   1. 样本数 10、batch_size 4:drop_last=true -> 每 epoch 2 批(8 样本,尾 2
//      样本确认被丢弃,不出现在任何批内容里);
//   2. drop_last=false -> 3 批且尾批 2 样本(shape[0]==2,内容为原尾 2 行);
//   3. 极端 batch_size==样本数(单批覆盖全部样本)与 batch_size==1(每批 1
//      样本)各一断言;
//   4. epoch 边界哨兵语义逐字验证:next() 在批序耗尽时返回空 optional 且
//      "原子推进到下一 epoch"(哨兵调用返回时 epoch() 已自增,不需要额外一次
//      next() 调用才能观察到),随后 next() 产出新 epoch 首批。

#include <cstdint>
#include <gtest/gtest.h>
#include <optional>
#include <vector>

#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/data/dataloader.h>
#include <frame/data/dataset.h>

#include "data_test_helpers.h"

namespace {

using frame::Result;
using frame::Shape;
using frame::Tensor;
using frame::data::DataLoader;
using frame::data::DataLoaderOptions;
using frame::data::TensorDataset;
using frame::testing::data::DataLoaderTestBase;

constexpr int64_t kNumSamples = 10;

class BatchBoundaryTest : public DataLoaderTestBase {
 protected:
  // 单列数据集,行 i = [i](恒等序下批内容与源行号一一对应,便于验证丢弃/尾批)。
  TensorDataset BuildDataset() {
    std::vector<float> values(static_cast<size_t>(kNumSamples));
    for (int64_t row = 0; row < kNumSamples; ++row) {
      values[static_cast<size_t>(row)] = static_cast<float>(row);
    }
    Tensor column = MakeFloatTensor(values, Shape({kNumSamples, 1}));
    Result<TensorDataset> dataset_result = TensorDataset::create({column});
    return std::move(dataset_result.value());
  }

  static std::vector<int64_t> BatchRowIndices(const Tensor& batch) {
    std::vector<int64_t> rows;
    const float* data = static_cast<const float*>(batch.raw_data());
    const int64_t batch_len = batch.shape().dim(0);
    rows.reserve(static_cast<size_t>(batch_len));
    for (int64_t i = 0; i < batch_len; ++i) rows.push_back(static_cast<int64_t>(data[i]));
    return rows;
  }
};

TEST_F(BatchBoundaryTest, DropLastTrueYieldsTwoFullBatchesAndDropsShortTail) {
  const DataLoaderOptions options{/*batch_size=*/4, /*shuffle=*/false, /*seed=*/0,
                                  /*drop_last=*/true};
  Result<DataLoader> loader_result = DataLoader::create(BuildDataset(), options);
  ASSERT_TRUE(loader_result.is_ok()) << loader_result.status().message();
  DataLoader loader = std::move(loader_result.value());

  ASSERT_EQ(loader.batches_per_epoch(), 2);  // 10/4 -> 2(整除批数),drop_last 丢弃尾 2 样本

  // 具名 const 引用绑定同一 optional 对象,使 has_value() 检查与随后的解引用
  // 落在同一变量上(而非各自重复调用 X.value()),满足 clang-tidy
  // bugprone-unchecked-optional-access 的检查-使用配对识别;本文件全部
  // next() 消费点统一采用此写法。
  Result<std::optional<std::vector<Tensor>>> batch0 = loader.next(*allocator_);
  ASSERT_TRUE(batch0.is_ok());
  const std::optional<std::vector<Tensor>>& batch0_opt = batch0.value();
  FRAME_ASSERT_HAS_VALUE_OR_RETURN(batch0_opt, "batch0 expected to have a value");
  const Tensor& batch0_tensor = (*batch0_opt)[0];
  EXPECT_EQ(batch0_tensor.shape().dim(0), 4);
  EXPECT_EQ(BatchRowIndices(batch0_tensor), (std::vector<int64_t>{0, 1, 2, 3}));

  Result<std::optional<std::vector<Tensor>>> batch1 = loader.next(*allocator_);
  ASSERT_TRUE(batch1.is_ok());
  const std::optional<std::vector<Tensor>>& batch1_opt = batch1.value();
  FRAME_ASSERT_HAS_VALUE_OR_RETURN(batch1_opt, "batch1 expected to have a value");
  const Tensor& batch1_tensor = (*batch1_opt)[0];
  EXPECT_EQ(batch1_tensor.shape().dim(0), 4);
  EXPECT_EQ(BatchRowIndices(batch1_tensor), (std::vector<int64_t>{4, 5, 6, 7}));

  // 尾 2 样本(行 8、9)被 drop_last 丢弃:全 epoch 只消费了 8 个样本,行 8/9
  // 未出现在任何已产出批次里(上面两个断言已穷举全部批内容,不含 8/9)。

  Result<std::optional<std::vector<Tensor>>> sentinel = loader.next(*allocator_);
  ASSERT_TRUE(sentinel.is_ok());
  EXPECT_FALSE(sentinel.value().has_value());
  EXPECT_EQ(loader.epoch(), 1u);
}

TEST_F(BatchBoundaryTest, DropLastFalseYieldsThreeBatchesWithShortTailOfTwo) {
  const DataLoaderOptions options{/*batch_size=*/4, /*shuffle=*/false, /*seed=*/0,
                                  /*drop_last=*/false};
  Result<DataLoader> loader_result = DataLoader::create(BuildDataset(), options);
  ASSERT_TRUE(loader_result.is_ok()) << loader_result.status().message();
  DataLoader loader = std::move(loader_result.value());

  ASSERT_EQ(loader.batches_per_epoch(), 3);  // ceil(10/4) -> 3,尾批长度 = 10 % 4 = 2

  Result<std::optional<std::vector<Tensor>>> batch0 = loader.next(*allocator_);
  const std::optional<std::vector<Tensor>>& batch0_opt = batch0.value();
  FRAME_ASSERT_HAS_VALUE_OR_RETURN(batch0_opt, "batch0 expected to have a value");
  EXPECT_EQ((*batch0_opt)[0].shape().dim(0), 4);
  EXPECT_EQ(BatchRowIndices((*batch0_opt)[0]), (std::vector<int64_t>{0, 1, 2, 3}));

  Result<std::optional<std::vector<Tensor>>> batch1 = loader.next(*allocator_);
  const std::optional<std::vector<Tensor>>& batch1_opt = batch1.value();
  FRAME_ASSERT_HAS_VALUE_OR_RETURN(batch1_opt, "batch1 expected to have a value");
  EXPECT_EQ((*batch1_opt)[0].shape().dim(0), 4);
  EXPECT_EQ(BatchRowIndices((*batch1_opt)[0]), (std::vector<int64_t>{4, 5, 6, 7}));

  Result<std::optional<std::vector<Tensor>>> batch2 = loader.next(*allocator_);
  const std::optional<std::vector<Tensor>>& batch2_opt = batch2.value();
  FRAME_ASSERT_HAS_VALUE_OR_RETURN(batch2_opt, "batch2 (tail) expected to have a value");
  const Tensor& tail = (*batch2_opt)[0];
  EXPECT_EQ(tail.shape().dim(0), 2);  // 尾批 shape[0]==2
  EXPECT_EQ(BatchRowIndices(tail), (std::vector<int64_t>{8, 9}));

  Result<std::optional<std::vector<Tensor>>> sentinel = loader.next(*allocator_);
  ASSERT_TRUE(sentinel.is_ok());
  EXPECT_FALSE(sentinel.value().has_value());
  EXPECT_EQ(loader.epoch(), 1u);
}

TEST_F(BatchBoundaryTest, BatchSizeEqualsDatasetSizeYieldsSingleFullBatch) {
  // 极端①:batch_size == 样本数,单批覆盖全部样本。
  const DataLoaderOptions options{/*batch_size=*/kNumSamples, /*shuffle=*/false, /*seed=*/0,
                                  /*drop_last=*/false};
  Result<DataLoader> loader_result = DataLoader::create(BuildDataset(), options);
  ASSERT_TRUE(loader_result.is_ok()) << loader_result.status().message();
  DataLoader loader = std::move(loader_result.value());

  ASSERT_EQ(loader.batches_per_epoch(), 1);

  Result<std::optional<std::vector<Tensor>>> batch = loader.next(*allocator_);
  ASSERT_TRUE(batch.is_ok());
  const std::optional<std::vector<Tensor>>& batch_opt = batch.value();
  FRAME_ASSERT_HAS_VALUE_OR_RETURN(batch_opt, "batch expected to have a value");
  const Tensor& single_batch = (*batch_opt)[0];
  EXPECT_EQ(single_batch.shape().dim(0), kNumSamples);
  std::vector<int64_t> expected_rows(static_cast<size_t>(kNumSamples));
  for (int64_t i = 0; i < kNumSamples; ++i) expected_rows[static_cast<size_t>(i)] = i;
  EXPECT_EQ(BatchRowIndices(single_batch), expected_rows);

  Result<std::optional<std::vector<Tensor>>> sentinel = loader.next(*allocator_);
  ASSERT_TRUE(sentinel.is_ok());
  EXPECT_FALSE(sentinel.value().has_value());
}

TEST_F(BatchBoundaryTest, BatchSizeOneYieldsOneSamplePerBatch) {
  // 极端②:batch_size == 1,每批恰 1 样本,批数 == 样本数。
  const DataLoaderOptions options{/*batch_size=*/1, /*shuffle=*/false, /*seed=*/0,
                                  /*drop_last=*/false};
  Result<DataLoader> loader_result = DataLoader::create(BuildDataset(), options);
  ASSERT_TRUE(loader_result.is_ok()) << loader_result.status().message();
  DataLoader loader = std::move(loader_result.value());

  ASSERT_EQ(loader.batches_per_epoch(), kNumSamples);

  for (int64_t row = 0; row < kNumSamples; ++row) {
    Result<std::optional<std::vector<Tensor>>> batch = loader.next(*allocator_);
    ASSERT_TRUE(batch.is_ok());
    const std::optional<std::vector<Tensor>>& batch_opt = batch.value();
    FRAME_ASSERT_HAS_VALUE_OR_RETURN(batch_opt, "row " << row << " expected a value");
    const Tensor& single = (*batch_opt)[0];
    EXPECT_EQ(single.shape().dim(0), 1) << "row " << row;
    EXPECT_EQ(BatchRowIndices(single), (std::vector<int64_t>{row})) << "row " << row;
  }

  Result<std::optional<std::vector<Tensor>>> sentinel = loader.next(*allocator_);
  ASSERT_TRUE(sentinel.is_ok());
  EXPECT_FALSE(sentinel.value().has_value());
}

TEST_F(BatchBoundaryTest, NextReturnsNulloptSentinelThenAtomicallyAdvancesEpochBeforeNextBatch) {
  // 头注释①字面契约:next() 在批序耗尽时返回 std::nullopt,并把内部状态"原子
  // 推进到下一 epoch"——即哨兵调用返回的那一刻 epoch() 已自增,不需要再调用一
  // 次 next() 才能观察到;紧随其后的 next() 调用即产出新 epoch 首批。
  const DataLoaderOptions options{/*batch_size=*/4, /*shuffle=*/false, /*seed=*/0,
                                  /*drop_last=*/true};
  Result<DataLoader> loader_result = DataLoader::create(BuildDataset(), options);
  ASSERT_TRUE(loader_result.is_ok()) << loader_result.status().message();
  DataLoader loader = std::move(loader_result.value());

  EXPECT_EQ(loader.epoch(), 0u);

  Result<std::optional<std::vector<Tensor>>> batch0 = loader.next(*allocator_);
  ASSERT_TRUE(batch0.value().has_value());
  EXPECT_EQ(loader.epoch(), 0u);  // 仍在 epoch 0 内(批序未耗尽)

  Result<std::optional<std::vector<Tensor>>> batch1 = loader.next(*allocator_);
  ASSERT_TRUE(batch1.value().has_value());
  EXPECT_EQ(loader.epoch(), 0u);

  // 第三次调用:批序已耗尽(batches_per_epoch()==2),哨兵触发。
  Result<std::optional<std::vector<Tensor>>> sentinel = loader.next(*allocator_);
  ASSERT_TRUE(sentinel.is_ok());
  ASSERT_FALSE(sentinel.value().has_value());
  EXPECT_EQ(loader.epoch(), 1u);  // 哨兵调用返回时已自增,原子推进语义

  // 紧随哨兵之后的 next() 产出新 epoch(epoch 1)首批;shuffle=false 恒等序,
  // 新 epoch 首批内容与 epoch 0 首批相同(行 0-3),验证"清洁重启"。
  Result<std::optional<std::vector<Tensor>>> new_epoch_batch0 = loader.next(*allocator_);
  ASSERT_TRUE(new_epoch_batch0.is_ok());
  const std::optional<std::vector<Tensor>>& new_epoch_batch0_opt = new_epoch_batch0.value();
  FRAME_ASSERT_HAS_VALUE_OR_RETURN(new_epoch_batch0_opt, "new epoch batch0 expected a value");
  EXPECT_EQ(loader.epoch(), 1u);
  EXPECT_EQ(BatchRowIndices((*new_epoch_batch0_opt)[0]), (std::vector<int64_t>{0, 1, 2, 3}));
}

}  // namespace
