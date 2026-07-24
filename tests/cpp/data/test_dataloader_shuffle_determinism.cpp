// 洗牌确定性测试组(M20 批2 Task 3,docs/architecture/nn-design.md ARCH-076 /
// 判定方法段"洗牌确定性"一项;逐字验证 include/frame/data/dataloader.h 头注释
// ①②契约)。覆盖:
//   1. 同 seed 两个独立 DataLoader(独立分配的 Tensor,不共享 Storage)逐 epoch
//      逐批张量数值逐位相等;
//   2. shuffle=false 时批序恒为恒等序([0,1,...,size()-1]),跨 epoch 不变;
//   3. 同 seed 不同 epoch 序列不同(seed ^ epoch 混合生效),并与独立复算的
//      mt19937_64(seed^epoch)+shuffle oracle 逐一核对;
//   4. 不同 seed 首 epoch 序列不同,同样与 oracle 核对;
//   5. reset(epoch_index) 直接跳转与逐 epoch 顺序推进 next() 结果一致(头注释
//      ②机械契约字面量)。

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
using frame::testing::data::ComputeShuffledIndices;
using frame::testing::data::DataLoaderTestBase;

constexpr int64_t kNumSamples = 8;
constexpr int64_t kFeatureCols = 2;
constexpr int64_t kBatchSize = 3;  // 8/3 -> 批长 3,3,2(drop_last=false)

class ShuffleDeterminismTest : public DataLoaderTestBase {
 protected:
  // 单列数据集,行 i = [i*10, i*10+1](逐行可区分,便于从批内容反推源行号)。
  TensorDataset BuildDataset() {
    std::vector<float> values(static_cast<size_t>(kNumSamples * kFeatureCols));
    for (int64_t row = 0; row < kNumSamples; ++row) {
      for (int64_t col = 0; col < kFeatureCols; ++col) {
        values[static_cast<size_t>(row * kFeatureCols + col)] = static_cast<float>(row * 10 + col);
      }
    }
    Tensor features = MakeFloatTensor(values, Shape({kNumSamples, kFeatureCols}));
    Result<TensorDataset> dataset_result = TensorDataset::create({features});
    return std::move(dataset_result.value());
  }

  // 从批张量的第 local_row 行反推源样本行号(依据 BuildDataset 的可区分构造:
  // 该行首列值恰为 row*10)。
  static int64_t RowIndexFromBatch(const Tensor& batch, int64_t local_row) {
    const float* data = static_cast<const float*>(batch.raw_data());
    const float first_col = data[local_row * kFeatureCols];
    return static_cast<int64_t>(first_col) / 10;
  }
};

TEST_F(ShuffleDeterminismTest,
       SameSeedTwoIndependentLoadersProduceBitIdenticalBatchesAcrossEpochs) {
  constexpr uint64_t kSeed = 20260718ULL;
  const DataLoaderOptions options{kBatchSize, /*shuffle=*/true, kSeed, /*drop_last=*/false};

  Result<DataLoader> loader_a_result = DataLoader::create(BuildDataset(), options);
  ASSERT_TRUE(loader_a_result.is_ok()) << loader_a_result.status().message();
  DataLoader loader_a = std::move(loader_a_result.value());

  Result<DataLoader> loader_b_result = DataLoader::create(BuildDataset(), options);
  ASSERT_TRUE(loader_b_result.is_ok()) << loader_b_result.status().message();
  DataLoader loader_b = std::move(loader_b_result.value());

  ASSERT_EQ(loader_a.batches_per_epoch(), loader_b.batches_per_epoch());
  const int64_t batches_per_epoch = loader_a.batches_per_epoch();
  ASSERT_GT(batches_per_epoch, 0);

  for (int epoch = 0; epoch < 2; ++epoch) {
    for (int64_t batch_index = 0; batch_index < batches_per_epoch; ++batch_index) {
      Result<std::optional<std::vector<Tensor>>> result_a = loader_a.next(*allocator_);
      Result<std::optional<std::vector<Tensor>>> result_b = loader_b.next(*allocator_);
      ASSERT_TRUE(result_a.is_ok()) << result_a.status().message();
      ASSERT_TRUE(result_b.is_ok()) << result_b.status().message();
      // 具名 const 引用绑定同一 optional 对象,使 has_value() 检查与随后的解引用
      // 落在同一变量上(而非各自重复调用 result_a.value()),满足 clang-tidy
      // bugprone-unchecked-optional-access 的检查-使用配对识别;本文件全部
      // next() 消费点统一采用此写法。
      const std::optional<std::vector<Tensor>>& batch_opt_a = result_a.value();
      const std::optional<std::vector<Tensor>>& batch_opt_b = result_b.value();
      FRAME_ASSERT_HAS_VALUE_OR_RETURN(
          batch_opt_a, "epoch " << epoch << " batch " << batch_index << " loader_a");
      FRAME_ASSERT_HAS_VALUE_OR_RETURN(
          batch_opt_b, "epoch " << epoch << " batch " << batch_index << " loader_b");

      const std::vector<Tensor>& batch_a = *batch_opt_a;
      const std::vector<Tensor>& batch_b = *batch_opt_b;
      ASSERT_EQ(batch_a.size(), batch_b.size());
      for (size_t col = 0; col < batch_a.size(); ++col) {
        ASSERT_EQ(batch_a[col].shape(), batch_b[col].shape());
        const int64_t numel = batch_a[col].numel();
        const float* data_a = static_cast<const float*>(batch_a[col].raw_data());
        const float* data_b = static_cast<const float*>(batch_b[col].raw_data());
        for (int64_t i = 0; i < numel; ++i) {
          EXPECT_EQ(data_a[i], data_b[i]) << "epoch " << epoch << " batch " << batch_index
                                          << " col " << col << " element " << i;
        }
      }
    }
    // epoch 边界哨兵:两个 loader 同步耗尽当前 epoch 批序。
    Result<std::optional<std::vector<Tensor>>> sentinel_a = loader_a.next(*allocator_);
    Result<std::optional<std::vector<Tensor>>> sentinel_b = loader_b.next(*allocator_);
    ASSERT_TRUE(sentinel_a.is_ok());
    ASSERT_TRUE(sentinel_b.is_ok());
    EXPECT_FALSE(sentinel_a.value().has_value());
    EXPECT_FALSE(sentinel_b.value().has_value());
  }
}

TEST_F(ShuffleDeterminismTest, ShuffleFalseYieldsIdentityBatchOrderAcrossEpochs) {
  const DataLoaderOptions options{kBatchSize, /*shuffle=*/false, /*seed=*/999,
                                  /*drop_last=*/false};
  Result<DataLoader> loader_result = DataLoader::create(BuildDataset(), options);
  ASSERT_TRUE(loader_result.is_ok()) << loader_result.status().message();
  DataLoader loader = std::move(loader_result.value());

  const int64_t batches_per_epoch = loader.batches_per_epoch();
  for (int epoch = 0; epoch < 2; ++epoch) {
    int64_t expected_row = 0;
    for (int64_t batch_index = 0; batch_index < batches_per_epoch; ++batch_index) {
      Result<std::optional<std::vector<Tensor>>> result = loader.next(*allocator_);
      ASSERT_TRUE(result.is_ok()) << result.status().message();
      const std::optional<std::vector<Tensor>>& batch_opt = result.value();
      FRAME_ASSERT_HAS_VALUE_OR_RETURN(batch_opt, "epoch " << epoch << " batch " << batch_index);
      const Tensor& batch = (*batch_opt)[0];
      const int64_t batch_len = batch.shape().dim(0);
      for (int64_t local_row = 0; local_row < batch_len; ++local_row) {
        // shuffle=false 恒等序:批序即样本原序,行号严格递增。
        EXPECT_EQ(RowIndexFromBatch(batch, local_row), expected_row)
            << "epoch " << epoch << " batch " << batch_index << " local_row " << local_row;
        ++expected_row;
      }
    }
    EXPECT_EQ(expected_row, kNumSamples);
    Result<std::optional<std::vector<Tensor>>> sentinel = loader.next(*allocator_);
    ASSERT_TRUE(sentinel.is_ok());
    EXPECT_FALSE(sentinel.value().has_value());
  }
}

TEST_F(ShuffleDeterminismTest, SameSeedDifferentEpochsYieldDifferentOrderMatchingOracle) {
  constexpr uint64_t kSeed = 424242ULL;
  const DataLoaderOptions options{kBatchSize, /*shuffle=*/true, kSeed, /*drop_last=*/false};
  Result<DataLoader> loader_result = DataLoader::create(BuildDataset(), options);
  ASSERT_TRUE(loader_result.is_ok()) << loader_result.status().message();
  DataLoader loader = std::move(loader_result.value());

  const int64_t batches_per_epoch = loader.batches_per_epoch();

  // 消费 epoch0 全部批次,拼出实际消费的样本行号序列。
  std::vector<int64_t> consumed_epoch0;
  for (int64_t batch_index = 0; batch_index < batches_per_epoch; ++batch_index) {
    Result<std::optional<std::vector<Tensor>>> result = loader.next(*allocator_);
    ASSERT_TRUE(result.is_ok());
    const std::optional<std::vector<Tensor>>& batch_opt = result.value();
    FRAME_ASSERT_HAS_VALUE_OR_RETURN(batch_opt, "epoch0 batch " << batch_index);
    const Tensor& batch = (*batch_opt)[0];
    const int64_t batch_len = batch.shape().dim(0);
    for (int64_t local_row = 0; local_row < batch_len; ++local_row) {
      consumed_epoch0.push_back(RowIndexFromBatch(batch, local_row));
    }
  }
  Result<std::optional<std::vector<Tensor>>> sentinel0 = loader.next(*allocator_);
  ASSERT_TRUE(sentinel0.is_ok());
  ASSERT_FALSE(sentinel0.value().has_value());
  ASSERT_EQ(loader.epoch(), 1u);

  // 消费 epoch1 全部批次。
  std::vector<int64_t> consumed_epoch1;
  for (int64_t batch_index = 0; batch_index < batches_per_epoch; ++batch_index) {
    Result<std::optional<std::vector<Tensor>>> result = loader.next(*allocator_);
    ASSERT_TRUE(result.is_ok());
    const std::optional<std::vector<Tensor>>& batch_opt = result.value();
    FRAME_ASSERT_HAS_VALUE_OR_RETURN(batch_opt, "epoch1 batch " << batch_index);
    const Tensor& batch = (*batch_opt)[0];
    const int64_t batch_len = batch.shape().dim(0);
    for (int64_t local_row = 0; local_row < batch_len; ++local_row) {
      consumed_epoch1.push_back(RowIndexFromBatch(batch, local_row));
    }
  }

  const std::vector<int64_t> oracle_epoch0 = ComputeShuffledIndices(kNumSamples, kSeed, 0);
  const std::vector<int64_t> oracle_epoch1 = ComputeShuffledIndices(kNumSamples, kSeed, 1);
  EXPECT_EQ(consumed_epoch0, oracle_epoch0);
  EXPECT_EQ(consumed_epoch1, oracle_epoch1);
  // seed^epoch 混合生效:两 epoch 序列本身应不同(该 seed 已实测校准,见文件头
  // 注释——两个 mt19937_64 状态不同,产生不同排列)。
  EXPECT_NE(consumed_epoch0, consumed_epoch1);
}

TEST_F(ShuffleDeterminismTest, DifferentSeedsYieldDifferentFirstEpochOrderMatchingOracle) {
  const DataLoaderOptions options_seed1{kBatchSize, /*shuffle=*/true, /*seed=*/1,
                                        /*drop_last=*/false};
  const DataLoaderOptions options_seed2{kBatchSize, /*shuffle=*/true, /*seed=*/2,
                                        /*drop_last=*/false};

  Result<DataLoader> loader1_result = DataLoader::create(BuildDataset(), options_seed1);
  ASSERT_TRUE(loader1_result.is_ok());
  DataLoader loader1 = std::move(loader1_result.value());
  Result<DataLoader> loader2_result = DataLoader::create(BuildDataset(), options_seed2);
  ASSERT_TRUE(loader2_result.is_ok());
  DataLoader loader2 = std::move(loader2_result.value());

  const int64_t batches_per_epoch = loader1.batches_per_epoch();
  ASSERT_EQ(batches_per_epoch, loader2.batches_per_epoch());

  std::vector<int64_t> consumed1;
  std::vector<int64_t> consumed2;
  for (int64_t batch_index = 0; batch_index < batches_per_epoch; ++batch_index) {
    Result<std::optional<std::vector<Tensor>>> result1 = loader1.next(*allocator_);
    Result<std::optional<std::vector<Tensor>>> result2 = loader2.next(*allocator_);
    ASSERT_TRUE(result1.is_ok());
    ASSERT_TRUE(result2.is_ok());
    const std::optional<std::vector<Tensor>>& batch_opt1 = result1.value();
    const std::optional<std::vector<Tensor>>& batch_opt2 = result2.value();
    FRAME_ASSERT_HAS_VALUE_OR_RETURN(batch_opt1, "batch_index " << batch_index << " loader1");
    FRAME_ASSERT_HAS_VALUE_OR_RETURN(batch_opt2, "batch_index " << batch_index << " loader2");
    const Tensor& batch1 = (*batch_opt1)[0];
    const Tensor& batch2 = (*batch_opt2)[0];
    for (int64_t local_row = 0; local_row < batch1.shape().dim(0); ++local_row) {
      consumed1.push_back(RowIndexFromBatch(batch1, local_row));
    }
    for (int64_t local_row = 0; local_row < batch2.shape().dim(0); ++local_row) {
      consumed2.push_back(RowIndexFromBatch(batch2, local_row));
    }
  }

  EXPECT_EQ(consumed1, ComputeShuffledIndices(kNumSamples, 1, 0));
  EXPECT_EQ(consumed2, ComputeShuffledIndices(kNumSamples, 2, 0));
  EXPECT_NE(consumed1, consumed2);
}

TEST_F(ShuffleDeterminismTest, ResetToEpochIndexMatchesSequentialAdvancement) {
  // 头注释②机械契约字面量:"reset() 直接跳转与逐 epoch 顺序推进 next() 结果
  // 一致,均只取决于 seed ^ epoch_index"。
  constexpr uint64_t kSeed = 7ULL;
  constexpr uint64_t kTargetEpoch = 3ULL;
  const DataLoaderOptions options{kBatchSize, /*shuffle=*/true, kSeed, /*drop_last=*/false};

  // 路径一:逐 epoch 顺序推进(消费 epoch0/1/2 全部批次 + 各自哨兵)到达 epoch3。
  Result<DataLoader> sequential_result = DataLoader::create(BuildDataset(), options);
  ASSERT_TRUE(sequential_result.is_ok());
  DataLoader sequential = std::move(sequential_result.value());
  const int64_t batches_per_epoch = sequential.batches_per_epoch();
  for (uint64_t epoch = 0; epoch < kTargetEpoch; ++epoch) {
    for (int64_t batch_index = 0; batch_index < batches_per_epoch; ++batch_index) {
      Result<std::optional<std::vector<Tensor>>> result = sequential.next(*allocator_);
      ASSERT_TRUE(result.is_ok());
      ASSERT_TRUE(result.value().has_value());
    }
    Result<std::optional<std::vector<Tensor>>> sentinel = sequential.next(*allocator_);
    ASSERT_TRUE(sentinel.is_ok());
    ASSERT_FALSE(sentinel.value().has_value());
  }
  ASSERT_EQ(sequential.epoch(), kTargetEpoch);

  // 路径二:全新 loader 实例 reset(3) 直接跳转。
  Result<DataLoader> jumped_result = DataLoader::create(BuildDataset(), options);
  ASSERT_TRUE(jumped_result.is_ok());
  DataLoader jumped = std::move(jumped_result.value());
  jumped.reset(kTargetEpoch);
  ASSERT_EQ(jumped.epoch(), kTargetEpoch);

  for (int64_t batch_index = 0; batch_index < batches_per_epoch; ++batch_index) {
    Result<std::optional<std::vector<Tensor>>> sequential_batch = sequential.next(*allocator_);
    Result<std::optional<std::vector<Tensor>>> jumped_batch = jumped.next(*allocator_);
    ASSERT_TRUE(sequential_batch.is_ok());
    ASSERT_TRUE(jumped_batch.is_ok());
    const std::optional<std::vector<Tensor>>& sequential_batch_opt = sequential_batch.value();
    const std::optional<std::vector<Tensor>>& jumped_batch_opt = jumped_batch.value();
    FRAME_ASSERT_HAS_VALUE_OR_RETURN(sequential_batch_opt,
                                     "batch_index " << batch_index << " sequential");
    FRAME_ASSERT_HAS_VALUE_OR_RETURN(jumped_batch_opt, "batch_index " << batch_index << " jumped");

    const Tensor& seq_tensor = (*sequential_batch_opt)[0];
    const Tensor& jump_tensor = (*jumped_batch_opt)[0];
    ASSERT_EQ(seq_tensor.shape(), jump_tensor.shape());
    const int64_t numel = seq_tensor.numel();
    const float* seq_data = static_cast<const float*>(seq_tensor.raw_data());
    const float* jump_data = static_cast<const float*>(jump_tensor.raw_data());
    for (int64_t i = 0; i < numel; ++i) {
      EXPECT_EQ(seq_data[i], jump_data[i]) << "batch " << batch_index << " element " << i;
    }
  }
}

}  // namespace
