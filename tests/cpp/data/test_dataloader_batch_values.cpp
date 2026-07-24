// 批数值正确测试组(M20 批2 Task 3,docs/architecture/nn-design.md ARCH-076 /
// 判定方法段"批组装数值正确"一项;逐字验证 include/frame/data/dataloader.h
// 头注释③"批组装"契约与 include/frame/data/dataset.h 头注释①~④校验契约)。
// 覆盖:
//   1. shuffle=false:2 列(features[10,3]、targets[10,1])填充可区分序列,
//      逐批逐元素与"按源行公式独立计算的期望值"精确相等(memcpy 语义下的精确
//      值搬运,非近似计算,故用 EXPECT_EQ 精确比较,不接入 BUILD-011 容差工具
//      ——该工具面向近似浮点比较,详见 data_test_helpers.h 头注释);
//   2. shuffle=true(固定 seed):批行内容与"按同 seed 独立复算的索引序" oracle
//      核对,验证行内容随索引正确搬运(非按物理批位置搬运);
//   3. TensorDataset 负例:列样本数不等 / 非 cpu 后端 / 空列——create() 报错且
//      英文消息含违例信息(dataset.h 头注释①③④机械契约)。

#include <cstdint>
#include <gtest/gtest.h>
#include <optional>
#include <string_view>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/data/dataloader.h>
#include <frame/data/dataset.h>

#include "data_test_helpers.h"

namespace {

using frame::DType;
using frame::ErrorCode;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::Tensor;
using frame::data::DataLoader;
using frame::data::DataLoaderOptions;
using frame::data::TensorDataset;
using frame::testing::data::ComputeShuffledIndices;
using frame::testing::data::DataLoaderTestBase;

constexpr int64_t kNumSamples = 10;
constexpr int64_t kFeatureCols = 3;
constexpr int64_t kTargetCols = 1;
constexpr int64_t kBatchSize = 4;  // 10/4 -> 批长 4,4,2

// 源行公式(独立于被测实现,供逐元素核对):
//   features[row][col] = row*10 + col + 1
//   targets[row][0]    = 1000 + row
float ExpectedFeatureValue(int64_t row, int64_t col) {
  return static_cast<float>(row * 10 + col + 1);
}
float ExpectedTargetValue(int64_t row) { return static_cast<float>(1000 + row); }

class BatchValueCorrectnessTest : public DataLoaderTestBase {
 protected:
  TensorDataset BuildTwoColumnDataset() {
    std::vector<float> feature_values(static_cast<size_t>(kNumSamples * kFeatureCols));
    for (int64_t row = 0; row < kNumSamples; ++row) {
      for (int64_t col = 0; col < kFeatureCols; ++col) {
        feature_values[static_cast<size_t>(row * kFeatureCols + col)] =
            ExpectedFeatureValue(row, col);
      }
    }
    std::vector<float> target_values(static_cast<size_t>(kNumSamples * kTargetCols));
    for (int64_t row = 0; row < kNumSamples; ++row) {
      target_values[static_cast<size_t>(row)] = ExpectedTargetValue(row);
    }

    Tensor features = MakeFloatTensor(feature_values, Shape({kNumSamples, kFeatureCols}));
    Tensor targets = MakeFloatTensor(target_values, Shape({kNumSamples, kTargetCols}));
    Result<TensorDataset> dataset_result = TensorDataset::create({features, targets});
    return std::move(dataset_result.value());
  }

  static void ExpectBatchRowMatchesSource(const Tensor& features_batch, const Tensor& targets_batch,
                                          int64_t local_row, int64_t source_row) {
    const float* feature_data = static_cast<const float*>(features_batch.raw_data());
    for (int64_t col = 0; col < kFeatureCols; ++col) {
      EXPECT_EQ(feature_data[local_row * kFeatureCols + col], ExpectedFeatureValue(source_row, col))
          << "local_row " << local_row << " col " << col << " source_row " << source_row;
    }
    const float* target_data = static_cast<const float*>(targets_batch.raw_data());
    EXPECT_EQ(target_data[local_row], ExpectedTargetValue(source_row))
        << "local_row " << local_row << " source_row " << source_row;
  }
};

// ---------------------------------------------------------------------------
// 1. shuffle=false:批内容与源行公式精确相等。
// ---------------------------------------------------------------------------

TEST_F(BatchValueCorrectnessTest, ShuffleFalseBatchesMatchSourceRowsExactly) {
  const DataLoaderOptions options{kBatchSize, /*shuffle=*/false, /*seed=*/0,
                                  /*drop_last=*/false};
  Result<DataLoader> loader_result = DataLoader::create(BuildTwoColumnDataset(), options);
  ASSERT_TRUE(loader_result.is_ok()) << loader_result.status().message();
  DataLoader loader = std::move(loader_result.value());

  ASSERT_EQ(loader.batches_per_epoch(), 3);  // ceil(10/4)

  int64_t next_source_row = 0;
  for (int64_t batch_index = 0; batch_index < loader.batches_per_epoch(); ++batch_index) {
    Result<std::optional<std::vector<Tensor>>> result = loader.next(*allocator_);
    ASSERT_TRUE(result.is_ok()) << result.status().message();
    // 具名 const 引用绑定同一 optional 对象,使 has_value() 检查与随后的解引用
    // 落在同一变量上(而非重复调用 result.value()),满足 clang-tidy
    // bugprone-unchecked-optional-access 的检查-使用配对识别。
    const std::optional<std::vector<Tensor>>& batch_opt = result.value();
    FRAME_ASSERT_HAS_VALUE_OR_RETURN(batch_opt,
                                     "batch_index " << batch_index << " expected a value");
    const std::vector<Tensor>& batch = *batch_opt;
    ASSERT_EQ(batch.size(), static_cast<size_t>(2));
    const Tensor& features_batch = batch[0];
    const Tensor& targets_batch = batch[1];
    const int64_t batch_len = features_batch.shape().dim(0);
    ASSERT_EQ(targets_batch.shape().dim(0), batch_len);

    for (int64_t local_row = 0; local_row < batch_len; ++local_row) {
      ExpectBatchRowMatchesSource(features_batch, targets_batch, local_row, next_source_row);
      ++next_source_row;
    }
  }
  EXPECT_EQ(next_source_row, kNumSamples);
}

// ---------------------------------------------------------------------------
// 2. shuffle=true(固定 seed):批行内容与独立复算索引序核对。
// ---------------------------------------------------------------------------

TEST_F(BatchValueCorrectnessTest, ShuffleTrueBatchesMatchSeedRecomputedIndices) {
  constexpr uint64_t kSeed = 777ULL;
  const DataLoaderOptions options{kBatchSize, /*shuffle=*/true, kSeed, /*drop_last=*/false};
  Result<DataLoader> loader_result = DataLoader::create(BuildTwoColumnDataset(), options);
  ASSERT_TRUE(loader_result.is_ok()) << loader_result.status().message();
  DataLoader loader = std::move(loader_result.value());

  const std::vector<int64_t> oracle_indices = ComputeShuffledIndices(kNumSamples, kSeed, 0);
  ASSERT_EQ(static_cast<int64_t>(oracle_indices.size()), kNumSamples);

  int64_t consumed = 0;
  for (int64_t batch_index = 0; batch_index < loader.batches_per_epoch(); ++batch_index) {
    Result<std::optional<std::vector<Tensor>>> result = loader.next(*allocator_);
    ASSERT_TRUE(result.is_ok()) << result.status().message();
    const std::optional<std::vector<Tensor>>& batch_opt = result.value();
    FRAME_ASSERT_HAS_VALUE_OR_RETURN(batch_opt,
                                     "batch_index " << batch_index << " expected a value");
    const std::vector<Tensor>& batch = *batch_opt;
    const Tensor& features_batch = batch[0];
    const Tensor& targets_batch = batch[1];
    const int64_t batch_len = features_batch.shape().dim(0);

    for (int64_t local_row = 0; local_row < batch_len; ++local_row) {
      const int64_t source_row = oracle_indices[static_cast<size_t>(consumed)];
      ExpectBatchRowMatchesSource(features_batch, targets_batch, local_row, source_row);
      ++consumed;
    }
  }
  EXPECT_EQ(consumed, kNumSamples);
}

// ---------------------------------------------------------------------------
// 3. TensorDataset 负例。
// ---------------------------------------------------------------------------

TEST_F(BatchValueCorrectnessTest, CreateRejectsMismatchedColumnSampleCounts) {
  Tensor features =
      MakeFloatTensor(std::vector<float>(static_cast<size_t>(10 * 3), 0.0F), Shape({10, 3}));
  // 违例列(下标 1):样本数 9 != 列 0 的样本数 10。
  Tensor targets = MakeFloatTensor(std::vector<float>(static_cast<size_t>(9), 0.0F), Shape({9, 1}));

  Result<TensorDataset> dataset_result = TensorDataset::create({features, targets});
  ASSERT_FALSE(dataset_result.is_ok());
  EXPECT_EQ(dataset_result.status().code(), ErrorCode::kInvalidArgument);
  const std::string_view message = dataset_result.status().message();
  EXPECT_NE(message.find("column 1"), std::string_view::npos) << "message: " << message;
  EXPECT_NE(message.find('9'), std::string_view::npos) << "message: " << message;
}

TEST_F(BatchValueCorrectnessTest, CreateRejectsNonCpuBackendColumn) {
  // Storage/device 一致性由调用方保证、Storage 本身不校验(同
  // tests/cpp/interop/test_onnx_weights.cpp::SaveRejectsNonCpuTensor 先例):
  // 用真实 cpu allocator 分配但显式声明非 cpu 的 device 元数据,构造出"设备
  // 元数据为非 cpu"的张量,驱动 TensorDataset::create 的③号校验,不需要真实
  // 注册 cuda 后端。
  const frame::Device fake_non_cpu_device{frame::kCudaBackendName, 0};
  Result<Tensor> non_cpu_result =
      Tensor::empty(Shape({10, 3}), DType::of<float>(), fake_non_cpu_device, *allocator_);
  ASSERT_TRUE(non_cpu_result.is_ok());
  Tensor targets =
      MakeFloatTensor(std::vector<float>(static_cast<size_t>(10), 0.0F), Shape({10, 1}));

  Result<TensorDataset> dataset_result = TensorDataset::create({non_cpu_result.value(), targets});
  ASSERT_FALSE(dataset_result.is_ok());
  EXPECT_EQ(dataset_result.status().code(), ErrorCode::kInvalidArgument);
  const std::string_view message = dataset_result.status().message();
  EXPECT_NE(message.find("column 0"), std::string_view::npos) << "message: " << message;
  EXPECT_NE(message.find("cuda"), std::string_view::npos) << "message: " << message;
}

TEST_F(BatchValueCorrectnessTest, CreateRejectsEmptyColumnsList) {
  Result<TensorDataset> dataset_result = TensorDataset::create({});
  ASSERT_FALSE(dataset_result.is_ok());
  EXPECT_EQ(dataset_result.status().code(), ErrorCode::kInvalidArgument);
  const std::string_view message = dataset_result.status().message();
  EXPECT_NE(message.find("empty"), std::string_view::npos) << "message: " << message;
}

}  // namespace
