// M27 SnnClassifier(with_bias=false) 固定种子小样本分类收敛与前向 IR golden。
// 训练真实经过 input Linear -> LIF spike -> output Linear -> 时间 sum，调用方
// 组合 MseLoss 与 SGD；无 bias，不能靠逐样本位置参数记忆标签。
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/nn/layers.h>
#include <frame/nn/module.h>

#include "convergence_test_helpers.h"

namespace {

using frame::DType;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::Tensor;
using frame::hal::Backend;
using frame::hal::BackendRegistry;
using frame::ir::Graph;
using frame::ir::Value;
using frame::nn::add_parameter_inputs;
using frame::nn::Module;
using frame::nn::MseLoss;
using frame::nn::ParamSpec;
using frame::nn::SnnClassifier;
using frame::nn::testing::AssertGraphMatchesGolden;
using frame::nn::testing::MakeCpuTensorType;
using frame::nn::testing::MakeTensorFromFloats;
using frame::nn::testing::ParamTypesOf;
using frame::nn::testing::RunFullBatchSgdTraining;

constexpr int64_t kBatch = 4;
constexpr int64_t kSteps = 3;
constexpr int64_t kInputDim = 2;
constexpr int64_t kHiddenDim = 4;
constexpr int64_t kClasses = 2;

struct SnnGraphBundle {
  Graph graph;
  std::vector<ParamSpec> param_specs;
};

Result<SnnGraphBundle> BuildSnnForwardGraph() {
  Graph graph("snn_classifier_forward");
  const Result<Value*> x = graph.add_graph_input(MakeCpuTensorType({kBatch, kSteps, kInputDim}));
  if (!x.is_ok()) return x.status();
  const Module model = SnnClassifier("snn", kBatch, kSteps, kInputDim, kHiddenDim, kClasses,
                                     /*decay=*/0.5, /*threshold=*/0.5, /*alpha=*/3.0,
                                     /*with_bias=*/false, DType::of<float>());
  const std::vector<ParamSpec> param_specs = model.parameters();
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, param_specs);
  if (!params.is_ok()) return params.status();
  const Result<std::vector<Value*>> logits =
      model.build(graph, std::vector<Value*>{x.value()}, params.value());
  if (!logits.is_ok()) return logits.status();
  const Result<Value*> target = graph.add_graph_input(MakeCpuTensorType({kBatch, kClasses}));
  if (!target.is_ok()) return target.status();
  const Result<std::vector<Value*>> loss = MseLoss("loss").build(
      graph, std::vector<Value*>{logits.value()[0], target.value()}, std::vector<Value*>{});
  if (!loss.is_ok()) return loss.status();
  const Status marked = graph.mark_output(loss.value()[0]);
  if (!marked.is_ok()) return marked;
  return SnnGraphBundle{std::move(graph), param_specs};
}

class SnnConvergenceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const Result<Backend*> backend = BackendRegistry::instance().get(frame::kCpuBackendName);
    ASSERT_TRUE(backend.is_ok()) << backend.status().message();
    allocator_ = backend.value()->allocator(frame::cpu_device());
    ASSERT_NE(allocator_, nullptr);
  }

  frame::hal::Allocator* allocator_ = nullptr;
};

TEST_F(SnnConvergenceTest, BiasFreeFixedSeedPulseClassificationConverges) {
  Result<SnnGraphBundle> bundle_result = BuildSnnForwardGraph();
  ASSERT_TRUE(bundle_result.is_ok()) << bundle_result.status().message();
  SnnGraphBundle bundle = std::move(bundle_result.value());
  ASSERT_EQ(bundle.param_specs.size(), 2U);

  // 两类脉冲分别编码在 feature0/feature1，样本内幅度有小扰动，不能按固定
  // 时间位置记忆；两行 input weight 产生互补 spike 模式。
  const std::vector<float> x_values{
      1.0F, 0.0F, 0.9F, 0.0F, 1.1F, 0.0F, 0.8F, 0.0F, 1.0F, 0.0F, 0.9F, 0.0F,
      0.0F, 1.0F, 0.0F, 0.9F, 0.0F, 1.1F, 0.0F, 0.8F, 0.0F, 1.0F, 0.0F, 0.9F,
  };
  const std::vector<float> target_values{1.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 1.0F};
  const Tensor x = MakeTensorFromFloats(x_values, Shape({kBatch, kSteps, kInputDim}),
                                        frame::cpu_device(), *allocator_);
  const Tensor target = MakeTensorFromFloats(target_values, Shape({kBatch, kClasses}),
                                             frame::cpu_device(), *allocator_);

  // 固定 seed 只驱动 output weight 初值；input weight 取可解释的互补脉冲
  // 编码，确保前向真实产生 spike，而非在无脉冲区依赖代理梯度偶然穿越阈值。
  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 rng(20260731U);
  std::uniform_real_distribution<float> output_dist(-0.15F, 0.15F);
  std::vector<float> output_weight(static_cast<size_t>(kHiddenDim * kClasses));
  for (float& value : output_weight) value = output_dist(rng);
  std::vector<Tensor> params{
      MakeTensorFromFloats({0.8F, 0.7F, -0.6F, -0.5F, -0.5F, -0.6F, 0.7F, 0.8F},
                           Shape({kInputDim, kHiddenDim}), frame::cpu_device(), *allocator_),
      MakeTensorFromFloats(output_weight, Shape({kHiddenDim, kClasses}), frame::cpu_device(),
                           *allocator_),
  };

  constexpr double kLearningRate = 0.03;
  constexpr int kTrainingSteps = 250;
  const Result<std::vector<double>> losses =
      RunFullBatchSgdTraining(bundle.graph, x, params, ParamTypesOf(bundle.param_specs), target,
                              kLearningRate, kTrainingSteps);
  ASSERT_TRUE(losses.is_ok()) << losses.status().message();
  ASSERT_EQ(losses.value().size(), static_cast<size_t>(kTrainingSteps));
  const double initial = losses.value().front();
  const double final = losses.value().back();
  EXPECT_LT(final, initial * 1e-3) << "initial_loss=" << initial << " final_loss=" << final;
  EXPECT_LT(final, 1e-5) << "final_loss=" << final;
}

TEST(SnnConvergenceGolden, ForwardGraphMatchesGolden) {
  Result<SnnGraphBundle> bundle_result = BuildSnnForwardGraph();
  ASSERT_TRUE(bundle_result.is_ok()) << bundle_result.status().message();
  SnnGraphBundle bundle = std::move(bundle_result.value());
  EXPECT_TRUE(AssertGraphMatchesGolden(
      bundle.graph, "tests/cpp/nn/testdata/snn_classifier_forward_expected.txt"));
}

}  // namespace
