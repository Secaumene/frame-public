// M25 Mamba/FourierMamba 固定种子小样本收敛冒烟。训练编排复用
// convergence_test_helpers.h，真实执行前向、autograd、SGD 更新闭环。
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
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
using frame::nn::FourierMamba;
using frame::nn::Mamba;
using frame::nn::Module;
using frame::nn::MseLoss;
using frame::nn::ParamSpec;
using frame::nn::testing::MakeCpuTensorType;
using frame::nn::testing::MakeTensorFromFloats;
using frame::nn::testing::MakeUniformParamTensors;
using frame::nn::testing::ParamTypesOf;
using frame::nn::testing::RunFullBatchSgdTraining;

constexpr int64_t kBatch = 2;
constexpr int64_t kChannels = 1;
constexpr int64_t kSteps = 3;
constexpr int64_t kKernel = 2;

struct SsmTrainingBundle {
  Graph graph;
  std::vector<ParamSpec> param_specs;
};

Result<SsmTrainingBundle> BuildTrainingForward(bool with_fourier) {
  Graph graph(with_fourier ? "fourier_mamba_convergence" : "mamba_convergence");
  const Result<Value*> x = graph.add_graph_input(MakeCpuTensorType({kBatch, kChannels, kSteps}));
  if (!x.is_ok()) return x.status();
  const Module model =
      with_fourier ? FourierMamba("fm", kBatch, kChannels, kSteps, kKernel, DType::of<float>())
                   : Mamba("m", kBatch, kChannels, kSteps, kKernel, DType::of<float>());
  const std::vector<ParamSpec> param_specs = model.parameters();
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, param_specs);
  if (!params.is_ok()) return params.status();
  const Result<std::vector<Value*>> outputs =
      model.build(graph, std::vector<Value*>{x.value()}, params.value());
  if (!outputs.is_ok()) return outputs.status();

  const Result<Value*> target =
      graph.add_graph_input(MakeCpuTensorType({kBatch, kChannels, kSteps}));
  if (!target.is_ok()) return target.status();
  const Result<std::vector<Value*>> loss = MseLoss("loss").build(
      graph, std::vector<Value*>{outputs.value()[0], target.value()}, std::vector<Value*>{});
  if (!loss.is_ok()) return loss.status();
  const Status marked = graph.mark_output(loss.value()[0]);
  if (!marked.is_ok()) return marked;
  return SsmTrainingBundle{std::move(graph), param_specs};
}

class SsmConvergenceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const Result<Backend*> backend = BackendRegistry::instance().get(frame::kCpuBackendName);
    ASSERT_TRUE(backend.is_ok()) << backend.status().message();
    allocator_ = backend.value()->allocator(frame::cpu_device());
    ASSERT_NE(allocator_, nullptr);
  }

  void ExpectConverges(bool with_fourier, uint32_t seed) {
    Result<SsmTrainingBundle> bundle_result = BuildTrainingForward(with_fourier);
    ASSERT_TRUE(bundle_result.is_ok()) << bundle_result.status().message();
    SsmTrainingBundle bundle = std::move(bundle_result.value());

    // 固定 seed 是可复现性合同;样本与教师目标均从同一 rng 顺序抽取。
    // NOLINTNEXTLINE(bugprone-random-generator-seed)
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> data_dist(-0.5F, 0.5F);
    std::vector<float> x_values(static_cast<size_t>(kBatch * kChannels * kSteps));
    for (float& value : x_values) value = data_dist(rng);
    std::vector<float> target_values(x_values.size());
    for (size_t i = 0; i < x_values.size(); ++i) {
      target_values[i] = 0.35F * x_values[i] + 0.05F;
    }

    const Shape shape({kBatch, kChannels, kSteps});
    const Tensor x = MakeTensorFromFloats(x_values, shape, frame::cpu_device(), *allocator_);
    const Tensor target =
        MakeTensorFromFloats(target_values, shape, frame::cpu_device(), *allocator_);
    std::vector<Tensor> params = MakeUniformParamTensors(bundle.param_specs, rng, -0.1F, 0.1F,
                                                         frame::cpu_device(), *allocator_);

    constexpr double kLearningRate = 0.1;
    constexpr int kTrainingSteps = 200;
    const Result<std::vector<double>> losses =
        RunFullBatchSgdTraining(bundle.graph, x, params, ParamTypesOf(bundle.param_specs), target,
                                kLearningRate, kTrainingSteps);
    ASSERT_TRUE(losses.is_ok()) << losses.status().message();
    ASSERT_EQ(losses.value().size(), static_cast<size_t>(kTrainingSteps));
    const double initial = losses.value().front();
    const double final = losses.value().back();

    // 固定 seed 实测:Mamba 0.0336126->4.6412e-08,FourierMamba
    // 0.0399469->2.66861e-09。共同阈值取下降 3 个数量级且 final<1e-5,
    // 分别较较差实测保留约 700 倍与 200 倍余量，属于收敛冒烟而非精确 golden。
    EXPECT_LT(final, initial * 1e-3) << "initial_loss=" << initial << " final_loss=" << final;
    EXPECT_LT(final, 1e-5) << "final_loss=" << final;
  }

  frame::hal::Allocator* allocator_ = nullptr;
};

TEST_F(SsmConvergenceTest, MambaFixedSeedSmallSampleConverges) {
  ExpectConverges(false, 20260729U);
}

TEST_F(SsmConvergenceTest, FourierMambaFixedSeedSmallSampleConverges) {
  ExpectConverges(true, 20260730U);
}

}  // namespace
