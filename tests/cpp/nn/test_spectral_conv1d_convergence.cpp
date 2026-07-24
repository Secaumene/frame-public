// SpectralConv1d 单块小回归收敛冒烟(M23 批5 T6,docs/plan/
// 2026-07-21-batch5-m23-fft.md §1.5/§3 验收硬门 5、设计门建议 4):
// SpectralConv1d("sc",...)->Flatten->Linear(24->1,bias)->MseLoss;
// x[4,2,8] 固定种子(20260713)合成回归数据。证明 rfft/irfft 互引用梯度微图
// (§1.3 决议点C)全链在真实训练循环(build_backward_graph+
// build_sgd_update_graph+runtime::compile("cpu"))中真跑通、loss 显著下降。
// 训练循环/参数轮换/断言结构 + golden 比对手法均复用
// convergence_test_helpers.h(REUSE-002,同 test_wavelet_convergence.cpp/
// test_cnn_convergence.cpp 先例)。
//
// 前向输出空间形状推导:SpectralConv1d(batch=4,in=2,out=3,n=8,modes=3,
// k=n/2+1=5,modes<k 触发零补分支)输出 [4,3,8];flatten 输出 [4,24]
// (3*8),故 Linear in_dim=24。
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

using frame::cpu_device;
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
using frame::nn::Flatten;
using frame::nn::Linear;
using frame::nn::Module;
using frame::nn::MseLoss;
using frame::nn::ParamSpec;
using frame::nn::Sequential;
using frame::nn::SpectralConv1d;
using frame::nn::testing::AssertGraphMatchesGolden;
using frame::nn::testing::MakeCpuTensorType;
using frame::nn::testing::MakeTensorFromFloats;
using frame::nn::testing::MakeUniformParamTensors;
using frame::nn::testing::ParamTypesOf;
using frame::nn::testing::RunFullBatchSgdTraining;

constexpr int64_t kBatch = 4;
constexpr int64_t kInChannels = 2;
constexpr int64_t kOutChannels = 3;
constexpr int64_t kN = 8;            // k = n/2+1 = 5
constexpr int64_t kModes = 3;        // < k,触发零补分支(同购构建纯度测试覆盖)
constexpr int64_t kFlattenDim = 24;  // out_channels(3) * n(8)
constexpr int64_t kOutDim = 1;

// 完整前向图(含 mse_loss、mark_output),图输入序 [x, sc.W_re, sc.W_im,
// linear.weight, linear.bias, target]。
struct SpectralConvGraphBundle {
  Graph graph;
  Value* x = nullptr;
  Value* target = nullptr;
  std::vector<ParamSpec> param_specs;
};

Result<SpectralConvGraphBundle> BuildSpectralConvForwardGraph() {
  Graph graph("spectral_conv1d_forward");

  const Result<Value*> x_result =
      graph.add_graph_input(MakeCpuTensorType({kBatch, kInChannels, kN}));
  if (!x_result.is_ok()) return x_result.status();
  Value* x = x_result.value();

  const Module model = Sequential(
      "spectral_smoke",
      {SpectralConv1d("sc", kBatch, kInChannels, kOutChannels, kN, kModes, DType::of<float>()),
       Flatten("flatten"),
       Linear("linear", kBatch, kFlattenDim, kOutDim, /*with_bias=*/true, DType::of<float>())});
  const std::vector<ParamSpec> param_specs = model.parameters();

  const Result<std::vector<Value*>> param_inputs = add_parameter_inputs(graph, param_specs);
  if (!param_inputs.is_ok()) return param_inputs.status();

  const Result<std::vector<Value*>> forward_outputs =
      model.build(graph, std::vector<Value*>{x}, param_inputs.value());
  if (!forward_outputs.is_ok()) return forward_outputs.status();

  const Result<Value*> target_result = graph.add_graph_input(MakeCpuTensorType({kBatch, kOutDim}));
  if (!target_result.is_ok()) return target_result.status();
  Value* target = target_result.value();

  const Result<std::vector<Value*>> loss_outputs = MseLoss("loss").build(
      graph, std::vector<Value*>{forward_outputs.value()[0], target}, std::vector<Value*>{});
  if (!loss_outputs.is_ok()) return loss_outputs.status();
  const Status mark_status = graph.mark_output(loss_outputs.value()[0]);
  if (!mark_status.is_ok()) return mark_status;

  SpectralConvGraphBundle bundle;
  bundle.graph = std::move(graph);
  bundle.x = x;
  bundle.target = target;
  bundle.param_specs = param_specs;
  return bundle;
}

class SpectralConv1dConvergenceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const Result<Backend*> backend_result = BackendRegistry::instance().get(frame::kCpuBackendName);
    ASSERT_TRUE(backend_result.is_ok());
    backend_ = backend_result.value();
    device_ = cpu_device();
    allocator_ = backend_->allocator(device_);
    ASSERT_NE(allocator_, nullptr);
  }

  Backend* backend_ = nullptr;
  frame::Device device_{};
  frame::hal::Allocator* allocator_ = nullptr;
};

// 收敛断言(BUILD-011 口径:阈值取自实测,留安全边际):固定种子
// (mt19937(20260713))下实测 initial_loss≈0.1111、final_loss≈1.334e-4(300 步,
// lr=0.02);断言留裕度——final < initial*0.1(实测比值≈0.0012,远优于该
// 阈值)。本用例证明 rfft/irfft 互引用梯度微图(§1.3 决议点C:rfft 梯度=
// n·irfft(...),irfft 梯度=(1/n)·rfft(...))在真实 SGD 训练循环
// (build_backward_graph+build_sgd_update_graph+runtime::compile("cpu"))中
// 数值稳定收敛,而非仅解析梯度数值对照(ops 层已由
// tests/cpp/ops/test_op_fft.cpp 覆盖)。
TEST_F(SpectralConv1dConvergenceTest, TrainingLossDecreasesSignificantly) {
  Result<SpectralConvGraphBundle> bundle_result = BuildSpectralConvForwardGraph();
  ASSERT_TRUE(bundle_result.is_ok()) << bundle_result.status().message();
  SpectralConvGraphBundle bundle = std::move(bundle_result.value());

  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 rng(20260713U);
  std::uniform_real_distribution<float> data_dist(-1.0F, 1.0F);
  std::uniform_real_distribution<float> teacher_dist(-0.15F, 0.15F);

  std::vector<float> x_values(static_cast<size_t>(kBatch * kInChannels * kN));
  for (float& v : x_values) v = data_dist(rng);

  constexpr int64_t kRawPixels = kInChannels * kN;  // 16,教师函数在原始输入上定义
  std::vector<float> teacher_weights(static_cast<size_t>(kRawPixels));
  for (float& v : teacher_weights) v = teacher_dist(rng);

  std::vector<float> target_values(static_cast<size_t>(kBatch * kOutDim), 0.0F);
  for (int64_t i = 0; i < kBatch; ++i) {
    float acc = 0.0F;
    for (int64_t k = 0; k < kRawPixels; ++k) {
      acc += x_values[static_cast<size_t>(i * kRawPixels + k)] *
             teacher_weights[static_cast<size_t>(k)];
    }
    target_values[static_cast<size_t>(i)] = acc;
  }

  const Tensor x_tensor =
      MakeTensorFromFloats(x_values, Shape({kBatch, kInChannels, kN}), device_, *allocator_);
  const Tensor target_tensor =
      MakeTensorFromFloats(target_values, Shape({kBatch, kOutDim}), device_, *allocator_);

  std::vector<Tensor> params =
      MakeUniformParamTensors(bundle.param_specs, rng, -0.1F, 0.1F, device_, *allocator_);
  const std::vector<frame::ir::TensorType> param_types = ParamTypesOf(bundle.param_specs);

  constexpr double kLearningRate = 0.02;
  constexpr int kNumSteps = 300;
  const Result<std::vector<double>> loss_history = RunFullBatchSgdTraining(
      bundle.graph, x_tensor, params, param_types, target_tensor, kLearningRate, kNumSteps);
  ASSERT_TRUE(loss_history.is_ok()) << loss_history.status().message();
  ASSERT_EQ(loss_history.value().size(), static_cast<size_t>(kNumSteps));

  const double initial_loss = loss_history.value().front();
  const double final_loss = loss_history.value().back();

  EXPECT_LT(final_loss, initial_loss * 0.1)
      << "initial_loss=" << initial_loss << " final_loss=" << final_loss;
}

TEST_F(SpectralConv1dConvergenceTest, ForwardGraphMatchesGolden) {
  Result<SpectralConvGraphBundle> bundle_result = BuildSpectralConvForwardGraph();
  ASSERT_TRUE(bundle_result.is_ok()) << bundle_result.status().message();
  SpectralConvGraphBundle bundle = std::move(bundle_result.value());

  EXPECT_TRUE(AssertGraphMatchesGolden(
      bundle.graph, "tests/cpp/nn/testdata/spectral_conv1d_forward_expected.txt"));
}

}  // namespace
