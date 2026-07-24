// AFF 固定种子收敛测试(M21 批3 T7,docs/plan/2026-07-18-batch3-m21-conv.md
// 第1.4/3节):Conv2d(1->4,3x3,p1)->Relu->MaxPool2d(2x2)得 X 支;
// Conv2d(4->4,1x1)(X)得 Y 支;AFF(4 channels)(X,Y)->Flatten->Linear(64->1,
// bias)->MseLoss;x[8,1,8,8] 固定种子合成回归数据。AFF 需要 2 路输入
// (X/Y),Sequential 单流转发无法表达分支+汇合,故本文件手工组合各子 Module
// (照抄 test_aff_smoke.cpp 的 build() 手工调用手法,而非套 Sequential),
// params 切片按 [conv, y_conv, aff, linear] 声明序手动分段(ARCH-071 切片
// 不变式对手工组合同样成立,同 nn::Sequential/AFF 实现的既有做法)。训练循环/
// 参数轮换/断言结构 + golden 比对均复用 convergence_test_helpers.h
// (REUSE-002,同 test_cnn_convergence.cpp)。
//
// 前向输出空间形状推导:conv 输出 [8,4,8,8];pool 输出 [8,4,4,4](=X);y_conv
// (1x1,s1,p0)输出仍 [8,4,4,4](=Y);AFF(X,Y)输出 [8,4,4,4];flatten 输出
// [8,64](4*4*4),故 Linear in_dim=64。
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <span>
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
using frame::nn::AFF;
using frame::nn::Conv2d;
using frame::nn::Flatten;
using frame::nn::Linear;
using frame::nn::MaxPool2d;
using frame::nn::Module;
using frame::nn::MseLoss;
using frame::nn::ParamSpec;
using frame::nn::Relu;
using frame::nn::testing::AssertGraphMatchesGolden;
using frame::nn::testing::MakeCpuTensorType;
using frame::nn::testing::MakeTensorFromFloats;
using frame::nn::testing::MakeUniformParamTensors;
using frame::nn::testing::ParamTypesOf;
using frame::nn::testing::RunFullBatchSgdTraining;

constexpr int64_t kBatch = 8;
constexpr int64_t kInChannels = 1;
constexpr int64_t kHeight = 8;
constexpr int64_t kWidth = 8;
constexpr int64_t kFusionChannels = 4;
constexpr int64_t kFlattenDim = 64;  // 4(pooled C) * 4(pooled H) * 4(pooled W)
constexpr int64_t kOutDim = 1;

// 完整前向图(含 mse_loss、mark_output),图输入序 [x, conv.weight, conv.bias,
// y_conv.weight, y_conv.bias, aff.c1.weight, aff.c1.bias, aff.c2.weight,
// aff.c2.bias, linear.weight, linear.bias, target]。
struct AffGraphBundle {
  Graph graph;
  Value* x = nullptr;
  Value* target = nullptr;
  std::vector<ParamSpec> param_specs;
};

Result<AffGraphBundle> BuildAffForwardGraph() {
  Graph graph("aff_forward");

  const Result<Value*> x_result =
      graph.add_graph_input(MakeCpuTensorType({kBatch, kInChannels, kHeight, kWidth}));
  if (!x_result.is_ok()) return x_result.status();
  Value* x = x_result.value();

  const Module conv = Conv2d("conv", kInChannels, kFusionChannels, /*kernel_hw=*/{3, 3},
                             /*stride_hw=*/{1, 1}, /*padding_hw=*/{1, 1}, /*groups=*/1,
                             /*with_bias=*/true, DType::of<float>());
  const Module relu = Relu("relu");
  const Module pool = MaxPool2d("pool", /*kernel_hw=*/{2, 2}, /*stride_hw=*/{2, 2},
                                /*padding_hw=*/{0, 0});
  const Module y_conv = Conv2d("y_conv", kFusionChannels, kFusionChannels, /*kernel_hw=*/{1, 1},
                               /*stride_hw=*/{1, 1}, /*padding_hw=*/{0, 0}, /*groups=*/1,
                               /*with_bias=*/true, DType::of<float>());
  const Module aff = AFF("aff", kFusionChannels, DType::of<float>());
  const Module flatten = Flatten("flatten");
  const Module linear =
      Linear("linear", kBatch, kFlattenDim, kOutDim, /*with_bias=*/true, DType::of<float>());

  std::vector<ParamSpec> param_specs;
  const std::vector<ParamSpec> conv_params_spec = conv.parameters();
  const std::vector<ParamSpec> y_conv_params_spec = y_conv.parameters();
  const std::vector<ParamSpec> aff_params_spec = aff.parameters();
  const std::vector<ParamSpec> linear_params_spec = linear.parameters();
  param_specs.insert(param_specs.end(), conv_params_spec.begin(), conv_params_spec.end());
  param_specs.insert(param_specs.end(), y_conv_params_spec.begin(), y_conv_params_spec.end());
  param_specs.insert(param_specs.end(), aff_params_spec.begin(), aff_params_spec.end());
  param_specs.insert(param_specs.end(), linear_params_spec.begin(), linear_params_spec.end());

  const Result<std::vector<Value*>> param_inputs = add_parameter_inputs(graph, param_specs);
  if (!param_inputs.is_ok()) return param_inputs.status();
  const std::span<Value* const> all_params(param_inputs.value());

  size_t offset = 0;
  const std::span<Value* const> conv_params = all_params.subspan(offset, conv_params_spec.size());
  offset += conv_params_spec.size();
  const std::span<Value* const> y_conv_params =
      all_params.subspan(offset, y_conv_params_spec.size());
  offset += y_conv_params_spec.size();
  const std::span<Value* const> aff_params = all_params.subspan(offset, aff_params_spec.size());
  offset += aff_params_spec.size();
  const std::span<Value* const> linear_params =
      all_params.subspan(offset, linear_params_spec.size());

  const Result<std::vector<Value*>> conv_out =
      conv.build(graph, std::vector<Value*>{x}, conv_params);
  if (!conv_out.is_ok()) return conv_out.status();
  const Result<std::vector<Value*>> relu_out =
      relu.build(graph, conv_out.value(), std::vector<Value*>{});
  if (!relu_out.is_ok()) return relu_out.status();
  const Result<std::vector<Value*>> pool_out =
      pool.build(graph, relu_out.value(), std::vector<Value*>{});
  if (!pool_out.is_ok()) return pool_out.status();
  Value* x_branch = pool_out.value()[0];

  const Result<std::vector<Value*>> y_out =
      y_conv.build(graph, std::vector<Value*>{x_branch}, y_conv_params);
  if (!y_out.is_ok()) return y_out.status();
  Value* y_branch = y_out.value()[0];

  const Result<std::vector<Value*>> aff_out =
      aff.build(graph, std::vector<Value*>{x_branch, y_branch}, aff_params);
  if (!aff_out.is_ok()) return aff_out.status();

  const Result<std::vector<Value*>> flatten_out =
      flatten.build(graph, aff_out.value(), std::vector<Value*>{});
  if (!flatten_out.is_ok()) return flatten_out.status();

  const Result<std::vector<Value*>> linear_out =
      linear.build(graph, flatten_out.value(), linear_params);
  if (!linear_out.is_ok()) return linear_out.status();
  Value* prediction = linear_out.value()[0];

  const Result<Value*> target_result = graph.add_graph_input(MakeCpuTensorType({kBatch, kOutDim}));
  if (!target_result.is_ok()) return target_result.status();
  Value* target = target_result.value();

  const Result<std::vector<Value*>> loss_outputs =
      MseLoss("loss").build(graph, std::vector<Value*>{prediction, target}, std::vector<Value*>{});
  if (!loss_outputs.is_ok()) return loss_outputs.status();
  const Status mark_status = graph.mark_output(loss_outputs.value()[0]);
  if (!mark_status.is_ok()) return mark_status;

  AffGraphBundle bundle;
  bundle.graph = std::move(graph);
  bundle.x = x;
  bundle.target = target;
  bundle.param_specs = param_specs;
  return bundle;
}

class AffConvergenceTest : public ::testing::Test {
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
// (mt19937(20260713))下实测 initial_loss≈0.1067、final_loss≈1.08e-05(300 步,
// lr=0.05——参数量远超样本数(8),故能几乎精确拟合教师线性目标);两断言均
// 留裕度——① final < initial*0.1;② final < 1e-3(约为实测 final_loss 的
// 93 倍)。
TEST_F(AffConvergenceTest, TrainingLossDecreasesSignificantly) {
  Result<AffGraphBundle> bundle_result = BuildAffForwardGraph();
  ASSERT_TRUE(bundle_result.is_ok()) << bundle_result.status().message();
  AffGraphBundle bundle = std::move(bundle_result.value());

  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 rng(20260713U);
  std::uniform_real_distribution<float> data_dist(-1.0F, 1.0F);
  std::uniform_real_distribution<float> teacher_dist(-0.15F, 0.15F);

  std::vector<float> x_values(static_cast<size_t>(kBatch * kInChannels * kHeight * kWidth));
  for (float& v : x_values) v = data_dist(rng);

  constexpr int64_t kRawPixels = kInChannels * kHeight * kWidth;  // 64,恰与 kFlattenDim 同值
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

  const Tensor x_tensor = MakeTensorFromFloats(
      x_values, Shape({kBatch, kInChannels, kHeight, kWidth}), device_, *allocator_);
  const Tensor target_tensor =
      MakeTensorFromFloats(target_values, Shape({kBatch, kOutDim}), device_, *allocator_);

  std::vector<Tensor> params =
      MakeUniformParamTensors(bundle.param_specs, rng, -0.1F, 0.1F, device_, *allocator_);
  const std::vector<frame::ir::TensorType> param_types = ParamTypesOf(bundle.param_specs);

  constexpr double kLearningRate = 0.05;
  constexpr int kNumSteps = 300;
  const Result<std::vector<double>> loss_history = RunFullBatchSgdTraining(
      bundle.graph, x_tensor, params, param_types, target_tensor, kLearningRate, kNumSteps);
  ASSERT_TRUE(loss_history.is_ok()) << loss_history.status().message();
  ASSERT_EQ(loss_history.value().size(), static_cast<size_t>(kNumSteps));

  const double initial_loss = loss_history.value().front();
  const double final_loss = loss_history.value().back();

  EXPECT_LT(final_loss, initial_loss * 0.1)
      << "initial_loss=" << initial_loss << " final_loss=" << final_loss;
  EXPECT_LT(final_loss, 1e-3) << "final_loss=" << final_loss;
}

TEST_F(AffConvergenceTest, ForwardGraphMatchesGolden) {
  Result<AffGraphBundle> bundle_result = BuildAffForwardGraph();
  ASSERT_TRUE(bundle_result.is_ok()) << bundle_result.status().message();
  AffGraphBundle bundle = std::move(bundle_result.value());

  EXPECT_TRUE(
      AssertGraphMatchesGolden(bundle.graph, "tests/cpp/nn/testdata/aff_forward_expected.txt"));
}

}  // namespace
