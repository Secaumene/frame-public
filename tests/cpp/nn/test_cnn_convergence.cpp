// CNN 固定种子收敛测试(M21 批3 T7,docs/plan/2026-07-18-batch3-m21-conv.md
// 第1.4/3节):Conv2d("conv",1->4,3x3,s1,p1,bias)->Relu->
// MaxPool2d(2x2,s2,p0)->Flatten->Linear(64->1,bias)->MseLoss;x[8,1,8,8]
// 固定种子合成回归数据(uniform_seeded,同 test_training_smoke.cpp 头注释
// 先例)。训练循环/参数轮换/断言结构照抄 test_training_smoke.cpp;三网络共用
// 训练线组装收敛进 convergence_test_helpers.h(REUSE-002)。
//
// 前向输出空间形状推导(compute_conv2d_geometry floor 口径,
// src/ops/schemas/conv.cpp):conv 输出 [8,4,8,8]((8+2*1-3)/1+1=8);pool 输出
// [8,4,4,4]((8-2)/2+1=4);flatten 输出 [8,64](4*4*4=64),故 Linear
// in_dim=64。
//
// golden IR 快照(构图确定性,NnCnnConvergence.ForwardGraphMatchesGolden):
// 完整前向图(含 mse_loss 节点、mark_output)dump_text 与
// testdata/cnn_forward_expected.txt 逐字节比对——生成方式:先跑一次冒烟测试
// 打印 dump_text 落盘,再以逐字节断言固化(不入库任何一次性生成代码)。
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
using frame::nn::Conv2d;
using frame::nn::Flatten;
using frame::nn::Linear;
using frame::nn::MaxPool2d;
using frame::nn::Module;
using frame::nn::MseLoss;
using frame::nn::ParamSpec;
using frame::nn::Relu;
using frame::nn::Sequential;
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
constexpr int64_t kConvOutChannels = 4;
constexpr int64_t kFlattenDim = 64;  // 4(pooled C) * 4(pooled H) * 4(pooled W)
constexpr int64_t kOutDim = 1;

// 完整前向图(含 mse_loss、mark_output),图输入序 [x, conv.weight, conv.bias,
// linear.weight, linear.bias, target]——add_parameter_inputs 紧随 x 之后、
// target 最后追加,同 test_training_smoke.cpp 装配序。
struct CnnGraphBundle {
  Graph graph;
  Value* x = nullptr;
  Value* target = nullptr;
  std::vector<ParamSpec> param_specs;
};

Result<CnnGraphBundle> BuildCnnForwardGraph() {
  Graph graph("cnn_forward");

  const Result<Value*> x_result =
      graph.add_graph_input(MakeCpuTensorType({kBatch, kInChannels, kHeight, kWidth}));
  if (!x_result.is_ok()) return x_result.status();
  Value* x = x_result.value();

  const Module model = Sequential(
      "cnn",
      {Conv2d("conv", kInChannels, kConvOutChannels, /*kernel_hw=*/{3, 3},
              /*stride_hw=*/{1, 1}, /*padding_hw=*/{1, 1}, /*groups=*/1, /*with_bias=*/true,
              DType::of<float>()),
       Relu("relu"),
       MaxPool2d("pool", /*kernel_hw=*/{2, 2}, /*stride_hw=*/{2, 2},
                 /*padding_hw=*/{0, 0}),
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

  CnnGraphBundle bundle;
  bundle.graph = std::move(graph);
  bundle.x = x;
  bundle.target = target;
  bundle.param_specs = param_specs;
  return bundle;
}

class CnnConvergenceTest : public ::testing::Test {
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
// (mt19937(20260713))下实测 initial_loss≈0.0836、final_loss≈4.33e-08(300 步,
// lr=0.05——网络参数量(112)远超样本数(8),故能几乎精确拟合教师线性目标);
// 两断言均留裕度——① final < initial*0.1;② final < 1e-4(约为实测 final_loss
// 的 2300 倍)。
TEST_F(CnnConvergenceTest, TrainingLossDecreasesSignificantly) {
  Result<CnnGraphBundle> bundle_result = BuildCnnForwardGraph();
  ASSERT_TRUE(bundle_result.is_ok()) << bundle_result.status().message();
  CnnGraphBundle bundle = std::move(bundle_result.value());

  // 常量种子是刻意选择(测试可复现性,非 bug),同 test_training_smoke.cpp
  // 抑制方式:
  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 rng(20260713U);
  std::uniform_real_distribution<float> data_dist(-1.0F, 1.0F);
  std::uniform_real_distribution<float> teacher_dist(-0.15F, 0.15F);

  std::vector<float> x_values(static_cast<size_t>(kBatch * kInChannels * kHeight * kWidth));
  for (float& v : x_values) v = data_dist(rng);

  // 教师权重:对每个样本的 64 个原始像素做线性组合(与网络实际的
  // conv->relu->pool->flatten->linear 特征空间不同基,不要求精确可拟合——同
  // test_training_smoke.cpp 文件头注释的既有取舍:目标只需可被充分下降,不要求
  // 网络零损失拟合)。
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
  EXPECT_LT(final_loss, 1e-4) << "final_loss=" << final_loss;
}

TEST_F(CnnConvergenceTest, ForwardGraphMatchesGolden) {
  Result<CnnGraphBundle> bundle_result = BuildCnnForwardGraph();
  ASSERT_TRUE(bundle_result.is_ok()) << bundle_result.status().message();
  CnnGraphBundle bundle = std::move(bundle_result.value());

  EXPECT_TRUE(
      AssertGraphMatchesGolden(bundle.graph, "tests/cpp/nn/testdata/cnn_forward_expected.txt"));
}

}  // namespace
