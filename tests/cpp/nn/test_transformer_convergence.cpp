// TransformerEncoderBlock 固定种子收敛测试(M22 批4 T6,docs/plan/
// 2026-07-19-batch4-m22-seq.md §1.7/§2 T6/§3 验收硬门第 5 条):
// x[rows=4,embed_dim=4]->TransformerEncoderBlock(heads=2,ffn=6)->
// y[4,4]->Linear(4->1,bias)->MseLoss;x 固定种子合成回归数据(uniform_seeded,
// 同 test_training_smoke.cpp 头注释先例)。TransformerEncoderBlock 的输入口径
// 本身即 [batch·seq_len, embed_dim] 2-D(layers.h 头注释,Linear 2-D 口径),
// 恰同 Linear 的输入形状,故 EncoderBlock 输出无需再经 Flatten 节点即可直接
// 喂给末端 Linear(§1.7 网络交付条目"EncoderBlock→Flatten 口径 Linear"中的
// "Flatten 口径"在本例已由 EncoderBlock 输出天然满足,literal Flatten 节点
// 对已是 rank-2 的输入是恒等操作,省略不改变语义)。EncoderBlock/Linear 均恰
// 1 输入恰 1 输出,可直接用 Sequential 表达(同 test_cnn_convergence.cpp/
// test_lstm_convergence.cpp 单流结构)。训练循环/参数轮换/断言结构 + golden
// 比对均复用 convergence_test_helpers.h(REUSE-002)。
//
// 教师目标构造(同 CNN/Wavelet/LSTM precedent 既有取舍):对每一行(样本)的
// embed_dim=4 个原始输入值做线性组合,与网络实际的
// TransformerEncoderBlock(注意力+FFN+LayerNorm)->Linear 特征空间不同基,不
// 要求精确可拟合——目标只需可被充分下降,不要求网络零损失拟合。
//
// golden IR 快照(构图确定性,
// TransformerConvergenceTest.ForwardGraphMatchesGolden):完整前向图(含
// mse_loss 节点、mark_output)dump_text 与
// testdata/transformer_forward_expected.txt 逐字节比对——生成方式同
// test_cnn_convergence.cpp 头注释:先跑一次冒烟测试打印 dump_text 落盘,再以
// 逐字节断言固化(不入库任何一次性生成代码)。
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
using frame::nn::Linear;
using frame::nn::Module;
using frame::nn::MseLoss;
using frame::nn::ParamSpec;
using frame::nn::Sequential;
using frame::nn::TransformerEncoderBlock;
using frame::nn::testing::AssertGraphMatchesGolden;
using frame::nn::testing::MakeCpuTensorType;
using frame::nn::testing::MakeTensorFromFloats;
using frame::nn::testing::MakeUniformParamTensors;
using frame::nn::testing::ParamTypesOf;
using frame::nn::testing::RunFullBatchSgdTraining;

constexpr int64_t kBatch = 2;
constexpr int64_t kSeqLen = 2;
constexpr int64_t kEmbedDim = 4;
constexpr int64_t kNumHeads = 2;
constexpr int64_t kFfnDim = 6;
constexpr int64_t kRows = kBatch * kSeqLen;  // 4,EncoderBlock/Linear 的 2-D 行数
constexpr int64_t kOutDim = 1;

// 完整前向图(含 mse_loss、mark_output),图输入序 [x, block.mha.q.weight,
// block.mha.q.bias, ..., block.ln2.gamma, block.ln2.beta, linear.weight,
// linear.bias, target]——add_parameter_inputs 紧随 x 之后、target 最后追加,
// 同 test_cnn_convergence.cpp 装配序。
struct TransformerGraphBundle {
  Graph graph;
  Value* x = nullptr;
  Value* target = nullptr;
  std::vector<ParamSpec> param_specs;
};

Result<TransformerGraphBundle> BuildTransformerForwardGraph() {
  Graph graph("transformer_forward");

  const Result<Value*> x_result = graph.add_graph_input(MakeCpuTensorType({kRows, kEmbedDim}));
  if (!x_result.is_ok()) return x_result.status();
  Value* x = x_result.value();

  const Module model = Sequential(
      "transformer_net",
      {TransformerEncoderBlock("block", kBatch, kSeqLen, kEmbedDim, kNumHeads, kFfnDim,
                               /*with_bias=*/true, DType::of<float>()),
       Linear("linear", kRows, kEmbedDim, kOutDim, /*with_bias=*/true, DType::of<float>())});
  const std::vector<ParamSpec> param_specs = model.parameters();

  const Result<std::vector<Value*>> param_inputs = add_parameter_inputs(graph, param_specs);
  if (!param_inputs.is_ok()) return param_inputs.status();

  const Result<std::vector<Value*>> forward_outputs =
      model.build(graph, std::vector<Value*>{x}, param_inputs.value());
  if (!forward_outputs.is_ok()) return forward_outputs.status();

  const Result<Value*> target_result = graph.add_graph_input(MakeCpuTensorType({kRows, kOutDim}));
  if (!target_result.is_ok()) return target_result.status();
  Value* target = target_result.value();

  const Result<std::vector<Value*>> loss_outputs = MseLoss("loss").build(
      graph, std::vector<Value*>{forward_outputs.value()[0], target}, std::vector<Value*>{});
  if (!loss_outputs.is_ok()) return loss_outputs.status();
  const Status mark_status = graph.mark_output(loss_outputs.value()[0]);
  if (!mark_status.is_ok()) return mark_status;

  TransformerGraphBundle bundle;
  bundle.graph = std::move(graph);
  bundle.x = x;
  bundle.target = target;
  bundle.param_specs = param_specs;
  return bundle;
}

class TransformerConvergenceTest : public ::testing::Test {
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
// (mt19937(20260713))下实测 initial_loss≈0.010890、final_loss≈1.13e-09(300
// 步,lr=0.05——网络参数量(约 245)远超样本数(4),故能几乎精确拟合教师线性
// 目标,同 CNN/Wavelet precedent 取舍)。loss 下降约 7 个数量级,满足 spec §3
// 验收硬门第 5 条"≥3 个数量级"要求。两断言均留裕度——① final <
// initial*1e-4(约 4 个数量级,低于实测的约 7 个数量级留安全边际);
// ② final < 1e-6(约为实测 final_loss 的 880 倍)。
TEST_F(TransformerConvergenceTest, TrainingLossDecreasesSignificantly) {
  Result<TransformerGraphBundle> bundle_result = BuildTransformerForwardGraph();
  ASSERT_TRUE(bundle_result.is_ok()) << bundle_result.status().message();
  TransformerGraphBundle bundle = std::move(bundle_result.value());

  // 常量种子是刻意选择(测试可复现性,非 bug),同 test_training_smoke.cpp
  // 抑制方式:
  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 rng(20260713U);
  std::uniform_real_distribution<float> data_dist(-1.0F, 1.0F);
  std::uniform_real_distribution<float> teacher_dist(-0.15F, 0.15F);

  std::vector<float> x_values(static_cast<size_t>(kRows * kEmbedDim));
  for (float& v : x_values) v = data_dist(rng);

  // 教师权重:对每一行(样本)的 embed_dim=4 个原始输入值做线性组合(与网络
  // 实际的 TransformerEncoderBlock->Linear 特征空间不同基,不要求精确可
  // 拟合——同 test_cnn_convergence.cpp 文件头注释的既有取舍)。
  std::vector<float> teacher_weights(static_cast<size_t>(kEmbedDim));
  for (float& v : teacher_weights) v = teacher_dist(rng);

  std::vector<float> target_values(static_cast<size_t>(kRows * kOutDim), 0.0F);
  for (int64_t i = 0; i < kRows; ++i) {
    float acc = 0.0F;
    for (int64_t k = 0; k < kEmbedDim; ++k) {
      acc += x_values[static_cast<size_t>(i * kEmbedDim + k)] *
             teacher_weights[static_cast<size_t>(k)];
    }
    target_values[static_cast<size_t>(i)] = acc;
  }

  const Tensor x_tensor =
      MakeTensorFromFloats(x_values, Shape({kRows, kEmbedDim}), device_, *allocator_);
  const Tensor target_tensor =
      MakeTensorFromFloats(target_values, Shape({kRows, kOutDim}), device_, *allocator_);

  std::vector<Tensor> params =
      MakeUniformParamTensors(bundle.param_specs, rng, -0.1F, 0.1F, device_, *allocator_);
  const std::vector<frame::ir::TensorType> param_types = ParamTypesOf(bundle.param_specs);

  constexpr double kLearningRate = 0.05;
  constexpr int kNumTrainSteps = 300;
  const Result<std::vector<double>> loss_history = RunFullBatchSgdTraining(
      bundle.graph, x_tensor, params, param_types, target_tensor, kLearningRate, kNumTrainSteps);
  ASSERT_TRUE(loss_history.is_ok()) << loss_history.status().message();
  ASSERT_EQ(loss_history.value().size(), static_cast<size_t>(kNumTrainSteps));

  const double initial_loss = loss_history.value().front();
  const double final_loss = loss_history.value().back();

  EXPECT_LT(final_loss, initial_loss * 1e-4)
      << "initial_loss=" << initial_loss << " final_loss=" << final_loss;
  EXPECT_LT(final_loss, 1e-6) << "final_loss=" << final_loss;
}

TEST_F(TransformerConvergenceTest, ForwardGraphMatchesGolden) {
  Result<TransformerGraphBundle> bundle_result = BuildTransformerForwardGraph();
  ASSERT_TRUE(bundle_result.is_ok()) << bundle_result.status().message();
  TransformerGraphBundle bundle = std::move(bundle_result.value());

  EXPECT_TRUE(AssertGraphMatchesGolden(bundle.graph,
                                       "tests/cpp/nn/testdata/transformer_forward_expected.txt"));
}

}  // namespace
