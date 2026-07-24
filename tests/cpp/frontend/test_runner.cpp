// run_training 端到端训练冒烟测试(docs/architecture/frontend-dsl.md,镜像
// tests/cpp/compiler/test_training_loop.cpp 的端到端训练循环结构)。禁用
// EXPECT_NEAR/精确数值 golden(BUILD-011:分布实现跨标准库不逐位一致),
// 仅用阈值断言判定收敛趋势——口径参照 test_training_loop.cpp 的
// 0.21 -> 5.6e-05 收敛论证:tiny_mlp 的 batch==hidden_dim==8 使第二层 W2
// 对固定隐藏特征矩阵而言是满秩线性方程组解,300 步 SGD(lr=0.05)足以把
// loss 压到远低于 1e-3 的量级(本机实测見下方注释)。

#include <cstddef>
#include <gtest/gtest.h>

#include <frame/core/status.h>
#include <frame/frontend/model_spec.h>
#include <frame/frontend/runner.h>

#include "tiny_mlp_spec_helper.h"

namespace {

using frame::Result;
using frame::frontend::ModelSpec;
using frame::frontend::run_training;
using frame::frontend::RunOptions;
using frame::frontend::RunReport;
using frame::frontend::testing::make_tiny_mlp_spec;

// 300 步收敛用例(阈值经本机实测校准,留有充分安全边际,不是刚好卡阈值):
// kSeed=20260713、kNumSteps=300、kLearningRate=0.05 下实测
// initial_loss≈0.4584、final_loss≈3.6e-06,余量约 2 个数量级(判据 1e-3)。
TEST(RunTrainingTest, ConvergesWithinThreeHundredSteps) {
  const ModelSpec spec = make_tiny_mlp_spec();
  const Result<RunReport> report_result = run_training(spec, RunOptions{"cpu"});
  ASSERT_TRUE(report_result.is_ok()) << report_result.status().message();
  const RunReport& report = report_result.value();

  ASSERT_EQ(report.loss_history.size(), static_cast<size_t>(spec.training.steps));
  EXPECT_EQ(report.loss_history.size(), 300u);

  const double initial_loss = report.loss_history.front();
  const double final_loss = report.loss_history.back();
  EXPECT_GT(initial_loss, final_loss)
      << "initial_loss=" << initial_loss << " final_loss=" << final_loss;
  EXPECT_LT(final_loss, 1e-3) << "final_loss=" << final_loss;
  EXPECT_LT(report.final_loss, 1e-3) << "report.final_loss=" << report.final_loss;

  // 推理阶段(lower_to_inference_graph 编译执行一次)产出的预测值与 loss 形状
  // 一致:tiny_mlp 输出层 [batch=8, out=1]。
  EXPECT_EQ(report.final_predictions.size(), 8u);
}

// 50 步版本:只断言首末 loss 呈下降趋势(不逐步断言,阈值同 BUILD-011 禁止
// 手写 EXPECT_NEAR 精神,单纯比较大小)。
TEST(RunTrainingTest, FiftyStepsShowsDecreasingTrend) {
  ModelSpec spec = make_tiny_mlp_spec();
  spec.training.steps = 50;

  const Result<RunReport> report_result = run_training(spec, RunOptions{"cpu"});
  ASSERT_TRUE(report_result.is_ok()) << report_result.status().message();
  const RunReport& report = report_result.value();

  ASSERT_EQ(report.loss_history.size(), 50u);
  EXPECT_GT(report.loss_history.front(), report.loss_history.back())
      << "initial_loss=" << report.loss_history.front()
      << " final_loss=" << report.loss_history.back();
}

// run_training 内部先调用 lower_to_graph -> validate(spec),失败原样透传。
TEST(RunTrainingTest, PropagatesValidateFailure) {
  ModelSpec spec = make_tiny_mlp_spec();
  spec.optimizer.learning_rate = -1.0;

  const Result<RunReport> report_result = run_training(spec, RunOptions{"cpu"});
  ASSERT_FALSE(report_result.is_ok());
  EXPECT_EQ(report_result.status().code(), frame::ErrorCode::kInvalidArgument);
}

}  // namespace
