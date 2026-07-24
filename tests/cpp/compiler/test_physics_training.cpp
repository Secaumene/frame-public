// M26 物理约束训练验收:PINN 以同一 build_backward_graph 连续三次变换
// 得到坐标二阶导与参数梯度;PINO 以 rfft/irfft 和物理波数构造周期谱二阶
// 导数。两条路径均真实编译执行并完成固定种子 SGD 收敛闭环。

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <numbers>
#include <random>
#include <vector>

#include <frame/compiler/autograd.h>
#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/executable.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/nn/layers.h>
#include <frame/nn/module.h>
#include <frame/ops/constant_utils.h>
#include <frame/ops/graph_builder.h>
#include <frame/ops/op_registry.h>
#include <frame/runtime/compile.h>

#include "../common/tolerance.h"
#include "../nn/convergence_test_helpers.h"

namespace {

using frame::DType;
using frame::DTypeCode;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::Tensor;
using frame::compiler::build_backward_graph;
using frame::compiler::build_sgd_update_graph;
using frame::hal::Backend;
using frame::hal::BackendRegistry;
using frame::hal::CompileOptions;
using frame::hal::Executable;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::Value;
using frame::nn::add_parameter_inputs;
using frame::nn::FourierFilter1d;
using frame::nn::Module;
using frame::nn::MseLoss;
using frame::nn::ParamSpec;
using frame::nn::testing::MakeCpuTensorType;
using frame::nn::testing::MakeTensorFromFloats;
using frame::nn::testing::MakeUniformParamTensors;
using frame::nn::testing::ParamTypesOf;
using frame::nn::testing::RunFullBatchSgdTraining;
using frame::ops::AttrMap;
using frame::ops::create_node_with_inferred_types;
using frame::testing::default_tolerance;
using frame::testing::relaxed_tolerance;
using frame::testing::tensor_all_close;

constexpr int64_t kPinnPoints = 4;
constexpr int64_t kPinoBatch = 1;
constexpr int64_t kPinoChannels = 1;
constexpr int64_t kPinoSteps = 8;

Result<Value*> BuildPeriodicSecondDerivative(Graph& graph, Value* x, double domain_length) {
  const Result<Node*> spectrum = create_node_with_inferred_types(graph, "rfft", {x});
  if (!spectrum.is_ok()) return spectrum.status();

  const Shape spectrum_shape = spectrum.value()->output(0)->type().shape;
  std::vector<double> multipliers(static_cast<size_t>(spectrum_shape.numel()), 0.0);
  const int64_t modes = kPinoSteps / 2 + 1;
  for (int64_t j = 0; j < modes; ++j) {
    const double wave_number =
        2.0 * std::numbers::pi_v<double> * static_cast<double>(j) / domain_length;
    const double multiplier = -wave_number * wave_number;
    multipliers[static_cast<size_t>(2 * j)] = multiplier;
    multipliers[static_cast<size_t>(2 * j + 1)] = multiplier;
  }
  const AttrMap constant_attrs{
      {"value", multipliers}, {"shape", spectrum_shape}, {"dtype", DType::of<float>()}};
  const Result<Node*> wave_number_node = create_node_with_inferred_types(
      graph, frame::ops::kConstantOpName, x->type().device, constant_attrs);
  if (!wave_number_node.is_ok()) return wave_number_node.status();
  const Result<Node*> weighted = create_node_with_inferred_types(
      graph, "mul", {spectrum.value()->output(0), wave_number_node.value()->output(0)});
  if (!weighted.is_ok()) return weighted.status();
  const AttrMap irfft_attrs{{"n", kPinoSteps}};
  const Result<Node*> derivative =
      create_node_with_inferred_types(graph, "irfft", {weighted.value()->output(0)}, irfft_attrs);
  if (!derivative.is_ok()) return derivative.status();
  return derivative.value()->output(0);
}

struct PinnTrainingBundle {
  Graph graph;
  frame::ir::TensorType weight_type;
};

Result<PinnTrainingBundle> BuildPinnTrainingGraph() {
  Graph forward("pinn_polynomial_forward");
  const Result<Value*> x = forward.add_graph_input(MakeCpuTensorType({kPinnPoints, 1}));
  if (!x.is_ok()) return x.status();
  const Result<Value*> weight = forward.add_graph_input(MakeCpuTensorType({1, 1}));
  if (!weight.is_ok()) return weight.status();
  const Result<Value*> target = forward.add_graph_input(MakeCpuTensorType({kPinnPoints, 1}));
  if (!target.is_ok()) return target.status();

  const Result<Node*> projected =
      create_node_with_inferred_types(forward, "matmul", {x.value(), weight.value()});
  if (!projected.is_ok()) return projected.status();
  const Result<Node*> solution =
      create_node_with_inferred_types(forward, "square", {projected.value()->output(0)});
  if (!solution.is_ok()) return solution.status();
  const AttrMap sum_all{{"axes", std::vector<int64_t>{}}};
  const Result<Node*> solution_sum =
      create_node_with_inferred_types(forward, "sum", {solution.value()->output(0)}, sum_all);
  if (!solution_sum.is_ok()) return solution_sum.status();
  Status status = forward.mark_output(solution.value()->output(0));
  if (!status.is_ok()) return status;
  status = forward.mark_output(solution_sum.value()->output(0));
  if (!status.is_ok()) return status;

  const std::vector<int32_t> wrt_x{0};
  Result<Graph> first_result = build_backward_graph(forward, /*loss_output_index=*/1, wrt_x);
  if (!first_result.is_ok()) return first_result.status();
  Graph first = std::move(first_result.value());
  const Result<Node*> first_derivative_sum =
      create_node_with_inferred_types(first, "sum", {first.outputs()[2]}, sum_all);
  if (!first_derivative_sum.is_ok()) return first_derivative_sum.status();
  status = first.mark_output(first_derivative_sum.value()->output(0));
  if (!status.is_ok()) return status;

  Result<Graph> second_result = build_backward_graph(first, /*loss_output_index=*/3, wrt_x);
  if (!second_result.is_ok()) return second_result.status();
  Graph second = std::move(second_result.value());
  const Result<Node*> residual_loss = create_node_with_inferred_types(
      second, "mse_loss", {second.outputs()[4], second.inputs()[2]});
  if (!residual_loss.is_ok()) return residual_loss.status();
  status = second.mark_output(residual_loss.value()->output(0));
  if (!status.is_ok()) return status;

  const std::vector<int32_t> wrt_weight{1};
  Result<Graph> third_result = build_backward_graph(second, /*loss_output_index=*/5, wrt_weight);
  if (!third_result.is_ok()) return third_result.status();
  PinnTrainingBundle bundle{std::move(third_result.value()), weight.value()->type()};
  return bundle;
}

class PhysicsTrainingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const Result<Backend*> backend = BackendRegistry::instance().get(frame::kCpuBackendName);
    ASSERT_TRUE(backend.is_ok()) << backend.status().message();
    allocator_ = backend.value()->allocator(frame::cpu_device());
    ASSERT_NE(allocator_, nullptr);
  }

  frame::hal::Allocator* allocator_ = nullptr;
};

TEST_F(PhysicsTrainingTest, PinnRepeatedTransformMatchesSecondDifferenceAndTrainsParameter) {
  Result<PinnTrainingBundle> bundle_result = BuildPinnTrainingGraph();
  ASSERT_TRUE(bundle_result.is_ok()) << bundle_result.status().message();
  PinnTrainingBundle bundle = std::move(bundle_result.value());
  ASSERT_EQ(bundle.graph.outputs().size(), 7U);
  ASSERT_TRUE(bundle.graph.verify(frame::ops::make_op_query()).is_ok());

  const Result<std::shared_ptr<Executable>> training_executable =
      frame::runtime::compile(bundle.graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(training_executable.is_ok()) << training_executable.status().message();
  const std::vector<frame::ir::TensorType> weight_types{bundle.weight_type};
  const Result<Graph> update_graph = build_sgd_update_graph(weight_types, 0.02);
  ASSERT_TRUE(update_graph.is_ok()) << update_graph.status().message();
  const Result<std::shared_ptr<Executable>> update_executable =
      frame::runtime::compile(update_graph.value(), frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(update_executable.is_ok()) << update_executable.status().message();

  const std::vector<float> x_values{-0.75F, -0.25F, 0.25F, 0.75F};
  const Tensor x =
      MakeTensorFromFloats(x_values, Shape({kPinnPoints, 1}), frame::cpu_device(), *allocator_);
  Tensor weight = MakeTensorFromFloats({0.4F}, Shape({1, 1}), frame::cpu_device(), *allocator_);
  const Tensor target =
      MakeTensorFromFloats(std::vector<float>(kPinnPoints, 2.0F), Shape({kPinnPoints, 1}),
                           frame::cpu_device(), *allocator_);

  std::vector<Tensor> initial_inputs{x, weight, target};
  const Result<std::vector<Tensor>> initial_outputs = frame::runtime::run_with_allocated_outputs(
      *training_executable.value(), frame::kCpuBackendName, initial_inputs);
  ASSERT_TRUE(initial_outputs.is_ok()) << initial_outputs.status().message();
  const Tensor expected_second =
      MakeTensorFromFloats(std::vector<float>(kPinnPoints, 2.0F * 0.4F * 0.4F),
                           Shape({kPinnPoints, 1}), frame::cpu_device(), *allocator_);
  EXPECT_TRUE(tensor_all_close(initial_outputs.value()[4], expected_second,
                               default_tolerance(DTypeCode::kFloat32)));

  constexpr double kDifferenceStep = 1e-2;
  std::vector<float> numeric_second;
  numeric_second.reserve(x_values.size());
  for (float coordinate : x_values) {
    const double center = static_cast<double>(coordinate) * 0.4;
    const double plus = (static_cast<double>(coordinate) + kDifferenceStep) * 0.4;
    const double minus = (static_cast<double>(coordinate) - kDifferenceStep) * 0.4;
    numeric_second.push_back(
        static_cast<float>((plus * plus - 2.0 * center * center + minus * minus) /
                           (kDifferenceStep * kDifferenceStep)));
  }
  const Tensor numeric_second_tensor = MakeTensorFromFloats(numeric_second, Shape({kPinnPoints, 1}),
                                                            frame::cpu_device(), *allocator_);
  EXPECT_TRUE(tensor_all_close(initial_outputs.value()[4], numeric_second_tensor,
                               relaxed_tolerance(DTypeCode::kFloat32)));

  double initial_loss = 0.0;
  double final_loss = 0.0;
  constexpr int kTrainingSteps = 100;
  for (int step = 0; step < kTrainingSteps; ++step) {
    std::vector<Tensor> train_inputs{x, weight, target};
    const Result<std::vector<Tensor>> train_outputs = frame::runtime::run_with_allocated_outputs(
        *training_executable.value(), frame::kCpuBackendName, train_inputs);
    ASSERT_TRUE(train_outputs.is_ok()) << train_outputs.status().message();
    const double loss =
        static_cast<double>(static_cast<const float*>(train_outputs.value()[5].raw_data())[0]);
    if (step == 0) initial_loss = loss;
    final_loss = loss;
    std::vector<Tensor> update_inputs{weight, train_outputs.value()[6]};
    const Result<std::vector<Tensor>> updated = frame::runtime::run_with_allocated_outputs(
        *update_executable.value(), frame::kCpuBackendName, update_inputs);
    ASSERT_TRUE(updated.is_ok()) << updated.status().message();
    weight = updated.value()[0];
  }
  EXPECT_LT(final_loss, initial_loss * 1e-4)
      << "initial_loss=" << initial_loss << " final_loss=" << final_loss;
  EXPECT_LT(final_loss, 1e-6) << "final_loss=" << final_loss;
}

TEST_F(PhysicsTrainingTest, PinoSpectralSecondDerivativeMatchesModeAndTrainsResidual) {
  Graph derivative_graph("pino_spectral_second_derivative");
  const Result<Value*> derivative_input =
      derivative_graph.add_graph_input(MakeCpuTensorType({kPinoBatch, kPinoChannels, kPinoSteps}));
  ASSERT_TRUE(derivative_input.is_ok()) << derivative_input.status().message();
  const Result<Value*> derivative = BuildPeriodicSecondDerivative(
      derivative_graph, derivative_input.value(), 2.0 * std::numbers::pi_v<double>);
  ASSERT_TRUE(derivative.is_ok()) << derivative.status().message();
  ASSERT_TRUE(derivative_graph.mark_output(derivative.value()).is_ok());
  ASSERT_TRUE(derivative_graph.verify(frame::ops::make_op_query()).is_ok());
  const Result<std::shared_ptr<Executable>> derivative_executable =
      frame::runtime::compile(derivative_graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(derivative_executable.is_ok()) << derivative_executable.status().message();

  std::vector<float> mode_values(static_cast<size_t>(kPinoSteps));
  for (int64_t i = 0; i < kPinoSteps; ++i) {
    mode_values[static_cast<size_t>(i)] = static_cast<float>(
        std::sin(2.0 * std::numbers::pi_v<double> * static_cast<double>(i) / kPinoSteps));
  }
  const Tensor mode =
      MakeTensorFromFloats(mode_values, Shape({kPinoBatch, kPinoChannels, kPinoSteps}),
                           frame::cpu_device(), *allocator_);
  std::vector<Tensor> derivative_inputs{mode};
  const Result<std::vector<Tensor>> derivative_outputs = frame::runtime::run_with_allocated_outputs(
      *derivative_executable.value(), frame::kCpuBackendName, derivative_inputs);
  ASSERT_TRUE(derivative_outputs.is_ok()) << derivative_outputs.status().message();
  std::vector<float> expected_values(mode_values.size());
  for (size_t i = 0; i < mode_values.size(); ++i) expected_values[i] = -mode_values[i];
  const Tensor expected =
      MakeTensorFromFloats(expected_values, Shape({kPinoBatch, kPinoChannels, kPinoSteps}),
                           frame::cpu_device(), *allocator_);
  EXPECT_TRUE(tensor_all_close(derivative_outputs.value()[0], expected,
                               relaxed_tolerance(DTypeCode::kFloat32)));

  Graph training_graph("pino_physics_residual_training");
  const Result<Value*> x =
      training_graph.add_graph_input(MakeCpuTensorType({kPinoBatch, kPinoChannels, kPinoSteps}));
  ASSERT_TRUE(x.is_ok()) << x.status().message();
  const Module filter =
      FourierFilter1d("pino", kPinoBatch, kPinoChannels, kPinoSteps, DType::of<float>());
  const std::vector<ParamSpec> param_specs = filter.parameters();
  const Result<std::vector<Value*>> params = add_parameter_inputs(training_graph, param_specs);
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  const Result<std::vector<Value*>> prediction =
      filter.build(training_graph, std::vector<Value*>{x.value()}, params.value());
  ASSERT_TRUE(prediction.is_ok()) << prediction.status().message();
  const Result<Value*> physical_second = BuildPeriodicSecondDerivative(
      training_graph, prediction.value()[0], 2.0 * std::numbers::pi_v<double>);
  ASSERT_TRUE(physical_second.is_ok()) << physical_second.status().message();
  const Result<Value*> target =
      training_graph.add_graph_input(MakeCpuTensorType({kPinoBatch, kPinoChannels, kPinoSteps}));
  ASSERT_TRUE(target.is_ok()) << target.status().message();
  const Result<std::vector<Value*>> loss =
      MseLoss("physics_loss")
          .build(training_graph, std::vector<Value*>{physical_second.value(), target.value()}, {});
  ASSERT_TRUE(loss.is_ok()) << loss.status().message();
  ASSERT_TRUE(training_graph.mark_output(loss.value()[0]).is_ok());

  std::vector<float> target_values(mode_values.size());
  for (size_t i = 0; i < mode_values.size(); ++i) target_values[i] = -2.0F * mode_values[i];
  const Tensor physics_target =
      MakeTensorFromFloats(target_values, Shape({kPinoBatch, kPinoChannels, kPinoSteps}),
                           frame::cpu_device(), *allocator_);
  // 固定 seed 是训练回归合同;纯 mode-1 输入使物理残差只约束对应谱权重。
  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 rng(20260731U);
  std::vector<Tensor> parameter_tensors =
      MakeUniformParamTensors(param_specs, rng, -0.1F, 0.1F, frame::cpu_device(), *allocator_);
  const Result<std::vector<double>> losses = RunFullBatchSgdTraining(
      training_graph, mode, parameter_tensors, ParamTypesOf(param_specs), physics_target,
      /*learning_rate=*/0.1, /*num_steps=*/120);
  ASSERT_TRUE(losses.is_ok()) << losses.status().message();
  EXPECT_LT(losses.value().back(), losses.value().front() * 1e-4)
      << "initial_loss=" << losses.value().front() << " final_loss=" << losses.value().back();
  EXPECT_LT(losses.value().back(), 1e-6) << "final_loss=" << losses.value().back();
}

}  // namespace
