// M27 LIFCell/SnnClassifier 工厂回归:手算静态递推、reset 完整代理梯度、
// 无参数/纯构图合同，以及分类器参数先序、固定流水线与错误路径。
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <string>
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
#include <frame/ops/graph_builder.h>
#include <frame/ops/op_registry.h>
#include <frame/runtime/compile.h>

#include "../common/tolerance.h"

namespace {

using frame::DType;
using frame::DTypeCode;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::Tensor;
using frame::hal::Backend;
using frame::hal::BackendRegistry;
using frame::hal::CompileOptions;
using frame::hal::Executable;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::TensorType;
using frame::ir::Value;
using frame::nn::add_parameter_inputs;
using frame::nn::LIFCell;
using frame::nn::Module;
using frame::nn::ParamSpec;
using frame::nn::SnnClassifier;
using frame::ops::OpRegistry;
using frame::testing::default_tolerance;
using frame::testing::tensor_all_close;

TensorType MakeCpuType(std::vector<int64_t> dims) {
  TensorType type;
  type.dtype = DType::of<float>();
  type.shape = Shape(std::move(dims));
  type.device = frame::cpu_device();
  return type;
}

int CountNodesWithOp(Graph& graph, const std::string& op) {
  int count = 0;
  for (Node* node : graph.topological_order()) {
    if (node->op() == op) ++count;
  }
  return count;
}

class SnnBatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const Result<Backend*> backend = BackendRegistry::instance().get(frame::kCpuBackendName);
    ASSERT_TRUE(backend.is_ok()) << backend.status().message();
    allocator_ = backend.value()->allocator(frame::cpu_device());
    ASSERT_NE(allocator_, nullptr);
  }

  Tensor MakeTensor(const Shape& shape, const std::vector<float>& values) {
    Tensor tensor =
        Tensor::empty(shape, DType::of<float>(), frame::cpu_device(), *allocator_).value();
    EXPECT_EQ(tensor.numel(), static_cast<int64_t>(values.size()));
    float* data = tensor.data<float>();
    for (size_t i = 0; i < values.size(); ++i) data[i] = values[i];
    return tensor;
  }

  frame::hal::Allocator* allocator_ = nullptr;
};

TEST(SnnBatchParameters, LifCellHasNoParameters) {
  const Module lif = LIFCell("lif", /*batch=*/2, /*num_steps=*/3, /*features=*/4,
                             /*decay=*/0.5, /*threshold=*/1.0, /*alpha=*/2.0, DType::of<float>());
  EXPECT_TRUE(lif.parameters().empty());
  EXPECT_TRUE(lif.children.empty());
}

TEST(SnnBatchParameters, ClassifierUsesApprovedChildPreorderAndShapesWithBias) {
  constexpr int64_t kBatch = 2;
  constexpr int64_t kSteps = 3;
  constexpr int64_t kInputDim = 4;
  constexpr int64_t kHiddenDim = 5;
  constexpr int64_t kClasses = 2;
  const Module model = SnnClassifier("snn", kBatch, kSteps, kInputDim, kHiddenDim, kClasses,
                                     /*decay=*/0.5, /*threshold=*/1.0, /*alpha=*/2.0,
                                     /*with_bias=*/true, DType::of<float>());
  ASSERT_EQ(model.children.size(), 3U);
  EXPECT_EQ(model.children[0].name, "input");
  EXPECT_EQ(model.children[1].name, "lif");
  EXPECT_EQ(model.children[2].name, "output");
  const std::vector<ParamSpec> params = model.parameters();
  ASSERT_EQ(params.size(), 4U);
  EXPECT_EQ(params[0].name, "snn.input.weight");
  EXPECT_EQ(params[0].type.shape, Shape({kInputDim, kHiddenDim}));
  EXPECT_EQ(params[1].name, "snn.input.bias");
  EXPECT_EQ(params[1].type.shape, Shape({kBatch * kSteps, kHiddenDim}));
  EXPECT_EQ(params[2].name, "snn.output.weight");
  EXPECT_EQ(params[2].type.shape, Shape({kHiddenDim, kClasses}));
  EXPECT_EQ(params[3].name, "snn.output.bias");
  EXPECT_EQ(params[3].type.shape, Shape({kBatch * kSteps, kClasses}));
}

TEST(SnnBatchParameters, ClassifierWithoutBiasHasOnlyTwoWeights) {
  const Module model =
      SnnClassifier("snn", 2, 3, 4, 5, 2, 0.5, 1.0, 2.0, false, DType::of<float>());
  const std::vector<ParamSpec> params = model.parameters();
  ASSERT_EQ(params.size(), 2U);
  EXPECT_EQ(params[0].name, "snn.input.weight");
  EXPECT_EQ(params[1].name, "snn.output.weight");
}

TEST_F(SnnBatchTest, LifForwardMatchesHandComputedMultiStepResetRecurrence) {
  constexpr int64_t kBatch = 1;
  constexpr int64_t kSteps = 4;
  constexpr int64_t kFeatures = 2;
  constexpr double kDecay = 0.5;
  constexpr double kThreshold = 1.0;
  Graph graph("lif_hand_recurrence");
  Value* x = graph.add_graph_input(MakeCpuType({kBatch, kSteps, kFeatures})).value();
  const Module lif = LIFCell("lif", kBatch, kSteps, kFeatures, kDecay, kThreshold,
                             /*alpha=*/2.0, DType::of<float>());
  const Result<std::vector<Value*>> outputs =
      lif.build(graph, std::vector<Value*>{x}, std::vector<Value*>{});
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  ASSERT_EQ(outputs.value().size(), 1U);
  ASSERT_TRUE(graph.mark_output(outputs.value()[0]).is_ok());
  const Result<std::shared_ptr<Executable>> executable =
      frame::runtime::compile(graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  // 特征 0:v_pre=[0.6,1.0(reset),0.2,1.0(reset)] -> [0,1,0,1]。
  // 特征 1:v_pre=[1.2(reset),0.1,0.55,0.675] -> [1,0,0,0]。
  const Tensor input = MakeTensor(Shape({kBatch, kSteps, kFeatures}),
                                  {0.6F, 1.2F, 0.7F, 0.1F, 0.2F, 0.5F, 0.9F, 0.4F});
  const std::vector<Tensor> run_inputs{input};
  const Result<std::vector<Tensor>> actual = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, run_inputs);
  ASSERT_TRUE(actual.is_ok()) << actual.status().message();
  const Tensor expected = MakeTensor(Shape({kBatch, kSteps, kFeatures}),
                                     {0.0F, 1.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F});
  EXPECT_TRUE(
      tensor_all_close(actual.value()[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(SnnBatchTest, LifResetGradientIncludesSpikeAndVoltageTerms) {
  constexpr double kDecay = 0.5;
  constexpr double kThreshold = 1.0;
  constexpr double kAlpha = 2.0;
  Graph forward("lif_reset_gradient");
  Value* x = forward.add_graph_input(MakeCpuType({1, 2, 1})).value();
  const Module lif = LIFCell("lif", 1, 2, 1, kDecay, kThreshold, kAlpha, DType::of<float>());
  const Result<std::vector<Value*>> spikes =
      lif.build(forward, std::vector<Value*>{x}, std::vector<Value*>{});
  ASSERT_TRUE(spikes.is_ok()) << spikes.status().message();
  const frame::ops::AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* loss =
      frame::ops::create_node_with_inferred_types(forward, "sum", {spikes.value()[0]}, sum_attrs)
          .value();
  ASSERT_TRUE(forward.mark_output(loss, 0).is_ok());
  const std::vector<int32_t> wrt{0};
  const Result<Graph> training = frame::compiler::build_backward_graph(forward, 0, wrt);
  ASSERT_TRUE(training.is_ok()) << training.status().message();
  const Result<std::shared_ptr<Executable>> executable =
      frame::runtime::compile(training.value(), frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  const std::vector<float> x_values{1.2F, 0.2F};
  const Tensor input = MakeTensor(Shape({1, 2, 1}), x_values);
  const std::vector<Tensor> run_inputs{input};
  const Result<std::vector<Tensor>> actual = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, run_inputs);
  ASSERT_TRUE(actual.is_ok()) << actual.status().message();
  ASSERT_EQ(actual.value().size(), 2U);

  const auto proxy_derivative = [](double voltage) {
    constexpr double alpha = kAlpha;
    constexpr double threshold = kThreshold;
    const double s = 1.0 / (1.0 + std::exp(-alpha * (voltage - threshold)));
    return alpha * s * (1.0 - s);
  };
  const double v_pre0 = 1.2;
  const double v_pre1 = 0.2;  // 第一步 spike=1 后 reset 到 0。
  const double q0 = proxy_derivative(v_pre0);
  const double q1 = proxy_derivative(v_pre1);
  const double gx1 = q1;
  // 完整 reset 局部导数:1-spike-v_pre*q；此处 spike0=1，不能只保留
  // (1-spike) 或把 reset detach，否则 gx0 不会等于下式。
  const double gx0 = q0 + kDecay * gx1 * (1.0 - 1.0 - v_pre0 * q0);
  const Tensor expected =
      MakeTensor(Shape({1, 2, 1}), {static_cast<float>(gx0), static_cast<float>(gx1)});
  EXPECT_TRUE(
      tensor_all_close(actual.value()[1], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST(SnnBatchConstructionPurity, LifBuildAddsOnlyPureGraphNodesAndPreservesShape) {
  Graph graph("lif_purity");
  Value* x = graph.add_graph_input(MakeCpuType({2, 3, 4})).value();
  const Module lif = LIFCell("lif", 2, 3, 4, 0.5, 1.0, 2.0, DType::of<float>());
  const size_t before = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      lif.build(graph, std::vector<Value*>{x}, std::vector<Value*>{});
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  EXPECT_GT(graph.topological_order().size(), before);
  ASSERT_EQ(outputs.value().size(), 1U);
  EXPECT_EQ(outputs.value()[0]->type().shape, Shape({2, 3, 4}));
  EXPECT_EQ(CountNodesWithOp(graph, "heaviside_surrogate"), 3);
  EXPECT_EQ(CountNodesWithOp(graph, "concat"), 1);
  for (Node* node : graph.topological_order()) {
    if (node->op() == "graph_input") continue;
    const auto* schema = OpRegistry::instance().find(node->op());
    ASSERT_NE(schema, nullptr);
    EXPECT_FALSE(schema->has_trait(frame::ops::OpTrait::kHasSideEffect));
  }
}

TEST(SnnBatchConstructionPurity, ClassifierUsesFixedTwoLinearSpikeAndTimeSumPipeline) {
  Graph graph("snn_classifier_pipeline");
  Value* x = graph.add_graph_input(MakeCpuType({2, 3, 4})).value();
  const Module model =
      SnnClassifier("snn", 2, 3, 4, 5, 2, 0.5, 1.0, 2.0, false, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, model.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  const Result<std::vector<Value*>> outputs =
      model.build(graph, std::vector<Value*>{x}, params.value());
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  ASSERT_EQ(outputs.value().size(), 1U);
  EXPECT_EQ(outputs.value()[0]->type().shape, Shape({2, 2}));
  EXPECT_EQ(CountNodesWithOp(graph, "matmul"), 2);
  EXPECT_EQ(CountNodesWithOp(graph, "heaviside_surrogate"), 3);
  EXPECT_EQ(CountNodesWithOp(graph, "sum"), 1);
  EXPECT_EQ(outputs.value()[0]->producer()->op(), "sum");
}

TEST(SnnBatchConstructionNegative, LifAndClassifierRejectInvalidContractsAtBuild) {
  const std::vector<Module> invalid_lif{
      LIFCell("lif", 0, 2, 1, 0.5, 1.0, 2.0, DType::of<float>()),
      LIFCell("lif", 1, 0, 1, 0.5, 1.0, 2.0, DType::of<float>()),
      LIFCell("lif", 1, 2, 0, 0.5, 1.0, 2.0, DType::of<float>()),
      LIFCell("lif", 1, 2, 1, -0.1, 1.0, 2.0, DType::of<float>()),
      LIFCell("lif", 1, 2, 1, 1.0, 1.0, 2.0, DType::of<float>()),
      LIFCell("lif", 1, 2, 1, 0.5, 0.0, 2.0, DType::of<float>()),
      LIFCell("lif", 1, 2, 1, 0.5, 1.0, 0.0, DType::of<float>()),
  };
  for (const Module& lif : invalid_lif) {
    Graph graph("invalid_lif");
    Value* x = graph.add_graph_input(MakeCpuType({1, 2, 1})).value();
    EXPECT_FALSE(lif.build(graph, std::vector<Value*>{x}, std::vector<Value*>{}).is_ok());
  }

  Graph wrong_shape("snn_wrong_shape");
  Value* bad_x = wrong_shape.add_graph_input(MakeCpuType({2, 4, 4})).value();
  const Module model =
      SnnClassifier("snn", 2, 3, 4, 5, 2, 0.5, 1.0, 2.0, false, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(wrong_shape, model.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  EXPECT_FALSE(model.build(wrong_shape, std::vector<Value*>{bad_x}, params.value()).is_ok());

  Graph wrong_inputs("snn_wrong_inputs");
  const Result<std::vector<Value*>> wrong_input_params =
      add_parameter_inputs(wrong_inputs, model.parameters());
  ASSERT_TRUE(wrong_input_params.is_ok()) << wrong_input_params.status().message();
  EXPECT_FALSE(
      model.build(wrong_inputs, std::vector<Value*>{}, wrong_input_params.value()).is_ok());

  Graph wrong_params("snn_wrong_params");
  Value* good_x = wrong_params.add_graph_input(MakeCpuType({2, 3, 4})).value();
  std::vector<Value*> short_params = add_parameter_inputs(wrong_params, model.parameters()).value();
  short_params.pop_back();
  EXPECT_FALSE(model.build(wrong_params, std::vector<Value*>{good_x}, short_params).is_ok());
}

TEST(SnnBatchConstructionNegative, LifAndClassifierRejectInt64ShapeOverflowAtomically) {
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  {
    Graph graph("lif_overflow");
    Value* x = graph.add_graph_input(MakeCpuType({1, 1, 1})).value();
    const Module lif = LIFCell("lif", kMax, 2, 2, 0.5, 1.0, 2.0, DType::of<float>());
    const size_t before = graph.topological_order().size();
    const Result<std::vector<Value*>> result =
        lif.build(graph, std::vector<Value*>{x}, std::vector<Value*>{});
    ASSERT_FALSE(result.is_ok());
    EXPECT_NE(result.status().message().find("int64 shape range"), std::string_view::npos);
    EXPECT_EQ(graph.topological_order().size(), before);
  }
  {
    Graph graph("snn_overflow");
    Value* x = graph.add_graph_input(MakeCpuType({1, 1, 1})).value();
    const Module model =
        SnnClassifier("snn", kMax, 2, 2, 2, 2, 0.5, 1.0, 2.0, false, DType::of<float>());
    const Result<std::vector<Value*>> params = add_parameter_inputs(graph, model.parameters());
    ASSERT_TRUE(params.is_ok()) << params.status().message();
    const size_t before = graph.topological_order().size();
    const Result<std::vector<Value*>> result =
        model.build(graph, std::vector<Value*>{x}, params.value());
    ASSERT_FALSE(result.is_ok());
    EXPECT_NE(result.status().message().find("int64 shape range"), std::string_view::npos);
    EXPECT_EQ(graph.topological_order().size(), before);
  }
}

}  // namespace
