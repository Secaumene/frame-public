// M25 Mamba/FourierMamba 工厂回归:参数先序与 shape、构图纯度、关键算子、
// 输出 shape，以及输入数/参数数/输入 shape 错误路径。
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/nn/layers.h>
#include <frame/nn/module.h>

namespace {

using frame::DType;
using frame::Result;
using frame::Shape;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::TensorType;
using frame::ir::Value;
using frame::nn::add_parameter_inputs;
using frame::nn::FourierMamba;
using frame::nn::Mamba;
using frame::nn::Module;
using frame::nn::ParamSpec;

constexpr int64_t kBatch = 2;
constexpr int64_t kChannels = 3;
constexpr int64_t kSteps = 4;
constexpr int64_t kKernel = 2;
constexpr int64_t kRows = kBatch * kSteps;

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

std::vector<std::string> ParamNames(const std::vector<ParamSpec>& params) {
  std::vector<std::string> names;
  names.reserve(params.size());
  for (const ParamSpec& param : params) names.push_back(param.name);
  return names;
}

TEST(SsmBatchParameters, MambaUsesApprovedPreorderNamesAndShapes) {
  const Module mamba = Mamba("m", kBatch, kChannels, kSteps, kKernel, DType::of<float>());
  const std::vector<ParamSpec> params = mamba.parameters();
  const std::vector<std::string> expected_names{
      "m.conv.weight", "m.conv.bias", "m.input.weight", "m.input.bias", "m.a.weight", "m.a.bias",
      "m.b.weight",    "m.b.bias",    "m.c.weight",     "m.c.bias",     "m.d.weight", "m.d.bias",
      "m.gate.weight", "m.gate.bias", "m.out.weight",   "m.out.bias",
  };
  EXPECT_EQ(ParamNames(params), expected_names);
  ASSERT_EQ(params.size(), 16U);
  EXPECT_EQ(params[0].type.shape, Shape({kChannels, 1, kKernel}));
  EXPECT_EQ(params[1].type.shape, Shape({kChannels}));
  for (size_t i = 2; i < params.size(); i += 2) {
    EXPECT_EQ(params[i].type.shape, Shape({kChannels, kChannels}));
    EXPECT_EQ(params[i + 1].type.shape, Shape({kRows, kChannels}));
    EXPECT_EQ(params[i].type.dtype, DType::of<float>());
    EXPECT_EQ(params[i + 1].type.dtype, DType::of<float>());
  }
}

TEST(SsmBatchParameters, FourierMambaAppendsReusedFourierFilterParameters) {
  const Module model = FourierMamba("fm", kBatch, kChannels, kSteps, kKernel, DType::of<float>());
  const std::vector<ParamSpec> params = model.parameters();
  ASSERT_EQ(params.size(), 18U);
  EXPECT_EQ(params.front().name, "fm.mamba.conv.weight");
  EXPECT_EQ(params[15].name, "fm.mamba.out.bias");
  EXPECT_EQ(params[16].name, "fm.fourier.w_re");
  EXPECT_EQ(params[16].type.shape, Shape({kBatch, kChannels, kSteps / 2 + 1, 1}));
  EXPECT_EQ(params[17].name, "fm.fourier.w_im");
  EXPECT_EQ(params[17].type.shape, Shape({kBatch, kChannels, kSteps / 2 + 1, 1}));
  ASSERT_EQ(model.children.size(), 2U);
  EXPECT_EQ(model.children[0].name, "mamba");
  EXPECT_EQ(model.children[1].name, "fourier");
}

TEST(SsmBatchConstructionPurity, MambaAddsExpectedGraphOnlyNodes) {
  Graph graph("mamba_purity");
  Value* x = graph.add_graph_input(MakeCpuType({kBatch, kChannels, kSteps})).value();
  const Module mamba = Mamba("m", kBatch, kChannels, kSteps, kKernel, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, mamba.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();

  const size_t before = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      mamba.build(graph, std::vector<Value*>{x}, params.value());
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  // conv/slice/布局变换、六路投影、scan+gate、out 投影合计 42 个纯 IR 节点。
  EXPECT_EQ(graph.topological_order().size() - before, 42U);
  ASSERT_EQ(outputs.value().size(), 1U);
  EXPECT_EQ(outputs.value()[0]->type().shape, Shape({kBatch, kChannels, kSteps}));
  EXPECT_EQ(CountNodesWithOp(graph, "conv1d"), 1);
  EXPECT_EQ(CountNodesWithOp(graph, "selective_scan"), 1);
  EXPECT_EQ(CountNodesWithOp(graph, "matmul"), 7);
  EXPECT_EQ(CountNodesWithOp(graph, "sigmoid"), 2);
  EXPECT_EQ(CountNodesWithOp(graph, "tanh"), 4);
}

TEST(SsmBatchConstructionPurity, FourierMambaReusesMambaAndFourierBranches) {
  Graph graph("fourier_mamba_purity");
  Value* x = graph.add_graph_input(MakeCpuType({kBatch, kChannels, kSteps})).value();
  const Module model = FourierMamba("fm", kBatch, kChannels, kSteps, kKernel, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, model.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();

  const size_t before = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      model.build(graph, std::vector<Value*>{x}, params.value());
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  // Mamba(42)+FourierFilter1d(13)+末端 add/tanh(2)=57。
  EXPECT_EQ(graph.topological_order().size() - before, 57U);
  ASSERT_EQ(outputs.value().size(), 1U);
  EXPECT_EQ(outputs.value()[0]->type().shape, Shape({kBatch, kChannels, kSteps}));
  EXPECT_EQ(CountNodesWithOp(graph, "selective_scan"), 1);
  EXPECT_EQ(CountNodesWithOp(graph, "rfft"), 1);
  EXPECT_EQ(CountNodesWithOp(graph, "irfft"), 1);
  EXPECT_EQ(outputs.value()[0]->producer()->op(), "tanh");
}

TEST(SsmBatchConstructionNegative, MambaRejectsWrongInputCount) {
  Graph graph("mamba_wrong_inputs");
  const Module mamba = Mamba("m", kBatch, kChannels, kSteps, kKernel, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, mamba.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  EXPECT_FALSE(mamba.build(graph, std::vector<Value*>{}, params.value()).is_ok());
}

TEST(SsmBatchConstructionNegative, FourierMambaRejectsWrongInputCount) {
  Graph graph("fourier_mamba_wrong_inputs");
  const Module model = FourierMamba("fm", kBatch, kChannels, kSteps, kKernel, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, model.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  EXPECT_FALSE(model.build(graph, std::vector<Value*>{}, params.value()).is_ok());
}

TEST(SsmBatchConstructionNegative, MambaRejectsWrongParameterCount) {
  Graph graph("mamba_wrong_params");
  Value* x = graph.add_graph_input(MakeCpuType({kBatch, kChannels, kSteps})).value();
  const Module mamba = Mamba("m", kBatch, kChannels, kSteps, kKernel, DType::of<float>());
  std::vector<Value*> params = add_parameter_inputs(graph, mamba.parameters()).value();
  params.pop_back();
  EXPECT_FALSE(mamba.build(graph, std::vector<Value*>{x}, params).is_ok());
}

TEST(SsmBatchConstructionNegative, MambaRejectsInputShapeMismatch) {
  Graph graph("mamba_wrong_shape");
  Value* x = graph.add_graph_input(MakeCpuType({kBatch, kChannels + 1, kSteps})).value();
  const Module mamba = Mamba("m", kBatch, kChannels, kSteps, kKernel, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, mamba.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  EXPECT_FALSE(mamba.build(graph, std::vector<Value*>{x}, params.value()).is_ok());
}

TEST(SsmBatchConstructionNegative, MambaRejectsShorterAndLongerTimeAxesAtomically) {
  const Module mamba = Mamba("m", kBatch, kChannels, kSteps, kKernel, DType::of<float>());
  for (const int64_t actual_steps : {kSteps - 1, kSteps + 1}) {
    Graph graph("mamba_wrong_time_axis");
    Value* x = graph.add_graph_input(MakeCpuType({kBatch, kChannels, actual_steps})).value();
    const Result<std::vector<Value*>> params = add_parameter_inputs(graph, mamba.parameters());
    ASSERT_TRUE(params.is_ok()) << params.status().message();
    const size_t before = graph.topological_order().size();
    const Result<std::vector<Value*>> result =
        mamba.build(graph, std::vector<Value*>{x}, params.value());
    EXPECT_FALSE(result.is_ok());
    EXPECT_EQ(result.status().code(), frame::ErrorCode::kInvalidArgument);
    EXPECT_EQ(graph.topological_order().size(), before);
  }
}

TEST(SsmBatchConstructionNegative, MambaRejectsInt64ShapeOverflowAtomically) {
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  const std::vector<Module> invalid{
      Mamba("rows", kMax, 2, 2, 2, DType::of<float>()),
      Mamba("kernel", 1, 2, 2, kMax, DType::of<float>()),
      Mamba("weight", 1, kMax, 1, 1, DType::of<float>()),
      FourierMamba("fourier", kMax, 2, 2, 2, DType::of<float>()),
      FourierMamba("fourier_packed", 1, 2, kMax / 2, 1, DType::of<float>()),
  };
  for (const Module& mamba : invalid) {
    Graph graph("mamba_overflow");
    Value* x = graph.add_graph_input(MakeCpuType({1, 1, 1})).value();
    const Result<std::vector<Value*>> params = add_parameter_inputs(graph, mamba.parameters());
    ASSERT_TRUE(params.is_ok()) << params.status().message();
    const size_t before = graph.topological_order().size();
    const Result<std::vector<Value*>> result =
        mamba.build(graph, std::vector<Value*>{x}, params.value());
    ASSERT_FALSE(result.is_ok());
    EXPECT_EQ(result.status().code(), frame::ErrorCode::kInvalidArgument);
    EXPECT_NE(result.status().message().find("int64 shape range"), std::string_view::npos);
    EXPECT_EQ(graph.topological_order().size(), before);
  }
}

}  // namespace
