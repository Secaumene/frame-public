// M28 GraphConv/HypergraphConv 固定拓扑工厂:参数先序、构图纯度、错误原子性
// 与手算有向度归一化/二段超图传播。
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/executable.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/nn/layers.h>
#include <frame/nn/module.h>
#include <frame/ops/constant_utils.h>
#include <frame/runtime/compile.h>

#include "../common/tolerance.h"

namespace {

using frame::DType;
using frame::DTypeCode;
using frame::Result;
using frame::Shape;
using frame::Tensor;
using frame::hal::BackendRegistry;
using frame::hal::Executable;
using frame::ir::Graph;
using frame::ir::TensorType;
using frame::ir::Value;
using frame::nn::add_parameter_inputs;
using frame::nn::GraphConv;
using frame::nn::HypergraphConv;
using frame::nn::Module;
using frame::nn::ParamSpec;
using frame::testing::default_tolerance;
using frame::testing::tensor_all_close;

TensorType MakeCpuType(std::vector<int64_t> dims, DType dtype = DType::of<float>()) {
  TensorType type;
  type.dtype = dtype;
  type.shape = Shape(std::move(dims));
  type.device = frame::cpu_device();
  return type;
}

int CountOp(Graph& graph, std::string_view op) {
  int count = 0;
  for (const auto* node : graph.topological_order()) {
    if (node->op() == op) ++count;
  }
  return count;
}

class GnnBatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const Result<frame::hal::Backend*> backend =
        BackendRegistry::instance().get(frame::kCpuBackendName);
    ASSERT_TRUE(backend.is_ok()) << backend.status().message();
    allocator_ = backend.value()->allocator(frame::cpu_device());
    ASSERT_NE(allocator_, nullptr);
  }

  Tensor MakeTensor(const Shape& shape, const std::vector<float>& values) {
    Tensor tensor =
        Tensor::empty(shape, DType::of<float>(), frame::cpu_device(), *allocator_).value();
    float* data = tensor.data<float>();
    for (size_t i = 0; i < values.size(); ++i) data[i] = values[i];
    return tensor;
  }

  Tensor Run(const Module& model, const Shape& input_shape, const std::vector<float>& input,
             const Shape& weight_shape, const std::vector<float>& weight) {
    Graph graph("gnn_numeric");
    Value* x = graph.add_graph_input(MakeCpuType(input_shape.dims())).value();
    const Result<std::vector<Value*>> params = add_parameter_inputs(graph, model.parameters());
    EXPECT_TRUE(params.is_ok()) << params.status().message();
    const Result<std::vector<Value*>> outputs =
        model.build(graph, std::vector<Value*>{x}, params.value());
    EXPECT_TRUE(outputs.is_ok()) << outputs.status().message();
    EXPECT_TRUE(graph.mark_output(outputs.value()[0]).is_ok());
    const Result<std::shared_ptr<Executable>> executable =
        frame::runtime::compile(graph, frame::kCpuBackendName, frame::hal::CompileOptions{});
    EXPECT_TRUE(executable.is_ok()) << executable.status().message();
    const std::vector<Tensor> tensors{MakeTensor(input_shape, input),
                                      MakeTensor(weight_shape, weight)};
    const Result<std::vector<Tensor>> result = frame::runtime::run_with_allocated_outputs(
        *executable.value(), frame::kCpuBackendName, tensors);
    EXPECT_TRUE(result.is_ok()) << result.status().message();
    return result.value()[0];
  }

  frame::hal::Allocator* allocator_ = nullptr;
};

TEST(GnnParametersTest, GraphConvAndHypergraphConvHaveOneLinearWeightChild) {
  const Module graph = GraphConv("g", 4, 2, 3, {0, 1}, {1, 2}, DType::of<float>());
  ASSERT_EQ(graph.children.size(), 1U);
  EXPECT_EQ(graph.children[0].name, "linear");
  const std::vector<ParamSpec> graph_params = graph.parameters();
  ASSERT_EQ(graph_params.size(), 1U);
  EXPECT_EQ(graph_params[0].name, "g.linear.weight");
  EXPECT_EQ(graph_params[0].type.shape, Shape({2, 3}));

  const Module hyper = HypergraphConv("h", 4, 3, 2, 3, {0, 1}, {0, 0}, DType::of<float>());
  ASSERT_EQ(hyper.children.size(), 1U);
  EXPECT_EQ(hyper.children[0].name, "linear");
  const std::vector<ParamSpec> hyper_params = hyper.parameters();
  ASSERT_EQ(hyper_params.size(), 1U);
  EXPECT_EQ(hyper_params[0].name, "h.linear.weight");
  EXPECT_EQ(hyper_params[0].type.shape, Shape({2, 3}));
}

TEST_F(GnnBatchTest, GraphConvMatchesDirectedNormalizationWithDuplicatesSelfLoopAndIsolates) {
  const Module model =
      GraphConv("g", 4, 2, 2, {0, 0, 1, 1, 1, 2}, {1, 1, 1, 2, 1, 2}, DType::of<float>());
  const Tensor actual =
      Run(model, Shape({4, 2}), {1, 2, 3, 4, 5, 6, 7, 8}, Shape({2, 2}), {1, 0, 0, 1});
  // 节点 1 同时接收重复 0->1 与重复自环 1->1；节点 3 无入边。
  const Tensor expected =
      MakeTensor(Shape({4, 2}), {0, 0, 2.4391576F, 3.723615F, 4.7602787F, 5.8756337F, 0, 0});
  EXPECT_TRUE(tensor_all_close(actual, expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(GnnBatchTest, HypergraphConvMatchesTwoStagePropagationWithEmptyEdgeAndIsolates) {
  const Module model =
      HypergraphConv("h", 4, 3, 2, 2, {0, 1, 1, 2}, {0, 0, 1, 1}, DType::of<float>());
  const Tensor actual =
      Run(model, Shape({4, 2}), {1, 2, 3, 4, 5, 6, 7, 8}, Shape({2, 2}), {1, 0, 0, 1});
  // 超边 2 为空、节点 3 孤立；其余节点按 Dv^-1/2 H De^-1 H^T Dv^-1/2。
  const Tensor expected = MakeTensor(Shape({4, 2}), {1.5606601F, 2.4142137F, 3.6213202F, 4.8284273F,
                                                     3.5606601F, 4.4142137F, 0, 0});
  EXPECT_TRUE(tensor_all_close(actual, expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(GnnBatchTest, HypergraphEmptyIncidenceProducesZeros) {
  const Module model = HypergraphConv("h", 3, 2, 2, 2, {}, {}, DType::of<float>());
  const Tensor actual = Run(model, Shape({3, 2}), {1, 2, 3, 4, 5, 6}, Shape({2, 2}), {1, 0, 0, 1});
  const Tensor expected = MakeTensor(Shape({3, 2}), std::vector<float>(6, 0));
  EXPECT_TRUE(tensor_all_close(actual, expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST(GnnConstructionTest, GraphConvAndHypergraphConvHavePureExpectedPipelines) {
  {
    Graph graph("graph_conv_purity");
    Value* x = graph.add_graph_input(MakeCpuType({4, 2})).value();
    const Module model = GraphConv("g", 4, 2, 3, {0, 1, 2}, {1, 2, 3}, DType::of<float>());
    const Result<std::vector<Value*>> params = add_parameter_inputs(graph, model.parameters());
    const Result<std::vector<Value*>> outputs =
        model.build(graph, std::vector<Value*>{x}, params.value());
    ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
    EXPECT_EQ(outputs.value()[0]->type().shape, Shape({4, 3}));
    EXPECT_EQ(CountOp(graph, "gather"), 1);
    EXPECT_EQ(CountOp(graph, "scatter_add"), 1);
    EXPECT_EQ(CountOp(graph, "matmul"), 1);
  }
  {
    Graph graph("hypergraph_conv_purity");
    Value* x = graph.add_graph_input(MakeCpuType({4, 2})).value();
    const Module model =
        HypergraphConv("h", 4, 3, 2, 3, {0, 1, 1, 2}, {0, 0, 1, 1}, DType::of<float>());
    const Result<std::vector<Value*>> params = add_parameter_inputs(graph, model.parameters());
    const Result<std::vector<Value*>> outputs =
        model.build(graph, std::vector<Value*>{x}, params.value());
    ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
    EXPECT_EQ(outputs.value()[0]->type().shape, Shape({4, 3}));
    EXPECT_EQ(CountOp(graph, "gather"), 2);
    EXPECT_EQ(CountOp(graph, "scatter_add"), 2);
    EXPECT_EQ(CountOp(graph, "matmul"), 1);
  }
}

template <typename Factory>
void ExpectBuildFailsWithoutAddingNodes(Factory&& factory, const Shape& input_shape,
                                        DType input_dtype = DType::of<float>(),
                                        bool provide_input = true, bool provide_params = true) {
  Graph graph("invalid_gnn");
  std::vector<Value*> inputs;
  if (provide_input)
    inputs.push_back(graph.add_graph_input(MakeCpuType(input_shape.dims(), input_dtype)).value());
  const Module model = factory();
  std::vector<Value*> params;
  if (provide_params) params = add_parameter_inputs(graph, model.parameters()).value();
  const size_t before = graph.topological_order().size();
  const Result<std::vector<Value*>> result = model.build(graph, inputs, params);
  EXPECT_FALSE(result.is_ok());
  EXPECT_EQ(graph.topological_order().size(), before);
}

TEST(GnnConstructionNegative, GraphConvRejectsTopologyInputAndParameterErrorsAtomically) {
  ExpectBuildFailsWithoutAddingNodes(
      [] { return GraphConv("g", 4, 2, 2, {}, {}, DType::of<float>()); }, Shape({4, 2}));
  ExpectBuildFailsWithoutAddingNodes(
      [] { return GraphConv("g", 4, 2, 2, {0}, {1, 2}, DType::of<float>()); }, Shape({4, 2}));
  ExpectBuildFailsWithoutAddingNodes(
      [] { return GraphConv("g", 4, 2, 2, {-1}, {1}, DType::of<float>()); }, Shape({4, 2}));
  ExpectBuildFailsWithoutAddingNodes(
      [] { return GraphConv("g", 4, 2, 2, {0}, {4}, DType::of<float>()); }, Shape({4, 2}));
  ExpectBuildFailsWithoutAddingNodes(
      [] { return GraphConv("g", 4, 2, 2, {0}, {1}, DType::of<float>()); }, Shape({3, 2}));
  ExpectBuildFailsWithoutAddingNodes(
      [] { return GraphConv("g", 4, 2, 2, {0}, {1}, DType::of<float>()); }, Shape({4, 2}),
      DType::of<frame::float16_t>());
  ExpectBuildFailsWithoutAddingNodes(
      [] { return GraphConv("g", 4, 2, 2, {0}, {1}, DType::of<float>()); }, Shape({4, 2}),
      DType::of<float>(), false);
  ExpectBuildFailsWithoutAddingNodes(
      [] { return GraphConv("g", 4, 2, 2, {0}, {1}, DType::of<float>()); }, Shape({4, 2}),
      DType::of<float>(), true, false);
  ExpectBuildFailsWithoutAddingNodes(
      [] {
        return GraphConv("g", std::numeric_limits<int64_t>::max(), 2, 2, {0}, {0},
                         DType::of<float>());
      },
      Shape({1, 1}));
}

TEST(GnnConstructionNegative, HypergraphRejectsTopologyInputAndParameterErrorsAtomically) {
  ExpectBuildFailsWithoutAddingNodes(
      [] { return HypergraphConv("h", 4, 2, 2, 2, {0}, {0, 1}, DType::of<float>()); },
      Shape({4, 2}));
  ExpectBuildFailsWithoutAddingNodes(
      [] { return HypergraphConv("h", 4, 2, 2, 2, {4}, {0}, DType::of<float>()); }, Shape({4, 2}));
  ExpectBuildFailsWithoutAddingNodes(
      [] { return HypergraphConv("h", 4, 2, 2, 2, {0}, {2}, DType::of<float>()); }, Shape({4, 2}));
  ExpectBuildFailsWithoutAddingNodes(
      [] { return HypergraphConv("h", 4, 2, 2, 2, {0, 0}, {1, 1}, DType::of<float>()); },
      Shape({4, 2}));
  ExpectBuildFailsWithoutAddingNodes(
      [] { return HypergraphConv("h", 4, 2, 2, 2, {0}, {0}, DType::of<float>()); }, Shape({3, 2}));
  ExpectBuildFailsWithoutAddingNodes(
      [] { return HypergraphConv("h", 4, 2, 2, 2, {0}, {0}, DType::of<float>()); }, Shape({4, 2}),
      DType::of<float>(), true, false);
  ExpectBuildFailsWithoutAddingNodes(
      [] {
        return HypergraphConv("h", std::numeric_limits<int64_t>::max(), 2, 2, 2, {0}, {0},
                              DType::of<float>());
      },
      Shape({1, 1}));
}

TEST(GnnConstructionNegative, TopologyIndicesBeyondDoubleExactRangeFailBeforeMaterialization) {
  const int64_t lossy_index = frame::ops::kMaxDoubleExactInteger + 1;
  const int64_t containing_dimension = lossy_index + 1;
  {
    Graph graph("graph_index_precision");
    Value* input = graph.add_graph_input(MakeCpuType({containing_dimension, 1})).value();
    const Module model =
        GraphConv("g", containing_dimension, 1, 1, {lossy_index}, {0}, DType::of<float>());
    const Result<std::vector<Value*>> params = add_parameter_inputs(graph, model.parameters());
    ASSERT_TRUE(params.is_ok()) << params.status().message();
    const size_t before = graph.topological_order().size();
    const Result<std::vector<Value*>> result =
        model.build(graph, std::vector<Value*>{input}, params.value());
    ASSERT_FALSE(result.is_ok());
    EXPECT_NE(result.status().message().find("double-exact integer bound"), std::string_view::npos);
    EXPECT_EQ(graph.topological_order().size(), before);
  }
  {
    Graph graph("hypergraph_node_index_precision");
    Value* input = graph.add_graph_input(MakeCpuType({containing_dimension, 1})).value();
    const Module model =
        HypergraphConv("h", containing_dimension, 1, 1, 1, {lossy_index}, {0}, DType::of<float>());
    const Result<std::vector<Value*>> params = add_parameter_inputs(graph, model.parameters());
    ASSERT_TRUE(params.is_ok()) << params.status().message();
    const size_t before = graph.topological_order().size();
    const Result<std::vector<Value*>> result =
        model.build(graph, std::vector<Value*>{input}, params.value());
    ASSERT_FALSE(result.is_ok());
    EXPECT_NE(result.status().message().find("double-exact integer bound"), std::string_view::npos);
    EXPECT_EQ(graph.topological_order().size(), before);
  }
  {
    Graph graph("hypergraph_edge_index_precision");
    Value* input = graph.add_graph_input(MakeCpuType({1, 1})).value();
    const Module model =
        HypergraphConv("h", 1, containing_dimension, 1, 1, {0}, {lossy_index}, DType::of<float>());
    const Result<std::vector<Value*>> params = add_parameter_inputs(graph, model.parameters());
    ASSERT_TRUE(params.is_ok()) << params.status().message();
    const size_t before = graph.topological_order().size();
    const Result<std::vector<Value*>> result =
        model.build(graph, std::vector<Value*>{input}, params.value());
    ASSERT_FALSE(result.is_ok());
    EXPECT_NE(result.status().message().find("double-exact integer bound"), std::string_view::npos);
    EXPECT_EQ(graph.topological_order().size(), before);
  }
}

}  // namespace
