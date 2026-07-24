// M28 GraphConv/HypergraphConv 固定小图收敛与 IR golden。两网络均只有
// linear.weight 一个参数，前向真实经过 gather/scatter_add。
#include <cmath>
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

using frame::DType;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::Tensor;
using frame::hal::BackendRegistry;
using frame::ir::Graph;
using frame::ir::Value;
using frame::nn::add_parameter_inputs;
using frame::nn::GraphConv;
using frame::nn::HypergraphConv;
using frame::nn::Module;
using frame::nn::MseLoss;
using frame::nn::ParamSpec;
using frame::nn::testing::AssertGraphMatchesGolden;
using frame::nn::testing::MakeCpuTensorType;
using frame::nn::testing::MakeTensorFromFloats;
using frame::nn::testing::MakeUniformParamTensors;
using frame::nn::testing::ParamTypesOf;
using frame::nn::testing::RunFullBatchSgdTraining;

constexpr int64_t kNodes = 4;
constexpr int64_t kIn = 2;
constexpr int64_t kOut = 1;

struct GnnBundle {
  Graph graph;
  std::vector<ParamSpec> params;
};

Result<GnnBundle> BuildGraph(bool hypergraph) {
  Graph graph(hypergraph ? "hypergraph_conv_forward" : "graph_conv_forward");
  const Result<Value*> x = graph.add_graph_input(MakeCpuTensorType({kNodes, kIn}));
  if (!x.is_ok()) return x.status();
  const Module model = hypergraph ? HypergraphConv("h", kNodes, 3, kIn, kOut, {0, 1, 1, 2, 2, 3},
                                                   {0, 0, 1, 1, 2, 2}, DType::of<float>())
                                  : GraphConv("g", kNodes, kIn, kOut, {0, 1, 2, 3}, {1, 2, 3, 0},
                                              DType::of<float>());
  const std::vector<ParamSpec> params = model.parameters();
  const Result<std::vector<Value*>> param_values = add_parameter_inputs(graph, params);
  if (!param_values.is_ok()) return param_values.status();
  const Result<std::vector<Value*>> output =
      model.build(graph, std::vector<Value*>{x.value()}, param_values.value());
  if (!output.is_ok()) return output.status();
  const Result<Value*> target = graph.add_graph_input(MakeCpuTensorType({kNodes, kOut}));
  if (!target.is_ok()) return target.status();
  const Result<std::vector<Value*>> loss = MseLoss("loss").build(
      graph, std::vector<Value*>{output.value()[0], target.value()}, std::vector<Value*>{});
  if (!loss.is_ok()) return loss.status();
  const Status marked = graph.mark_output(loss.value()[0]);
  if (!marked.is_ok()) return marked;
  return GnnBundle{std::move(graph), params};
}

// x/weight 是教师函数的固定输入契约,二者同型但语义不可互换。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::vector<float> HypergraphTeacher(const std::vector<float>& x,
                                     const std::vector<float>& weight) {
  const std::vector<int64_t> nodes{0, 1, 1, 2, 2, 3};
  const std::vector<int64_t> edges{0, 0, 1, 1, 2, 2};
  const std::vector<double> node_degree{1, 2, 2, 1};
  std::vector<double> edge_values(3 * kIn, 0.0);
  for (size_t i = 0; i < nodes.size(); ++i) {
    for (int64_t f = 0; f < kIn; ++f) {
      edge_values[static_cast<size_t>(edges[i] * kIn + f)] +=
          static_cast<double>(x[static_cast<size_t>(nodes[i] * kIn + f)]) /
          std::sqrt(node_degree[static_cast<size_t>(nodes[i])]) / 2.0;
    }
  }
  std::vector<double> propagated(kNodes * kIn, 0.0);
  for (size_t i = 0; i < nodes.size(); ++i) {
    for (int64_t f = 0; f < kIn; ++f) {
      propagated[static_cast<size_t>(nodes[i] * kIn + f)] +=
          edge_values[static_cast<size_t>(edges[i] * kIn + f)];
    }
  }
  std::vector<float> target(kNodes, 0.0F);
  for (int64_t node = 0; node < kNodes; ++node) {
    double value = 0.0;
    for (int64_t f = 0; f < kIn; ++f) {
      value += propagated[static_cast<size_t>(node * kIn + f)] /
               std::sqrt(node_degree[static_cast<size_t>(node)]) *
               static_cast<double>(weight[static_cast<size_t>(f)]);
    }
    target[static_cast<size_t>(node)] = static_cast<float>(value);
  }
  return target;
}

class GnnConvergenceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const Result<frame::hal::Backend*> backend =
        BackendRegistry::instance().get(frame::kCpuBackendName);
    ASSERT_TRUE(backend.is_ok()) << backend.status().message();
    allocator_ = backend.value()->allocator(frame::cpu_device());
    ASSERT_NE(allocator_, nullptr);
  }

  void ExpectConverges(bool hypergraph, uint32_t seed) {
    Result<GnnBundle> bundle_result = BuildGraph(hypergraph);
    ASSERT_TRUE(bundle_result.is_ok()) << bundle_result.status().message();
    GnnBundle bundle = std::move(bundle_result.value());
    ASSERT_EQ(bundle.params.size(), 1U);
    const std::vector<float> x_values{1, 0, 0, 1, 1, 1, -1, 0.5F};
    const std::vector<float> teacher{0.4F, -0.2F};
    const std::vector<float> target_values = hypergraph
                                                 ? HypergraphTeacher(x_values, teacher)
                                                 : std::vector<float>{-0.5F, 0.4F, -0.2F, 0.2F};
    const Tensor x =
        MakeTensorFromFloats(x_values, Shape({kNodes, kIn}), frame::cpu_device(), *allocator_);
    const Tensor target = MakeTensorFromFloats(target_values, Shape({kNodes, kOut}),
                                               frame::cpu_device(), *allocator_);
    // NOLINTNEXTLINE(bugprone-random-generator-seed)
    std::mt19937 rng(seed);
    std::vector<Tensor> params =
        MakeUniformParamTensors(bundle.params, rng, -0.2F, 0.2F, frame::cpu_device(), *allocator_);
    // 超图归一化算子的谱半径更小，同一学习率下收敛较慢；固定 1000 步仍是
    // 小图毫秒级冒烟，并保持与 GNN 相同的严格下降判据。
    const int training_steps = hypergraph ? 1000 : 200;
    const Result<std::vector<double>> losses = RunFullBatchSgdTraining(
        bundle.graph, x, params, ParamTypesOf(bundle.params), target, 0.1, training_steps);
    ASSERT_TRUE(losses.is_ok()) << losses.status().message();
    EXPECT_LT(losses.value().back(), losses.value().front() * 1e-4);
    EXPECT_LT(losses.value().back(), 1e-6);
  }

  frame::hal::Allocator* allocator_ = nullptr;
};

TEST_F(GnnConvergenceTest, GraphConvFixedGraphConverges) { ExpectConverges(false, 20260801U); }

TEST_F(GnnConvergenceTest, HypergraphConvFixedGraphConverges) { ExpectConverges(true, 20260802U); }

TEST(GnnConvergenceGolden, GraphConvForwardMatchesGolden) {
  Result<GnnBundle> bundle = BuildGraph(false);
  ASSERT_TRUE(bundle.is_ok()) << bundle.status().message();
  EXPECT_TRUE(AssertGraphMatchesGolden(bundle.value().graph,
                                       "tests/cpp/nn/testdata/graph_conv_forward_expected.txt"));
}

TEST(GnnConvergenceGolden, HypergraphConvForwardMatchesGolden) {
  Result<GnnBundle> bundle = BuildGraph(true);
  ASSERT_TRUE(bundle.is_ok()) << bundle.status().message();
  EXPECT_TRUE(AssertGraphMatchesGolden(
      bundle.value().graph, "tests/cpp/nn/testdata/hypergraph_conv_forward_expected.txt"));
}

}  // namespace
