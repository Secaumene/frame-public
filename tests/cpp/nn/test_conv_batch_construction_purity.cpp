// M21 批3 T6 卷积批次工厂的构图纯度测试(ARCH-072,docs/architecture/
// nn-design.md §2/§8;工厂契约见 docs/plan/2026-07-18-batch3-m21-conv.md
// 第1.4节)。覆盖 Conv2d/Conv1d/MaxPool2d/AvgPool2d/Sigmoid/Flatten/AFF/Dwt2d/
// Dwt1d 九个工厂:build() 前后图节点计数增量恰等于预期新增 op 节点数、返回
// 的 Value* 仍属本图(手法同 tests/cpp/nn/test_construction_purity.cpp,不
// 重复该文件头注释已记录的判据来源说明)。文件末尾另含负例
// (channels<=0/kernel 非正/wavelet_kind 非法)——均应在 build() 期经 Result
// 报错,不静默产出错误图(ARCH-031 口径)。
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/nn/layers.h>
#include <frame/nn/module.h>

namespace {

using frame::cpu_device;
using frame::DType;
using frame::Result;
using frame::Shape;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::TensorType;
using frame::ir::Value;
using frame::nn::add_parameter_inputs;
using frame::nn::AFF;
using frame::nn::AvgPool2d;
using frame::nn::Conv1d;
using frame::nn::Conv2d;
using frame::nn::Dwt1d;
using frame::nn::Dwt2d;
using frame::nn::Flatten;
using frame::nn::MaxPool2d;
using frame::nn::Module;
using frame::nn::ParamSpec;
using frame::nn::Sigmoid;
using frame::nn::WaveletKind;

TensorType MakeCpuTensorType(std::vector<int64_t> dims) {
  TensorType type;
  type.dtype = DType::of<float>();
  type.shape = Shape(std::move(dims));
  type.device = cpu_device();
  return type;
}

// 核对 value 确属 graph(同 test_construction_purity.cpp::ExpectValueBelongsToGraph
// 先例,REUSE-002:结构相同,分属不同 TU 各自持一份匿名命名空间实现)。
void ExpectValueBelongsToGraph(Graph& graph, Value* value) {
  const std::vector<Node*>& topo = graph.topological_order();
  EXPECT_NE(std::find(topo.begin(), topo.end(), value->producer()), topo.end());
  const Result<Node*> probe = graph.create_node("relu", {value}, {value->type()});
  EXPECT_TRUE(probe.is_ok()) << (probe.is_ok() ? std::string() : probe.status().message());
}

TEST(ConvBatchConstructionPurity, Conv2dWithBiasAddsExactlyOneConvNode) {
  Graph graph("purity_conv2d_with_bias");
  Value* x = graph.add_graph_input(MakeCpuTensorType({2, 3, 8, 8})).value();

  const Module conv = Conv2d("conv", /*in_channels=*/3, /*out_channels=*/4, {3, 3}, {1, 1}, {1, 1},
                             /*groups=*/1, /*with_bias=*/true, DType::of<float>());
  const std::vector<ParamSpec> param_specs = conv.parameters();
  ASSERT_EQ(param_specs.size(), 2u);  // weight + bias

  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, param_specs);
  ASSERT_TRUE(params.is_ok()) << params.status().message();

  const size_t pre_build = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      conv.build(graph, std::vector<Value*>{x}, params.value());
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  EXPECT_EQ(graph.topological_order().size() - pre_build, 1u);
  ExpectValueBelongsToGraph(graph, outputs.value()[0]);
  EXPECT_EQ(outputs.value()[0]->producer()->op(), "conv2d");
  EXPECT_EQ(outputs.value()[0]->producer()->inputs().size(), 3u);  // x, weight, bias
}

TEST(ConvBatchConstructionPurity, Conv2dWithoutBiasAddsExactlyOneConvNode) {
  Graph graph("purity_conv2d_without_bias");
  Value* x = graph.add_graph_input(MakeCpuTensorType({2, 3, 8, 8})).value();

  const Module conv = Conv2d("conv", /*in_channels=*/3, /*out_channels=*/4, {3, 3}, {1, 1}, {1, 1},
                             /*groups=*/1, /*with_bias=*/false, DType::of<float>());
  const std::vector<ParamSpec> param_specs = conv.parameters();
  ASSERT_EQ(param_specs.size(), 1u);  // weight only

  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, param_specs);
  ASSERT_TRUE(params.is_ok()) << params.status().message();

  const size_t pre_build = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      conv.build(graph, std::vector<Value*>{x}, params.value());
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  EXPECT_EQ(graph.topological_order().size() - pre_build, 1u);
  EXPECT_EQ(outputs.value()[0]->producer()->op(), "conv2d");
  EXPECT_EQ(outputs.value()[0]->producer()->inputs().size(), 2u);  // x, weight
}

TEST(ConvBatchConstructionPurity, Conv1dWithBiasAddsExactlyOneConvNode) {
  Graph graph("purity_conv1d_with_bias");
  Value* x = graph.add_graph_input(MakeCpuTensorType({2, 3, 16})).value();

  const Module conv =
      Conv1d("conv", /*in_channels=*/3, /*out_channels=*/5, /*kernel=*/3,
             /*stride=*/1, /*padding=*/1, /*groups=*/1, /*with_bias=*/true, DType::of<float>());
  const std::vector<ParamSpec> param_specs = conv.parameters();
  ASSERT_EQ(param_specs.size(), 2u);

  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, param_specs);
  ASSERT_TRUE(params.is_ok()) << params.status().message();

  const size_t pre_build = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      conv.build(graph, std::vector<Value*>{x}, params.value());
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  EXPECT_EQ(graph.topological_order().size() - pre_build, 1u);
  EXPECT_EQ(outputs.value()[0]->producer()->op(), "conv1d");
}

TEST(ConvBatchConstructionPurity, MaxPool2dHasNoParametersAndAddsExactlyOneNode) {
  Graph graph("purity_max_pool2d");
  Value* x = graph.add_graph_input(MakeCpuTensorType({2, 3, 8, 8})).value();

  const Module pool = MaxPool2d("pool", {2, 2}, {2, 2}, {0, 0});
  EXPECT_EQ(pool.parameters().size(), 0u);

  const size_t pre_build = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      pool.build(graph, std::vector<Value*>{x}, std::vector<Value*>{});
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  EXPECT_EQ(graph.topological_order().size() - pre_build, 1u);
  EXPECT_EQ(outputs.value()[0]->producer()->op(), "max_pool2d");
}

TEST(ConvBatchConstructionPurity, AvgPool2dHasNoParametersAndAddsExactlyOneNode) {
  Graph graph("purity_avg_pool2d");
  Value* x = graph.add_graph_input(MakeCpuTensorType({2, 3, 8, 8})).value();

  const Module pool = AvgPool2d("pool", {2, 2}, {2, 2}, {0, 0});
  EXPECT_EQ(pool.parameters().size(), 0u);

  const size_t pre_build = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      pool.build(graph, std::vector<Value*>{x}, std::vector<Value*>{});
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  EXPECT_EQ(graph.topological_order().size() - pre_build, 1u);
  EXPECT_EQ(outputs.value()[0]->producer()->op(), "avg_pool2d");
}

TEST(ConvBatchConstructionPurity, SigmoidHasNoParametersAndAddsExactlyOneNode) {
  Graph graph("purity_sigmoid");
  Value* x = graph.add_graph_input(MakeCpuTensorType({4, 3})).value();

  const Module sigmoid = Sigmoid("act");
  EXPECT_EQ(sigmoid.parameters().size(), 0u);

  const size_t pre_build = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      sigmoid.build(graph, std::vector<Value*>{x}, std::vector<Value*>{});
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  EXPECT_EQ(graph.topological_order().size() - pre_build, 1u);
  EXPECT_EQ(outputs.value()[0]->producer()->op(), "sigmoid");
}

TEST(ConvBatchConstructionPurity, FlattenComputesNProdShapeAndAddsExactlyOneReshapeNode) {
  Graph graph("purity_flatten");
  Value* x = graph.add_graph_input(MakeCpuTensorType({2, 3, 4, 5})).value();

  const Module flatten = Flatten("flatten");
  EXPECT_EQ(flatten.parameters().size(), 0u);

  const size_t pre_build = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      flatten.build(graph, std::vector<Value*>{x}, std::vector<Value*>{});
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  EXPECT_EQ(graph.topological_order().size() - pre_build, 1u);
  Node* reshape_node = outputs.value()[0]->producer();
  EXPECT_EQ(reshape_node->op(), "reshape");
  // N=2, prod(3,4,5)=60(ARCH-072③:Flatten 形状推导判据)。
  EXPECT_EQ(reshape_node->output(0)->type().shape, Shape({2, 60}));
}

TEST(ConvBatchConstructionPurity, FlattenOnRankOneInputProducesTrailingUnitDim) {
  // rank=1 输入([N])无"其余维"可连乘,prod 恒为空积=1(target_shape=[N,1])——
  // 边界形态,同一份实现路径覆盖。
  Graph graph("purity_flatten_rank1");
  Value* x = graph.add_graph_input(MakeCpuTensorType({7})).value();
  const Result<std::vector<Value*>> outputs =
      Flatten("flatten").build(graph, std::vector<Value*>{x}, std::vector<Value*>{});
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  EXPECT_EQ(outputs.value()[0]->producer()->output(0)->type().shape, Shape({7, 1}));
}

TEST(ConvBatchConstructionPurity, AFFBuildsExpectedTwelveNodeGraphAndChildrenAreConv2d) {
  Graph graph("purity_aff");
  Value* x = graph.add_graph_input(MakeCpuTensorType({2, 4, 6, 6})).value();
  Value* y = graph.add_graph_input(MakeCpuTensorType({2, 4, 6, 6})).value();

  const Module aff = AFF("aff", /*channels=*/4, DType::of<float>());
  ASSERT_EQ(aff.children.size(), 2u);
  // children=[c1,c2] 均为 Conv2d(1x1 卷积:weight[4,4,1,1] + bias[4],
  // with_bias=true 的工程取舍,见 src/nn/layers.cpp::AFF 头注释)。
  const std::vector<ParamSpec> param_specs = aff.parameters();
  ASSERT_EQ(param_specs.size(), 4u);  // c1.weight, c1.bias, c2.weight, c2.bias
  EXPECT_EQ(param_specs[0].name, "aff.c1.weight");
  EXPECT_EQ(param_specs[1].name, "aff.c1.bias");
  EXPECT_EQ(param_specs[2].name, "aff.c2.weight");
  EXPECT_EQ(param_specs[3].name, "aff.c2.bias");

  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, param_specs);
  ASSERT_TRUE(params.is_ok()) << params.status().message();

  const size_t pre_build = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      aff.build(graph, std::vector<Value*>{x, y}, params.value());
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  ASSERT_EQ(outputs.value().size(), 1u);
  // 节点计数(src/nn/layers.cpp::AFF 头注释推导):add, conv2d(c1), relu,
  // conv2d(c2), sigmoid, constant(ones), constant(neg_one), mul, add, mul,
  // mul, add = 12。
  EXPECT_EQ(graph.topological_order().size() - pre_build, 12u);
  ExpectValueBelongsToGraph(graph, outputs.value()[0]);
  EXPECT_EQ(outputs.value()[0]->producer()->op(), "add");  // 末节点是 add(mx,my)
}

TEST(ConvBatchConstructionPurity, Dwt2dAddsConstantThenConv2dNode) {
  Graph graph("purity_dwt2d");
  Value* x = graph.add_graph_input(MakeCpuTensorType({2, 3, 8, 8})).value();

  const Module dwt = Dwt2d("dwt", /*channels=*/3, WaveletKind::kHaar);
  EXPECT_EQ(dwt.parameters().size(), 0u);  // 固定滤波器非 ParamSpec,不训练

  const size_t pre_build = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      dwt.build(graph, std::vector<Value*>{x}, std::vector<Value*>{});
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  EXPECT_EQ(graph.topological_order().size() - pre_build, 2u);  // constant + conv2d
  Node* conv_node = outputs.value()[0]->producer();
  EXPECT_EQ(conv_node->op(), "conv2d");
  ASSERT_EQ(conv_node->inputs().size(), 2u);
  EXPECT_EQ(conv_node->inputs()[1]->producer()->op(), "constant");
}

TEST(ConvBatchConstructionPurity, Dwt1dAddsConstantThenConv1dNode) {
  Graph graph("purity_dwt1d");
  Value* x = graph.add_graph_input(MakeCpuTensorType({2, 3, 16})).value();

  const Module dwt = Dwt1d("dwt", /*channels=*/3, WaveletKind::kDb4);
  EXPECT_EQ(dwt.parameters().size(), 0u);

  const size_t pre_build = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      dwt.build(graph, std::vector<Value*>{x}, std::vector<Value*>{});
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  EXPECT_EQ(graph.topological_order().size() - pre_build, 2u);  // constant + conv1d
  Node* conv_node = outputs.value()[0]->producer();
  EXPECT_EQ(conv_node->op(), "conv1d");
  ASSERT_EQ(conv_node->inputs().size(), 2u);
  EXPECT_EQ(conv_node->inputs()[1]->producer()->op(), "constant");
}

// ---------------------------------------------------------------------------
// 负例:channels<=0 / kernel 非正 / wavelet_kind 非法(计划 1.4 节交付项)——
// 均由对应算子 shape 推断在 build() 期经 Result 报错(不静默产出错误图)。
// ---------------------------------------------------------------------------

TEST(ConvBatchConstructionPurityNegative, Conv2dNonPositiveOutChannelsFailsAtBuild) {
  Graph graph("negative_conv2d_out_channels");
  Value* x = graph.add_graph_input(MakeCpuTensorType({1, 3, 8, 8})).value();
  const Module conv = Conv2d("conv", /*in_channels=*/3, /*out_channels=*/0, {3, 3}, {1, 1}, {1, 1},
                             /*groups=*/1, /*with_bias=*/false, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, conv.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  const Result<std::vector<Value*>> outputs =
      conv.build(graph, std::vector<Value*>{x}, params.value());
  EXPECT_FALSE(outputs.is_ok());
}

TEST(ConvBatchConstructionPurityNegative, Conv2dNonPositiveKernelFailsAtBuild) {
  Graph graph("negative_conv2d_kernel");
  Value* x = graph.add_graph_input(MakeCpuTensorType({1, 3, 8, 8})).value();
  const Module conv = Conv2d("conv", /*in_channels=*/3, /*out_channels=*/4, {0, 3}, {1, 1}, {1, 1},
                             /*groups=*/1, /*with_bias=*/false, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, conv.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  const Result<std::vector<Value*>> outputs =
      conv.build(graph, std::vector<Value*>{x}, params.value());
  EXPECT_FALSE(outputs.is_ok());
}

TEST(ConvBatchConstructionPurityNegative, MaxPool2dNonPositiveKernelFailsAtBuild) {
  Graph graph("negative_max_pool2d_kernel");
  Value* x = graph.add_graph_input(MakeCpuTensorType({1, 3, 8, 8})).value();
  const Result<std::vector<Value*>> outputs =
      MaxPool2d("pool", {0, 2}, {2, 2}, {0, 0})
          .build(graph, std::vector<Value*>{x}, std::vector<Value*>{});
  EXPECT_FALSE(outputs.is_ok());
}

TEST(ConvBatchConstructionPurityNegative, AFFNonPositiveChannelsFailsAtBuild) {
  Graph graph("negative_aff_channels");
  Value* x = graph.add_graph_input(MakeCpuTensorType({1, 0, 4, 4})).value();
  Value* y = graph.add_graph_input(MakeCpuTensorType({1, 0, 4, 4})).value();
  const Module aff = AFF("aff", /*channels=*/0, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, aff.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  const Result<std::vector<Value*>> outputs =
      aff.build(graph, std::vector<Value*>{x, y}, params.value());
  EXPECT_FALSE(outputs.is_ok());
}

TEST(ConvBatchConstructionPurityNegative, Dwt2dRejectsDb4InV1) {
  Graph graph("negative_dwt2d_db4");
  Value* x = graph.add_graph_input(MakeCpuTensorType({1, 2, 8, 8})).value();
  const Result<std::vector<Value*>> outputs =
      Dwt2d("dwt", /*channels=*/2, WaveletKind::kDb4)
          .build(graph, std::vector<Value*>{x}, std::vector<Value*>{});
  EXPECT_FALSE(outputs.is_ok());
}

TEST(ConvBatchConstructionPurityNegative, Dwt1dRejectsUnrecognizedWaveletKindValue) {
  Graph graph("negative_dwt1d_bad_kind");
  Value* x = graph.add_graph_input(MakeCpuTensorType({1, 2, 8})).value();
  const Result<std::vector<Value*>> outputs =
      Dwt1d("dwt", /*channels=*/2, static_cast<WaveletKind>(99))
          .build(graph, std::vector<Value*>{x}, std::vector<Value*>{});
  EXPECT_FALSE(outputs.is_ok());
}

}  // namespace
