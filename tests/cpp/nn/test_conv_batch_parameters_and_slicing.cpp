// M21 批3 T6 卷积批次工厂的 parameters() 路径命名 golden(ARCH-073)与 AFF 的
// params 切片不变式(ARCH-071,两个 Conv2d 子树)。手法分别同
// tests/cpp/nn/test_parameters_golden.cpp、
// tests/cpp/nn/test_params_slicing_invariant.cpp 先例(不重复其头注释已记录
// 的判据来源)。
#include <cstddef>
#include <gtest/gtest.h>
#include <span>
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
using frame::ir::TensorType;
using frame::ir::Value;
using frame::nn::add_parameter_inputs;
using frame::nn::AFF;
using frame::nn::Conv1d;
using frame::nn::Conv2d;
using frame::nn::Module;
using frame::nn::ParamSpec;

TensorType MakeCpuTensorType(std::vector<int64_t> dims) {
  TensorType type;
  type.dtype = DType::of<float>();
  type.shape = Shape(std::move(dims));
  type.device = cpu_device();
  return type;
}

TEST(ConvBatchParametersGolden, Conv2dWithBiasParameterNamesAndShapes) {
  const Module conv = Conv2d("conv0", /*in_channels=*/6, /*out_channels=*/4, {3, 3}, {1, 1}, {1, 1},
                             /*groups=*/2, /*with_bias=*/true, DType::of<float>());
  const std::vector<ParamSpec> params = conv.parameters();
  ASSERT_EQ(params.size(), 2u);
  EXPECT_EQ(params[0].name, "conv0.weight");
  // Cin/groups = 6/2 = 3(计划 1.4 节 ParamSpec 口径:weight[Cout,Cin/g,KH,KW])。
  EXPECT_EQ(params[0].type.shape, Shape({4, 3, 3, 3}));
  EXPECT_EQ(params[1].name, "conv0.bias");
  EXPECT_EQ(params[1].type.shape, Shape({4}));  // 裁决点①:bias=[Cout],非全形
}

TEST(ConvBatchParametersGolden, Conv1dWithoutBiasParameterNames) {
  const Module conv =
      Conv1d("conv1", /*in_channels=*/4, /*out_channels=*/8, /*kernel=*/5,
             /*stride=*/2, /*padding=*/1, /*groups=*/1, /*with_bias=*/false, DType::of<float>());
  const std::vector<ParamSpec> params = conv.parameters();
  ASSERT_EQ(params.size(), 1u);
  EXPECT_EQ(params[0].name, "conv1.weight");
  EXPECT_EQ(params[0].type.shape, Shape({8, 4, 5}));
}

TEST(ConvBatchParametersGolden, AFFParameterNamesFollowChildPrefixPreorder) {
  const Module aff = AFF("aff0", /*channels=*/3, DType::of<float>());
  const std::vector<ParamSpec> params = aff.parameters();
  // 先序遍历:AFF 自身无直接参数 -> c1 子树(weight,bias)-> c2 子树
  // (weight,bias),ARCH-073。
  const std::vector<std::string> expected_names{"aff0.c1.weight", "aff0.c1.bias", "aff0.c2.weight",
                                                "aff0.c2.bias"};
  ASSERT_EQ(params.size(), expected_names.size());
  for (size_t i = 0; i < params.size(); ++i) {
    EXPECT_EQ(params[i].name, expected_names[i]);
  }
  // c1/c2 均为 1x1 卷积:weight[3,3,1,1]、bias[3]。
  EXPECT_EQ(params[0].type.shape, Shape({3, 3, 1, 1}));
  EXPECT_EQ(params[1].type.shape, Shape({3}));
}

// ARCH-071 params 切片不变式(AFF 组合模块的两个 Conv2d 子树):传入 AFF::build
// 的 params 恰为 aff.parameters() 的同序全集,按 [c1 子树…, c2 子树…] 先序
// 分段切片,每段长度 = 对应子 parameters().size()——本测试以真实端到端 build
// 调用间接验证(白盒断言:c1/c2 各自消费到正确长度的切片,由 build() 成功
// 且各自的 conv2d 节点权重输入恰是该段对应的 graph_input Value 佐证)。
TEST(AFFParamsSlicingInvariant, BuildConsumesC1ThenC2SubtreeSlicesInOrder) {
  Graph graph("aff_params_slicing");
  Value* x = graph.add_graph_input(MakeCpuTensorType({1, 2, 4, 4})).value();
  Value* y = graph.add_graph_input(MakeCpuTensorType({1, 2, 4, 4})).value();

  const Module aff = AFF("aff", /*channels=*/2, DType::of<float>());
  const std::vector<ParamSpec> flat_params = aff.parameters();
  ASSERT_EQ(flat_params.size(), 4u);

  const Result<std::vector<Value*>> param_inputs = add_parameter_inputs(graph, flat_params);
  ASSERT_TRUE(param_inputs.is_ok()) << param_inputs.status().message();
  const std::vector<Value*>& params = param_inputs.value();

  const Result<std::vector<Value*>> outputs = aff.build(graph, std::vector<Value*>{x, y}, params);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  // add(X,Y) -> c1 的 conv2d 节点应恰以 params[0]/params[1](c1.weight/bias)
  // 为其权重/偏置输入;relu -> c2 的 conv2d 节点应恰以 params[2]/params[3]
  // (c2.weight/bias)为其权重/偏置输入——直接证明切片边界落在正确位置
  // (0..2)/(2..4),而非错位或越界(ARCH-071 判定核心)。
  const std::vector<frame::ir::Node*>& topo = graph.topological_order();
  frame::ir::Node* c1_conv = nullptr;
  frame::ir::Node* c2_conv = nullptr;
  int conv_seen = 0;
  for (frame::ir::Node* node : topo) {
    if (node->op() == "conv2d") {
      if (conv_seen == 0) {
        c1_conv = node;
      } else if (conv_seen == 1) {
        c2_conv = node;
      }
      ++conv_seen;
    }
  }
  ASSERT_EQ(conv_seen, 2);
  ASSERT_NE(c1_conv, nullptr);
  ASSERT_NE(c2_conv, nullptr);
  ASSERT_EQ(c1_conv->inputs().size(), 3u);
  EXPECT_EQ(c1_conv->inputs()[1], params[0]);  // c1.weight
  EXPECT_EQ(c1_conv->inputs()[2], params[1]);  // c1.bias
  ASSERT_EQ(c2_conv->inputs().size(), 3u);
  EXPECT_EQ(c2_conv->inputs()[1], params[2]);  // c2.weight
  EXPECT_EQ(c2_conv->inputs()[2], params[3]);  // c2.bias
}

}  // namespace
