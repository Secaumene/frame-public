// M23 批5 T5 频域批次工厂的 parameters() 路径命名 golden(ARCH-073)与
// Fno1dBlock 的 params 切片不变式(ARCH-071,组合模块子树数=2:spectral +
// bypass)。手法同 tests/cpp/nn/test_seq_batch_parameters_and_slicing.cpp
// 先例(不重复其头注释已记录的判据来源)。
#include <cstddef>
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
using frame::nn::Fno1dBlock;
using frame::nn::FourierFilter1d;
using frame::nn::Module;
using frame::nn::ParamSpec;
using frame::nn::SpectralConv1d;

TensorType MakeCpuTensorType(std::vector<int64_t> dims) {
  TensorType type;
  type.dtype = DType::of<float>();
  type.shape = Shape(std::move(dims));
  type.device = cpu_device();
  return type;
}

// 查找以 target 为其第 input_index 个输入、op 匹配的节点(ARCH-071 切片不
// 变式的定位 helper,REUSE-002:同
// test_seq_batch_parameters_and_slicing.cpp::FindNodeWithInput 先例)。
Node* FindNodeWithInput(Graph& graph, const std::string& op, size_t input_index, Value* target) {
  for (Node* node : graph.topological_order()) {
    if (node->op() != op) continue;
    if (node->inputs().size() <= input_index) continue;
    if (node->inputs()[input_index] == target) return node;
  }
  return nullptr;
}

// 计数以 target 为其第 input_index 个输入、op 匹配的节点数(SpectralConv1d 的
// 权重列 slice 每模态各一次,单个 FindNodeWithInput 不足以佐证"逐模态均消费
// 到同一份切片参数",故另设计数版本)。
int CountNodesWithInput(Graph& graph, const std::string& op, size_t input_index, Value* target) {
  int count = 0;
  for (Node* node : graph.topological_order()) {
    if (node->op() != op) continue;
    if (node->inputs().size() <= input_index) continue;
    if (node->inputs()[input_index] == target) ++count;
  }
  return count;
}

TEST(FftBatchParametersGolden, SpectralConv1dParameterNamesAndShapes) {
  const Module sc = SpectralConv1d("sc0", /*batch=*/2, /*in_channels=*/3, /*out_channels=*/4,
                                   /*n=*/8, /*modes=*/2, DType::of<float>());
  const std::vector<ParamSpec> params = sc.parameters();
  ASSERT_EQ(params.size(), 2u);
  EXPECT_EQ(params[0].name, "sc0.W_re");
  EXPECT_EQ(params[0].type.shape, Shape({3, 8}));  // [in_channels, modes*out_channels]=[3,2*4]
  EXPECT_EQ(params[1].name, "sc0.W_im");
  EXPECT_EQ(params[1].type.shape, Shape({3, 8}));
}

TEST(FftBatchParametersGolden, FourierFilter1dParameterNamesAndShapes) {
  const Module ff =
      FourierFilter1d("ff0", /*batch=*/2, /*channels=*/3, /*n=*/8, DType::of<float>());
  const std::vector<ParamSpec> params = ff.parameters();
  ASSERT_EQ(params.size(), 2u);
  EXPECT_EQ(params[0].name, "ff0.w_re");
  EXPECT_EQ(params[0].type.shape, Shape({2, 3, 5, 1}));  // [batch, channels, k=n/2+1, 1]
  EXPECT_EQ(params[1].name, "ff0.w_im");
  EXPECT_EQ(params[1].type.shape, Shape({2, 3, 5, 1}));
}

TEST(FftBatchParametersGolden, Fno1dBlockParameterNamesFollowChildPrefixPreorder) {
  const Module block = Fno1dBlock("fno0", /*batch=*/2, /*in_channels=*/3, /*out_channels=*/4,
                                  /*n=*/8, /*modes=*/2, DType::of<float>());
  const std::vector<ParamSpec> params = block.parameters();
  const std::vector<std::string> expected_names{"fno0.spectral.W_re", "fno0.spectral.W_im",
                                                "fno0.bypass.weight", "fno0.bypass.bias"};
  ASSERT_EQ(params.size(), expected_names.size());
  for (size_t i = 0; i < params.size(); ++i) {
    EXPECT_EQ(params[i].name, expected_names[i]);
  }
  EXPECT_EQ(params[0].type.shape, Shape({3, 8}));     // spectral W_re [in,modes*out]
  EXPECT_EQ(params[2].type.shape, Shape({4, 3, 1}));  // bypass weight [out,in,kernel=1]
  EXPECT_EQ(params[3].type.shape, Shape({4}));        // bypass bias [out]
}

// ARCH-071 params 切片不变式(Fno1dBlock 的 2 个子树:spectral + bypass):传入
// Fno1dBlock::build 的 params 恰为 fno.parameters() 的同序全集,按
// [spectral 子树参数…, bypass 子树参数…] 先序分段切片——本测试以真实端到端
// build 调用间接验证(白盒断言:spectral 分支消费到 W_re/W_im 切片、bypass
// 分支消费到 weight/bias 切片,佐证由各自算子的输入恰是该段对应的
// graph_input Value)。
TEST(FftBatchParamsSlicingInvariant, Fno1dBlockBuildConsumesSpectralAndBypassSubtreeSlicesInOrder) {
  constexpr int64_t kBatch = 2;
  constexpr int64_t kIn = 3;
  constexpr int64_t kOut = 4;
  constexpr int64_t kN = 8;
  constexpr int64_t kModes = 2;

  Graph graph("fno1d_block_params_slicing");
  Value* x = graph.add_graph_input(MakeCpuTensorType({kBatch, kIn, kN})).value();

  const Module block = Fno1dBlock("fno", kBatch, kIn, kOut, kN, kModes, DType::of<float>());
  const std::vector<ParamSpec> flat_params = block.parameters();
  ASSERT_EQ(flat_params.size(), 4u);

  const Result<std::vector<Value*>> param_inputs = add_parameter_inputs(graph, flat_params);
  ASSERT_TRUE(param_inputs.is_ok()) << param_inputs.status().message();
  const std::vector<Value*>& params = param_inputs.value();

  const Result<std::vector<Value*>> outputs = block.build(graph, std::vector<Value*>{x}, params);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  // spectral 段(0..2):W_re/W_im 各恰被 modes(2)次权重列 slice 节点消费
  // (每模态一次)。
  EXPECT_EQ(CountNodesWithInput(graph, "slice", 0, params[0]), kModes);
  EXPECT_EQ(CountNodesWithInput(graph, "slice", 0, params[1]), kModes);

  // rfft 节点(spectral 分支入口)恰以 x 为其唯一输入。
  Node* rfft_node = FindNodeWithInput(graph, "rfft", 0, x);
  ASSERT_NE(rfft_node, nullptr);

  // bypass 段(2..4):conv1d 节点恰以 x 为其第一输入、params[2]/params[3] 为
  // 其权重/偏置输入。
  Node* conv1d_node = FindNodeWithInput(graph, "conv1d", 0, x);
  ASSERT_NE(conv1d_node, nullptr);
  ASSERT_EQ(conv1d_node->inputs().size(), 3u);
  EXPECT_EQ(conv1d_node->inputs()[1], params[2]);
  EXPECT_EQ(conv1d_node->inputs()[2], params[3]);

  // 最终输出即 tanh(add(spectral(x), conv1x1(x))) 的末节点,其两个加数分别
  // 来自 irfft(spectral 分支)与 conv1d(bypass 分支)。
  ASSERT_EQ(outputs.value().size(), 1u);
  Node* tanh_node = outputs.value()[0]->producer();
  EXPECT_EQ(tanh_node->op(), "tanh");
  Node* add_node = tanh_node->inputs()[0]->producer();
  EXPECT_EQ(add_node->op(), "add");
  EXPECT_EQ(add_node->inputs()[0]->producer()->op(), "irfft");
  EXPECT_EQ(add_node->inputs()[1], conv1d_node->output(0));
}

}  // namespace
