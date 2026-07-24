// M22 批4 T5 序列批次工厂的 parameters() 路径命名 golden(ARCH-073)与
// MultiheadAttention/TransformerEncoderBlock 的 params 切片不变式(ARCH-071,
// 组合模块子树数分别为 4/5)。手法分别同
// tests/cpp/nn/test_conv_batch_parameters_and_slicing.cpp 先例(不重复其头
// 注释已记录的判据来源)。
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
using frame::nn::LayerNorm;
using frame::nn::LSTM;
using frame::nn::Module;
using frame::nn::MultiheadAttention;
using frame::nn::ParamSpec;
using frame::nn::TransformerEncoderBlock;

TensorType MakeCpuTensorType(std::vector<int64_t> dims) {
  TensorType type;
  type.dtype = DType::of<float>();
  type.shape = Shape(std::move(dims));
  type.device = cpu_device();
  return type;
}

// 查找以 target 为其第 input_index 个输入、op 匹配的节点(ARCH-071 切片不
// 变式的定位 helper:params 各元素是全局唯一 Value*,借此定位其消费节点,比
// 按拓扑序位置数节点更稳健——REUSE-002,本文件内部,不入公开 API)。
Node* FindNodeWithInput(Graph& graph, const std::string& op, size_t input_index, Value* target) {
  for (Node* node : graph.topological_order()) {
    if (node->op() != op) continue;
    if (node->inputs().size() <= input_index) continue;
    if (node->inputs()[input_index] == target) return node;
  }
  return nullptr;
}

TEST(SeqBatchParametersGolden, LayerNormParameterNamesAndShapes) {
  const Module ln = LayerNorm("ln0", /*dim=*/5, /*eps=*/1e-5, DType::of<float>());
  const std::vector<ParamSpec> params = ln.parameters();
  ASSERT_EQ(params.size(), 2u);
  EXPECT_EQ(params[0].name, "ln0.gamma");
  EXPECT_EQ(params[0].type.shape, Shape({5}));
  EXPECT_EQ(params[1].name, "ln0.beta");
  EXPECT_EQ(params[1].type.shape, Shape({5}));
}

TEST(SeqBatchParametersGolden, LSTMParameterNamesAndShapes) {
  const Module lstm = LSTM("lstm0", /*batch=*/6, /*num_steps=*/4, /*input_dim=*/3,
                           /*hidden_dim=*/5, DType::of<float>());
  const std::vector<ParamSpec> params = lstm.parameters();
  ASSERT_EQ(params.size(), 3u);
  EXPECT_EQ(params[0].name, "lstm0.W_ih");
  EXPECT_EQ(params[0].type.shape, Shape({3, 20}));  // [input_dim, 4H]
  EXPECT_EQ(params[1].name, "lstm0.W_hh");
  EXPECT_EQ(params[1].type.shape, Shape({5, 20}));  // [hidden_dim, 4H]
  EXPECT_EQ(params[2].name, "lstm0.bias");
  EXPECT_EQ(params[2].type.shape, Shape({6, 20}));  // [batch, 4H]
}

TEST(SeqBatchParametersGolden, MultiheadAttentionParameterNamesFollowChildPrefixPreorder) {
  const Module mha = MultiheadAttention("mha0", /*batch=*/2, /*seq_len=*/3, /*embed_dim=*/8,
                                        /*num_heads=*/4, /*with_bias=*/true, DType::of<float>());
  const std::vector<ParamSpec> params = mha.parameters();
  const std::vector<std::string> expected_names{"mha0.q.weight", "mha0.q.bias",   "mha0.k.weight",
                                                "mha0.k.bias",   "mha0.v.weight", "mha0.v.bias",
                                                "mha0.o.weight", "mha0.o.bias"};
  ASSERT_EQ(params.size(), expected_names.size());
  for (size_t i = 0; i < params.size(); ++i) {
    EXPECT_EQ(params[i].name, expected_names[i]);
  }
  // 4 个 Linear 均 embed_dim(8)->embed_dim(8),batch 形参=batch*seq_len=6。
  EXPECT_EQ(params[0].type.shape, Shape({8, 8}));
  EXPECT_EQ(params[1].type.shape, Shape({6, 8}));
}

TEST(SeqBatchParametersGolden, TransformerEncoderBlockParameterNamesFollowChildPrefixPreorder) {
  const Module block = TransformerEncoderBlock("block0", /*batch=*/1, /*seq_len=*/2,
                                               /*embed_dim=*/4, /*num_heads=*/2, /*ffn_dim=*/6,
                                               /*with_bias=*/true, DType::of<float>());
  const std::vector<ParamSpec> params = block.parameters();
  const std::vector<std::string> expected_names{
      "block0.mha.q.weight", "block0.mha.q.bias", "block0.mha.k.weight", "block0.mha.k.bias",
      "block0.mha.v.weight", "block0.mha.v.bias", "block0.mha.o.weight", "block0.mha.o.bias",
      "block0.ln1.gamma",    "block0.ln1.beta",   "block0.ffn1.weight",  "block0.ffn1.bias",
      "block0.ffn2.weight",  "block0.ffn2.bias",  "block0.ln2.gamma",    "block0.ln2.beta"};
  ASSERT_EQ(params.size(), expected_names.size());
  for (size_t i = 0; i < params.size(); ++i) {
    EXPECT_EQ(params[i].name, expected_names[i]);
  }
}

// ARCH-071 params 切片不变式(MultiheadAttention 的 4 个 Linear 子树):传入
// MultiheadAttention::build 的 params 恰为 mha.parameters() 的同序全集,按
// [q 子树…, k 子树…, v 子树…, o 子树…] 先序分段切片,每段长度=对应子
// parameters().size()——本测试以真实端到端 build 调用间接验证(白盒断言:
// 各 Linear 子模块消费到正确切片,由各自 matmul/add 节点的权重/偏置输入恰是
// 该段对应的 graph_input Value 佐证)。
TEST(SeqBatchParamsSlicingInvariant, MultiheadAttentionBuildConsumesQKVOSubtreeSlicesInOrder) {
  constexpr int64_t kBatch = 1;
  constexpr int64_t kSeqLen = 2;
  constexpr int64_t kEmbedDim = 4;
  constexpr int64_t kNumHeads = 2;

  Graph graph("mha_params_slicing");
  Value* x = graph.add_graph_input(MakeCpuTensorType({kBatch * kSeqLen, kEmbedDim})).value();

  const Module mha = MultiheadAttention("mha", kBatch, kSeqLen, kEmbedDim, kNumHeads,
                                        /*with_bias=*/true, DType::of<float>());
  const std::vector<ParamSpec> flat_params = mha.parameters();
  ASSERT_EQ(flat_params.size(), 8u);

  const Result<std::vector<Value*>> param_inputs = add_parameter_inputs(graph, flat_params);
  ASSERT_TRUE(param_inputs.is_ok()) << param_inputs.status().message();
  const std::vector<Value*>& params = param_inputs.value();

  const Result<std::vector<Value*>> outputs = mha.build(graph, std::vector<Value*>{x}, params);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  // q/k/v 三个 Linear 均以 x 为其唯一输入,各自 matmul 节点的第二输入(权重)
  // 恰是对应切片段首元素;matmul 节点的第一输入恰是 x(佐证子模块确实消费到
  // 原始输入,而非切片错位)。
  Node* q_matmul = FindNodeWithInput(graph, "matmul", 1, params[0]);
  ASSERT_NE(q_matmul, nullptr);
  EXPECT_EQ(q_matmul->inputs()[0], x);
  Node* q_add = FindNodeWithInput(graph, "add", 1, params[1]);
  ASSERT_NE(q_add, nullptr);
  EXPECT_EQ(q_add->inputs()[0], q_matmul->output(0));

  Node* k_matmul = FindNodeWithInput(graph, "matmul", 1, params[2]);
  ASSERT_NE(k_matmul, nullptr);
  EXPECT_EQ(k_matmul->inputs()[0], x);
  Node* k_add = FindNodeWithInput(graph, "add", 1, params[3]);
  ASSERT_NE(k_add, nullptr);

  Node* v_matmul = FindNodeWithInput(graph, "matmul", 1, params[4]);
  ASSERT_NE(v_matmul, nullptr);
  EXPECT_EQ(v_matmul->inputs()[0], x);
  Node* v_add = FindNodeWithInput(graph, "add", 1, params[5]);
  ASSERT_NE(v_add, nullptr);

  // o 投影不以 x 为输入(消费的是头/批拼接结果),但仍恰以 params[6]/params[7]
  // 为其权重/偏置输入,且其输出即整个 MultiheadAttention 的唯一输出。
  Node* o_matmul = FindNodeWithInput(graph, "matmul", 1, params[6]);
  ASSERT_NE(o_matmul, nullptr);
  EXPECT_NE(o_matmul->inputs()[0], x);
  Node* o_add = FindNodeWithInput(graph, "add", 1, params[7]);
  ASSERT_NE(o_add, nullptr);
  EXPECT_EQ(o_add->inputs()[0], o_matmul->output(0));
  ASSERT_EQ(outputs.value().size(), 1u);
  EXPECT_EQ(outputs.value()[0], o_add->output(0));
}

// ARCH-071 params 切片不变式(TransformerEncoderBlock 的 5 个子树:mha 4
// Linear + ln1 + ffn1 + ffn2 + ln2)。同上手法,聚焦跨子树边界(mha 段
// (0..8)/ln1 段(8..10)/ffn1 段(10..12)/ffn2 段(12..14)/ln2 段(14..16))
// 各自消费到正确切片。
TEST(SeqBatchParamsSlicingInvariant, TransformerEncoderBlockBuildConsumesFiveSubtreeSlicesInOrder) {
  constexpr int64_t kBatch = 1;
  constexpr int64_t kSeqLen = 2;
  constexpr int64_t kEmbedDim = 4;
  constexpr int64_t kNumHeads = 2;
  constexpr int64_t kFfnDim = 6;

  Graph graph("transformer_block_params_slicing");
  Value* x = graph.add_graph_input(MakeCpuTensorType({kBatch * kSeqLen, kEmbedDim})).value();

  const Module block = TransformerEncoderBlock("block", kBatch, kSeqLen, kEmbedDim, kNumHeads,
                                               kFfnDim, /*with_bias=*/true, DType::of<float>());
  const std::vector<ParamSpec> flat_params = block.parameters();
  ASSERT_EQ(flat_params.size(), 16u);

  const Result<std::vector<Value*>> param_inputs = add_parameter_inputs(graph, flat_params);
  ASSERT_TRUE(param_inputs.is_ok()) << param_inputs.status().message();
  const std::vector<Value*>& params = param_inputs.value();

  const Result<std::vector<Value*>> outputs = block.build(graph, std::vector<Value*>{x}, params);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  // mha 段(0..8):q 子模块以 x 为输入,权重恰是 params[0]。
  Node* q_matmul = FindNodeWithInput(graph, "matmul", 1, params[0]);
  ASSERT_NE(q_matmul, nullptr);
  EXPECT_EQ(q_matmul->inputs()[0], x);

  // ln1 段(8..10):layer_norm 节点的 gamma/beta 恰是 params[8]/params[9]。
  Node* ln1_node = FindNodeWithInput(graph, "layer_norm", 1, params[8]);
  ASSERT_NE(ln1_node, nullptr);
  EXPECT_EQ(ln1_node->inputs()[2], params[9]);

  // ffn1 段(10..12):matmul 节点权重恰是 params[10],输入为 ln1 的输出。
  Node* ffn1_matmul = FindNodeWithInput(graph, "matmul", 1, params[10]);
  ASSERT_NE(ffn1_matmul, nullptr);
  EXPECT_EQ(ffn1_matmul->inputs()[0], ln1_node->output(0));

  // ffn2 段(12..14):matmul 节点权重恰是 params[12]。
  Node* ffn2_matmul = FindNodeWithInput(graph, "matmul", 1, params[12]);
  ASSERT_NE(ffn2_matmul, nullptr);

  // ln2 段(14..16):layer_norm 节点的 gamma/beta 恰是 params[14]/params[15],
  // 且其输出即整个 TransformerEncoderBlock 的唯一输出(Post-LN 末节点)。
  Node* ln2_node = FindNodeWithInput(graph, "layer_norm", 1, params[14]);
  ASSERT_NE(ln2_node, nullptr);
  EXPECT_EQ(ln2_node->inputs()[2], params[15]);
  ASSERT_EQ(outputs.value().size(), 1u);
  EXPECT_EQ(outputs.value()[0], ln2_node->output(0));
}

}  // namespace
