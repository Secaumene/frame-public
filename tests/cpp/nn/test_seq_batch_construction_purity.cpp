// M22 批4 T5 序列批次工厂的构图纯度测试(ARCH-072,docs/architecture/
// nn-design.md §2/§8;工厂契约见 docs/plan/2026-07-19-batch4-m22-seq.md
// §1.7)。覆盖 LayerNorm/LSTM/MultiheadAttention/TransformerEncoderBlock 四
// 工厂:build() 前后图节点计数增量恰等于预期新增 op 节点数、关键算子存在、
// 输出 shape/dtype 正确(手法同 tests/cpp/nn/test_conv_batch_construction_purity.cpp,
// 不重复该文件头注释已记录的判据来源说明)。文件末尾另含负例(eps<=0、
// embed_dim 不整除 num_heads、输入数错)——均应在 build() 期经 Result 报错,
// 不静默产出错误图(ARCH-031 口径)。
#include <algorithm>
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

// 计数指定 op 在图中出现的节点数(关键算子存在性断言,REUSE-002:同
// test_conv_batch_construction_purity.cpp::ExpectValueBelongsToGraph 先例,
// 各文件按需持一份轻量私有 helper)。
int CountNodesWithOp(Graph& graph, const std::string& op) {
  int count = 0;
  for (Node* node : graph.topological_order()) {
    if (node->op() == op) ++count;
  }
  return count;
}

// ---------------------------------------------------------------------------
// LayerNorm(name, dim, eps, dtype):单 layer_norm 节点。
// ---------------------------------------------------------------------------

TEST(SeqBatchConstructionPurity, LayerNormAddsExactlyOneNode) {
  Graph graph("purity_layer_norm");
  Value* x = graph.add_graph_input(MakeCpuTensorType({3, 4})).value();

  const Module ln = LayerNorm("ln", /*dim=*/4, /*eps=*/1e-5, DType::of<float>());
  const std::vector<ParamSpec> param_specs = ln.parameters();
  ASSERT_EQ(param_specs.size(), 2u);  // gamma, beta
  EXPECT_EQ(param_specs[0].name, "ln.gamma");
  EXPECT_EQ(param_specs[0].type.shape, Shape({4}));
  EXPECT_EQ(param_specs[1].name, "ln.beta");
  EXPECT_EQ(param_specs[1].type.shape, Shape({4}));

  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, param_specs);
  ASSERT_TRUE(params.is_ok()) << params.status().message();

  const size_t pre_build = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      ln.build(graph, std::vector<Value*>{x}, params.value());
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  EXPECT_EQ(graph.topological_order().size() - pre_build, 1u);
  ASSERT_EQ(outputs.value().size(), 1u);
  Node* ln_node = outputs.value()[0]->producer();
  EXPECT_EQ(ln_node->op(), "layer_norm");
  EXPECT_EQ(ln_node->inputs().size(), 3u);  // x, gamma, beta
  EXPECT_EQ(ln_node->output(0)->type().shape, Shape({3, 4}));
  EXPECT_EQ(ln_node->output(0)->type().dtype, DType::of<float>());
}

// ---------------------------------------------------------------------------
// LSTM(name, batch, num_steps, input_dim, hidden_dim, dtype):静态展开
// num_steps 步,每步 19 节点(§1.7 展开序列:slice+reshape+2 matmul+2 add+4
// slice+4 激活+2 mul+1 add+1 tanh+1 mul)+ h0/c0 两个 constant splat。
// ---------------------------------------------------------------------------

TEST(SeqBatchConstructionPurity, LSTMAddsExpectedNodeCountAndFinalOpIsMul) {
  constexpr int64_t kBatch = 2;
  constexpr int64_t kNumSteps = 3;
  constexpr int64_t kInputDim = 2;
  constexpr int64_t kHiddenDim = 3;

  Graph graph("purity_lstm");
  Value* x = graph.add_graph_input(MakeCpuTensorType({kBatch, kNumSteps, kInputDim})).value();

  const Module lstm = LSTM("lstm", kBatch, kNumSteps, kInputDim, kHiddenDim, DType::of<float>());
  const std::vector<ParamSpec> param_specs = lstm.parameters();
  ASSERT_EQ(param_specs.size(), 3u);  // W_ih, W_hh, bias
  EXPECT_EQ(param_specs[0].name, "lstm.W_ih");
  EXPECT_EQ(param_specs[0].type.shape, Shape({kInputDim, 4 * kHiddenDim}));
  EXPECT_EQ(param_specs[1].name, "lstm.W_hh");
  EXPECT_EQ(param_specs[1].type.shape, Shape({kHiddenDim, 4 * kHiddenDim}));
  EXPECT_EQ(param_specs[2].name, "lstm.bias");
  EXPECT_EQ(param_specs[2].type.shape, Shape({kBatch, 4 * kHiddenDim}));

  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, param_specs);
  ASSERT_TRUE(params.is_ok()) << params.status().message();

  const size_t pre_build = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      lstm.build(graph, std::vector<Value*>{x}, params.value());
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  // 2(h0,c0)+19*num_steps=2+57=59(实测校准值,构图期机械可复算)。
  EXPECT_EQ(graph.topological_order().size() - pre_build, 59u);
  ASSERT_EQ(outputs.value().size(), 1u);
  Node* h_node = outputs.value()[0]->producer();
  EXPECT_EQ(h_node->op(), "mul");  // h_T = o⊙tanh(c) 的末节点
  EXPECT_EQ(h_node->output(0)->type().shape, Shape({kBatch, kHiddenDim}));
  EXPECT_EQ(h_node->output(0)->type().dtype, DType::of<float>());
  EXPECT_EQ(CountNodesWithOp(graph, "sigmoid"), 3 * kNumSteps);  // i,f,o 三门 * num_steps
  EXPECT_EQ(CountNodesWithOp(graph, "tanh"), 2 * kNumSteps);     // g 门 + tanh(c) * num_steps
}

// ---------------------------------------------------------------------------
// MultiheadAttention(name, batch, seq_len, embed_dim, num_heads, with_bias,
// dtype):children=4 个 Linear(q/k/v/o);per-(b,h) 静态展开。
// ---------------------------------------------------------------------------

TEST(SeqBatchConstructionPurity, MultiheadAttentionAddsExpectedNodeCountAndKeyOps) {
  constexpr int64_t kBatch = 1;
  constexpr int64_t kSeqLen = 2;
  constexpr int64_t kEmbedDim = 4;
  constexpr int64_t kNumHeads = 2;

  Graph graph("purity_mha");
  Value* x = graph.add_graph_input(MakeCpuTensorType({kBatch * kSeqLen, kEmbedDim})).value();

  const Module mha = MultiheadAttention("mha", kBatch, kSeqLen, kEmbedDim, kNumHeads,
                                        /*with_bias=*/true, DType::of<float>());
  ASSERT_EQ(mha.children.size(), 4u);  // q, k, v, o
  const std::vector<ParamSpec> param_specs = mha.parameters();
  ASSERT_EQ(param_specs.size(), 8u);  // 4 个 Linear(with_bias) 各 weight+bias
  EXPECT_EQ(param_specs[0].name, "mha.q.weight");
  EXPECT_EQ(param_specs[1].name, "mha.q.bias");
  EXPECT_EQ(param_specs[2].name, "mha.k.weight");
  EXPECT_EQ(param_specs[4].name, "mha.v.weight");
  EXPECT_EQ(param_specs[6].name, "mha.o.weight");
  EXPECT_EQ(param_specs[7].name, "mha.o.bias");

  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, param_specs);
  ASSERT_TRUE(params.is_ok()) << params.status().message();

  const size_t pre_build = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      mha.build(graph, std::vector<Value*>{x}, params.value());
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  // 实测校准值(见 implementer probe 脚本推导,构图期机械可复算):31。
  EXPECT_EQ(graph.topological_order().size() - pre_build, 31u);
  ASSERT_EQ(outputs.value().size(), 1u);
  Node* out_node = outputs.value()[0]->producer();
  EXPECT_EQ(out_node->op(), "add");  // o 投影(with_bias=true)的末节点
  EXPECT_EQ(out_node->output(0)->type().shape, Shape({kBatch * kSeqLen, kEmbedDim}));
  // softmax 节点数 == batch*num_heads(每个 (b,h) 恰 1 次 softmax)。
  EXPECT_EQ(CountNodesWithOp(graph, "softmax"), kBatch * kNumHeads);
  // concat 节点数 == batch(头拼接)+ 1(批拼接)。
  EXPECT_EQ(CountNodesWithOp(graph, "concat"), kBatch + 1);
}

TEST(SeqBatchConstructionPurity, MultiheadAttentionWithoutBiasHasFourWeightOnlyParams) {
  Graph graph("purity_mha_no_bias");
  Value* x = graph.add_graph_input(MakeCpuTensorType({4, 4})).value();
  const Module mha =
      MultiheadAttention("mha", /*batch=*/2, /*seq_len=*/2, /*embed_dim=*/4, /*num_heads=*/2,
                         /*with_bias=*/false, DType::of<float>());
  const std::vector<ParamSpec> param_specs = mha.parameters();
  ASSERT_EQ(param_specs.size(), 4u);  // q.weight, k.weight, v.weight, o.weight
  EXPECT_EQ(param_specs[0].name, "mha.q.weight");
  EXPECT_EQ(param_specs[1].name, "mha.k.weight");
  EXPECT_EQ(param_specs[2].name, "mha.v.weight");
  EXPECT_EQ(param_specs[3].name, "mha.o.weight");

  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, param_specs);
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  const Result<std::vector<Value*>> outputs =
      mha.build(graph, std::vector<Value*>{x}, params.value());
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  EXPECT_EQ(outputs.value()[0]->producer()->op(), "matmul");  // o 投影(无 bias)的末节点
}

// ---------------------------------------------------------------------------
// Transformer 编码器块:TransformerEncoderBlock(name, batch, seq_len,
// embed_dim, num_heads, ffn_dim, with_bias, dtype):children=[mha, ln1, ffn1,
// ffn2, ln2]。
// ---------------------------------------------------------------------------

TEST(SeqBatchConstructionPurity, TransformerEncoderBlockChildrenParamsAndOutputShape) {
  constexpr int64_t kBatch = 1;
  constexpr int64_t kSeqLen = 2;
  constexpr int64_t kEmbedDim = 4;
  constexpr int64_t kNumHeads = 2;
  constexpr int64_t kFfnDim = 6;

  Graph graph("purity_transformer_block");
  Value* x = graph.add_graph_input(MakeCpuTensorType({kBatch * kSeqLen, kEmbedDim})).value();

  const Module block = TransformerEncoderBlock("block", kBatch, kSeqLen, kEmbedDim, kNumHeads,
                                               kFfnDim, /*with_bias=*/true, DType::of<float>());
  ASSERT_EQ(block.children.size(), 5u);  // mha, ln1, ffn1, ffn2, ln2
  const std::vector<ParamSpec> param_specs = block.parameters();
  // mha(4*2=8)+ln1(2)+ffn1(2)+ffn2(2)+ln2(2)=16。
  ASSERT_EQ(param_specs.size(), 16u);
  EXPECT_EQ(param_specs[0].name, "block.mha.q.weight");
  EXPECT_EQ(param_specs[8].name, "block.ln1.gamma");
  EXPECT_EQ(param_specs[9].name, "block.ln1.beta");
  EXPECT_EQ(param_specs[10].name, "block.ffn1.weight");
  EXPECT_EQ(param_specs[10].type.shape, Shape({kEmbedDim, kFfnDim}));
  EXPECT_EQ(param_specs[12].name, "block.ffn2.weight");
  EXPECT_EQ(param_specs[12].type.shape, Shape({kFfnDim, kEmbedDim}));
  EXPECT_EQ(param_specs[14].name, "block.ln2.gamma");
  EXPECT_EQ(param_specs[15].name, "block.ln2.beta");

  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, param_specs);
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  const Result<std::vector<Value*>> outputs =
      block.build(graph, std::vector<Value*>{x}, params.value());
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  ASSERT_EQ(outputs.value().size(), 1u);
  Node* out_node = outputs.value()[0]->producer();
  EXPECT_EQ(out_node->op(), "layer_norm");  // ln2 的末节点(Post-LN)
  EXPECT_EQ(out_node->output(0)->type().shape, Shape({kBatch * kSeqLen, kEmbedDim}));
  EXPECT_EQ(CountNodesWithOp(graph, "layer_norm"), 2);  // ln1 + ln2
  EXPECT_EQ(CountNodesWithOp(graph, "relu"), 1);        // FFN 内联 relu(非独立 child)
}

// ---------------------------------------------------------------------------
// 负例:eps<=0 / embed_dim 不整除 num_heads / 输入数错——均由 build() 期经
// Result 报错(不静默产出错误图,ARCH-031 口径)。
// ---------------------------------------------------------------------------

TEST(SeqBatchConstructionPurityNegative, LayerNormZeroEpsFailsAtBuild) {
  Graph graph("negative_layer_norm_zero_eps");
  Value* x = graph.add_graph_input(MakeCpuTensorType({2, 3})).value();
  const Module ln = LayerNorm("ln", /*dim=*/3, /*eps=*/0.0, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, ln.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  const Result<std::vector<Value*>> outputs =
      ln.build(graph, std::vector<Value*>{x}, params.value());
  EXPECT_FALSE(outputs.is_ok());
}

TEST(SeqBatchConstructionPurityNegative, LayerNormNegativeEpsFailsAtBuild) {
  Graph graph("negative_layer_norm_negative_eps");
  Value* x = graph.add_graph_input(MakeCpuTensorType({2, 3})).value();
  const Module ln = LayerNorm("ln", /*dim=*/3, /*eps=*/-1e-5, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, ln.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  const Result<std::vector<Value*>> outputs =
      ln.build(graph, std::vector<Value*>{x}, params.value());
  EXPECT_FALSE(outputs.is_ok());
}

TEST(SeqBatchConstructionPurityNegative, LayerNormWrongInputCountFailsAtBuild) {
  Graph graph("negative_layer_norm_wrong_input_count");
  Value* x0 = graph.add_graph_input(MakeCpuTensorType({2, 3})).value();
  Value* x1 = graph.add_graph_input(MakeCpuTensorType({2, 3})).value();
  const Module ln = LayerNorm("ln", /*dim=*/3, /*eps=*/1e-5, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, ln.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  const Result<std::vector<Value*>> outputs =
      ln.build(graph, std::vector<Value*>{x0, x1}, params.value());
  EXPECT_FALSE(outputs.is_ok());
}

TEST(SeqBatchConstructionPurityNegative, LSTMWrongInputCountFailsAtBuild) {
  Graph graph("negative_lstm_wrong_input_count");
  const Module lstm = LSTM("lstm", /*batch=*/2, /*num_steps=*/2, /*input_dim=*/2,
                           /*hidden_dim=*/2, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, lstm.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  const Result<std::vector<Value*>> outputs =
      lstm.build(graph, std::vector<Value*>{}, params.value());
  EXPECT_FALSE(outputs.is_ok());
}

TEST(SeqBatchConstructionPurityNegative, MultiheadAttentionEmbedDimNotDivisibleFailsAtBuild) {
  Graph graph("negative_mha_embed_dim_not_divisible");
  Value* x = graph.add_graph_input(MakeCpuTensorType({2, 5})).value();
  const Module mha = MultiheadAttention("mha", /*batch=*/1, /*seq_len=*/2, /*embed_dim=*/5,
                                        /*num_heads=*/2, /*with_bias=*/false, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, mha.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  const Result<std::vector<Value*>> outputs =
      mha.build(graph, std::vector<Value*>{x}, params.value());
  EXPECT_FALSE(outputs.is_ok());
}

TEST(SeqBatchConstructionPurityNegative, MultiheadAttentionWrongInputCountFailsAtBuild) {
  Graph graph("negative_mha_wrong_input_count");
  const Module mha = MultiheadAttention("mha", /*batch=*/1, /*seq_len=*/2, /*embed_dim=*/4,
                                        /*num_heads=*/2, /*with_bias=*/false, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, mha.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  const Result<std::vector<Value*>> outputs =
      mha.build(graph, std::vector<Value*>{}, params.value());
  EXPECT_FALSE(outputs.is_ok());
}

TEST(SeqBatchConstructionPurityNegative, TransformerEncoderBlockWrongInputCountFailsAtBuild) {
  Graph graph("negative_transformer_block_wrong_input_count");
  Value* x0 = graph.add_graph_input(MakeCpuTensorType({2, 4})).value();
  Value* x1 = graph.add_graph_input(MakeCpuTensorType({2, 4})).value();
  const Module block = TransformerEncoderBlock("block", /*batch=*/1, /*seq_len=*/2, /*embed_dim=*/4,
                                               /*num_heads=*/2, /*ffn_dim=*/6,
                                               /*with_bias=*/false, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, block.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  const Result<std::vector<Value*>> outputs =
      block.build(graph, std::vector<Value*>{x0, x1}, params.value());
  EXPECT_FALSE(outputs.is_ok());
}

}  // namespace
