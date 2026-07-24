// M23 批5 T5 频域批次工厂的构图纯度测试(ARCH-072,docs/architecture/
// nn-design.md §2/§8;工厂契约见 docs/plan/2026-07-21-batch5-m23-fft.md
// §1.5)。覆盖 SpectralConv1d/FourierFilter1d/Fno1dBlock 三工厂:build() 前后
// 图节点计数增量恰等于预期新增 op 节点数(实测校准值,构图期机械可复算,同
// tests/cpp/nn/test_seq_batch_construction_purity.cpp 先例)、关键算子存在、
// 输出 shape/dtype 正确。文件末尾另含负例(modes>n/2+1、输入数错)——均应在
// build() 期经 Result 报错,不静默产出错误图(ARCH-031 口径)。
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

// 计数指定 op 在图中出现的节点数(REUSE-002:同
// test_seq_batch_construction_purity.cpp::CountNodesWithOp 先例,各文件按需
// 持一份轻量私有 helper)。
int CountNodesWithOp(Graph& graph, const std::string& op) {
  int count = 0;
  for (Node* node : graph.topological_order()) {
    if (node->op() == op) ++count;
  }
  return count;
}

// ---------------------------------------------------------------------------
// SpectralConv1d(name, batch, in_channels, out_channels, n, modes, dtype):
// rfft -> slice(modes) -> 逐模态展开(每模态 18 节点)-> concat 回拼 -> 零补
// (modes<k 时 +2)-> irfft。
// ---------------------------------------------------------------------------

TEST(FftBatchConstructionPurity,
     SpectralConv1dTruncatedModesAddsExpectedNodeCountAndFinalOpIsIrfft) {
  constexpr int64_t kBatch = 2;
  constexpr int64_t kIn = 3;
  constexpr int64_t kOut = 4;
  constexpr int64_t kN = 8;      // k = n/2+1 = 5
  constexpr int64_t kModes = 2;  // modes < k,触发零补 concat

  Graph graph("purity_spectral_conv1d_truncated");
  Value* x = graph.add_graph_input(MakeCpuTensorType({kBatch, kIn, kN})).value();

  const Module sc = SpectralConv1d("sc", kBatch, kIn, kOut, kN, kModes, DType::of<float>());
  const std::vector<ParamSpec> param_specs = sc.parameters();
  ASSERT_EQ(param_specs.size(), 2u);  // W_re, W_im
  EXPECT_EQ(param_specs[0].name, "sc.W_re");
  EXPECT_EQ(param_specs[0].type.shape, Shape({kIn, kModes * kOut}));
  EXPECT_EQ(param_specs[1].name, "sc.W_im");
  EXPECT_EQ(param_specs[1].type.shape, Shape({kIn, kModes * kOut}));

  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, param_specs);
  ASSERT_TRUE(params.is_ok()) << params.status().message();

  const size_t pre_build = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      sc.build(graph, std::vector<Value*>{x}, params.value());
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  // 每模态 18 节点(2 次模态/re-im 拆分 slice + 2 次 reshape + 2 次权重列
  // slice + 4 次 matmul + 1 次 constant(-1) + 1 次 mul + 2 次 add + 2 次
  // reshape 回拼 + 1 次 concat(re,im)= 3+2+2+4+1+1+2+2+1=18)* modes(2)=36,
  // 外层 rfft(1)+slice(modes)(1)+concat(modes 回拼)(1)+zero-pad
  // constant+concat(2,modes<k)+irfft(1)=6;合计 42(实测校准值)。
  EXPECT_EQ(graph.topological_order().size() - pre_build, 42u);
  ASSERT_EQ(outputs.value().size(), 1u);
  Node* out_node = outputs.value()[0]->producer();
  EXPECT_EQ(out_node->op(), "irfft");
  EXPECT_EQ(out_node->output(0)->type().shape, Shape({kBatch, kOut, kN}));
  EXPECT_EQ(out_node->output(0)->type().dtype, DType::of<float>());
  EXPECT_EQ(CountNodesWithOp(graph, "rfft"), 1);
  EXPECT_EQ(CountNodesWithOp(graph, "irfft"), 1);
  EXPECT_EQ(CountNodesWithOp(graph, "matmul"), 4 * kModes);
  // concat:modes 次 re/im 回拼 + 1 次模态回拼 + 1 次零补回拼。
  EXPECT_EQ(CountNodesWithOp(graph, "concat"), kModes + 2);
  // constant:modes 次 -1 splat(复乘组合)+ 1 次零补 splat。
  EXPECT_EQ(CountNodesWithOp(graph, "constant"), kModes + 1);
}

TEST(FftBatchConstructionPurity, SpectralConv1dFullModesSkipsZeroPadConcat) {
  constexpr int64_t kBatch = 2;
  constexpr int64_t kIn = 3;
  constexpr int64_t kOut = 4;
  constexpr int64_t kN = 8;
  constexpr int64_t kModes = 5;  // == k,零补分支应被跳过

  Graph graph("purity_spectral_conv1d_full");
  Value* x = graph.add_graph_input(MakeCpuTensorType({kBatch, kIn, kN})).value();

  const Module sc = SpectralConv1d("sc", kBatch, kIn, kOut, kN, kModes, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, sc.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();

  const size_t pre_build = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      sc.build(graph, std::vector<Value*>{x}, params.value());
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  // 18*modes(5)+外层 4(rfft+slice(modes)+concat(modes 回拼)+irfft,无零补
  // constant/concat)=90+4=94(实测校准值)。
  EXPECT_EQ(graph.topological_order().size() - pre_build, 94u);
  // 无零补:constant 节点数恰 == modes(仅 -1 splat,无零补 splat)。
  EXPECT_EQ(CountNodesWithOp(graph, "constant"), kModes);
}

// ---------------------------------------------------------------------------
// FourierFilter1d(name, batch, channels, n, dtype):rfft -> 逐元素复乘(slice
// re/im + 5 次 mul/add 组合 + 1 次 constant(-1))-> concat -> irfft。
// ---------------------------------------------------------------------------

TEST(FftBatchConstructionPurity, FourierFilter1dAddsExpectedNodeCountAndFinalOpIsIrfft) {
  constexpr int64_t kBatch = 2;
  constexpr int64_t kChannels = 3;
  constexpr int64_t kN = 8;  // k = 5

  Graph graph("purity_fourier_filter1d");
  Value* x = graph.add_graph_input(MakeCpuTensorType({kBatch, kChannels, kN})).value();

  const Module ff = FourierFilter1d("ff", kBatch, kChannels, kN, DType::of<float>());
  const std::vector<ParamSpec> param_specs = ff.parameters();
  ASSERT_EQ(param_specs.size(), 2u);  // w_re, w_im
  EXPECT_EQ(param_specs[0].name, "ff.w_re");
  EXPECT_EQ(param_specs[0].type.shape, Shape({kBatch, kChannels, 5, 1}));
  EXPECT_EQ(param_specs[1].name, "ff.w_im");
  EXPECT_EQ(param_specs[1].type.shape, Shape({kBatch, kChannels, 5, 1}));

  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, param_specs);
  ASSERT_TRUE(params.is_ok()) << params.status().message();

  const size_t pre_build = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      ff.build(graph, std::vector<Value*>{x}, params.value());
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  // rfft(1)+slice re/im(2)+mul(wre*xre,wim*xim,neg*wimxim,wre*xim,wim*xre)
  // (5)+constant(-1)(1)+add(y_re,y_im)(2)+concat(1)+irfft(1)=13
  // (实测校准值)。
  EXPECT_EQ(graph.topological_order().size() - pre_build, 13u);
  ASSERT_EQ(outputs.value().size(), 1u);
  Node* out_node = outputs.value()[0]->producer();
  EXPECT_EQ(out_node->op(), "irfft");
  EXPECT_EQ(out_node->output(0)->type().shape, Shape({kBatch, kChannels, kN}));
  EXPECT_EQ(out_node->output(0)->type().dtype, DType::of<float>());
  EXPECT_EQ(CountNodesWithOp(graph, "rfft"), 1);
  EXPECT_EQ(CountNodesWithOp(graph, "slice"), 2);
  EXPECT_EQ(CountNodesWithOp(graph, "mul"), 5);
  EXPECT_EQ(CountNodesWithOp(graph, "add"), 2);
  EXPECT_EQ(CountNodesWithOp(graph, "concat"), 1);
  EXPECT_EQ(CountNodesWithOp(graph, "constant"), 1);
}

// ---------------------------------------------------------------------------
// Fno1dBlock(name, batch, in_channels, out_channels, n, modes, dtype):频域
// 分支与逐点旁路两条支路相加后取 tanh,children=[SpectralConv1d, Conv1d];
// y=tanh(add(spectral(x), conv1x1(x)))。
// ---------------------------------------------------------------------------

TEST(FftBatchConstructionPurity, Fno1dBlockAddsExpectedNodeCountAndFinalOpIsTanh) {
  constexpr int64_t kBatch = 2;
  constexpr int64_t kIn = 3;
  constexpr int64_t kOut = 4;
  constexpr int64_t kN = 8;
  constexpr int64_t kModes = 2;

  Graph graph("purity_fno1d_block");
  Value* x = graph.add_graph_input(MakeCpuTensorType({kBatch, kIn, kN})).value();

  const Module block = Fno1dBlock("fno", kBatch, kIn, kOut, kN, kModes, DType::of<float>());
  ASSERT_EQ(block.children.size(), 2u);  // spectral, bypass
  const std::vector<ParamSpec> param_specs = block.parameters();
  ASSERT_EQ(param_specs.size(), 4u);  // spectral(W_re,W_im) + bypass(weight,bias)
  EXPECT_EQ(param_specs[0].name, "fno.spectral.W_re");
  EXPECT_EQ(param_specs[1].name, "fno.spectral.W_im");
  EXPECT_EQ(param_specs[2].name, "fno.bypass.weight");
  EXPECT_EQ(param_specs[2].type.shape, Shape({kOut, kIn, 1}));
  EXPECT_EQ(param_specs[3].name, "fno.bypass.bias");
  EXPECT_EQ(param_specs[3].type.shape, Shape({kOut}));

  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, param_specs);
  ASSERT_TRUE(params.is_ok()) << params.status().message();

  const size_t pre_build = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      block.build(graph, std::vector<Value*>{x}, params.value());
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  // spectral(42,modes<k)+conv1d(1)+add(1)+tanh(1)=45(实测校准值)。
  EXPECT_EQ(graph.topological_order().size() - pre_build, 45u);
  ASSERT_EQ(outputs.value().size(), 1u);
  Node* out_node = outputs.value()[0]->producer();
  EXPECT_EQ(out_node->op(), "tanh");
  EXPECT_EQ(out_node->output(0)->type().shape, Shape({kBatch, kOut, kN}));
  EXPECT_EQ(CountNodesWithOp(graph, "conv1d"), 1);
  EXPECT_EQ(CountNodesWithOp(graph, "tanh"), 1);
}

// ---------------------------------------------------------------------------
// 负例:modes>n/2+1 / 输入数错——均由 build() 期经 Result 报错(不静默产出
// 错误图,ARCH-031 口径)。
// ---------------------------------------------------------------------------

TEST(FftBatchConstructionPurityNegative, SpectralConv1dModesExceedingKFailsAtBuild) {
  Graph graph("negative_spectral_conv1d_modes_exceed_k");
  Value* x = graph.add_graph_input(MakeCpuTensorType({2, 3, 8})).value();  // k=5
  const Module sc = SpectralConv1d("sc", /*batch=*/2, /*in_channels=*/3, /*out_channels=*/4,
                                   /*n=*/8, /*modes=*/6, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, sc.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  const Result<std::vector<Value*>> outputs =
      sc.build(graph, std::vector<Value*>{x}, params.value());
  EXPECT_FALSE(outputs.is_ok());
}

TEST(FftBatchConstructionPurityNegative, SpectralConv1dWrongInputCountFailsAtBuild) {
  Graph graph("negative_spectral_conv1d_wrong_input_count");
  const Module sc = SpectralConv1d("sc", /*batch=*/2, /*in_channels=*/3, /*out_channels=*/4,
                                   /*n=*/8, /*modes=*/2, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, sc.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  const Result<std::vector<Value*>> outputs =
      sc.build(graph, std::vector<Value*>{}, params.value());
  EXPECT_FALSE(outputs.is_ok());
}

TEST(FftBatchConstructionPurityNegative, FourierFilter1dWrongInputCountFailsAtBuild) {
  Graph graph("negative_fourier_filter1d_wrong_input_count");
  const Module ff = FourierFilter1d("ff", /*batch=*/2, /*channels=*/3, /*n=*/8, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, ff.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  const Result<std::vector<Value*>> outputs =
      ff.build(graph, std::vector<Value*>{}, params.value());
  EXPECT_FALSE(outputs.is_ok());
}

TEST(FftBatchConstructionPurityNegative, Fno1dBlockWrongInputCountFailsAtBuild) {
  Graph graph("negative_fno1d_block_wrong_input_count");
  const Module block = Fno1dBlock("fno", /*batch=*/2, /*in_channels=*/3, /*out_channels=*/4,
                                  /*n=*/8, /*modes=*/2, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, block.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  const Result<std::vector<Value*>> outputs =
      block.build(graph, std::vector<Value*>{}, params.value());
  EXPECT_FALSE(outputs.is_ok());
}

TEST(FftBatchConstructionPurityNegative, Fno1dBlockModesExceedingKFailsAtBuildViaSpectralChild) {
  Graph graph("negative_fno1d_block_modes_exceed_k");
  Value* x = graph.add_graph_input(MakeCpuTensorType({2, 3, 8})).value();  // k=5
  const Module block = Fno1dBlock("fno", /*batch=*/2, /*in_channels=*/3, /*out_channels=*/4,
                                  /*n=*/8, /*modes=*/6, DType::of<float>());
  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, block.parameters());
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  const Result<std::vector<Value*>> outputs =
      block.build(graph, std::vector<Value*>{x}, params.value());
  EXPECT_FALSE(outputs.is_ok());
}

}  // namespace
