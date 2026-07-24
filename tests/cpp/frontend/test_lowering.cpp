// lower_to_graph / lower_to_inference_graph 单测(docs/architecture/
// frontend-dsl.md 第 4 章):对 tiny_mlp spec 构图 -> verify(make_op_query)
// -> dump_text 与 testdata/ 下的 golden 文本逐字节比对;再断言
// param_names/wrt_input_indices 与第 4 章"图输入序 [数据输入...,
// 逐层 weight,bias..., target]"契约一致。
//
// golden 更新流程:确认 emitter/lowering 行为变更符合预期后,用测试打印的
// 实际输出(比对失败时下方 AssertLoweringMatchesGolden 会把 actual/expected
// 双方全文打印到测试日志)覆盖 testdata/tiny_mlp_forward_expected.txt /
// testdata/tiny_mlp_inference_expected.txt,并人工核对拓扑
// (matmul/add/relu/matmul[/mse_loss])未出现非预期变化后再提交。
//
// 2026-07-18 重定基(M20 批2 Task4,ADR-0020/docs/architecture/nn-design.md
// ARCH-074):lowering 网络结构段改经 frame::nn 模块构图(逐层独立
// nn::Linear[+nn::Relu] + nn::add_parameter_inputs + module.build),批量
// 参数图输入前置于计算节点构图之前,发射序整体重排(旧序 weight-input/
// matmul/bias-input/add 交错,新序参数图输入整体先行、计算节点整体随后);
// 算子节点集与数据流边集与重定基前拓扑等价(ARCH-074 判定方法②),
// LoweredModel 三元组与图输入总序契约逐位不变(判定方法①,见下方
// LowerToGraphParamNamesAndWrtIndicesMatchInputOrderContract)。
//
// 2026-07-18 追加(code-reviewer 终审修复):逐层构图改用 value_by_name 名字
// 解析(与旧 AppendLayer 同一套流程),不再包一整棵 Sequential——层间数据流
// 按 layer.input 名字取值,而非假定"末层输出即 prediction、每层输入恰为
// 前一层输出"。下方 LowerToGraphSupportsPredictionAtNonLastLayer /
// LowerToGraphSupportsSkipConnectionsAndDataInputReuse 两个新用例覆盖
// frontend-dsl.md FE-002 明确合法、旧实现能正确处理的两类拓扑:
// loss.prediction 指向非末层、层输入跳连或复用数据输入。

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <vector>

#include <frame/core/status.h>
#include <frame/frontend/lowering.h>
#include <frame/frontend/model_spec.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ir/serialization.h>
#include <frame/ops/op_registry.h>

#include "../compiler/golden_test_helpers.h"
#include "tiny_mlp_spec_helper.h"

namespace {

using frame::Result;
using frame::frontend::InitKind;
using frame::frontend::InputSpec;
using frame::frontend::LinearLayerSpec;
using frame::frontend::lower_to_graph;
using frame::frontend::lower_to_inference_graph;
using frame::frontend::LoweredModel;
using frame::frontend::ModelSpec;
using frame::frontend::TensorDataSpec;
using frame::frontend::testing::make_tiny_mlp_spec;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::OpQuery;

// verify(make_op_query()) 通过后取 dump_text,与 expected_path(仓库根相对
// 路径)逐字节比对;手法镜像 golden_test_helpers.h::run_pass_matches_golden,
// 复用其 read_file_contents,但入口是一张已构好的 Graph(而非"解析 input 文本
// + 跑 pass"),故不直接调用该函数,自行实现同风格的诊断输出。
::testing::AssertionResult AssertLoweringMatchesGolden(Graph& graph,
                                                       std::string_view expected_path) {
  const OpQuery query = frame::ops::make_op_query();
  const frame::Status verify_status = graph.verify(query);
  if (!verify_status.is_ok()) {
    return ::testing::AssertionFailure()
           << "AssertLoweringMatchesGolden: graph.verify() failed: " << verify_status.message();
  }

  const std::string actual_text = frame::ir::dump_text(graph);

  const Result<std::string> expected_result =
      frame::compiler::testing::read_file_contents(expected_path);
  if (!expected_result.is_ok()) {
    return ::testing::AssertionFailure()
           << "AssertLoweringMatchesGolden: failed to load expected '" << expected_path
           << "': " << expected_result.status().message();
  }
  const std::string& expected_text = expected_result.value();

  if (actual_text != expected_text) {
    return ::testing::AssertionFailure()
           << "AssertLoweringMatchesGolden: dump_text mismatch\n"
           << "--- actual ---\n"
           << actual_text << "--- expected (from " << expected_path << ") ---\n"
           << expected_text;
  }
  return ::testing::AssertionSuccess();
}

TEST(LoweringTest, LowerToGraphMatchesForwardGolden) {
  const ModelSpec spec = make_tiny_mlp_spec();
  Result<LoweredModel> lowered = lower_to_graph(spec);
  ASSERT_TRUE(lowered.is_ok()) << lowered.status().message();

  EXPECT_TRUE(AssertLoweringMatchesGolden(
      lowered.value().forward, "tests/cpp/frontend/testdata/tiny_mlp_forward_expected.txt"));
}

TEST(LoweringTest, LowerToInferenceGraphMatchesInferenceGolden) {
  const ModelSpec spec = make_tiny_mlp_spec();
  Result<Graph> inference_graph = lower_to_inference_graph(spec);
  ASSERT_TRUE(inference_graph.is_ok()) << inference_graph.status().message();

  EXPECT_TRUE(AssertLoweringMatchesGolden(
      inference_graph.value(), "tests/cpp/frontend/testdata/tiny_mlp_inference_expected.txt"));
}

// 图输入序契约(frontend-dsl.md 第 4 章):[数据输入..., 逐层
// weight,bias..., target]。tiny_mlp:x(index0) -> layer0_weight(index1) ->
// layer0_bias(index2) -> layer1_weight(index3)(layer1 无 bias) ->
// target(index4,仅训练图)。
TEST(LoweringTest, LowerToGraphParamNamesAndWrtIndicesMatchInputOrderContract) {
  const ModelSpec spec = make_tiny_mlp_spec();
  Result<LoweredModel> lowered = lower_to_graph(spec);
  ASSERT_TRUE(lowered.is_ok()) << lowered.status().message();
  const LoweredModel& model = lowered.value();

  const std::vector<std::string> expected_param_names{"layer0_weight", "layer0_bias",
                                                      "layer1_weight"};
  EXPECT_EQ(model.param_names, expected_param_names);

  const std::vector<int32_t> expected_wrt_indices{1, 2, 3};
  EXPECT_EQ(model.wrt_input_indices, expected_wrt_indices);

  ASSERT_EQ(model.param_types.size(), expected_param_names.size());
  // 图输入总数 = x + layer0_weight + layer0_bias + layer1_weight + target。
  EXPECT_EQ(model.forward.inputs().size(), 5u);
}

TEST(LoweringTest, LowerToInferenceGraphHasNoTargetInputAndSingleOutput) {
  const ModelSpec spec = make_tiny_mlp_spec();
  Result<Graph> inference_graph = lower_to_inference_graph(spec);
  ASSERT_TRUE(inference_graph.is_ok()) << inference_graph.status().message();

  // 推理图输入序 = 训练图去掉末尾 target,共 4 个(x + 逐层 weight,bias...)。
  EXPECT_EQ(inference_graph.value().inputs().size(), 4u);
  EXPECT_EQ(inference_graph.value().outputs().size(), 1u);
}

// validate() 失败原样透传(lower_to_graph/lower_to_inference_graph 内部均先
// 调用 validate)。
TEST(LoweringTest, LowerToGraphPropagatesValidateFailure) {
  ModelSpec spec = make_tiny_mlp_spec();
  spec.loss.prediction = "nonexistent_layer";

  const Result<LoweredModel> lowered = lower_to_graph(spec);
  ASSERT_FALSE(lowered.is_ok());
  EXPECT_EQ(lowered.status().code(), frame::ErrorCode::kInvalidArgument);
}

TEST(LoweringTest, LowerToInferenceGraphPropagatesValidateFailure) {
  ModelSpec spec = make_tiny_mlp_spec();
  spec.loss.prediction = "nonexistent_layer";

  const Result<Graph> inference_graph = lower_to_inference_graph(spec);
  ASSERT_FALSE(inference_graph.is_ok());
  EXPECT_EQ(inference_graph.status().code(), frame::ErrorCode::kInvalidArgument);
}

// 均匀采样的数据条目 helper(lo/hi 固定为 [-1, 1),下方两个手工构造的小 spec
// 共用;仅需 lower_to_graph 成功并产出正确拓扑,不关心具体数值,故沿用同一
// 组范围即可)。
TensorDataSpec MakeUniformDataSpec() {
  TensorDataSpec data;
  data.kind = InitKind::kUniformSeeded;
  data.lo = -1.0F;
  data.hi = 1.0F;
  return data;
}

// frontend-dsl.md FE-002 合法、旧实现(AppendLayer + value_by_name)能正确
// lower 的拓扑①:loss.prediction 指向非末层。two-layer spec:a<-x,
// b<-a(声明但不参与 loss),prediction="a"。断言:①两层均建图(与旧实现
// 同款行为,不因 prediction 非末层而提前终止);②LoweredModel 契约覆盖两层
// 参数;③结构性核验 mse_loss 的 prediction 操作数确系 a 层输出(matmul_a),
// 而非最后声明的 b 层输出——证明未被"末层即 prediction"的错误假设污染。
TEST(LoweringTest, LowerToGraphSupportsPredictionAtNonLastLayer) {
  ModelSpec spec;
  spec.name = "pred_not_last";
  spec.batch = 2;

  InputSpec input;
  input.name = "x";
  input.shape = {2, 3};
  spec.inputs = {input};

  LinearLayerSpec layer_a;
  layer_a.name = "a";
  layer_a.input = "x";
  layer_a.weight_shape = {3, 4};

  LinearLayerSpec layer_b;
  layer_b.name = "b";
  layer_b.input = "a";
  layer_b.weight_shape = {4, 5};

  spec.layers = {layer_a, layer_b};

  spec.loss.prediction = "a";  // 非末层
  spec.loss.target_shape = {2, 4};

  spec.optimizer.learning_rate = 0.1;
  spec.training.steps = 1;
  spec.training.seed = 1;
  spec.training.log_every = 0;

  spec.data["x"] = MakeUniformDataSpec();
  spec.data["target"] = MakeUniformDataSpec();
  spec.param_init.weight_lo = -0.1F;
  spec.param_init.weight_hi = 0.1F;
  spec.param_init.bias_lo = 0.0F;
  spec.param_init.bias_hi = 1.0F;

  const Result<LoweredModel> lowered = lower_to_graph(spec);
  ASSERT_TRUE(lowered.is_ok()) << lowered.status().message();
  const LoweredModel& model = lowered.value();

  EXPECT_EQ(model.param_names, (std::vector<std::string>{"a_weight", "b_weight"}));
  EXPECT_EQ(model.wrt_input_indices, (std::vector<int32_t>{1, 2}));
  EXPECT_EQ(model.forward.inputs().size(), 4u);  // x, a_weight, b_weight, target

  const std::vector<Node*>& nodes = model.forward.topological_order();
  ASSERT_EQ(nodes.size(), 7u);  // x, a_weight, b_weight, matmul_a, matmul_b, target, mse_loss
  EXPECT_EQ(nodes[0]->op(), "graph_input");  // x
  EXPECT_EQ(nodes[1]->op(), "graph_input");  // a_weight
  EXPECT_EQ(nodes[2]->op(), "graph_input");  // b_weight
  ASSERT_EQ(nodes[3]->op(), "matmul");       // matmul_a(x, a_weight)
  EXPECT_EQ(nodes[3]->inputs()[0], nodes[0]->output(0));
  EXPECT_EQ(nodes[3]->inputs()[1], nodes[1]->output(0));
  ASSERT_EQ(nodes[4]->op(), "matmul");  // matmul_b(matmul_a 输出, b_weight)——仍建图但不参与 loss
  EXPECT_EQ(nodes[4]->inputs()[0], nodes[3]->output(0));
  EXPECT_EQ(nodes[4]->inputs()[1], nodes[2]->output(0));
  EXPECT_EQ(nodes[5]->op(), "graph_input");  // target
  ASSERT_EQ(nodes[6]->op(), "mse_loss");
  // 关键断言:prediction 操作数取自 a 层输出(matmul_a,nodes[3]),不是最后
  // 声明的 b 层输出(matmul_b,nodes[4])。
  EXPECT_EQ(nodes[6]->inputs()[0], nodes[3]->output(0));
  EXPECT_EQ(nodes[6]->inputs()[1], nodes[5]->output(0));

  const OpQuery query = frame::ops::make_op_query();
  EXPECT_TRUE(model.forward.verify(query).is_ok());
}

// frontend-dsl.md FE-002 合法、旧实现能正确 lower 的拓扑②:层输入跳连
// (取非紧邻前层)与复用数据输入(多层直接消费同一数据输入)。three-layer
// spec:p<-x,q<-x(复用数据输入,而非紧邻前层 p 的输出),r<-p(跳连:声明序
// 上紧邻前层是 q,但取的是更早声明的 p),prediction="r"。断言按上例同一手法
// 做结构性核验。
TEST(LoweringTest, LowerToGraphSupportsSkipConnectionsAndDataInputReuse) {
  ModelSpec spec;
  spec.name = "skip_connection";
  spec.batch = 2;

  InputSpec input;
  input.name = "x";
  input.shape = {2, 3};
  spec.inputs = {input};

  LinearLayerSpec layer_p;
  layer_p.name = "p";
  layer_p.input = "x";
  layer_p.weight_shape = {3, 4};

  LinearLayerSpec layer_q;
  layer_q.name = "q";
  layer_q.input = "x";  // 复用数据输入,而非紧邻前层 p 的输出
  layer_q.weight_shape = {3, 4};

  LinearLayerSpec layer_r;
  layer_r.name = "r";
  layer_r.input = "p";  // 跳连:声明序上紧邻前层是 q,取的却是更早的 p
  layer_r.weight_shape = {4, 2};

  spec.layers = {layer_p, layer_q, layer_r};

  spec.loss.prediction = "r";
  spec.loss.target_shape = {2, 2};

  spec.optimizer.learning_rate = 0.1;
  spec.training.steps = 1;
  spec.training.seed = 1;
  spec.training.log_every = 0;

  spec.data["x"] = MakeUniformDataSpec();
  spec.data["target"] = MakeUniformDataSpec();
  spec.param_init.weight_lo = -0.1F;
  spec.param_init.weight_hi = 0.1F;
  spec.param_init.bias_lo = 0.0F;
  spec.param_init.bias_hi = 1.0F;

  const Result<LoweredModel> lowered = lower_to_graph(spec);
  ASSERT_TRUE(lowered.is_ok()) << lowered.status().message();
  const LoweredModel& model = lowered.value();

  EXPECT_EQ(model.param_names, (std::vector<std::string>{"p_weight", "q_weight", "r_weight"}));
  EXPECT_EQ(model.wrt_input_indices, (std::vector<int32_t>{1, 2, 3}));
  EXPECT_EQ(model.forward.inputs().size(), 5u);  // x, p_weight, q_weight, r_weight, target

  const std::vector<Node*>& nodes = model.forward.topological_order();
  ASSERT_EQ(nodes.size(),
            9u);  // x,p_weight,q_weight,r_weight,matmul_p,matmul_q,matmul_r,target,mse_loss
  EXPECT_EQ(nodes[0]->op(), "graph_input");  // x
  EXPECT_EQ(nodes[1]->op(), "graph_input");  // p_weight
  EXPECT_EQ(nodes[2]->op(), "graph_input");  // q_weight
  EXPECT_EQ(nodes[3]->op(), "graph_input");  // r_weight
  ASSERT_EQ(nodes[4]->op(), "matmul");       // matmul_p(x, p_weight)
  EXPECT_EQ(nodes[4]->inputs()[0], nodes[0]->output(0));
  EXPECT_EQ(nodes[4]->inputs()[1], nodes[1]->output(0));
  ASSERT_EQ(nodes[5]->op(), "matmul");  // matmul_q(x, q_weight):复用数据输入
  EXPECT_EQ(nodes[5]->inputs()[0], nodes[0]->output(0));
  EXPECT_EQ(nodes[5]->inputs()[1], nodes[2]->output(0));
  ASSERT_EQ(nodes[6]->op(), "matmul");  // matmul_r(matmul_p 输出, r_weight):跳连取 p 而非 q
  EXPECT_EQ(nodes[6]->inputs()[0], nodes[4]->output(0));
  EXPECT_EQ(nodes[6]->inputs()[1], nodes[3]->output(0));
  EXPECT_EQ(nodes[7]->op(), "graph_input");  // target
  ASSERT_EQ(nodes[8]->op(), "mse_loss");
  EXPECT_EQ(nodes[8]->inputs()[0], nodes[6]->output(0));
  EXPECT_EQ(nodes[8]->inputs()[1], nodes[7]->output(0));

  const OpQuery query = frame::ops::make_op_query();
  EXPECT_TRUE(model.forward.verify(query).is_ok());
}

}  // namespace
