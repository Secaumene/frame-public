#pragma once
// MLP 前向图(含 mse_loss)构造 helper(namespace frame::compiler::testing)。
// 提取原因:test_training_loop.cpp(M18 训练闭环收敛用例)与
// test_onnx_weights.cpp(ADR-0013 train-save-import-infer 链路用例,判定②)
// 都需要构造完全相同的训练用前向图,原地各写一份将产生 ≥20 行同构重复
// (REUSE-002)。图形状取舍(v0 无广播下 bias 建模为与 matmul 输出同形张量)的
// 完整论证见 test_training_loop.cpp 文件头注释,本文件不重复。

#include <cstdint>
#include <gtest/gtest.h>

#include <frame/core/dtype.h>
#include <frame/core/status.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/graph_builder.h>

#include "../ops/elementwise_op_test_helpers.h"

namespace frame::compiler::testing {

// 网络维度(小规模,加速测试执行):x[8,4] -> matmul(W1[4,8]) -> add(b1[8,8])
// -> relu -> matmul(W2[8,1]) -> mse_loss(., target[8,1])。
inline constexpr int64_t kMlpBatchSize = 8;
inline constexpr int64_t kMlpInputDim = 4;
inline constexpr int64_t kMlpHiddenDim = 8;
inline constexpr int64_t kMlpOutputDim = 1;

// graph_inputs 按序 [x, w1, b1, w2, target](下标 0..4);训练场景下
// wrt = {1, 2, 3}(w1/b1/w2 是待训练参数,x/target 是每步固定不变的数据,
// 不求梯度)。
inline frame::ir::Graph BuildMlpForwardGraph() {
  using frame::DType;
  using frame::ir::Graph;
  using frame::ir::Node;
  using frame::ir::Value;
  using frame::ops::create_node_with_inferred_types;
  using frame::ops::testing::MakeType;

  Graph graph("mlp_forward");
  Value* x =
      graph.add_graph_input(MakeType(DType::of<float>(), {kMlpBatchSize, kMlpInputDim})).value();
  Value* w1 =
      graph.add_graph_input(MakeType(DType::of<float>(), {kMlpInputDim, kMlpHiddenDim})).value();
  Value* b1 =
      graph.add_graph_input(MakeType(DType::of<float>(), {kMlpBatchSize, kMlpHiddenDim})).value();
  Value* w2 =
      graph.add_graph_input(MakeType(DType::of<float>(), {kMlpHiddenDim, kMlpOutputDim})).value();
  Value* target =
      graph.add_graph_input(MakeType(DType::of<float>(), {kMlpBatchSize, kMlpOutputDim})).value();

  Node* matmul1 = create_node_with_inferred_types(graph, "matmul", {x, w1}).value();
  Node* add1 = create_node_with_inferred_types(graph, "add", {matmul1->output(0), b1}).value();
  Node* relu1 = create_node_with_inferred_types(graph, "relu", {add1->output(0)}).value();
  Node* matmul2 = create_node_with_inferred_types(graph, "matmul", {relu1->output(0), w2}).value();
  Node* loss_node =
      create_node_with_inferred_types(graph, "mse_loss", {matmul2->output(0), target}).value();
  const frame::Status mark_status = graph.mark_output(loss_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

}  // namespace frame::compiler::testing
