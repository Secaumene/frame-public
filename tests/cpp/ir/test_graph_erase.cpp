// Graph::erase_node 单测:输出被其他节点引用/被图输出引用时拒绝删除(拒绝制造
// 悬挂引用);孤立节点可删除且拓扑序/图输入列表同步收缩。全程纯主机内存。
#include <gtest/gtest.h>
#include <string_view>
#include <vector>

#include <frame/ir/graph.h>

#include "ir_test_helpers.h"

namespace {

using frame::ErrorCode;
using frame::Result;
using frame::Status;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::Value;
using frame::ir::testing::MakeFloat32Type;

TEST(GraphEraseTest, ErasingNodeReferencedByAnotherNodesInputReturnsError) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* producer = graph.create_node("relu", {input}, {MakeFloat32Type({4})}).value();
  // consumer 引用 producer 的输出,使 producer 不再孤立。
  graph.create_node("relu2", {producer->output(0)}, {MakeFloat32Type({4})});
  ASSERT_EQ(graph.topological_order().size(), 3u);

  const Status status = graph.erase_node(producer);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("referenced by another node's input"), std::string_view::npos);
  // 拒绝后节点应仍留在图中(erase 失败不改变图状态)。
  EXPECT_EQ(graph.topological_order().size(), 3u);
}

TEST(GraphEraseTest, ErasingNodeReferencedByGraphOutputReturnsError) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* producer = graph.create_node("relu", {input}, {MakeFloat32Type({4})}).value();
  ASSERT_TRUE(graph.mark_output(producer->output(0)).is_ok());

  const Status status = graph.erase_node(producer);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("referenced by a graph output"), std::string_view::npos);
  EXPECT_EQ(graph.topological_order().size(), 2u);
  EXPECT_EQ(graph.outputs().size(), 1u);
}

TEST(GraphEraseTest, ErasingIsolatedNodeSucceedsAndShrinksTopoOrderInPlace) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* keep_first = graph.create_node("relu", {input}, {MakeFloat32Type({4})}).value();
  Node* isolated = graph.create_node("dead_op", {input}, {MakeFloat32Type({4})}).value();
  Node* keep_last =
      graph.create_node("relu2", {keep_first->output(0)}, {MakeFloat32Type({4})}).value();
  ASSERT_EQ(graph.topological_order().size(), 4u);

  const Status status = graph.erase_node(isolated);
  ASSERT_TRUE(status.is_ok());

  const std::vector<Node*>& topo = graph.topological_order();
  ASSERT_EQ(topo.size(), 3u);
  // 拓扑序原地收缩:isolated 消失,其余节点保持原有相对顺序。
  EXPECT_EQ(topo[0], input->producer());
  EXPECT_EQ(topo[1], keep_first);
  EXPECT_EQ(topo[2], keep_last);
}

TEST(GraphEraseTest, ErasingUnusedGraphInputNodeShrinksInputsListToo) {
  Graph graph;
  Value* used_input = graph.add_graph_input(MakeFloat32Type({4})).value();
  Value* unused_input = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* only_consumer = graph.create_node("relu", {used_input}, {MakeFloat32Type({4})}).value();
  ASSERT_TRUE(graph.mark_output(only_consumer->output(0)).is_ok());
  ASSERT_EQ(graph.inputs().size(), 2u);
  ASSERT_NE(unused_input->producer(), nullptr);

  const Status status = graph.erase_node(unused_input->producer());
  ASSERT_TRUE(status.is_ok());

  ASSERT_EQ(graph.inputs().size(), 1u);
  EXPECT_EQ(graph.inputs()[0], used_input);
  EXPECT_EQ(graph.topological_order().size(), 2u);
}

TEST(GraphEraseTest, ErasingNullptrReturnsError) {
  Graph graph;
  const Status status = graph.erase_node(nullptr);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
}

TEST(GraphEraseTest, ErasingNodeNotOwnedByThisGraphReturnsError) {
  Graph graph_a;
  Graph graph_b;
  // add_node 已移除(决议点 5-④);跨图归属场景改用 create_node 在 graph_b 上
  // 建立零输入零输出节点,保留"节点属于另一图"这一测试意图不变。
  const Result<Node*> foreign_result = graph_b.create_node("const_fill", {}, {});
  ASSERT_TRUE(foreign_result.is_ok());
  Node* foreign = foreign_result.value();
  ASSERT_NE(foreign, nullptr);

  const Status status = graph_a.erase_node(foreign);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("does not belong to this graph"), std::string_view::npos);
}

}  // namespace
