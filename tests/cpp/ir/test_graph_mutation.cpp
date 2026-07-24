// Graph::replace_all_uses / Graph::swap_node_inputs 单测(M8 决议点 B,受控
// 图变异 API;M9 裁决修订2 扩展第二豁免为一般规则,graph.h 头注释"三种情形"):
//   replace_all_uses——正常替换(消费者输入 + 图输出列表双更新)、第二豁免的
//   两个正例(0 输入 producer 重定位 / 非 0 输入但全部输入均早于 from 的一般
//   情形,均要求替换后 dump_text 拓扑序合法且 verify 通过)、四类报错路径
//   (类型四元组不等 / 跨图 / from==to / 拓扑序不变式违例,后者即第二豁免的
//   反例:to 依赖 from 的输出 -> 报错且图不变)。
//   swap_node_inputs——正常交换、三类报错路径(越界 / i==j / 跨图)。
// 全程纯主机内存,不依赖任何已注册后端(TensorType::device 只是数据字段,同
// ir_test_helpers.h 既有纪律);OpQuery 用本文件内的"全放行" fake 回调,不
// 依赖 ops 层(ARCH-001,同 test_graph_verify.cpp/test_graph_erase.cpp 既有
// 先例)。
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <vector>

#include <frame/core/device.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ir/serialization.h>

#include "ir_test_helpers.h"

namespace {

using frame::Device;
using frame::ErrorCode;
using frame::Result;
using frame::Status;
using frame::ir::dump_text;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::OpQuery;
using frame::ir::TensorType;
using frame::ir::Value;
using frame::ir::testing::MakeFloat32Type;

// 全放行 OpQuery:本文件全部节点均为测试专用 op 名(未注册进任何真实
// OpRegistry),verify() 只需结构层面通过,不关心具体 schema 约束。
OpQuery AlwaysValidQuery() {
  OpQuery query;
  query.op_registered = [](std::string_view) { return true; };
  query.check_schema = [](const Node&) { return Status::ok(); };
  return query;
}

// ---------------------------------------------------------------------------
// replace_all_uses:正常路径。
// ---------------------------------------------------------------------------

TEST(GraphReplaceAllUsesTest, UpdatesConsumerInputAndGraphOutput) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  // to_producer 先于 from_producer 创建,天然满足拓扑序不变式(无需 0 输入
  // 重定位),用于聚焦"替换范围"这一测试意图。
  Node* to_producer = graph.create_node("relu", {input}, {MakeFloat32Type({4})}).value();
  Node* from_producer = graph.create_node("relu2", {input}, {MakeFloat32Type({4})}).value();
  Node* consumer =
      graph.create_node("relu3", {from_producer->output(0)}, {MakeFloat32Type({4})}).value();
  ASSERT_TRUE(graph.mark_output(from_producer->output(0)).is_ok());
  ASSERT_EQ(graph.outputs().size(), 1u);

  const Status status = graph.replace_all_uses(from_producer->output(0), to_producer->output(0));
  ASSERT_TRUE(status.is_ok()) << status.message();

  // 消费者输入侧更新。
  ASSERT_EQ(consumer->inputs().size(), 1u);
  EXPECT_EQ(consumer->inputs()[0], to_producer->output(0));
  // 图输出列表侧同步更新(图输出也是一种 use)。
  ASSERT_EQ(graph.outputs().size(), 1u);
  EXPECT_EQ(graph.outputs()[0], to_producer->output(0));

  EXPECT_TRUE(graph.verify(AlwaysValidQuery()).is_ok());
}

TEST(GraphReplaceAllUsesTest, ZeroInputProducerIsRelocatedBeforeFromInTopologicalOrder) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* from_producer = graph.create_node("relu", {input}, {MakeFloat32Type({4})}).value();
  Node* consumer =
      graph.create_node("relu2", {from_producer->output(0)}, {MakeFloat32Type({4})}).value();
  // to_producer 是 0 输入节点,创建于 from_producer 之后(拓扑序上排在
  // from_producer 之后,若不重定位则违反不变式)——命中重定位豁免。
  Node* to_producer = graph.create_node("const_fill", {}, {MakeFloat32Type({4})}).value();
  ASSERT_TRUE(graph.mark_output(consumer->output(0)).is_ok());

  const std::vector<Node*> topo_before = graph.topological_order();
  ASSERT_EQ(topo_before.size(), 4u);
  // 重定位前:[graph_input, from_producer, consumer, to_producer]。
  EXPECT_EQ(topo_before[1], from_producer);
  EXPECT_EQ(topo_before[3], to_producer);

  const Status status = graph.replace_all_uses(from_producer->output(0), to_producer->output(0));
  ASSERT_TRUE(status.is_ok()) << status.message();

  const std::vector<Node*>& topo_after = graph.topological_order();
  ASSERT_EQ(topo_after.size(), 4u);
  // 重定位后:to_producer 就地移到 from_producer 紧前,其余节点相对顺序不变。
  EXPECT_EQ(topo_after[0], input->producer());
  EXPECT_EQ(topo_after[1], to_producer);
  EXPECT_EQ(topo_after[2], from_producer);
  EXPECT_EQ(topo_after[3], consumer);

  EXPECT_EQ(consumer->inputs()[0], to_producer->output(0));

  // 替换后 dump_text 拓扑序合法且 verify 通过(交付清单显式要求的两项断言)。
  const std::string text = dump_text(graph);
  EXPECT_NE(text.find("const_fill"), std::string::npos);
  EXPECT_TRUE(graph.verify(AlwaysValidQuery()).is_ok());
}

TEST(GraphReplaceAllUsesTest, SecondExemptionRelocatesWhenAllInputsPrecedeFromIndex) {
  // M9 裁决修订2 第二豁免的一般(非 0 输入)正例:to 的 producer 带 2 个输入,
  // 二者的 producer 拓扑下标均严格 < from 的 producer 下标,应触发重定位
  // (0 输入豁免是本条件的平凡实例,见 ZeroInputProducerIsRelocatedBeforeFrom
  // InTopologicalOrder;本用例覆盖"非 0 输入但仍全部早于 from"的一般情形)。
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();                        // idx0
  Node* helper = graph.create_node("id_helper", {input}, {MakeFloat32Type({4})}).value();    // idx1
  Node* from_producer = graph.create_node("relu", {input}, {MakeFloat32Type({4})}).value();  // idx2
  Node* consumer = graph.create_node("relu2", {from_producer->output(0)}, {MakeFloat32Type({4})})
                       .value();  // idx3
  // to_producer 的两个输入的 producer 分别是 graph_input(idx0)与 helper
  // (idx1),均严格 < from_producer 的下标(idx2)。
  Node* to_producer =
      graph.create_node("add", {input, helper->output(0)}, {MakeFloat32Type({4})}).value();  // idx4
  ASSERT_TRUE(graph.mark_output(consumer->output(0)).is_ok());

  const std::vector<Node*> topo_before = graph.topological_order();
  ASSERT_EQ(topo_before.size(), 5u);
  EXPECT_EQ(topo_before[2], from_producer);
  EXPECT_EQ(topo_before[4], to_producer);

  const Status status = graph.replace_all_uses(from_producer->output(0), to_producer->output(0));
  ASSERT_TRUE(status.is_ok()) << status.message();

  const std::vector<Node*>& topo_after = graph.topological_order();
  ASSERT_EQ(topo_after.size(), 5u);
  // 重定位后:to_producer 就地移到 from_producer 原位置(idx2),其余节点
  // (from_producer/consumer)整体右移一位,input/helper 位置不变。
  EXPECT_EQ(topo_after[0], input->producer());
  EXPECT_EQ(topo_after[1], helper);
  EXPECT_EQ(topo_after[2], to_producer);
  EXPECT_EQ(topo_after[3], from_producer);
  EXPECT_EQ(topo_after[4], consumer);

  EXPECT_EQ(consumer->inputs()[0], to_producer->output(0));

  // 重定位成功 + verify 通过 + dump 拓扑合法(交付清单显式要求的三项断言)。
  const std::string text = dump_text(graph);
  EXPECT_NE(text.find("add("), std::string::npos);
  EXPECT_TRUE(graph.verify(AlwaysValidQuery()).is_ok());
}

// ---------------------------------------------------------------------------
// replace_all_uses:报错路径。
// ---------------------------------------------------------------------------

TEST(GraphReplaceAllUsesTest, MismatchedTensorTypeQuadrupleIsRejected) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* from_producer = graph.create_node("relu", {input}, {MakeFloat32Type({4})}).value();
  // to 的 shape 与 from 不同(4 维 tuple 中的 shape 分量不等)。
  Node* to_producer = graph.create_node("relu2", {input}, {MakeFloat32Type({8})}).value();

  const Status status = graph.replace_all_uses(from_producer->output(0), to_producer->output(0));
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("different tensor types"), std::string_view::npos);
}

TEST(GraphReplaceAllUsesTest, CrossGraphFromIsRejected) {
  Graph graph_a;
  Graph graph_b;
  Value* input_a = graph_a.add_graph_input(MakeFloat32Type({4})).value();
  Node* from_in_b = graph_b.create_node("const_fill", {}, {MakeFloat32Type({4})}).value();
  Node* to_in_a = graph_a.create_node("relu", {input_a}, {MakeFloat32Type({4})}).value();

  const Status status = graph_a.replace_all_uses(from_in_b->output(0), to_in_a->output(0));
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("does not belong to this graph"), std::string_view::npos);
}

TEST(GraphReplaceAllUsesTest, CrossGraphToIsRejected) {
  Graph graph_a;
  Graph graph_b;
  Value* input_a = graph_a.add_graph_input(MakeFloat32Type({4})).value();
  Node* from_in_a = graph_a.create_node("relu", {input_a}, {MakeFloat32Type({4})}).value();
  Node* to_in_b = graph_b.create_node("const_fill", {}, {MakeFloat32Type({4})}).value();

  const Status status = graph_a.replace_all_uses(from_in_a->output(0), to_in_b->output(0));
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("does not belong to this graph"), std::string_view::npos);
}

TEST(GraphReplaceAllUsesTest, FromEqualsToIsRejected) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* node = graph.create_node("relu", {input}, {MakeFloat32Type({4})}).value();

  const Status status = graph.replace_all_uses(node->output(0), node->output(0));
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("must be different values"), std::string_view::npos);
}

TEST(GraphReplaceAllUsesTest, TopologicalOrderInvariantViolationIsRejected) {
  // M9 裁决修订2(第二豁免,严格版):重定位要求 to 的 producer 的**每一个
  // 输入**的 producer 拓扑下标严格 < from 的 producer 下标。本用例构造
  // to_producer 依赖 from_producer 自身的输出——该输入的 producer(即
  // from_producer)拓扑下标恰等于 from_index,不满足"严格小于",故不豁免
  // (反例:to 依赖 from 的输出 → 报错且图不变,graph.h 头注释"删除或等于支"
  // 一节)。
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* from_producer = graph.create_node("relu", {input}, {MakeFloat32Type({4})}).value();
  Node* to_producer =
      graph.create_node("relu2", {from_producer->output(0)}, {MakeFloat32Type({4})}).value();
  const std::vector<Node*> topo_before = graph.topological_order();

  const Status status = graph.replace_all_uses(from_producer->output(0), to_producer->output(0));
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("topological order invariant violated"), std::string_view::npos);
  EXPECT_NE(status.message().find("relu2"), std::string_view::npos);
  EXPECT_NE(status.message().find("relu"), std::string_view::npos);
  // 违例时不做任何改动:拓扑序保持不变。
  EXPECT_EQ(graph.topological_order(), topo_before);
}

// ---------------------------------------------------------------------------
// swap_node_inputs:正常路径。
// ---------------------------------------------------------------------------

TEST(GraphSwapNodeInputsTest, SwapsTwoInputSlots) {
  Graph graph;
  Value* a = graph.add_graph_input(MakeFloat32Type({4})).value();
  Value* b = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* node = graph.create_node("add", {a, b}, {MakeFloat32Type({4})}).value();
  ASSERT_EQ(node->inputs()[0], a);
  ASSERT_EQ(node->inputs()[1], b);

  const Status status = graph.swap_node_inputs(node, 0, 1);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_EQ(node->inputs()[0], b);
  EXPECT_EQ(node->inputs()[1], a);
}

// ---------------------------------------------------------------------------
// swap_node_inputs:报错路径。
// ---------------------------------------------------------------------------

TEST(GraphSwapNodeInputsTest, OutOfRangeIndexIsRejected) {
  Graph graph;
  Value* a = graph.add_graph_input(MakeFloat32Type({4})).value();
  Value* b = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* node = graph.create_node("add", {a, b}, {MakeFloat32Type({4})}).value();

  const Status status = graph.swap_node_inputs(node, 0, 2);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("out of range"), std::string_view::npos);
  // 拒绝后不改变原有输入顺序。
  EXPECT_EQ(node->inputs()[0], a);
  EXPECT_EQ(node->inputs()[1], b);
}

TEST(GraphSwapNodeInputsTest, EqualIndicesAreRejected) {
  Graph graph;
  Value* a = graph.add_graph_input(MakeFloat32Type({4})).value();
  Value* b = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* node = graph.create_node("add", {a, b}, {MakeFloat32Type({4})}).value();

  const Status status = graph.swap_node_inputs(node, 1, 1);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("must be different indices"), std::string_view::npos);
}

TEST(GraphSwapNodeInputsTest, NodeNotOwnedByThisGraphIsRejected) {
  Graph graph_a;
  Graph graph_b;
  Value* a = graph_b.add_graph_input(MakeFloat32Type({4})).value();
  Value* b = graph_b.add_graph_input(MakeFloat32Type({4})).value();
  Node* foreign_node = graph_b.create_node("add", {a, b}, {MakeFloat32Type({4})}).value();

  const Status status = graph_a.swap_node_inputs(foreign_node, 0, 1);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("does not belong to this graph"), std::string_view::npos);
}

}  // namespace
