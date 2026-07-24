// Graph 构图 API 单测:create_node/add_graph_input/mark_output 正常路径、
// topological_order 与插入顺序一致、跨图 Value 引用被拒、kGraphInputOp 保留名
// 被拒。全程纯主机内存,不依赖任何后端。
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
using frame::ir::kGraphInputOp;
using frame::ir::kGraphOutputMarker;
using frame::ir::Node;
using frame::ir::Value;
using frame::ir::testing::MakeFloat32Type;

TEST(GraphConstructionTest, CreateNodeAddGraphInputMarkOutputHappyPath) {
  Graph graph("mlp");
  EXPECT_EQ(graph.name(), "mlp");

  const Result<Value*> input_result = graph.add_graph_input(MakeFloat32Type({2, 3}));
  ASSERT_TRUE(input_result.is_ok());
  Value* input = input_result.value();
  ASSERT_NE(input, nullptr);
  ASSERT_NE(input->producer(), nullptr);
  EXPECT_EQ(input->producer()->op(), kGraphInputOp);
  ASSERT_EQ(graph.inputs().size(), 1u);
  EXPECT_EQ(graph.inputs()[0], input);

  const Result<Node*> node_result = graph.create_node("relu", {input}, {MakeFloat32Type({2, 3})});
  ASSERT_TRUE(node_result.is_ok());
  Node* node = node_result.value();
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->op(), "relu");
  ASSERT_EQ(node->inputs().size(), 1u);
  EXPECT_EQ(node->inputs()[0], input);
  ASSERT_EQ(node->outputs().size(), 1u);
  EXPECT_EQ(node->outputs()[0].producer(), node);
  EXPECT_EQ(node->outputs()[0].output_index(), 0);

  const Status mark_status = graph.mark_output(node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  ASSERT_EQ(graph.outputs().size(), 1u);
  EXPECT_EQ(graph.outputs()[0], node->output(0));
}

TEST(GraphConstructionTest, NodeOutputReturnsNullptrForOutOfRangeIndex) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* node = graph.create_node("relu", {input}, {MakeFloat32Type({4})}).value();
  // node 只有 1 个输出(index 0 合法);负数与越界正数均须返回 nullptr
  // (include/frame/ir/node.h::Node::output 头注释)。
  EXPECT_EQ(node->output(-1), nullptr);
  EXPECT_EQ(node->output(1), nullptr);
  EXPECT_NE(node->output(0), nullptr);  // 合法路径对照,证明越界返回并非恒为
                                        // nullptr 的巧合
}

TEST(GraphConstructionTest, MarkOutputByNodeAndIndexRejectsOutOfRangeIndex) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* node = graph.create_node("relu", {input}, {MakeFloat32Type({4})}).value();

  // node 只有 1 个输出,index 1 越界:mark_output(Node*, int32_t) 内部经
  // Node::output(index) 取得 nullptr 后转为 Status 错误(src/ir/graph.cpp),
  // 消息含违例的 index 值。
  const Status status = graph.mark_output(node, 1);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("is out of range for node"), std::string_view::npos);
  EXPECT_NE(status.message().find('1'), std::string_view::npos);
  EXPECT_TRUE(graph.outputs().empty());  // 拒绝时不产生副作用

  // 合法路径对照:index 0 应成功登记。
  const Status ok_status = graph.mark_output(node, 0);
  EXPECT_TRUE(ok_status.is_ok());
  ASSERT_EQ(graph.outputs().size(), 1u);
  EXPECT_EQ(graph.outputs()[0], node->output(0));
}

TEST(GraphConstructionTest, MarkOutputSameValueTwiceAppendsTwoEntries) {
  // mark_output 头注释:"同一 Value 可重复登记(每次调用各占序列化中的一行)"。
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  ASSERT_TRUE(graph.mark_output(input).is_ok());
  ASSERT_TRUE(graph.mark_output(input).is_ok());
  ASSERT_EQ(graph.outputs().size(), 2u);
  EXPECT_EQ(graph.outputs()[0], input);
  EXPECT_EQ(graph.outputs()[1], input);
}

TEST(GraphConstructionTest, TopologicalOrderReflectsInsertionOrder) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* first = graph.create_node("op_a", {input}, {MakeFloat32Type({4})}).value();
  Node* second = graph.create_node("op_b", {first->output(0)}, {MakeFloat32Type({4})}).value();
  Node* third = graph.create_node("op_c", {}, {}).value();
  ASSERT_NE(third, nullptr);

  const std::vector<Node*>& topo = graph.topological_order();
  ASSERT_EQ(topo.size(), 4u);
  EXPECT_EQ(topo[0], input->producer());
  EXPECT_EQ(topo[1], first);
  EXPECT_EQ(topo[2], second);
  EXPECT_EQ(topo[3], third);
}

TEST(GraphConstructionTest, CreateNodeRejectsValueFromAnotherGraph) {
  Graph graph_a;
  Graph graph_b;
  Value* input_from_b = graph_b.add_graph_input(MakeFloat32Type({4})).value();

  const Result<Node*> result = graph_a.create_node("relu", {input_from_b}, {MakeFloat32Type({4})});
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("does not belong to this graph"),
            std::string_view::npos);
  // 拒绝时图内不应留下半成品节点。
  EXPECT_TRUE(graph_a.topological_order().empty());
}

TEST(GraphConstructionTest, CreateNodeRejectsReservedGraphInputOpName) {
  Graph graph;
  const Result<Node*> result = graph.create_node(std::string(kGraphInputOp), {}, {});
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("reserved"), std::string_view::npos);
  EXPECT_TRUE(graph.topological_order().empty());
}

TEST(GraphConstructionTest, CreateNodeRejectsReservedGraphOutputMarkerOpName) {
  // "graph_output" 是序列化层图输出标记的保留名(见 graph.h 头注释),create_node
  // 必须拒绝,消息含固定短语(src/ir/graph.cpp create_node 第二个保留名分支)。
  Graph graph;
  const Result<Node*> result = graph.create_node(std::string(kGraphOutputMarker), {}, {});
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("reserved as the serialization graph output marker"),
            std::string_view::npos);
  EXPECT_TRUE(graph.topological_order().empty());
}

TEST(GraphConstructionTest, CreateNodeRejectsOpNamesViolatingCharset) {
  // 字符集校验 `^[a-z][a-z0-9_]*$`(src/ir/graph.cpp::matches_op_name_charset),
  // 逐一覆盖:空串、大写字母、含空格、含 '%'、数字开头。
  const std::vector<std::string> invalid_op_names = {"", "Add", "my op", "op%", "1op"};
  for (const std::string& op : invalid_op_names) {
    SCOPED_TRACE(op);
    Graph graph;
    const Result<Node*> result = graph.create_node(op, {}, {});
    ASSERT_FALSE(result.is_ok());
    EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
    EXPECT_NE(result.status().message().find("must match ^[a-z][a-z0-9_]*$"),
              std::string_view::npos);
    if (op.empty()) {
      // 空串是任意字符串的子串,直接断言"消息含违例名"无区分度;改按
      // create_node 错误消息的实际拼接文本断言(见 src/ir/graph.cpp)。
      EXPECT_NE(result.status().message().find("op name '' is invalid"), std::string_view::npos);
    } else {
      EXPECT_NE(result.status().message().find(op), std::string_view::npos);
    }
    EXPECT_TRUE(graph.topological_order().empty());
  }
}

}  // namespace
