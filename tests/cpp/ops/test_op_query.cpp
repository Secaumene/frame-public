// make_op_query 单测:op_registered 命中/未命中;check_schema 覆盖输入计数不符、
// 变长输入计数不足/达标(=min 与 >min,M9 前置设计:OpSchema 变长输入支持)、
// 输出计数不符、必需属性缺失、属性类型不符、未知属性拒绝五类违例,断言错误消息
// 不带 "V4: " 前缀(include/frame/ops/op_registry.h 头注释:前缀纪律由
// Graph::verify 统一施加,check_schema 自身不得重复加前缀,否则双重前缀破坏
// golden 文本对齐);再经 Graph::verify(query) 断言消息中 "V4: " 前缀出现且仅
// 出现一次(不双重)。全程纯主机内存,不依赖任何已注册后端(TensorType::device
// 只是数据字段,同 tests/cpp/ir/ir_test_helpers.h)。ops 层注册的 op 名跨全体
// tests/cpp/ops/ 测试文件保持进程级唯一,本文件以 "test_query_" 前缀区分。
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <string_view>

#include <frame/ir/attribute.h>
#include <frame/ir/graph.h>
#include <frame/ops/op_registry.h>

#include "../ir/ir_test_helpers.h"

FRAME_REGISTER_OP("test_query_binary_op")
    .input("lhs", "left operand")
    .input("rhs", "right operand")
    .output("out", "result");

FRAME_REGISTER_OP("test_query_attr_op")
    .input("x", "input tensor")
    .output("y", "output tensor")
    .attr("axis", frame::ir::AttrType::kInt64, /*required=*/true);

// M9 前置设计:OpSchema 变长输入支持(note A/B)。0 个定长输入 + 尾随变长组
// (min_count=2) => min_input_count()=2。用于驱动 check_schema 的变长分支
// (op_registry.cpp make_op_query,V4)。
FRAME_REGISTER_OP("test_query_variadic_op")
    .variadic_input("xs", "variadic inputs", /*min_count=*/2)
    .output("out", "result");

namespace {

using frame::ErrorCode;
using frame::Status;
using frame::ir::AttrValue;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::OpQuery;
using frame::ir::Value;
using frame::ir::testing::MakeFloat32Type;

// 统计 haystack 内 needle 的非重叠出现次数,用于校验 Graph::verify 不会对
// check_schema 已产出的消息重复施加 "V4: " 前缀。
size_t CountOccurrences(std::string_view haystack, std::string_view needle) {
  size_t count = 0;
  size_t pos = 0;
  while ((pos = haystack.find(needle, pos)) != std::string_view::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

TEST(MakeOpQueryTest, OpRegisteredReflectsRegistryState) {
  const OpQuery query = frame::ops::make_op_query();
  EXPECT_TRUE(query.op_registered("test_query_binary_op"));
  EXPECT_FALSE(query.op_registered("test_query_never_registered_xyz"));
}

TEST(MakeOpQueryTest, CheckSchemaRejectsInputCountMismatchWithoutV4Prefix) {
  Graph graph;
  Value* lhs = graph.add_graph_input(MakeFloat32Type({4})).value();
  // schema 声明 2 输入(lhs/rhs),这里只给 1 个。
  Node* node = graph.create_node("test_query_binary_op", {lhs}, {MakeFloat32Type({4})}).value();

  const OpQuery query = frame::ops::make_op_query();
  const Status status = query.check_schema(*node);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("expects 2 input(s), got 1"), std::string_view::npos);
  EXPECT_EQ(status.message().find("V4: "), std::string_view::npos);
}

TEST(MakeOpQueryTest, CheckSchemaRejectsVariadicInputCountBelowMinimumWithoutV4Prefix) {
  Graph graph;
  Value* a = graph.add_graph_input(MakeFloat32Type({4})).value();
  // schema 声明变长组 min_count=2(定长部分为空),这里只给 1 个输入,低于
  // min_input_count()=2。
  Node* node = graph.create_node("test_query_variadic_op", {a}, {MakeFloat32Type({4})}).value();

  const OpQuery query = frame::ops::make_op_query();
  const Status status = query.check_schema(*node);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("test_query_variadic_op"), std::string_view::npos);
  EXPECT_NE(status.message().find("expects at least 2 input(s), got 1"), std::string_view::npos);
  EXPECT_EQ(status.message().find("V4: "), std::string_view::npos);
}

TEST(MakeOpQueryTest, CheckSchemaAcceptsVariadicInputCountExactlyAtMinimum) {
  Graph graph;
  Value* a = graph.add_graph_input(MakeFloat32Type({4})).value();
  Value* b = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* node = graph.create_node("test_query_variadic_op", {a, b}, {MakeFloat32Type({4})}).value();

  const OpQuery query = frame::ops::make_op_query();
  EXPECT_TRUE(query.check_schema(*node).is_ok());
}

TEST(MakeOpQueryTest, CheckSchemaAcceptsVariadicInputCountAboveMinimum) {
  Graph graph;
  Value* a = graph.add_graph_input(MakeFloat32Type({4})).value();
  Value* b = graph.add_graph_input(MakeFloat32Type({4})).value();
  Value* c = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* node =
      graph.create_node("test_query_variadic_op", {a, b, c}, {MakeFloat32Type({4})}).value();

  const OpQuery query = frame::ops::make_op_query();
  EXPECT_TRUE(query.check_schema(*node).is_ok());
}

TEST(MakeOpQueryTest, CheckSchemaRejectsOutputCountMismatchWithoutV4Prefix) {
  Graph graph;
  Value* lhs = graph.add_graph_input(MakeFloat32Type({4})).value();
  Value* rhs = graph.add_graph_input(MakeFloat32Type({4})).value();
  // 输入数正确(2 个),但输出给了 2 个(schema 声明恰 1 个)。
  Node* node = graph
                   .create_node("test_query_binary_op", {lhs, rhs},
                                {MakeFloat32Type({4}), MakeFloat32Type({4})})
                   .value();

  const OpQuery query = frame::ops::make_op_query();
  const Status status = query.check_schema(*node);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("expects 1 output(s), got 2"), std::string_view::npos);
  EXPECT_EQ(status.message().find("V4: "), std::string_view::npos);
}

TEST(MakeOpQueryTest, CheckSchemaRejectsMissingRequiredAttrWithoutV4Prefix) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  // 未设置必需属性 axis。
  Node* node = graph.create_node("test_query_attr_op", {input}, {MakeFloat32Type({4})}).value();

  const OpQuery query = frame::ops::make_op_query();
  const Status status = query.check_schema(*node);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("missing required attribute 'axis'"), std::string_view::npos);
  EXPECT_EQ(status.message().find("V4: "), std::string_view::npos);
}

TEST(MakeOpQueryTest, CheckSchemaRejectsAttrTypeMismatchWithoutV4Prefix) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* node = graph.create_node("test_query_attr_op", {input}, {MakeFloat32Type({4})}).value();
  node->set_attr("axis", AttrValue{1.5});  // schema 声明 kInt64,这里给 double

  const OpQuery query = frame::ops::make_op_query();
  const Status status = query.check_schema(*node);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("attribute 'axis' has type double, expected int64"),
            std::string_view::npos);
  EXPECT_EQ(status.message().find("V4: "), std::string_view::npos);
}

TEST(MakeOpQueryTest, CheckSchemaRejectsUndeclaredAttrWithoutV4Prefix) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* node = graph.create_node("test_query_attr_op", {input}, {MakeFloat32Type({4})}).value();
  node->set_attr("axis", AttrValue{int64_t{0}});
  node->set_attr("extra", AttrValue{int64_t{1}});  // schema 未声明该属性

  const OpQuery query = frame::ops::make_op_query();
  const Status status = query.check_schema(*node);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("undeclared attribute 'extra'"), std::string_view::npos);
  EXPECT_EQ(status.message().find("V4: "), std::string_view::npos);
}

TEST(MakeOpQueryTest, GraphVerifyAddsExactlyOneV4PrefixToCheckSchemaError) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  // 未设置必需属性 axis,复用 check_schema 缺失必需属性这一违例路径驱动
  // Graph::verify(不需要该节点指针,仅需其副作用体现在图内)。
  ASSERT_TRUE(graph.create_node("test_query_attr_op", {input}, {MakeFloat32Type({4})}).is_ok());

  const OpQuery query = frame::ops::make_op_query();
  const Status status = graph.verify(query);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_EQ(CountOccurrences(status.message(), "V4: "), 1u);
  EXPECT_NE(status.message().find("missing required attribute 'axis'"), std::string_view::npos);
}

}  // namespace
