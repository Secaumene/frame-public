// 图 IR 文本序列化单测:dump_text 确定性、8 种属性类型的 dump→parse→dump 往返、
// 多输出/零输出节点往返、空图往返、属性按名字典序排列、parse_text 错误路径
// (含 1-based 行号)、M9 layout 尾缀往返(row_major 尾缀 dump→parse→dump 逐
// 字节一致 / kUnknown 图与既有格式逐字节一致,serialization.h 头注释第3a条)。
// 格式细节的唯一权威见 include/frame/ir/serialization.h 头注释(非
// docs/architecture/ir-design.md 第3章正文——正文示例用简写 "f32" 仅作插图,
// 并非字面格式契约)。
#include <cstdint>
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

using frame::cpu_device;
using frame::DType;
using frame::DTypeCode;
using frame::ErrorCode;
using frame::Result;
using frame::Shape;
using frame::ir::AttrValue;
using frame::ir::dump_text;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::parse_text;
using frame::ir::TensorType;
using frame::ir::Value;
using frame::ir::testing::MakeFloat32Type;

TEST(SerializationTest, DumpTextMatchesDocumentedFormatForMlpLikeGraph) {
  // 复刻 ir-design.md 第3章示例的结构(matmul+relu),但按 serialization.h
  // 头注释的权威格式使用 DType::name() 全称 "float32"(而非文档插图简写
  // "f32")。MakeFloat32Type 恒置 layout=kRowMajor(ir_test_helpers.h),故
  // M9 起每个类型后缀均带 ":row_major" 尾缀(serialization.h 头注释第3a条)。
  Graph graph;
  Value* x = graph.add_graph_input(MakeFloat32Type({32, 784})).value();
  Value* w = graph.add_graph_input(MakeFloat32Type({784, 256})).value();
  Node* matmul = graph.create_node("matmul", {x, w}, {MakeFloat32Type({32, 256})}).value();
  Node* relu = graph.create_node("relu", {matmul->output(0)}, {MakeFloat32Type({32, 256})}).value();
  ASSERT_TRUE(graph.mark_output(relu->output(0)).is_ok());

  const std::string expected =
      "%0 = graph_input() : float32[32,784]@cpu:0:row_major\n"
      "%1 = graph_input() : float32[784,256]@cpu:0:row_major\n"
      "%2 = matmul(%0, %1) : float32[32,256]@cpu:0:row_major\n"
      "%3 = relu(%2) : float32[32,256]@cpu:0:row_major\n"
      "graph_output(%3)\n";
  EXPECT_EQ(dump_text(graph), expected);
}

TEST(SerializationTest, DumpTextIsByteForByteDeterministicAcrossRepeatedCalls) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({2, 3})).value();
  Node* node = graph.create_node("relu", {input}, {MakeFloat32Type({2, 3})}).value();
  node->set_attr("axis", AttrValue{int64_t{1}});
  ASSERT_TRUE(graph.mark_output(node->output(0)).is_ok());

  const std::string first = dump_text(graph);
  const std::string second = dump_text(graph);
  EXPECT_EQ(first, second);
}

TEST(SerializationTest, AttributesAppearInDictionaryOrderByName) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({2})).value();
  Node* node = graph.create_node("configurable_op", {input}, {MakeFloat32Type({2})}).value();
  // 故意乱序 set_attr,验证 dump_text 按名字典序而非插入序输出(node.attrs()
  // 底层是 unordered_map,枚举顺序不确定;字典序是序列化层自身的契约)。
  node->set_attr("zeta", AttrValue{int64_t{1}});
  node->set_attr("alpha", AttrValue{int64_t{2}});
  node->set_attr("mid", AttrValue{int64_t{3}});

  const std::string text = dump_text(graph);
  const size_t pos_alpha = text.find("alpha=");
  const size_t pos_mid = text.find("mid=");
  const size_t pos_zeta = text.find("zeta=");
  ASSERT_NE(pos_alpha, std::string::npos);
  ASSERT_NE(pos_mid, std::string::npos);
  ASSERT_NE(pos_zeta, std::string::npos);
  EXPECT_LT(pos_alpha, pos_mid);
  EXPECT_LT(pos_mid, pos_zeta);
}

TEST(SerializationTest, RoundTripPreservesEmptyGraph) {
  const Graph graph;
  EXPECT_EQ(dump_text(graph), "");

  const Result<Graph> parsed = parse_text("");
  ASSERT_TRUE(parsed.is_ok());
  EXPECT_EQ(dump_text(parsed.value()), "");
  EXPECT_TRUE(parsed.value().topological_order().empty());
}

TEST(SerializationTest, RoundTripPreservesAllEightAttributeTypes) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({2})).value();
  Node* node = graph.create_node("configurable_op", {input}, {MakeFloat32Type({2})}).value();
  node->set_attr("a_int", AttrValue{int64_t{-7}});
  node->set_attr("b_double", AttrValue{2.5});
  node->set_attr("c_string", AttrValue{std::string("hello \"world\"\nnext")});
  node->set_attr("d_bool", AttrValue{true});
  node->set_attr("e_int_array", AttrValue{std::vector<int64_t>{1, -2, 3}});
  node->set_attr("f_double_array", AttrValue{std::vector<double>{1.5, -2.25}});
  node->set_attr("g_dtype", AttrValue{DType(DTypeCode::kInt32)});
  node->set_attr("h_shape", AttrValue{Shape({2, 3, 4})});
  ASSERT_TRUE(graph.mark_output(node->output(0)).is_ok());
  ASSERT_EQ(node->attrs().size(), 8u);

  const std::string first = dump_text(graph);
  const Result<Graph> parsed = parse_text(first);
  ASSERT_TRUE(parsed.is_ok());
  const std::string second = dump_text(parsed.value());
  EXPECT_EQ(first, second);
}

TEST(SerializationTest, RoundTripPreservesMultiOutputAndZeroOutputNodes) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  // 零输出节点(如就地写回类算子的简化替身):output_types 为空。
  graph.create_node("no_output_op", {input}, {});
  // 双输出节点。
  Node* split =
      graph.create_node("split", {input}, {MakeFloat32Type({2}), MakeFloat32Type({2})}).value();
  ASSERT_TRUE(graph.mark_output(split->output(0)).is_ok());
  ASSERT_TRUE(graph.mark_output(split->output(1)).is_ok());

  const std::string first = dump_text(graph);
  // 零输出节点行不含 " = " 前缀(serialization.h 头注释第1条)。
  EXPECT_NE(first.find("no_output_op(%0)\n"), std::string::npos);

  const Result<Graph> parsed = parse_text(first);
  ASSERT_TRUE(parsed.is_ok());
  const std::string second = dump_text(parsed.value());
  EXPECT_EQ(first, second);
}

TEST(SerializationTest, ParseTextMalformedLineReportsEnglishErrorWithLineNumber) {
  const Result<Graph> parsed = parse_text("this is not a valid node line\n");
  ASSERT_FALSE(parsed.is_ok());
  EXPECT_EQ(parsed.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(parsed.status().message().find("line 1"), std::string_view::npos);
  EXPECT_NE(parsed.status().message().find("missing '('"), std::string_view::npos);
}

TEST(SerializationTest, ParseTextRejectsNonCanonicalCommaSpacing) {
  // dump_text 恒以 ", "(逗号+空格)分隔多输入;此处故意省略空格构造"非规范
  // 形态",验证 parse_text 仅接受 dump_text 产出的规范形态(serialization.h
  // 头注释:"仅接受 dump_text 产出的规范形态")。
  const std::string malformed_text =
      "%0 = graph_input() : float32[2]@cpu:0\n"
      "%1 = graph_input() : float32[2]@cpu:0\n"
      "%2 = add(%0,%1) : float32[2]@cpu:0\n";  // 第3行 "%0,%1" 缺少规范分隔空格
  const Result<Graph> parsed = parse_text(malformed_text);
  ASSERT_FALSE(parsed.is_ok());
  EXPECT_EQ(parsed.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(parsed.status().message().find("line 3"), std::string_view::npos);
}

TEST(SerializationTest, ParseTextRejectsReferenceToUndefinedValue) {
  const Result<Graph> parsed = parse_text("%0 = relu(%99) : float32[2]@cpu:0\n");
  ASSERT_FALSE(parsed.is_ok());
  EXPECT_EQ(parsed.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(parsed.status().message().find("line 1"), std::string_view::npos);
  EXPECT_NE(parsed.status().message().find("undefined value"), std::string_view::npos);
}

TEST(SerializationTest, ParseTextRejectsDuplicateValueIdDefinition) {
  // 两行各自产出 %0(非规范文本,dump_text 恒产出全局单调递增 id,不会自然
  // 出现此形态),验证 register_value_id 的重复定义拒绝(src/ir/serialization.cpp)。
  const std::string duplicate_id_text =
      "%0 = graph_input() : float32[2]@cpu:0\n"
      "%0 = graph_input() : float32[2]@cpu:0\n";
  const Result<Graph> parsed = parse_text(duplicate_id_text);
  ASSERT_FALSE(parsed.is_ok());
  EXPECT_EQ(parsed.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(parsed.status().message().find("line 2"), std::string_view::npos);
  EXPECT_NE(parsed.status().message().find("duplicate definition of value %"),
            std::string_view::npos);
}

TEST(SerializationTest, IntegerValuedDoubleAttributeDumpsWithDotZeroSuffix) {
  // format_double: std::to_chars(5.0) 产出看起来像整数的 "5",serialization.h
  // 头注释第2条 kDouble 分支要求强制追加 ".0",与 kInt64 的文本表示消歧。
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({2})).value();
  Node* node = graph.create_node("configurable_op", {input}, {MakeFloat32Type({2})}).value();
  node->set_attr("scale", AttrValue{5.0});

  const std::string text = dump_text(graph);
  EXPECT_NE(text.find("{scale=5.0}"), std::string::npos);
}

TEST(SerializationTest, RoundTripPreservesRowMajorLayoutSuffix) {
  // M9 决议点 A + 裁决修订7:layout != kUnknown 时类型后缀追加 ":row_major"
  // 尾段(serialization.h 头注释第3a条)。MakeFloat32Type 恒置 kRowMajor,故
  // dump 结果含该尾缀;往返(dump->parse->dump)须逐字节相同。
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({3})).value();
  Node* node = graph.create_node("relu", {input}, {MakeFloat32Type({3})}).value();
  ASSERT_TRUE(graph.mark_output(node->output(0)).is_ok());

  const std::string first = dump_text(graph);
  EXPECT_NE(first.find(":row_major"), std::string::npos);
  const Result<Graph> parsed = parse_text(first);
  ASSERT_TRUE(parsed.is_ok()) << parsed.status().message();
  const std::string second = dump_text(parsed.value());
  EXPECT_EQ(first, second);
}

TEST(SerializationTest, KUnknownLayoutGraphMatchesPreM9FormatByteForByte) {
  // layout == kUnknown 时不追加任何尾段,产出与 M9 之前逐字节一致的格式
  // (serialization.h 头注释第3a条:向后兼容)。TensorType 默认 layout 即
  // kUnknown(node.h),此处不经 MakeFloat32Type(其恒置 kRowMajor),直接构造。
  Graph graph;
  TensorType type;
  type.dtype = DType::of<float>();
  type.shape = Shape({3});
  type.device = cpu_device();

  Value* input = graph.add_graph_input(type).value();
  Node* node = graph.create_node("relu", {input}, {type}).value();
  ASSERT_TRUE(graph.mark_output(node->output(0)).is_ok());

  const std::string expected =
      "%0 = graph_input() : float32[3]@cpu:0\n"
      "%1 = relu(%0) : float32[3]@cpu:0\n"
      "graph_output(%1)\n";
  EXPECT_EQ(dump_text(graph), expected);

  const Result<Graph> parsed = parse_text(expected);
  ASSERT_TRUE(parsed.is_ok()) << parsed.status().message();
  EXPECT_EQ(dump_text(parsed.value()), expected);
}

TEST(SerializationTest, RoundTripPreservesNonTerminatingDecimalDoubleValue) {
  // 0.1 在二进制浮点下不可精确表示,验证 to_chars/from_chars 最短往返对
  // (而非十进制截断/舍入)使 dump→parse→dump 逐字节相同。
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({2})).value();
  Node* node = graph.create_node("configurable_op", {input}, {MakeFloat32Type({2})}).value();
  node->set_attr("scale", AttrValue{0.1});
  ASSERT_TRUE(graph.mark_output(node->output(0)).is_ok());

  const std::string first = dump_text(graph);
  EXPECT_NE(first.find("scale=0.1"), std::string::npos);
  const Result<Graph> parsed = parse_text(first);
  ASSERT_TRUE(parsed.is_ok());
  const std::string second = dump_text(parsed.value());
  EXPECT_EQ(first, second);
}

}  // namespace
