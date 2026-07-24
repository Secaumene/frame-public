// 节点属性单测:Node::set_attr/find_attr/attr<T> 存取语义、attrs() 枚举、
// attr_type_of 对 AttrValue 8 种备选项的映射(ARCH-020)、attr_type_name 英文名。
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <frame/ir/attribute.h>
#include <frame/ir/node.h>

namespace {

using frame::DType;
using frame::DTypeCode;
using frame::Shape;
using frame::ir::attr_type_name;
using frame::ir::attr_type_of;
using frame::ir::AttrType;
using frame::ir::AttrValue;
using frame::ir::Node;

TEST(NodeAttributeTest, SetAttrThenFindAttrReturnsStoredValue) {
  Node node("op");
  node.set_attr("axis", AttrValue{int64_t{42}});
  const AttrValue* value = node.find_attr("axis");
  ASSERT_NE(value, nullptr);
  ASSERT_TRUE(std::holds_alternative<int64_t>(*value));
  EXPECT_EQ(std::get<int64_t>(*value), 42);
}

TEST(NodeAttributeTest, FindAttrOnMissingNameReturnsNullptr) {
  const Node node("op");
  EXPECT_EQ(node.find_attr("missing"), nullptr);
}

TEST(NodeAttributeTest, SetAttrOverwritesExistingValueByName) {
  Node node("op");
  node.set_attr("axis", AttrValue{int64_t{1}});
  node.set_attr("axis", AttrValue{int64_t{2}});
  ASSERT_EQ(node.attrs().size(), 1u);
  const int64_t* value = node.attr<int64_t>("axis");
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(*value, 2);
}

TEST(NodeAttributeTest, AttrTemplateHitReturnsPointerToTypedValue) {
  Node node("op");
  node.set_attr("name", AttrValue{std::string("relu")});
  const std::string* value = node.attr<std::string>("name");
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(*value, "relu");
}

TEST(NodeAttributeTest, AttrTemplateOnMissingNameReturnsNullptr) {
  const Node node("op");
  EXPECT_EQ(node.attr<int64_t>("missing"), nullptr);
}

TEST(NodeAttributeTest, AttrTemplateOnTypeMismatchReturnsNullptr) {
  Node node("op");
  node.set_attr("axis", AttrValue{int64_t{1}});
  // 存的是 int64,按 double 取回应为 nullptr(不做隐式数值转换)。
  EXPECT_EQ(node.attr<double>("axis"), nullptr);
}

TEST(NodeAttributeTest, AttrsEnumeratesAllSetAttributesRegardlessOfOrder) {
  Node node("op");
  node.set_attr("a", AttrValue{int64_t{1}});
  node.set_attr("b", AttrValue{true});
  node.set_attr("c", AttrValue{std::string("x")});
  ASSERT_EQ(node.attrs().size(), 3u);
  EXPECT_NE(node.find_attr("a"), nullptr);
  EXPECT_NE(node.find_attr("b"), nullptr);
  EXPECT_NE(node.find_attr("c"), nullptr);
}

// ---------------------------------------------------------------------------
// attr_type_of / attr_type_name:AttrValue 的 8 种备选项(ARCH-020)逐一映射到
// 对应 AttrType 与英文名,数据源是 attribute.h/attribute.cpp 声明顺序。
// ---------------------------------------------------------------------------

struct AttrTypeExpectation {
  AttrValue value;
  AttrType type;
  std::string_view name;
};

std::vector<AttrTypeExpectation> AllAttrTypeExpectations() {
  return {
      {AttrValue{int64_t{-3}}, AttrType::kInt64, "int64"},
      {AttrValue{1.5}, AttrType::kDouble, "double"},
      {AttrValue{std::string("s")}, AttrType::kString, "string"},
      {AttrValue{true}, AttrType::kBool, "bool"},
      {AttrValue{std::vector<int64_t>{1, 2, 3}}, AttrType::kInt64Array, "int64_array"},
      {AttrValue{std::vector<double>{1.0, 2.0}}, AttrType::kDoubleArray, "double_array"},
      {AttrValue{DType(DTypeCode::kFloat32)}, AttrType::kDType, "dtype"},
      {AttrValue{Shape({2, 3})}, AttrType::kShape, "shape"},
  };
}

TEST(AttrTypeOfTest, MapsEachVariantAlternativeToMatchingAttrType) {
  for (const AttrTypeExpectation& expectation : AllAttrTypeExpectations()) {
    SCOPED_TRACE(expectation.name);
    EXPECT_EQ(attr_type_of(expectation.value), expectation.type);
  }
}

TEST(AttrTypeNameTest, ReturnsExpectedEnglishNameForEachAttrType) {
  for (const AttrTypeExpectation& expectation : AllAttrTypeExpectations()) {
    SCOPED_TRACE(expectation.name);
    EXPECT_EQ(attr_type_name(expectation.type), expectation.name);
  }
}

}  // namespace
