// OpSchema builder 单测:input/output/attr/trait/shape_infer/decomposition 链式
// 调用后全字段可读回(builder 方法均返回 *this,链式作用于同一对象,非产出副本);
// NodeContext::attr<T> 命中/缺失/类型不符三路径 + attrs 指针为空(无属性)时的
// nullptr 语义;M9 前置设计 variadic_input() 字段读回 + 两条硬约束(至多一组/
// 须尾随)的 builder 期 fail-fast 死亡测试。OpSchema 公开默认构造,本文件全程
// 构造局部实例,不触碰 OpRegistry 单例 —— 与经 OpRegistry::register_op/
// FRAME_REGISTER_OP 走"名字注入"路径的 tests/cpp/ops/test_op_registry.cpp
// 职责分离。
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <vector>

#include <frame/core/shape.h>
#include <frame/ir/attribute.h>
#include <frame/ir/graph.h>
#include <frame/ops/op_schema.h>

namespace {

using frame::Result;
using frame::Shape;
using frame::ir::AttrType;
using frame::ir::AttrValue;
using frame::ir::Graph;
using frame::ops::NodeContext;
using frame::ops::OpSchema;
using frame::ops::OpTrait;

Result<std::vector<Shape>> DummyShapeInfer(const NodeContext&) { return std::vector<Shape>{}; }

Result<Graph> DummyDecompose(const NodeContext&) { return Graph(); }

Result<Graph> DummyGradient(const NodeContext&) { return Graph(); }

TEST(OpSchemaTest, DefaultConstructedSchemaHasEmptyFieldsAndNullFunctionPointers) {
  const OpSchema schema;
  EXPECT_TRUE(schema.name().empty());
  EXPECT_TRUE(schema.inputs().empty());
  EXPECT_TRUE(schema.outputs().empty());
  EXPECT_TRUE(schema.attrs().empty());
  EXPECT_FALSE(schema.has_trait(OpTrait::kElementwise));
  EXPECT_FALSE(schema.has_trait(OpTrait::kFusable));
  EXPECT_FALSE(schema.has_trait(OpTrait::kHasSideEffect));
  EXPECT_FALSE(schema.has_trait(OpTrait::kCommutative));
  EXPECT_EQ(schema.shape_infer(), nullptr);
  EXPECT_EQ(schema.decomposition(), nullptr);
  EXPECT_EQ(schema.gradient(), nullptr);  // M17
}

TEST(OpSchemaTest, BuilderChainPopulatesAllFieldsForReadback) {
  OpSchema schema;
  OpSchema& chained = schema.input("lhs", "left operand")
                          .input("rhs", "right operand")
                          .output("out", "result tensor")
                          .attr("axis", AttrType::kInt64, /*required=*/true)
                          .attr("keepdims", AttrType::kBool, /*required=*/false)
                          .trait(OpTrait::kElementwise)
                          .trait(OpTrait::kCommutative)
                          .shape_infer(DummyShapeInfer)
                          .decomposition(DummyDecompose)
                          .gradient(DummyGradient);

  // builder 方法均返回 *this(链式调用作用于同一对象,而非产出新副本)。
  EXPECT_EQ(&chained, &schema);

  ASSERT_EQ(schema.inputs().size(), 2u);
  EXPECT_EQ(schema.inputs()[0].name, "lhs");
  EXPECT_EQ(schema.inputs()[0].doc, "left operand");
  EXPECT_EQ(schema.inputs()[1].name, "rhs");
  EXPECT_EQ(schema.inputs()[1].doc, "right operand");

  ASSERT_EQ(schema.outputs().size(), 1u);
  EXPECT_EQ(schema.outputs()[0].name, "out");
  EXPECT_EQ(schema.outputs()[0].doc, "result tensor");

  ASSERT_EQ(schema.attrs().size(), 2u);
  EXPECT_EQ(schema.attrs()[0].name, "axis");
  EXPECT_EQ(schema.attrs()[0].type, AttrType::kInt64);
  EXPECT_TRUE(schema.attrs()[0].required);
  EXPECT_EQ(schema.attrs()[1].name, "keepdims");
  EXPECT_EQ(schema.attrs()[1].type, AttrType::kBool);
  EXPECT_FALSE(schema.attrs()[1].required);

  EXPECT_TRUE(schema.has_trait(OpTrait::kElementwise));
  EXPECT_TRUE(schema.has_trait(OpTrait::kCommutative));
  EXPECT_FALSE(schema.has_trait(OpTrait::kFusable));
  EXPECT_FALSE(schema.has_trait(OpTrait::kHasSideEffect));

  EXPECT_EQ(schema.shape_infer(), DummyShapeInfer);
  EXPECT_EQ(schema.decomposition(), DummyDecompose);
  EXPECT_EQ(schema.gradient(), DummyGradient);  // M17
}

// M17(ARCH-062):带 kHasSideEffect trait 的算子不可微,gradient() 在 builder
// 期即时 fail-fast(与 M9 variadic_input 两条硬约束同一纪律、同一份
// fatal_registration_error 实现,REUSE-002)。校验顺序敏感:必须先
// trait(OpTrait::kHasSideEffect) 再调用 gradient(),否则 has_trait() 尚未置位、
// 不会触发本分支(见 op_schema.h 对应声明处的顺序敏感说明)。
TEST(OpSchemaGradientTest, GradientCalledAfterHasSideEffectTraitAborts) {
  OpSchema schema;
  schema.trait(OpTrait::kHasSideEffect);

  EXPECT_DEATH(
      { schema.gradient(DummyGradient); },
      "gradient\\(\\) rejected: op has OpTrait::kHasSideEffect");
}

TEST(OpSchemaVariadicInputTest, BuilderChainPopulatesVariadicFieldsForReadback) {
  OpSchema schema;
  OpSchema& chained =
      schema.input("fixed", "fixed input").variadic_input("xs", "variadic inputs", /*min_count=*/2);
  EXPECT_EQ(&chained, &schema);

  EXPECT_TRUE(schema.has_variadic_inputs());
  ASSERT_EQ(schema.inputs().size(), 1u);  // 变长组不计入定长 inputs()
  EXPECT_EQ(schema.inputs()[0].name, "fixed");
  EXPECT_EQ(schema.min_input_count(), 3);  // 定长数(1) + min_count(2)
}

TEST(OpSchemaVariadicInputTest, DefaultConstructedSchemaHasNoVariadicGroup) {
  const OpSchema schema;
  EXPECT_FALSE(schema.has_variadic_inputs());
}

// M9 note A:变长组两条硬约束(至多一组 / 须尾随全部定长 input())均在 builder
// 方法内即时 fail-fast(经 fatal_registration_error,与 OpRegistry::register_op
// 共用同一份 fprintf+abort 实现,REUSE-002),参照 tests/cpp/ops/test_ops_stub.cpp
// 的重名注册死亡测试写法:局部 OpSchema 实例(不触碰 OpRegistry 单例),先在
// "父进程"完成一次合法调用,EXPECT_DEATH(fast/fork 风格)内的"子进程"里触发
// 违例。诊断串取自 op_schema.cpp 对应分支的 reason 文本,与其余 builder fatal
// (OpRegistry::register_op 三类)可区分。

TEST(OpSchemaVariadicInputTest, SecondVariadicInputCallAborts) {
  OpSchema schema;
  schema.variadic_input("xs", "variadic inputs", /*min_count=*/0);

  EXPECT_DEATH(
      { schema.variadic_input("ys", "second variadic group", /*min_count=*/0); },
      "variadic_input\\(\\) called more than once");
}

TEST(OpSchemaVariadicInputTest, InputCallAfterVariadicInputAborts) {
  OpSchema schema;
  schema.variadic_input("xs", "variadic inputs", /*min_count=*/0);

  EXPECT_DEATH(
      { schema.input("trailing", "input declared after variadic group"); },
      "input\\(\\) called after variadic_input\\(\\)");
}

TEST(NodeContextAttrTest, HitReturnsPointerToStoredValue) {
  std::unordered_map<std::string, AttrValue> attrs;
  attrs.emplace("axis", AttrValue{int64_t{2}});
  NodeContext ctx;
  ctx.op = "test_op";
  ctx.attrs = &attrs;

  const int64_t* value = ctx.attr<int64_t>("axis");
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(*value, 2);
}

TEST(NodeContextAttrTest, MissingNameReturnsNullptr) {
  std::unordered_map<std::string, AttrValue> attrs;
  attrs.emplace("axis", AttrValue{int64_t{2}});
  NodeContext ctx;
  ctx.op = "test_op";
  ctx.attrs = &attrs;

  EXPECT_EQ(ctx.attr<int64_t>("missing"), nullptr);
}

TEST(NodeContextAttrTest, TypeMismatchReturnsNullptr) {
  std::unordered_map<std::string, AttrValue> attrs;
  attrs.emplace("axis", AttrValue{int64_t{2}});
  NodeContext ctx;
  ctx.op = "test_op";
  ctx.attrs = &attrs;

  // 存的是 int64,按 double 取回应为 nullptr(不做隐式数值转换,同 Node::attr<T> 语义)。
  EXPECT_EQ(ctx.attr<double>("axis"), nullptr);
}

TEST(NodeContextAttrTest, NullAttrsPointerMeansNoAttributesAndReturnsNullptr) {
  NodeContext ctx;
  ctx.op = "test_op";
  ASSERT_EQ(ctx.attrs, nullptr);  // 默认值即"无属性"(op_schema.h 头注释:可空 = 无属性)
  EXPECT_EQ(ctx.attr<int64_t>("axis"), nullptr);
}

}  // namespace
