// fused_elementwise_internal 算子测试(M9,决议点 B 覆盖版):schema 字段
// (src/ops/schemas/fused_elementwise.cpp)+ shape_infer 经
// decode_and_validate_fused_chain(src/ops/fused_elementwise_utils.cpp,note C)
// 暴露的四类报错路径:①arities 与输入数不自洽;②sub-op 未注册;③sub-op 非
// (kElementwise+kFusable)——用已注册的真实算子 "sum"(无任何 trait,见
// tests/cpp/ops/test_op_sum.cpp 头注释)驱动,不新增辅助 op 注册;
// ④"ops"/"arities" 两属性段数不一致(此路径不经 encode_fused_chain——后者的
// FRAME_CHECK 保证调用方 ops.size()==arities.size(),需手工构造 attrs 才能
// 触达该独立校验分支)。cpu kernel 组合调用数值路径见
// tests/cpp/compiler/test_operator_fusion.cpp 的 ARCH-052 数值等价用例(经
// operator_fusion pass 产出真实融合节点后由 runtime::compile 执行,聚焦"融合
// 是否数值等价"而非本文件聚焦的"schema 校验单独触发的报错路径")。
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <vector>

#include <frame/core/dtype.h>
#include <frame/ir/attribute.h>
#include <frame/ops/fused_elementwise_utils.h>
#include <frame/ops/op_registry.h>

#include "elementwise_op_test_helpers.h"

namespace {

using frame::DType;
using frame::ErrorCode;
using frame::ir::AttrValue;
using frame::ops::encode_fused_chain;
using frame::ops::kFusedElementwiseOpName;
using frame::ops::NodeContext;
using frame::ops::OpRegistry;
using frame::ops::OpSchema;
using frame::ops::OpTrait;
using frame::ops::testing::MakeType;

// ---------------------------------------------------------------------------
// 1. schema 断言。
// ---------------------------------------------------------------------------

TEST(FusedElementwiseOpSchemaTest, RegisteredWithVariadicMinCountOneAndElementwiseOnlyTrait) {
  const OpSchema* schema = OpRegistry::instance().find(kFusedElementwiseOpName);
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->name(), kFusedElementwiseOpName);

  // 变长输入(min_count=1),无定长输入段(前置设计:OpSchema 变长输入支持)。
  EXPECT_TRUE(schema->has_variadic_inputs());
  EXPECT_TRUE(schema->inputs().empty());
  EXPECT_EQ(schema->min_input_count(), 1);

  ASSERT_EQ(schema->outputs().size(), 1u);
  EXPECT_EQ(schema->outputs()[0].name, "out");

  ASSERT_EQ(schema->attrs().size(), 2u);
  // attrs() 枚举顺序取决于 builder 调用序(vector,非 unordered_map),schema
  // 注册文件按 "ops" 后 "arities" 声明。
  EXPECT_EQ(schema->attrs()[0].name, "ops");
  EXPECT_EQ(schema->attrs()[0].type, frame::ir::AttrType::kString);
  EXPECT_TRUE(schema->attrs()[0].required);
  EXPECT_EQ(schema->attrs()[1].name, "arities");
  EXPECT_EQ(schema->attrs()[1].type, frame::ir::AttrType::kInt64Array);
  EXPECT_TRUE(schema->attrs()[1].required);

  // 决议点 B:融合产物仅标 kElementwise,不标 kFusable(v0 不做融合节点再
  // 融合)。
  EXPECT_TRUE(schema->has_trait(OpTrait::kElementwise));
  EXPECT_FALSE(schema->has_trait(OpTrait::kFusable));

  EXPECT_NE(schema->shape_infer(), nullptr);
}

// ---------------------------------------------------------------------------
// 2. shape_infer 报错路径(经 decode_and_validate_fused_chain,note C)。
// ---------------------------------------------------------------------------

TEST(FusedElementwiseShapeInferTest, RejectsAritiesInconsistentWithInputCount) {
  const OpSchema* schema = OpRegistry::instance().find(kFusedElementwiseOpName);
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  // ops="add;relu"(2 段),arities=[2,1] => sum=3,computed_input_count =
  // 3 - (2-1) = 2;但下方 ctx.input_types 给 3 个,与 2 不符。
  std::unordered_map<std::string, AttrValue> attrs;
  encode_fused_chain({"add", "relu"}, {2, 1}, attrs);

  NodeContext ctx;
  ctx.op = kFusedElementwiseOpName;
  ctx.input_types = {MakeType(DType::of<float>(), {4}), MakeType(DType::of<float>(), {4}),
                     MakeType(DType::of<float>(), {4})};
  ctx.attrs = &attrs;

  const auto result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("arities are inconsistent with input count"),
            std::string::npos);
  EXPECT_NE(result.status().message().find("sum(arities)=3"), std::string::npos);
  EXPECT_NE(result.status().message().find("expected 3"), std::string::npos);
}

TEST(FusedElementwiseShapeInferTest, RejectsUnregisteredSubOp) {
  const OpSchema* schema = OpRegistry::instance().find(kFusedElementwiseOpName);
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  std::unordered_map<std::string, AttrValue> attrs;
  encode_fused_chain({"nonexistent_fused_test_sub_op"}, {1}, attrs);

  NodeContext ctx;
  ctx.op = kFusedElementwiseOpName;
  ctx.input_types = {MakeType(DType::of<float>(), {4})};
  ctx.attrs = &attrs;

  const auto result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kNotFound);
  EXPECT_NE(result.status().message().find("nonexistent_fused_test_sub_op"), std::string::npos);
  EXPECT_NE(result.status().message().find("is not registered"), std::string::npos);
}

TEST(FusedElementwiseShapeInferTest, RejectsSubOpMissingElementwiseOrFusableTrait) {
  const OpSchema* schema = OpRegistry::instance().find(kFusedElementwiseOpName);
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  // "sum" 是已注册的真实归约算子,不标任何 trait(见
  // src/ops/schemas/reduction.cpp、tests/cpp/ops/test_op_sum.cpp 头注释),
  // 天然满足"已注册但非 kElementwise+kFusable"这一违例条件,无需新注册辅助
  // op(REUSE-001:先搜后写,复用已有产物算子)。
  std::unordered_map<std::string, AttrValue> attrs;
  encode_fused_chain({"sum"}, {1}, attrs);

  NodeContext ctx;
  ctx.op = kFusedElementwiseOpName;
  ctx.input_types = {MakeType(DType::of<float>(), {4})};
  ctx.attrs = &attrs;

  const auto result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("'sum'"), std::string::npos);
  EXPECT_NE(result.status().message().find("must have both kElementwise and kFusable traits"),
            std::string::npos);
}

TEST(FusedElementwiseShapeInferTest, RejectsMismatchedOpsAndAritiesSegmentCounts) {
  const OpSchema* schema = OpRegistry::instance().find(kFusedElementwiseOpName);
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  // 手工构造 attrs(不经 encode_fused_chain——后者的 FRAME_CHECK 强制调用方
  // ops.size()==arities.size(),此分支验证 decode 侧对"attrs 本身已不自洽"
  // 的独立防御,故绕过 encode 直接拼装 kString/kInt64Array):"ops" 2 段,
  // "arities" 3 个元素。
  std::unordered_map<std::string, AttrValue> attrs;
  attrs["ops"] = AttrValue{std::string("add;relu")};
  attrs["arities"] = AttrValue{std::vector<int64_t>{1, 1, 1}};

  NodeContext ctx;
  ctx.op = kFusedElementwiseOpName;
  ctx.input_types = {MakeType(DType::of<float>(), {4})};
  ctx.attrs = &attrs;

  const auto result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("2 segment(s)"), std::string::npos);
  EXPECT_NE(result.status().message().find("3 element(s)"), std::string::npos);
  EXPECT_NE(result.status().message().find("expected equal counts"), std::string::npos);
}

}  // namespace
