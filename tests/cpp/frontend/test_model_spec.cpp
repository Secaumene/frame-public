// frontend::validate 单测(docs/architecture/frontend-dsl.md 第 3 节
// FE-002~005)。每条规则至少一个否定用例(断言错误消息含违例字段名)+ 一个
// 合法 spec 通过用例。FE-001(schema_version 校验)属 JSON 层职责
// (tools/frame_dslc,ADR-0018),不在本库/本文件范围。

#include <gtest/gtest.h>
#include <string_view>
#include <vector>

#include <frame/core/status.h>
#include <frame/frontend/model_spec.h>

#include "tiny_mlp_spec_helper.h"

namespace {

using frame::ErrorCode;
using frame::Status;
using frame::frontend::Activation;
using frame::frontend::InitKind;
using frame::frontend::ModelSpec;
using frame::frontend::TensorDataSpec;
using frame::frontend::validate;
using frame::frontend::testing::make_tiny_mlp_spec;

// ---------------------------------------------------------------------------
// 合法 spec 通过用例。
// ---------------------------------------------------------------------------

TEST(ModelSpecValidateTest, AcceptsValidTinyMlpSpec) {
  const ModelSpec spec = make_tiny_mlp_spec();
  const Status status = validate(spec);
  EXPECT_TRUE(status.is_ok()) << status.message();
}

// ---------------------------------------------------------------------------
// FE-002:名字引用闭包(layers[].input / loss.prediction / 名字唯一非空)。
// ---------------------------------------------------------------------------

TEST(ModelSpecValidateTest, Fe002RejectsUndefinedLayerInputReference) {
  ModelSpec spec = make_tiny_mlp_spec();
  spec.layers[0].input = "does_not_exist";

  const Status status = validate(spec);
  ASSERT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("layers['layer0'].input"), std::string_view::npos)
      << status.message();
  EXPECT_NE(status.message().find("does_not_exist"), std::string_view::npos) << status.message();
}

TEST(ModelSpecValidateTest, Fe002RejectsUndefinedLossPredictionReference) {
  ModelSpec spec = make_tiny_mlp_spec();
  spec.loss.prediction = "nonexistent_layer";

  const Status status = validate(spec);
  ASSERT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("loss.prediction"), std::string_view::npos) << status.message();
  EXPECT_NE(status.message().find("nonexistent_layer"), std::string_view::npos) << status.message();
}

TEST(ModelSpecValidateTest, Fe002RejectsDuplicateNameAcrossInputAndLayer) {
  ModelSpec spec = make_tiny_mlp_spec();
  spec.layers[0].name = "x";  // 与 inputs[0].name 撞名。

  const Status status = validate(spec);
  ASSERT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("duplicate name"), std::string_view::npos) << status.message();
  EXPECT_NE(status.message().find("'x'"), std::string_view::npos) << status.message();
}

TEST(ModelSpecValidateTest, Fe002RejectsEmptyLayerName) {
  ModelSpec spec = make_tiny_mlp_spec();
  spec.layers[0].name = "";

  const Status status = validate(spec);
  ASSERT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("layers[].name"), std::string_view::npos) << status.message();
}

// ---------------------------------------------------------------------------
// FE-003:形状链一致(layer 输入末维/bias_shape/loss.target_shape)。
// ---------------------------------------------------------------------------

TEST(ModelSpecValidateTest, Fe003RejectsMismatchedInputWeightShape) {
  ModelSpec spec = make_tiny_mlp_spec();
  spec.layers[0].weight_shape = {5, 8};  // x 末维是 4,与 weight_shape[0]=5 不一致。

  const Status status = validate(spec);
  ASSERT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("layers['layer0']"), std::string_view::npos) << status.message();
  EXPECT_NE(status.message().find("weight_shape[0]"), std::string_view::npos) << status.message();
}

TEST(ModelSpecValidateTest, Fe003RejectsMismatchedBiasShape) {
  ModelSpec spec = make_tiny_mlp_spec();
  spec.layers[0].bias_shape = std::vector<int64_t>{8, 4};  // 应为 [batch=8, out=8]。

  const Status status = validate(spec);
  ASSERT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("layers['layer0'].bias_shape"), std::string_view::npos)
      << status.message();
}

TEST(ModelSpecValidateTest, Fe003RejectsMismatchedLossTargetShape) {
  ModelSpec spec = make_tiny_mlp_spec();
  spec.loss.target_shape = {8, 2};  // layer1 输出是 [8,1]。

  const Status status = validate(spec);
  ASSERT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("loss.target_shape"), std::string_view::npos) << status.message();
  EXPECT_NE(status.message().find("layer1"), std::string_view::npos) << status.message();
}

TEST(ModelSpecValidateTest, Fe003RejectsNonRank2WeightShape) {
  ModelSpec spec = make_tiny_mlp_spec();
  spec.layers[0].weight_shape = {4, 8, 1};

  const Status status = validate(spec);
  ASSERT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("layers['layer0'].weight_shape"), std::string_view::npos)
      << status.message();
  EXPECT_NE(status.message().find("rank 2"), std::string_view::npos) << status.message();
}

// ---------------------------------------------------------------------------
// FE-004:枚举白名单(Activation)。C++ 类型系统本已约束合法取值,本用例经
// static_cast 构造超出枚举定义域的位模式,专门验证 validate() 的运行期防御
// ——JSON 层(tools/frame_dslc)把无法识别的字符串映射为不可信位模式时同样
// 依赖这一道防线。
// ---------------------------------------------------------------------------

TEST(ModelSpecValidateTest, Fe004RejectsUnrecognizedActivationValue) {
  ModelSpec spec = make_tiny_mlp_spec();
  spec.layers[0].activation = static_cast<Activation>(99);

  const Status status = validate(spec);
  ASSERT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("layers['layer0'].activation"), std::string_view::npos)
      << status.message();
  EXPECT_NE(status.message().find("unrecognized"), std::string_view::npos) << status.message();
}

// ---------------------------------------------------------------------------
// FE-005:data 完整性(缺项/inline 元素数不匹配/uniform 范围/data.params)。
// ---------------------------------------------------------------------------

TEST(ModelSpecValidateTest, Fe005RejectsMissingInputData) {
  ModelSpec spec = make_tiny_mlp_spec();
  spec.data.erase("x");

  const Status status = validate(spec);
  ASSERT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("data['x']"), std::string_view::npos) << status.message();
  EXPECT_NE(status.message().find("is required"), std::string_view::npos) << status.message();
}

TEST(ModelSpecValidateTest, Fe005RejectsMissingTargetData) {
  ModelSpec spec = make_tiny_mlp_spec();
  spec.data.erase("target");

  const Status status = validate(spec);
  ASSERT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("data['target']"), std::string_view::npos) << status.message();
  EXPECT_NE(status.message().find("is required"), std::string_view::npos) << status.message();
}

TEST(ModelSpecValidateTest, Fe005RejectsInlineValuesCountMismatch) {
  ModelSpec spec = make_tiny_mlp_spec();
  TensorDataSpec bad_input;
  bad_input.kind = InitKind::kInline;
  bad_input.values = {1.0F, 2.0F, 3.0F};  // x 是 [8,4],numel=32,与 3 不符。
  spec.data["x"] = bad_input;

  const Status status = validate(spec);
  ASSERT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("data['x'].values"), std::string_view::npos) << status.message();
  EXPECT_NE(status.message().find("element count"), std::string_view::npos) << status.message();
}

TEST(ModelSpecValidateTest, Fe005RejectsInvalidUniformRange) {
  ModelSpec spec = make_tiny_mlp_spec();
  spec.data.at("x").lo = 1.0F;
  spec.data.at("x").hi = 1.0F;  // lo < hi 不满足(相等)。

  const Status status = validate(spec);
  ASSERT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("data['x']"), std::string_view::npos) << status.message();
  EXPECT_NE(status.message().find("lo < hi"), std::string_view::npos) << status.message();
}

TEST(ModelSpecValidateTest, Fe005RejectsInvalidParamWeightRange) {
  ModelSpec spec = make_tiny_mlp_spec();
  spec.param_init.weight_lo = 0.5F;
  spec.param_init.weight_hi = 0.1F;

  const Status status = validate(spec);
  ASSERT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("data.params.weight_range"), std::string_view::npos)
      << status.message();
}

TEST(ModelSpecValidateTest, Fe005RejectsInvalidParamBiasRange) {
  ModelSpec spec = make_tiny_mlp_spec();
  spec.param_init.bias_lo = 0.5F;
  spec.param_init.bias_hi = 0.1F;

  const Status status = validate(spec);
  ASSERT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("data.params.bias_range"), std::string_view::npos)
      << status.message();
}

}  // namespace
