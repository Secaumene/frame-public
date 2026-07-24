// sigmoid 算子测试(schema + cpu kernel + 解析梯度 ≡ 数值微分,
// src/ops/schemas/elementwise.cpp、src/backends/cpu/kernels/elementwise.cpp
// 已实化的行为,M21 批3 T4)。sigmoid(x)=1/(1+e^-x),一元逐元素算子;kernel
// 侧用数值稳定分式(x>=0 走 1/(1+e^-x),x<0 走 e^x/(1+e^x),避免 e^-x 在 x
// 很负时上溢,见 kernel 头注释)。
//   1. OpRegistry::find("sigmoid") 的 schema 字段、traits(仅 kElementwise,
//      不标 kFusable——fused_elementwise 内核暂不认识 sigmoid,同文件相关
//      待办说明)、shape_infer()/decomposition()/gradient() 三函数指针状态;
//   2. infer_sigmoid_shape 的合法路径(恒等 shape)+ 输入数错负例;
//   3. eager 数值路径(ARCH-011 第 3 类):fp32 与双精度参考值
//      1/(1+exp(-x))比对、fp16/bf16(位级转换,同一组数值)、极值稳定性
//      (±30 不产生 NaN/Inf);
//   4. sigmoid_cpu_kernel 自身的防御性拒绝路径(dtype 不支持 / shape 不一致);
//   5. 解析梯度 ≡ 数值微分。
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <string_view>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/ir/node.h>
#include <frame/ops/graph_builder.h>
#include <frame/ops/kernel_registry.h>
#include <frame/ops/op_registry.h>

#include "../common/tolerance.h"
#include "elementwise_op_test_helpers.h"
#include "gradient_check_test_helpers.h"

namespace {

using frame::bfloat16_t;
using frame::cpu_device;
using frame::DType;
using frame::DTypeCode;
using frame::ErrorCode;
using frame::float16_t;
using frame::float_to_bfloat16;
using frame::float_to_float16;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::Tensor;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::Value;
using frame::ops::AttrMap;
using frame::ops::create_node_with_inferred_types;
using frame::ops::NodeContext;
using frame::ops::OpRegistry;
using frame::ops::OpSchema;
using frame::ops::OpTrait;
using frame::ops::testing::CheckGradientMatchesNumeric;
using frame::ops::testing::MakeType;
using frame::testing::default_tolerance;
using frame::testing::tensor_all_close;

// 双精度参考实现:1/(1+exp(-x)),供 fp32/fp16/bf16 三档已知值交叉核对——
// 与被测 kernel 的数值稳定分支实现(x>=0/x<0 两支)相互独立,不复用被测代码。
double SigmoidReference(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// ---------------------------------------------------------------------------
// 1. schema 断言。
// ---------------------------------------------------------------------------

TEST(SigmoidOpSchemaTest, RegisteredWithOneInputOneOutput) {
  const OpSchema* schema = OpRegistry::instance().find("sigmoid");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 1U);
  EXPECT_EQ(schema->outputs().size(), 1U);
}

TEST(SigmoidOpSchemaTest, TraitsMatchElementwiseOnlyNotFusableNotCommutative) {
  const OpSchema* schema = OpRegistry::instance().find("sigmoid");
  ASSERT_NE(schema, nullptr);
  EXPECT_TRUE(schema->has_trait(OpTrait::kElementwise));
  // fused_elementwise 内核暂不认识 sigmoid 表达式(src/ops/schemas/
  // elementwise.cpp 同一处待办说明),不标 kFusable。
  EXPECT_FALSE(schema->has_trait(OpTrait::kFusable));
  EXPECT_FALSE(schema->has_trait(OpTrait::kCommutative));
  EXPECT_FALSE(schema->has_trait(OpTrait::kHasSideEffect));
}

TEST(SigmoidOpSchemaTest, HasShapeInferButNoDecompositionAndHasGradient) {
  const OpSchema* schema = OpRegistry::instance().find("sigmoid");
  ASSERT_NE(schema, nullptr);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

TEST(SigmoidShapeInferTest, OutputShapeIsIdenticalToInputShape) {
  const OpSchema* schema = OpRegistry::instance().find("sigmoid");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "sigmoid";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({2, 3}));
}

TEST(SigmoidShapeInferTest, WrongInputCountIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("sigmoid");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "sigmoid";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {2, 3})};
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("expects 1 input, got 2"), std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 3. eager 数值(ARCH-011 第 3 类)。
// ---------------------------------------------------------------------------

class SigmoidOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(SigmoidOpEagerTest, Float32MatchesDoublePrecisionReference) {
  const std::vector<double> x_values{-3.0, -1.0, 0.0, 1.0, 3.0};
  std::vector<float> x_floats;
  std::vector<float> expected_floats;
  for (double v : x_values) {
    x_floats.push_back(static_cast<float>(v));
    expected_floats.push_back(static_cast<float>(SigmoidReference(v)));
  }
  Tensor x = MakeTensor1D<float>(x_floats);
  Tensor expected = MakeTensor1D<float>(expected_floats);
  Tensor out = MakeTensor1D<float>(std::vector<float>(x_floats.size(), 0.0F));

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "sigmoid";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(SigmoidOpEagerTest, Float16MatchesDoublePrecisionReferenceViaBitLevelConversion) {
  const std::vector<double> x_values{-3.0, -1.0, 0.0, 1.0, 3.0};
  std::vector<float16_t> x_fp16;
  std::vector<float16_t> expected_fp16;
  for (double v : x_values) {
    x_fp16.push_back(float_to_float16(static_cast<float>(v)));
    expected_fp16.push_back(float_to_float16(static_cast<float>(SigmoidReference(v))));
  }
  Tensor x = MakeTensor1D<float16_t>(x_fp16);
  Tensor expected = MakeTensor1D<float16_t>(expected_fp16);
  Tensor out = MakeTensor1D<float16_t>(std::vector<float16_t>(x_fp16.size(), float16_t{0}));

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "sigmoid";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat16)));
}

TEST_F(SigmoidOpEagerTest, BFloat16MatchesDoublePrecisionReferenceViaBitLevelConversion) {
  const std::vector<double> x_values{-3.0, -1.0, 0.0, 1.0, 3.0};
  std::vector<bfloat16_t> x_bf16;
  std::vector<bfloat16_t> expected_bf16;
  for (double v : x_values) {
    x_bf16.push_back(float_to_bfloat16(static_cast<float>(v)));
    expected_bf16.push_back(float_to_bfloat16(static_cast<float>(SigmoidReference(v))));
  }
  Tensor x = MakeTensor1D<bfloat16_t>(x_bf16);
  Tensor expected = MakeTensor1D<bfloat16_t>(expected_bf16);
  Tensor out = MakeTensor1D<bfloat16_t>(std::vector<bfloat16_t>(x_bf16.size(), bfloat16_t{0}));

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "sigmoid";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kBFloat16)));
}

TEST_F(SigmoidOpEagerTest, ExtremeValuesRemainFiniteAndCloseToSaturationBounds) {
  // ±30 远超 float32 下 exp() 上溢阈值(exp(88)才上溢),数值稳定分支
  // (kernel 头注释)在此量级已充分体现:sigmoid(30)≈1(在 float32 精度下
  // 舍入为恰好 1.0),sigmoid(-30)≈0(≈9.36e-14,在 default_tolerance 的
  // atol=1e-6 内视为 0)。
  const float pos_extreme = 30.0F;
  const float neg_extreme = -30.0F;
  Tensor x = MakeTensor1D<float>({pos_extreme, neg_extreme});
  Tensor out = MakeTensor1D<float>({0.0F, 0.0F});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "sigmoid";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();

  const float* data = outputs[0].data<float>();
  EXPECT_FALSE(std::isnan(data[0]));
  EXPECT_FALSE(std::isinf(data[0]));
  EXPECT_FALSE(std::isnan(data[1]));
  EXPECT_FALSE(std::isinf(data[1]));

  Tensor expected = MakeTensor1D<float>({1.0F, 0.0F});
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 4. sigmoid_cpu_kernel 自身的防御性拒绝路径。
// ---------------------------------------------------------------------------

struct SigmoidOpNameTag {
  static constexpr std::string_view kOpName = "sigmoid";
};
using SigmoidOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<SigmoidOpNameTag>;

TEST_F(SigmoidOpKernelTest, RejectsUnsupportedDtypeInt32) {
  Tensor x = MakeTensor<std::int32_t>(Shape({2}));
  Tensor out = MakeTensor<std::int32_t>(Shape({2}));

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("does not support dtype 'int32'"), std::string_view::npos);
}

TEST_F(SigmoidOpKernelTest, RejectsShapeMismatchBetweenXAndOut) {
  Tensor x = MakeTensor<float>(Shape({2, 3}));
  Tensor out = MakeTensor<float>(Shape({3, 2}));

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("requires x/out of the same shape"), std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 5. 解析梯度 ≡ 数值微分。
// ---------------------------------------------------------------------------

class SigmoidGradientTest : public frame::ops::testing::ElementwiseEagerTestBase {};

Graph BuildSigmoidLossGraph() {
  Graph graph("sigmoid_loss");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {4})).value();
  Node* sigmoid_node = create_node_with_inferred_types(graph, "sigmoid", {x}).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {sigmoid_node->output(0)}, sum_attrs).value();
  const Status mark_status = graph.mark_output(sum_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

// 与 test_conv2d.cpp 同一实测区间(fp32,BUILD-011 建议 1e-3 量级起调)。
constexpr double kSigmoidCentralDifferenceH = 1e-2;

TEST_F(SigmoidGradientTest, GradientMatchesNumericForX) {
  const Graph forward = BuildSigmoidLossGraph();
  Tensor x = MakeTensor1D<float>({-1.5F, -0.5F, 0.5F, 1.5F});
  std::vector<Tensor> inputs{x};
  const std::vector<int32_t> wrt{0};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kSigmoidCentralDifferenceH));
}

}  // namespace
