// max_pool2d/avg_pool2d 算子测试(schema + cpu kernel + 解析梯度 ≡ 数值微分,
// src/ops/schemas/pool.cpp、src/backends/cpu/kernels/pool.cpp 已实化的行为,
// M21 批3 T4)。二者均 x[N,C,H,W],attrs kernel/stride/padding 各 kInt64Array
// 2元;max_pool2d padding 视为 -inf 语义(只在有效位置参与取最大),avg_pool2d
// 分母恒 KH*KW(include padding,与 cuDNN INCLUDE_PADDING 口径一致)。三个内部
// 梯度算子(max_pool2d_grad_internal/max_pool2d_select_internal/
// avg_pool2d_grad_internal)均自身注册 GradientFn(R11 封闭,
// docs/plan/2026-07-18-batch3-m21-conv.md 第1.2节);max_pool2d 的 argmax 平局
// 约定:取窗口内最低线性索引(kh*KW+kw,行优先遍历 + 严格 `>` 比较天然实现,
// src/backends/cpu/kernels/pool.cpp 头注释)。
//   1. OpRegistry::find 的 schema 字段(公开两算子 + 内部三算子)、
//      shape_infer()/decomposition()/gradient() 函数指针状态;
//   2. shape_infer 合法路径 + 负例(rank 错、输入数错、kernel/stride/padding
//      非法、padding*2>kernel 逐维、输出维<1;内部算子的 input_shape/g-x 形状
//      不自洽);
//   3. eager 数值路径(ARCH-011 第 3 类):max/avg 4x4→2x2 无 padding 已知值、
//      max padding 情形(-inf 语义,不选 padding 位)、avg padding 情形(分母恒
//      KH*KW 的数值区分断言)、fp16(max)/bf16(avg)已知值;
//   4. 平局用例:直接调用 max_pool2d_grad_internal kernel,两个不同窗口位置各
//      构造一组相等最大值,验证梯度落在窗口最低线性索引位;
//   5. 内部算子 max_pool2d_select_internal/avg_pool2d_grad_internal 的基本
//      数值路径(非平局,gather / 均匀回撒验证);
//   6. max_pool2d_cpu_kernel 自身的防御性拒绝路径;
//   7. 解析梯度 ≡ 数值微分(max_pool2d/avg_pool2d wrt x,无平局输入)。
#include <cstdint>
#include <gtest/gtest.h>
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
using frame::ops::testing::CheckGradientMatchesNumeric;
using frame::ops::testing::MakeType;
using frame::testing::default_tolerance;
using frame::testing::tensor_all_close;

// ---------------------------------------------------------------------------
// 1. schema 断言(公开两算子 + 内部三算子)。
// ---------------------------------------------------------------------------

TEST(PoolOpSchemaTest, MaxPool2dRegisteredWithOneInputThreeAttrsAndGradient) {
  const OpSchema* schema = OpRegistry::instance().find("max_pool2d");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 1U);
  EXPECT_EQ(schema->outputs().size(), 1U);
  ASSERT_EQ(schema->attrs().size(), 3U);
  EXPECT_EQ(schema->attrs()[0].name, "kernel");
  EXPECT_EQ(schema->attrs()[1].name, "stride");
  EXPECT_EQ(schema->attrs()[2].name, "padding");
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);
}

TEST(PoolOpSchemaTest, AvgPool2dRegisteredWithOneInputThreeAttrsAndGradient) {
  const OpSchema* schema = OpRegistry::instance().find("avg_pool2d");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 1U);
  EXPECT_EQ(schema->outputs().size(), 1U);
  ASSERT_EQ(schema->attrs().size(), 3U);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);
}

TEST(PoolOpSchemaTest, MaxPool2dGradInternalRegisteredWithTwoInputsAndGradient) {
  const OpSchema* schema = OpRegistry::instance().find("max_pool2d_grad_internal");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 2U);
  EXPECT_EQ(schema->outputs().size(), 1U);
  ASSERT_EQ(schema->attrs().size(), 4U);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);  // R11:内部算子自身注册 GradientFn
}

TEST(PoolOpSchemaTest, MaxPool2dSelectInternalRegisteredWithTwoInputsAndGradient) {
  const OpSchema* schema = OpRegistry::instance().find("max_pool2d_select_internal");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 2U);
  EXPECT_EQ(schema->outputs().size(), 1U);
  ASSERT_EQ(schema->attrs().size(), 3U);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);
}

TEST(PoolOpSchemaTest, AvgPool2dGradInternalRegisteredWithOneInputAndGradient) {
  const OpSchema* schema = OpRegistry::instance().find("avg_pool2d_grad_internal");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 1U);
  EXPECT_EQ(schema->outputs().size(), 1U);
  ASSERT_EQ(schema->attrs().size(), 4U);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);
}

// ---------------------------------------------------------------------------
// 2. shape_infer 合法路径 + 负例。
// ---------------------------------------------------------------------------

const AttrMap& StandardPoolAttrs() {
  static const AttrMap attrs{
      {"kernel", std::vector<int64_t>{2, 2}},
      {"stride", std::vector<int64_t>{2, 2}},
      {"padding", std::vector<int64_t>{0, 0}},
  };
  return attrs;
}

TEST(PoolShapeInferTest, MaxPool2dBasicProducesExpectedOutputShape) {
  const OpSchema* schema = OpRegistry::instance().find("max_pool2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "max_pool2d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 4, 4})};
  ctx.attrs = &StandardPoolAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({1, 1, 2, 2}));
}

TEST(PoolShapeInferTest, AvgPool2dBasicProducesExpectedOutputShape) {
  const OpSchema* schema = OpRegistry::instance().find("avg_pool2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "avg_pool2d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 4, 4})};
  ctx.attrs = &StandardPoolAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({1, 1, 2, 2}));
}

TEST(PoolShapeInferTest, MaxPool2dRankFourRequiredRejectedWithActualRank) {
  const OpSchema* schema = OpRegistry::instance().find("max_pool2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "max_pool2d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 4, 4})};
  ctx.attrs = &StandardPoolAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires x to be rank-4 [N, C, H, W], got rank 3"),
            std::string_view::npos);
}

TEST(PoolShapeInferTest, MaxPool2dWrongInputCountIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("max_pool2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "max_pool2d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 4, 4}),
                     MakeType(DType::of<float>(), {1, 1, 4, 4})};
  ctx.attrs = &StandardPoolAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("expects 1 input, got 2"), std::string_view::npos);
}

TEST(PoolShapeInferTest, KernelEntryLessThanOneIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("max_pool2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "max_pool2d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 4, 4})};
  const AttrMap attrs{
      {"kernel", std::vector<int64_t>{0, 2}},
      {"stride", std::vector<int64_t>{2, 2}},
      {"padding", std::vector<int64_t>{0, 0}},
  };
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("attribute 'kernel' entries must be >= 1, got [0, 2]"),
            std::string_view::npos);
}

TEST(PoolShapeInferTest, StrideEntryLessThanOneIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("max_pool2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "max_pool2d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 4, 4})};
  const AttrMap attrs{
      {"kernel", std::vector<int64_t>{2, 2}},
      {"stride", std::vector<int64_t>{0, 2}},
      {"padding", std::vector<int64_t>{0, 0}},
  };
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("attribute 'stride' entries must be >= 1, got [0, 2]"),
            std::string_view::npos);
}

TEST(PoolShapeInferTest, PaddingEntryLessThanZeroIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("max_pool2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "max_pool2d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 4, 4})};
  const AttrMap attrs{
      {"kernel", std::vector<int64_t>{2, 2}},
      {"stride", std::vector<int64_t>{2, 2}},
      {"padding", std::vector<int64_t>{-1, 0}},
  };
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("attribute 'padding' entries must be >= 0, got [-1, 0]"),
            std::string_view::npos);
}

TEST(PoolShapeInferTest, PaddingHeightTimesTwoExceedsKernelHeightIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("max_pool2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "max_pool2d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 4, 4})};
  // 手算:KH=2, padding_h=2: padding_h*2=4 > KH=2。
  const AttrMap attrs{
      {"kernel", std::vector<int64_t>{2, 2}},
      {"stride", std::vector<int64_t>{1, 1}},
      {"padding", std::vector<int64_t>{2, 0}},
  };
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("padding_h*2=4 exceeds KH=2"), std::string_view::npos);
}

TEST(PoolShapeInferTest, PaddingWidthTimesTwoExceedsKernelWidthIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("avg_pool2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "avg_pool2d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 4, 4})};
  // 手算:KW=2, padding_w=2: padding_w*2=4 > KW=2。
  const AttrMap attrs{
      {"kernel", std::vector<int64_t>{2, 2}},
      {"stride", std::vector<int64_t>{1, 1}},
      {"padding", std::vector<int64_t>{0, 2}},
  };
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("padding_w*2=4 exceeds KW=2"), std::string_view::npos);
}

TEST(PoolShapeInferTest, NonPositiveOutputHeightIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("max_pool2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "max_pool2d";
  // H=1, KH=3, pad_h=0(0*2<=3 满足 padding*2<=kernel 约束), stride_h=1:
  // out_h=floor(1+0-3)/1+1=floor(-2)+1=-1<1(KW=1,W=3 保持合法,只命中高度
  // 分支)。
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 1, 3})};
  const AttrMap attrs{
      {"kernel", std::vector<int64_t>{3, 1}},
      {"stride", std::vector<int64_t>{1, 1}},
      {"padding", std::vector<int64_t>{0, 0}},
  };
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("produces non-positive output height"),
            std::string_view::npos);
}

TEST(PoolShapeInferTest, MaxPool2dGradInternalRejectsXShapeNotMatchingInputShapeAttr) {
  const OpSchema* schema = OpRegistry::instance().find("max_pool2d_grad_internal");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "max_pool2d_grad_internal";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 2, 2}),
                     MakeType(DType::of<float>(), {1, 1, 4, 4})};
  const AttrMap attrs{
      {"input_shape", Shape({1, 1, 3, 3})},  // 与 x 实际 shape [1,1,4,4] 不符
      {"kernel", std::vector<int64_t>{2, 2}},
      {"stride", std::vector<int64_t>{2, 2}},
      {"padding", std::vector<int64_t>{0, 0}},
  };
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires x shape to match attribute 'input_shape'"),
            std::string_view::npos);
}

TEST(PoolShapeInferTest, MaxPool2dSelectInternalRejectsGAndXShapeMismatch) {
  const OpSchema* schema = OpRegistry::instance().find("max_pool2d_select_internal");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "max_pool2d_select_internal";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 4, 4}),
                     MakeType(DType::of<float>(), {1, 1, 3, 3})};
  const AttrMap attrs{
      {"kernel", std::vector<int64_t>{2, 2}},
      {"stride", std::vector<int64_t>{2, 2}},
      {"padding", std::vector<int64_t>{0, 0}},
  };
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires g and x of the same shape"),
            std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 3. eager 数值(ARCH-011 第 3 类)。
// ---------------------------------------------------------------------------

class PoolOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(PoolOpEagerTest, MaxPool2dNoPaddingKnownValues) {
  // x 4x4(行优先):
  //   1 3 2 4
  //   5 6 8 7
  //   9 1 4 2
  //   3 5 6 9
  // kernel2x2 stride2 pad0 → 逐窗取最大:[6,8,9,9]。
  Tensor x = MakeTensorWithShape<float>(
      Shape({1, 1, 4, 4}), {1.0F, 3.0F, 2.0F, 4.0F, 5.0F, 6.0F, 8.0F, 7.0F, 9.0F, 1.0F, 4.0F, 2.0F,
                            3.0F, 5.0F, 6.0F, 9.0F});
  Tensor expected = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {6.0F, 8.0F, 9.0F, 9.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "max_pool2d";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &StandardPoolAttrs();
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(PoolOpEagerTest, AvgPool2dNoPaddingKnownValues) {
  // 同上 x,avg:(1+3+5+6)/4=3.75  (2+4+8+7)/4=5.25
  //           (9+1+3+5)/4=4.5   (4+2+6+9)/4=5.25
  Tensor x = MakeTensorWithShape<float>(
      Shape({1, 1, 4, 4}), {1.0F, 3.0F, 2.0F, 4.0F, 5.0F, 6.0F, 8.0F, 7.0F, 9.0F, 1.0F, 4.0F, 2.0F,
                            3.0F, 5.0F, 6.0F, 9.0F});
  Tensor expected = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {3.75F, 5.25F, 4.5F, 5.25F});
  Tensor out = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "avg_pool2d";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &StandardPoolAttrs();
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(PoolOpEagerTest, MaxPool2dPaddingUsesNegativeInfinitySemanticsNotZero) {
  // x=[[-1,-2],[-3,-4]](全负),kernel2x2 stride1 pad1:若 padding 误用 0 语义,
  // 0 会被误判为每窗最大值(因 x 全负);-inf 语义下 padding 位置永不入选,
  // 各窗最大值来自窗口内实际存在的负值。逐窗手算(见文件设计记录):
  //   [-1,-1,-2, -1,-1,-2, -3,-3,-4]
  Tensor x = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {-1.0F, -2.0F, -3.0F, -4.0F});
  Tensor expected = MakeTensorWithShape<float>(
      Shape({1, 1, 3, 3}), {-1.0F, -1.0F, -2.0F, -1.0F, -1.0F, -2.0F, -3.0F, -3.0F, -4.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({1, 1, 3, 3}),
                                          {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});

  const AttrMap attrs{
      {"kernel", std::vector<int64_t>{2, 2}},
      {"stride", std::vector<int64_t>{1, 1}},
      {"padding", std::vector<int64_t>{1, 1}},
  };
  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "max_pool2d";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(PoolOpEagerTest, AvgPool2dPaddingDenominatorIsAlwaysKernelHeightTimesKernelWidth) {
  // x=[[1,2],[3,4]],kernel2x2 stride1 pad1:角窗仅 1 个有效位置,若分母随有效
  // 位置数变化(exclude-padding)会得到与本预期不同的值(如角窗分母应为 4 而非
  // 1)。逐窗手算(分母恒为 4,见文件设计记录):
  //   [0.25,0.75,0.5, 1.0,2.5,1.5, 0.75,1.75,1.0]
  Tensor x = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {1.0F, 2.0F, 3.0F, 4.0F});
  Tensor expected = MakeTensorWithShape<float>(
      Shape({1, 1, 3, 3}), {0.25F, 0.75F, 0.5F, 1.0F, 2.5F, 1.5F, 0.75F, 1.75F, 1.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({1, 1, 3, 3}),
                                          {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});

  const AttrMap attrs{
      {"kernel", std::vector<int64_t>{2, 2}},
      {"stride", std::vector<int64_t>{1, 1}},
      {"padding", std::vector<int64_t>{1, 1}},
  };
  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "avg_pool2d";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(PoolOpEagerTest, MaxPool2dFloat16KnownValuesViaBitLevelConversion) {
  // 复用 NoPadding 已知整数值(fp16 尾数内精确可表示):6,8,9,9。
  Tensor x = MakeTensorWithShape<float16_t>(
      Shape({1, 1, 4, 4}), {float_to_float16(1.0F), float_to_float16(3.0F), float_to_float16(2.0F),
                            float_to_float16(4.0F), float_to_float16(5.0F), float_to_float16(6.0F),
                            float_to_float16(8.0F), float_to_float16(7.0F), float_to_float16(9.0F),
                            float_to_float16(1.0F), float_to_float16(4.0F), float_to_float16(2.0F),
                            float_to_float16(3.0F), float_to_float16(5.0F), float_to_float16(6.0F),
                            float_to_float16(9.0F)});
  Tensor expected = MakeTensorWithShape<float16_t>(
      Shape({1, 1, 2, 2}), {float_to_float16(6.0F), float_to_float16(8.0F), float_to_float16(9.0F),
                            float_to_float16(9.0F)});
  Tensor out = MakeTensorWithShape<float16_t>(
      Shape({1, 1, 2, 2}), {float16_t{0}, float16_t{0}, float16_t{0}, float16_t{0}});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "max_pool2d";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &StandardPoolAttrs();
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat16)));
}

TEST_F(PoolOpEagerTest, AvgPool2dBFloat16KnownValuesViaBitLevelConversion) {
  // 复用 NoPadding 已知值(3.75/5.25/4.5/5.25,分母均为 2 的幂,bf16 精确表示)。
  Tensor x = MakeTensorWithShape<bfloat16_t>(
      Shape({1, 1, 4, 4}),
      {float_to_bfloat16(1.0F), float_to_bfloat16(3.0F), float_to_bfloat16(2.0F),
       float_to_bfloat16(4.0F), float_to_bfloat16(5.0F), float_to_bfloat16(6.0F),
       float_to_bfloat16(8.0F), float_to_bfloat16(7.0F), float_to_bfloat16(9.0F),
       float_to_bfloat16(1.0F), float_to_bfloat16(4.0F), float_to_bfloat16(2.0F),
       float_to_bfloat16(3.0F), float_to_bfloat16(5.0F), float_to_bfloat16(6.0F),
       float_to_bfloat16(9.0F)});
  Tensor expected = MakeTensorWithShape<bfloat16_t>(
      Shape({1, 1, 2, 2}), {float_to_bfloat16(3.75F), float_to_bfloat16(5.25F),
                            float_to_bfloat16(4.5F), float_to_bfloat16(5.25F)});
  Tensor out = MakeTensorWithShape<bfloat16_t>(
      Shape({1, 1, 2, 2}), {bfloat16_t{0}, bfloat16_t{0}, bfloat16_t{0}, bfloat16_t{0}});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "avg_pool2d";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &StandardPoolAttrs();
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kBFloat16)));
}

// ---------------------------------------------------------------------------
// 4. 平局用例:直接调用 max_pool2d_grad_internal kernel,验证梯度落在窗口
//    最低线性索引位(严格伴随性依赖此约定)。
// ---------------------------------------------------------------------------

class MaxPool2dGradInternalTieBreakTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(MaxPool2dGradInternalTieBreakTest, TieAtLinearIndexZeroVersusOneSelectsIndexZero) {
  // 单窗口(kernel=stride=2x2,整张 2x2 输入即窗口):x=[5,5,1,2],线性索引0
  // (kh=0,kw=0)与索引1(kh=0,kw=1)同为最大值5;约定取最低线性索引→选索引0。
  // dy=[7]→期望 dx=[7,0,0,0]。
  const Result<frame::ops::KernelFn> found = frame::ops::KernelRegistry::instance().find(
      "max_pool2d_grad_internal", frame::kCpuBackendName);
  ASSERT_TRUE(found.is_ok()) << found.status().message();
  const frame::ops::KernelFn kernel = found.value();

  Tensor x = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {5.0F, 5.0F, 1.0F, 2.0F});
  Tensor dy = MakeTensorWithShape<float>(Shape({1, 1, 1, 1}), {7.0F});
  Tensor dx = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {0.0F, 0.0F, 0.0F, 0.0F});
  Tensor expected = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {7.0F, 0.0F, 0.0F, 0.0F});

  const AttrMap attrs{
      {"input_shape", Shape({1, 1, 2, 2})},
      {"kernel", std::vector<int64_t>{2, 2}},
      {"stride", std::vector<int64_t>{2, 2}},
      {"padding", std::vector<int64_t>{0, 0}},
  };
  std::vector<Tensor> inputs{dy, x};
  std::vector<Tensor> outputs{dx};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.attrs = &attrs;
  ctx.device = cpu_device();

  const Status status = kernel(ctx);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(MaxPool2dGradInternalTieBreakTest, TieAtLinearIndexTwoVersusThreeSelectsIndexTwo) {
  // 同一窗口,tie 改在线性索引2(kh=1,kw=0)与索引3(kh=1,kw=1):x=[1,2,5,5]。
  // 约定取最低线性索引→选索引2(即 (1,0) 位置)。dy=[9]→期望 dx=[0,0,9,0]。
  const Result<frame::ops::KernelFn> found = frame::ops::KernelRegistry::instance().find(
      "max_pool2d_grad_internal", frame::kCpuBackendName);
  ASSERT_TRUE(found.is_ok()) << found.status().message();
  const frame::ops::KernelFn kernel = found.value();

  Tensor x = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {1.0F, 2.0F, 5.0F, 5.0F});
  Tensor dy = MakeTensorWithShape<float>(Shape({1, 1, 1, 1}), {9.0F});
  Tensor dx = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {0.0F, 0.0F, 0.0F, 0.0F});
  Tensor expected = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {0.0F, 0.0F, 9.0F, 0.0F});

  const AttrMap attrs{
      {"input_shape", Shape({1, 1, 2, 2})},
      {"kernel", std::vector<int64_t>{2, 2}},
      {"stride", std::vector<int64_t>{2, 2}},
      {"padding", std::vector<int64_t>{0, 0}},
  };
  std::vector<Tensor> inputs{dy, x};
  std::vector<Tensor> outputs{dx};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.attrs = &attrs;
  ctx.device = cpu_device();

  const Status status = kernel(ctx);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 5. 内部算子基本数值路径(非平局)。
// ---------------------------------------------------------------------------

class PoolInternalOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(PoolInternalOpEagerTest, MaxPool2dSelectInternalGathersGAtArgmaxPositions) {
  // 复用 NoPadding 用例的 x(4x4,argmax 分别落在线性索引5/6/8/15,均无平局);
  // g[i]=i+1(1..16),期望 out=[g[5],g[6],g[8],g[15]]=[6,7,9,16]。
  const Result<frame::ops::KernelFn> found = frame::ops::KernelRegistry::instance().find(
      "max_pool2d_select_internal", frame::kCpuBackendName);
  ASSERT_TRUE(found.is_ok()) << found.status().message();
  const frame::ops::KernelFn kernel = found.value();

  Tensor x = MakeTensorWithShape<float>(
      Shape({1, 1, 4, 4}), {1.0F, 3.0F, 2.0F, 4.0F, 5.0F, 6.0F, 8.0F, 7.0F, 9.0F, 1.0F, 4.0F, 2.0F,
                            3.0F, 5.0F, 6.0F, 9.0F});
  std::vector<float> g_values;
  for (int i = 1; i <= 16; ++i) g_values.push_back(static_cast<float>(i));
  Tensor g = MakeTensorWithShape<float>(Shape({1, 1, 4, 4}), g_values);
  Tensor out = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {0.0F, 0.0F, 0.0F, 0.0F});
  Tensor expected = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {6.0F, 7.0F, 9.0F, 16.0F});

  const AttrMap& attrs = StandardPoolAttrs();
  std::vector<Tensor> inputs{g, x};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.attrs = &attrs;
  ctx.device = cpu_device();

  const Status status = kernel(ctx);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(PoolInternalOpEagerTest, AvgPool2dGradInternalSpreadsDyEvenlyOverNonOverlappingWindows) {
  // stride==kernel(无重叠),input_shape=[1,1,4,4],dy=[10,20,30,40]:每个 2x2
  // 窗口内 4 个位置均分 dy/4(分母恒 KH*KW,同前向语义)。
  const Result<frame::ops::KernelFn> found = frame::ops::KernelRegistry::instance().find(
      "avg_pool2d_grad_internal", frame::kCpuBackendName);
  ASSERT_TRUE(found.is_ok()) << found.status().message();
  const frame::ops::KernelFn kernel = found.value();

  Tensor dy = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {10.0F, 20.0F, 30.0F, 40.0F});
  Tensor dx = MakeTensorWithShape<float>(
      Shape({1, 1, 4, 4}), {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                            0.0F, 0.0F, 0.0F, 0.0F});
  Tensor expected = MakeTensorWithShape<float>(
      Shape({1, 1, 4, 4}), {2.5F, 2.5F, 5.0F, 5.0F, 2.5F, 2.5F, 5.0F, 5.0F, 7.5F, 7.5F, 10.0F,
                            10.0F, 7.5F, 7.5F, 10.0F, 10.0F});

  const AttrMap attrs{
      {"input_shape", Shape({1, 1, 4, 4})},
      {"kernel", std::vector<int64_t>{2, 2}},
      {"stride", std::vector<int64_t>{2, 2}},
      {"padding", std::vector<int64_t>{0, 0}},
  };
  std::vector<Tensor> inputs{dy};
  std::vector<Tensor> outputs{dx};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.attrs = &attrs;
  ctx.device = cpu_device();

  const Status status = kernel(ctx);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 6. max_pool2d_cpu_kernel 自身的防御性拒绝路径。
// ---------------------------------------------------------------------------

struct MaxPool2dOpNameTag {
  static constexpr std::string_view kOpName = "max_pool2d";
};
using MaxPool2dOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<MaxPool2dOpNameTag>;

TEST_F(MaxPool2dOpKernelTest, RejectsUnsupportedDtypeInt32) {
  Tensor x = MakeTensor<std::int32_t>(Shape({1, 1, 4, 4}));
  Tensor out = MakeTensor<std::int32_t>(Shape({1, 1, 2, 2}));

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.attrs = &StandardPoolAttrs();
  ctx.device = cpu_device();

  const Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("does not support dtype 'int32'"), std::string_view::npos);
}

TEST_F(MaxPool2dOpKernelTest, RejectsOutShapeMismatch) {
  Tensor x = MakeTensor<float>(Shape({1, 1, 4, 4}));
  Tensor out = MakeTensor<float>(Shape({1, 1, 3, 3}));  // 期望 [1,1,2,2],故意给错

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.attrs = &StandardPoolAttrs();
  ctx.device = cpu_device();

  const Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("requires out shape to match the pooling result"),
            std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 7. 解析梯度 ≡ 数值微分(max_pool2d/avg_pool2d wrt x,无平局输入)。
// ---------------------------------------------------------------------------

class PoolGradientTest : public frame::ops::testing::ElementwiseEagerTestBase {};

// x[1,1,4,4](全部互异,无平局窗口)->pool->sum(axes=[])作标量 loss。
Graph BuildPoolLossGraph(const char* pool_op) {
  Graph graph("pool_loss");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {1, 1, 4, 4})).value();
  const AttrMap pool_attrs{
      {"kernel", std::vector<int64_t>{2, 2}},
      {"stride", std::vector<int64_t>{2, 2}},
      {"padding", std::vector<int64_t>{0, 0}},
  };
  Node* pool_node = create_node_with_inferred_types(graph, pool_op, {x}, pool_attrs).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {pool_node->output(0)}, sum_attrs).value();
  const Status mark_status = graph.mark_output(sum_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

// 与 test_conv2d.cpp 同一实测区间(fp32,BUILD-011 建议 1e-3 量级起调)。
constexpr double kPoolCentralDifferenceH = 1e-2;

TEST_F(PoolGradientTest, MaxPool2dGradientMatchesNumericForX) {
  const Graph forward = BuildPoolLossGraph("max_pool2d");
  // 与 NoPadding 已知值同一组数据(全部互异,窗口内无平局,避开 max 的
  // 不可导拐点,同 relu kink 点处理惯例)。
  Tensor x = MakeTensorWithShape<float>(
      Shape({1, 1, 4, 4}), {1.0F, 3.0F, 2.0F, 4.0F, 5.0F, 6.0F, 8.0F, 7.0F, 9.0F, 1.0F, 4.0F, 2.0F,
                            3.0F, 5.0F, 6.0F, 9.0F});
  std::vector<Tensor> inputs{x};
  const std::vector<int32_t> wrt{0};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kPoolCentralDifferenceH));
}

TEST_F(PoolGradientTest, AvgPool2dGradientMatchesNumericForX) {
  const Graph forward = BuildPoolLossGraph("avg_pool2d");
  Tensor x = MakeTensorWithShape<float>(
      Shape({1, 1, 4, 4}), {1.0F, 3.0F, 2.0F, 4.0F, 5.0F, 6.0F, 8.0F, 7.0F, 9.0F, 1.0F, 4.0F, 2.0F,
                            3.0F, 5.0F, 6.0F, 9.0F});
  std::vector<Tensor> inputs{x};
  const std::vector<int32_t> wrt{0};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kPoolCentralDifferenceH));
}

}  // namespace
