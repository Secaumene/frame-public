// conv2d 算子测试(schema + cpu kernel + 图编译路径 + 解析梯度 ≡ 数值微分,
// src/ops/schemas/conv.cpp、src/backends/cpu/kernels/conv.cpp 已实化的行为,
// M21 批3 T4)。conv2d 是 NCHW 二维卷积:x[N,Cin,H,W] + w[Cout,Cin/g,KH,KW] +
// 可选 bias[Cout](variadic_input("bias", min_count=0),裁决点①,
// docs/plan/2026-07-18-batch3-m21-conv.md 第1.1节);属性 stride/padding 各
// kInt64Array 2元、groups kInt64;输出维 floor 口径。
//   1. OpRegistry::find("conv2d") 的 schema 字段(2 定长输入 + 变长 bias 组、
//      1 输出、3 必填属性)、shape_infer()/decomposition()/gradient() 三函数
//      指针状态;
//   2. infer_conv2d_shape 的合法路径(基本形态/groups>1/含 bias)+ 负例逐条
//      (x/w 秩错、Cin%groups、Cout%groups、w 第二维错、bias 秩/长度错、
//      bias 组>1、输出维<1、stride<1、padding<0),消息含实际值关键片段;
//   3. eager 数值路径(ARCH-011 第 3 类):3×3 输入/2×2 核 stride1 pad0 逐位
//      手算、stride2、padding1、groups=2 条带核对、bias 有/无两态、fp16/bf16
//      (位级转换,复用 stride1 pad0 已知值);
//   4. 图编译路径:conv2d 经 standard_pipeline(frame::runtime::compile 内部
//      调用,examples/03_custom_op/main.cpp 末尾注释同一模式)+ cpu 执行的结果
//      同时与①直接 Backend::launch(eager,不经 pass pipeline)结果、②手算
//      参考值交叉核对;
//   5. conv2d_cpu_kernel 自身的防御性拒绝路径(dtype 不支持 / out shape 不符),
//      经 KernelRegistry::find("conv2d", cpu) 直接取 KernelFn 调用驱动;
//   6. 解析梯度 ≡ 数值微分(BUILD-011「解析梯度 ≡ 数值微分校验」专款,放宽
//      一档):x/w/bias 三路,经 tests/cpp/ops/gradient_check_test_helpers.h
//      共用的 build_backward_graph 全链模式(照抄 tests/cpp/compiler/
//      test_autograd.cpp 先例)。
//
// 共用设施复用 tests/cpp/ops/elementwise_op_test_helpers.h(MakeType/eager
// fixture/kernel fixture 模板,REUSE-002)与
// tests/cpp/ops/gradient_check_test_helpers.h(梯度检查断言,REUSE-002)。
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string_view>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/executable.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/graph_builder.h>
#include <frame/ops/kernel_registry.h>
#include <frame/ops/op_registry.h>
#include <frame/runtime/compile.h>

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
using frame::hal::CompileOptions;
using frame::hal::Executable;
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
// 1. schema 断言。
// ---------------------------------------------------------------------------

TEST(Conv2dOpSchemaTest, RegisteredAndFindable) {
  const OpSchema* schema = OpRegistry::instance().find("conv2d");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->name(), "conv2d");
}

TEST(Conv2dOpSchemaTest, HasTwoFixedInputsVariadicBiasAndThreeAttrs) {
  const OpSchema* schema = OpRegistry::instance().find("conv2d");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 2U);  // x, w(定长);bias 是尾随变长组
  EXPECT_TRUE(schema->has_variadic_inputs());
  EXPECT_EQ(schema->min_input_count(), 2);  // 变长组 min_count=0(bias 可选)
  EXPECT_EQ(schema->outputs().size(), 1U);
  ASSERT_EQ(schema->attrs().size(), 3U);
  EXPECT_EQ(schema->attrs()[0].name, "stride");
  EXPECT_EQ(schema->attrs()[1].name, "padding");
  EXPECT_EQ(schema->attrs()[2].name, "groups");
}

TEST(Conv2dOpSchemaTest, HasShapeInferButNoDecompositionAndHasGradient) {
  const OpSchema* schema = OpRegistry::instance().find("conv2d");
  ASSERT_NE(schema, nullptr);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

const AttrMap& StandardAttrs() {
  static const AttrMap attrs{
      {"stride", std::vector<int64_t>{1, 1}},
      {"padding", std::vector<int64_t>{0, 0}},
      {"groups", int64_t{1}},
  };
  return attrs;
}

TEST(Conv2dShapeInferTest, BasicNoBiasProducesExpectedOutputShape) {
  const OpSchema* schema = OpRegistry::instance().find("conv2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "conv2d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 3, 3}),
                     MakeType(DType::of<float>(), {1, 1, 2, 2})};
  ctx.attrs = &StandardAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({1, 1, 2, 2}));
}

TEST(Conv2dShapeInferTest, GroupsGreaterThanOneProducesExpectedOutputShape) {
  const OpSchema* schema = OpRegistry::instance().find("conv2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "conv2d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 4, 5, 5}),
                     MakeType(DType::of<float>(), {2, 2, 3, 3})};
  const AttrMap attrs{
      {"stride", std::vector<int64_t>{1, 1}},
      {"padding", std::vector<int64_t>{0, 0}},
      {"groups", int64_t{2}},
  };
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({1, 2, 3, 3}));
}

TEST(Conv2dShapeInferTest, WithBiasOfMatchingCoutProducesExpectedOutputShape) {
  const OpSchema* schema = OpRegistry::instance().find("conv2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "conv2d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 3, 3}),
                     MakeType(DType::of<float>(), {2, 1, 2, 2}), MakeType(DType::of<float>(), {2})};
  ctx.attrs = &StandardAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({1, 2, 2, 2}));
}

TEST(Conv2dShapeInferTest, RankFourRequiredForXRejectedWithActualRank) {
  const OpSchema* schema = OpRegistry::instance().find("conv2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "conv2d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 3}),
                     MakeType(DType::of<float>(), {1, 1, 2, 2})};
  ctx.attrs = &StandardAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires x to be rank-4 [N, Cin, H, W], got rank 3"),
            std::string_view::npos);
}

TEST(Conv2dShapeInferTest, RankFourRequiredForWRejectedWithActualRank) {
  const OpSchema* schema = OpRegistry::instance().find("conv2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "conv2d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 3, 3}),
                     MakeType(DType::of<float>(), {1, 1, 2})};
  ctx.attrs = &StandardAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find(
                "requires w to be rank-4 [Cout, Cin/groups, KH, KW], got rank 3"),
            std::string_view::npos);
}

TEST(Conv2dShapeInferTest, CinNotDivisibleByGroupsIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("conv2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "conv2d";
  // Cin=3, groups=2:3%2!=0(Cout=2/groups=2=0 恰好整除,确保只命中 Cin 分支)。
  ctx.input_types = {MakeType(DType::of<float>(), {1, 3, 4, 4}),
                     MakeType(DType::of<float>(), {2, 1, 2, 2})};
  const AttrMap attrs{
      {"stride", std::vector<int64_t>{1, 1}},
      {"padding", std::vector<int64_t>{0, 0}},
      {"groups", int64_t{2}},
  };
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("Cin=3 is not divisible by groups=2"),
            std::string_view::npos);
}

TEST(Conv2dShapeInferTest, CoutNotDivisibleByGroupsIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("conv2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "conv2d";
  // Cin=4, groups=2: 4%2==0(不命中 Cin 分支);Cout=3, groups=2: 3%2!=0。
  ctx.input_types = {MakeType(DType::of<float>(), {1, 4, 4, 4}),
                     MakeType(DType::of<float>(), {3, 2, 2, 2})};
  const AttrMap attrs{
      {"stride", std::vector<int64_t>{1, 1}},
      {"padding", std::vector<int64_t>{0, 0}},
      {"groups", int64_t{2}},
  };
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("Cout=3 is not divisible by groups=2"),
            std::string_view::npos);
}

TEST(Conv2dShapeInferTest, WSecondDimensionMismatchIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("conv2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "conv2d";
  // Cin=4, groups=2 => 期望 Cin/groups=2;w 第二维给 3。
  ctx.input_types = {MakeType(DType::of<float>(), {1, 4, 4, 4}),
                     MakeType(DType::of<float>(), {2, 3, 2, 2})};
  const AttrMap attrs{
      {"stride", std::vector<int64_t>{1, 1}},
      {"padding", std::vector<int64_t>{0, 0}},
      {"groups", int64_t{2}},
  };
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find(
                "w's second dimension (Cin/groups) is 3, expected 2 (Cin=4, groups=2)"),
            std::string_view::npos);
}

TEST(Conv2dShapeInferTest, BiasRankNotOneIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("conv2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "conv2d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 3, 3}),
                     MakeType(DType::of<float>(), {1, 1, 2, 2}),
                     MakeType(DType::of<float>(), {1, 1})};  // rank 2
  ctx.attrs = &StandardAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires bias to be rank-1, got rank 2"),
            std::string_view::npos);
}

TEST(Conv2dShapeInferTest, BiasLengthMismatchWithCoutIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("conv2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "conv2d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 3, 3}),
                     MakeType(DType::of<float>(), {1, 1, 2, 2}),
                     MakeType(DType::of<float>(), {2})};  // Cout=1,bias 长度给 2
  ctx.attrs = &StandardAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires bias size to equal Cout=1, got 2"),
            std::string_view::npos);
}

TEST(Conv2dShapeInferTest, MoreThanOneBiasTensorIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("conv2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "conv2d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 3, 3}),
                     MakeType(DType::of<float>(), {1, 1, 2, 2}), MakeType(DType::of<float>(), {1}),
                     MakeType(DType::of<float>(), {1})};
  ctx.attrs = &StandardAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("bias group expects at most 1 tensor, got 2"),
            std::string_view::npos);
}

TEST(Conv2dShapeInferTest, NonPositiveOutputHeightIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("conv2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "conv2d";
  // H=2, KH=3, pad_h=0, stride_h=1: out_h=floor(2-3)/1+1=0<1(W=3,KW=1 保持
  // 合法,只命中高度分支)。
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 2, 3}),
                     MakeType(DType::of<float>(), {1, 1, 3, 1})};
  ctx.attrs = &StandardAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("produces non-positive output height 0"),
            std::string_view::npos);
}

TEST(Conv2dShapeInferTest, StrideLessThanOneIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("conv2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "conv2d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 3, 3}),
                     MakeType(DType::of<float>(), {1, 1, 2, 2})};
  const AttrMap attrs{
      {"stride", std::vector<int64_t>{0, 1}},
      {"padding", std::vector<int64_t>{0, 0}},
      {"groups", int64_t{1}},
  };
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("attribute 'stride' entries must be >= 1, got [0, 1]"),
            std::string_view::npos);
}

TEST(Conv2dShapeInferTest, PaddingLessThanZeroIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("conv2d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "conv2d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 3, 3}),
                     MakeType(DType::of<float>(), {1, 1, 2, 2})};
  const AttrMap attrs{
      {"stride", std::vector<int64_t>{1, 1}},
      {"padding", std::vector<int64_t>{-1, 0}},
      {"groups", int64_t{1}},
  };
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("attribute 'padding' entries must be >= 0, got [-1, 0]"),
            std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 3. eager 数值(ARCH-011 第 3 类)。
// ---------------------------------------------------------------------------

class Conv2dOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(Conv2dOpEagerTest, Float32Stride1Pad0KnownValuesElementwiseCheck) {
  // x 3x3=[1..9],w 2x2=[1,2;3,4],stride1 pad0 无 bias:
  //   out[0,0]=1*1+2*2+4*3+5*4=1+4+12+20=37
  //   out[0,1]=2*1+3*2+5*3+6*4=2+6+15+24=47
  //   out[1,0]=4*1+5*2+7*3+8*4=4+10+21+32=67
  //   out[1,1]=5*1+6*2+8*3+9*4=5+12+24+36=77
  Tensor x = MakeTensorWithShape<float>(Shape({1, 1, 3, 3}),
                                        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F});
  Tensor w = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {1.0F, 2.0F, 3.0F, 4.0F});
  Tensor expected = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {37.0F, 47.0F, 67.0F, 77.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{x, w};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "conv2d";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &StandardAttrs();
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(Conv2dOpEagerTest, Float32Stride2KnownValues) {
  // x 5x5=[1..25](行优先),w=[[1,0],[0,1]](取对角和),stride2 pad0:
  //   out[0,0]=x[0,0]+x[1,1]=1+7=8      out[0,1]=x[0,2]+x[1,3]=3+9=12
  //   out[1,0]=x[2,0]+x[3,1]=11+17=28   out[1,1]=x[2,2]+x[3,3]=13+19=32
  std::vector<float> x_values;
  for (int i = 1; i <= 25; ++i) x_values.push_back(static_cast<float>(i));
  Tensor x = MakeTensorWithShape<float>(Shape({1, 1, 5, 5}), x_values);
  Tensor w = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {1.0F, 0.0F, 0.0F, 1.0F});
  Tensor expected = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {8.0F, 12.0F, 28.0F, 32.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {0.0F, 0.0F, 0.0F, 0.0F});

  const AttrMap attrs{
      {"stride", std::vector<int64_t>{2, 2}},
      {"padding", std::vector<int64_t>{0, 0}},
      {"groups", int64_t{1}},
  };
  std::vector<Tensor> inputs{x, w};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "conv2d";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(Conv2dOpEagerTest, Float32Padding1KnownValues) {
  // x 2x2=[1,2;3,4],w 2x2 全1(窗口求和),pad1 stride1:padded 4x4 零边框,
  // out 3x3 逐窗手算(见文件设计记录):
  //   [1,3,2, 4,10,6, 3,7,4]
  Tensor x = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {1.0F, 2.0F, 3.0F, 4.0F});
  Tensor w = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {1.0F, 1.0F, 1.0F, 1.0F});
  Tensor expected = MakeTensorWithShape<float>(
      Shape({1, 1, 3, 3}), {1.0F, 3.0F, 2.0F, 4.0F, 10.0F, 6.0F, 3.0F, 7.0F, 4.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({1, 1, 3, 3}),
                                          {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});

  const AttrMap attrs{
      {"stride", std::vector<int64_t>{1, 1}},
      {"padding", std::vector<int64_t>{1, 1}},
      {"groups", int64_t{1}},
  };
  std::vector<Tensor> inputs{x, w};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "conv2d";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(Conv2dOpEagerTest, GroupsTwoBandwiseIndependentKnownValues) {
  // Cin=Cout=2,groups=2(cin_per_group=cout_per_group=1),1x1 核:每组各自独立
  // 缩放对应通道,无跨通道混合。channel0=[1,2,3,4]*2=[2,4,6,8];
  // channel1=[5,6,7,8]*3=[15,18,21,24]。
  Tensor x = MakeTensorWithShape<float>(Shape({1, 2, 2, 2}),
                                        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
  Tensor w = MakeTensorWithShape<float>(Shape({2, 1, 1, 1}), {2.0F, 3.0F});
  Tensor expected = MakeTensorWithShape<float>(
      Shape({1, 2, 2, 2}), {2.0F, 4.0F, 6.0F, 8.0F, 15.0F, 18.0F, 21.0F, 24.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({1, 2, 2, 2}),
                                          {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});

  const AttrMap attrs{
      {"stride", std::vector<int64_t>{1, 1}},
      {"padding", std::vector<int64_t>{0, 0}},
      {"groups", int64_t{2}},
  };
  std::vector<Tensor> inputs{x, w};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "conv2d";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(Conv2dOpEagerTest, WithBiasAddsBiasPerOutputChannel) {
  // 复用 Stride1Pad0 已知值(37,47,67,77),bias=[100]:期望逐位 +100。
  Tensor x = MakeTensorWithShape<float>(Shape({1, 1, 3, 3}),
                                        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F});
  Tensor w = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {1.0F, 2.0F, 3.0F, 4.0F});
  Tensor bias = MakeTensorWithShape<float>(Shape({1}), {100.0F});
  Tensor expected =
      MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {137.0F, 147.0F, 167.0F, 177.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{x, w, bias};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "conv2d";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &StandardAttrs();
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(Conv2dOpEagerTest, Float16KnownValuesViaBitLevelConversion) {
  // 复用 Stride1Pad0 已知整数值(fp16 尾数内精确可表示,<2048):37,47,67,77。
  Tensor x = MakeTensorWithShape<float16_t>(
      Shape({1, 1, 3, 3}),
      {float_to_float16(1.0F), float_to_float16(2.0F), float_to_float16(3.0F),
       float_to_float16(4.0F), float_to_float16(5.0F), float_to_float16(6.0F),
       float_to_float16(7.0F), float_to_float16(8.0F), float_to_float16(9.0F)});
  Tensor w = MakeTensorWithShape<float16_t>(
      Shape({1, 1, 2, 2}), {float_to_float16(1.0F), float_to_float16(2.0F), float_to_float16(3.0F),
                            float_to_float16(4.0F)});
  Tensor expected = MakeTensorWithShape<float16_t>(
      Shape({1, 1, 2, 2}), {float_to_float16(37.0F), float_to_float16(47.0F),
                            float_to_float16(67.0F), float_to_float16(77.0F)});
  Tensor out = MakeTensorWithShape<float16_t>(
      Shape({1, 1, 2, 2}), {float16_t{0}, float16_t{0}, float16_t{0}, float16_t{0}});

  std::vector<Tensor> inputs{x, w};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "conv2d";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &StandardAttrs();
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat16)));
}

TEST_F(Conv2dOpEagerTest, BFloat16KnownValuesViaBitLevelConversion) {
  // 与 fp16 用例同一组整数值(<256,bf16 7 位尾数内精确可表示)。
  Tensor x = MakeTensorWithShape<bfloat16_t>(
      Shape({1, 1, 3, 3}),
      {float_to_bfloat16(1.0F), float_to_bfloat16(2.0F), float_to_bfloat16(3.0F),
       float_to_bfloat16(4.0F), float_to_bfloat16(5.0F), float_to_bfloat16(6.0F),
       float_to_bfloat16(7.0F), float_to_bfloat16(8.0F), float_to_bfloat16(9.0F)});
  Tensor w = MakeTensorWithShape<bfloat16_t>(Shape({1, 1, 2, 2}),
                                             {float_to_bfloat16(1.0F), float_to_bfloat16(2.0F),
                                              float_to_bfloat16(3.0F), float_to_bfloat16(4.0F)});
  Tensor expected = MakeTensorWithShape<bfloat16_t>(
      Shape({1, 1, 2, 2}), {float_to_bfloat16(37.0F), float_to_bfloat16(47.0F),
                            float_to_bfloat16(67.0F), float_to_bfloat16(77.0F)});
  Tensor out = MakeTensorWithShape<bfloat16_t>(
      Shape({1, 1, 2, 2}), {bfloat16_t{0}, bfloat16_t{0}, bfloat16_t{0}, bfloat16_t{0}});

  std::vector<Tensor> inputs{x, w};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "conv2d";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &StandardAttrs();
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kBFloat16)));
}

// ---------------------------------------------------------------------------
// 4. 图编译路径:standard_pipeline(frame::runtime::compile 内部调用)+ cpu
//    执行 == 直接 Backend::launch(eager)结果 == 手算参考值。
// ---------------------------------------------------------------------------

class Conv2dOpGraphCompileTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(Conv2dOpGraphCompileTest, CompiledGraphMatchesDirectKernelAndHandComputedReference) {
  Graph graph("conv2d_compile_smoke");
  Value* x_in = graph.add_graph_input(MakeType(DType::of<float>(), {1, 1, 3, 3})).value();
  Value* w_in = graph.add_graph_input(MakeType(DType::of<float>(), {1, 1, 2, 2})).value();
  Value* bias_in = graph.add_graph_input(MakeType(DType::of<float>(), {1})).value();
  const AttrMap attrs{
      {"stride", std::vector<int64_t>{1, 1}},
      {"padding", std::vector<int64_t>{0, 0}},
      {"groups", int64_t{1}},
  };
  const Result<Node*> conv_node =
      create_node_with_inferred_types(graph, "conv2d", {x_in, w_in, bias_in}, attrs);
  ASSERT_TRUE(conv_node.is_ok()) << conv_node.status().message();
  ASSERT_TRUE(graph.mark_output(conv_node.value(), 0).is_ok());

  const Result<std::shared_ptr<Executable>> executable =
      frame::runtime::compile(graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  Tensor x = MakeTensorWithShape<float>(Shape({1, 1, 3, 3}),
                                        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F});
  Tensor w = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {1.0F, 2.0F, 3.0F, 4.0F});
  Tensor bias = MakeTensorWithShape<float>(Shape({1}), {100.0F});
  std::vector<Tensor> inputs{x, w, bias};

  const Result<std::vector<Tensor>> compiled_outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(compiled_outputs.is_ok()) << compiled_outputs.status().message();

  // ① 与直接 Backend::launch(eager,不经 pass pipeline)结果核对:同一组输入
  // 直接调用 conv2d kernel。
  Tensor out = MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {0.0F, 0.0F, 0.0F, 0.0F});
  std::vector<Tensor> launch_inputs{x, w, bias};
  std::vector<Tensor> launch_outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "conv2d";
  invocation.inputs = launch_inputs;
  invocation.outputs = launch_outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;
  const Status launch_status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(launch_status.is_ok()) << launch_status.message();
  EXPECT_TRUE(tensor_all_close(compiled_outputs.value()[0], launch_outputs[0],
                               default_tolerance(DTypeCode::kFloat32)));

  // ② 与手算参考值核对(§eager 数值小节 WithBiasAddsBiasPerOutputChannel 同一
  // 组数据):保证两条路径不是共同偏离真值。
  Tensor expected =
      MakeTensorWithShape<float>(Shape({1, 1, 2, 2}), {137.0F, 147.0F, 167.0F, 177.0F});
  EXPECT_TRUE(tensor_all_close(compiled_outputs.value()[0], expected,
                               default_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 5. conv2d_cpu_kernel 自身的防御性拒绝路径。
// ---------------------------------------------------------------------------

struct Conv2dOpNameTag {
  static constexpr std::string_view kOpName = "conv2d";
};
using Conv2dOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<Conv2dOpNameTag>;

TEST_F(Conv2dOpKernelTest, RejectsUnsupportedDtypeInt32) {
  Tensor x = MakeTensor<std::int32_t>(Shape({1, 1, 3, 3}));
  Tensor w = MakeTensor<std::int32_t>(Shape({1, 1, 2, 2}));
  Tensor out = MakeTensor<std::int32_t>(Shape({1, 1, 2, 2}));

  const AttrMap attrs{
      {"stride", std::vector<int64_t>{1, 1}},
      {"padding", std::vector<int64_t>{0, 0}},
      {"groups", int64_t{1}},
  };
  std::vector<Tensor> inputs{x, w};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.attrs = &attrs;
  ctx.device = cpu_device();

  const Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("does not support dtype 'int32'"), std::string_view::npos);
}

TEST_F(Conv2dOpKernelTest, RejectsOutShapeMismatch) {
  Tensor x = MakeTensor<float>(Shape({1, 1, 3, 3}));
  Tensor w = MakeTensor<float>(Shape({1, 1, 2, 2}));
  Tensor out = MakeTensor<float>(Shape({1, 1, 3, 3}));  // 期望 [1,1,2,2],故意给错

  const AttrMap attrs{
      {"stride", std::vector<int64_t>{1, 1}},
      {"padding", std::vector<int64_t>{0, 0}},
      {"groups", int64_t{1}},
  };
  std::vector<Tensor> inputs{x, w};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.attrs = &attrs;
  ctx.device = cpu_device();

  const Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("requires out shape to match the convolution result"),
            std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 6. 解析梯度 ≡ 数值微分(x/w/bias 三路,BUILD-011 专款放宽一档)。
// ---------------------------------------------------------------------------

class Conv2dGradientTest : public frame::ops::testing::ElementwiseEagerTestBase {
 protected:
  Tensor MakeFilledTensor(const Shape& shape, float start, float step) {
    Tensor tensor = Tensor::empty(shape, DType::of<float>(), device_, *allocator_).value();
    float* data = tensor.data<float>();
    for (int64_t i = 0; i < tensor.numel(); ++i) {
      data[i] = start + static_cast<float>(i) * step;
    }
    return tensor;
  }
};

// x[1,1,3,3],w[1,1,2,2](+可选 bias[1])->conv2d->sum(axes=[])作标量 loss。
Graph BuildConv2dLossGraph(bool with_bias) {
  Graph graph("conv2d_loss");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {1, 1, 3, 3})).value();
  Value* w = graph.add_graph_input(MakeType(DType::of<float>(), {1, 1, 2, 2})).value();
  std::vector<Value*> conv_inputs{x, w};
  if (with_bias) {
    Value* bias = graph.add_graph_input(MakeType(DType::of<float>(), {1})).value();
    conv_inputs.push_back(bias);
  }
  const AttrMap conv_attrs{
      {"stride", std::vector<int64_t>{1, 1}},
      {"padding", std::vector<int64_t>{0, 0}},
      {"groups", int64_t{1}},
  };
  Node* conv_node =
      create_node_with_inferred_types(graph, "conv2d", conv_inputs, conv_attrs).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {conv_node->output(0)}, sum_attrs).value();
  const Status mark_status = graph.mark_output(sum_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

// 中心差分步长:与 tests/cpp/compiler/test_autograd.cpp 的 kCentralDifferenceH
// 同一实测区间(fp32,BUILD-011 建议 1e-3 量级起调,该文件已就 conv 输入量级
// 相近的 add/mul/matmul 等用例实测 1e-2 稳定通过 relaxed_tolerance);本文件
// conv2d 用例复用同一取值,不重新定案。
constexpr double kConv2dCentralDifferenceH = 1e-2;

TEST_F(Conv2dGradientTest, GradientMatchesNumericForXAndWeightNoBias) {
  const Graph forward = BuildConv2dLossGraph(/*with_bias=*/false);
  Tensor x = MakeFilledTensor(Shape({1, 1, 3, 3}), 0.3F, 0.1F);
  Tensor w = MakeFilledTensor(Shape({1, 1, 2, 2}), 0.2F, -0.05F);
  std::vector<Tensor> inputs{x, w};
  const std::vector<int32_t> wrt{0, 1};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kConv2dCentralDifferenceH));
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 1, inputs, kConv2dCentralDifferenceH));
}

TEST_F(Conv2dGradientTest, GradientMatchesNumericForBias) {
  const Graph forward = BuildConv2dLossGraph(/*with_bias=*/true);
  Tensor x = MakeFilledTensor(Shape({1, 1, 3, 3}), 0.3F, 0.1F);
  Tensor w = MakeFilledTensor(Shape({1, 1, 2, 2}), 0.2F, -0.05F);
  Tensor bias = MakeFilledTensor(Shape({1}), 0.5F, 0.0F);
  std::vector<Tensor> inputs{x, w, bias};
  const std::vector<int32_t> wrt{0, 1, 2};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 2, inputs, kConv2dCentralDifferenceH));
}

}  // namespace
