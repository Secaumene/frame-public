// conv1d 算子测试(schema + cpu kernel + decomposition + 解析梯度 ≡ 数值微分,
// src/ops/schemas/conv.cpp、src/backends/cpu/kernels/conv.cpp 已实化的行为,
// M21 批3 T4)。conv1d 与 conv2d 同构(x[N,Cin,L] + w[Cout,Cin/g,K] + 可选
// bias[Cout],attrs stride/padding/groups 均 kInt64 标量而非数组);裁决点②
// (docs/plan/2026-07-18-batch3-m21-conv.md 第1.1节):CUDA 侧经 decomposition
// (reshape→conv2d(H=1)→reshape)落 cuDNN,CPU 侧另有直循环参考 kernel
// (ARCH-041),二者须数值一致——本文件专设一节验证。
//   1. OpRegistry::find("conv1d") 的 schema 字段、shape_infer()/
//      decomposition()/gradient() 三函数指针状态(decomposition 非空);
//   2. infer_conv1d_shape 的合法路径 + 负例抽查(x 秩错、Cin%groups、输出维<1、
//      bias 长度错);
//   3. eager 数值路径(ARCH-011 第 3 类):stride1 pad0 无 bias 已知值、
//      stride2、padding1、groups=2 条带核对、bias 已知值、fp16/bf16(复用
//      stride1 pad0 已知值);
//   4. decomposition 微图:schema->decomposition()(ctx) 产出的
//      reshape→conv2d→reshape 微图独立编译执行,数值与直接 conv1d cpu kernel
//      一致(裁决点②数值等价性验证);
//   5. conv1d_cpu_kernel 自身的防御性拒绝路径;
//   6. 解析梯度 ≡ 数值微分(x/w/bias 三路)。
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

TEST(Conv1dOpSchemaTest, RegisteredAndFindable) {
  const OpSchema* schema = OpRegistry::instance().find("conv1d");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->name(), "conv1d");
}

TEST(Conv1dOpSchemaTest, HasTwoFixedInputsVariadicBiasAndThreeScalarAttrs) {
  const OpSchema* schema = OpRegistry::instance().find("conv1d");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 2U);
  EXPECT_TRUE(schema->has_variadic_inputs());
  EXPECT_EQ(schema->min_input_count(), 2);
  EXPECT_EQ(schema->outputs().size(), 1U);
  ASSERT_EQ(schema->attrs().size(), 3U);
  EXPECT_EQ(schema->attrs()[0].name, "stride");
  EXPECT_EQ(schema->attrs()[0].type, frame::ir::AttrType::kInt64);  // 标量,非数组(区别于 conv2d)
  EXPECT_EQ(schema->attrs()[1].name, "padding");
  EXPECT_EQ(schema->attrs()[1].type, frame::ir::AttrType::kInt64);
  EXPECT_EQ(schema->attrs()[2].name, "groups");
}

TEST(Conv1dOpSchemaTest, HasShapeInferDecompositionAndGradient) {
  const OpSchema* schema = OpRegistry::instance().find("conv1d");
  ASSERT_NE(schema, nullptr);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_NE(schema->decomposition(), nullptr);  // 裁决点②:CUDA 侧走 decomposition
  EXPECT_NE(schema->gradient(), nullptr);
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例(合法路径 + 负例抽查)。
// ---------------------------------------------------------------------------

const AttrMap& StandardAttrs() {
  static const AttrMap attrs{
      {"stride", int64_t{1}},
      {"padding", int64_t{0}},
      {"groups", int64_t{1}},
  };
  return attrs;
}

TEST(Conv1dShapeInferTest, BasicNoBiasProducesExpectedOutputShape) {
  const OpSchema* schema = OpRegistry::instance().find("conv1d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "conv1d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 4}),
                     MakeType(DType::of<float>(), {1, 1, 2})};
  ctx.attrs = &StandardAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({1, 1, 3}));
}

TEST(Conv1dShapeInferTest, RankThreeRequiredForXRejectedWithActualRank) {
  const OpSchema* schema = OpRegistry::instance().find("conv1d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "conv1d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1}), MakeType(DType::of<float>(), {1, 1, 2})};
  ctx.attrs = &StandardAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires x to be rank-3 [N, Cin, L], got rank 2"),
            std::string_view::npos);
}

TEST(Conv1dShapeInferTest, CinNotDivisibleByGroupsIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("conv1d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "conv1d";
  // Cin=3, groups=2:3%2!=0(Cout=2/groups=2==0,只命中 Cin 分支)。
  ctx.input_types = {MakeType(DType::of<float>(), {1, 3, 5}),
                     MakeType(DType::of<float>(), {2, 1, 2})};
  const AttrMap attrs{{"stride", int64_t{1}}, {"padding", int64_t{0}}, {"groups", int64_t{2}}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("Cin=3 is not divisible by groups=2"),
            std::string_view::npos);
}

TEST(Conv1dShapeInferTest, NonPositiveOutputLengthIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("conv1d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "conv1d";
  // 手算:L=2, K=3, pad=0, stride=1: out_l=floor(2-3)/1+1=0<1。
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 2}),
                     MakeType(DType::of<float>(), {1, 1, 3})};
  ctx.attrs = &StandardAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("produces non-positive output length 0"),
            std::string_view::npos);
}

TEST(Conv1dShapeInferTest, BiasLengthMismatchWithCoutIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("conv1d");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "conv1d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 4}),
                     MakeType(DType::of<float>(), {1, 1, 2}),
                     MakeType(DType::of<float>(), {2})};  // Cout=1,bias 长度给 2
  ctx.attrs = &StandardAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires bias size to equal Cout=1, got 2"),
            std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 3. eager 数值(ARCH-011 第 3 类)。
// ---------------------------------------------------------------------------

class Conv1dOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(Conv1dOpEagerTest, Float32Stride1Pad0KnownValues) {
  // 手算:x=[1,2,3,4,5],w=[1,2],stride1 pad0:
  //   out[0]=1*1+2*2=5   out[1]=2*1+3*2=8   out[2]=3*1+4*2=11   out[3]=4*1+5*2=14
  Tensor x = MakeTensorWithShape<float>(Shape({1, 1, 5}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
  Tensor w = MakeTensorWithShape<float>(Shape({1, 1, 2}), {1.0F, 2.0F});
  Tensor expected = MakeTensorWithShape<float>(Shape({1, 1, 4}), {5.0F, 8.0F, 11.0F, 14.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({1, 1, 4}), {0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{x, w};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "conv1d";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &StandardAttrs();
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(Conv1dOpEagerTest, Float32Stride2KnownValues) {
  // 手算:x=[1..7],w=[1,2],stride2 pad0:out_l=floor((7-2)/2)+1=3。
  //   out[0]=x0+2x1=1+4=5  out[1]=x2+2x3=3+8=11  out[2]=x4+2x5=5+12=17
  Tensor x =
      MakeTensorWithShape<float>(Shape({1, 1, 7}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F});
  Tensor w = MakeTensorWithShape<float>(Shape({1, 1, 2}), {1.0F, 2.0F});
  Tensor expected = MakeTensorWithShape<float>(Shape({1, 1, 3}), {5.0F, 11.0F, 17.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({1, 1, 3}), {0.0F, 0.0F, 0.0F});

  const AttrMap attrs{{"stride", int64_t{2}}, {"padding", int64_t{0}}, {"groups", int64_t{1}}};
  std::vector<Tensor> inputs{x, w};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "conv1d";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(Conv1dOpEagerTest, Float32Padding1KnownValues) {
  // x=[1,2,3],w=[1,1](窗口求和),pad1 stride1:padded=[0,1,2,3,0],out_l=4。
  //   out=[0+1, 1+2, 2+3, 3+0]=[1,3,5,3]
  Tensor x = MakeTensorWithShape<float>(Shape({1, 1, 3}), {1.0F, 2.0F, 3.0F});
  Tensor w = MakeTensorWithShape<float>(Shape({1, 1, 2}), {1.0F, 1.0F});
  Tensor expected = MakeTensorWithShape<float>(Shape({1, 1, 4}), {1.0F, 3.0F, 5.0F, 3.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({1, 1, 4}), {0.0F, 0.0F, 0.0F, 0.0F});

  const AttrMap attrs{{"stride", int64_t{1}}, {"padding", int64_t{1}}, {"groups", int64_t{1}}};
  std::vector<Tensor> inputs{x, w};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "conv1d";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(Conv1dOpEagerTest, GroupsTwoBandwiseIndependentKnownValues) {
  // Cin=Cout=2,groups=2,K=1:channel0=[1,2,3,4]*2=[2,4,6,8];
  // channel1=[5,6,7,8]*3=[15,18,21,24],无跨通道混合。
  Tensor x = MakeTensorWithShape<float>(Shape({1, 2, 4}),
                                        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
  Tensor w = MakeTensorWithShape<float>(Shape({2, 1, 1}), {2.0F, 3.0F});
  Tensor expected = MakeTensorWithShape<float>(
      Shape({1, 2, 4}), {2.0F, 4.0F, 6.0F, 8.0F, 15.0F, 18.0F, 21.0F, 24.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({1, 2, 4}),
                                          {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});

  const AttrMap attrs{{"stride", int64_t{1}}, {"padding", int64_t{0}}, {"groups", int64_t{2}}};
  std::vector<Tensor> inputs{x, w};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "conv1d";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(Conv1dOpEagerTest, WithBiasAddsBiasPerOutputChannel) {
  // 复用 Stride1Pad0 已知值(5,8,11,14),bias=[100]:期望逐位 +100。
  Tensor x = MakeTensorWithShape<float>(Shape({1, 1, 5}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
  Tensor w = MakeTensorWithShape<float>(Shape({1, 1, 2}), {1.0F, 2.0F});
  Tensor bias = MakeTensorWithShape<float>(Shape({1}), {100.0F});
  Tensor expected = MakeTensorWithShape<float>(Shape({1, 1, 4}), {105.0F, 108.0F, 111.0F, 114.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({1, 1, 4}), {0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{x, w, bias};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "conv1d";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &StandardAttrs();
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(Conv1dOpEagerTest, Float16KnownValuesViaBitLevelConversion) {
  // 复用 Stride1Pad0 已知整数值(fp16 尾数内精确可表示):5,8,11,14。
  Tensor x = MakeTensorWithShape<float16_t>(
      Shape({1, 1, 5}), {float_to_float16(1.0F), float_to_float16(2.0F), float_to_float16(3.0F),
                         float_to_float16(4.0F), float_to_float16(5.0F)});
  Tensor w = MakeTensorWithShape<float16_t>(Shape({1, 1, 2}),
                                            {float_to_float16(1.0F), float_to_float16(2.0F)});
  Tensor expected = MakeTensorWithShape<float16_t>(
      Shape({1, 1, 4}), {float_to_float16(5.0F), float_to_float16(8.0F), float_to_float16(11.0F),
                         float_to_float16(14.0F)});
  Tensor out = MakeTensorWithShape<float16_t>(
      Shape({1, 1, 4}), {float16_t{0}, float16_t{0}, float16_t{0}, float16_t{0}});

  std::vector<Tensor> inputs{x, w};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "conv1d";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &StandardAttrs();
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat16)));
}

TEST_F(Conv1dOpEagerTest, BFloat16KnownValuesViaBitLevelConversion) {
  Tensor x = MakeTensorWithShape<bfloat16_t>(
      Shape({1, 1, 5}), {float_to_bfloat16(1.0F), float_to_bfloat16(2.0F), float_to_bfloat16(3.0F),
                         float_to_bfloat16(4.0F), float_to_bfloat16(5.0F)});
  Tensor w = MakeTensorWithShape<bfloat16_t>(Shape({1, 1, 2}),
                                             {float_to_bfloat16(1.0F), float_to_bfloat16(2.0F)});
  Tensor expected = MakeTensorWithShape<bfloat16_t>(
      Shape({1, 1, 4}), {float_to_bfloat16(5.0F), float_to_bfloat16(8.0F), float_to_bfloat16(11.0F),
                         float_to_bfloat16(14.0F)});
  Tensor out = MakeTensorWithShape<bfloat16_t>(
      Shape({1, 1, 4}), {bfloat16_t{0}, bfloat16_t{0}, bfloat16_t{0}, bfloat16_t{0}});

  std::vector<Tensor> inputs{x, w};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "conv1d";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &StandardAttrs();
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kBFloat16)));
}

// ---------------------------------------------------------------------------
// 4. decomposition 微图(裁决点②)数值与直接 cpu kernel 一致。
// ---------------------------------------------------------------------------

class Conv1dDecompositionTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(Conv1dDecompositionTest, MicrographProducesSameResultAsDirectCpuKernel) {
  const OpSchema* schema = OpRegistry::instance().find("conv1d");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->decomposition(), nullptr);

  NodeContext ctx;
  ctx.op = "conv1d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 4}),
                     MakeType(DType::of<float>(), {1, 1, 2}), MakeType(DType::of<float>(), {1})};
  const AttrMap attrs{{"stride", int64_t{1}}, {"padding", int64_t{0}}, {"groups", int64_t{1}}};
  ctx.attrs = &attrs;

  const Result<Graph> micrograph = schema->decomposition()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();

  const Result<std::shared_ptr<Executable>> executable =
      frame::runtime::compile(micrograph.value(), frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  // x=[1,2,3,4],w=[1,2],bias=[10]:out=[1*1+2*2,2*1+3*2,3*1+4*2]+10
  //   =[5,8,11]+10=[15,18,21]。
  Tensor x = MakeTensorWithShape<float>(Shape({1, 1, 4}), {1.0F, 2.0F, 3.0F, 4.0F});
  Tensor w = MakeTensorWithShape<float>(Shape({1, 1, 2}), {1.0F, 2.0F});
  Tensor bias = MakeTensorWithShape<float>(Shape({1}), {10.0F});
  std::vector<Tensor> inputs{x, w, bias};

  const Result<std::vector<Tensor>> decomposed_outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(decomposed_outputs.is_ok()) << decomposed_outputs.status().message();

  Tensor expected = MakeTensorWithShape<float>(Shape({1, 1, 3}), {15.0F, 18.0F, 21.0F});
  EXPECT_TRUE(tensor_all_close(decomposed_outputs.value()[0], expected,
                               default_tolerance(DTypeCode::kFloat32)));

  // 与直接 conv1d cpu kernel(Backend::launch,不经 decomposition)结果核对
  // (裁决点②数值等价性:CPU 侧直循环参考 kernel 与 CUDA 侧使用的 decomposition
  // 微图必须一致)。
  Tensor out = MakeTensorWithShape<float>(Shape({1, 1, 3}), {0.0F, 0.0F, 0.0F});
  std::vector<Tensor> direct_inputs{x, w, bias};
  std::vector<Tensor> direct_outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "conv1d";
  invocation.inputs = direct_inputs;
  invocation.outputs = direct_outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;
  const Status launch_status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(launch_status.is_ok()) << launch_status.message();
  EXPECT_TRUE(tensor_all_close(decomposed_outputs.value()[0], direct_outputs[0],
                               default_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 5. conv1d_cpu_kernel 自身的防御性拒绝路径。
// ---------------------------------------------------------------------------

struct Conv1dOpNameTag {
  static constexpr std::string_view kOpName = "conv1d";
};
using Conv1dOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<Conv1dOpNameTag>;

TEST_F(Conv1dOpKernelTest, RejectsUnsupportedDtypeInt32) {
  Tensor x = MakeTensor<std::int32_t>(Shape({1, 1, 4}));
  Tensor w = MakeTensor<std::int32_t>(Shape({1, 1, 2}));
  Tensor out = MakeTensor<std::int32_t>(Shape({1, 1, 3}));

  const AttrMap attrs{{"stride", int64_t{1}}, {"padding", int64_t{0}}, {"groups", int64_t{1}}};
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

TEST_F(Conv1dOpKernelTest, RejectsOutShapeMismatch) {
  Tensor x = MakeTensor<float>(Shape({1, 1, 4}));
  Tensor w = MakeTensor<float>(Shape({1, 1, 2}));
  Tensor out = MakeTensor<float>(Shape({1, 1, 4}));  // 期望 [1,1,3],故意给错

  const AttrMap attrs{{"stride", int64_t{1}}, {"padding", int64_t{0}}, {"groups", int64_t{1}}};
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
// 6. 解析梯度 ≡ 数值微分(x/w/bias 三路)。
// ---------------------------------------------------------------------------

class Conv1dGradientTest : public frame::ops::testing::ElementwiseEagerTestBase {
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

Graph BuildConv1dLossGraph(bool with_bias) {
  Graph graph("conv1d_loss");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {1, 1, 4})).value();
  Value* w = graph.add_graph_input(MakeType(DType::of<float>(), {1, 1, 2})).value();
  std::vector<Value*> conv_inputs{x, w};
  if (with_bias) {
    Value* bias = graph.add_graph_input(MakeType(DType::of<float>(), {1})).value();
    conv_inputs.push_back(bias);
  }
  const AttrMap conv_attrs{{"stride", int64_t{1}}, {"padding", int64_t{0}}, {"groups", int64_t{1}}};
  Node* conv_node =
      create_node_with_inferred_types(graph, "conv1d", conv_inputs, conv_attrs).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {conv_node->output(0)}, sum_attrs).value();
  const Status mark_status = graph.mark_output(sum_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

// 与 test_conv2d.cpp 同一实测区间(fp32,BUILD-011 建议 1e-3 量级起调)。
constexpr double kConv1dCentralDifferenceH = 1e-2;

TEST_F(Conv1dGradientTest, GradientMatchesNumericForXAndWeightNoBias) {
  const Graph forward = BuildConv1dLossGraph(/*with_bias=*/false);
  Tensor x = MakeFilledTensor(Shape({1, 1, 4}), 0.3F, 0.1F);
  Tensor w = MakeFilledTensor(Shape({1, 1, 2}), 0.2F, -0.05F);
  std::vector<Tensor> inputs{x, w};
  const std::vector<int32_t> wrt{0, 1};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kConv1dCentralDifferenceH));
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 1, inputs, kConv1dCentralDifferenceH));
}

TEST_F(Conv1dGradientTest, GradientMatchesNumericForBias) {
  const Graph forward = BuildConv1dLossGraph(/*with_bias=*/true);
  Tensor x = MakeFilledTensor(Shape({1, 1, 4}), 0.3F, 0.1F);
  Tensor w = MakeFilledTensor(Shape({1, 1, 2}), 0.2F, -0.05F);
  Tensor bias = MakeFilledTensor(Shape({1}), 0.5F, 0.0F);
  std::vector<Tensor> inputs{x, w, bias};
  const std::vector<int32_t> wrt{0, 1, 2};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 2, inputs, kConv1dCentralDifferenceH));
}

}  // namespace
