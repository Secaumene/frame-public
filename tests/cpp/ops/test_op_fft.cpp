// rfft/irfft 算子测试(schema + cpu kernel + 解析梯度 ≡ 数值微分,
// src/ops/schemas/fft.cpp、src/backends/cpu/kernels/fft.cpp 已实化的行为,
// M23 批5 T3,docs/plan/2026-07-21-batch5-m23-fft.md §1.1–1.3/1.6、验收硬门
// §2/3)。rfft(x[...,n]) -> [...,k,2](k=n/2+1,末轴 2=(re,im) 交错);
// irfft(z[...,k,2]; n) -> [...,n],二者互为逆变换(numpy 口径,不归一化 /
// 归一化 1/n)。
//   1. schema 字段/三函数指针状态(rfft、irfft 各一);
//   2. shape_infer 合法路径 + 负例(rfft:输入数/rank<1/n<2;irfft:输入数/
//      rank<2/末轴!=2/缺 n/k!=n/2+1);
//   3. eager 数值:解析谱 golden(冲激/常量/单频,偶 n=4、奇 n=5 各三例,参考值
//      按 DFT 定义手算,见各用例注释);
//   4. rfft_cpu_kernel/irfft_cpu_kernel 自身的防御性拒绝路径(非 fp32);
//   5. 图编译路径(经 runtime::compile,非仅 eager);
//   6. roundtrip:irfft(rfft(x), n)≡x(fp32 容差,偶/奇 n 各一,另加一个批量
//      leading 维用例验证 kernel 的 batch 遍历);
//   7. 解析梯度 ≡ 数值微分(BUILD-011 容差,rfft/irfft 各偶/奇 n 两组共四例)。
#include <cmath>
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
#include <frame/ir/node.h>
#include <frame/ops/graph_builder.h>
#include <frame/ops/kernel_registry.h>
#include <frame/ops/op_registry.h>
#include <frame/runtime/compile.h>

#include "../common/tolerance.h"
#include "elementwise_op_test_helpers.h"
#include "gradient_check_test_helpers.h"

namespace {

using frame::cpu_device;
using frame::DType;
using frame::DTypeCode;
using frame::ErrorCode;
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
// 1. schema 断言。
// ---------------------------------------------------------------------------

TEST(FftOpSchemaTest, RfftRegisteredWithOneInputOneOutputNoAttrs) {
  const OpSchema* schema = OpRegistry::instance().find("rfft");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 1U);
  EXPECT_EQ(schema->outputs().size(), 1U);
  EXPECT_EQ(schema->attrs().size(), 0U);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);
}

TEST(FftOpSchemaTest, IrfftRegisteredWithOneInputOneOutputOneAttr) {
  const OpSchema* schema = OpRegistry::instance().find("irfft");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 1U);
  EXPECT_EQ(schema->outputs().size(), 1U);
  ASSERT_EQ(schema->attrs().size(), 1U);
  EXPECT_EQ(schema->attrs()[0].name, "n");
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

TEST(RfftShapeInferTest, EvenNProducesKTimesTwoShape) {
  const OpSchema* schema = OpRegistry::instance().find("rfft");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "rfft";
  ctx.input_types = {MakeType(DType::of<float>(), {4})};
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({3, 2}));
}

TEST(RfftShapeInferTest, OddNProducesKTimesTwoShape) {
  const OpSchema* schema = OpRegistry::instance().find("rfft");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "rfft";
  ctx.input_types = {MakeType(DType::of<float>(), {5})};
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({3, 2}));
}

TEST(RfftShapeInferTest, LeadingBatchDimsArePreserved) {
  const OpSchema* schema = OpRegistry::instance().find("rfft");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "rfft";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 6})};
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({2, 4, 2}));
}

TEST(RfftShapeInferTest, WrongInputCountIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("rfft");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "rfft";
  ctx.input_types = {MakeType(DType::of<float>(), {4}), MakeType(DType::of<float>(), {4})};
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("expects 1 input, got 2"), std::string_view::npos);
}

TEST(RfftShapeInferTest, RankZeroIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("rfft");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "rfft";
  ctx.input_types = {MakeType(DType::of<float>(), {})};
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires x to have rank >= 1, got rank 0"),
            std::string_view::npos);
}

TEST(RfftShapeInferTest, NLessThanTwoIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("rfft");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "rfft";
  ctx.input_types = {MakeType(DType::of<float>(), {1})};
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires the last dimension (n) to be >= 2, got n=1"),
            std::string_view::npos);
}

TEST(IrfftShapeInferTest, EvenNProducesNShape) {
  const OpSchema* schema = OpRegistry::instance().find("irfft");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "irfft";
  ctx.input_types = {MakeType(DType::of<float>(), {3, 2})};
  const AttrMap attrs{{"n", int64_t{4}}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({4}));
}

TEST(IrfftShapeInferTest, OddNProducesNShape) {
  const OpSchema* schema = OpRegistry::instance().find("irfft");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "irfft";
  ctx.input_types = {MakeType(DType::of<float>(), {3, 2})};
  const AttrMap attrs{{"n", int64_t{5}}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({5}));
}

TEST(IrfftShapeInferTest, LeadingBatchDimsArePreserved) {
  const OpSchema* schema = OpRegistry::instance().find("irfft");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "irfft";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 4, 2})};
  const AttrMap attrs{{"n", int64_t{6}}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({2, 6}));
}

TEST(IrfftShapeInferTest, WrongInputCountIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("irfft");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "irfft";
  ctx.input_types = {MakeType(DType::of<float>(), {3, 2}), MakeType(DType::of<float>(), {3, 2})};
  const AttrMap attrs{{"n", int64_t{4}}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("expects 1 input, got 2"), std::string_view::npos);
}

TEST(IrfftShapeInferTest, RankLessThanTwoIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("irfft");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "irfft";
  ctx.input_types = {MakeType(DType::of<float>(), {3})};
  const AttrMap attrs{{"n", int64_t{4}}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find(
                "requires z to have rank >= 2 (trailing axes are [k, 2]), got rank 1"),
            std::string_view::npos);
}

TEST(IrfftShapeInferTest, LastDimensionNotTwoIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("irfft");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "irfft";
  ctx.input_types = {MakeType(DType::of<float>(), {3, 3})};
  const AttrMap attrs{{"n", int64_t{4}}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(
      result.status().message().find("requires the last dimension to be 2 (interleaved re/im)"),
      std::string_view::npos);
}

TEST(IrfftShapeInferTest, MissingNIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("irfft");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "irfft";
  ctx.input_types = {MakeType(DType::of<float>(), {3, 2})};
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("missing required attribute 'n'"),
            std::string_view::npos);
}

TEST(IrfftShapeInferTest, KNotMatchingNIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("irfft");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "irfft";
  ctx.input_types = {MakeType(DType::of<float>(), {4, 2})};  // k=4
  const AttrMap attrs{{"n", int64_t{4}}};                    // expected k=3
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires k=n/2+1 for attribute n=4, expected k=3, "
                                           "got k=4"),
            std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 3. eager 数值:解析谱 golden(冲激/常量/单频,偶 n=4、奇 n=5 各三例;参考值
// 按 DFT 定义 X_k = sum_j x_j * exp(-2*pi*i*j*k/n) 手算,forward 不归一化)。
// ---------------------------------------------------------------------------

class RfftOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(RfftOpEagerTest, ImpulseEvenN) {
  // x=[1,0,0,0](n=4):X_k=x_0=1(对全部 k),故三个频点均为 (1,0)。
  Tensor x = MakeTensorWithShape<float>(Shape({4}), {1.0F, 0.0F, 0.0F, 0.0F});
  Tensor expected = MakeTensorWithShape<float>(Shape({3, 2}), {1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({3, 2}), {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "rfft";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(RfftOpEagerTest, ConstantEvenN) {
  // x=[2,2,2,2](n=4):X_0=sum=8;X_1=X_2=0(非零频点上常量信号的 DFT 恒为 0)。
  Tensor x = MakeTensorWithShape<float>(Shape({4}), {2.0F, 2.0F, 2.0F, 2.0F});
  Tensor expected = MakeTensorWithShape<float>(Shape({3, 2}), {8.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({3, 2}), {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "rfft";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(RfftOpEagerTest, SingleFrequencyEvenN) {
  // x=[1,0,-1,0]=cos(2*pi*j/4)(n=4):单频信号能量集中在 k=1,
  // X_1=n/2=2(实数,无 Nyquist 泄漏);X_0=X_2(Nyquist)=0。
  Tensor x = MakeTensorWithShape<float>(Shape({4}), {1.0F, 0.0F, -1.0F, 0.0F});
  Tensor expected = MakeTensorWithShape<float>(Shape({3, 2}), {0.0F, 0.0F, 2.0F, 0.0F, 0.0F, 0.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({3, 2}), {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "rfft";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(RfftOpEagerTest, ImpulseOddN) {
  // x=[1,0,0,0,0](n=5,k=3):同偶 n 冲激例,三个频点均为 (1,0)。
  Tensor x = MakeTensorWithShape<float>(Shape({5}), {1.0F, 0.0F, 0.0F, 0.0F, 0.0F});
  Tensor expected = MakeTensorWithShape<float>(Shape({3, 2}), {1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({3, 2}), {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "rfft";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(RfftOpEagerTest, ConstantOddN) {
  // x=[2,2,2,2,2](n=5):X_0=sum=10;X_1=X_2=0。
  Tensor x = MakeTensorWithShape<float>(Shape({5}), {2.0F, 2.0F, 2.0F, 2.0F, 2.0F});
  Tensor expected =
      MakeTensorWithShape<float>(Shape({3, 2}), {10.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({3, 2}), {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "rfft";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(RfftOpEagerTest, SingleFrequencyOddN) {
  // x_j=cos(2*pi*j/5)(n=5,j=0..4):偶对称信号(x_j=x_{n-j}),DFT 全实数。
  // X_1=n/2=2.5(单频能量,无 Nyquist 位可泄漏,奇 n 无 Nyquist 频点);
  // X_0=X_2=0(解析推导:sum_j cos(2*pi*j/5)=0,cos 正交性消去 X_2)。
  Tensor x = MakeTensorWithShape<float>(
      Shape({5}), {1.0F, 0.30901699F, -0.80901699F, -0.80901699F, 0.30901699F});
  Tensor expected = MakeTensorWithShape<float>(Shape({3, 2}), {0.0F, 0.0F, 2.5F, 0.0F, 0.0F, 0.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({3, 2}), {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "rfft";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 4. rfft_cpu_kernel/irfft_cpu_kernel 自身的防御性拒绝路径(非 fp32)。
// ---------------------------------------------------------------------------

struct RfftOpNameTag {
  static constexpr std::string_view kOpName = "rfft";
};
using RfftOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<RfftOpNameTag>;

TEST_F(RfftOpKernelTest, RejectsUnsupportedDtypeInt32) {
  Tensor x = MakeTensor<std::int32_t>(Shape({4}));
  Tensor out = MakeTensor<std::int32_t>(Shape({3, 2}));

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("requires x/out to be float32"), std::string_view::npos);
}

struct IrfftOpNameTag {
  static constexpr std::string_view kOpName = "irfft";
};
using IrfftOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<IrfftOpNameTag>;

TEST_F(IrfftOpKernelTest, RejectsUnsupportedDtypeInt32) {
  Tensor z = MakeTensor<std::int32_t>(Shape({3, 2}));
  Tensor out = MakeTensor<std::int32_t>(Shape({4}));

  std::vector<Tensor> inputs{z};
  std::vector<Tensor> outputs{out};
  const AttrMap attrs{{"n", int64_t{4}}};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.attrs = &attrs;
  ctx.device = cpu_device();

  const Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("requires z/out to be float32"), std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 5. 图编译路径。
// ---------------------------------------------------------------------------

class RfftOpCompileTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(RfftOpCompileTest, CompiledExecutionMatchesKnownValues) {
  Graph graph("rfft_only");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {4})).value();
  Node* rfft_node = create_node_with_inferred_types(graph, "rfft", {x}).value();
  ASSERT_TRUE(graph.mark_output(rfft_node, 0).is_ok());

  const Result<std::shared_ptr<frame::hal::Executable>> executable =
      frame::runtime::compile(graph, frame::kCpuBackendName, frame::hal::CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  Tensor x_tensor = MakeTensorWithShape<float>(Shape({4}), {1.0F, 0.0F, -1.0F, 0.0F});
  std::vector<Tensor> inputs{x_tensor};
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  Tensor expected = MakeTensorWithShape<float>(Shape({3, 2}), {0.0F, 0.0F, 2.0F, 0.0F, 0.0F, 0.0F});
  EXPECT_TRUE(
      tensor_all_close(outputs.value()[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

class IrfftOpCompileTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(IrfftOpCompileTest, CompiledExecutionMatchesKnownValues) {
  Graph graph("irfft_only");
  Value* z = graph.add_graph_input(MakeType(DType::of<float>(), {3, 2})).value();
  const AttrMap attrs{{"n", int64_t{4}}};
  Node* irfft_node = create_node_with_inferred_types(graph, "irfft", {z}, attrs).value();
  ASSERT_TRUE(graph.mark_output(irfft_node, 0).is_ok());

  const Result<std::shared_ptr<frame::hal::Executable>> executable =
      frame::runtime::compile(graph, frame::kCpuBackendName, frame::hal::CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  // z 是 rfft([1,0,-1,0]) 的谱(见上方 SingleFrequencyEvenN),irfft 应还原 x。
  Tensor z_tensor = MakeTensorWithShape<float>(Shape({3, 2}), {0.0F, 0.0F, 2.0F, 0.0F, 0.0F, 0.0F});
  std::vector<Tensor> inputs{z_tensor};
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  Tensor expected = MakeTensorWithShape<float>(Shape({4}), {1.0F, 0.0F, -1.0F, 0.0F});
  EXPECT_TRUE(
      tensor_all_close(outputs.value()[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 6. roundtrip:irfft(rfft(x), n)≡x(经图编译路径,偶/奇 n 各一,另加一个批量
// leading 维用例验证 kernel 的 batch 遍历,即 src/backends/cpu/kernels/
// fft.cpp 的"batch=前导维乘积"设计,决议点F)。
// ---------------------------------------------------------------------------

class FftRoundtripTest : public frame::ops::testing::ElementwiseEagerTestBase {};

Graph BuildRoundtripGraph(const Shape& x_shape, int64_t n) {
  Graph graph("fft_roundtrip");
  frame::ir::TensorType x_type;
  x_type.dtype = DType::of<float>();
  x_type.shape = x_shape;
  x_type.layout = frame::ir::Layout::kRowMajor;
  x_type.device = frame::cpu_device();
  Value* x = graph.add_graph_input(x_type).value();
  Node* rfft_node = create_node_with_inferred_types(graph, "rfft", {x}).value();
  const AttrMap irfft_attrs{{"n", n}};
  Node* irfft_node =
      create_node_with_inferred_types(graph, "irfft", {rfft_node->output(0)}, irfft_attrs).value();
  const Status mark_status = graph.mark_output(irfft_node, 0);
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

TEST_F(FftRoundtripTest, EvenNRank1) {
  const int64_t n = 6;
  const Graph graph = BuildRoundtripGraph(Shape({n}), n);
  const Result<std::shared_ptr<frame::hal::Executable>> executable =
      frame::runtime::compile(graph, frame::kCpuBackendName, frame::hal::CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  Tensor x = MakeTensorWithShape<float>(Shape({n}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  std::vector<Tensor> inputs{x};
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  EXPECT_TRUE(tensor_all_close(outputs.value()[0], x, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(FftRoundtripTest, OddNRank1) {
  const int64_t n = 5;
  const Graph graph = BuildRoundtripGraph(Shape({n}), n);
  const Result<std::shared_ptr<frame::hal::Executable>> executable =
      frame::runtime::compile(graph, frame::kCpuBackendName, frame::hal::CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  Tensor x = MakeTensorWithShape<float>(Shape({n}), {1.0F, -2.0F, 3.0F, 0.5F, 4.0F});
  std::vector<Tensor> inputs{x};
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  EXPECT_TRUE(tensor_all_close(outputs.value()[0], x, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(FftRoundtripTest, BatchedLeadingDim) {
  // leading 维=2 的批量输入,验证 kernel 不手写 batch 循环、经 pocketfft
  // 完整 shape+axis 自行遍历其余维度(决议点F)的正确性。
  const int64_t n = 6;
  const Graph graph = BuildRoundtripGraph(Shape({2, n}), n);
  const Result<std::shared_ptr<frame::hal::Executable>> executable =
      frame::runtime::compile(graph, frame::kCpuBackendName, frame::hal::CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  Tensor x = MakeTensorWithShape<float>(
      Shape({2, n}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, -1.0F, 0.5F, 2.5F, -3.0F, 4.5F, 0.0F});
  std::vector<Tensor> inputs{x};
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  EXPECT_TRUE(tensor_all_close(outputs.value()[0], x, default_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 7. 解析梯度 ≡ 数值微分(BUILD-011 容差;rfft/irfft 均为线性变换,loss=
// sum(y) 亦线性,数值微分截断误差理论上为零,仍按惯例走 relaxed_tolerance +
// CheckGradientMatchesNumeric 统一路径,不手写 EXPECT_NEAR)。
// ---------------------------------------------------------------------------

class RfftGradientTest : public frame::ops::testing::ElementwiseEagerTestBase {};
class IrfftGradientTest : public frame::ops::testing::ElementwiseEagerTestBase {};

Graph BuildRfftLossGraph(int64_t n) {
  Graph graph("rfft_loss");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {n})).value();
  Node* rfft_node = create_node_with_inferred_types(graph, "rfft", {x}).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {rfft_node->output(0)}, sum_attrs).value();
  const Status mark_status = graph.mark_output(sum_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

Graph BuildIrfftLossGraph(int64_t n) {
  Graph graph("irfft_loss");
  const int64_t k = n / 2 + 1;
  Value* z = graph.add_graph_input(MakeType(DType::of<float>(), {k, 2})).value();
  const AttrMap irfft_attrs{{"n", n}};
  Node* irfft_node = create_node_with_inferred_types(graph, "irfft", {z}, irfft_attrs).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {irfft_node->output(0)}, sum_attrs).value();
  const Status mark_status = graph.mark_output(sum_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

constexpr double kFftCentralDifferenceH = 1e-2;

TEST_F(RfftGradientTest, GradientMatchesNumericEvenN) {
  const Graph forward = BuildRfftLossGraph(4);
  Tensor x = MakeTensorWithShape<float>(Shape({4}), {1.0F, 2.0F, 3.0F, 4.0F});
  std::vector<Tensor> inputs{x};
  const std::vector<int32_t> wrt{0};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kFftCentralDifferenceH));
}

TEST_F(RfftGradientTest, GradientMatchesNumericOddN) {
  const Graph forward = BuildRfftLossGraph(5);
  Tensor x = MakeTensorWithShape<float>(Shape({5}), {1.0F, -2.0F, 3.0F, 0.5F, 4.0F});
  std::vector<Tensor> inputs{x};
  const std::vector<int32_t> wrt{0};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kFftCentralDifferenceH));
}

TEST_F(IrfftGradientTest, GradientMatchesNumericEvenN) {
  const Graph forward = BuildIrfftLossGraph(4);  // k=3
  Tensor z = MakeTensorWithShape<float>(Shape({3, 2}), {1.0F, 0.5F, -0.5F, 2.0F, 0.25F, -1.0F});
  std::vector<Tensor> inputs{z};
  const std::vector<int32_t> wrt{0};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kFftCentralDifferenceH));
}

TEST_F(IrfftGradientTest, GradientMatchesNumericOddN) {
  const Graph forward = BuildIrfftLossGraph(5);  // k=3
  Tensor z = MakeTensorWithShape<float>(Shape({3, 2}), {0.5F, -1.0F, 2.0F, 0.75F, -0.25F, 1.5F});
  std::vector<Tensor> inputs{z};
  const std::vector<int32_t> wrt{0};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kFftCentralDifferenceH));
}

}  // namespace
