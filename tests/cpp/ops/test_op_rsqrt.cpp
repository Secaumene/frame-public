// rsqrt 算子测试(schema + cpu kernel + 解析梯度 ≡ 数值微分,
// src/ops/schemas/elementwise.cpp、src/backends/cpu/kernels/elementwise.cpp
// 已实化的行为,M22 批4 T3,spec 外增项——layer_norm 梯度需要 1/√(σ²+ε),
// 设计门批注见 docs/plan/2026-07-19-batch4-m22-seq.md §1.2)。
// rsqrt(x)=x^(-1/2) 一元逐元素算子(x>0);梯度 gx=gy*(-0.5)*r*r*r
// (r=rsqrt(x)),经 constant(-0.5) splat + mul 组合。测试结构照抄
// test_op_sigmoid.cpp/test_op_tanh.cpp。
//   1. schema 字段/traits/三函数指针状态;
//   2. shape_infer 合法路径 + 输入数错负例;
//   3. eager 数值:fp32 与 1/std::sqrt 参考值比对、bf16(位级转换);
//   4. rsqrt_cpu_kernel 自身的防御性拒绝路径;
//   5. 图编译路径;
//   6. 解析梯度 ≡ 数值微分(输入取正值区间,避开 x=0 的不可导点)。
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

using frame::bfloat16_t;
using frame::cpu_device;
using frame::DType;
using frame::DTypeCode;
using frame::ErrorCode;
using frame::float_to_bfloat16;
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

float RsqrtReference(float x) { return 1.0F / std::sqrt(x); }

// ---------------------------------------------------------------------------
// 1. schema 断言。
// ---------------------------------------------------------------------------

TEST(RsqrtOpSchemaTest, RegisteredWithOneInputOneOutput) {
  const OpSchema* schema = OpRegistry::instance().find("rsqrt");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 1U);
  EXPECT_EQ(schema->outputs().size(), 1U);
}

TEST(RsqrtOpSchemaTest, TraitsMatchElementwiseOnly) {
  const OpSchema* schema = OpRegistry::instance().find("rsqrt");
  ASSERT_NE(schema, nullptr);
  EXPECT_TRUE(schema->has_trait(OpTrait::kElementwise));
  EXPECT_FALSE(schema->has_trait(OpTrait::kFusable));
  EXPECT_FALSE(schema->has_trait(OpTrait::kCommutative));
  EXPECT_FALSE(schema->has_trait(OpTrait::kHasSideEffect));
}

TEST(RsqrtOpSchemaTest, HasShapeInferButNoDecompositionAndHasGradient) {
  const OpSchema* schema = OpRegistry::instance().find("rsqrt");
  ASSERT_NE(schema, nullptr);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

TEST(RsqrtShapeInferTest, OutputShapeIsIdenticalToInputShape) {
  const OpSchema* schema = OpRegistry::instance().find("rsqrt");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "rsqrt";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({2, 3}));
}

TEST(RsqrtShapeInferTest, WrongInputCountIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("rsqrt");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "rsqrt";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {2, 3})};
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("expects 1 input, got 2"), std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 3. eager 数值。
// ---------------------------------------------------------------------------

class RsqrtOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(RsqrtOpEagerTest, Float32MatchesReciprocalSqrtReference) {
  const std::vector<float> x_values{0.25F, 1.0F, 4.0F, 16.0F};
  std::vector<float> expected_values;
  expected_values.reserve(x_values.size());
  for (float v : x_values) expected_values.push_back(RsqrtReference(v));
  Tensor x = MakeTensor1D<float>(x_values);
  Tensor expected = MakeTensor1D<float>(expected_values);
  Tensor out = MakeTensor1D<float>(std::vector<float>(x_values.size(), 0.0F));

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "rsqrt";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(RsqrtOpEagerTest, BFloat16MatchesReferenceViaBitLevelConversion) {
  const std::vector<double> x_values{0.25, 1.0, 4.0, 16.0};
  std::vector<bfloat16_t> x_bf16;
  std::vector<bfloat16_t> expected_bf16;
  for (double v : x_values) {
    x_bf16.push_back(float_to_bfloat16(static_cast<float>(v)));
    expected_bf16.push_back(float_to_bfloat16(RsqrtReference(static_cast<float>(v))));
  }
  Tensor x = MakeTensor1D<bfloat16_t>(x_bf16);
  Tensor expected = MakeTensor1D<bfloat16_t>(expected_bf16);
  Tensor out = MakeTensor1D<bfloat16_t>(std::vector<bfloat16_t>(x_bf16.size(), bfloat16_t{0}));

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "rsqrt";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kBFloat16)));
}

// ---------------------------------------------------------------------------
// 4. rsqrt_cpu_kernel 自身的防御性拒绝路径。
// ---------------------------------------------------------------------------

struct RsqrtOpNameTag {
  static constexpr std::string_view kOpName = "rsqrt";
};
using RsqrtOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<RsqrtOpNameTag>;

TEST_F(RsqrtOpKernelTest, RejectsUnsupportedDtypeInt32) {
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

TEST_F(RsqrtOpKernelTest, RejectsShapeMismatchBetweenXAndOut) {
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
// 5. 图编译路径。
// ---------------------------------------------------------------------------

class RsqrtOpCompileTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(RsqrtOpCompileTest, CompiledExecutionMatchesReciprocalSqrtReference) {
  Graph graph("rsqrt_only");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {4})).value();
  Node* rsqrt_node = create_node_with_inferred_types(graph, "rsqrt", {x}).value();
  ASSERT_TRUE(graph.mark_output(rsqrt_node, 0).is_ok());

  const Result<std::shared_ptr<frame::hal::Executable>> executable =
      frame::runtime::compile(graph, frame::kCpuBackendName, frame::hal::CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  Tensor x_tensor = MakeTensor1D<float>({0.25F, 1.0F, 4.0F, 16.0F});
  std::vector<Tensor> inputs{x_tensor};
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  Tensor expected = MakeTensor1D<float>(
      {RsqrtReference(0.25F), RsqrtReference(1.0F), RsqrtReference(4.0F), RsqrtReference(16.0F)});
  EXPECT_TRUE(
      tensor_all_close(outputs.value()[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 6. 解析梯度 ≡ 数值微分(x 取正值区间,rsqrt 定义域要求)。
// ---------------------------------------------------------------------------

class RsqrtGradientTest : public frame::ops::testing::ElementwiseEagerTestBase {};

Graph BuildRsqrtLossGraph() {
  Graph graph("rsqrt_loss");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {4})).value();
  Node* rsqrt_node = create_node_with_inferred_types(graph, "rsqrt", {x}).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {rsqrt_node->output(0)}, sum_attrs).value();
  const Status mark_status = graph.mark_output(sum_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

constexpr double kRsqrtCentralDifferenceH = 1e-2;

TEST_F(RsqrtGradientTest, GradientMatchesNumericForX) {
  const Graph forward = BuildRsqrtLossGraph();
  Tensor x = MakeTensor1D<float>({0.5F, 1.0F, 2.0F, 4.0F});
  std::vector<Tensor> inputs{x};
  const std::vector<int32_t> wrt{0};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kRsqrtCentralDifferenceH));
}

}  // namespace
