// tanh 算子测试(schema + cpu kernel + 解析梯度 ≡ 数值微分,
// src/ops/schemas/elementwise.cpp、src/backends/cpu/kernels/elementwise.cpp
// 已实化的行为,M22 批4 T3)。tanh(x) 一元逐元素算子;梯度
// gx=gy*(1-tanh(x)^2),经 constant(±1) splat + mul/add 组合(v0 无 sub 算子,
// 同 sigmoid_gradient 机制)。测试结构照抄 test_op_sigmoid.cpp。
//   1. OpRegistry::find("tanh") 的 schema 字段、traits、三函数指针状态;
//   2. infer_tanh_shape 合法路径 + 输入数错负例;
//   3. eager 数值路径:fp32 与 std::tanh 参考值比对、bf16(位级转换);
//   4. tanh_cpu_kernel 自身的防御性拒绝路径;
//   5. 图编译路径(runtime::compile("cpu"));
//   6. 解析梯度 ≡ 数值微分。
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

// ---------------------------------------------------------------------------
// 1. schema 断言。
// ---------------------------------------------------------------------------

TEST(TanhOpSchemaTest, RegisteredWithOneInputOneOutput) {
  const OpSchema* schema = OpRegistry::instance().find("tanh");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 1U);
  EXPECT_EQ(schema->outputs().size(), 1U);
}

TEST(TanhOpSchemaTest, TraitsMatchElementwiseOnly) {
  const OpSchema* schema = OpRegistry::instance().find("tanh");
  ASSERT_NE(schema, nullptr);
  EXPECT_TRUE(schema->has_trait(OpTrait::kElementwise));
  EXPECT_FALSE(schema->has_trait(OpTrait::kFusable));
  EXPECT_FALSE(schema->has_trait(OpTrait::kCommutative));
  EXPECT_FALSE(schema->has_trait(OpTrait::kHasSideEffect));
}

TEST(TanhOpSchemaTest, HasShapeInferButNoDecompositionAndHasGradient) {
  const OpSchema* schema = OpRegistry::instance().find("tanh");
  ASSERT_NE(schema, nullptr);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

TEST(TanhShapeInferTest, OutputShapeIsIdenticalToInputShape) {
  const OpSchema* schema = OpRegistry::instance().find("tanh");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "tanh";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({2, 3}));
}

TEST(TanhShapeInferTest, WrongInputCountIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("tanh");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "tanh";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {2, 3})};
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("expects 1 input, got 2"), std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 3. eager 数值。
// ---------------------------------------------------------------------------

class TanhOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(TanhOpEagerTest, Float32MatchesStdTanhReference) {
  const std::vector<float> x_values{-3.0F, -1.0F, 0.0F, 1.0F, 3.0F};
  std::vector<float> expected_values;
  expected_values.reserve(x_values.size());
  for (float v : x_values) expected_values.push_back(std::tanh(v));
  Tensor x = MakeTensor1D<float>(x_values);
  Tensor expected = MakeTensor1D<float>(expected_values);
  Tensor out = MakeTensor1D<float>(std::vector<float>(x_values.size(), 0.0F));

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "tanh";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(TanhOpEagerTest, BFloat16MatchesStdTanhReferenceViaBitLevelConversion) {
  const std::vector<double> x_values{-2.0, -0.5, 0.0, 0.5, 2.0};
  std::vector<bfloat16_t> x_bf16;
  std::vector<bfloat16_t> expected_bf16;
  for (double v : x_values) {
    x_bf16.push_back(float_to_bfloat16(static_cast<float>(v)));
    expected_bf16.push_back(float_to_bfloat16(static_cast<float>(std::tanh(v))));
  }
  Tensor x = MakeTensor1D<bfloat16_t>(x_bf16);
  Tensor expected = MakeTensor1D<bfloat16_t>(expected_bf16);
  Tensor out = MakeTensor1D<bfloat16_t>(std::vector<bfloat16_t>(x_bf16.size(), bfloat16_t{0}));

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "tanh";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kBFloat16)));
}

// ---------------------------------------------------------------------------
// 4. tanh_cpu_kernel 自身的防御性拒绝路径。
// ---------------------------------------------------------------------------

struct TanhOpNameTag {
  static constexpr std::string_view kOpName = "tanh";
};
using TanhOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<TanhOpNameTag>;

TEST_F(TanhOpKernelTest, RejectsUnsupportedDtypeInt32) {
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

TEST_F(TanhOpKernelTest, RejectsShapeMismatchBetweenXAndOut) {
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
// 5. 图编译路径(runtime::compile("cpu"))。
// ---------------------------------------------------------------------------

class TanhOpCompileTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(TanhOpCompileTest, CompiledExecutionMatchesStdTanhReference) {
  Graph graph("tanh_only");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {4})).value();
  Node* tanh_node = create_node_with_inferred_types(graph, "tanh", {x}).value();
  ASSERT_TRUE(graph.mark_output(tanh_node, 0).is_ok());

  const Result<std::shared_ptr<frame::hal::Executable>> executable =
      frame::runtime::compile(graph, frame::kCpuBackendName, frame::hal::CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  Tensor x_tensor = MakeTensor1D<float>({-1.0F, -0.25F, 0.25F, 1.0F});
  std::vector<Tensor> inputs{x_tensor};
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  Tensor expected =
      MakeTensor1D<float>({std::tanh(-1.0F), std::tanh(-0.25F), std::tanh(0.25F), std::tanh(1.0F)});
  EXPECT_TRUE(
      tensor_all_close(outputs.value()[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 6. 解析梯度 ≡ 数值微分。
// ---------------------------------------------------------------------------

class TanhGradientTest : public frame::ops::testing::ElementwiseEagerTestBase {};

Graph BuildTanhLossGraph() {
  Graph graph("tanh_loss");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {4})).value();
  Node* tanh_node = create_node_with_inferred_types(graph, "tanh", {x}).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {tanh_node->output(0)}, sum_attrs).value();
  const Status mark_status = graph.mark_output(sum_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

constexpr double kTanhCentralDifferenceH = 1e-2;

TEST_F(TanhGradientTest, GradientMatchesNumericForX) {
  const Graph forward = BuildTanhLossGraph();
  Tensor x = MakeTensor1D<float>({-1.5F, -0.5F, 0.5F, 1.5F});
  std::vector<Tensor> inputs{x};
  const std::vector<int32_t> wrt{0};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kTanhCentralDifferenceH));
}

}  // namespace
