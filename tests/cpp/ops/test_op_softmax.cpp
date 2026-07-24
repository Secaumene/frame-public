// softmax 算子测试(schema + cpu kernel + 解析梯度 ≡ 数值微分,
// src/ops/schemas/sequence.cpp、src/backends/cpu/kernels/sequence.cpp 已实化
// 的行为,M22 批4 T3,§1.2/1.3 决议点B/C)。softmax(x[N,D]):限 rank-2,末轴,
// 无属性;kernel 内减行最大值数值稳定。梯度微图:t=softmax(x) 重算;
// s=sum(t·gy,axes=[1]);gx=t·(gy−bcast_col(s))。
//   1. schema 字段/三函数指针状态;
//   2. shape_infer 合法路径 + rank 非 2 负例;
//   3. eager 数值:2x2 手算已知值(fp32 精确到容差)、bf16(位级转换);
//   4. softmax_cpu_kernel 自身的防御性拒绝路径;
//   5. 图编译路径;
//   6. 解析梯度 ≡ 数值微分(loss=sum(softmax(x)^2),避开
//      sum(softmax(x))恒为常数 N 的退化 loss)。
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
using frame::ops::testing::CheckGradientMatchesNumeric;
using frame::ops::testing::MakeType;
using frame::testing::default_tolerance;
using frame::testing::tensor_all_close;

// ---------------------------------------------------------------------------
// 1. schema 断言。
// ---------------------------------------------------------------------------

TEST(SoftmaxOpSchemaTest, RegisteredWithOneInputOneOutputNoAttrs) {
  const OpSchema* schema = OpRegistry::instance().find("softmax");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 1U);
  EXPECT_EQ(schema->outputs().size(), 1U);
  EXPECT_EQ(schema->attrs().size(), 0U);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

TEST(SoftmaxShapeInferTest, Rank2ProducesIdenticalOutputShape) {
  const OpSchema* schema = OpRegistry::instance().find("softmax");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "softmax";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({2, 3}));
}

TEST(SoftmaxShapeInferTest, Rank3IsRejectedWithActualRank) {
  const OpSchema* schema = OpRegistry::instance().find("softmax");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "softmax";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3, 4})};
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires x to be rank-2 [N, D], got rank 3"),
            std::string_view::npos);
}

TEST(SoftmaxShapeInferTest, WrongInputCountIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("softmax");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "softmax";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {2, 3})};
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("expects 1 input, got 2"), std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 3. eager 数值。
// ---------------------------------------------------------------------------

class SoftmaxOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(SoftmaxOpEagerTest, TwoByTwoKnownValues) {
  // x=[[1,2],[3,4]]:每行差值均为 1,softmax([-1,0])=[e^-1/(e^-1+1),
  // 1/(e^-1+1)]≈[0.26894142,0.73105858],两行结果相同(手算)。
  Tensor x = MakeTensorWithShape<float>(Shape({2, 2}), {1.0F, 2.0F, 3.0F, 4.0F});
  Tensor expected = MakeTensorWithShape<float>(
      Shape({2, 2}), {0.26894142F, 0.73105858F, 0.26894142F, 0.73105858F});
  Tensor out = MakeTensorWithShape<float>(Shape({2, 2}), {0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "softmax";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(SoftmaxOpEagerTest, BFloat16KnownValuesViaBitLevelConversion) {
  Tensor x = MakeTensorWithShape<bfloat16_t>(
      Shape({2, 2}), {float_to_bfloat16(1.0F), float_to_bfloat16(2.0F), float_to_bfloat16(3.0F),
                      float_to_bfloat16(4.0F)});
  Tensor expected = MakeTensorWithShape<bfloat16_t>(
      Shape({2, 2}), {float_to_bfloat16(0.26894142F), float_to_bfloat16(0.73105858F),
                      float_to_bfloat16(0.26894142F), float_to_bfloat16(0.73105858F)});
  Tensor out = MakeTensorWithShape<bfloat16_t>(
      Shape({2, 2}), {bfloat16_t{0}, bfloat16_t{0}, bfloat16_t{0}, bfloat16_t{0}});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "softmax";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kBFloat16)));
}

// ---------------------------------------------------------------------------
// 4. softmax_cpu_kernel 自身的防御性拒绝路径。
// ---------------------------------------------------------------------------

struct SoftmaxOpNameTag {
  static constexpr std::string_view kOpName = "softmax";
};
using SoftmaxOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<SoftmaxOpNameTag>;

TEST_F(SoftmaxOpKernelTest, RejectsUnsupportedDtypeInt32) {
  Tensor x = MakeTensor<std::int32_t>(Shape({2, 2}));
  Tensor out = MakeTensor<std::int32_t>(Shape({2, 2}));

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

TEST_F(SoftmaxOpKernelTest, RejectsRankNotTwo) {
  Tensor x = MakeTensor<float>(Shape({2, 2, 2}));
  Tensor out = MakeTensor<float>(Shape({2, 2, 2}));

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("requires x to be rank-2 [N, D]"), std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 5. 图编译路径。
// ---------------------------------------------------------------------------

class SoftmaxOpCompileTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(SoftmaxOpCompileTest, CompiledExecutionMatchesKnownValues) {
  Graph graph("softmax_only");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {2, 2})).value();
  Node* softmax_node = create_node_with_inferred_types(graph, "softmax", {x}).value();
  ASSERT_TRUE(graph.mark_output(softmax_node, 0).is_ok());

  const Result<std::shared_ptr<frame::hal::Executable>> executable =
      frame::runtime::compile(graph, frame::kCpuBackendName, frame::hal::CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  Tensor x_tensor = MakeTensorWithShape<float>(Shape({2, 2}), {1.0F, 2.0F, 3.0F, 4.0F});
  std::vector<Tensor> inputs{x_tensor};
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  Tensor expected = MakeTensorWithShape<float>(
      Shape({2, 2}), {0.26894142F, 0.73105858F, 0.26894142F, 0.73105858F});
  EXPECT_TRUE(
      tensor_all_close(outputs.value()[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 6. 解析梯度 ≡ 数值微分。
// ---------------------------------------------------------------------------

class SoftmaxGradientTest : public frame::ops::testing::ElementwiseEagerTestBase {};

// loss=sum(softmax(x)^2)(非退化:sum(softmax(x))对任意 x 恒为常数 N,梯度
// 恒零,无法检验 softmax_gradient 本身;平方后 loss 随 x 变化,梯度非平凡)。
Graph BuildSoftmaxSquaredLossGraph() {
  Graph graph("softmax_squared_loss");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  Node* softmax_node = create_node_with_inferred_types(graph, "softmax", {x}).value();
  Node* square_node =
      create_node_with_inferred_types(graph, "square", {softmax_node->output(0)}).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {square_node->output(0)}, sum_attrs).value();
  const Status mark_status = graph.mark_output(sum_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

constexpr double kSoftmaxCentralDifferenceH = 1e-2;

TEST_F(SoftmaxGradientTest, GradientMatchesNumericForX) {
  const Graph forward = BuildSoftmaxSquaredLossGraph();
  Tensor x = MakeTensorWithShape<float>(Shape({2, 3}), {0.1F, 1.5F, -0.5F, -1.0F, 0.3F, 2.0F});
  std::vector<Tensor> inputs{x};
  const std::vector<int32_t> wrt{0};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kSoftmaxCentralDifferenceH));
}

}  // namespace
