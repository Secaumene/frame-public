// concat 算子测试(schema + cpu kernel + 解析梯度 ≡ 数值微分,
// src/ops/schemas/shape.cpp、src/backends/cpu/kernels/shape.cpp 已实化的
// 行为,M22 批4 T3,§1.4 决议点D)。concat(xs...; axis):variadic_input
// min_count=1(单输入退化=恒等拷贝,为 slice 满切片梯度所需);梯度
// gx_i=slice(gy,axis,start_i,stop_i)(各输入宽度前缀和)。
//   1. schema 字段(variadic 输入组)/三函数指针状态;
//   2. shape_infer 合法路径 + 负例(axis 越界/rank 不一/非 axis 维不符/
//      dtype 不一致/0 输入);
//   3. eager 数值:axis=0 与 axis=1 两组已知值(fp32)、bf16(位级转换)、
//      单输入退化=恒等拷贝;
//   4. concat_cpu_kernel 自身的防御性拒绝路径;
//   5. 图编译路径;
//   6. 解析梯度 ≡ 数值微分(loss=sum(concat(a,b,axis)^2),两输入各自验证)。
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

TEST(ConcatOpSchemaTest, RegisteredWithVariadicInputMinCountOneAndOneAttr) {
  const OpSchema* schema = OpRegistry::instance().find("concat");
  ASSERT_NE(schema, nullptr);
  EXPECT_TRUE(schema->has_variadic_inputs());
  EXPECT_EQ(schema->min_input_count(), 1);
  EXPECT_EQ(schema->outputs().size(), 1U);
  ASSERT_EQ(schema->attrs().size(), 1U);
  EXPECT_EQ(schema->attrs()[0].name, "axis");
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

TEST(ConcatShapeInferTest, TwoInputsAxisZeroProducesSummedDim) {
  const OpSchema* schema = OpRegistry::instance().find("concat");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "concat";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {5, 3})};
  const AttrMap attrs{{"axis", int64_t{0}}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({7, 3}));
}

TEST(ConcatShapeInferTest, ZeroInputsIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("concat");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "concat";
  const AttrMap attrs{{"axis", int64_t{0}}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("expects at least 1 input, got 0"),
            std::string_view::npos);
}

TEST(ConcatShapeInferTest, AxisOutOfRangeIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("concat");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "concat";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};
  const AttrMap attrs{{"axis", int64_t{2}}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("attribute 'axis' 2 is out of range for rank 2"),
            std::string_view::npos);
}

TEST(ConcatShapeInferTest, RankMismatchIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("concat");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "concat";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {2, 3, 1})};
  const AttrMap attrs{{"axis", int64_t{0}}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires all inputs of the same rank"),
            std::string_view::npos);
}

TEST(ConcatShapeInferTest, NonAxisDimensionMismatchIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("concat");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "concat";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {2, 4})};
  const AttrMap attrs{{"axis", int64_t{0}}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires all inputs to match on non-axis dimension"),
            std::string_view::npos);
}

TEST(ConcatShapeInferTest, DtypeMismatchIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("concat");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "concat";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}),
                     MakeType(DType::of<frame::bfloat16_t>(), {2, 3})};
  const AttrMap attrs{{"axis", int64_t{0}}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires all inputs of the same dtype"),
            std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 3. eager 数值。
// ---------------------------------------------------------------------------

class ConcatOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(ConcatOpEagerTest, AxisZeroKnownValues) {
  // 手算:a=[[1,2],[3,4]](2x2),b=[[5,6]](1x2),axis=0 -> [[1,2],[3,4],[5,6]](3x2)。
  Tensor a = MakeTensorWithShape<float>(Shape({2, 2}), {1.0F, 2.0F, 3.0F, 4.0F});
  Tensor b = MakeTensorWithShape<float>(Shape({1, 2}), {5.0F, 6.0F});
  Tensor expected = MakeTensorWithShape<float>(Shape({3, 2}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({3, 2}), {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{a, b};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "concat";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  const AttrMap attrs{{"axis", int64_t{0}}};
  invocation.attrs = &attrs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(ConcatOpEagerTest, AxisOneKnownValues) {
  // 手算:a=[[1,2],[3,4]](2x2),b=[[5],[6]](2x1),axis=1 -> [[1,2,5],[3,4,6]](2x3)。
  Tensor a = MakeTensorWithShape<float>(Shape({2, 2}), {1.0F, 2.0F, 3.0F, 4.0F});
  Tensor b = MakeTensorWithShape<float>(Shape({2, 1}), {5.0F, 6.0F});
  Tensor expected = MakeTensorWithShape<float>(Shape({2, 3}), {1.0F, 2.0F, 5.0F, 3.0F, 4.0F, 6.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({2, 3}), {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{a, b};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "concat";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  const AttrMap attrs{{"axis", int64_t{1}}};
  invocation.attrs = &attrs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(ConcatOpEagerTest, SingleInputDegeneratesToIdentity) {
  // min_count=1 的退化路径:单输入 concat 恒等拷贝(slice 满切片梯度所需的
  // 前提,§1.4 决议点D)。
  Tensor a = MakeTensorWithShape<float>(Shape({2, 2}), {1.0F, 2.0F, 3.0F, 4.0F});
  Tensor expected = MakeTensorWithShape<float>(Shape({2, 2}), {1.0F, 2.0F, 3.0F, 4.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({2, 2}), {0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{a};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "concat";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  const AttrMap attrs{{"axis", int64_t{0}}};
  invocation.attrs = &attrs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(ConcatOpEagerTest, BFloat16AxisZeroKnownValuesViaBitLevelConversion) {
  Tensor a = MakeTensorWithShape<bfloat16_t>(Shape({1, 2}),
                                             {float_to_bfloat16(1.0F), float_to_bfloat16(2.0F)});
  Tensor b = MakeTensorWithShape<bfloat16_t>(Shape({1, 2}),
                                             {float_to_bfloat16(3.0F), float_to_bfloat16(4.0F)});
  Tensor expected = MakeTensorWithShape<bfloat16_t>(
      Shape({2, 2}), {float_to_bfloat16(1.0F), float_to_bfloat16(2.0F), float_to_bfloat16(3.0F),
                      float_to_bfloat16(4.0F)});
  Tensor out = MakeTensorWithShape<bfloat16_t>(
      Shape({2, 2}), {bfloat16_t{0}, bfloat16_t{0}, bfloat16_t{0}, bfloat16_t{0}});

  std::vector<Tensor> inputs{a, b};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "concat";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  const AttrMap attrs{{"axis", int64_t{0}}};
  invocation.attrs = &attrs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kBFloat16)));
}

// ---------------------------------------------------------------------------
// 4. concat_cpu_kernel 自身的防御性拒绝路径。
// ---------------------------------------------------------------------------

struct ConcatOpNameTag {
  static constexpr std::string_view kOpName = "concat";
};
using ConcatOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<ConcatOpNameTag>;

TEST_F(ConcatOpKernelTest, RejectsUnsupportedDtypeInt32) {
  Tensor a = MakeTensor<std::int32_t>(Shape({2, 2}));
  Tensor out = MakeTensor<std::int32_t>(Shape({2, 2}));

  std::vector<Tensor> inputs{a};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  const AttrMap attrs{{"axis", int64_t{0}}};
  ctx.attrs = &attrs;
  ctx.device = cpu_device();

  const Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("does not support dtype 'int32'"), std::string_view::npos);
}

TEST_F(ConcatOpKernelTest, RejectsOutShapeMismatch) {
  Tensor a = MakeTensor<float>(Shape({2, 2}));
  Tensor b = MakeTensor<float>(Shape({1, 2}));
  Tensor out = MakeTensor<float>(Shape({2, 2}));  // 期望 [3,2],故意给错

  std::vector<Tensor> inputs{a, b};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  const AttrMap attrs{{"axis", int64_t{0}}};
  ctx.attrs = &attrs;
  ctx.device = cpu_device();

  const Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("requires out shape to match the concatenation result"),
            std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 5. 图编译路径。
// ---------------------------------------------------------------------------

class ConcatOpCompileTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(ConcatOpCompileTest, CompiledExecutionMatchesKnownValues) {
  Graph graph("concat_only");
  Value* a = graph.add_graph_input(MakeType(DType::of<float>(), {2, 2})).value();
  Value* b = graph.add_graph_input(MakeType(DType::of<float>(), {1, 2})).value();
  const AttrMap attrs{{"axis", int64_t{0}}};
  Node* concat_node = create_node_with_inferred_types(graph, "concat", {a, b}, attrs).value();
  ASSERT_TRUE(graph.mark_output(concat_node, 0).is_ok());

  const Result<std::shared_ptr<frame::hal::Executable>> executable =
      frame::runtime::compile(graph, frame::kCpuBackendName, frame::hal::CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  Tensor a_tensor = MakeTensorWithShape<float>(Shape({2, 2}), {1.0F, 2.0F, 3.0F, 4.0F});
  Tensor b_tensor = MakeTensorWithShape<float>(Shape({1, 2}), {5.0F, 6.0F});
  std::vector<Tensor> inputs{a_tensor, b_tensor};
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  Tensor expected = MakeTensorWithShape<float>(Shape({3, 2}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  EXPECT_TRUE(
      tensor_all_close(outputs.value()[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 6. 解析梯度 ≡ 数值微分。
// ---------------------------------------------------------------------------

class ConcatGradientTest : public frame::ops::testing::ElementwiseEagerTestBase {};

// loss=sum(concat(a,b,axis=0)^2):经实际构图 + 反向执行验证 concat_gradient
// 的前缀和切片偏移是否正确(理由同 test_op_transpose.cpp 头注释)。
Graph BuildConcatSquaredLossGraph() {
  Graph graph("concat_squared_loss");
  Value* a = graph.add_graph_input(MakeType(DType::of<float>(), {2, 2})).value();
  Value* b = graph.add_graph_input(MakeType(DType::of<float>(), {1, 2})).value();
  const AttrMap attrs{{"axis", int64_t{0}}};
  Node* concat_node = create_node_with_inferred_types(graph, "concat", {a, b}, attrs).value();
  Node* square_node =
      create_node_with_inferred_types(graph, "square", {concat_node->output(0)}).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {square_node->output(0)}, sum_attrs).value();
  const Status mark_status = graph.mark_output(sum_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

constexpr double kConcatCentralDifferenceH = 1e-2;

TEST_F(ConcatGradientTest, GradientMatchesNumericForAAndB) {
  const Graph forward = BuildConcatSquaredLossGraph();
  Tensor a = MakeTensorWithShape<float>(Shape({2, 2}), {1.0F, -2.0F, 3.0F, -4.0F});
  Tensor b = MakeTensorWithShape<float>(Shape({1, 2}), {0.5F, -1.5F});
  std::vector<Tensor> inputs{a, b};
  const std::vector<int32_t> wrt{0, 1};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kConcatCentralDifferenceH));
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 1, inputs, kConcatCentralDifferenceH));
}

}  // namespace
