// slice 算子测试(schema + cpu kernel + 解析梯度 ≡ 数值微分,
// src/ops/schemas/shape.cpp、src/backends/cpu/kernels/shape.cpp 已实化的
// 行为,M22 批4 T3,§1.4 决议点D)。slice(x; axis, start, stop):单轴连续
// 切片,0<=start<stop<=dim;梯度 gx=concat(前置零 splat, gy, 后置零 splat;
// axis),零段宽 0 时省略、满切片时退化为 concat 单输入(min_count=1 由此
// 必要)。
//   1. schema 字段(三 attr)/三函数指针状态;
//   2. shape_infer 合法路径 + 负例(axis 越界/start 越界/stop 非法/输入数错);
//   3. eager 数值:axis=1 部分切片已知值(fp32)、bf16(位级转换);
//   4. slice_cpu_kernel 自身的防御性拒绝路径;
//   5. 图编译路径;
//   6. 解析梯度 ≡ 数值微分:部分切片(concat 前置+gy+后置三输入路径)与满
//      切片(concat 单输入退化路径,min_count=1 存在的直接原因)各一例。
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

TEST(SliceOpSchemaTest, RegisteredWithOneInputOneOutputThreeAttrs) {
  const OpSchema* schema = OpRegistry::instance().find("slice");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 1U);
  EXPECT_EQ(schema->outputs().size(), 1U);
  ASSERT_EQ(schema->attrs().size(), 3U);
  EXPECT_EQ(schema->attrs()[0].name, "axis");
  EXPECT_EQ(schema->attrs()[1].name, "start");
  EXPECT_EQ(schema->attrs()[2].name, "stop");
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

const AttrMap& StandardSliceAttrs() {
  static const AttrMap attrs{{"axis", int64_t{1}}, {"start", int64_t{1}}, {"stop", int64_t{3}}};
  return attrs;
}

TEST(SliceShapeInferTest, ValidRangeProducesSlicedShape) {
  const OpSchema* schema = OpRegistry::instance().find("slice");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "slice";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 4})};
  ctx.attrs = &StandardSliceAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({2, 2}));
}

TEST(SliceShapeInferTest, WrongInputCountIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("slice");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "slice";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 4}), MakeType(DType::of<float>(), {2, 4})};
  ctx.attrs = &StandardSliceAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("expects 1 input, got 2"), std::string_view::npos);
}

TEST(SliceShapeInferTest, AxisOutOfRangeIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("slice");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "slice";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 4})};
  const AttrMap attrs{{"axis", int64_t{2}}, {"start", int64_t{0}}, {"stop", int64_t{1}}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("attribute 'axis' 2 is out of range for rank 2"),
            std::string_view::npos);
}

TEST(SliceShapeInferTest, StartOutOfRangeIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("slice");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "slice";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 4})};
  const AttrMap attrs{{"axis", int64_t{1}}, {"start", int64_t{-1}}, {"stop", int64_t{2}}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("attribute 'start' -1 is out of range for dimension 4"),
            std::string_view::npos);
}

TEST(SliceShapeInferTest, StopNotGreaterThanStartIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("slice");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "slice";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 4})};
  const AttrMap attrs{{"axis", int64_t{1}}, {"start", int64_t{2}}, {"stop", int64_t{2}}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("attribute 'stop' 2 is invalid for start=2"),
            std::string_view::npos);
}

TEST(SliceShapeInferTest, StopExceedsDimensionIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("slice");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "slice";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 4})};
  const AttrMap attrs{{"axis", int64_t{1}}, {"start", int64_t{0}}, {"stop", int64_t{5}}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("attribute 'stop' 5 is invalid for start=0"),
            std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 3. eager 数值。
// ---------------------------------------------------------------------------

class SliceOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(SliceOpEagerTest, AxisOneKnownValues) {
  // 手算:x=[[1,2,3,4],[5,6,7,8]](2x4),axis=1,[1,3) -> [[2,3],[6,7]](2x2)。
  Tensor x =
      MakeTensorWithShape<float>(Shape({2, 4}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
  Tensor expected = MakeTensorWithShape<float>(Shape({2, 2}), {2.0F, 3.0F, 6.0F, 7.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({2, 2}), {0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "slice";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &StandardSliceAttrs();
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(SliceOpEagerTest, BFloat16FullSliceKnownValuesViaBitLevelConversion) {
  // start=0,stop=dim(满切片)=恒等拷贝。
  Tensor x = MakeTensorWithShape<bfloat16_t>(
      Shape({1, 3}), {float_to_bfloat16(1.0F), float_to_bfloat16(2.0F), float_to_bfloat16(3.0F)});
  Tensor expected = MakeTensorWithShape<bfloat16_t>(
      Shape({1, 3}), {float_to_bfloat16(1.0F), float_to_bfloat16(2.0F), float_to_bfloat16(3.0F)});
  Tensor out =
      MakeTensorWithShape<bfloat16_t>(Shape({1, 3}), {bfloat16_t{0}, bfloat16_t{0}, bfloat16_t{0}});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "slice";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  const AttrMap attrs{{"axis", int64_t{1}}, {"start", int64_t{0}}, {"stop", int64_t{3}}};
  invocation.attrs = &attrs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kBFloat16)));
}

// ---------------------------------------------------------------------------
// 4. slice_cpu_kernel 自身的防御性拒绝路径。
// ---------------------------------------------------------------------------

struct SliceOpNameTag {
  static constexpr std::string_view kOpName = "slice";
};
using SliceOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<SliceOpNameTag>;

TEST_F(SliceOpKernelTest, RejectsUnsupportedDtypeInt32) {
  Tensor x = MakeTensor<std::int32_t>(Shape({2, 4}));
  Tensor out = MakeTensor<std::int32_t>(Shape({2, 2}));

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.attrs = &StandardSliceAttrs();
  ctx.device = cpu_device();

  const Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("does not support dtype 'int32'"), std::string_view::npos);
}

TEST_F(SliceOpKernelTest, RejectsOutShapeMismatch) {
  Tensor x = MakeTensor<float>(Shape({2, 4}));
  Tensor out = MakeTensor<float>(Shape({2, 4}));  // 期望 [2,2],故意给错

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.attrs = &StandardSliceAttrs();
  ctx.device = cpu_device();

  const Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("requires out shape to match the sliced result"),
            std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 5. 图编译路径。
// ---------------------------------------------------------------------------

class SliceOpCompileTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(SliceOpCompileTest, CompiledExecutionMatchesKnownValues) {
  Graph graph("slice_only");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {2, 4})).value();
  Node* slice_node =
      create_node_with_inferred_types(graph, "slice", {x}, StandardSliceAttrs()).value();
  ASSERT_TRUE(graph.mark_output(slice_node, 0).is_ok());

  const Result<std::shared_ptr<frame::hal::Executable>> executable =
      frame::runtime::compile(graph, frame::kCpuBackendName, frame::hal::CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  Tensor x_tensor =
      MakeTensorWithShape<float>(Shape({2, 4}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
  std::vector<Tensor> inputs{x_tensor};
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  Tensor expected = MakeTensorWithShape<float>(Shape({2, 2}), {2.0F, 3.0F, 6.0F, 7.0F});
  EXPECT_TRUE(
      tensor_all_close(outputs.value()[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 6. 解析梯度 ≡ 数值微分:部分切片(concat 前置+gy+后置三输入路径)+ 满切片
//    (concat 单输入退化路径)各一例。
// ---------------------------------------------------------------------------

class SliceGradientTest : public frame::ops::testing::ElementwiseEagerTestBase {};

Graph BuildSliceSquaredLossGraph(const AttrMap& slice_attrs) {
  Graph graph("slice_squared_loss");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {2, 4})).value();
  Node* slice_node = create_node_with_inferred_types(graph, "slice", {x}, slice_attrs).value();
  Node* square_node =
      create_node_with_inferred_types(graph, "square", {slice_node->output(0)}).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {square_node->output(0)}, sum_attrs).value();
  const Status mark_status = graph.mark_output(sum_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

constexpr double kSliceCentralDifferenceH = 1e-2;

TEST_F(SliceGradientTest, PartialSliceGradientMatchesNumericForX) {
  // start=1,stop=3(dim=4):前置宽1 + gy宽2 + 后置宽1,concat 三输入路径。
  const Graph forward = BuildSliceSquaredLossGraph(StandardSliceAttrs());
  Tensor x = MakeTensorWithShape<float>(Shape({2, 4}),
                                        {1.0F, -2.0F, 3.0F, -4.0F, 0.5F, -1.5F, 2.5F, -3.5F});
  std::vector<Tensor> inputs{x};
  const std::vector<int32_t> wrt{0};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kSliceCentralDifferenceH));
}

TEST_F(SliceGradientTest, FullSliceGradientMatchesNumericForXViaConcatSingleInputPath) {
  // start=0,stop=dim=4(满切片):concat 退化为单输入路径(min_count=1 存在的
  // 直接原因,§1.4 决议点D)。
  const AttrMap full_slice_attrs{{"axis", int64_t{1}}, {"start", int64_t{0}}, {"stop", int64_t{4}}};
  const Graph forward = BuildSliceSquaredLossGraph(full_slice_attrs);
  Tensor x = MakeTensorWithShape<float>(Shape({2, 4}),
                                        {1.0F, -2.0F, 3.0F, -4.0F, 0.5F, -1.5F, 2.5F, -3.5F});
  std::vector<Tensor> inputs{x};
  const std::vector<int32_t> wrt{0};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kSliceCentralDifferenceH));
}

}  // namespace
