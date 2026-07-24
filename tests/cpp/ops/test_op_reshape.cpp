// reshape 算子测试(schema + cpu kernel + 解析梯度 ≡ 数值微分,
// src/ops/schemas/shape.cpp、src/backends/cpu/kernels/shape.cpp 已实化的
// 行为,M21 批3 T4)。reshape 是唯一属性 target_shape(kShape)、numel 守恒的
// 形状变换算子;cpu kernel 为逐字节拷贝(不经 dispatch_dtype,对任意 dtype
// 一视同仁,详见 kernel 头注释)。
//   1. OpRegistry::find("reshape") 的 schema 字段、shape_infer()/
//      decomposition()/gradient() 三函数指针状态;
//   2. infer_reshape_shape 的合法路径 + numel 不等负例(消息含双方 numel)+
//      输入数错负例;
//   3. eager 数值路径(ARCH-011 第 3 类):fp32 数值透传(行优先展平顺序不变,
//      [2,3]→[3,2])、dtype 无关性(int32 同样透传,不受 v0 浮点三档限制);
//   4. reshape_cpu_kernel 自身的防御性拒绝路径(dtype 不一致 / numel 不等);
//   5. 解析梯度 ≡ 数值微分(reshape 回原 shape)。
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

TEST(ReshapeOpSchemaTest, RegisteredWithOneInputOneAttrOneOutput) {
  const OpSchema* schema = OpRegistry::instance().find("reshape");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 1U);
  EXPECT_EQ(schema->outputs().size(), 1U);
  ASSERT_EQ(schema->attrs().size(), 1U);
  EXPECT_EQ(schema->attrs()[0].name, "target_shape");
  EXPECT_EQ(schema->attrs()[0].type, frame::ir::AttrType::kShape);
  EXPECT_TRUE(schema->attrs()[0].required);
}

TEST(ReshapeOpSchemaTest, HasShapeInferButNoDecompositionAndHasGradient) {
  const OpSchema* schema = OpRegistry::instance().find("reshape");
  ASSERT_NE(schema, nullptr);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

TEST(ReshapeShapeInferTest, TargetShapeWithSameNumelIsAccepted) {
  const OpSchema* schema = OpRegistry::instance().find("reshape");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "reshape";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};
  const AttrMap attrs{{"target_shape", Shape({3, 2})}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({3, 2}));
}

TEST(ReshapeShapeInferTest, WrongInputCountIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("reshape");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "reshape";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {2, 3})};
  const AttrMap attrs{{"target_shape", Shape({3, 2})}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("expects 1 input, got 2"), std::string_view::npos);
}

TEST(ReshapeShapeInferTest, NumelMismatchIsRejectedWithBothNumelInMessage) {
  const OpSchema* schema = OpRegistry::instance().find("reshape");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "reshape";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};  // numel=6
  const AttrMap attrs{{"target_shape", Shape({4, 2})}};      // numel=8
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  const std::string_view message = result.status().message();
  EXPECT_NE(message.find("numel=6"), std::string_view::npos);
  EXPECT_NE(message.find("numel=8"), std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 3. eager 数值(ARCH-011 第 3 类)。
// ---------------------------------------------------------------------------

class ReshapeOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(ReshapeOpEagerTest, Float32TransparentlyPassesThroughInFlattenedOrder) {
  // [2,3]→[3,2]:reshape 是逐字节拷贝,行优先展平顺序不变。
  Tensor x = MakeTensorWithShape<float>(Shape({2, 3}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  Tensor expected = MakeTensorWithShape<float>(Shape({3, 2}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({3, 2}), {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "reshape";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(ReshapeOpEagerTest, Int32TransparentlyPassesThroughDtypeAgnosticKernel) {
  // reshape kernel 不经 dispatch_dtype、不限 v0 浮点三档(逐字节拷贝天然与
  // dtype 无关,见 src/backends/cpu/kernels/shape.cpp 头注释);int32 同样应
  // 透传正确。
  Tensor x = MakeTensorWithShape<std::int32_t>(Shape({4}), {10, 20, 30, 40});
  Tensor expected = MakeTensorWithShape<std::int32_t>(Shape({2, 2}), {10, 20, 30, 40});
  Tensor out = MakeTensorWithShape<std::int32_t>(Shape({2, 2}), {0, 0, 0, 0});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "reshape";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kInt32)));
}

// ---------------------------------------------------------------------------
// 4. reshape_cpu_kernel 自身的防御性拒绝路径。
// ---------------------------------------------------------------------------

struct ReshapeOpNameTag {
  static constexpr std::string_view kOpName = "reshape";
};
using ReshapeOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<ReshapeOpNameTag>;

TEST_F(ReshapeOpKernelTest, RejectsDtypeMismatchBetweenXAndOut) {
  Tensor x = MakeTensor<float>(Shape({4}));
  Tensor out = MakeTensor<std::int32_t>(Shape({4}));

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("requires x/out of the same dtype"), std::string_view::npos);
}

TEST_F(ReshapeOpKernelTest, RejectsNumelMismatch) {
  Tensor x = MakeTensor<float>(Shape({4}));
  Tensor out = MakeTensor<float>(Shape({5}));  // numel 4 vs 5

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("requires x/out numel to match"), std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 5. 解析梯度 ≡ 数值微分(reshape 回原 shape)。
// ---------------------------------------------------------------------------

class ReshapeGradientTest : public frame::ops::testing::ElementwiseEagerTestBase {};

Graph BuildReshapeLossGraph() {
  Graph graph("reshape_loss");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  const AttrMap reshape_attrs{{"target_shape", Shape({3, 2})}};
  Node* reshape_node =
      create_node_with_inferred_types(graph, "reshape", {x}, reshape_attrs).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {reshape_node->output(0)}, sum_attrs).value();
  const Status mark_status = graph.mark_output(sum_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

// 与 test_conv2d.cpp 同一实测区间(fp32,BUILD-011 建议 1e-3 量级起调)。
constexpr double kReshapeCentralDifferenceH = 1e-2;

TEST_F(ReshapeGradientTest, GradientMatchesNumericForX) {
  const Graph forward = BuildReshapeLossGraph();
  Tensor x = MakeTensorWithShape<float>(Shape({2, 3}), {0.5F, -1.2F, 2.3F, -0.7F, 1.1F, 0.2F});
  std::vector<Tensor> inputs{x};
  const std::vector<int32_t> wrt{0};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kReshapeCentralDifferenceH));
}

}  // namespace
