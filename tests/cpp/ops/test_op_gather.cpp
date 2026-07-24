// gather 算子测试(schema + cpu kernel + 解析梯度 ≡ 数值微分,
// src/ops/schemas/gather.cpp、src/backends/cpu/kernels/gather.cpp 已实化的
// 行为,M22 批4 T3,§1.5 决议点E)。gather(x[N,F], indices[K]) -> out[K,F],
// axis 固定 0;x 限 rank-2 浮点三档,indices 限 rank-1 且 dtype∈
// {int32,int64}。梯度:wrt x=gather_grad_internal(gy,indices)+input_shape,
// wrt indices=constant(0) 整数 splat。gather_grad_internal 为
// scatter-add(重复索引累加)。
//   1. schema 字段(gather/gather_grad_internal)/三函数指针状态;
//   2. shape_infer 合法路径 + 负例(rank/indices dtype/input_shape 校验);
//   3. eager 数值:4x2 已知值(fp32,含 int32/int64 indices 各一)、
//      bf16(位级转换);
//   4. gather_cpu_kernel/gather_grad_internal_cpu_kernel 自身的防御性拒绝
//      路径(dtype、越界索引);
//   5. 重复索引累加手算例(直接调用 gather_grad_internal kernel);
//   6. 图编译路径;
//   7. 解析梯度 ≡ 数值微分(对 x,indices 取含重复索引的样本以覆盖累加路径)。
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
#include <frame/ir/serialization.h>
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

TEST(GatherOpSchemaTest, RegisteredWithTwoInputsOneOutputAndGradient) {
  const OpSchema* schema = OpRegistry::instance().find("gather");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 2U);
  EXPECT_EQ(schema->outputs().size(), 1U);
  EXPECT_EQ(schema->attrs().size(), 0U);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);
}

TEST(GatherOpSchemaTest, GatherGradInternalRegisteredWithTwoInputsOneAttrAndGradient) {
  const OpSchema* schema = OpRegistry::instance().find("gather_grad_internal");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 2U);
  EXPECT_EQ(schema->outputs().size(), 1U);
  ASSERT_EQ(schema->attrs().size(), 1U);
  EXPECT_EQ(schema->attrs()[0].name, "input_shape");
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);  // R11:内部算子自身注册 GradientFn
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

TEST(GatherShapeInferTest, ValidInputsProduceKByFShape) {
  const OpSchema* schema = OpRegistry::instance().find("gather");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "gather";
  ctx.input_types = {MakeType(DType::of<float>(), {4, 2}),
                     MakeType(DType::of<std::int64_t>(), {3})};
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({3, 2}));
}

TEST(GatherShapeInferTest, XRankNotTwoIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("gather");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "gather";
  ctx.input_types = {MakeType(DType::of<float>(), {4}), MakeType(DType::of<std::int64_t>(), {3})};
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires x to be rank-2 [N, F], got rank 1"),
            std::string_view::npos);
}

TEST(GatherShapeInferTest, IndicesRankNotOneIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("gather");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "gather";
  ctx.input_types = {MakeType(DType::of<float>(), {4, 2}),
                     MakeType(DType::of<std::int64_t>(), {3, 1})};
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires indices to be rank-1 [K], got rank 2"),
            std::string_view::npos);
}

TEST(GatherShapeInferTest, IndicesDtypeOutsideWhitelistIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("gather");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "gather";
  ctx.input_types = {MakeType(DType::of<float>(), {4, 2}), MakeType(DType::of<float>(), {3})};
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires indices dtype to be int32 or int64"),
            std::string_view::npos);
}

TEST(GatherShapeInferTest, XDtypeOutsideWhitelistIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("gather");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "gather";
  ctx.input_types = {MakeType(DType::of<std::int32_t>(), {4, 2}),
                     MakeType(DType::of<std::int64_t>(), {3})};
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("does not support x dtype 'int32'"),
            std::string_view::npos);
}

TEST(GatherGradInternalShapeInferTest, ValidInputsProduceInputShape) {
  const OpSchema* schema = OpRegistry::instance().find("gather_grad_internal");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "gather_grad_internal";
  ctx.input_types = {MakeType(DType::of<float>(), {3, 2}),
                     MakeType(DType::of<std::int64_t>(), {3})};
  const AttrMap attrs{{"input_shape", Shape({4, 2})}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({4, 2}));
}

TEST(GatherGradInternalShapeInferTest, InputShapeSecondDimensionMismatchIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("gather_grad_internal");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "gather_grad_internal";
  ctx.input_types = {MakeType(DType::of<float>(), {3, 2}),
                     MakeType(DType::of<std::int64_t>(), {3})};
  const AttrMap attrs{{"input_shape", Shape({4, 5})}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("second dimension must equal gy's F=2"),
            std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 3. eager 数值。
// ---------------------------------------------------------------------------

class GatherOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(GatherOpEagerTest, Int64IndicesKnownValues) {
  // x=[[1,2],[3,4],[5,6],[7,8]](4x2),indices=[2,0,2](含重复,K=3)
  // -> out=[[5,6],[1,2],[5,6]]。
  Tensor x =
      MakeTensorWithShape<float>(Shape({4, 2}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
  Tensor indices = MakeTensor1D<std::int64_t>({2, 0, 2});
  Tensor expected = MakeTensorWithShape<float>(Shape({3, 2}), {5.0F, 6.0F, 1.0F, 2.0F, 5.0F, 6.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({3, 2}), {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{x, indices};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "gather";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(GatherOpEagerTest, Int32IndicesKnownValues) {
  Tensor x =
      MakeTensorWithShape<float>(Shape({4, 2}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
  Tensor indices = MakeTensor1D<std::int32_t>({3, 1});
  Tensor expected = MakeTensorWithShape<float>(Shape({2, 2}), {7.0F, 8.0F, 3.0F, 4.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({2, 2}), {0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{x, indices};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "gather";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(GatherOpEagerTest, BFloat16KnownValuesViaBitLevelConversion) {
  Tensor x = MakeTensorWithShape<bfloat16_t>(
      Shape({2, 2}), {float_to_bfloat16(1.0F), float_to_bfloat16(2.0F), float_to_bfloat16(3.0F),
                      float_to_bfloat16(4.0F)});
  Tensor indices = MakeTensor1D<std::int64_t>({1, 0});
  Tensor expected = MakeTensorWithShape<bfloat16_t>(
      Shape({2, 2}), {float_to_bfloat16(3.0F), float_to_bfloat16(4.0F), float_to_bfloat16(1.0F),
                      float_to_bfloat16(2.0F)});
  Tensor out = MakeTensorWithShape<bfloat16_t>(
      Shape({2, 2}), {bfloat16_t{0}, bfloat16_t{0}, bfloat16_t{0}, bfloat16_t{0}});

  std::vector<Tensor> inputs{x, indices};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "gather";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kBFloat16)));
}

// ---------------------------------------------------------------------------
// 4. gather_cpu_kernel/gather_grad_internal_cpu_kernel 自身的防御性拒绝路径
//    (dtype、越界索引)。
// ---------------------------------------------------------------------------

struct GatherOpNameTag {
  static constexpr std::string_view kOpName = "gather";
};
using GatherOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<GatherOpNameTag>;

TEST_F(GatherOpKernelTest, RejectsUnsupportedXDtype) {
  Tensor x = MakeTensor<std::int32_t>(Shape({4, 2}));
  Tensor indices = MakeTensor<std::int64_t>(Shape({2}));
  Tensor out = MakeTensor<std::int32_t>(Shape({2, 2}));

  std::vector<Tensor> inputs{x, indices};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("does not support dtype 'int32'"), std::string_view::npos);
}

TEST_F(GatherOpKernelTest, RejectsUnsupportedIndicesDtype) {
  Tensor x = MakeTensor<float>(Shape({4, 2}));
  Tensor indices = MakeTensor<float>(Shape({2}));  // 浮点 indices,非法
  Tensor out = MakeTensor<float>(Shape({2, 2}));

  std::vector<Tensor> inputs{x, indices};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("requires indices dtype to be int32 or int64"),
            std::string_view::npos);
}

// 以下两类用例(越界索引、重复索引累加)需要构造带具体数值的 indices/gy
// 张量,ElementwiseKernelTestBase::MakeTensor 只分配不填值;改用
// ElementwiseEagerTestBase(MakeTensor1D/MakeTensorWithShape)+
// KernelRegistry::find 手动取 KernelFn 直接调用(同 test_pool2d.cpp
// MaxPool2dGradInternalTieBreakTest 先例)。

class GatherKernelValueTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(GatherKernelValueTest, RejectsOutOfRangeIndex) {
  // N=4,indices 含 4(越界,合法范围 [0,4))。
  const Result<frame::ops::KernelFn> found =
      frame::ops::KernelRegistry::instance().find("gather", frame::kCpuBackendName);
  ASSERT_TRUE(found.is_ok()) << found.status().message();
  const frame::ops::KernelFn kernel = found.value();

  Tensor x =
      MakeTensorWithShape<float>(Shape({4, 2}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
  Tensor indices = MakeTensor1D<std::int64_t>({0, 4});
  Tensor out = MakeTensorWithShape<float>(Shape({2, 2}), {0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{x, indices};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const Status status = kernel(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("indices[1]=4"), std::string_view::npos);
  EXPECT_NE(status.message().find("out of range for x's first dimension N=4"),
            std::string_view::npos);
}

TEST_F(GatherKernelValueTest, GatherGradInternalRejectsOutOfRangeIndex) {
  const Result<frame::ops::KernelFn> found =
      frame::ops::KernelRegistry::instance().find("gather_grad_internal", frame::kCpuBackendName);
  ASSERT_TRUE(found.is_ok()) << found.status().message();
  const frame::ops::KernelFn kernel = found.value();

  Tensor gy = MakeTensorWithShape<float>(Shape({2, 2}), {1.0F, 2.0F, 3.0F, 4.0F});
  Tensor indices = MakeTensor1D<std::int64_t>({0, 4});  // N=4,4 越界
  Tensor gx =
      MakeTensorWithShape<float>(Shape({4, 2}), {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{gy, indices};
  std::vector<Tensor> outputs{gx};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  const AttrMap attrs{{"input_shape", Shape({4, 2})}};
  ctx.attrs = &attrs;
  ctx.device = cpu_device();

  const Status status = kernel(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("indices[1]=4"), std::string_view::npos);
  EXPECT_NE(status.message().find("out of range for gx's first dimension N=4"),
            std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 5. 重复索引累加手算例(直接调用 gather_grad_internal kernel)。
// ---------------------------------------------------------------------------

TEST_F(GatherKernelValueTest, GatherGradInternalDuplicateIndicesAccumulate) {
  const Result<frame::ops::KernelFn> found =
      frame::ops::KernelRegistry::instance().find("gather_grad_internal", frame::kCpuBackendName);
  ASSERT_TRUE(found.is_ok()) << found.status().message();
  const frame::ops::KernelFn kernel = found.value();

  // indices=[2,0,2],gy=[[10,20],[1,1],[100,200]]:
  //   行0(来自 indices[1]=0)= gy[1] = [1,1]
  //   行2(来自 indices[0]=2 与 indices[2]=2)= gy[0]+gy[2] = [110,220]
  //   行1/3 无贡献 = [0,0]
  Tensor gy = MakeTensorWithShape<float>(Shape({3, 2}), {10.0F, 20.0F, 1.0F, 1.0F, 100.0F, 200.0F});
  Tensor indices = MakeTensor1D<std::int64_t>({2, 0, 2});
  Tensor gx =
      MakeTensorWithShape<float>(Shape({4, 2}), {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});
  Tensor expected = MakeTensorWithShape<float>(
      Shape({4, 2}), {1.0F, 1.0F, 0.0F, 0.0F, 110.0F, 220.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{gy, indices};
  std::vector<Tensor> outputs{gx};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  const AttrMap attrs{{"input_shape", Shape({4, 2})}};
  ctx.attrs = &attrs;
  ctx.device = cpu_device();

  const Status status = kernel(ctx);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 6. 图编译路径。
// ---------------------------------------------------------------------------

class GatherOpCompileTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(GatherOpCompileTest, CompiledExecutionMatchesKnownValues) {
  Graph graph("gather_only");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {4, 2})).value();
  Value* indices = graph.add_graph_input(MakeType(DType::of<std::int64_t>(), {3})).value();
  Node* gather_node = create_node_with_inferred_types(graph, "gather", {x, indices}).value();
  ASSERT_TRUE(graph.mark_output(gather_node, 0).is_ok());

  const Result<std::shared_ptr<frame::hal::Executable>> executable =
      frame::runtime::compile(graph, frame::kCpuBackendName, frame::hal::CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  Tensor x_tensor =
      MakeTensorWithShape<float>(Shape({4, 2}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
  Tensor indices_tensor = MakeTensor1D<std::int64_t>({2, 0, 2});
  std::vector<Tensor> inputs{x_tensor, indices_tensor};
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  Tensor expected = MakeTensorWithShape<float>(Shape({3, 2}), {5.0F, 6.0F, 1.0F, 2.0F, 5.0F, 6.0F});
  EXPECT_TRUE(
      tensor_all_close(outputs.value()[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 7. 解析梯度 ≡ 数值微分(对 x;indices 取含重复索引的样本以覆盖累加路径)。
// ---------------------------------------------------------------------------

class GatherGradientTest : public frame::ops::testing::ElementwiseEagerTestBase {};

// loss=sum(gather(x,indices)):线性组合(各行按 indices 出现次数加权求和),
// 梯度 gx[i,:]=count(indices==i)(重复索引 2 出现两次,累加路径非平凡覆盖)。
Graph BuildGatherLossGraph() {
  Graph graph("gather_loss");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {4, 2})).value();
  Value* indices = graph.add_graph_input(MakeType(DType::of<std::int64_t>(), {3})).value();
  Node* gather_node = create_node_with_inferred_types(graph, "gather", {x, indices}).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {gather_node->output(0)}, sum_attrs).value();
  const Status mark_status = graph.mark_output(sum_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

constexpr double kGatherCentralDifferenceH = 1e-2;

TEST_F(GatherGradientTest, GradientMatchesNumericForXWithDuplicateIndices) {
  const Graph forward = BuildGatherLossGraph();
  Tensor x = MakeTensorWithShape<float>(Shape({4, 2}),
                                        {1.0F, -2.0F, 3.0F, -4.0F, 0.5F, -1.5F, 2.5F, -3.5F});
  Tensor indices = MakeTensor1D<std::int64_t>({2, 0, 2});  // 索引 2 重复,覆盖累加路径
  std::vector<Tensor> inputs{x, indices};
  const std::vector<int32_t> wrt{0};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kGatherCentralDifferenceH));
}

TEST(GatherCompatibilityTest, InternalRejectsPublicAttributeNameWithLegacyDiagnostic) {
  const OpSchema* schema = OpRegistry::instance().find("gather_grad_internal");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "gather_grad_internal";
  ctx.input_types = {MakeType(DType::of<float>(), {3, 2}), MakeType(DType::of<int64_t>(), {3})};
  const AttrMap attrs{{"output_shape", Shape({4, 2})}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_NE(result.status().message().find(
                "op 'gather_grad_internal' is missing required attribute 'input_shape'"),
            std::string_view::npos);
}

TEST(GatherCompatibilityTest, GradientStillProducesLegacyInternalNodeAndAttribute) {
  const OpSchema* schema = OpRegistry::instance().find("gather");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "gather";
  ctx.input_types = {MakeType(DType::of<float>(), {4, 2}), MakeType(DType::of<int64_t>(), {3})};
  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  const std::string dump = frame::ir::dump_text(micrograph.value());
  EXPECT_NE(dump.find("gather_grad_internal"), std::string::npos);
  EXPECT_NE(dump.find("input_shape=shape[4,2]"), std::string::npos);
  EXPECT_EQ(dump.find("scatter_add"), std::string::npos);
  EXPECT_EQ(dump.find("output_shape"), std::string::npos);
}

}  // namespace
