// layer_norm 算子测试(schema + cpu kernel + 解析梯度 ≡ 数值微分,
// src/ops/schemas/sequence.cpp、src/backends/cpu/kernels/sequence.cpp 已实化
// 的行为,M22 批4 T3,§1.2/1.3 决议点B/C)。layer_norm(x[N,D], gamma[D],
// beta[D]; eps):行归一化 + 仿射,gamma/beta 算子内沿行广播(conv bias 先例)。
// 梯度微图约 48 个非输入节点(§1.2 表推导),经 sum/mul/add/constant/reshape/
// matmul/rsqrt 全部已注册可微算子表达(R11)。
//   1. schema 字段/三函数指针状态;
//   2. shape_infer 合法路径 + 负例(x rank/gamma rank/gamma size/beta rank/
//      beta size/eps 缺失/eps<=0);
//   3. eager 数值:2 行手算已知值(fp32 精确到容差,gamma/beta 非均匀验证
//      仿射部分)、fp16(位级转换);
//   4. layer_norm_cpu_kernel 自身的防御性拒绝路径;
//   5. 图编译路径;
//   6. 解析梯度 ≡ 数值微分(x/gamma/beta 三输入全验,gamma 非均匀避开
//      "sum(xhat)恒零"退化梯度)。
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
using frame::float16_t;
using frame::float_to_float16;
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

// 标准 3 输入 layer_norm attrs(eps=1e-5)供多处用例共用。
const AttrMap& StandardLayerNormAttrs() {
  static const AttrMap attrs{{"eps", 1e-5}};
  return attrs;
}

// ---------------------------------------------------------------------------
// 1. schema 断言。
// ---------------------------------------------------------------------------

TEST(LayerNormOpSchemaTest, RegisteredWithThreeInputsOneOutputOneAttr) {
  const OpSchema* schema = OpRegistry::instance().find("layer_norm");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 3U);
  EXPECT_EQ(schema->outputs().size(), 1U);
  ASSERT_EQ(schema->attrs().size(), 1U);
  EXPECT_EQ(schema->attrs()[0].name, "eps");
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

TEST(LayerNormShapeInferTest, ValidInputsProduceXShape) {
  const OpSchema* schema = OpRegistry::instance().find("layer_norm");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "layer_norm";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 4}), MakeType(DType::of<float>(), {4}),
                     MakeType(DType::of<float>(), {4})};
  ctx.attrs = &StandardLayerNormAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({2, 4}));
}

TEST(LayerNormShapeInferTest, WrongInputCountIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("layer_norm");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "layer_norm";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 4})};
  ctx.attrs = &StandardLayerNormAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("expects 3 inputs (x, gamma, beta), got 1"),
            std::string_view::npos);
}

TEST(LayerNormShapeInferTest, XRankNotTwoIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("layer_norm");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "layer_norm";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 4, 1}), MakeType(DType::of<float>(), {4}),
                     MakeType(DType::of<float>(), {4})};
  ctx.attrs = &StandardLayerNormAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires x to be rank-2 [N, D], got rank 3"),
            std::string_view::npos);
}

TEST(LayerNormShapeInferTest, GammaRankNotOneIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("layer_norm");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "layer_norm";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 4}), MakeType(DType::of<float>(), {1, 4}),
                     MakeType(DType::of<float>(), {4})};
  ctx.attrs = &StandardLayerNormAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires gamma to be rank-1 [D], got rank 2"),
            std::string_view::npos);
}

TEST(LayerNormShapeInferTest, GammaSizeMismatchIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("layer_norm");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "layer_norm";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 4}), MakeType(DType::of<float>(), {3}),
                     MakeType(DType::of<float>(), {4})};
  ctx.attrs = &StandardLayerNormAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires gamma size to equal D=4, got 3"),
            std::string_view::npos);
}

TEST(LayerNormShapeInferTest, BetaRankNotOneIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("layer_norm");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "layer_norm";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 4}), MakeType(DType::of<float>(), {4}),
                     MakeType(DType::of<float>(), {4, 1})};
  ctx.attrs = &StandardLayerNormAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires beta to be rank-1 [D], got rank 2"),
            std::string_view::npos);
}

TEST(LayerNormShapeInferTest, BetaSizeMismatchIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("layer_norm");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "layer_norm";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 4}), MakeType(DType::of<float>(), {4}),
                     MakeType(DType::of<float>(), {5})};
  ctx.attrs = &StandardLayerNormAttrs();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires beta size to equal D=4, got 5"),
            std::string_view::npos);
}

TEST(LayerNormShapeInferTest, MissingEpsIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("layer_norm");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "layer_norm";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 4}), MakeType(DType::of<float>(), {4}),
                     MakeType(DType::of<float>(), {4})};
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("missing required attribute 'eps'"),
            std::string_view::npos);
}

TEST(LayerNormShapeInferTest, NonPositiveEpsIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("layer_norm");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "layer_norm";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 4}), MakeType(DType::of<float>(), {4}),
                     MakeType(DType::of<float>(), {4})};
  const AttrMap attrs{{"eps", 0.0}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("attribute 'eps' must be positive"),
            std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 3. eager 数值(2 行手算已知值,gamma/beta 非均匀验证仿射部分;参考值经
// Python 独立算得:mean/var/rsqrt/仿射,eps=1e-5)。
// ---------------------------------------------------------------------------

class LayerNormOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(LayerNormOpEagerTest, TwoRowsKnownValues) {
  Tensor x =
      MakeTensorWithShape<float>(Shape({2, 4}), {1.0F, 2.0F, 3.0F, 4.0F, 0.0F, 2.0F, 4.0F, 6.0F});
  Tensor gamma = MakeTensor1D<float>({2.0F, 0.5F, 1.0F, -1.0F});
  Tensor beta = MakeTensor1D<float>({0.1F, -0.2F, 0.0F, 1.0F});
  Tensor expected = MakeTensorWithShape<float>(
      Shape({2, 4}), {-2.5832708F, -0.4236059F, 0.4472118F, -0.3416354F, -2.5832789F, -0.4236066F,
                      0.4472131F, -0.3416394F});
  Tensor out =
      MakeTensorWithShape<float>(Shape({2, 4}), {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{x, gamma, beta};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "layer_norm";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &StandardLayerNormAttrs();
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(LayerNormOpEagerTest, Float16KnownValuesGammaOneBetaZero) {
  // gamma=1/beta=0(纯归一化,fp16 精度验证):x=[1,2,3,4],
  // mean=2.5,var=1.25,r=1/sqrt(1.25+eps)≈0.8944236,
  // xhat≈[-1.3416354,-0.4472118,0.4472118,1.3416354]。
  Tensor x = MakeTensorWithShape<float16_t>(
      Shape({1, 4}), {float_to_float16(1.0F), float_to_float16(2.0F), float_to_float16(3.0F),
                      float_to_float16(4.0F)});
  Tensor gamma = MakeTensor1D<float16_t>({float_to_float16(1.0F), float_to_float16(1.0F),
                                          float_to_float16(1.0F), float_to_float16(1.0F)});
  Tensor beta = MakeTensor1D<float16_t>({float_to_float16(0.0F), float_to_float16(0.0F),
                                         float_to_float16(0.0F), float_to_float16(0.0F)});
  Tensor expected = MakeTensorWithShape<float16_t>(
      Shape({1, 4}), {float_to_float16(-1.3416354F), float_to_float16(-0.4472118F),
                      float_to_float16(0.4472118F), float_to_float16(1.3416354F)});
  Tensor out = MakeTensorWithShape<float16_t>(
      Shape({1, 4}), {float16_t{0}, float16_t{0}, float16_t{0}, float16_t{0}});

  std::vector<Tensor> inputs{x, gamma, beta};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "layer_norm";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &StandardLayerNormAttrs();
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat16)));
}

// ---------------------------------------------------------------------------
// 4. layer_norm_cpu_kernel 自身的防御性拒绝路径。
// ---------------------------------------------------------------------------

struct LayerNormOpNameTag {
  static constexpr std::string_view kOpName = "layer_norm";
};
using LayerNormOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<LayerNormOpNameTag>;

TEST_F(LayerNormOpKernelTest, RejectsUnsupportedDtypeInt32) {
  Tensor x = MakeTensor<std::int32_t>(Shape({2, 4}));
  Tensor gamma = MakeTensor<std::int32_t>(Shape({4}));
  Tensor beta = MakeTensor<std::int32_t>(Shape({4}));
  Tensor out = MakeTensor<std::int32_t>(Shape({2, 4}));

  std::vector<Tensor> inputs{x, gamma, beta};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.attrs = &StandardLayerNormAttrs();
  ctx.device = cpu_device();

  const Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("does not support dtype 'int32'"), std::string_view::npos);
}

TEST_F(LayerNormOpKernelTest, RejectsGammaSizeMismatch) {
  Tensor x = MakeTensor<float>(Shape({2, 4}));
  Tensor gamma = MakeTensor<float>(Shape({3}));
  Tensor beta = MakeTensor<float>(Shape({4}));
  Tensor out = MakeTensor<float>(Shape({2, 4}));

  std::vector<Tensor> inputs{x, gamma, beta};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.attrs = &StandardLayerNormAttrs();
  ctx.device = cpu_device();

  const Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("requires gamma to be rank-1 [D=4]"), std::string_view::npos);
}

TEST_F(LayerNormOpKernelTest, RejectsMissingEps) {
  Tensor x = MakeTensor<float>(Shape({2, 4}));
  Tensor gamma = MakeTensor<float>(Shape({4}));
  Tensor beta = MakeTensor<float>(Shape({4}));
  Tensor out = MakeTensor<float>(Shape({2, 4}));

  std::vector<Tensor> inputs{x, gamma, beta};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();  // 未设置 ctx.attrs

  const Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("missing required attribute 'eps'"), std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 5. 图编译路径。
// ---------------------------------------------------------------------------

class LayerNormOpCompileTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(LayerNormOpCompileTest, CompiledExecutionMatchesKnownValues) {
  Graph graph("layer_norm_only");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {2, 4})).value();
  Value* gamma = graph.add_graph_input(MakeType(DType::of<float>(), {4})).value();
  Value* beta = graph.add_graph_input(MakeType(DType::of<float>(), {4})).value();
  Node* ln_node = create_node_with_inferred_types(graph, "layer_norm", {x, gamma, beta},
                                                  StandardLayerNormAttrs())
                      .value();
  ASSERT_TRUE(graph.mark_output(ln_node, 0).is_ok());

  const Result<std::shared_ptr<frame::hal::Executable>> executable =
      frame::runtime::compile(graph, frame::kCpuBackendName, frame::hal::CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  Tensor x_tensor =
      MakeTensorWithShape<float>(Shape({2, 4}), {1.0F, 2.0F, 3.0F, 4.0F, 0.0F, 2.0F, 4.0F, 6.0F});
  Tensor gamma_tensor = MakeTensor1D<float>({2.0F, 0.5F, 1.0F, -1.0F});
  Tensor beta_tensor = MakeTensor1D<float>({0.1F, -0.2F, 0.0F, 1.0F});
  std::vector<Tensor> inputs{x_tensor, gamma_tensor, beta_tensor};
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  Tensor expected = MakeTensorWithShape<float>(
      Shape({2, 4}), {-2.5832708F, -0.4236059F, 0.4472118F, -0.3416354F, -2.5832789F, -0.4236066F,
                      0.4472131F, -0.3416394F});
  EXPECT_TRUE(
      tensor_all_close(outputs.value()[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 6. 解析梯度 ≡ 数值微分(x/gamma/beta 三输入全验)。
// ---------------------------------------------------------------------------

class LayerNormGradientTest : public frame::ops::testing::ElementwiseEagerTestBase {};

// gamma 非均匀(避开"sum(xhat,axes=1)恒零→d(sum loss)/dx 结构性退化"的边界
// 情形,§1.2 表 ggamma=sum(gy·x̂,axes=[0]) 同样需要非均匀 gy 才有意义)。
Graph BuildLayerNormLossGraph() {
  Graph graph("layer_norm_loss");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {2, 4})).value();
  Value* gamma = graph.add_graph_input(MakeType(DType::of<float>(), {4})).value();
  Value* beta = graph.add_graph_input(MakeType(DType::of<float>(), {4})).value();
  Node* ln_node = create_node_with_inferred_types(graph, "layer_norm", {x, gamma, beta},
                                                  StandardLayerNormAttrs())
                      .value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {ln_node->output(0)}, sum_attrs).value();
  const Status mark_status = graph.mark_output(sum_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

constexpr double kLayerNormCentralDifferenceH = 1e-2;

TEST_F(LayerNormGradientTest, GradientMatchesNumericForXGammaAndBeta) {
  const Graph forward = BuildLayerNormLossGraph();
  Tensor x =
      MakeTensorWithShape<float>(Shape({2, 4}), {1.0F, 2.0F, 3.0F, 4.0F, 0.0F, 2.0F, 4.0F, 6.0F});
  Tensor gamma = MakeTensor1D<float>({2.0F, 0.5F, 1.0F, -1.0F});
  Tensor beta = MakeTensor1D<float>({0.1F, -0.2F, 0.0F, 1.0F});
  std::vector<Tensor> inputs{x, gamma, beta};
  const std::vector<int32_t> wrt{0, 1, 2};
  EXPECT_TRUE(
      CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kLayerNormCentralDifferenceH));
  EXPECT_TRUE(
      CheckGradientMatchesNumeric(forward, 0, wrt, 1, inputs, kLayerNormCentralDifferenceH));
  EXPECT_TRUE(
      CheckGradientMatchesNumeric(forward, 0, wrt, 2, inputs, kLayerNormCentralDifferenceH));
}

}  // namespace
