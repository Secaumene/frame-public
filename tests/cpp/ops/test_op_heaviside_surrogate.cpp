// M27 heaviside_surrogate 回归:精确阶跃前向、schema/属性拒绝路径、三档
// CPU 数值，以及代理梯度闭式值与平滑 sigmoid 中心差分。BUILD-011 明确
// 禁止把该 GradientFn 与离散阶跃前向做中心差分，本文件只对平滑代理差分。
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <frame/compiler/autograd.h>
#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/backend.h>
#include <frame/hal/executable.h>
#include <frame/ir/attribute.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/graph_builder.h>
#include <frame/ops/op_registry.h>
#include <frame/runtime/compile.h>

#include "../common/tolerance.h"
#include "elementwise_op_test_helpers.h"

namespace {

using frame::bfloat16_t;
using frame::DType;
using frame::DTypeCode;
using frame::ErrorCode;
using frame::float16_t;
using frame::float_to_bfloat16;
using frame::float_to_float16;
using frame::kDynamicDim;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::Tensor;
using frame::hal::CompileOptions;
using frame::hal::Executable;
using frame::ir::AttrValue;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::Value;
using frame::ops::AttrMap;
using frame::ops::create_node_with_inferred_types;
using frame::ops::NodeContext;
using frame::ops::OpRegistry;
using frame::ops::OpSchema;
using frame::ops::OpTrait;
using frame::ops::testing::MakeType;
using frame::testing::default_tolerance;
using frame::testing::relaxed_tolerance;
using frame::testing::tensor_all_close;

constexpr double kAlpha = 2.0;

NodeContext MakeContext(DType dtype, std::initializer_list<int64_t> dims, const AttrMap* attrs) {
  NodeContext ctx;
  ctx.op = "heaviside_surrogate";
  ctx.input_types = {MakeType(dtype, dims)};
  ctx.attrs = attrs;
  return ctx;
}

TEST(HeavisideSurrogateSchemaTest, RegisteredWithRequiredDoubleAlphaAndGradient) {
  const OpSchema* schema = OpRegistry::instance().find("heaviside_surrogate");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 1U);
  EXPECT_EQ(schema->outputs().size(), 1U);
  ASSERT_EQ(schema->attrs().size(), 1U);
  EXPECT_EQ(schema->attrs()[0].name, "alpha");
  EXPECT_EQ(schema->attrs()[0].type, frame::ir::AttrType::kDouble);
  EXPECT_TRUE(schema->attrs()[0].required);
  EXPECT_TRUE(schema->has_trait(OpTrait::kElementwise));
  EXPECT_FALSE(schema->has_trait(OpTrait::kFusable));
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);
}

TEST(HeavisideSurrogateShapeInferTest, PreservesStaticShapeAndSupportedDtype) {
  const OpSchema* schema = OpRegistry::instance().find("heaviside_surrogate");
  ASSERT_NE(schema, nullptr);
  const AttrMap attrs{{"alpha", kAlpha}};
  for (DType dtype : {DType::of<float>(), DType::of<float16_t>(), DType::of<bfloat16_t>()}) {
    const Result<std::vector<Shape>> result =
        schema->shape_infer()(MakeContext(dtype, {2, 3}, &attrs));
    ASSERT_TRUE(result.is_ok()) << result.status().message();
    ASSERT_EQ(result.value().size(), 1U);
    EXPECT_EQ(result.value()[0], Shape({2, 3}));
  }
}

TEST(HeavisideSurrogateShapeInferTest, RejectsWrongInputCount) {
  const OpSchema* schema = OpRegistry::instance().find("heaviside_surrogate");
  ASSERT_NE(schema, nullptr);
  const AttrMap attrs{{"alpha", kAlpha}};
  NodeContext ctx = MakeContext(DType::of<float>(), {2}, &attrs);
  ctx.input_types.push_back(MakeType(DType::of<float>(), {2}));
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
}

TEST(HeavisideSurrogateShapeInferTest, RejectsDynamicShapeAndUnsupportedDtype) {
  const OpSchema* schema = OpRegistry::instance().find("heaviside_surrogate");
  ASSERT_NE(schema, nullptr);
  const AttrMap attrs{{"alpha", kAlpha}};
  const Result<std::vector<Shape>> dynamic =
      schema->shape_infer()(MakeContext(DType::of<float>(), {2, kDynamicDim}, &attrs));
  ASSERT_FALSE(dynamic.is_ok());
  EXPECT_EQ(dynamic.status().code(), ErrorCode::kInvalidArgument);
  const Result<std::vector<Shape>> integer =
      schema->shape_infer()(MakeContext(DType::of<int32_t>(), {2, 3}, &attrs));
  ASSERT_FALSE(integer.is_ok());
  EXPECT_EQ(integer.status().code(), ErrorCode::kInvalidArgument);
  const Result<std::vector<Shape>> overflow = schema->shape_infer()(
      MakeContext(DType::of<float>(), {std::numeric_limits<int64_t>::max(), 2}, &attrs));
  ASSERT_FALSE(overflow.is_ok());
  EXPECT_EQ(overflow.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(overflow.status().message().find("overflows int64"), std::string_view::npos);
}

TEST(HeavisideSurrogateShapeInferTest, RejectsMissingNonPositiveAndNonFiniteAlpha) {
  const OpSchema* schema = OpRegistry::instance().find("heaviside_surrogate");
  ASSERT_NE(schema, nullptr);
  NodeContext missing = MakeContext(DType::of<float>(), {2}, nullptr);
  EXPECT_FALSE(schema->shape_infer()(missing).is_ok());
  for (double alpha : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
    const AttrMap attrs{{"alpha", alpha}};
    const Result<std::vector<Shape>> result =
        schema->shape_infer()(MakeContext(DType::of<float>(), {2}, &attrs));
    ASSERT_FALSE(result.is_ok());
    EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  }
}

class HeavisideSurrogateCpuTest : public frame::ops::testing::ElementwiseEagerTestBase {
 protected:
  template <typename T>
  Tensor Run(const std::vector<T>& values, double alpha = kAlpha) {
    Tensor x = MakeTensor1D<T>(values);
    Tensor out = MakeTensor1D<T>(std::vector<T>(values.size(), T{}));
    std::vector<Tensor> inputs{x};
    std::vector<Tensor> outputs{out};
    const std::unordered_map<std::string, AttrValue> attrs{{"alpha", alpha}};
    frame::hal::KernelInvocation invocation;
    invocation.op = "heaviside_surrogate";
    invocation.inputs = inputs;
    invocation.outputs = outputs;
    invocation.attrs = &attrs;
    invocation.device = device_;
    const Status status = backend_->launch(invocation, nullptr);
    EXPECT_TRUE(status.is_ok()) << status.message();
    return outputs[0];
  }
};

TEST_F(HeavisideSurrogateCpuTest, Float32HandlesNegativeSignedZerosAndPositiveValues) {
  const float negative_zero = -0.0F;
  const Tensor actual = Run<float>({-2.0F, negative_zero, 0.0F, 0.25F});
  const Tensor expected = MakeTensor1D<float>({0.0F, 1.0F, 1.0F, 1.0F});
  EXPECT_TRUE(tensor_all_close(actual, expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(HeavisideSurrogateCpuTest, Float16MatchesExactStep) {
  const Tensor actual = Run<float16_t>({float_to_float16(-1.0F), float_to_float16(-0.0F),
                                        float_to_float16(0.0F), float_to_float16(1.0F)});
  const Tensor expected = MakeTensor1D<float16_t>({float_to_float16(0.0F), float_to_float16(1.0F),
                                                   float_to_float16(1.0F), float_to_float16(1.0F)});
  EXPECT_TRUE(tensor_all_close(actual, expected, default_tolerance(DTypeCode::kFloat16)));
}

TEST_F(HeavisideSurrogateCpuTest, BFloat16MatchesExactStep) {
  const Tensor actual = Run<bfloat16_t>({float_to_bfloat16(-1.0F), float_to_bfloat16(-0.0F),
                                         float_to_bfloat16(0.0F), float_to_bfloat16(1.0F)});
  const Tensor expected =
      MakeTensor1D<bfloat16_t>({float_to_bfloat16(0.0F), float_to_bfloat16(1.0F),
                                float_to_bfloat16(1.0F), float_to_bfloat16(1.0F)});
  EXPECT_TRUE(tensor_all_close(actual, expected, default_tolerance(DTypeCode::kBFloat16)));
}

TEST_F(HeavisideSurrogateCpuTest, EmptyTensorReturnsWithoutAccessingStorage) {
  const Tensor actual = Run<float>({});
  EXPECT_EQ(actual.shape(), Shape({0}));
  EXPECT_EQ(actual.numel(), 0);
  EXPECT_EQ(actual.raw_data(), nullptr);
}

TEST_F(HeavisideSurrogateCpuTest, EmptyGradientMicrographRunsWithoutAccessingStorage) {
  const OpSchema* schema = OpRegistry::instance().find("heaviside_surrogate");
  ASSERT_NE(schema, nullptr);
  const AttrMap attrs{{"alpha", kAlpha}};
  NodeContext ctx = MakeContext(DType::of<float>(), {0}, &attrs);
  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  const Result<std::shared_ptr<frame::hal::Executable>> executable = frame::runtime::compile(
      micrograph.value(), frame::kCpuBackendName, frame::hal::CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();
  const Tensor empty = MakeTensor1D<float>({});
  const std::vector<Tensor> inputs{empty, empty, empty};
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  ASSERT_EQ(outputs.value().size(), 1U);
  EXPECT_EQ(outputs.value()[0].shape(), Shape({0}));
  EXPECT_EQ(outputs.value()[0].raw_data(), nullptr);
}

TEST_F(HeavisideSurrogateCpuTest, KernelRejectsInvalidAlpha) {
  Tensor x = MakeTensor1D<float>({-1.0F, 1.0F});
  Tensor out = MakeTensor1D<float>({0.0F, 0.0F});
  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  for (double alpha : {0.0, -1.0, std::numeric_limits<double>::infinity()}) {
    const std::unordered_map<std::string, AttrValue> attrs{{"alpha", alpha}};
    frame::hal::KernelInvocation invocation;
    invocation.op = "heaviside_surrogate";
    invocation.inputs = inputs;
    invocation.outputs = outputs;
    invocation.attrs = &attrs;
    invocation.device = device_;
    const Status status = backend_->launch(invocation, nullptr);
    EXPECT_FALSE(status.is_ok());
    EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  }
}

Graph BuildWeightedHeavisideLossGraph(const Shape& shape, double alpha) {
  Graph graph("heaviside_surrogate_weighted_loss");
  frame::ir::TensorType type;
  type.dtype = DType::of<float>();
  type.shape = shape;
  type.device = frame::cpu_device();
  Value* x = graph.add_graph_input(type).value();
  Value* gy = graph.add_graph_input(type).value();
  const AttrMap attrs{{"alpha", alpha}};
  Node* spike = create_node_with_inferred_types(graph, "heaviside_surrogate", {x}, attrs).value();
  Node* weighted = create_node_with_inferred_types(graph, "mul", {spike->output(0), gy}).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* loss =
      create_node_with_inferred_types(graph, "sum", {weighted->output(0)}, sum_attrs).value();
  EXPECT_TRUE(graph.mark_output(loss, 0).is_ok());
  return graph;
}

TEST_F(HeavisideSurrogateCpuTest, ProxyGradientMatchesClosedFormAndSmoothCentralDifference) {
  constexpr double kGradientAlpha = 3.0;
  constexpr double kCentralDifferenceH = 1e-3;
  const Shape shape({4});
  const Graph forward = BuildWeightedHeavisideLossGraph(shape, kGradientAlpha);
  const std::vector<int32_t> wrt{0};
  const Result<Graph> training = frame::compiler::build_backward_graph(forward, 0, wrt);
  ASSERT_TRUE(training.is_ok()) << training.status().message();
  const Result<std::shared_ptr<Executable>> executable =
      frame::runtime::compile(training.value(), frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  const std::vector<float> x_values{-1.25F, -0.2F, 0.0F, 0.8F};
  const std::vector<float> gy_values{0.5F, -1.5F, 2.0F, 0.75F};
  const Tensor x = MakeTensorWithShape<float>(shape, x_values);
  const Tensor gy = MakeTensorWithShape<float>(shape, gy_values);
  const std::vector<Tensor> run_inputs{x, gy};
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, run_inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  ASSERT_EQ(outputs.value().size(), 2U);

  std::vector<float> closed_form;
  std::vector<float> smooth_numeric;
  closed_form.reserve(x_values.size());
  smooth_numeric.reserve(x_values.size());
  for (size_t i = 0; i < x_values.size(); ++i) {
    const double x_value = static_cast<double>(x_values[i]);
    const double gy_value = static_cast<double>(gy_values[i]);
    const double sigmoid = 1.0 / (1.0 + std::exp(-kGradientAlpha * x_value));
    closed_form.push_back(
        static_cast<float>(gy_value * kGradientAlpha * sigmoid * (1.0 - sigmoid)));
    const auto smooth_proxy = [&](double value) {
      return gy_value / (1.0 + std::exp(-kGradientAlpha * value));
    };
    smooth_numeric.push_back(static_cast<float>((smooth_proxy(x_value + kCentralDifferenceH) -
                                                 smooth_proxy(x_value - kCentralDifferenceH)) /
                                                (2.0 * kCentralDifferenceH)));
  }

  const Tensor expected_closed = MakeTensorWithShape<float>(shape, closed_form);
  EXPECT_TRUE(tensor_all_close(outputs.value()[1], expected_closed,
                               default_tolerance(DTypeCode::kFloat32)));
  const Tensor expected_numeric = MakeTensorWithShape<float>(shape, smooth_numeric);
  // BUILD-011 数值微分专款:平滑 sigmoid 代理的 fp32 中心差分含 O(h^2)
  // 截断误差，按规范使用下一档容差；没有对离散阶跃前向做差分。
  EXPECT_TRUE(tensor_all_close(outputs.value()[1], expected_numeric,
                               relaxed_tolerance(DTypeCode::kFloat32)));
}

}  // namespace
