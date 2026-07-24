// selective_scan 算子回归(M25,docs/plan/2026-07-23-batch6-m25-ssm.md):
// schema 正负例、CPU 手算递推、steps=1、任意前导维，以及五输入解析梯度与
// fp32 中心差分的一致性。数值断言统一走 BUILD-011 容差工具。
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <string_view>
#include <vector>

#include <frame/compiler/autograd.h>
#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/executable.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/graph_builder.h>
#include <frame/ops/op_registry.h>
#include <frame/runtime/compile.h>

#include "../common/tolerance.h"
#include "elementwise_op_test_helpers.h"
#include "gradient_check_test_helpers.h"

namespace {

using frame::DType;
using frame::DTypeCode;
using frame::ErrorCode;
using frame::kDynamicDim;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::Tensor;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::Value;
using frame::ops::create_node_with_inferred_types;
using frame::ops::NodeContext;
using frame::ops::OpRegistry;
using frame::ops::OpSchema;
using frame::ops::testing::CheckGradientMatchesNumeric;
using frame::ops::testing::MakeType;
using frame::testing::default_tolerance;
using frame::testing::tensor_all_close;

std::vector<frame::ir::TensorType> MakeFiveTypes(DType dtype, std::initializer_list<int64_t> dims) {
  const frame::ir::TensorType type = MakeType(dtype, dims);
  return {type, type, type, type, type};
}

frame::ir::TensorType MakeTypeWithShape(DType dtype, const Shape& shape) {
  frame::ir::TensorType type;
  type.dtype = dtype;
  type.shape = shape;
  type.layout = frame::ir::Layout::kRowMajor;
  type.device = frame::cpu_device();
  return type;
}

TEST(SelectiveScanSchemaTest, RegisteredWithFiveInputsOneOutputAndGradient) {
  const OpSchema* schema = OpRegistry::instance().find("selective_scan");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 5U);
  EXPECT_EQ(schema->outputs().size(), 1U);
  EXPECT_TRUE(schema->attrs().empty());
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);
}

TEST(SelectiveScanShapeInferTest, RankOneAndLeadingDimensionsPreserveShape) {
  const OpSchema* schema = OpRegistry::instance().find("selective_scan");
  ASSERT_NE(schema, nullptr);
  for (const auto& input_types :
       {MakeFiveTypes(DType::of<float>(), {4}), MakeFiveTypes(DType::of<float>(), {2, 3, 4})}) {
    NodeContext ctx;
    ctx.op = "selective_scan";
    ctx.input_types = input_types;
    const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
    ASSERT_TRUE(result.is_ok()) << result.status().message();
    ASSERT_EQ(result.value().size(), 1U);
    EXPECT_EQ(result.value()[0], input_types[0].shape);
  }
}

TEST(SelectiveScanShapeInferTest, WrongInputCountIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("selective_scan");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "selective_scan";
  ctx.input_types = MakeFiveTypes(DType::of<float>(), {3});
  ctx.input_types.pop_back();
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("expects 5 inputs"), std::string_view::npos);
}

TEST(SelectiveScanShapeInferTest, ShapeMismatchIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("selective_scan");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "selective_scan";
  ctx.input_types = MakeFiveTypes(DType::of<float>(), {2, 3});
  ctx.input_types[2] = MakeType(DType::of<float>(), {2, 4});
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("all input shapes to match x"), std::string_view::npos);
}

TEST(SelectiveScanShapeInferTest, DtypeMismatchIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("selective_scan");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "selective_scan";
  ctx.input_types = MakeFiveTypes(DType::of<float>(), {3});
  ctx.input_types[4] = MakeType(DType::of<frame::float16_t>(), {3});
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("all input dtypes to match x"), std::string_view::npos);
}

TEST(SelectiveScanShapeInferTest, RankZeroIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("selective_scan");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "selective_scan";
  ctx.input_types = MakeFiveTypes(DType::of<float>(), {});
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires rank >= 1"), std::string_view::npos);
}

TEST(SelectiveScanShapeInferTest, DynamicDimensionIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("selective_scan");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "selective_scan";
  ctx.input_types = MakeFiveTypes(DType::of<float>(), {2, kDynamicDim});
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("dynamic dimension"), std::string_view::npos);
}

TEST(SelectiveScanShapeInferTest, NegativeAndOverflowingLeadingDimensionsAreRejected) {
  const OpSchema* schema = OpRegistry::instance().find("selective_scan");
  ASSERT_NE(schema, nullptr);
  for (const Shape& shape : {Shape({-2, 3}), Shape({std::numeric_limits<int64_t>::max(), 2})}) {
    NodeContext ctx;
    ctx.op = "selective_scan";
    frame::ir::TensorType type = MakeTypeWithShape(DType::of<float>(), shape);
    ctx.input_types = {type, type, type, type, type};
    const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
    ASSERT_FALSE(result.is_ok());
    EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  }
}

TEST(SelectiveScanShapeInferTest, ZeroStepsIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("selective_scan");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "selective_scan";
  ctx.input_types = MakeFiveTypes(DType::of<float>(), {2, 0});
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("steps) to be >= 1"), std::string_view::npos);
}

TEST(SelectiveScanShapeInferTest, UnsupportedDtypeIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("selective_scan");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "selective_scan";
  ctx.input_types = MakeFiveTypes(DType::of<int32_t>(), {3});
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("does not support dtype"), std::string_view::npos);
}

class SelectiveScanCpuTest : public frame::ops::testing::ElementwiseEagerTestBase {
 protected:
  Tensor Run(const Shape& shape, const std::vector<std::vector<float>>& values) {
    Graph graph("selective_scan_cpu");
    std::vector<Value*> graph_inputs;
    graph_inputs.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      graph_inputs.push_back(
          graph.add_graph_input(MakeTypeWithShape(DType::of<float>(), shape)).value());
    }
    Node* scan = create_node_with_inferred_types(graph, "selective_scan", graph_inputs).value();
    EXPECT_TRUE(graph.mark_output(scan, 0).is_ok());
    const Result<std::shared_ptr<frame::hal::Executable>> executable =
        frame::runtime::compile(graph, frame::kCpuBackendName, frame::hal::CompileOptions{});
    EXPECT_TRUE(executable.is_ok()) << executable.status().message();
    std::vector<Tensor> inputs;
    inputs.reserve(values.size());
    for (const std::vector<float>& input_values : values) {
      inputs.push_back(MakeTensorWithShape<float>(shape, input_values));
    }
    const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
        *executable.value(), frame::kCpuBackendName, inputs);
    EXPECT_TRUE(outputs.is_ok()) << outputs.status().message();
    return outputs.value()[0];
  }
};

TEST_F(SelectiveScanCpuTest, HandComputedRecurrenceMatches) {
  const Shape shape({3});
  const Tensor actual = Run(shape, {{1.0F, 2.0F, -1.0F},
                                    {0.5F, 0.25F, -0.5F},
                                    {2.0F, 1.0F, 3.0F},
                                    {1.0F, 2.0F, 0.5F},
                                    {0.1F, -1.0F, 2.0F}});
  // h=[2,2.5,-4.25],故 y=[2.1,3,-4.125]。
  const Tensor expected = MakeTensorWithShape<float>(shape, {2.1F, 3.0F, -4.125F});
  EXPECT_TRUE(tensor_all_close(actual, expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(SelectiveScanCpuTest, SingleStepUsesZeroInitialState) {
  const Shape shape({1});
  const Tensor actual = Run(shape, {{3.0F}, {0.2F}, {4.0F}, {0.5F}, {-1.0F}});
  const Tensor expected = MakeTensorWithShape<float>(shape, {3.0F});
  EXPECT_TRUE(tensor_all_close(actual, expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(SelectiveScanCpuTest, LeadingDimensionsDefineIndependentSequences) {
  const Shape shape({2, 2, 3});
  const std::vector<float> x{1, 2, 3, 4, 5, 6, -1, -2, -3, 2, 0, -2};
  const std::vector<float> a(12, 0.5F);
  const std::vector<float> b(12, 1.0F);
  const std::vector<float> c(12, 1.0F);
  const std::vector<float> d(12, 0.0F);
  const Tensor actual = Run(shape, {x, a, b, c, d});
  const Tensor expected = MakeTensorWithShape<float>(
      shape, {1, 2.5F, 4.25F, 4, 7, 9.5F, -1, -2.5F, -4.25F, 2, 1, -1.5F});
  EXPECT_TRUE(tensor_all_close(actual, expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(SelectiveScanCpuTest, EmptyLeadingDimensionReturnsWithoutAccessingStorage) {
  const Shape shape({0, 3});
  const Tensor actual = Run(shape, {{}, {}, {}, {}, {}});
  EXPECT_EQ(actual.shape(), shape);
  EXPECT_EQ(actual.numel(), 0);
  EXPECT_EQ(actual.raw_data(), nullptr);
}

TEST_F(SelectiveScanCpuTest, EmptyGradientMicrographRunsWithoutAccessingStorage) {
  const OpSchema* schema = OpRegistry::instance().find("selective_scan");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "selective_scan";
  ctx.input_types = MakeFiveTypes(DType::of<float>(), {0, 3});
  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  const Result<std::shared_ptr<frame::hal::Executable>> executable = frame::runtime::compile(
      micrograph.value(), frame::kCpuBackendName, frame::hal::CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();
  const Tensor empty = MakeTensorWithShape<float>(Shape({0, 3}), {});
  const std::vector<Tensor> inputs{empty, empty, empty, empty, empty, empty, empty};
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  ASSERT_EQ(outputs.value().size(), 5U);
  for (const Tensor& output : outputs.value()) {
    EXPECT_EQ(output.shape(), Shape({0, 3}));
    EXPECT_EQ(output.raw_data(), nullptr);
  }
}

Graph BuildSelectiveScanLossGraph(const Shape& shape) {
  Graph graph("selective_scan_loss");
  std::vector<Value*> inputs;
  inputs.reserve(5);
  for (int i = 0; i < 5; ++i) {
    inputs.push_back(graph.add_graph_input(MakeTypeWithShape(DType::of<float>(), shape)).value());
  }
  Node* scan = create_node_with_inferred_types(graph, "selective_scan", inputs).value();
  const frame::ops::AttrMap attrs{{"axes", std::vector<int64_t>{}}};
  Node* loss = create_node_with_inferred_types(graph, "sum", {scan->output(0)}, attrs).value();
  EXPECT_TRUE(graph.mark_output(loss, 0).is_ok());
  return graph;
}

class SelectiveScanGradientTest : public frame::ops::testing::ElementwiseEagerTestBase {};

constexpr double kSelectiveScanCentralDifferenceH = 1e-3;

TEST_F(SelectiveScanGradientTest, FiveInputGradientsMatchFloat32CentralDifference) {
  const Shape shape({3});
  const Graph forward = BuildSelectiveScanLossGraph(shape);
  std::vector<Tensor> inputs{MakeTensorWithShape<float>(shape, {0.4F, -0.7F, 1.2F}),
                             MakeTensorWithShape<float>(shape, {0.2F, -0.3F, 0.4F}),
                             MakeTensorWithShape<float>(shape, {0.8F, 0.5F, -0.6F}),
                             MakeTensorWithShape<float>(shape, {1.1F, -0.9F, 0.7F}),
                             MakeTensorWithShape<float>(shape, {0.1F, 0.2F, -0.15F})};
  const std::vector<int32_t> wrt{0, 1, 2, 3, 4};
  // BUILD-011 数值微分专款:fp32 中心差分 h=1e-3,断言由公共 helper 使用
  // relaxed_tolerance 放宽一档,不用于普通前向数值比较。
  for (size_t position = 0; position < wrt.size(); ++position) {
    EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, position, inputs,
                                            kSelectiveScanCentralDifferenceH));
  }
}

TEST_F(SelectiveScanGradientTest, SingleStepFiveInputGradientsMatchCentralDifference) {
  const Shape shape({1});
  const Graph forward = BuildSelectiveScanLossGraph(shape);
  std::vector<Tensor> inputs{
      MakeTensorWithShape<float>(shape, {0.4F}), MakeTensorWithShape<float>(shape, {0.2F}),
      MakeTensorWithShape<float>(shape, {0.8F}), MakeTensorWithShape<float>(shape, {1.1F}),
      MakeTensorWithShape<float>(shape, {0.1F})};
  const std::vector<int32_t> wrt{0, 1, 2, 3, 4};
  // 同上,steps=1 仍经同一解析微图公式与 BUILD-011 数值微分容差。
  for (size_t position = 0; position < wrt.size(); ++position) {
    EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, position, inputs,
                                            kSelectiveScanCentralDifferenceH));
  }
}

}  // namespace
