// M28 scatter_add 回归:公开 output_shape 合同、六种 dtype 组合、重复累加、
// 空输出行、越界与属性负例、编译执行及两输入梯度。旧 internal 兼容断言仍
// 留在 test_op_gather.cpp，避免公共属性名渗入旧 IR。
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
#include <frame/core/tensor.h>
#include <frame/hal/executable.h>
#include <frame/ir/graph.h>
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

TEST(ScatterAddSchemaTest, UsesOnlyRequiredOutputShapeAndHasGradient) {
  const OpSchema* schema = OpRegistry::instance().find("scatter_add");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 2U);
  EXPECT_EQ(schema->outputs().size(), 1U);
  ASSERT_EQ(schema->attrs().size(), 1U);
  EXPECT_EQ(schema->attrs()[0].name, "output_shape");
  EXPECT_EQ(schema->attrs()[0].type, frame::ir::AttrType::kShape);
  EXPECT_TRUE(schema->attrs()[0].required);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);
}

Result<std::vector<Shape>> InferScatter(DType updates_dtype, const Shape& updates_shape,
                                        DType indices_dtype, const Shape& indices_shape,
                                        const AttrMap* attrs) {
  const OpSchema* schema = OpRegistry::instance().find("scatter_add");
  NodeContext ctx;
  ctx.op = "scatter_add";
  frame::ir::TensorType updates = MakeType(updates_dtype, {});
  updates.shape = updates_shape;
  frame::ir::TensorType indices = MakeType(indices_dtype, {});
  indices.shape = indices_shape;
  ctx.input_types = {updates, indices};
  ctx.attrs = attrs;
  return schema->shape_infer()(ctx);
}

TEST(ScatterAddShapeInferTest, AcceptsThreeFloatAndTwoIndexDtypes) {
  const AttrMap attrs{{"output_shape", Shape({4, 2})}};
  for (DType value_dtype : {DType::of<float>(), DType::of<float16_t>(), DType::of<bfloat16_t>()}) {
    for (DType index_dtype : {DType::of<int32_t>(), DType::of<int64_t>()}) {
      const Result<std::vector<Shape>> result =
          InferScatter(value_dtype, Shape({3, 2}), index_dtype, Shape({3}), &attrs);
      ASSERT_TRUE(result.is_ok()) << result.status().message();
      EXPECT_EQ(result.value()[0], Shape({4, 2}));
    }
  }
}

TEST(ScatterAddShapeInferTest, AcceptsZeroUpdatesAndZeroOutputRows) {
  const AttrMap attrs{{"output_shape", Shape({0, 2})}};
  const Result<std::vector<Shape>> result =
      InferScatter(DType::of<float>(), Shape({0, 2}), DType::of<int64_t>(), Shape({0}), &attrs);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({0, 2}));
}

TEST(ScatterAddShapeInferTest, RejectsNegativeAndOverflowingElementCounts) {
  const AttrMap negative_output{{"output_shape", Shape({-2, 2})}};
  EXPECT_FALSE(InferScatter(DType::of<float>(), Shape({3, 2}), DType::of<int64_t>(), Shape({3}),
                            &negative_output)
                   .is_ok());

  const int64_t maximum = std::numeric_limits<int64_t>::max();
  const AttrMap overflowing_output{{"output_shape", Shape({maximum, 2})}};
  const Result<std::vector<Shape>> output_result = InferScatter(
      DType::of<float>(), Shape({0, 2}), DType::of<int64_t>(), Shape({0}), &overflowing_output);
  ASSERT_FALSE(output_result.is_ok());
  EXPECT_NE(output_result.status().message().find("output_shape"), std::string_view::npos);
  EXPECT_NE(output_result.status().message().find("overflows int64"), std::string_view::npos);

  const AttrMap valid_output{{"output_shape", Shape({1, 2})}};
  const Result<std::vector<Shape>> input_result =
      InferScatter(DType::of<float>(), Shape({maximum, 2}), DType::of<int64_t>(), Shape({maximum}),
                   &valid_output);
  ASSERT_FALSE(input_result.is_ok());
  EXPECT_NE(input_result.status().message().find("input 0"), std::string_view::npos);
  EXPECT_NE(input_result.status().message().find("overflows int64"), std::string_view::npos);
}

TEST(ScatterAddShapeInferTest, RejectsShapeDtypeAndAttributeViolations) {
  const AttrMap valid{{"output_shape", Shape({4, 2})}};
  EXPECT_FALSE(
      InferScatter(DType::of<float>(), Shape({3}), DType::of<int64_t>(), Shape({3}), &valid)
          .is_ok());
  EXPECT_FALSE(
      InferScatter(DType::of<float>(), Shape({3, 2}), DType::of<int64_t>(), Shape({3, 1}), &valid)
          .is_ok());
  EXPECT_FALSE(
      InferScatter(DType::of<float>(), Shape({3, 2}), DType::of<int64_t>(), Shape({2}), &valid)
          .is_ok());
  EXPECT_FALSE(
      InferScatter(DType::of<int32_t>(), Shape({3, 2}), DType::of<int64_t>(), Shape({3}), &valid)
          .is_ok());
  EXPECT_FALSE(
      InferScatter(DType::of<float>(), Shape({3, 2}), DType::of<float>(), Shape({3}), &valid)
          .is_ok());
  EXPECT_FALSE(
      InferScatter(DType::of<float>(), Shape({3, 2}), DType::of<int64_t>(), Shape({3}), nullptr)
          .is_ok());
  const AttrMap old_name{{"input_shape", Shape({4, 2})}};
  EXPECT_FALSE(
      InferScatter(DType::of<float>(), Shape({3, 2}), DType::of<int64_t>(), Shape({3}), &old_name)
          .is_ok());
  for (const Shape& invalid : {Shape({4}), Shape({4, 3}), Shape({kDynamicDim, 2})}) {
    const AttrMap attrs{{"output_shape", invalid}};
    EXPECT_FALSE(
        InferScatter(DType::of<float>(), Shape({3, 2}), DType::of<int64_t>(), Shape({3}), &attrs)
            .is_ok());
  }
}

class ScatterAddCpuTest : public frame::ops::testing::ElementwiseEagerTestBase {
 protected:
  template <typename ValueT, typename IndexT>
  void ExpectDuplicateAccumulation(const std::vector<ValueT>& updates_values,
                                   const std::vector<IndexT>& indices_values,
                                   const std::vector<ValueT>& expected_values, DTypeCode code) {
    const Tensor updates = MakeTensorWithShape<ValueT>(Shape({3, 2}), updates_values);
    const Tensor indices = MakeTensor1D<IndexT>(indices_values);
    Graph graph("scatter_add_compiled");
    Value* updates_value = graph.add_graph_input(MakeType(DType::of<ValueT>(), {3, 2})).value();
    Value* indices_value = graph.add_graph_input(MakeType(DType::of<IndexT>(), {3})).value();
    const AttrMap attrs{{"output_shape", Shape({4, 2})}};
    Node* scatter =
        create_node_with_inferred_types(graph, "scatter_add", {updates_value, indices_value}, attrs)
            .value();
    EXPECT_TRUE(graph.mark_output(scatter, 0).is_ok());
    const Result<std::shared_ptr<frame::hal::Executable>> executable =
        frame::runtime::compile(graph, frame::kCpuBackendName, frame::hal::CompileOptions{});
    ASSERT_TRUE(executable.is_ok()) << executable.status().message();
    const std::vector<Tensor> inputs{updates, indices};
    const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
        *executable.value(), frame::kCpuBackendName, inputs);
    ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
    const Tensor expected = MakeTensorWithShape<ValueT>(Shape({4, 2}), expected_values);
    EXPECT_TRUE(tensor_all_close(outputs.value()[0], expected, default_tolerance(code)));
  }

  void ExpectEmptyUpdates(const Shape& output_shape) {
    const Tensor updates = MakeTensorWithShape<float>(Shape({0, 2}), {});
    const Tensor indices = MakeTensor1D<int64_t>({});
    Graph graph("scatter_add_empty_updates");
    Value* updates_value = graph.add_graph_input(MakeType(DType::of<float>(), {0, 2})).value();
    Value* indices_value = graph.add_graph_input(MakeType(DType::of<int64_t>(), {0})).value();
    const AttrMap attrs{{"output_shape", output_shape}};
    Node* scatter =
        create_node_with_inferred_types(graph, "scatter_add", {updates_value, indices_value}, attrs)
            .value();
    EXPECT_TRUE(graph.mark_output(scatter, 0).is_ok());
    const Result<std::shared_ptr<frame::hal::Executable>> executable =
        frame::runtime::compile(graph, frame::kCpuBackendName, frame::hal::CompileOptions{});
    ASSERT_TRUE(executable.is_ok()) << executable.status().message();
    const std::vector<Tensor> inputs{updates, indices};
    const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
        *executable.value(), frame::kCpuBackendName, inputs);
    ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
    EXPECT_EQ(outputs.value()[0].shape(), output_shape);
    const Tensor expected = MakeTensorWithShape<float>(
        output_shape, std::vector<float>(static_cast<size_t>(output_shape.numel()), 0.0F));
    EXPECT_TRUE(
        tensor_all_close(outputs.value()[0], expected, default_tolerance(DTypeCode::kFloat32)));
  }
};

TEST_F(ScatterAddCpuTest, EmptyUpdatesZeroFillNonEmptyOutput) { ExpectEmptyUpdates(Shape({3, 2})); }

TEST_F(ScatterAddCpuTest, EmptyUpdatesSupportEmptyOutput) { ExpectEmptyUpdates(Shape({0, 2})); }

TEST_F(ScatterAddCpuTest, Float32Int32DuplicateIndicesCompileAndAccumulate) {
  ExpectDuplicateAccumulation<float, int32_t>({10, 20, 1, 2, 100, 200}, {2, 0, 2},
                                              {1, 2, 0, 0, 110, 220, 0, 0}, DTypeCode::kFloat32);
}

TEST_F(ScatterAddCpuTest, Float32Int64DuplicateIndicesCompileAndAccumulate) {
  ExpectDuplicateAccumulation<float, int64_t>({10, 20, 1, 2, 100, 200}, {2, 0, 2},
                                              {1, 2, 0, 0, 110, 220, 0, 0}, DTypeCode::kFloat32);
}

TEST_F(ScatterAddCpuTest, Float16BothIndexDtypesAccumulate) {
  const std::vector<float16_t> updates{float_to_float16(1), float_to_float16(2),
                                       float_to_float16(3), float_to_float16(4),
                                       float_to_float16(5), float_to_float16(6)};
  const std::vector<float16_t> expected{
      float_to_float16(3), float_to_float16(4), float_to_float16(0), float_to_float16(0),
      float_to_float16(6), float_to_float16(8), float_to_float16(0), float_to_float16(0)};
  ExpectDuplicateAccumulation<float16_t, int32_t>(updates, {2, 0, 2}, expected,
                                                  DTypeCode::kFloat16);
  ExpectDuplicateAccumulation<float16_t, int64_t>(updates, {2, 0, 2}, expected,
                                                  DTypeCode::kFloat16);
}

TEST_F(ScatterAddCpuTest, BFloat16BothIndexDtypesAccumulate) {
  const std::vector<bfloat16_t> updates{float_to_bfloat16(1), float_to_bfloat16(2),
                                        float_to_bfloat16(3), float_to_bfloat16(4),
                                        float_to_bfloat16(5), float_to_bfloat16(6)};
  const std::vector<bfloat16_t> expected{
      float_to_bfloat16(3), float_to_bfloat16(4), float_to_bfloat16(0), float_to_bfloat16(0),
      float_to_bfloat16(6), float_to_bfloat16(8), float_to_bfloat16(0), float_to_bfloat16(0)};
  ExpectDuplicateAccumulation<bfloat16_t, int32_t>(updates, {2, 0, 2}, expected,
                                                   DTypeCode::kBFloat16);
  ExpectDuplicateAccumulation<bfloat16_t, int64_t>(updates, {2, 0, 2}, expected,
                                                   DTypeCode::kBFloat16);
}

TEST_F(ScatterAddCpuTest, KernelRejectsOutOfRangeIndex) {
  const Result<frame::ops::KernelFn> found =
      frame::ops::KernelRegistry::instance().find("scatter_add", frame::kCpuBackendName);
  ASSERT_TRUE(found.is_ok()) << found.status().message();
  Tensor updates = MakeTensorWithShape<float>(Shape({2, 2}), {1, 2, 3, 4});
  Tensor indices = MakeTensor1D<int64_t>({0, 4});
  Tensor out = MakeTensorWithShape<float>(Shape({4, 2}), std::vector<float>(8, 0));
  std::vector<Tensor> inputs{updates, indices};
  std::vector<Tensor> outputs{out};
  const AttrMap attrs{{"output_shape", Shape({4, 2})}};
  frame::ops::KernelContext ctx{inputs, outputs, &attrs, frame::cpu_device(), nullptr};
  const Status status = found.value()(ctx);
  ASSERT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("indices[1]=4"), std::string_view::npos);
  EXPECT_NE(status.message().find("out of range for out's first dimension V=4"),
            std::string_view::npos);
}

Graph BuildScatterLossGraph() {
  Graph graph("scatter_add_loss");
  Value* updates = graph.add_graph_input(MakeType(DType::of<float>(), {3, 2})).value();
  Value* indices = graph.add_graph_input(MakeType(DType::of<int64_t>(), {3})).value();
  Value* weights = graph.add_graph_input(MakeType(DType::of<float>(), {4, 2})).value();
  const AttrMap attrs{{"output_shape", Shape({4, 2})}};
  Node* scatter =
      create_node_with_inferred_types(graph, "scatter_add", {updates, indices}, attrs).value();
  Node* weighted =
      create_node_with_inferred_types(graph, "mul", {scatter->output(0), weights}).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* loss =
      create_node_with_inferred_types(graph, "sum", {weighted->output(0)}, sum_attrs).value();
  EXPECT_TRUE(graph.mark_output(loss, 0).is_ok());
  return graph;
}

TEST_F(ScatterAddCpuTest,
       UpdatesGradientMatchesClosedFormAndCentralDifferenceIndicesGradientIsZero) {
  const Graph forward = BuildScatterLossGraph();
  Tensor updates = MakeTensorWithShape<float>(Shape({3, 2}), {1, 2, 3, 4, 5, 6});
  Tensor indices = MakeTensor1D<int64_t>({2, 0, 2});
  Tensor weights = MakeTensorWithShape<float>(Shape({4, 2}), {1, 2, 3, 4, 5, 6, 7, 8});
  std::vector<Tensor> inputs{updates, indices, weights};
  const std::vector<int32_t> updates_only{0};
  // BUILD-011 数值微分专款:fp32 中心差分按公共 helper 的放宽一档容差。
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, updates_only, 0, inputs, 1e-3));

  const std::vector<int32_t> wrt{0, 1};
  const Result<Graph> training = frame::compiler::build_backward_graph(forward, 0, wrt);
  ASSERT_TRUE(training.is_ok()) << training.status().message();
  const Result<std::shared_ptr<frame::hal::Executable>> executable = frame::runtime::compile(
      training.value(), frame::kCpuBackendName, frame::hal::CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  ASSERT_EQ(outputs.value().size(), 3U);
  const Tensor expected_updates = MakeTensorWithShape<float>(Shape({3, 2}), {5, 6, 1, 2, 5, 6});
  EXPECT_TRUE(tensor_all_close(outputs.value()[1], expected_updates,
                               default_tolerance(DTypeCode::kFloat32)));
  const Tensor expected_indices = MakeTensor1D<int64_t>({0, 0, 0});
  EXPECT_EQ(outputs.value()[2].dtype(), DType::of<int64_t>());
  EXPECT_EQ(outputs.value()[2].shape(), Shape({3}));
  EXPECT_TRUE(
      tensor_all_close(outputs.value()[2], expected_indices, default_tolerance(DTypeCode::kInt64)));
}

}  // namespace
