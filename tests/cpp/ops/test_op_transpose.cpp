// transpose 算子测试(schema + cpu kernel + 解析梯度 ≡ 数值微分,
// src/ops/schemas/shape.cpp、src/backends/cpu/kernels/shape.cpp 已实化的
// 行为,M22 批4 T3,§1.4 决议点D)。transpose(x; perm):任意秩,perm 必须为
// [0,rank) 的排列;梯度 gx=transpose(gy, inverse_perm)(构图期求逆,自伴
// 闭包)。
//   1. schema 字段/三函数指针状态;
//   2. shape_infer 合法路径(rank-2)+ 负例(perm 长度/越界/重复/输入数错);
//   3. eager 数值:rank-2 手算已知值(fp32)、rank-3 高秩一例、bf16(位级
//      转换);
//   4. transpose_cpu_kernel 自身的防御性拒绝路径;
//   5. 图编译路径;
//   6. 解析梯度 ≡ 数值微分(loss=sum(transpose(x,perm)^2),经实际
//      构图+反向执行验证 inverse_perm 计算是否正确——若求逆有误,分析梯度会
//      落在错误位置而与数值梯度不符,详见文件内注释)。
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

TEST(TransposeOpSchemaTest, RegisteredWithOneInputOneOutputOneAttr) {
  const OpSchema* schema = OpRegistry::instance().find("transpose");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 1U);
  EXPECT_EQ(schema->outputs().size(), 1U);
  ASSERT_EQ(schema->attrs().size(), 1U);
  EXPECT_EQ(schema->attrs()[0].name, "perm");
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

TEST(TransposeShapeInferTest, Rank2SwapProducesPermutedShape) {
  const OpSchema* schema = OpRegistry::instance().find("transpose");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "transpose";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};
  const AttrMap attrs{{"perm", std::vector<int64_t>{1, 0}}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({3, 2}));
}

TEST(TransposeShapeInferTest, Rank3PermutationProducesPermutedShape) {
  const OpSchema* schema = OpRegistry::instance().find("transpose");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "transpose";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3, 4})};
  const AttrMap attrs{{"perm", std::vector<int64_t>{2, 0, 1}}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({4, 2, 3}));
}

TEST(TransposeShapeInferTest, WrongInputCountIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("transpose");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "transpose";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {2, 3})};
  const AttrMap attrs{{"perm", std::vector<int64_t>{1, 0}}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("expects 1 input, got 2"), std::string_view::npos);
}

TEST(TransposeShapeInferTest, PermLengthMismatchIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("transpose");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "transpose";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3, 4})};
  const AttrMap attrs{{"perm", std::vector<int64_t>{1, 0}}};  // 长度 2,期望 3
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("must have rank=3 element(s), got 2"),
            std::string_view::npos);
}

TEST(TransposeShapeInferTest, PermEntryOutOfRangeIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("transpose");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "transpose";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};
  const AttrMap attrs{{"perm", std::vector<int64_t>{0, 2}}};  // 2 越界(rank=2)
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("entry 2 is out of range for rank 2"),
            std::string_view::npos);
}

TEST(TransposeShapeInferTest, PermEntryDuplicatedIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("transpose");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "transpose";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};
  const AttrMap attrs{{"perm", std::vector<int64_t>{0, 0}}};
  ctx.attrs = &attrs;
  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("entry 0 is duplicated"), std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 3. eager 数值。
// ---------------------------------------------------------------------------

class TransposeOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(TransposeOpEagerTest, Rank2SwapKnownValues) {
  // 手算:x=[[1,2,3],[4,5,6]](2x3),perm=[1,0] -> out=[[1,4],[2,5],[3,6]](3x2)。
  Tensor x = MakeTensorWithShape<float>(Shape({2, 3}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  Tensor expected = MakeTensorWithShape<float>(Shape({3, 2}), {1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F});
  Tensor out = MakeTensorWithShape<float>(Shape({3, 2}), {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "transpose";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  const AttrMap attrs{{"perm", std::vector<int64_t>{1, 0}}};
  invocation.attrs = &attrs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(TransposeOpEagerTest, Rank3PermutationKnownValues) {
  // x shape [2,3,4](行优先 0..23),perm=[2,0,1] -> out shape [4,2,3];
  // out[k,i,j]=x[i,j,k](手算取若干位置校验,展平比对整张张量)。
  std::vector<float> x_values(24);
  for (int i = 0; i < 24; ++i) x_values[static_cast<size_t>(i)] = static_cast<float>(i);
  Tensor x = MakeTensorWithShape<float>(Shape({2, 3, 4}), x_values);

  std::vector<float> expected_values(24, 0.0F);
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 3; ++j) {
      for (int64_t k = 0; k < 4; ++k) {
        const int64_t x_linear = (i * 3 + j) * 4 + k;
        const int64_t out_linear = (k * 2 + i) * 3 + j;
        expected_values[static_cast<size_t>(out_linear)] = x_values[static_cast<size_t>(x_linear)];
      }
    }
  }
  Tensor expected = MakeTensorWithShape<float>(Shape({4, 2, 3}), expected_values);
  Tensor out = MakeTensorWithShape<float>(Shape({4, 2, 3}), std::vector<float>(24, 0.0F));

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "transpose";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  const AttrMap attrs{{"perm", std::vector<int64_t>{2, 0, 1}}};
  invocation.attrs = &attrs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(TransposeOpEagerTest, BFloat16Rank2SwapKnownValuesViaBitLevelConversion) {
  Tensor x = MakeTensorWithShape<bfloat16_t>(
      Shape({2, 2}), {float_to_bfloat16(1.0F), float_to_bfloat16(2.0F), float_to_bfloat16(3.0F),
                      float_to_bfloat16(4.0F)});
  Tensor expected = MakeTensorWithShape<bfloat16_t>(
      Shape({2, 2}), {float_to_bfloat16(1.0F), float_to_bfloat16(3.0F), float_to_bfloat16(2.0F),
                      float_to_bfloat16(4.0F)});
  Tensor out = MakeTensorWithShape<bfloat16_t>(
      Shape({2, 2}), {bfloat16_t{0}, bfloat16_t{0}, bfloat16_t{0}, bfloat16_t{0}});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "transpose";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  const AttrMap attrs{{"perm", std::vector<int64_t>{1, 0}}};
  invocation.attrs = &attrs;
  invocation.device = device_;
  const Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kBFloat16)));
}

// ---------------------------------------------------------------------------
// 4. transpose_cpu_kernel 自身的防御性拒绝路径。
// ---------------------------------------------------------------------------

struct TransposeOpNameTag {
  static constexpr std::string_view kOpName = "transpose";
};
using TransposeOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<TransposeOpNameTag>;

TEST_F(TransposeOpKernelTest, RejectsUnsupportedDtypeInt32) {
  Tensor x = MakeTensor<std::int32_t>(Shape({2, 3}));
  Tensor out = MakeTensor<std::int32_t>(Shape({3, 2}));

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  const AttrMap attrs{{"perm", std::vector<int64_t>{1, 0}}};
  ctx.attrs = &attrs;
  ctx.device = cpu_device();

  const Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("does not support dtype 'int32'"), std::string_view::npos);
}

TEST_F(TransposeOpKernelTest, RejectsOutShapeMismatch) {
  Tensor x = MakeTensor<float>(Shape({2, 3}));
  Tensor out = MakeTensor<float>(Shape({2, 3}));  // 期望 [3,2],故意给错

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  const AttrMap attrs{{"perm", std::vector<int64_t>{1, 0}}};
  ctx.attrs = &attrs;
  ctx.device = cpu_device();

  const Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("requires out shape to match the permuted result"),
            std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 5. 图编译路径。
// ---------------------------------------------------------------------------

class TransposeOpCompileTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(TransposeOpCompileTest, CompiledExecutionMatchesKnownValues) {
  Graph graph("transpose_only");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  const AttrMap attrs{{"perm", std::vector<int64_t>{1, 0}}};
  Node* transpose_node = create_node_with_inferred_types(graph, "transpose", {x}, attrs).value();
  ASSERT_TRUE(graph.mark_output(transpose_node, 0).is_ok());

  const Result<std::shared_ptr<frame::hal::Executable>> executable =
      frame::runtime::compile(graph, frame::kCpuBackendName, frame::hal::CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  Tensor x_tensor = MakeTensorWithShape<float>(Shape({2, 3}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  std::vector<Tensor> inputs{x_tensor};
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  Tensor expected = MakeTensorWithShape<float>(Shape({3, 2}), {1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F});
  EXPECT_TRUE(
      tensor_all_close(outputs.value()[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 6. 解析梯度 ≡ 数值微分。
// ---------------------------------------------------------------------------

class TransposeGradientTest : public frame::ops::testing::ElementwiseEagerTestBase {};

// loss=sum(transpose(x,perm)^2):数学上与 perm 无关(恒等于 sum(x^2)),但
// 分析梯度经"实际构图 + 反向执行"求得(若 transpose_gradient 内 inverse_perm
// 求逆有误,gx 会把值散布到错误位置,x 取互异值时与数值梯度立即失配),与
// "扰动实际前向输出"得到的数值梯度独立核对,并非平凡退化用例(参考
// test_op_transpose.cpp 头注释)。dims/perm 参数化:2-轴对换与 rank-3 循环
// 置换两用例共用本构图(REUSE-002)。
// 相邻同型 (dims, perm) 形参:dims=输入形状、perm=轴排列,固定契约序且调用
// 点均以字面量传入;误置换时 perm 的长度/值域/排列校验(shape_infer)会立即
// 拒绝构图,不会静默产出错误图(同 conv.cpp::nchw_index 先例论证)。
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
Graph BuildTransposeSquaredLossGraph(const std::vector<int64_t>& dims,
                                     const std::vector<int64_t>& perm) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  Graph graph("transpose_squared_loss");
  frame::ir::TensorType x_type = MakeType(DType::of<float>(), {});
  x_type.shape = Shape(dims);
  Value* x = graph.add_graph_input(x_type).value();
  const AttrMap attrs{{"perm", perm}};
  Node* transpose_node = create_node_with_inferred_types(graph, "transpose", {x}, attrs).value();
  Node* square_node =
      create_node_with_inferred_types(graph, "square", {transpose_node->output(0)}).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {square_node->output(0)}, sum_attrs).value();
  const Status mark_status = graph.mark_output(sum_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

constexpr double kTransposeCentralDifferenceH = 1e-2;

TEST_F(TransposeGradientTest, GradientMatchesNumericForX) {
  const Graph forward = BuildTransposeSquaredLossGraph({2, 3}, {1, 0});
  Tensor x = MakeTensorWithShape<float>(Shape({2, 3}), {1.0F, -2.0F, 3.0F, -4.0F, 5.0F, -6.0F});
  std::vector<Tensor> inputs{x};
  const std::vector<int32_t> wrt{0};
  EXPECT_TRUE(
      CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kTransposeCentralDifferenceH));
}

// 非自逆排列路径(code-reviewer 批4建议③):2-轴对换是自逆置换
// (inverse_perm==perm),无法暴露"误用 perm 本身充当逆排列"一类求逆缺陷;
// rank-3 循环置换 perm=[2,0,1] 的逆 = [1,2,0] ≠ perm,若求逆有误,互异 x 值
// 的分析梯度会落错位置,与数值梯度立即失配。
TEST_F(TransposeGradientTest, GradientMatchesNumericForNonInvolutivePerm) {
  const Graph forward = BuildTransposeSquaredLossGraph({2, 3, 4}, {2, 0, 1});
  std::vector<float> values(24);
  for (size_t i = 0; i < values.size(); ++i) {
    const float sign = (i % 2 == 0) ? 1.0F : -1.0F;
    values[i] = 0.25F * static_cast<float>(i + 1) * sign;
  }
  Tensor x = MakeTensorWithShape<float>(Shape({2, 3, 4}), values);
  std::vector<Tensor> inputs{x};
  const std::vector<int32_t> wrt{0};
  EXPECT_TRUE(
      CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kTransposeCentralDifferenceH));
}

}  // namespace
