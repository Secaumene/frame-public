// matmul_grad_{lhs,rhs}_internal 算子测试(M17,matmul 的梯度 internal 算子;
// src/ops/schemas/matmul.cpp、src/backends/cpu/kernels/matmul.cpp 已实化的
// 行为)。二者均不面向用户(_internal 后缀,PY-021 天然豁免),kernel 内转置
// 索引直算、不物化 transpose(见两 kernel 头注释)。
//   1. OpRegistry::find 的 schema 字段(2 输入 1 输出)、
//      shape_infer()/decomposition()/gradient() 三函数指针状态;
//   2. eager 数值路径(ARCH-011 第 3 类):对手算期望的已知小矩阵各一条;
//   3. kernel 自身的防御性拒绝路径(dtype 不支持 / rank≠2 / 收缩维不一致 /
//      out shape 不符),经 KernelRegistry::find 直接取 KernelFn 调用驱动。
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
#include <frame/ops/kernel_registry.h>
#include <frame/ops/op_registry.h>

#include "../common/tolerance.h"
#include "elementwise_op_test_helpers.h"

namespace {

using frame::cpu_device;
using frame::DType;
using frame::DTypeCode;
using frame::ErrorCode;
using frame::Shape;
using frame::Tensor;
using frame::ops::NodeContext;
using frame::ops::OpRegistry;
using frame::ops::OpSchema;
using frame::ops::testing::MakeType;
using frame::testing::default_tolerance;
using frame::testing::tensor_all_close;

// ---------------------------------------------------------------------------
// 1. schema 断言(两算子同构,合并覆盖)。
// ---------------------------------------------------------------------------

TEST(MatmulGradLhsInternalOpSchemaTest, RegisteredWithTwoInputsAndOneOutput) {
  const OpSchema* schema = OpRegistry::instance().find("matmul_grad_lhs_internal");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 2u);
  EXPECT_EQ(schema->outputs().size(), 1u);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);  // M26:二阶图变换闭包(ARCH-068)
}

TEST(MatmulGradRhsInternalOpSchemaTest, RegisteredWithTwoInputsAndOneOutput) {
  const OpSchema* schema = OpRegistry::instance().find("matmul_grad_rhs_internal");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 2u);
  EXPECT_EQ(schema->outputs().size(), 1u);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);  // M26:二阶图变换闭包(ARCH-068)
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

TEST(MatmulGradLhsInternalShapeInferTest, ProducesMShapeByK) {
  const OpSchema* schema = OpRegistry::instance().find("matmul_grad_lhs_internal");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "matmul_grad_lhs_internal";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 2}), MakeType(DType::of<float>(), {3, 2})};
  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok());
  EXPECT_EQ(result.value()[0], Shape({2, 3}));
}

TEST(MatmulGradRhsInternalShapeInferTest, ProducesKShapeByN) {
  const OpSchema* schema = OpRegistry::instance().find("matmul_grad_rhs_internal");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "matmul_grad_rhs_internal";
  ctx.input_types = {MakeType(DType::of<float>(), {3, 2}), MakeType(DType::of<float>(), {3, 2})};
  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok());
  EXPECT_EQ(result.value()[0], Shape({2, 2}));
}

// ---------------------------------------------------------------------------
// 3. eager 数值(ARCH-011 第 3 类)。
// ---------------------------------------------------------------------------

class MatmulGradInternalOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(MatmulGradInternalOpEagerTest, LhsInternalComputesGyTimesBTranspose) {
  // gy=[[1,2],[3,4]](2x2),b=[[1,0],[0,1],[1,1]](3x2,原 matmul 的 rhs);
  // ga=gy·bᵀ 独立手算:row0=[1,2,3],row1=[3,4,7],shape [2,3]。
  Tensor gy = MakeTensorWithShape<float>(Shape({2, 2}), {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor b = MakeTensorWithShape<float>(Shape({3, 2}), {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f});
  Tensor expected = MakeTensorWithShape<float>(Shape({2, 3}), {1.0f, 2.0f, 3.0f, 3.0f, 4.0f, 7.0f});
  Tensor out = MakeTensorWithShape<float>(Shape({2, 3}), {0, 0, 0, 0, 0, 0});

  std::vector<Tensor> inputs{gy, b};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "matmul_grad_lhs_internal";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(MatmulGradInternalOpEagerTest, RhsInternalComputesATransposeTimesGy) {
  // a=[[1,2],[3,4],[5,6]](3x2,原 matmul 的 lhs),gy=[[1,0],[0,1],[1,1]](3x2);
  // gb=aᵀ·gy 独立手算:row0=[6,8],row1=[8,10],shape [2,2]。
  Tensor a = MakeTensorWithShape<float>(Shape({3, 2}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor gy = MakeTensorWithShape<float>(Shape({3, 2}), {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f});
  Tensor expected = MakeTensorWithShape<float>(Shape({2, 2}), {6.0f, 8.0f, 8.0f, 10.0f});
  Tensor out = MakeTensorWithShape<float>(Shape({2, 2}), {0, 0, 0, 0});

  std::vector<Tensor> inputs{a, gy};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "matmul_grad_rhs_internal";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 4. kernel 自身的防御性拒绝路径。
// ---------------------------------------------------------------------------

struct MatmulGradLhsInternalOpNameTag {
  static constexpr std::string_view kOpName = "matmul_grad_lhs_internal";
};
using MatmulGradLhsInternalOpKernelTest =
    frame::ops::testing::ElementwiseKernelTestBase<MatmulGradLhsInternalOpNameTag>;

TEST_F(MatmulGradLhsInternalOpKernelTest, RejectsUnsupportedDtypeInt32) {
  Tensor gy = MakeTensor<std::int32_t>(Shape({2, 2}));
  Tensor b = MakeTensor<std::int32_t>(Shape({2, 2}));
  Tensor ga = MakeTensor<std::int32_t>(Shape({2, 2}));

  std::vector<Tensor> inputs{gy, b};
  std::vector<Tensor> outputs{ga};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("does not support dtype 'int32'"), std::string_view::npos);
}

TEST_F(MatmulGradLhsInternalOpKernelTest, RejectsContractionDimensionMismatch) {
  Tensor gy = MakeTensor<float>(Shape({2, 3}));  // n=3
  Tensor b = MakeTensor<float>(Shape({4, 5}));   // n=5,与 gy 的 n 不一致
  Tensor ga = MakeTensor<float>(Shape({2, 4}));

  std::vector<Tensor> inputs{gy, b};
  std::vector<Tensor> outputs{ga};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("contraction dimension mismatch"), std::string_view::npos);
}

TEST_F(MatmulGradLhsInternalOpKernelTest, RejectsOutShapeMismatch) {
  Tensor gy = MakeTensor<float>(Shape({2, 3}));
  Tensor b = MakeTensor<float>(Shape({4, 3}));
  Tensor ga = MakeTensor<float>(Shape({2, 5}));  // 期望 [2,4],故意给 [2,5]

  std::vector<Tensor> inputs{gy, b};
  std::vector<Tensor> outputs{ga};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("requires ga(out) shape to match [m, k]"),
            std::string_view::npos);
}

struct MatmulGradRhsInternalOpNameTag {
  static constexpr std::string_view kOpName = "matmul_grad_rhs_internal";
};
using MatmulGradRhsInternalOpKernelTest =
    frame::ops::testing::ElementwiseKernelTestBase<MatmulGradRhsInternalOpNameTag>;

TEST_F(MatmulGradRhsInternalOpKernelTest, RejectsRankMismatch) {
  Tensor a = MakeTensor<float>(Shape({3}));  // rank 1,应拒绝
  Tensor gy = MakeTensor<float>(Shape({3, 2}));
  Tensor gb = MakeTensor<float>(Shape({2, 2}));

  std::vector<Tensor> inputs{a, gy};
  std::vector<Tensor> outputs{gb};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("requires a to be rank-2, got rank 1"), std::string_view::npos);
}

TEST_F(MatmulGradRhsInternalOpKernelTest, RejectsOutShapeMismatch) {
  Tensor a = MakeTensor<float>(Shape({3, 2}));
  Tensor gy = MakeTensor<float>(Shape({3, 4}));
  Tensor gb = MakeTensor<float>(Shape({2, 5}));  // 期望 [2,4],故意给 [2,5]

  std::vector<Tensor> inputs{a, gy};
  std::vector<Tensor> outputs{gb};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("requires gb(out) shape to match [k, n]"),
            std::string_view::npos);
}

}  // namespace
