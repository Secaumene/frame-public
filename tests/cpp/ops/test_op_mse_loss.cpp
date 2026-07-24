// mse_loss + mse_loss_grad_internal 算子测试(M17,ADR-0008/autograd.md 第4
// 章;src/ops/schemas/loss.cpp、src/backends/cpu/kernels/loss.cpp 已实化的
// 行为)。mse_loss 是面向用户算子(ARCH-064,独立注册、不以 sub/mean 组合
// 表达);mse_loss_grad_internal 是其梯度专用 internal 算子(不面向用户,
// PY-021 天然豁免)。
//   1. OpRegistry::find 的 schema 字段、shape_infer()/decomposition()/
//      gradient() 函数指针状态；mse_loss 与 M26 补齐的 internal 算子均带
//      可再次变换的 GradientFn(ARCH-068)；
//   2. eager 数值路径(ARCH-011 第 3 类):对手算期望;
//   3. kernel 自身的防御性拒绝路径。
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
// 1. schema 断言。
// ---------------------------------------------------------------------------

TEST(MseLossOpSchemaTest, RegisteredWithTwoInputsAndOneOutput) {
  const OpSchema* schema = OpRegistry::instance().find("mse_loss");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->name(), "mse_loss");
  EXPECT_EQ(schema->inputs().size(), 2u);
  EXPECT_EQ(schema->outputs().size(), 1u);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);  // M17:mse_loss_gradient 注册读回(ARCH-063)
}

TEST(MseLossGradInternalOpSchemaTest, RegisteredWithThreeInputsAndOneOutput) {
  const OpSchema* schema = OpRegistry::instance().find("mse_loss_grad_internal");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 3u);
  EXPECT_EQ(schema->outputs().size(), 1u);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);  // M26:二阶图变换闭包(ARCH-068)
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

TEST(MseLossShapeInferTest, OutputIsRankZeroScalar) {
  const OpSchema* schema = OpRegistry::instance().find("mse_loss");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "mse_loss";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 2}), MakeType(DType::of<float>(), {2, 2})};
  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok());
  EXPECT_EQ(result.value()[0], Shape());
}

TEST(MseLossGradInternalShapeInferTest, OutputMatchesPredShape) {
  const OpSchema* schema = OpRegistry::instance().find("mse_loss_grad_internal");
  ASSERT_NE(schema, nullptr);
  NodeContext ctx;
  ctx.op = "mse_loss_grad_internal";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 2}), MakeType(DType::of<float>(), {2, 2}),
                     MakeType(DType::of<float>(), {})};
  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok());
  EXPECT_EQ(result.value()[0], Shape({2, 2}));
}

// ---------------------------------------------------------------------------
// 3. eager 数值(ARCH-011 第 3 类)。
// ---------------------------------------------------------------------------

class MseLossOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(MseLossOpEagerTest, ComputesMeanSquaredError) {
  // pred=[1,2,3,4],target=[1,1,1,1];diff=[0,1,2,3],squared=[0,1,4,9],
  // sum=14,mean=14/4=3.5。
  Tensor pred = MakeTensor1D<float>({1.0f, 2.0f, 3.0f, 4.0f});
  Tensor target = MakeTensor1D<float>({1.0f, 1.0f, 1.0f, 1.0f});
  Tensor expected = MakeTensorWithShape<float>(Shape(), {3.5f});
  Tensor out = MakeTensorWithShape<float>(Shape(), {0.0f});

  std::vector<Tensor> inputs{pred, target};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "mse_loss";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(MseLossOpEagerTest, GradInternalComputesTwoTimesDiffOverNTimesGy) {
  // 同上 pred/target,gy=2.0(标量);coefficient=2*gy/N=2*2/4=1.0;
  // gpred=coefficient*(pred-target)=[0,1,2,3]。
  Tensor pred = MakeTensor1D<float>({1.0f, 2.0f, 3.0f, 4.0f});
  Tensor target = MakeTensor1D<float>({1.0f, 1.0f, 1.0f, 1.0f});
  Tensor gy = MakeTensorWithShape<float>(Shape(), {2.0f});
  Tensor expected = MakeTensor1D<float>({0.0f, 1.0f, 2.0f, 3.0f});
  Tensor out = MakeTensor1D<float>({0.0f, 0.0f, 0.0f, 0.0f});

  std::vector<Tensor> inputs{pred, target, gy};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "mse_loss_grad_internal";
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

struct MseLossOpNameTag {
  static constexpr std::string_view kOpName = "mse_loss";
};
using MseLossOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<MseLossOpNameTag>;

TEST_F(MseLossOpKernelTest, RejectsUnsupportedDtypeInt32) {
  Tensor pred = MakeTensor<std::int32_t>(Shape({2}));
  Tensor target = MakeTensor<std::int32_t>(Shape({2}));
  Tensor out = MakeTensor<std::int32_t>(Shape());

  std::vector<Tensor> inputs{pred, target};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("does not support dtype 'int32'"), std::string_view::npos);
}

TEST_F(MseLossOpKernelTest, RejectsOutShapeNotRankZero) {
  Tensor pred = MakeTensor<float>(Shape({2}));
  Tensor target = MakeTensor<float>(Shape({2}));
  Tensor out = MakeTensor<float>(Shape({1}));  // 非 rank-0

  std::vector<Tensor> inputs{pred, target};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("requires out shape to be rank-0 (scalar)"),
            std::string_view::npos);
}

struct MseLossGradInternalOpNameTag {
  static constexpr std::string_view kOpName = "mse_loss_grad_internal";
};
using MseLossGradInternalOpKernelTest =
    frame::ops::testing::ElementwiseKernelTestBase<MseLossGradInternalOpNameTag>;

TEST_F(MseLossGradInternalOpKernelTest, RejectsGyNotScalar) {
  Tensor pred = MakeTensor<float>(Shape({2}));
  Tensor target = MakeTensor<float>(Shape({2}));
  Tensor gy = MakeTensor<float>(Shape({2}));  // 非标量
  Tensor gpred = MakeTensor<float>(Shape({2}));

  std::vector<Tensor> inputs{pred, target, gy};
  std::vector<Tensor> outputs{gpred};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("requires gy to be scalar"), std::string_view::npos);
}

TEST_F(MseLossGradInternalOpKernelTest, RejectsPredTargetShapeMismatch) {
  Tensor pred = MakeTensor<float>(Shape({2}));
  Tensor target = MakeTensor<float>(Shape({3}));  // 与 pred 不一致
  Tensor gy = MakeTensor<float>(Shape());
  Tensor gpred = MakeTensor<float>(Shape({2}));

  std::vector<Tensor> inputs{pred, target, gy};
  std::vector<Tensor> outputs{gpred};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("requires pred and target of the same shape"),
            std::string_view::npos);
}

}  // namespace
