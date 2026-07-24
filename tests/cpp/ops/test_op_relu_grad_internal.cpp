// relu_grad_internal 算子测试(M17,relu 的梯度 internal 算子;
// src/ops/schemas/elementwise.cpp、src/backends/cpu/kernels/elementwise.cpp
// 已实化的行为)。relu_grad_internal(x, gy) = x>0 处透传 gy,余 0;不面向用户
// (_internal 后缀,PY-021 天然豁免),shape/kernel 校验骨架复用二元逐元素
// 共用实现(REUSE-002,同 add/mul/relu 先例,见 src/ops/schemas/
// elementwise.cpp::infer_relu_grad_internal_shape、src/backends/cpu/kernels/
// elementwise.cpp::relu_grad_internal_cpu_kernel 头注释),故本文件的 schema/
// kernel 拒绝路径覆盖点与 test_op_add.cpp/test_op_relu.cpp 同构:
//   1. OpRegistry::find("relu_grad_internal") 的 schema 字段(2 输入 1 输出)、
//      shape_infer()/decomposition()/gradient() 三函数指针状态；M26 已为该
//      internal 算子注册可再次变换的 GradientFn(ARCH-068)；
//   2. eager 数值路径(ARCH-011 第 3 类:单算子单元测试):fp32 覆盖 x 的正/
//      负/±0.0 四种情形(对手算期望 gy 或 0),fp16 一条已知位级值;
//   3. kernel 自身的防御性拒绝路径(dtype 不支持 / x·gy 二者 shape 不一致),
//      经 KernelRegistry::find 直接取 KernelFn 调用驱动。
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
using frame::float16_t;
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

TEST(ReluGradInternalOpSchemaTest, RegisteredWithTwoInputsAndOneOutput) {
  const OpSchema* schema = OpRegistry::instance().find("relu_grad_internal");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->name(), "relu_grad_internal");
  EXPECT_EQ(schema->inputs().size(), 2u);
  EXPECT_EQ(schema->outputs().size(), 1u);
}

TEST(ReluGradInternalOpSchemaTest, HasShapeInferAndHigherOrderGradient) {
  const OpSchema* schema = OpRegistry::instance().find("relu_grad_internal");
  ASSERT_NE(schema, nullptr);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  // M26:internal 算子已注册二阶图变换闭包(ARCH-068)。
  EXPECT_NE(schema->gradient(), nullptr);
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

TEST(ReluGradInternalShapeInferTest, OutputShapeMatchesXAndGy) {
  const OpSchema* schema = OpRegistry::instance().find("relu_grad_internal");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  NodeContext ctx;
  ctx.op = "relu_grad_internal";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {2, 3})};

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok());
  ASSERT_EQ(result.value().size(), 1u);
  EXPECT_EQ(result.value()[0], Shape({2, 3}));
}

// ---------------------------------------------------------------------------
// 3. eager 数值(ARCH-011 第 3 类:单算子单元测试)。
// ---------------------------------------------------------------------------

class ReluGradInternalOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(ReluGradInternalOpEagerTest, Float32PassesGyWhereXPositiveElseZero) {
  // x = [3.5, -2.5, 0.0, -0.0]:严格 x>0 才透传 gy,0.0/-0.0 均判为"非正"清零
  // (与 relu 本身的 kernel 头注释"该 IEEE-754 边界值不敏感"一致精神)。
  Tensor x = MakeTensor1D<float>({3.5f, -2.5f, 0.0f, -0.0f});
  Tensor gy = MakeTensor1D<float>({1.0f, 2.0f, 3.0f, 4.0f});
  Tensor expected = MakeTensor1D<float>({1.0f, 0.0f, 0.0f, 0.0f});
  Tensor out = MakeTensor1D<float>({0.0f, 0.0f, 0.0f, 0.0f});

  std::vector<Tensor> inputs{x, gy};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "relu_grad_internal";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(ReluGradInternalOpEagerTest, Float16PassesGyWhereXPositiveElseZero) {
  // 位级构造 fp16 已知值(与 test_op_relu.cpp 复用同一组已交叉验证过的位
  // 模式):1.5(0x3E00,x>0 透传 gy)、-2.0(0xC000,x<=0 清零)。
  Tensor x = MakeTensor1D<float16_t>({float16_t{0x3E00u}, float16_t{0xC000u}});
  Tensor gy = MakeTensor1D<float16_t>({float16_t{0x3E00u}, float16_t{0x3E00u}});  // 1.5, 1.5
  Tensor expected = MakeTensor1D<float16_t>({float16_t{0x3E00u}, float16_t{0x0000u}});
  Tensor out = MakeTensor1D<float16_t>({float16_t{0x0000u}, float16_t{0x0000u}});

  std::vector<Tensor> inputs{x, gy};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "relu_grad_internal";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat16)));
}

// ---------------------------------------------------------------------------
// 4. relu_grad_internal_cpu_kernel 自身的防御性拒绝路径。
// ---------------------------------------------------------------------------

struct ReluGradInternalOpNameTag {
  static constexpr std::string_view kOpName = "relu_grad_internal";
};
using ReluGradInternalOpKernelTest =
    frame::ops::testing::ElementwiseKernelTestBase<ReluGradInternalOpNameTag>;

TEST_F(ReluGradInternalOpKernelTest, RejectsUnsupportedDtypeInt32) {
  Tensor x = MakeTensor<std::int32_t>(Shape({2}));
  Tensor gy = MakeTensor<std::int32_t>(Shape({2}));
  Tensor out = MakeTensor<std::int32_t>(Shape({2}));

  std::vector<Tensor> inputs{x, gy};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("does not support dtype 'int32'"), std::string_view::npos);
}

TEST_F(ReluGradInternalOpKernelTest, RejectsShapeMismatchBetweenXAndGy) {
  Tensor x = MakeTensor<float>(Shape({2, 3}));
  Tensor gy = MakeTensor<float>(Shape({3, 2}));  // 与 x 不一致
  Tensor out = MakeTensor<float>(Shape({2, 3}));

  std::vector<Tensor> inputs{x, gy};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("requires lhs/rhs/out of the same shape"),
            std::string_view::npos);
}

}  // namespace
