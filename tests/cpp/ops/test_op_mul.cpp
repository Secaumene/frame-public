// mul 算子测试(schema + cpu kernel,src/ops/schemas/elementwise.cpp、
// src/backends/cpu/kernels/elementwise.cpp 已实化的行为,与 add 同构 ——
// 二者共用 infer_binary_elementwise_shape/binary_elementwise_cpu_kernel 校验
// 骨架,仅运算体不同):
//   1. OpRegistry::find("mul") 的 schema 字段(inputs/outputs 数量)、traits
//      命中/未命中、shape_infer()/decomposition() 函数指针状态;
//   2. infer_mul_shape 的合法路径 + 三类拒绝路径(shape 不一致/dtype 不一致/
//      输入数不为 2),错误消息 op 名为 'mul';
//   3. eager 数值路径(ARCH-011 第 3 类:单算子单元测试,tests/ 不在
//      Backend::launch 调用点白名单扫描范围,天然放行):经 BackendRegistry 取
//      cpu 后端真实 Allocator 构造输入/输出 Tensor,Backend::launch 执行
//      "mul",数值断言统一用 tests/cpp/common/tolerance.h 的 tensor_all_close +
//      default_tolerance(BUILD-011)—— fp32/fp16/bf16 各一条(fp16/bf16 用
//      位级构造已知值,期望值独立手算,见各用例注释),另加 launch 未注册
//      算子名报错负例;
//   4. mul_cpu_kernel 自身的两类防御性拒绝路径(dtype 不在 v0 支持的三档浮点
//      内 / lhs·rhs·out 三者 shape 不一致),经 KernelRegistry::find("mul",
//      kCpuBackendName) 直接取 KernelFn 调用驱动(不经 Backend::launch 这层
//      薄壳,聚焦 kernel 自身的校验逻辑)。
//
// 共用设施(MakeType/eager fixture/kernel fixture 模板)见
// tests/cpp/ops/elementwise_op_test_helpers.h(与 test_op_add.cpp、
// test_op_relu.cpp 共用,REUSE-002,不在本文件维护第二份复制)。全程复用
// cpu 后端真实 Allocator
// (经 BackendRegistry 取得,hal 已实化),不使用 FakeAllocator。本文件不新增
// 任何 op/kernel 注册,仅消费已由 src/ 静态注册好的 "mul";launch 负例用的
// 未注册算子名以 "test_op_mul_" 前缀,跨全体 tests/cpp/ops/ 测试文件保持进程级
// 唯一(同 test_ops_stub.cpp 头注释纪律)。
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

using frame::bfloat16_t;
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
using frame::ops::OpTrait;
using frame::ops::testing::MakeType;
using frame::testing::default_tolerance;
using frame::testing::tensor_all_close;

// ---------------------------------------------------------------------------
// 1. schema 断言。
// ---------------------------------------------------------------------------

TEST(MulOpSchemaTest, RegisteredAndFindable) {
  const OpSchema* schema = OpRegistry::instance().find("mul");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->name(), "mul");
}

TEST(MulOpSchemaTest, HasTwoInputsAndOneOutput) {
  const OpSchema* schema = OpRegistry::instance().find("mul");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 2u);
  EXPECT_EQ(schema->outputs().size(), 1u);
}

TEST(MulOpSchemaTest, TraitsMatchElementwiseFusableCommutativeOnly) {
  const OpSchema* schema = OpRegistry::instance().find("mul");
  ASSERT_NE(schema, nullptr);
  EXPECT_TRUE(schema->has_trait(OpTrait::kElementwise));
  EXPECT_TRUE(schema->has_trait(OpTrait::kFusable));
  EXPECT_TRUE(schema->has_trait(OpTrait::kCommutative));
  EXPECT_FALSE(schema->has_trait(OpTrait::kHasSideEffect));
}

TEST(MulOpSchemaTest, HasShapeInferButNoDecomposition) {
  const OpSchema* schema = OpRegistry::instance().find("mul");
  ASSERT_NE(schema, nullptr);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);  // M17:mul_gradient 注册读回(ARCH-063)
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

TEST(MulShapeInferTest, SameShapeInputsProduceSameOutputShape) {
  const OpSchema* schema = OpRegistry::instance().find("mul");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  NodeContext ctx;
  ctx.op = "mul";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {2, 3})};

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok());
  ASSERT_EQ(result.value().size(), 1u);
  EXPECT_EQ(result.value()[0], Shape({2, 3}));
}

TEST(MulShapeInferTest, MismatchedShapeIsRejectedWithNoBroadcastingMessage) {
  const OpSchema* schema = OpRegistry::instance().find("mul");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  NodeContext ctx;
  ctx.op = "mul";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {3, 2})};

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("same shape"), std::string_view::npos);
  EXPECT_NE(result.status().message().find("no broadcasting"), std::string_view::npos);
  // op 名须为 'mul'(与 add 共用校验骨架,消息须能区分具体是哪个算子违例)。
  EXPECT_NE(result.status().message().find("op 'mul'"), std::string_view::npos);
}

TEST(MulShapeInferTest, MismatchedDtypeIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("mul");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  NodeContext ctx;
  ctx.op = "mul";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}),
                     MakeType(DType::of<std::int32_t>(), {2, 3})};

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("same dtype"), std::string_view::npos);
}

TEST(MulShapeInferTest, WrongInputCountIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("mul");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  NodeContext ctx;
  ctx.op = "mul";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};  // 只给 1 个,schema 要求 2 个

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("expects 2 inputs, got 1"), std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 3. eager 数值(ARCH-011 第 3 类:单算子单元测试)。经 BackendRegistry 取 cpu
//    后端真实 Allocator 构造输入/输出 Tensor,Backend::launch 执行 "mul"。
// ---------------------------------------------------------------------------

class MulOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(MulOpEagerTest, Float32LaunchComputesElementwiseProduct) {
  Tensor lhs = MakeTensor1D<float>({2.0f, -3.0f, 0.0f});
  Tensor rhs = MakeTensor1D<float>({4.0f, 5.0f, 7.0f});
  Tensor expected = MakeTensor1D<float>({8.0f, -15.0f, 0.0f});
  Tensor out = MakeTensor1D<float>({0.0f, 0.0f, 0.0f});

  std::vector<Tensor> inputs{lhs, rhs};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "mul";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(MulOpEagerTest, Float16LaunchComputesElementwiseProductViaBitLevelValues) {
  // 位级构造 fp16 已知值(独立手算 + Python struct 交叉验证,未调用被测转换
  // 函数推导):
  //   1.5(0x3E00) * 2.0(0x4000) = 3.0(0x4200) —— 均为 2 的幂次相关的精确值,
  //   积也精确可表示,无舍入误差;
  //   -2.0(0xC000) * 0.5(0x3800) = -1.0(0xBC00) —— 覆盖负值路径。
  Tensor lhs = MakeTensor1D<float16_t>({float16_t{0x3E00u}, float16_t{0xC000u}});
  Tensor rhs = MakeTensor1D<float16_t>({float16_t{0x4000u}, float16_t{0x3800u}});
  Tensor expected = MakeTensor1D<float16_t>({float16_t{0x4200u}, float16_t{0xBC00u}});
  Tensor out = MakeTensor1D<float16_t>({float16_t{0x0000u}, float16_t{0x0000u}});

  std::vector<Tensor> inputs{lhs, rhs};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "mul";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat16)));
}

TEST_F(MulOpEagerTest, BFloat16LaunchComputesElementwiseProductViaBitLevelValues) {
  // 位级构造 bf16 已知值(独立手算 + Python struct 交叉验证,未调用被测转换
  // 函数推导):
  //   1.5(0x3FC0) * 2.0(0x4000) = 3.0(0x4040) —— 积精确可表示,无舍入误差;
  //   -2.0(0xC000) * 0.5(0x3F00) = -1.0(0xBF80) —— 覆盖负值路径。
  Tensor lhs = MakeTensor1D<bfloat16_t>({bfloat16_t{0x3FC0u}, bfloat16_t{0xC000u}});
  Tensor rhs = MakeTensor1D<bfloat16_t>({bfloat16_t{0x4000u}, bfloat16_t{0x3F00u}});
  Tensor expected = MakeTensor1D<bfloat16_t>({bfloat16_t{0x4040u}, bfloat16_t{0xBF80u}});
  Tensor out = MakeTensor1D<bfloat16_t>({bfloat16_t{0x0000u}, bfloat16_t{0x0000u}});

  std::vector<Tensor> inputs{lhs, rhs};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "mul";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kBFloat16)));
}

TEST_F(MulOpEagerTest, LaunchUnregisteredOpNameReturnsError) {
  frame::hal::KernelInvocation invocation;
  invocation.op = "test_op_mul_never_registered_xyz";
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kNotFound);
}

// ---------------------------------------------------------------------------
// 4. mul_cpu_kernel 自身的防御性拒绝路径:经 KernelRegistry::find("mul", cpu)
//    直接取 KernelFn 调用,不经 Backend::launch(聚焦 kernel 校验逻辑本身)。
// ---------------------------------------------------------------------------

struct MulOpNameTag {
  static constexpr std::string_view kOpName = "mul";
};
using MulOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<MulOpNameTag>;

TEST_F(MulOpKernelTest, RejectsUnsupportedDtypeInt32) {
  Tensor lhs = MakeTensor<std::int32_t>(Shape({2}));
  Tensor rhs = MakeTensor<std::int32_t>(Shape({2}));
  Tensor out = MakeTensor<std::int32_t>(Shape({2}));

  std::vector<Tensor> inputs{lhs, rhs};
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

TEST_F(MulOpKernelTest, RejectsShapeMismatchAmongLhsRhsOut) {
  Tensor lhs = MakeTensor<float>(Shape({2, 3}));
  Tensor rhs = MakeTensor<float>(Shape({2, 3}));
  Tensor out = MakeTensor<float>(Shape({3, 2}));  // 与 lhs/rhs 不一致

  std::vector<Tensor> inputs{lhs, rhs};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("same shape"), std::string_view::npos);
}

}  // namespace
