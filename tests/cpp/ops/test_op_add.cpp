// add 算子测试(schema + cpu kernel,src/ops/schemas/elementwise.cpp、
// src/backends/cpu/kernels/elementwise.cpp 已实化的行为):
//   1. OpRegistry::find("add") 的 schema 字段(inputs/outputs 数量)、traits
//      命中/未命中、shape_infer()/decomposition() 函数指针状态;
//   2. infer_add_shape 的合法路径 + 三类拒绝路径(shape 不一致/dtype 不一致/
//      输入数不为 2);
//   3. eager 数值路径(ARCH-011 第 3 类:单算子单元测试,tests/ 不在
//      Backend::launch 调用点白名单扫描范围,天然放行):经 BackendRegistry 取
//      cpu 后端真实 Allocator 构造输入/输出 Tensor,Backend::launch 执行
//      "add",数值断言统一用 tests/cpp/common/tolerance.h 的 tensor_all_close +
//      default_tolerance(BUILD-011)—— fp32/fp16/bf16 各一条(fp16/bf16 用
//      位级构造已知值,期望值独立手算,见各用例注释),另加 launch 未注册
//      算子名报错负例;
//   4. add_cpu_kernel 自身的两类防御性拒绝路径(dtype 不在 v0 支持的三档浮点
//      内 / lhs·rhs·out 三者 shape 不一致),经 KernelRegistry::find("add",
//      kCpuBackendName) 直接取 KernelFn 调用驱动(不经 Backend::launch 这层
//      薄壳,聚焦 kernel 自身的校验逻辑)。
//
// 共用设施(MakeType/eager fixture/kernel fixture 模板)见
// tests/cpp/ops/elementwise_op_test_helpers.h(与 test_op_mul.cpp、
// test_op_relu.cpp 共用,REUSE-002,不在本文件维护第二份复制)。全程复用
// cpu 后端真实 Allocator
// (经 BackendRegistry 取得,hal 已实化),不使用 FakeAllocator。本文件不新增
// 任何 op/kernel 注册,仅消费已由 src/ 静态注册好的 "add";launch 负例用的
// 未注册算子名以 "test_op_add_" 前缀,跨全体 tests/cpp/ops/ 测试文件保持进程级
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
using frame::ir::Graph;
using frame::ir::Value;
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

TEST(AddOpSchemaTest, RegisteredAndFindable) {
  const OpSchema* schema = OpRegistry::instance().find("add");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->name(), "add");
}

TEST(AddOpSchemaTest, HasTwoInputsAndOneOutput) {
  const OpSchema* schema = OpRegistry::instance().find("add");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 2u);
  EXPECT_EQ(schema->outputs().size(), 1u);
}

TEST(AddOpSchemaTest, TraitsMatchElementwiseFusableCommutativeOnly) {
  const OpSchema* schema = OpRegistry::instance().find("add");
  ASSERT_NE(schema, nullptr);
  EXPECT_TRUE(schema->has_trait(OpTrait::kElementwise));
  EXPECT_TRUE(schema->has_trait(OpTrait::kFusable));
  EXPECT_TRUE(schema->has_trait(OpTrait::kCommutative));
  EXPECT_FALSE(schema->has_trait(OpTrait::kHasSideEffect));
}

TEST(AddOpSchemaTest, HasShapeInferButNoDecomposition) {
  const OpSchema* schema = OpRegistry::instance().find("add");
  ASSERT_NE(schema, nullptr);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);  // M17:add_gradient 注册读回(ARCH-063)
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

TEST(AddShapeInferTest, SameShapeInputsProduceSameOutputShape) {
  const OpSchema* schema = OpRegistry::instance().find("add");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  NodeContext ctx;
  ctx.op = "add";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {2, 3})};

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok());
  ASSERT_EQ(result.value().size(), 1u);
  EXPECT_EQ(result.value()[0], Shape({2, 3}));
}

TEST(AddShapeInferTest, MismatchedShapeIsRejectedWithNoBroadcastingMessage) {
  const OpSchema* schema = OpRegistry::instance().find("add");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  NodeContext ctx;
  ctx.op = "add";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {3, 2})};

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("same shape"), std::string_view::npos);
  EXPECT_NE(result.status().message().find("no broadcasting"), std::string_view::npos);
}

TEST(AddShapeInferTest, MismatchedDtypeIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("add");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  NodeContext ctx;
  ctx.op = "add";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}),
                     MakeType(DType::of<std::int32_t>(), {2, 3})};

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("same dtype"), std::string_view::npos);
}

TEST(AddShapeInferTest, WrongInputCountIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("add");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  NodeContext ctx;
  ctx.op = "add";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};  // 只给 1 个,schema 要求 2 个

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("expects 2 inputs, got 1"), std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 3. eager 数值(ARCH-011 第 3 类:单算子单元测试)。经 BackendRegistry 取 cpu
//    后端真实 Allocator 构造输入/输出 Tensor,Backend::launch 执行 "add"。
// ---------------------------------------------------------------------------

class AddOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(AddOpEagerTest, Float32LaunchComputesElementwiseSum) {
  Tensor lhs = MakeTensor1D<float>({1.5f, -2.0f, 0.0f});
  Tensor rhs = MakeTensor1D<float>({2.25f, 4.5f, 3.0f});
  Tensor expected = MakeTensor1D<float>({3.75f, 2.5f, 3.0f});
  Tensor out = MakeTensor1D<float>({0.0f, 0.0f, 0.0f});

  std::vector<Tensor> inputs{lhs, rhs};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "add";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(AddOpEagerTest, Float16LaunchComputesElementwiseSumViaBitLevelValues) {
  // 位级构造 fp16 已知值(独立手算,未调用被测转换函数推导):
  //   1.5(0x3E00) + 2.25(0x4080) = 3.75(0x4380) —— 均在 fp16 10 位尾数精度内
  //   精确可表示,和值也精确可表示,无舍入误差;
  //   -1.0(0xBC00) + -0.5(0xB800) = -1.5(0xBE00) —— 覆盖负值路径。
  Tensor lhs = MakeTensor1D<float16_t>({float16_t{0x3E00u}, float16_t{0xBC00u}});
  Tensor rhs = MakeTensor1D<float16_t>({float16_t{0x4080u}, float16_t{0xB800u}});
  Tensor expected = MakeTensor1D<float16_t>({float16_t{0x4380u}, float16_t{0xBE00u}});
  Tensor out = MakeTensor1D<float16_t>({float16_t{0x0000u}, float16_t{0x0000u}});

  std::vector<Tensor> inputs{lhs, rhs};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "add";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat16)));
}

TEST_F(AddOpEagerTest, BFloat16LaunchComputesElementwiseSumViaBitLevelValues) {
  // 位级构造 bf16 已知值(独立手算,未调用被测转换函数推导):
  //   1.5(0x3FC0) + 2.25(0x4010) = 3.75(0x4070) —— bf16 7 位尾数足以精确表示
  //   1.875 的二进制小数部分(0.111),和值精确、无舍入误差;
  //   -1.0(0xBF80) + -0.5(0xBF00) = -1.5(0xBFC0) —— 覆盖负值路径。
  Tensor lhs = MakeTensor1D<bfloat16_t>({bfloat16_t{0x3FC0u}, bfloat16_t{0xBF80u}});
  Tensor rhs = MakeTensor1D<bfloat16_t>({bfloat16_t{0x4010u}, bfloat16_t{0xBF00u}});
  Tensor expected = MakeTensor1D<bfloat16_t>({bfloat16_t{0x4070u}, bfloat16_t{0xBFC0u}});
  Tensor out = MakeTensor1D<bfloat16_t>({bfloat16_t{0x0000u}, bfloat16_t{0x0000u}});

  std::vector<Tensor> inputs{lhs, rhs};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "add";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kBFloat16)));
}

TEST_F(AddOpEagerTest, Int32LaunchComputesElementwiseSumExactly) {
  // int32 直加(M22,批4 T3,决议点A):精确无容差(tolerance.h 整数档
  // rtol=atol=0),含大数值验证不经 float 桥接(1000000000+1000000000 超出
  // float32 24 位尾数精确整数表示上限 2^24,若误经 float 计算会失精;和值
  // 2000000000 仍在 int32 值域内,不溢出)。
  Tensor lhs = MakeTensor1D<std::int32_t>({1, -2, 1000000000});
  Tensor rhs = MakeTensor1D<std::int32_t>({2, 4, 1000000000});
  Tensor expected = MakeTensor1D<std::int32_t>({3, 2, 2000000000});
  Tensor out = MakeTensor1D<std::int32_t>({0, 0, 0});

  std::vector<Tensor> inputs{lhs, rhs};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "add";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kInt32)));
}

TEST_F(AddOpEagerTest, Int64LaunchComputesElementwiseSumExactly) {
  Tensor lhs = MakeTensor1D<std::int64_t>({1, -2, 9007199254740993LL});
  Tensor rhs = MakeTensor1D<std::int64_t>({2, 4, 1LL});
  Tensor expected = MakeTensor1D<std::int64_t>({3, 2, 9007199254740994LL});
  Tensor out = MakeTensor1D<std::int64_t>({0, 0, 0});

  std::vector<Tensor> inputs{lhs, rhs};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "add";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok()) << status.message();
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kInt64)));
}

TEST_F(AddOpEagerTest, LaunchUnregisteredOpNameReturnsError) {
  frame::hal::KernelInvocation invocation;
  invocation.op = "test_op_add_never_registered_xyz";
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kNotFound);
}

// ---------------------------------------------------------------------------
// 4. add_cpu_kernel 自身的防御性拒绝路径:经 KernelRegistry::find("add", cpu)
//    直接取 KernelFn 调用,不经 Backend::launch(聚焦 kernel 校验逻辑本身)。
// ---------------------------------------------------------------------------

struct AddOpNameTag {
  static constexpr std::string_view kOpName = "add";
};
using AddOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<AddOpNameTag>;

// int32/int64(M22,批4 T3,决议点A:整数索引最小接触面——add 是"梯度累加
// 安全网",反向引擎以 add 累加整数输入的形式零梯度,整数直加、精确无容差)
// 已扩入 add 的 dtype 白名单,不再是拒绝用例;int8 仍在白名单外(整数索引
// 扩容仅覆盖 int32/int64 二档,§1.1 决议点A表)。
TEST_F(AddOpKernelTest, RejectsUnsupportedDtypeInt8) {
  Tensor lhs = MakeTensor<std::int8_t>(Shape({2}));
  Tensor rhs = MakeTensor<std::int8_t>(Shape({2}));
  Tensor out = MakeTensor<std::int8_t>(Shape({2}));

  std::vector<Tensor> inputs{lhs, rhs};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("does not support dtype 'int8'"), std::string_view::npos);
}

TEST_F(AddOpKernelTest, RejectsShapeMismatchAmongLhsRhsOut) {
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

// ---------------------------------------------------------------------------
// 5. GradientFn 微图契约白盒校验(M17,ARCH-063:「同一 Value 可重复
//    mark_output(M2 既有能力,如 add 的两个输入梯度都是 gy 本身)」)。经
//    schema->gradient() 直接取函数指针调用——该函数定义于
//    src/ops/schemas/elementwise.cpp 的匿名命名空间内,无法按符号名链接,但
//    其地址已由 OpSchema 注册读回,可直接以 NodeContext 调用,本文件不新增
//    任何 op/kernel 注册。与
//    tests/cpp/compiler/test_autograd.cpp::AddGradientMatchesNumericForBothOperands
//    职责互补:后者经完整 build_backward_graph → verify → compile → run
//    端到端校验数值结果;本用例白盒校验微图结构本身——两个输出确系同一
//    Value(而非各自独立、恰好数值相等的两份 gy 拷贝)。
// ---------------------------------------------------------------------------

TEST(AddGradientMicrographTest, BothOutputsRepeatSameGyValue) {
  const OpSchema* schema = OpRegistry::instance().find("add");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "add";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {2, 3})};

  const frame::Result<Graph> micro = schema->gradient()(ctx);
  ASSERT_TRUE(micro.is_ok()) << micro.status().message();

  // 按位契约(ARCH-063):graph_inputs=[a, b, y, gy](2 输入 + 1 输出 + 1 输出
  // 梯度全部占位声明),图输出=[ga, gb](2 个前向输入各一份梯度)。
  ASSERT_EQ(micro.value().inputs().size(), 4u);
  ASSERT_EQ(micro.value().outputs().size(), 2u);

  const Value* gy = micro.value().inputs()[3];
  EXPECT_EQ(micro.value().outputs()[0], gy);
  EXPECT_EQ(micro.value().outputs()[1], gy);
}

}  // namespace
