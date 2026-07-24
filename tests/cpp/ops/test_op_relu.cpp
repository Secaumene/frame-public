// relu 算子测试(schema + cpu kernel,src/ops/schemas/elementwise.cpp、
// src/backends/cpu/kernels/elementwise.cpp 已实化的行为)。relu 是一元逐元素
// 算子,与 add/mul 的二元校验骨架并列(infer_unary_elementwise_shape/
// unary_elementwise_cpu_kernel,见两文件头注释:输入个数不同、错误消息模板
// 不同,未强行与二元版合并):
//   1. OpRegistry::find("relu") 的 schema 字段(inputs/outputs 数量)、traits
//      命中/未命中(含 kCommutative 语义仅适用于二元可交换运算,relu 不应
//      标注)、shape_infer()/decomposition() 函数指针状态;
//   2. infer_relu_shape 的合法路径(恒等 shape)+ 输入数不为 1 的拒绝路径;
//   3. eager 数值路径(ARCH-011 第 3 类:单算子单元测试,tests/ 不在
//      Backend::launch 调用点白名单扫描范围,天然放行):经 BackendRegistry 取
//      cpu 后端真实 Allocator 构造输入/输出 Tensor,Backend::launch 执行
//      "relu"。fp32 一条覆盖正值/负值/+0.0/-0.0/NaN/±inf 七元素:容差工具
//      对 NaN/inf 一律判不等(tests/cpp/common/tolerance.h 头注释),故拆成
//      两个张量 —— 常规值(正值/负值/+0.0/-0.0)四元素走 tensor_all_close +
//      default_tolerance(BUILD-011),特殊值(NaN/+inf/-inf)三元素单独逐元素用
//      std::isnan/std::isinf 显式断言(relu(x)=std::max(x,0.f) 的 NaN 传播/
//      ±inf 语义见 src/backends/cpu/kernels/elementwise.cpp::relu_cpu_kernel
//      头注释,本文件用例据此独立推导期望值,不直接复用被测实现的分支判断);
//      fp16/bf16 各一条常规值(位级构造已知值,与 test_op_add.cpp/
//      test_op_mul.cpp 复用同一组已交叉验证过的位模式:1.5/-2.0/0.0);
//   4. relu_cpu_kernel 自身的两类防御性拒绝路径(dtype 不在 v0 支持的三档浮点
//      内 / x·out 二者 shape 不一致),经 KernelRegistry::find("relu",
//      kCpuBackendName) 直接取 KernelFn 调用驱动(不经 Backend::launch 这层
//      薄壳,聚焦 kernel 自身的校验逻辑)。
//
// 共用设施(MakeType/eager fixture/kernel fixture 模板)见
// tests/cpp/ops/elementwise_op_test_helpers.h(与 test_op_add.cpp、
// test_op_mul.cpp 共用,REUSE-002,不在本文件维护第二份复制;两个共用 fixture
// 本就与算子是一元还是二元无关,详见该头文件头注释)。全程复用 cpu 后端真实
// Allocator(经 BackendRegistry 取得,hal 已实化),不使用 FakeAllocator。本
// 文件不新增任何 op/kernel 注册,仅消费已由 src/ 静态注册好的 "relu"。
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
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

TEST(ReluOpSchemaTest, RegisteredAndFindable) {
  const OpSchema* schema = OpRegistry::instance().find("relu");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->name(), "relu");
}

TEST(ReluOpSchemaTest, HasOneInputAndOneOutput) {
  const OpSchema* schema = OpRegistry::instance().find("relu");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 1u);
  EXPECT_EQ(schema->outputs().size(), 1u);
}

TEST(ReluOpSchemaTest, TraitsMatchElementwiseFusableOnlyNotCommutative) {
  const OpSchema* schema = OpRegistry::instance().find("relu");
  ASSERT_NE(schema, nullptr);
  EXPECT_TRUE(schema->has_trait(OpTrait::kElementwise));
  EXPECT_TRUE(schema->has_trait(OpTrait::kFusable));
  // relu 是一元算子:kCommutative 语义仅适用于二元可交换运算(op_schema.h
  // OpTrait 定义),不应标注。
  EXPECT_FALSE(schema->has_trait(OpTrait::kCommutative));
  EXPECT_FALSE(schema->has_trait(OpTrait::kHasSideEffect));
}

TEST(ReluOpSchemaTest, HasShapeInferButNoDecomposition) {
  const OpSchema* schema = OpRegistry::instance().find("relu");
  ASSERT_NE(schema, nullptr);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);  // M17:relu_gradient 注册读回(ARCH-063)
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

TEST(ReluShapeInferTest, OutputShapeIsIdenticalToInputShape) {
  const OpSchema* schema = OpRegistry::instance().find("relu");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  NodeContext ctx;
  ctx.op = "relu";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok());
  ASSERT_EQ(result.value().size(), 1u);
  EXPECT_EQ(result.value()[0], Shape({2, 3}));
}

TEST(ReluShapeInferTest, WrongInputCountIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("relu");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  NodeContext ctx;
  ctx.op = "relu";
  // relu 是一元算子,schema 要求恰 1 个输入;这里给 2 个触发拒绝路径。
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {2, 3})};

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_EQ(result.status().message(), "op 'relu' expects 1 input, got 2");
}

// ---------------------------------------------------------------------------
// 3. eager 数值(ARCH-011 第 3 类:单算子单元测试)。经 BackendRegistry 取 cpu
//    后端真实 Allocator 构造输入/输出 Tensor,Backend::launch 执行 "relu"。
// ---------------------------------------------------------------------------

class ReluOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(ReluOpEagerTest, Float32LaunchComputesReluAcrossRegularAndSpecialValues) {
  // 常规值(正值/负值/+0.0/-0.0)四元素:全部有限,可安全走 tensor_all_close。
  // relu(x)=max(x,0):正值原样保留、负值清零、±0.0 两种输入均产出量级为 0 的
  // 输出(C++ 中 0.0f == -0.0f 恒真,tensor_all_close 内部按此做浮点判等,无需
  // 关心具体符号位,呼应 relu_cpu_kernel 头注释"产出 +0.0 或 -0.0 均视为合规")。
  {
    Tensor x = MakeTensor1D<float>({3.5f, -2.5f, 0.0f, -0.0f});
    Tensor expected = MakeTensor1D<float>({3.5f, 0.0f, 0.0f, 0.0f});
    Tensor out = MakeTensor1D<float>({0.0f, 0.0f, 0.0f, 0.0f});

    std::vector<Tensor> inputs{x};
    std::vector<Tensor> outputs{out};
    frame::hal::KernelInvocation invocation;
    invocation.op = "relu";
    invocation.inputs = inputs;
    invocation.outputs = outputs;
    invocation.device = device_;

    const frame::Status status = backend_->launch(invocation, nullptr);
    ASSERT_TRUE(status.is_ok());
    EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
  }

  // 特殊值(NaN/+inf/-inf)三元素:容差工具对 NaN/inf 一律判不等
  // (tolerance.h 头注释),不能进 tensor_all_close 的期望,单独逐元素显式断言。
  // 期望值独立推导(std::max(a,b) 语义 = a<b 时取 b 否则取 a):
  //   relu(NaN)  = std::max(NaN, 0.f)   → NaN<0.f 恒假 → 取 a=NaN(传播);
  //   relu(+inf) = std::max(+inf, 0.f)  → +inf<0.f 假  → 取 a=+inf;
  //   relu(-inf) = std::max(-inf, 0.f)  → -inf<0.f 真  → 取 b=0.f(清零)。
  {
    const float nan_value = std::numeric_limits<float>::quiet_NaN();
    const float pos_inf = std::numeric_limits<float>::infinity();
    const float neg_inf = -std::numeric_limits<float>::infinity();

    Tensor x = MakeTensor1D<float>({nan_value, pos_inf, neg_inf});
    Tensor out = MakeTensor1D<float>({0.0f, 0.0f, 0.0f});

    std::vector<Tensor> inputs{x};
    std::vector<Tensor> outputs{out};
    frame::hal::KernelInvocation invocation;
    invocation.op = "relu";
    invocation.inputs = inputs;
    invocation.outputs = outputs;
    invocation.device = device_;

    const frame::Status status = backend_->launch(invocation, nullptr);
    ASSERT_TRUE(status.is_ok());

    const float* data = outputs[0].data<float>();
    EXPECT_TRUE(std::isnan(data[0]));                    // relu(NaN) = NaN
    EXPECT_TRUE(std::isinf(data[1]) && data[1] > 0.0f);  // relu(+inf) = +inf
    EXPECT_EQ(data[2], 0.0f);                            // relu(-inf) = 0.0(有限)
  }
}

TEST_F(ReluOpEagerTest, Float16LaunchComputesReluViaBitLevelValues) {
  // 位级构造 fp16 已知值(与 test_op_add.cpp/test_op_mul.cpp 复用同一组已
  // 交叉验证过的位模式):1.5(0x3E00,正值原样保留)、-2.0(0xC000,负值清零
  // 为 0x0000)。
  Tensor x = MakeTensor1D<float16_t>({float16_t{0x3E00u}, float16_t{0xC000u}});
  Tensor expected = MakeTensor1D<float16_t>({float16_t{0x3E00u}, float16_t{0x0000u}});
  Tensor out = MakeTensor1D<float16_t>({float16_t{0x0000u}, float16_t{0x0000u}});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "relu";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat16)));
}

TEST_F(ReluOpEagerTest, BFloat16LaunchComputesReluViaBitLevelValues) {
  // 位级构造 bf16 已知值(与 test_op_add.cpp/test_op_mul.cpp 复用同一组已
  // 交叉验证过的位模式):1.5(0x3FC0,正值原样保留)、-2.0(0xC000,负值清零
  // 为 0x0000)。
  Tensor x = MakeTensor1D<bfloat16_t>({bfloat16_t{0x3FC0u}, bfloat16_t{0xC000u}});
  Tensor expected = MakeTensor1D<bfloat16_t>({bfloat16_t{0x3FC0u}, bfloat16_t{0x0000u}});
  Tensor out = MakeTensor1D<bfloat16_t>({bfloat16_t{0x0000u}, bfloat16_t{0x0000u}});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "relu";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kBFloat16)));
}

// ---------------------------------------------------------------------------
// 4. relu_cpu_kernel 自身的防御性拒绝路径:经 KernelRegistry::find("relu", cpu)
//    直接取 KernelFn 调用,不经 Backend::launch(聚焦 kernel 校验逻辑本身)。
// ---------------------------------------------------------------------------

struct ReluOpNameTag {
  static constexpr std::string_view kOpName = "relu";
};
using ReluOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<ReluOpNameTag>;

TEST_F(ReluOpKernelTest, RejectsUnsupportedDtypeInt32) {
  Tensor x = MakeTensor<std::int32_t>(Shape({2}));
  Tensor out = MakeTensor<std::int32_t>(Shape({2}));

  std::vector<Tensor> inputs{x};
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

TEST_F(ReluOpKernelTest, RejectsShapeMismatchBetweenXAndOut) {
  Tensor x = MakeTensor<float>(Shape({2, 3}));
  Tensor out = MakeTensor<float>(Shape({3, 2}));  // 与 x 不一致

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("requires x/out of the same shape"), std::string_view::npos);
}

}  // namespace
