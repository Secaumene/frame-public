// matmul 算子测试(schema + cpu kernel,src/ops/schemas/matmul.cpp、
// src/backends/cpu/kernels/matmul.cpp 已实化的行为)。matmul 是 v0
// 2D-only(design-reviewer 决议,m5-design-brief 决议点 2)的矩阵乘:非逐
// 元素、不可交换(AB≠BA),不标任何 OpTrait。
//   1. OpRegistry::find("matmul") 的 schema 字段(2 输入 1 输出)、四个
//      OpTrait 全部未命中、shape_infer()/decomposition() 函数指针状态;
//   2. infer_matmul_shape 的合法路径(非方阵)+ 五类拒绝路径(lhs rank≠2/
//      rhs rank≠2/收缩维不一致/dtype 不一致/输入数≠2),消息以
//      src/ops/schemas/matmul.cpp 原文为准;
//   3. eager 数值路径(ARCH-011 第 3 类:单算子单元测试,tests/ 不在
//      Backend::launch 调用点白名单扫描范围,天然放行):fp32 已知 2x2 矩阵
//      乘(教科书经典例)、fp32 非方阵(独立手算)、单位矩阵性质(I×A=A)、
//      fp16/bf16 各一条已知小矩阵(位级构造,期望值独立算出并经 Python
//      struct 交叉验证位模式);
//   4. matmul_cpu_kernel 自身的防御性拒绝路径(dtype 不在 v0 支持的三档
//      浮点内 / 收缩维不一致 / out shape 与 [m,n] 不符),经
//      KernelRegistry::find("matmul", kCpuBackendName) 直接取 KernelFn 调用
//      驱动(不经 Backend::launch 这层薄壳,聚焦 kernel 自身的校验逻辑)。
//
// 共用设施(MakeType/eager fixture/kernel fixture 模板,含按显式 Shape + 展平
// 值列表构造 Tensor 的 MakeTensorWithShape<T>)复用
// tests/cpp/ops/elementwise_op_test_helpers.h(与 test_op_add.cpp/
// test_op_mul.cpp/test_op_relu.cpp/test_op_sum.cpp 共用,REUSE-002)—— matmul
// 的输入/输出均是 2D 矩阵,直接继承基类已提供的 MakeTensorWithShape<T>,不在
// 本文件重复定义;文件名虽带 "elementwise" 字样,但内容已是算子无关的通用
// 设施(该头文件头注释已说明)。全程复用 cpu 后端真实 Allocator(经
// BackendRegistry 取得,hal 已实化),不使用 FakeAllocator。本文件不新增任何
// op/kernel 注册,仅消费已由 src/ 静态注册好的 "matmul"。
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

TEST(MatmulOpSchemaTest, RegisteredAndFindable) {
  const OpSchema* schema = OpRegistry::instance().find("matmul");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->name(), "matmul");
}

TEST(MatmulOpSchemaTest, HasTwoInputsAndOneOutput) {
  const OpSchema* schema = OpRegistry::instance().find("matmul");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 2u);
  EXPECT_EQ(schema->outputs().size(), 1u);
}

TEST(MatmulOpSchemaTest, HasNoTraitsSet) {
  const OpSchema* schema = OpRegistry::instance().find("matmul");
  ASSERT_NE(schema, nullptr);
  // matmul 非逐元素、不可交换(AB≠BA),四个 OpTrait 全部未标注。
  EXPECT_FALSE(schema->has_trait(OpTrait::kElementwise));
  EXPECT_FALSE(schema->has_trait(OpTrait::kFusable));
  EXPECT_FALSE(schema->has_trait(OpTrait::kHasSideEffect));
  EXPECT_FALSE(schema->has_trait(OpTrait::kCommutative));
}

TEST(MatmulOpSchemaTest, HasShapeInferButNoDecomposition) {
  const OpSchema* schema = OpRegistry::instance().find("matmul");
  ASSERT_NE(schema, nullptr);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);  // M17:matmul_gradient 注册读回(ARCH-063)
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

TEST(MatmulShapeInferTest, NonSquareMatricesProduceExpectedOutputShape) {
  const OpSchema* schema = OpRegistry::instance().find("matmul");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  NodeContext ctx;
  ctx.op = "matmul";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {3, 4})};

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok());
  ASSERT_EQ(result.value().size(), 1u);
  EXPECT_EQ(result.value()[0], Shape({2, 4}));
}

TEST(MatmulShapeInferTest, RankOneLhsIsRejectedWithActualRankInMessage) {
  const OpSchema* schema = OpRegistry::instance().find("matmul");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  NodeContext ctx;
  ctx.op = "matmul";
  ctx.input_types = {MakeType(DType::of<float>(), {3}), MakeType(DType::of<float>(), {3, 4})};

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires lhs to be rank-2, got rank 1"),
            std::string_view::npos);
}

TEST(MatmulShapeInferTest, RankThreeRhsIsRejectedWithActualRankInMessage) {
  const OpSchema* schema = OpRegistry::instance().find("matmul");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  NodeContext ctx;
  ctx.op = "matmul";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {3, 4, 5})};

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("requires rhs to be rank-2, got rank 3"),
            std::string_view::npos);
}

TEST(MatmulShapeInferTest, ContractionDimensionMismatchIsRejectedWithBothKValues) {
  const OpSchema* schema = OpRegistry::instance().find("matmul");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  NodeContext ctx;
  ctx.op = "matmul";
  // lhs [2,3] 的 k=3,rhs [4,5] 的 k2=4,二者不一致。
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {4, 5})};

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  const std::string_view message = result.status().message();
  EXPECT_NE(message.find("contraction dimension mismatch"), std::string_view::npos);
  EXPECT_NE(message.find("k=3"), std::string_view::npos);
  EXPECT_NE(message.find("k=4"), std::string_view::npos);
}

TEST(MatmulShapeInferTest, DtypeMismatchIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("matmul");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  NodeContext ctx;
  ctx.op = "matmul";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}),
                     MakeType(DType::of<std::int32_t>(), {3, 4})};

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("same dtype"), std::string_view::npos);
}

TEST(MatmulShapeInferTest, WrongInputCountIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("matmul");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  NodeContext ctx;
  ctx.op = "matmul";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};  // 只给 1 个,schema 要求 2 个

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("expects 2 inputs, got 1"), std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 3. eager 数值(ARCH-011 第 3 类:单算子单元测试)。经 BackendRegistry 取 cpu
//    后端真实 Allocator 构造输入/输出 Tensor,Backend::launch 执行 "matmul"。
// ---------------------------------------------------------------------------

class MatmulOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(MatmulOpEagerTest, Float32KnownTwoByTwoMatrixProduct) {
  // 教科书经典例:[[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
  // ([0][0]=1*5+2*7=19,[0][1]=1*6+2*8=22,[1][0]=3*5+4*7=43,[1][1]=3*6+4*8=50)。
  Tensor lhs = MakeTensorWithShape<float>(Shape({2, 2}), {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor rhs = MakeTensorWithShape<float>(Shape({2, 2}), {5.0f, 6.0f, 7.0f, 8.0f});
  Tensor expected = MakeTensorWithShape<float>(Shape({2, 2}), {19.0f, 22.0f, 43.0f, 50.0f});
  Tensor out = MakeTensorWithShape<float>(Shape({2, 2}), {0.0f, 0.0f, 0.0f, 0.0f});

  std::vector<Tensor> inputs{lhs, rhs};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "matmul";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(MatmulOpEagerTest, Float32NonSquareMatricesProduceIndependentlyComputedProduct) {
  // lhs [2,3] x rhs [3,2] -> out [2,2];独立手算(不依赖被测 kernel 本身):
  //   [0][0]=1*7+2*9+3*11=7+18+33=58   [0][1]=1*8+2*10+3*12=8+20+36=64
  //   [1][0]=4*7+5*9+6*11=28+45+66=139 [1][1]=4*8+5*10+6*12=32+50+72=154
  Tensor lhs = MakeTensorWithShape<float>(Shape({2, 3}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor rhs = MakeTensorWithShape<float>(Shape({3, 2}), {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
  Tensor expected = MakeTensorWithShape<float>(Shape({2, 2}), {58.0f, 64.0f, 139.0f, 154.0f});
  Tensor out = MakeTensorWithShape<float>(Shape({2, 2}), {0.0f, 0.0f, 0.0f, 0.0f});

  std::vector<Tensor> inputs{lhs, rhs};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "matmul";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(MatmulOpEagerTest, IdentityMatrixTimesAnyMatrixEqualsThatMatrix) {
  // 单位矩阵性质:I x A = A。
  Tensor identity = MakeTensorWithShape<float>(Shape({2, 2}), {1.0f, 0.0f, 0.0f, 1.0f});
  Tensor a = MakeTensorWithShape<float>(Shape({2, 2}), {2.0f, 3.0f, 4.0f, 5.0f});
  Tensor expected = MakeTensorWithShape<float>(Shape({2, 2}), {2.0f, 3.0f, 4.0f, 5.0f});
  Tensor out = MakeTensorWithShape<float>(Shape({2, 2}), {0.0f, 0.0f, 0.0f, 0.0f});

  std::vector<Tensor> inputs{identity, a};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "matmul";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(MatmulOpEagerTest, Float16KnownTwoByTwoMatrixProductViaBitLevelValues) {
  // 位级构造 fp16 已知值(与 fp32 用例同一组小整数,均 <2048、fp16 尾数内精确
  // 可表示,位模式独立用 Python struct 交叉验证):
  //   [[1(0x3C00),2(0x4000)],[3(0x4200),4(0x4400)]] x
  //   [[5(0x4500),6(0x4600)],[7(0x4700),8(0x4800)]] =
  //   [[19(0x4CC0),22(0x4D80)],[43(0x5160),50(0x5240)]]
  // 内积以 float32 精度累加(matmul_compute),四个输出元素的内积和均为精确
  // 整数,无舍入误差。
  Tensor lhs = MakeTensorWithShape<float16_t>(
      Shape({2, 2}),
      {float16_t{0x3C00u}, float16_t{0x4000u}, float16_t{0x4200u}, float16_t{0x4400u}});
  Tensor rhs = MakeTensorWithShape<float16_t>(
      Shape({2, 2}),
      {float16_t{0x4500u}, float16_t{0x4600u}, float16_t{0x4700u}, float16_t{0x4800u}});
  Tensor expected = MakeTensorWithShape<float16_t>(
      Shape({2, 2}),
      {float16_t{0x4CC0u}, float16_t{0x4D80u}, float16_t{0x5160u}, float16_t{0x5240u}});
  Tensor out = MakeTensorWithShape<float16_t>(
      Shape({2, 2}),
      {float16_t{0x0000u}, float16_t{0x0000u}, float16_t{0x0000u}, float16_t{0x0000u}});

  std::vector<Tensor> inputs{lhs, rhs};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "matmul";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat16)));
}

TEST_F(MatmulOpEagerTest, BFloat16KnownTwoByTwoMatrixProductViaBitLevelValues) {
  // 位级构造 bf16 已知值(与上一条 fp16 用例同一组十进制值,位模式不同,均
  // 经 Python struct 交叉验证):
  //   [[1(0x3F80),2(0x4000)],[3(0x4040),4(0x4080)]] x
  //   [[5(0x40A0),6(0x40C0)],[7(0x40E0),8(0x4100)]] =
  //   [[19(0x4198),22(0x41B0)],[43(0x422C),50(0x4248)]]
  // 全部输出整数均 <256,在 bf16 7 位尾数精度内精确可表示,无舍入误差。
  Tensor lhs = MakeTensorWithShape<bfloat16_t>(
      Shape({2, 2}),
      {bfloat16_t{0x3F80u}, bfloat16_t{0x4000u}, bfloat16_t{0x4040u}, bfloat16_t{0x4080u}});
  Tensor rhs = MakeTensorWithShape<bfloat16_t>(
      Shape({2, 2}),
      {bfloat16_t{0x40A0u}, bfloat16_t{0x40C0u}, bfloat16_t{0x40E0u}, bfloat16_t{0x4100u}});
  Tensor expected = MakeTensorWithShape<bfloat16_t>(
      Shape({2, 2}),
      {bfloat16_t{0x4198u}, bfloat16_t{0x41B0u}, bfloat16_t{0x422Cu}, bfloat16_t{0x4248u}});
  Tensor out = MakeTensorWithShape<bfloat16_t>(
      Shape({2, 2}),
      {bfloat16_t{0x0000u}, bfloat16_t{0x0000u}, bfloat16_t{0x0000u}, bfloat16_t{0x0000u}});

  std::vector<Tensor> inputs{lhs, rhs};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "matmul";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kBFloat16)));
}

// ---------------------------------------------------------------------------
// 4. matmul_cpu_kernel 自身的防御性拒绝路径:经
//    KernelRegistry::find("matmul", cpu) 直接取 KernelFn 调用,不经
//    Backend::launch(聚焦 kernel 校验逻辑本身)。
// ---------------------------------------------------------------------------

struct MatmulOpNameTag {
  static constexpr std::string_view kOpName = "matmul";
};
using MatmulOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<MatmulOpNameTag>;

TEST_F(MatmulOpKernelTest, RejectsUnsupportedDtypeInt32) {
  Tensor lhs = MakeTensor<std::int32_t>(Shape({2, 2}));
  Tensor rhs = MakeTensor<std::int32_t>(Shape({2, 2}));
  Tensor out = MakeTensor<std::int32_t>(Shape({2, 2}));

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

TEST_F(MatmulOpKernelTest, RejectsContractionDimensionMismatch) {
  // lhs [2,3] 的 k=3,rhs [4,5] 的 k2=4,二者不一致。
  Tensor lhs = MakeTensor<float>(Shape({2, 3}));
  Tensor rhs = MakeTensor<float>(Shape({4, 5}));
  Tensor out = MakeTensor<float>(Shape({2, 5}));

  std::vector<Tensor> inputs{lhs, rhs};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("contraction dimension mismatch"), std::string_view::npos);
}

TEST_F(MatmulOpKernelTest, RejectsOutShapeMismatch) {
  // lhs [2,3] x rhs [3,4] 按定义应产出 [2,4];这里故意把 out 分配成 [2,5],
  // 触发"out shape 与 [m,n] 不符"拒绝路径。
  Tensor lhs = MakeTensor<float>(Shape({2, 3}));
  Tensor rhs = MakeTensor<float>(Shape({3, 4}));
  Tensor out = MakeTensor<float>(Shape({2, 5}));

  std::vector<Tensor> inputs{lhs, rhs};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("requires out shape to match [m, n]"), std::string_view::npos);
}

}  // namespace
