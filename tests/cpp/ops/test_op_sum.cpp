// sum 算子测试(schema + cpu kernel,src/ops/schemas/reduction.cpp、
// src/backends/cpu/kernels/reduction.cpp 已实化的行为)。sum 是归约算子:有
// 跨元素依赖,不标任何 OpTrait(与 add/mul/relu 逐元素算子不同,见
// reduction.cpp 头注释)。
//   1. OpRegistry::find("sum") 的 schema 字段(1 输入 1 输出)、无任何 trait
//      命中、attrs 读回(axes: kInt64Array required=true;keepdims: kBool
//      required=false)、shape_infer()/decomposition() 函数指针状态;
//   2. infer_sum_shape 的合法路径(axes=[0] keepdims=false/true、axes=[]、
//      axes=[0,1]全维显式枚举)+ 四类拒绝路径(负值/越界/重复/缺 axes),
//      消息以 src/ops/schemas/reduction.cpp 原文为准;
//   3. eager 数值路径(ARCH-011 第 3 类:单算子单元测试,tests/ 不在
//      Backend::launch 调用点白名单扫描范围,天然放行):fp32 覆盖全归约/
//      单轴(两条)/单轴+keepdims 四种归约形态,fp16/bf16 各一条全归约(升
//      float 精度累加),外加 BUILD-011 大规模归约放宽一档的首个真实用例
//      (见下方该用例内的专门注释);
//   4. sum_cpu_kernel 自身的防御性拒绝路径(dtype 不在 v0 支持的三档浮点内 /
//      out shape 与按 axes/keepdims 推得的归约结果不符 / 缺 axes 的两种 kernel
//      侧消息变体),经 KernelRegistry::find("sum", kCpuBackendName) 直接取
//      KernelFn 调用驱动(不经 Backend::launch 这层薄壳,聚焦 kernel 自身的
//      校验逻辑)。
//
// 共用设施(MakeType/eager fixture/kernel fixture 模板,含按显式 Shape + 展平
// 值列表构造 Tensor 的 MakeTensorWithShape<T>)复用
// tests/cpp/ops/elementwise_op_test_helpers.h(与 test_op_add.cpp/
// test_op_mul.cpp/test_op_relu.cpp/test_op_matmul.cpp 共用,REUSE-002)——
// 该头文件内的两个 fixture(取 cpu 后端真实 Allocator、按值列表/shape 构造
// Tensor)本就与具体算子是逐元素、归约还是矩阵乘无关,sum 直接继承使用,不在
// 本文件重复定义任何一份;文件名虽带 "elementwise" 字样,但内容已是算子无关
// 的通用设施(该头文件头注释已说明)。全程复用 cpu 后端真实 Allocator(经
// BackendRegistry 取得,hal 已实化),不使用 FakeAllocator。本文件不新增任何
// op/kernel 注册,仅消费已由 src/ 静态注册好的 "sum"。
#include <cstdint>
#include <gtest/gtest.h>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/ir/attribute.h>
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
using frame::ir::AttrType;
using frame::ir::AttrValue;
using frame::ops::NodeContext;
using frame::ops::OpRegistry;
using frame::ops::OpSchema;
using frame::ops::OpTrait;
using frame::ops::testing::MakeType;
using frame::testing::default_tolerance;
using frame::testing::relaxed_tolerance;
using frame::testing::tensor_all_close;

// ---------------------------------------------------------------------------
// 1. schema 断言。
// ---------------------------------------------------------------------------

TEST(SumOpSchemaTest, RegisteredAndFindable) {
  const OpSchema* schema = OpRegistry::instance().find("sum");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->name(), "sum");
}

TEST(SumOpSchemaTest, HasOneInputAndOneOutput) {
  const OpSchema* schema = OpRegistry::instance().find("sum");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 1u);
  EXPECT_EQ(schema->outputs().size(), 1u);
}

TEST(SumOpSchemaTest, HasNoTraitsSet) {
  const OpSchema* schema = OpRegistry::instance().find("sum");
  ASSERT_NE(schema, nullptr);
  // 归约算子有跨元素依赖,不是逐元素(非 kElementwise);当前 schema 注册也
  // 未标注 kFusable/kHasSideEffect/kCommutative 中任何一项。
  EXPECT_FALSE(schema->has_trait(OpTrait::kElementwise));
  EXPECT_FALSE(schema->has_trait(OpTrait::kFusable));
  EXPECT_FALSE(schema->has_trait(OpTrait::kHasSideEffect));
  EXPECT_FALSE(schema->has_trait(OpTrait::kCommutative));
}

TEST(SumOpSchemaTest, AttrsMatchAxesRequiredAndKeepdimsOptional) {
  const OpSchema* schema = OpRegistry::instance().find("sum");
  ASSERT_NE(schema, nullptr);
  ASSERT_EQ(schema->attrs().size(), 2u);
  EXPECT_EQ(schema->attrs()[0].name, "axes");
  EXPECT_EQ(schema->attrs()[0].type, AttrType::kInt64Array);
  EXPECT_TRUE(schema->attrs()[0].required);
  EXPECT_EQ(schema->attrs()[1].name, "keepdims");
  EXPECT_EQ(schema->attrs()[1].type, AttrType::kBool);
  EXPECT_FALSE(schema->attrs()[1].required);
}

TEST(SumOpSchemaTest, HasShapeInferButNoDecomposition) {
  const OpSchema* schema = OpRegistry::instance().find("sum");
  ASSERT_NE(schema, nullptr);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);  // M17:sum_gradient 注册读回(ARCH-063)
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

TEST(SumShapeInferTest, Axis0KeepdimsFalseReducesToRankOneShape) {
  const OpSchema* schema = OpRegistry::instance().find("sum");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  const std::unordered_map<std::string, AttrValue> attrs{
      {"axes", std::vector<int64_t>{0}},
  };
  NodeContext ctx;
  ctx.op = "sum";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};
  ctx.attrs = &attrs;

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok());
  ASSERT_EQ(result.value().size(), 1u);
  EXPECT_EQ(result.value()[0], Shape({3}));
}

TEST(SumShapeInferTest, Axis0KeepdimsTrueReducesToRankTwoShapeWithSizeOneDim) {
  const OpSchema* schema = OpRegistry::instance().find("sum");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  const std::unordered_map<std::string, AttrValue> attrs{
      {"axes", std::vector<int64_t>{0}},
      {"keepdims", true},
  };
  NodeContext ctx;
  ctx.op = "sum";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};
  ctx.attrs = &attrs;

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok());
  ASSERT_EQ(result.value().size(), 1u);
  EXPECT_EQ(result.value()[0], Shape({1, 3}));
}

TEST(SumShapeInferTest, EmptyAxesArrayMeansFullReductionToRankZero) {
  const OpSchema* schema = OpRegistry::instance().find("sum");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  const std::unordered_map<std::string, AttrValue> attrs{
      {"axes", std::vector<int64_t>{}},
  };
  NodeContext ctx;
  ctx.op = "sum";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};
  ctx.attrs = &attrs;

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok());
  ASSERT_EQ(result.value().size(), 1u);
  EXPECT_EQ(result.value()[0], Shape());
}

TEST(SumShapeInferTest, ExplicitAllAxesAlsoReducesToRankZero) {
  const OpSchema* schema = OpRegistry::instance().find("sum");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  const std::unordered_map<std::string, AttrValue> attrs{
      {"axes", std::vector<int64_t>{0, 1}},
  };
  NodeContext ctx;
  ctx.op = "sum";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};
  ctx.attrs = &attrs;

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok());
  ASSERT_EQ(result.value().size(), 1u);
  EXPECT_EQ(result.value()[0], Shape());
}

TEST(SumShapeInferTest, NegativeAxisIsRejectedWithNoNormalizationMessage) {
  const OpSchema* schema = OpRegistry::instance().find("sum");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  const std::unordered_map<std::string, AttrValue> attrs{
      {"axes", std::vector<int64_t>{-1}},
  };
  NodeContext ctx;
  ctx.op = "sum";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};
  ctx.attrs = &attrs;

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("does not normalize negative indices"),
            std::string_view::npos);
}

TEST(SumShapeInferTest, OutOfRangeAxisIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("sum");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  const std::unordered_map<std::string, AttrValue> attrs{
      {"axes", std::vector<int64_t>{2}},  // rank=2,合法范围 [0, 2)
  };
  NodeContext ctx;
  ctx.op = "sum";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};
  ctx.attrs = &attrs;

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("is out of range for rank 2"), std::string_view::npos);
}

TEST(SumShapeInferTest, DuplicateAxisIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("sum");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  const std::unordered_map<std::string, AttrValue> attrs{
      {"axes", std::vector<int64_t>{0, 0}},
  };
  NodeContext ctx;
  ctx.op = "sum";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};
  ctx.attrs = &attrs;

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("is duplicated"), std::string_view::npos);
}

TEST(SumShapeInferTest, MissingAxesAttributeIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("sum");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  NodeContext ctx;
  ctx.op = "sum";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};
  // ctx.attrs 保持默认 nullptr(无属性),缺失必填的 'axes'。

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_EQ(result.status().message(),
            "op 'sum' is missing required attribute 'axes' (int64 array)");
}

// ---------------------------------------------------------------------------
// 3. eager 数值(ARCH-011 第 3 类:单算子单元测试)。经 BackendRegistry 取 cpu
//    后端真实 Allocator 构造输入/输出 Tensor,Backend::launch 执行 "sum"。
// ---------------------------------------------------------------------------

class SumOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(SumOpEagerTest, Float32FullReductionSumsAllElements) {
  // x = [[1,2,3],[4,5,6]],全归约(axes=[])= 1+2+3+4+5+6 = 21,输出 rank-0
  // (Shape{},numel==1)。
  Tensor x = MakeTensorWithShape<float>(Shape({2, 3}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor expected = MakeTensorWithShape<float>(Shape(), {21.0f});
  Tensor out = MakeTensorWithShape<float>(Shape(), {0.0f});

  const std::unordered_map<std::string, AttrValue> attrs{
      {"axes", std::vector<int64_t>{}},
  };
  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "sum";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(SumOpEagerTest, Float32ReduceAxisZeroProducesColumnSums) {
  // axis=0(尺寸 2 的维)归约:列和 = [1+4, 2+5, 3+6] = [5, 7, 9],输出 shape [3]。
  Tensor x = MakeTensorWithShape<float>(Shape({2, 3}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor expected = MakeTensorWithShape<float>(Shape({3}), {5.0f, 7.0f, 9.0f});
  Tensor out = MakeTensorWithShape<float>(Shape({3}), {0.0f, 0.0f, 0.0f});

  const std::unordered_map<std::string, AttrValue> attrs{
      {"axes", std::vector<int64_t>{0}},
  };
  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "sum";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(SumOpEagerTest, Float32ReduceAxisOneProducesRowSums) {
  // axis=1(尺寸 3 的维)归约:行和 = [1+2+3, 4+5+6] = [6, 15],输出 shape [2]。
  Tensor x = MakeTensorWithShape<float>(Shape({2, 3}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor expected = MakeTensorWithShape<float>(Shape({2}), {6.0f, 15.0f});
  Tensor out = MakeTensorWithShape<float>(Shape({2}), {0.0f, 0.0f});

  const std::unordered_map<std::string, AttrValue> attrs{
      {"axes", std::vector<int64_t>{1}},
  };
  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "sum";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(SumOpEagerTest, Float32ReduceAxisOneKeepdimsTrueProducesRankTwoShape) {
  // 同上 axis=1 归约,但 keepdims=true:输出 shape [2,1],展平值仍是 [6, 15]。
  Tensor x = MakeTensorWithShape<float>(Shape({2, 3}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor expected = MakeTensorWithShape<float>(Shape({2, 1}), {6.0f, 15.0f});
  Tensor out = MakeTensorWithShape<float>(Shape({2, 1}), {0.0f, 0.0f});

  const std::unordered_map<std::string, AttrValue> attrs{
      {"axes", std::vector<int64_t>{1}},
      {"keepdims", true},
  };
  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "sum";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(SumOpEagerTest, Float16FullReductionAccumulatesInFloatPrecision) {
  // 位级构造 fp16 已知值(独立手算 + Python struct 交叉验证):
  //   0.5(0x3800) + 1.5(0x3E00) + 2.5(0x4100) + 3.5(0x4300) = 8.0(0x4800),
  //   累加以 float 精度进行(sum_cpu_kernel::to_accum 逐元素升 float),四个
  //   加数均为 2 的负/正整数次幂组合,和值精确可表示,无舍入误差。
  Tensor x = MakeTensorWithShape<float16_t>(
      Shape({4}), {float16_t{0x3800u}, float16_t{0x3E00u}, float16_t{0x4100u}, float16_t{0x4300u}});
  Tensor expected = MakeTensorWithShape<float16_t>(Shape(), {float16_t{0x4800u}});
  Tensor out = MakeTensorWithShape<float16_t>(Shape(), {float16_t{0x0000u}});

  const std::unordered_map<std::string, AttrValue> attrs{
      {"axes", std::vector<int64_t>{}},
  };
  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "sum";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat16)));
}

TEST_F(SumOpEagerTest, BFloat16FullReductionAccumulatesInFloatPrecision) {
  // 位级构造 bf16 已知值(独立手算 + Python struct 交叉验证,与上一条 fp16
  // 用例同一组十进制值,位模式不同):
  //   0.5(0x3F00) + 1.5(0x3FC0) + 2.5(0x4020) + 3.5(0x4060) = 8.0(0x4100),
  //   和值精确可表示,无舍入误差。
  Tensor x = MakeTensorWithShape<bfloat16_t>(
      Shape({4}),
      {bfloat16_t{0x3F00u}, bfloat16_t{0x3FC0u}, bfloat16_t{0x4020u}, bfloat16_t{0x4060u}});
  Tensor expected = MakeTensorWithShape<bfloat16_t>(Shape(), {bfloat16_t{0x4100u}});
  Tensor out = MakeTensorWithShape<bfloat16_t>(Shape(), {bfloat16_t{0x0000u}});

  const std::unordered_map<std::string, AttrValue> attrs{
      {"axes", std::vector<int64_t>{}},
  };
  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "sum";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kBFloat16)));
}

TEST_F(SumOpEagerTest, Float32LargeReductionAtBuildEleven2Pow20AccumulationsUsesRelaxedTolerance) {
  // BUILD-011 大规模归约放宽一档条款(docs/standards/build-and-test.md:
  // "大规模归约(单个输出元素的累加次数 >= 2^20 的 reduction/matmul 类用例)
  // 允许放宽一档")的首个真实用例:单输出元素(全归约,rank-0)的累加次数恰为
  // 2^20 = 1048576,触发该条款,故本用例改用 relaxed_tolerance(而非
  // default_tolerance)。
  //
  // 输入分布:value[i] = 0.1F * (i%7 + 1),即 {0.1,0.2,...,0.7} 循环取值。
  // 0.1F 本身是 float32 不可精确表示的小数(二进制无限循环),且该分布不像
  // 早期版本的 i%7(纯整数、每步部分和均 < 2^24,float32 累加零舍入)那样是
  // 退化输入——这里累加从 0 单调递增到约 4.19e5(远超 2^24=16777216 的
  // float32 尾数精度边界之前那一段无影响,但随部分和增长,越往后每步加法的
  // 有效精度越低),真实触发可观测的舍入误差累积。
  //
  // 实测结论(替换早期版本"不可避免产生舍入误差"这一推测性表述——已用本
  // 用例本身实测验证,而非停留在理论推断):把下方断言暂改为
  // default_tolerance(DTypeCode::kFloat32) 重跑 ctest,SumOpEagerTest 的这条
  // 用例确凿失败,gtest 报告的实测结果为
  //   actual=418342(sum_cpu_kernel 的 float32 顺序累加结果)
  //   expected=419429.8125(本用例下方以 double 精度独立顺序累加的参考值)
  //   |actual-expected|≈1087.8(相对误差≈0.259%)
  // 而 default_tolerance(fp32 档 {rtol=1e-5,atol=1e-6})在 |expected|≈4.19e5
  // 处允许的误差上限仅 ≈4.19,实测偏差约为该上限的 259 倍,牢固证明放宽确有
  // 必要(而非"放宽了但其实不测试default_tolerance本可通过"的空放宽)。确认
  // 失败后已把断言切回 relaxed_tolerance(fp32 放宽后取 fp16 档
  // {rtol=1e-2,atol=1e-3},此时误差上限 ≈4194.3,实测偏差 1087.8 落在其内,
  // 断言通过)。
  constexpr int64_t kLargeReductionNumel = int64_t{1} << 20;  // 2^20,BUILD-011 放宽边界值
  std::vector<float> values(static_cast<size_t>(kLargeReductionNumel));
  double expected_sum_double = 0.0;
  for (int64_t i = 0; i < kLargeReductionNumel; ++i) {
    const float value = 0.1F * static_cast<float>(i % 7 + 1);
    values[static_cast<size_t>(i)] = value;
    expected_sum_double += static_cast<double>(value);  // 独立的 double 精度参考累加
  }

  Tensor x = MakeTensorWithShape<float>(Shape({kLargeReductionNumel}), values);
  Tensor expected = MakeTensorWithShape<float>(Shape(), {static_cast<float>(expected_sum_double)});
  Tensor out = MakeTensorWithShape<float>(Shape(), {0.0f});

  const std::unordered_map<std::string, AttrValue> attrs{
      {"axes", std::vector<int64_t>{}},
  };
  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "sum";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, relaxed_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 4. sum_cpu_kernel 自身的防御性拒绝路径:经 KernelRegistry::find("sum", cpu)
//    直接取 KernelFn 调用,不经 Backend::launch(聚焦 kernel 校验逻辑本身)。
// ---------------------------------------------------------------------------

struct SumOpNameTag {
  static constexpr std::string_view kOpName = "sum";
};
using SumOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<SumOpNameTag>;

TEST_F(SumOpKernelTest, RejectsUnsupportedDtypeInt32) {
  // dtype 校验先于 axes 读取(见 sum_cpu_kernel 顺序),故本用例无需提供
  // attrs 即可触发 dtype 拒绝路径。
  Tensor x = MakeTensor<std::int32_t>(Shape({4}));
  Tensor out = MakeTensor<std::int32_t>(Shape());

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

TEST_F(SumOpKernelTest, RejectsOutShapeMismatchWithReductionResult) {
  // x shape [2,3],axes=[](全归约)按定义应产出 rank-0 输出;这里故意把 out
  // 分配成 shape [1](非 rank-0),触发"out shape 与归约结果不符"拒绝路径。
  Tensor x = MakeTensor<float>(Shape({2, 3}));
  Tensor out = MakeTensor<float>(Shape({1}));

  const std::unordered_map<std::string, AttrValue> attrs{
      {"axes", std::vector<int64_t>{}},
  };
  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.attrs = &attrs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("requires out shape to match the reduction result"),
            std::string_view::npos);
}

TEST_F(SumOpKernelTest, RejectsMissingAxesWhenAttrsPointerIsNull) {
  // kernel 侧消息变体之一:ctx.attrs 整体为 nullptr(无属性表),消息含
  // "no attrs provided" 后缀,与 schema 侧的通用缺失消息不同。
  Tensor x = MakeTensor<float>(Shape({4}));
  Tensor out = MakeTensor<float>(Shape());

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();
  // ctx.attrs 保持默认 nullptr。

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("no attrs provided"), std::string_view::npos);
}

TEST_F(SumOpKernelTest, RejectsMissingAxesWhenAxesKeyAbsent) {
  // kernel 侧消息变体之二:ctx.attrs 非空但不含 'axes' 键,消息不带
  // "no attrs provided" 后缀,仍报缺失必填属性。
  Tensor x = MakeTensor<float>(Shape({4}));
  Tensor out = MakeTensor<float>(Shape());

  const std::unordered_map<std::string, AttrValue> attrs{};  // 非空指针,但无 'axes' 键
  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.attrs = &attrs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  const std::string_view message = status.message();
  EXPECT_NE(message.find("is missing required attribute 'axes'"), std::string_view::npos);
  EXPECT_EQ(message.find("no attrs provided"), std::string_view::npos);
}

}  // namespace
