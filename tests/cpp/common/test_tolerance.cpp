// tolerance 工具自身单测(tests/cpp/common/tolerance.h 的唯一单测文件)。覆盖:
// default_tolerance 三档默认表数值精确断言 + 整型/bool 精确比较档 + 表外浮点
// (kFloat64)FRAME_CHECK 死亡测试;relaxed_tolerance 两档映射(fp32→fp16 行、
// fp16→bf16 行)+ bf16 末行与非浮点(kInt32)FRAME_CHECK 死亡测试;
// tensor_all_close 通过 / 首违例报告(索引+期望值+实际值出现在消息中)/
// shape 不一致(消息含双方 shape)/ dtype 不一致(消息含双方 dtype 名)/
// 容差边界(恰好等于 atol+rtol*|e| 应通过)/ NaN 不相等 / 整型精确比较(相等
// 过、差 1 拒),外加 fp16、bf16 各至少一条路径(用 float16_t/bfloat16_t 位级
// 构造已知值,已知位模式取自 tests/cpp/core/test_dtype.cpp 验证过的常量)。
// 全程使用 host 内存的 FakeAllocator(tests/cpp/core/fake_allocator.h),
// 与 tests/cpp/core/test_tensor.cpp 保持一致,不依赖任何已注册后端。
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

#include <frame/core/dtype.h>
#include <frame/core/tensor.h>

#include "../core/fake_allocator.h"
#include "tolerance.h"

namespace {

using frame::bfloat16_t;
using frame::cpu_device;
using frame::DType;
using frame::DTypeCode;
using frame::float16_t;
using frame::Shape;
using frame::Tensor;
using frame::testing::default_tolerance;
using frame::testing::FakeAllocator;
using frame::testing::relaxed_tolerance;
using frame::testing::tensor_all_close;
using frame::testing::Tolerance;

// 按给定值列表构造 1D Tensor(host 内存,FakeAllocator)。
template <typename T>
Tensor MakeTensor1D(FakeAllocator& allocator, const std::vector<T>& values) {
  frame::Result<Tensor> result = Tensor::empty(Shape({static_cast<int64_t>(values.size())}),
                                               DType::of<T>(), cpu_device(), allocator);
  EXPECT_TRUE(result.is_ok());
  Tensor tensor = result.value();
  T* data = tensor.data<T>();
  for (size_t i = 0; i < values.size(); ++i) data[i] = values[i];
  return tensor;
}

// ---------------------------------------------------------------------------
// default_tolerance:BUILD-011 默认表三档数值精确断言 + 整型/bool 精确档 +
// 表外浮点(kFloat64)fail-fast。
// ---------------------------------------------------------------------------

TEST(DefaultToleranceTest, Float32MatchesBuildEleven) {
  const Tolerance tol = default_tolerance(DTypeCode::kFloat32);
  EXPECT_EQ(tol.rtol, 1e-5);
  EXPECT_EQ(tol.atol, 1e-6);
}

TEST(DefaultToleranceTest, Float16MatchesBuildEleven) {
  const Tolerance tol = default_tolerance(DTypeCode::kFloat16);
  EXPECT_EQ(tol.rtol, 1e-2);
  EXPECT_EQ(tol.atol, 1e-3);
}

TEST(DefaultToleranceTest, BFloat16MatchesBuildEleven) {
  const Tolerance tol = default_tolerance(DTypeCode::kBFloat16);
  EXPECT_EQ(tol.rtol, 2e-2);
  EXPECT_EQ(tol.atol, 2e-3);
}

TEST(DefaultToleranceTest, IntegralAndBoolTypesAreExact) {
  constexpr std::array<DTypeCode, 6> kCodes = {
      DTypeCode::kInt8,  DTypeCode::kInt16, DTypeCode::kInt32,
      DTypeCode::kInt64, DTypeCode::kUInt8, DTypeCode::kBool,
  };
  for (DTypeCode code : kCodes) {
    SCOPED_TRACE(static_cast<int>(code));
    const Tolerance tol = default_tolerance(code);
    EXPECT_EQ(tol.rtol, 0.0);
    EXPECT_EQ(tol.atol, 0.0);
  }
}

TEST(DefaultToleranceDeathTest, Float64IsOutOfTableAndAborts) {
  // BUILD-011 表未列出 fp64 容差值:default_tolerance 禁止臆造,须 fail-fast。
  EXPECT_DEATH({ default_tolerance(DTypeCode::kFloat64); }, "FRAME_CHECK failed");
}

// ---------------------------------------------------------------------------
// relaxed_tolerance:大规模归约放宽一档,两档映射 + bf16 末行/非浮点 fail-fast。
// ---------------------------------------------------------------------------

TEST(RelaxedToleranceTest, Float32RelaxesToFloat16Row) {
  const Tolerance tol = relaxed_tolerance(DTypeCode::kFloat32);
  EXPECT_EQ(tol.rtol, 1e-2);
  EXPECT_EQ(tol.atol, 1e-3);
}

TEST(RelaxedToleranceTest, Float16RelaxesToBFloat16Row) {
  const Tolerance tol = relaxed_tolerance(DTypeCode::kFloat16);
  EXPECT_EQ(tol.rtol, 2e-2);
  EXPECT_EQ(tol.atol, 2e-3);
}

TEST(RelaxedToleranceDeathTest, BFloat16HasNoFurtherRowAndAborts) {
  // bf16 已是 BUILD-011 表末行,继续放宽须调用方个案显式构造 Tolerance,
  // relaxed_tolerance 本身必须 fail-fast,不得代为决定放宽到什么程度。
  EXPECT_DEATH({ relaxed_tolerance(DTypeCode::kBFloat16); }, "FRAME_CHECK failed");
}

TEST(RelaxedToleranceDeathTest, IntegralTypeHasNoRelaxConceptAndAborts) {
  // 整型是精确比较,不存在"放宽一档"的语义,relaxed_tolerance 须 fail-fast。
  EXPECT_DEATH({ relaxed_tolerance(DTypeCode::kInt32); }, "FRAME_CHECK failed");
}

// ---------------------------------------------------------------------------
// tensor_all_close:通过 / 首违例报告 / shape 不一致 / dtype 不一致 / 边界 / NaN。
// ---------------------------------------------------------------------------

TEST(TensorAllCloseTest, PassesForFloat32WithinDefaultTolerance) {
  FakeAllocator allocator;
  Tensor actual = MakeTensor1D<float>(allocator, {1.0f, 2.0f, 3.0f});
  Tensor expected = MakeTensor1D<float>(allocator, {1.0f, 2.0f, 3.0f});
  EXPECT_TRUE(tensor_all_close(actual, expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST(TensorAllCloseTest, ReportsFirstViolationIndexAndValues) {
  FakeAllocator allocator;
  Tensor actual = MakeTensor1D<float>(allocator, {1.0f, 2.0f, 300.0f});
  Tensor expected = MakeTensor1D<float>(allocator, {1.0f, 2.0f, 3.0f});
  const ::testing::AssertionResult result =
      tensor_all_close(actual, expected, default_tolerance(DTypeCode::kFloat32));
  ASSERT_FALSE(result);
  const std::string message = result.message();
  EXPECT_NE(message.find("index 2"), std::string::npos);
  EXPECT_NE(message.find("actual=300"), std::string::npos);
  EXPECT_NE(message.find("expected=3"), std::string::npos);
}

TEST(TensorAllCloseTest, FailsWithBothShapesOnShapeMismatch) {
  FakeAllocator allocator;
  Tensor actual = MakeTensor1D<float>(allocator, {1.0f, 2.0f});
  Tensor expected = MakeTensor1D<float>(allocator, {1.0f, 2.0f, 3.0f});
  const ::testing::AssertionResult result =
      tensor_all_close(actual, expected, default_tolerance(DTypeCode::kFloat32));
  ASSERT_FALSE(result);
  const std::string message = result.message();
  EXPECT_NE(message.find("[2]"), std::string::npos);  // actual shape
  EXPECT_NE(message.find("[3]"), std::string::npos);  // expected shape
}

TEST(TensorAllCloseTest, FailsOnDtypeMismatch) {
  FakeAllocator allocator;
  Tensor actual = MakeTensor1D<float>(allocator, {1.0f, 2.0f});
  Tensor expected = MakeTensor1D<int32_t>(allocator, {1, 2});
  const ::testing::AssertionResult result =
      tensor_all_close(actual, expected, default_tolerance(DTypeCode::kFloat32));
  ASSERT_FALSE(result);
  const std::string message = result.message();
  EXPECT_NE(message.find("float32"), std::string::npos);
  EXPECT_NE(message.find("int32"), std::string::npos);
}

TEST(TensorAllCloseTest, BoundaryExactlyAtToleranceSumPasses) {
  FakeAllocator allocator;
  // rtol=0 避免乘法舍入误差干扰边界判定;100/105/5 均为 float32 精确可表示的
  // 整数(远小于 2^24),故 |105-100|=5.0 与 atol=5.0 在 double 下逐位相等,
  // 判据 |a-e|<=atol+rtol*|e| 取等号,构造出确定性(非近似)的边界穿越用例。
  const Tolerance tol{/*rtol=*/0.0, /*atol=*/5.0};
  Tensor expected = MakeTensor1D<float>(allocator, {100.0f});
  Tensor actual = MakeTensor1D<float>(allocator, {105.0f});
  EXPECT_TRUE(tensor_all_close(actual, expected, tol));
}

TEST(TensorAllCloseTest, NanIsNeverEqualEvenToItself) {
  FakeAllocator allocator;
  const float nan_value = std::numeric_limits<float>::quiet_NaN();
  Tensor actual = MakeTensor1D<float>(allocator, {nan_value});
  Tensor expected = MakeTensor1D<float>(allocator, {nan_value});
  // IEEE 754 语义:NaN 不与任何值相等(含自身),tensor_all_close 头注释已声明
  // 不对 NaN 做特殊放行。
  EXPECT_FALSE(tensor_all_close(actual, expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST(TensorAllCloseTest, IntegralExactComparisonPassesOnEqualRejectsOnOffByOne) {
  FakeAllocator allocator;
  Tensor a = MakeTensor1D<int32_t>(allocator, {1, 2, 3});
  Tensor equal = MakeTensor1D<int32_t>(allocator, {1, 2, 3});
  Tensor off_by_one = MakeTensor1D<int32_t>(allocator, {1, 2, 4});
  const Tolerance tol = default_tolerance(DTypeCode::kInt32);
  EXPECT_TRUE(tensor_all_close(a, equal, tol));
  EXPECT_FALSE(tensor_all_close(a, off_by_one, tol));
}

// fp16 路径:位级构造已知值(0x3C00 == float_to_float16(1.0f).bits,
// 0x4000 == float_to_float16(2.0f).bits,均见 tests/cpp/core/test_dtype.cpp
// Float16ConversionTest.KnownValuesMatchExpectedBitPattern 独立验证过的常量)。
TEST(TensorAllCloseTest, Float16PathComparesViaBitLevelConversion) {
  FakeAllocator allocator;
  Tensor one = MakeTensor1D<float16_t>(allocator, {float16_t{0x3C00u}});
  Tensor one_again = MakeTensor1D<float16_t>(allocator, {float16_t{0x3C00u}});
  Tensor two = MakeTensor1D<float16_t>(allocator, {float16_t{0x4000u}});
  const Tolerance tol = default_tolerance(DTypeCode::kFloat16);
  EXPECT_TRUE(tensor_all_close(one, one_again, tol));
  EXPECT_FALSE(tensor_all_close(one, two, tol));
}

// bf16 路径:位级构造已知值(0x3F80 == float_to_bfloat16(1.0f).bits,
// 0xBF80 == float_to_bfloat16(-1.0f).bits,均见 test_dtype.cpp 已验证的常量)。
TEST(TensorAllCloseTest, BFloat16PathComparesViaBitLevelConversion) {
  FakeAllocator allocator;
  Tensor one = MakeTensor1D<bfloat16_t>(allocator, {bfloat16_t{0x3F80u}});
  Tensor one_again = MakeTensor1D<bfloat16_t>(allocator, {bfloat16_t{0x3F80u}});
  Tensor minus_one = MakeTensor1D<bfloat16_t>(allocator, {bfloat16_t{0xBF80u}});
  const Tolerance tol = default_tolerance(DTypeCode::kBFloat16);
  EXPECT_TRUE(tensor_all_close(one, one_again, tol));
  EXPECT_FALSE(tensor_all_close(one, minus_one, tol));
}

}  // namespace
