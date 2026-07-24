// dtype 模块单测:itemsize()/name() 全表覆盖、dispatch_dtype 逐 dtype 实例化、
// fp16/bf16 ↔ fp32 转换的已知值与往返一致性。
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <string_view>

#include <frame/core/dtype.h>

namespace {

using frame::DType;
using frame::DTypeCode;

struct DTypeExpectation {
  DTypeCode code;
  size_t itemsize;
  std::string_view name;
};

// 全部 10 个非哨兵 DTypeCode 的期望 itemsize/name;数据源是 dtype.h 中各
// dtype_traits 特化(itemsize()/name() 的唯一事实来源),逐个手写核对。
constexpr std::array<DTypeExpectation, 10> kAllDTypes = {{
    {DTypeCode::kFloat32, 4, "float32"},
    {DTypeCode::kFloat64, 8, "float64"},
    {DTypeCode::kFloat16, 2, "float16"},
    {DTypeCode::kBFloat16, 2, "bfloat16"},
    {DTypeCode::kInt8, 1, "int8"},
    {DTypeCode::kInt16, 2, "int16"},
    {DTypeCode::kInt32, 4, "int32"},
    {DTypeCode::kInt64, 8, "int64"},
    {DTypeCode::kUInt8, 1, "uint8"},
    {DTypeCode::kBool, 1, "bool"},
}};

TEST(DTypeTest, ItemsizeMatchesExpectedForAllCodes) {
  for (const auto& expectation : kAllDTypes) {
    SCOPED_TRACE(expectation.name);
    EXPECT_EQ(DType(expectation.code).itemsize(), expectation.itemsize);
  }
}

TEST(DTypeTest, NameMatchesExpectedForAllCodes) {
  for (const auto& expectation : kAllDTypes) {
    SCOPED_TRACE(expectation.name);
    EXPECT_EQ(DType(expectation.code).name(), expectation.name);
  }
}

// 探针 1:dispatch_dtype 实例化出的 T 的 sizeof 是否与 itemsize 对照一致。
struct SizeOfProbe {
  template <typename T>
  size_t operator()() const {
    return sizeof(T);
  }
};

// 探针 2:dispatch_dtype 实例化出的 T 本身是否正确(比 sizeof 更严格 ——
// int8/uint8/bool 的 size 都是 1,单靠 sizeof 无法区分是否分派到了错误的 T)。
struct CodeProbe {
  template <typename T>
  DTypeCode operator()() const {
    return frame::dtype_traits<T>::code;
  }
};

TEST(DTypeTest, DispatchDtypeInstantiatesMatchingTypeForAllCodes) {
  for (const auto& expectation : kAllDTypes) {
    SCOPED_TRACE(expectation.name);
    EXPECT_EQ(frame::dispatch_dtype(expectation.code, SizeOfProbe{}), expectation.itemsize);
    EXPECT_EQ(frame::dispatch_dtype(expectation.code, CodeProbe{}), expectation.code);
  }
}

// ---------------------------------------------------------------------------
// fp16 ↔ fp32:已知值 + 往返一致性。
// 期望位模式为独立算出(fp16 用 Python numpy.float16 的位视图 —— IEEE754
// binary16 round-to-nearest-even 的成熟参考实现;bf16 用手工 IEEE754 截断+
// round-to-nearest-even 推导,均未照抄被测代码的实现),仅做位级/值级精确比较,
// 不涉及浮点容差,故不使用 BUILD-011 的容差工具。
// ---------------------------------------------------------------------------

TEST(Float16ConversionTest, KnownValuesMatchExpectedBitPattern) {
  EXPECT_EQ(frame::float_to_float16(0.0f).bits, 0x0000u);
  EXPECT_EQ(frame::float_to_float16(-0.0f).bits, 0x8000u);
  EXPECT_EQ(frame::float_to_float16(1.0f).bits, 0x3C00u);
  EXPECT_EQ(frame::float_to_float16(-1.0f).bits, 0xBC00u);
  EXPECT_EQ(frame::float_to_float16(-2.5f).bits, 0xC100u);
  // 最大规格值(finite max,65504)。
  EXPECT_EQ(frame::float_to_float16(65504.0f).bits, 0x7BFFu);
  // 最小规格值(正规数下限,2^-14)。
  EXPECT_EQ(frame::float_to_float16(6.103515625e-05f).bits, 0x0400u);
}

TEST(Float16ConversionTest, RoundTripPreservesExactlyRepresentableValues) {
  const std::array<float, 7> values = {0.0f, -0.0f, 1.0f, -1.0f, -2.5f, 65504.0f, 6.103515625e-05f};
  for (float value : values) {
    SCOPED_TRACE(value);
    EXPECT_EQ(frame::float16_to_float(frame::float_to_float16(value)), value);
  }
}

TEST(BFloat16ConversionTest, KnownValuesMatchExpectedBitPattern) {
  EXPECT_EQ(frame::float_to_bfloat16(0.0f).bits, 0x0000u);
  EXPECT_EQ(frame::float_to_bfloat16(-0.0f).bits, 0x8000u);
  EXPECT_EQ(frame::float_to_bfloat16(1.0f).bits, 0x3F80u);
  EXPECT_EQ(frame::float_to_bfloat16(-1.0f).bits, 0xBF80u);
  EXPECT_EQ(frame::float_to_bfloat16(-2.5f).bits, 0xC020u);
  // 最大规格值(finite max,约 3.3895313892515355e+38)。
  EXPECT_EQ(frame::float_to_bfloat16(3.3895313892515355e+38f).bits, 0x7F7Fu);
  // 最小规格值(正规数下限,2^-126)。
  EXPECT_EQ(frame::float_to_bfloat16(1.1754943508222875e-38f).bits, 0x0080u);
}

TEST(BFloat16ConversionTest, RoundTripPreservesExactlyRepresentableValues) {
  const std::array<float, 7> values = {
      0.0f, -0.0f, 1.0f, -1.0f, -2.5f, 3.3895313892515355e+38f, 1.1754943508222875e-38f};
  for (float value : values) {
    SCOPED_TRACE(value);
    EXPECT_EQ(frame::bfloat16_to_float(frame::float_to_bfloat16(value)), value);
  }
}

// ---------------------------------------------------------------------------
// 边界值测试(补上方"已知值"测试遗漏的特殊分支:NaN / ±inf / 上溢 / 次正规 /
// 正规区 RTNE 平局)。期望位模式的推导方法:
//   - fp16 参考值用 numpy.float16 的位视图独立算出(与文件开头方法一致);
//   - bf16 参考值用与被测代码完全独立的 Python 手写 RTNE 截断算法交叉验证;
//   - 平局/进位类用例额外手工按 IEEE754 尾数-指数进位规则推导 round bit/sticky
//     位,与工具算出的参考值互相印证(均未运行 dtype.h 中的被测函数本身)。
// 涉及位级平局的输入一律用 FloatFromBits 从明确的 fp32 位模式构造,避免十进制
// 字面量的隐式舍入带来歧义;全部断言比较 uint16_t/uint32_t 位模式而非浮点相等
// (NaN 不可用 == 比较)。
// ---------------------------------------------------------------------------

// 从原始 IEEE754 位模式构造 float,用于精确构造边界/平局输入。
constexpr float FloatFromBits(uint32_t bits) { return std::bit_cast<float>(bits); }

TEST(Float16ConversionTest, NanCollapsesToCanonicalQuietNanPreservingSign) {
  // 规范正 qNaN:exp=0xFF,mantissa=0x400000(quiet 位置位,无 payload);以及其
  // 符号取反。位模式直接构造,不依赖 std::numeric_limits<float>::quiet_NaN()
  // 在各平台上的具体 payload(该值本身不是 v0 规范保证的稳定位模式)。
  const float positive_nan = FloatFromBits(0x7FC00000u);
  const float negative_nan = FloatFromBits(0xFFC00000u);
  // fp16 NaN 统一折叠为规范 quiet NaN 0x7e00,不保留 payload;符号位照常搬运。
  EXPECT_EQ(frame::float_to_float16(positive_nan).bits, 0x7e00u);
  EXPECT_EQ(frame::float_to_float16(negative_nan).bits, 0xfe00u);
}

TEST(BFloat16ConversionTest, NanPreservesPayloadAndForcesQuietBit) {
  // 信令 NaN,payload 仅落在被截断的低 16 位:exp=0xFF,mantissa=0x000001。
  // 若不强制置位 quiet 位,截断后高 16 位恰为 0x7f80——退化成 bf16 无穷大;
  // 强制 quiet 位后应为 0x7fc0(sign=0)/0xffc0(sign=1)。
  EXPECT_EQ(frame::float_to_bfloat16(FloatFromBits(0x7F800001u)).bits, 0x7fc0u);
  EXPECT_EQ(frame::float_to_bfloat16(FloatFromBits(0xFF800001u)).bits, 0xffc0u);
  // payload 落在保留的高 16 位内(mantissa 高 7 位 = 0b0101010,十六进制 0x2A):
  // 截断后应保留该 payload 而非仅得到与"无 payload"情形相同的规范位模式,
  // 证明 payload 确实被保留,而不只是巧合地强制了 quiet 位。
  EXPECT_EQ(frame::float_to_bfloat16(FloatFromBits(0x7FAA0000u)).bits, 0x7feau);
}

TEST(Float16ConversionTest, InfinityKnownBitPatternAndRoundTrip) {
  EXPECT_EQ(frame::float_to_float16(std::numeric_limits<float>::infinity()).bits, 0x7c00u);
  EXPECT_EQ(frame::float_to_float16(-std::numeric_limits<float>::infinity()).bits, 0xfc00u);
  // 反向转换应精确还原为 fp32 的 ±inf 位模式(0x7f800000 / 0xff800000)。
  EXPECT_EQ(std::bit_cast<uint32_t>(frame::float16_to_float(frame::float16_t{0x7c00u})),
            0x7f800000u);
  EXPECT_EQ(std::bit_cast<uint32_t>(frame::float16_to_float(frame::float16_t{0xfc00u})),
            0xff800000u);
}

TEST(BFloat16ConversionTest, InfinityKnownBitPatternAndRoundTrip) {
  EXPECT_EQ(frame::float_to_bfloat16(std::numeric_limits<float>::infinity()).bits, 0x7f80u);
  EXPECT_EQ(frame::float_to_bfloat16(-std::numeric_limits<float>::infinity()).bits, 0xff80u);
  EXPECT_EQ(std::bit_cast<uint32_t>(frame::bfloat16_to_float(frame::bfloat16_t{0x7f80u})),
            0x7f800000u);
  EXPECT_EQ(std::bit_cast<uint32_t>(frame::bfloat16_to_float(frame::bfloat16_t{0xff80u})),
            0xff800000u);
}

TEST(Float16ConversionTest, OverflowNearTiePointFollowsRoundToNearestEven) {
  // fp16 最大规格值 65504 所在指数段 ulp=32,半 ulp=16,平局点=65504+16=65520。
  // 65504 尾数全 1(奇),inf 的尾数视为 0(偶);RTNE 平局取偶 → 舍入到 inf。
  EXPECT_EQ(frame::float_to_float16(65520.0f).bits, 0x7c00u);
  // 平局点下方最近可表示的 fp32 值(65520.0f 的 fp32 前驱;该量级下 fp32
  // ulp=2^15*2^-23=2^-8,故前驱为 65520-2^-8=65519.99609375):未达半 ulp,
  // 舍入到 65504(与文件开头 KnownValues 用例中的 0x7BFF 一致)。
  EXPECT_EQ(frame::float_to_float16(FloatFromBits(0x477FEFFFu)).bits, 0x7BFFu);
}

TEST(BFloat16ConversionTest, FiniteMaxOverflowsToInfinity) {
  // FLT_MAX 尾数 23 位全 1:截断到 bf16 7 位尾数时被截断的 16 位同样全 1,
  // round bit=1 且 sticky=1(非平局,无条件进位);尾数进位溢出到指数位,
  // 自然产生 bf16 无穷大 0x7f80,不需要单独的溢出判断分支。
  EXPECT_EQ(frame::float_to_bfloat16(std::numeric_limits<float>::max()).bits, 0x7f80u);
}

TEST(Float16ConversionTest, SubnormalBoundaryValues) {
  // 2^-24:fp16 最小次正规值,尾数=1,可被精确表示,无需舍入。
  EXPECT_EQ(frame::float_to_float16(1.0f / 16777216.0f).bits, 0x0001u);
  // 2^-25:恰为最小次正规值的一半(平局点)。候选为 0(尾数视为偶)与最小次正规
  // 值(尾数=1,奇);RTNE 平局取偶 → 0。
  EXPECT_EQ(frame::float_to_float16(1.0f / 33554432.0f).bits, 0x0000u);
  // 取 2^-25 的 fp32 位模式并 +1(即上调一个 fp32 ulp):使被舍入位的 sticky
  // 部分非零,打破平局,RTNE 强制进位到最小次正规值 0x0001。
  EXPECT_EQ(frame::float_to_float16(FloatFromBits(0x33000001u)).bits, 0x0001u);
  // 次正规→正规进位边界:构造 1023.5 * 2^-24 = 2^-14 - 2^-25,恰为次正规最大值
  // (11 位合并尾数 1023,奇)与正规最小值(11 位合并尾数 1024,偶)之间的平局点。
  // RTNE 平局取偶 → 进位为正规最小值 2^-14,位模式为已知值 0x0400
  // (与文件开头 KnownValues 用例中 6.103515625e-05f 对应的 0x0400 一致)。
  EXPECT_EQ(frame::float_to_float16(FloatFromBits(0x387FE000u)).bits, 0x0400u);
}

TEST(Float16ConversionTest, SubnormalWideningToFloat32MatchesIndependentlyDerivedBits) {
  // float16_to_float 的次正规归一化分支(exp==0 且 mantissa!=0):循环左移尾数
  // 直到隐含 1 落在第 10 位,同步推导 fp32 侧的偏置指数(exp32=113-shift)。
  // 现有用例只测了 float_to_float16 这一正向;以下三例直接调用 float16_to_float
  // 覆盖该 while 循环所需左移次数的两端与中段,期望位模式按 fp16 次正规定义
  // value = mantissa * 2^-24 独立推导(先由 mantissa 的 leading-one 比特位求出
  // 指数,再将剩余比特左对齐得 23 位 fp32 尾数),不依赖被测代码本身的 shift/
  // exp32 计算过程。
  //
  // mantissa=1(0x0001,fp16 最小次正规,leading one 在第 0 位,循环需左移 10 次,
  // 达到 shift 上限):value=2^-24,是 2 的整数次幂,fraction=0;
  // 偏置指数 E=127-24=103=0x67,fp32 位模式=E<<23=0x33800000。
  EXPECT_EQ(std::bit_cast<uint32_t>(frame::float16_to_float(frame::float16_t{0x0001u})),
            0x33800000u);
  // mantissa=0x0155=341(居中的移位量):leading one 在第 8 位(2^8=256<=341<512),
  // 循环需左移 2 次;exponent=8-24=-16,偏置指数 E=127-16=111=0x6F;
  // fraction=(341-256)<<(23-8)=85<<15=0x2A8000;
  // fp32 位模式=E<<23 | fraction = 0x37800000 | 0x2A8000 = 0x37AA8000。
  EXPECT_EQ(std::bit_cast<uint32_t>(frame::float16_to_float(frame::float16_t{0x0155u})),
            0x37AA8000u);
  // mantissa=0x03FF=1023(fp16 次正规最大值,leading one 在第 9 位,循环只需左移
  // 1 次,shift 下限):exponent=9-24=-15,偏置指数 E=127-15=112=0x70;
  // fraction=(1023-512)<<(23-9)=511<<14=0x7FC000;
  // fp32 位模式=E<<23 | fraction = 0x38000000 | 0x7FC000 = 0x387FC000。
  EXPECT_EQ(std::bit_cast<uint32_t>(frame::float16_to_float(frame::float16_t{0x03FFu})),
            0x387FC000u);
}

TEST(Float16ConversionTest, NormalRegionRoundToNearestEvenTies) {
  // 均以 2.0f(fp32 尾数全 0,fp16 位模式 0x4000)为基准,在转换时被截断的低 13
  // 位(23 位 fp32 尾数 - 10 位 fp16 尾数 = 13 位)内构造 round bit / sticky 位。
  // round bit=1、sticky=0(恰为平局),目标 10 位尾数当前为偶(0)→ 舍入不变。
  EXPECT_EQ(frame::float_to_float16(FloatFromBits(0x40001000u)).bits, 0x4000u);
  // round bit=1、sticky=0(恰为平局),目标 10 位尾数当前为奇(1)→ 进位到偶(2)。
  EXPECT_EQ(frame::float_to_float16(FloatFromBits(0x40003000u)).bits, 0x4002u);
  // round bit=1、sticky≠0(超过半 ulp,非平局)→ 无条件进位,得尾数 1。
  EXPECT_EQ(frame::float_to_float16(FloatFromBits(0x40001001u)).bits, 0x4001u);
}

TEST(BFloat16ConversionTest, RoundToNearestEvenTies) {
  // bf16 = fp32 高 16 位 + RTNE;仍以 2.0f(fp32 高 16 位 = 0x4000)为基准,在
  // 被截断的低 16 位内构造 round bit / sticky 位。
  // round bit=1、sticky=0(0x????8000 平局),目标尾数当前为偶(0)→ 舍入不变。
  EXPECT_EQ(frame::float_to_bfloat16(FloatFromBits(0x40008000u)).bits, 0x4000u);
  // round bit=1、sticky=0(0x????8000 平局),目标尾数当前为奇(1)→ 进位到偶(2)。
  EXPECT_EQ(frame::float_to_bfloat16(FloatFromBits(0x40018000u)).bits, 0x4002u);
  // round bit=1、sticky≠0(0x????8001,非平局)→ 无条件进位,得尾数 1。
  EXPECT_EQ(frame::float_to_bfloat16(FloatFromBits(0x40008001u)).bits, 0x4001u);
}

}  // namespace
