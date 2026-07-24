#pragma once
// 数据类型系统。设计原则(铁律 #1②):C++ 类型到 dtype 的映射在编译期完成;
// 运行时 dtype 只在 dispatch_dtype() 这一个点转回编译期类型,此后全程模板。

#include <bit>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>

#include <frame/core/macros.h>

namespace frame {

// dtype 编码封闭枚举。新增 dtype 必须加在 kCount 之前,并补 dtype_traits 特化。
enum class DTypeCode : uint8_t {
  kFloat32 = 0,
  kFloat64,
  kFloat16,
  kBFloat16,
  kInt8,
  kInt16,
  kInt32,
  kInt64,
  kUInt8,
  kBool,
  kCount,  // 哨兵:dtype 总数
};

// 编译期映射:主模板不定义 → 未支持类型直接编译失败(而非运行时报错)。
template <typename T>
struct dtype_traits;

// 存储型半精度浮点(v0 仅存储/搬运,不提供算术运算符)。
// 注意:与 C++23 std::float16_t/std::bfloat16_t 同名但语义不同(本类型非算术类型),
// 勿混用;项目基线 C++20(ADR-0004)。
struct float16_t {
  uint16_t bits;
};
struct bfloat16_t {
  uint16_t bits;
};

// ---------------------------------------------------------------------------
// fp16/bf16 ↔ fp32 位级转换(自研,constexpr)。fp32 → fp16/bf16 一律
// round-to-nearest-even(RTNE);fp16/bf16 → fp32 为精确加宽,无需舍入。
// ---------------------------------------------------------------------------

// float32 位模式 → float16(IEEE 754 binary16),round-to-nearest-even。
constexpr float16_t float_to_float16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  const uint32_t sign = (bits >> 16) & 0x8000U;
  const uint32_t abs_bits = bits & 0x7fffffffU;

  if (abs_bits > 0x7f800000U) {
    // NaN:统一折成一个 quiet NaN 位模式,不保留原始 payload。
    return float16_t{static_cast<uint16_t>(sign | 0x7e00U)};
  }
  if (abs_bits == 0x7f800000U) {
    // 无穷大。
    return float16_t{static_cast<uint16_t>(sign | 0x7c00U)};
  }
  if (abs_bits < 0x33000000U) {
    // 小于最小次正规数一半(2^-25):RTNE 舍入到 0。
    return float16_t{static_cast<uint16_t>(sign)};
  }
  if (abs_bits < 0x38800000U) {
    // 次正规数区间:补隐含位后右移到 10 位尾数,按 RTNE 处理被移出的位。
    const uint32_t shift = 126U - (abs_bits >> 23);  // 取值范围 [14, 24]
    const uint32_t mantissa_full = (abs_bits & 0x7fffffU) | 0x800000U;
    uint32_t half_mantissa = mantissa_full >> shift;
    const uint32_t round_bit = (mantissa_full >> (shift - 1)) & 1U;
    const uint32_t sticky_mask = (1U << (shift - 1)) - 1U;
    const bool sticky = (mantissa_full & sticky_mask) != 0U;
    if (round_bit != 0U && (sticky || (half_mantissa & 1U) != 0U)) ++half_mantissa;
    return float16_t{static_cast<uint16_t>(sign | half_mantissa)};
  }

  // 正规数区间:指数偏置由 127 调整为 15,尾数 23 位舍入到 10 位(RTNE)。
  // 舍入后若进位溢出到指数全 1,自然落到 0x7c00(无穷大),无需单独分支。
  const uint32_t rounded = abs_bits + 0xfffU + ((abs_bits >> 13) & 1U);
  if (rounded >= 0x47800000U) {
    return float16_t{static_cast<uint16_t>(sign | 0x7c00U)};
  }
  return float16_t{static_cast<uint16_t>(sign | ((rounded - 0x38000000U) >> 13))};
}

// float16 → float32:精确加宽(含次正规数归一化),无信息损失,无需舍入。
constexpr float float16_to_float(float16_t value) {
  const uint32_t half = value.bits;
  const uint32_t sign = (half & 0x8000U) << 16;
  const uint32_t exp = (half >> 10) & 0x1fU;
  uint32_t mantissa = half & 0x3ffU;

  if (exp == 0U) {
    if (mantissa == 0U) return std::bit_cast<float>(sign);  // ±0
    // 次正规数:左移尾数直到隐含 1 落在第 10 位,同步推导 fp32 侧的偏置指数。
    uint32_t shift = 0U;
    while ((mantissa & 0x400U) == 0U) {
      mantissa <<= 1;
      ++shift;
    }
    mantissa &= 0x3ffU;
    const uint32_t exp32 = 113U - shift;
    return std::bit_cast<float>(sign | (exp32 << 23) | (mantissa << 13));
  }
  if (exp == 0x1fU) {
    // 无穷大或 NaN:阶码全 1,尾数原样加宽。
    return std::bit_cast<float>(sign | 0x7f800000U | (mantissa << 13));
  }
  const uint32_t exp32 = exp - 15U + 127U;
  return std::bit_cast<float>(sign | (exp32 << 23) | (mantissa << 13));
}

// float32 位模式 → bfloat16:截断高 16 位,round-to-nearest-even。
constexpr bfloat16_t float_to_bfloat16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    // NaN:保留高位尾数并强制 quiet 位,避免退化为无穷大。
    return bfloat16_t{static_cast<uint16_t>((bits >> 16) | 0x0040U)};
  }
  const uint32_t rounding_bias = 0x7fffU + ((bits >> 16) & 1U);
  const uint32_t rounded = bits + rounding_bias;
  return bfloat16_t{static_cast<uint16_t>(rounded >> 16)};
}

// bfloat16 → float32:精确加宽(低 16 位补零),无需舍入。
constexpr float bfloat16_to_float(bfloat16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value.bits) << 16);
}

template <>
struct dtype_traits<float> {
  static constexpr DTypeCode code = DTypeCode::kFloat32;
  static constexpr size_t size = 4;
  static constexpr std::string_view name = "float32";
};
template <>
struct dtype_traits<double> {
  static constexpr DTypeCode code = DTypeCode::kFloat64;
  static constexpr size_t size = 8;
  static constexpr std::string_view name = "float64";
};
template <>
struct dtype_traits<std::int8_t> {
  static constexpr DTypeCode code = DTypeCode::kInt8;
  static constexpr size_t size = 1;
  static constexpr std::string_view name = "int8";
};
template <>
struct dtype_traits<std::int16_t> {
  static constexpr DTypeCode code = DTypeCode::kInt16;
  static constexpr size_t size = 2;
  static constexpr std::string_view name = "int16";
};
template <>
struct dtype_traits<std::int32_t> {
  static constexpr DTypeCode code = DTypeCode::kInt32;
  static constexpr size_t size = 4;
  static constexpr std::string_view name = "int32";
};
template <>
struct dtype_traits<std::int64_t> {
  static constexpr DTypeCode code = DTypeCode::kInt64;
  static constexpr size_t size = 8;
  static constexpr std::string_view name = "int64";
};
template <>
struct dtype_traits<std::uint8_t> {
  static constexpr DTypeCode code = DTypeCode::kUInt8;
  static constexpr size_t size = 1;
  static constexpr std::string_view name = "uint8";
};
template <>
struct dtype_traits<bool> {
  static constexpr DTypeCode code = DTypeCode::kBool;
  static constexpr size_t size = 1;
  static constexpr std::string_view name = "bool";
};
template <>
struct dtype_traits<float16_t> {
  static constexpr DTypeCode code = DTypeCode::kFloat16;
  static constexpr size_t size = 2;
  static constexpr std::string_view name = "float16";
};
template <>
struct dtype_traits<bfloat16_t> {
  static constexpr DTypeCode code = DTypeCode::kBFloat16;
  static constexpr size_t size = 2;
  static constexpr std::string_view name = "bfloat16";
};

// 概念:可作为张量元素的类型(存在 dtype_traits 特化即满足)。
template <typename T>
concept ScalarType = requires { dtype_traits<std::remove_cv_t<T>>::code; };

// 运行时 dtype 值类型:constexpr 友好、无虚函数、按值传递。
class DType {
 public:
  constexpr explicit DType(DTypeCode code) : code_(code) {}

  template <ScalarType T>
  static constexpr DType of() {
    return DType(dtype_traits<std::remove_cv_t<T>>::code);
  }

  constexpr DTypeCode code() const { return code_; }

  // 单元素字节数。constexpr 成员隐式 inline,定义必须写在本头文件内
  // (写进 .cpp 会导致其他翻译单元未定义引用/无法常量求值),见下方类外定义。
  constexpr size_t itemsize() const;

  // dtype 英文名。定义同样必须写在本头文件内,见下方类外定义。
  constexpr std::string_view name() const;

  constexpr bool operator==(const DType&) const = default;

 private:
  DTypeCode code_;
};

// DType::itemsize()/name() 的 constexpr 查表定义(以 dtype_traits::size/name 为
// 唯一数据源,逐 dtype 穷举 switch;kCount 或越界值属违反不变量,走 FRAME_CHECK
// fatal——按 CPP-020,核心库不使用异常)。
constexpr size_t DType::itemsize() const {
  switch (code_) {
    case DTypeCode::kFloat32:
      return dtype_traits<float>::size;
    case DTypeCode::kFloat64:
      return dtype_traits<double>::size;
    case DTypeCode::kFloat16:
      return dtype_traits<float16_t>::size;
    case DTypeCode::kBFloat16:
      return dtype_traits<bfloat16_t>::size;
    case DTypeCode::kInt8:
      return dtype_traits<std::int8_t>::size;
    case DTypeCode::kInt16:
      return dtype_traits<std::int16_t>::size;
    case DTypeCode::kInt32:
      return dtype_traits<std::int32_t>::size;
    case DTypeCode::kInt64:
      return dtype_traits<std::int64_t>::size;
    case DTypeCode::kUInt8:
      return dtype_traits<std::uint8_t>::size;
    case DTypeCode::kBool:
      return dtype_traits<bool>::size;
    case DTypeCode::kCount:
      break;
  }
  FRAME_CHECK(false);  // 非法 dtype code(kCount 或越界值),违反调用方不变量
  return 0;            // 不可达:仅为满足函数返回路径的编译要求
}

constexpr std::string_view DType::name() const {
  switch (code_) {
    case DTypeCode::kFloat32:
      return dtype_traits<float>::name;
    case DTypeCode::kFloat64:
      return dtype_traits<double>::name;
    case DTypeCode::kFloat16:
      return dtype_traits<float16_t>::name;
    case DTypeCode::kBFloat16:
      return dtype_traits<bfloat16_t>::name;
    case DTypeCode::kInt8:
      return dtype_traits<std::int8_t>::name;
    case DTypeCode::kInt16:
      return dtype_traits<std::int16_t>::name;
    case DTypeCode::kInt32:
      return dtype_traits<std::int32_t>::name;
    case DTypeCode::kInt64:
      return dtype_traits<std::int64_t>::name;
    case DTypeCode::kUInt8:
      return dtype_traits<std::uint8_t>::name;
    case DTypeCode::kBool:
      return dtype_traits<bool>::name;
    case DTypeCode::kCount:
      break;
  }
  FRAME_CHECK(false);  // 非法 dtype code(kCount 或越界值),违反调用方不变量
  return {};           // 不可达:仅为满足函数返回路径的编译要求
}

// 运行时 → 编译期的唯一桥:fn 是泛型可调用体,以 dtype_traits<T> 对应的 T
// 实例化后被调用。这是全框架唯一允许的 dtype 运行时分派点(switch 一次,之后静态)。
// 定义必须写在本头文件内(使用处须可见,写进 .cpp 无法实例化)。
// 前置条件:fn 对全部合法 T 返回同一类型(decltype(auto) 由各 case 的 return
// 语句共同推导)。
template <typename Fn>
decltype(auto) dispatch_dtype(DTypeCode code, Fn&& fn) {
  switch (code) {
    case DTypeCode::kFloat32:
      return std::forward<Fn>(fn).template operator()<float>();
    case DTypeCode::kFloat64:
      return std::forward<Fn>(fn).template operator()<double>();
    case DTypeCode::kFloat16:
      return std::forward<Fn>(fn).template operator()<float16_t>();
    case DTypeCode::kBFloat16:
      return std::forward<Fn>(fn).template operator()<bfloat16_t>();
    case DTypeCode::kInt8:
      return std::forward<Fn>(fn).template operator()<std::int8_t>();
    case DTypeCode::kInt16:
      return std::forward<Fn>(fn).template operator()<std::int16_t>();
    case DTypeCode::kInt32:
      return std::forward<Fn>(fn).template operator()<std::int32_t>();
    case DTypeCode::kInt64:
      return std::forward<Fn>(fn).template operator()<std::int64_t>();
    case DTypeCode::kUInt8:
      return std::forward<Fn>(fn).template operator()<std::uint8_t>();
    case DTypeCode::kBool:
      return std::forward<Fn>(fn).template operator()<bool>();
    case DTypeCode::kCount:
      break;
  }
  // 非法/kCount 一律走 fatal:switch 已穷举全部合法取值,执行到此处即违反不变量。
  FRAME_CHECK(false);
  return std::forward<Fn>(fn).template operator()<float>();  // 不可达:仅满足返回类型推导
}

}  // namespace frame
