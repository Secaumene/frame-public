#pragma once
// 数值容差工具:BUILD-011(docs/standards/build-and-test.md 第 6 节)的唯一实现
// 载体,全仓库数值对比测试一律经本文件比较,禁止手写容差断言/自造阈值。
// BUILD-011 的 grep 执法规则会扫描 tests/cpp/ 下手写容差断言宏的字面名字
// (含注释),故本文件全文(含注释)不得出现该字面名字。
//
// NaN 语义(IEEE 754):任何一侧为 NaN 一律判定为"不相等"(不与任何值相等,含
// 自身);tensor_all_close 因此不对 NaN 做特殊放行。期望结果本身含 NaN 的用例
// (例如验证某算子在特定输入下产生 NaN)不能依赖本工具判定"通过",须调用方在
// 测试代码中自行显式检查(如 std::isnan),不属本文件覆盖范围。
//
// 无穷大语义:两侧同为 inf 时差值为 NaN,同样判定为"不相等"——期望结果含
// ±inf 的用例(如溢出行为校验)与 NaN 同理,须调用方自行显式检查(std::isinf
// 与符号比对),不能依赖本工具放行。
//
// dtype 分派经 frame::dispatch_dtype(include/frame/core/dtype.h)完成,全程无
// virtual/dynamic_cast/异常(铁律 #1)。

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <type_traits>

#include <frame/core/dtype.h>
#include <frame/core/macros.h>
#include <frame/core/shape.h>
#include <frame/core/tensor.h>

namespace frame::testing {

// 数值容差:相对误差 rtol + 绝对误差 atol,判据 |a-e| <= atol + rtol*|e|。
struct Tolerance {
  double rtol;
  double atol;
};

// BUILD-011 默认容差表(唯一数据源,不得在别处复制数值)。表外浮点类型(当前
// 仅 kFloat64,BUILD-011 未列出其容差值)与非法/哨兵 code 一律 FRAME_CHECK
// fail-fast:禁止臆造未经规范认定的容差,个案需要时由调用方在用例处显式构造
// Tolerance{...} 并写明依据。整型与 bool 是精确比较(rtol=atol=0)。
inline Tolerance default_tolerance(DTypeCode code) {
  switch (code) {
    case DTypeCode::kFloat32:
      return Tolerance{1e-5, 1e-6};
    case DTypeCode::kFloat16:
      return Tolerance{1e-2, 1e-3};
    case DTypeCode::kBFloat16:
      return Tolerance{2e-2, 2e-3};
    case DTypeCode::kInt8:
    case DTypeCode::kInt16:
    case DTypeCode::kInt32:
    case DTypeCode::kInt64:
    case DTypeCode::kUInt8:
    case DTypeCode::kBool:
      return Tolerance{0.0, 0.0};
    case DTypeCode::kFloat64:
    case DTypeCode::kCount:
      break;
  }
  FRAME_CHECK(false);          // dtype 不在 BUILD-011 表内,禁止臆造容差(fail-fast)
  return Tolerance{0.0, 0.0};  // 不可达:仅满足返回路径的编译要求
}

// 大规模归约放宽一档(BUILD-011:单输出元素累加次数 >= 2^20 的 reduction/matmul
// 类用例):取默认表中下一行的值,fp32 → fp16 行、fp16 → bf16 行。bf16 已是表
// 末行,继续放宽须调用方在用例处显式构造 Tolerance{...} 并以中文注释写明依据,
// 本函数对 bf16 同样 fail-fast,不代为决定放宽到什么程度。整型/bool 不存在
// "放宽"概念(精确比较),同样 fail-fast。
inline Tolerance relaxed_tolerance(DTypeCode code) {
  switch (code) {
    case DTypeCode::kFloat32:
      return default_tolerance(DTypeCode::kFloat16);
    case DTypeCode::kFloat16:
      return default_tolerance(DTypeCode::kBFloat16);
    case DTypeCode::kBFloat16:
    case DTypeCode::kInt8:
    case DTypeCode::kInt16:
    case DTypeCode::kInt32:
    case DTypeCode::kInt64:
    case DTypeCode::kUInt8:
    case DTypeCode::kBool:
    case DTypeCode::kFloat64:
    case DTypeCode::kCount:
      break;
  }
  FRAME_CHECK(false);          // 无下一档可放宽(bf16 末行)或非浮点,须调用方个案处理
  return Tolerance{0.0, 0.0};  // 不可达:仅满足返回路径的编译要求
}

// BUILD-011 fp32(allow_tf32) 档(ADR-0019;独立于主档位表,不参与「放宽一档」
// 的行序推导)。值已定案(2026-07-18 本机 K=512 matmul 实测回填:最大相对
// 偏差 1.58e-4、最大绝对偏差 2.93e-3,合成判据下 rtol 项覆盖,rtol 留 6 倍
// 裕度)。近零抵消场景(cpu 参考值趋零)不放宽本档:按 BUILD-011 既有口径由
// 调用方个案显式构造 Tolerance{...} 并以中文注释写明依据。
// 仅限以 CompileOptions::allow_tf32 = true 编译的用例取用;cpu 参考实现恒为
// 严格 fp32(ARCH-041 不变),TF32 开启时的 cpu-cuda 比对用本档;禁止用于
// 严格 fp32 比较(防放松洗白,判定方法见 BUILD-011)。
inline Tolerance tf32_tolerance() { return Tolerance{1e-3, 1e-4}; }

namespace detail {

// 单元素转 double,供比较判据与失败消息展示:fp16/bf16 经既有位级转换升 float
// 再隐式升 double;其余算术类型(含整型/bool)直接 static_cast。仅用于展示与
// 浮点判据计算,整型判等走 ScalarClose 的独立分支,不经本函数中转(避免大整数
// 经 double 损失精度)。
template <typename T>
double ToDouble(T value) {
  if constexpr (std::is_same_v<T, float16_t>) {
    return static_cast<double>(float16_to_float(value));
  } else if constexpr (std::is_same_v<T, bfloat16_t>) {
    return static_cast<double>(bfloat16_to_float(value));
  } else {
    return static_cast<double>(value);
  }
}

// 单元素判据:浮点类(含 fp16/bf16)按 |a-e|<=atol+rtol*|e| 比较,且任一侧为
// NaN 一律判不等(IEEE 754 语义,见本文件头注释);整型/bool 用原生类型精确
// 比较,不经 double 中转(避免 int64 大数值经 double 损失精度)。
template <typename T>
bool ScalarClose(T actual, T expected, const Tolerance& tol) {
  if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, float16_t> ||
                std::is_same_v<T, bfloat16_t>) {
    const double a = ToDouble(actual);
    const double e = ToDouble(expected);
    if (std::isnan(a) || std::isnan(e)) return false;
    return std::fabs(a - e) <= tol.atol + tol.rtol * std::fabs(e);
  } else {
    return actual == expected;
  }
}

}  // namespace detail

// 逐元素判等:先比对 shape/dtype,不一致直接失败并输出双方差异详情;一致后逐
// 元素按 ScalarClose 判据比较,失败报告首个违例的线性索引、期望值、实际值与
// 容差。dtype 分派经 dispatch_dtype 完成(fp16/bf16 走位级转换升 float/double
// 比较,见 detail::ToDouble)。
inline ::testing::AssertionResult tensor_all_close(const Tensor& actual, const Tensor& expected,
                                                   Tolerance tol) {
  if (!(actual.shape() == expected.shape())) {
    return ::testing::AssertionFailure()
           << "tensor_all_close: shape mismatch, actual=" << actual.shape().to_string()
           << " expected=" << expected.shape().to_string();
  }
  if (actual.dtype() != expected.dtype()) {
    return ::testing::AssertionFailure()
           << "tensor_all_close: dtype mismatch, actual=" << actual.dtype().name()
           << " expected=" << expected.dtype().name();
  }

  const int64_t numel = actual.numel();
  return dispatch_dtype(actual.dtype().code(), [&]<typename T>() -> ::testing::AssertionResult {
    const T* actual_data = static_cast<const T*>(actual.raw_data());
    const T* expected_data = static_cast<const T*>(expected.raw_data());
    for (int64_t i = 0; i < numel; ++i) {
      if (!detail::ScalarClose(actual_data[i], expected_data[i], tol)) {
        return ::testing::AssertionFailure()
               << "tensor_all_close: mismatch at linear index " << i
               << ", actual=" << detail::ToDouble(actual_data[i])
               << ", expected=" << detail::ToDouble(expected_data[i])
               << ", tolerance(rtol=" << tol.rtol << ", atol=" << tol.atol << ")";
      }
    }
    return ::testing::AssertionSuccess();
  });
}

}  // namespace frame::testing
