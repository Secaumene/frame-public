#pragma once
// cpu kernel 侧共享的 float 累加位级转换工具(M22 批4 T3 判重收敛)。
// 收敛动机(铁律 5):to_accum/from_accum 曾在 reduction/conv/pool 与本批
// gather/sequence 五个 kernel 文件各持一份逐字相同实现(code-reviewer 判重
// 命中 REUSE-002),同目录可共享,收敛为单份;仅供 src/backends/cpu/kernels/
// 内部包含,不入公开 API。其余 dtype 分支返回默认值——调用方在
// dispatch_dtype 前已拒绝这些 dtype,运行时不可达,仅为满足 dispatch_dtype
// 对全体 DTypeCode 的编译期穷举要求。matmul.cpp 的收缩维累加另有遍历顺序
// 语义,刻意不并入(其文件内注释已作论证)。

#include <type_traits>

#include <frame/core/dtype.h>

namespace frame::backends::cpu {

// 元素读入时统一升 float 累加(fp16/bf16 借用既有 float16_to_float/
// bfloat16_to_float 位级转换)。
template <typename T>
inline float to_accum(T value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else if constexpr (std::is_same_v<T, frame::float16_t>) {
    return frame::float16_to_float(value);
  } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
    return frame::bfloat16_to_float(value);
  } else {
    return 0.0F;
  }
}

// float 累加结果转回 T:与 to_accum 对称。
template <typename T>
inline T from_accum(float value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else if constexpr (std::is_same_v<T, frame::float16_t>) {
    return frame::float_to_float16(value);
  } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
    return frame::float_to_bfloat16(value);
  } else {
    return T{};
  }
}

}  // namespace frame::backends::cpu
