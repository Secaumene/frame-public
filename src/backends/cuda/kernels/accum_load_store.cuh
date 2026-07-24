#pragma once
// cuda kernel 层共享的 device 端位型 <-> float 转换桥接工具(M22,批4 T4,
// code-reviewer 建议①判重收敛):elementwise_load/elementwise_store 曾在
// elementwise.cu(M19)、pool.cu(M21)、sequence.cu、gather.cu(均 M22 批4 T4)
// 四个 .cu 文件各持一份逐字相同实现;reduction.cu(M17)另以 reduction_load/
// reduction_store 命名持第五份同构实现(收编增量核查发现,一并并入)。同目录
// 可共享,收敛为单份(铁律 5);
// 函数名保持不变(调用点约 30 处,不做重命名 churn)。数值语义与 cpu 参考的
// float16_to_float/float_to_float16 等位级转换同语义,但改用 CUDA intrinsic
// 在设备侧完成(REUSE-011:cuda 侧不得复用 cpu 侧的纯 host constexpr 实现,
// 与 cpu 侧 accum_cast.h::to_accum/from_accum 同一动机、不同实现载体)。
// dispatch_dtype 对 DTypeCode 全体成员做编译期穷举,故本函数也会为 int8_t/
// bool 等其余 dtype 实例化;这些分支本体留空(返回 0.0f / 不写入)——调用方
// 在派发前已按白名单拒绝这些 dtype,运行时不可达,仅为满足 if constexpr 分支
// 穷举与编译期实例化要求。仅供 src/backends/cuda/kernels/ 内部 .cu 翻译单元
// 包含,不入公开 API。

#include <cstdint>
#include <type_traits>

#include <frame/core/dtype.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace frame::backends::cuda {

// 从 T* 指针读第 i 个元素并升 float(fp32 直读,fp16/bf16 经 __half2float/
// __bfloat162float)。
template <typename T>
__device__ __forceinline__ float elementwise_load(const T* ptr, int64_t i) {
  if constexpr (std::is_same_v<T, float>) {
    return ptr[i];
  } else if constexpr (std::is_same_v<T, frame::float16_t>) {
    return __half2float(*reinterpret_cast<const __half*>(&ptr[i]));
  } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&ptr[i]));
  } else {
    return 0.0F;
  }
}

// 把 float 值写回 T* 指针第 i 个元素(fp32 直写,fp16/bf16 经 __float2half/
// __float2bfloat16 转换后按位写回)。
template <typename T>
__device__ __forceinline__ void elementwise_store(T* ptr, int64_t i, float value) {
  if constexpr (std::is_same_v<T, float>) {
    ptr[i] = value;
  } else if constexpr (std::is_same_v<T, frame::float16_t>) {
    const __half h = __float2half(value);
    ptr[i] = *reinterpret_cast<const frame::float16_t*>(&h);
  } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
    const __nv_bfloat16 b = __float2bfloat16(value);
    ptr[i] = *reinterpret_cast<const frame::bfloat16_t*>(&b);
  }
  // 其余 dtype:调用方已按白名单拒绝,运行时不可达,无操作。
}

}  // namespace frame::backends::cuda
