// CUDA 逐元素内核(阶段 C-11):add/mul/relu/square/sigmoid(sigmoid 为 M21,
// 批3 T5 新增,镜像 relu 向量化路径)/tanh/rsqrt(M22,批4 T4,新增)。dtype 差异
// 经 dispatch_dtype 编译期展开(ARCH-042,禁运行时 dtype/设备分支);fp16/bf16
// 以位型 reinterpret 到 __half/__nv_bfloat16 升 float 计算再转回,与 cpu 参考
// 同语义(见 src/backends/cpu/kernels/elementwise.cpp)。launch 配置
// (grid/block 计算)集中在 compute_launch_config,不在每个 kernel 内重复
// (docs/backends/cuda.md 第6章)。
//
// M19 Task 5:在标量 kernel 之外新增宽访存路径(float4/__half2/__nv_bfloat162),
// 运行期按指针 16 字节对齐与 numel 可整除性判定启用与否,判据与 rationale 见
// 下方"向量化访存路径"注释块;数值语义与标量路径逐位一致,仅访存宽度不同。
//
// M22(批4 T4,§1.1 决议点A):add 额外支持 int32/int64(梯度累加安全网,整数
// 直加,不经 float 桥接——理由与 cpu 侧 add_cpu_kernel 独立成一份实现同款,
// 见 src/backends/cpu/kernels/elementwise.cpp 头注释);mul/relu/square/sigmoid/
// tanh/rsqrt 的 dtype 白名单不变(仍是 v0 浮点三档)。

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/stream.h>
#include <frame/ops/kernel_registry.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "../cuda_status.h"
#include "accum_load_store.cuh"
#include "launch_config.cuh"

namespace {

// device 端位型 <-> float 转换:同目录共享工具(铁律 5 收敛,M22 批4 判重,见
// accum_load_store.cuh 头注释)。
using frame::backends::cuda::elementwise_load;
using frame::backends::cuda::elementwise_store;

// ---------------------------------------------------------------------------
// 向量化访存路径(M19 Task 5)。启用判据是"指针是否 16 字节对齐"与"numel 是否
// 整除向量宽度"这两个与 dtype 值无关的运行期谓词(不是 CPP-012 意义上的 dtype
// 运行时分支——向量宽度 kVectorWidth<T> 是纯编译期特化值,由 dispatch_dtype
// 已经展开出的具体 T 决定,不在运行时按 dtype 取值分叉)。16 字节取自 float4
// 一次搬运 4 个 float 的载荷宽度;fp16/bf16 场景本可用更松的 4 字节门槛
// (__half2/__nv_bfloat162 各 2 元素共 4 字节),但本文件统一取 16 字节作唯一
// 对齐判据,判据更保守但形式统一、便于审计。分配器默认对齐
// (include/frame/core/storage.h::kDefaultAlignment=64)与 cudaMalloc 默认对齐
// 均远超 16 字节,故 Tensor::empty 分配出的整图张量恒可用宽访存;该判据存在
// 是为兜住未来可能出现的非默认对齐设备指针(如异形切片、外部内存导入)。
// ---------------------------------------------------------------------------

// 编译期 dtype -> 向量宽度(一次搬运的元素个数)。默认 0 表示不支持向量化;
// 调用处以 `if constexpr (kVectorWidth<T> > 0)` 判定,dispatch_dtype 仍会展开
// 到的其余 dtype(int8_t/bool 等)因宽度为 0,该分支被编译期丢弃,不产生任何
// 向量化 kernel 实例化。
template <typename T>
inline constexpr int kVectorWidth = 0;
template <>
inline constexpr int kVectorWidth<float> = 4;  // float4:4 个 float 共 16 字节
template <>
inline constexpr int kVectorWidth<frame::float16_t> = 2;  // __half2:2 个半精度元素
template <>
inline constexpr int kVectorWidth<frame::bfloat16_t> = 2;  // __nv_bfloat162:同上

// dtype -> 宽访存打包类型。仅为受支持的三种 dtype 特化;不支持的 dtype 因
// kVectorWidth 为 0,调用处的 if constexpr 分支永不实例化本 trait,故无需
// (也不应)提供主模板定义。
template <typename T>
struct VecPacked;
template <>
struct VecPacked<float> {
  using Type = float4;
};
template <>
struct VecPacked<frame::float16_t> {
  using Type = __half2;
};
template <>
struct VecPacked<frame::bfloat16_t> {
  using Type = __nv_bfloat162;
};

// 宽访存打包类型 -> 4 路 float 寄存器(fp16/bf16 只写前 2 路)。逐元素转换与
// elementwise_load 同语义(升 float 计算的既有惯例),只是一次转换 2/4 个
// 元素而非 1 个,不改变每个元素自身的转换结果。
template <typename T>
__device__ __forceinline__ void vec_unpack(const typename VecPacked<T>::Type& packed,
                                           float (&lanes)[4]) {
  if constexpr (std::is_same_v<T, float>) {
    lanes[0] = packed.x;
    lanes[1] = packed.y;
    lanes[2] = packed.z;
    lanes[3] = packed.w;
  } else if constexpr (std::is_same_v<T, frame::float16_t>) {
    const float2 pair = __half22float2(packed);
    lanes[0] = pair.x;
    lanes[1] = pair.y;
  } else {
    static_assert(std::is_same_v<T, frame::bfloat16_t>,
                  "vec_unpack only instantiated for float/float16_t/bfloat16_t");
    const float2 pair = __bfloat1622float2(packed);
    lanes[0] = pair.x;
    lanes[1] = pair.y;
  }
}

// 4 路 float 寄存器(fp16/bf16 只读前 2 路)-> 宽访存打包类型。转换 intrinsic
// (__floats2half2_rn/__floats2bfloat162_rn)在 sm_80 及以上架构(含本机实测
// 的 sm_120,见 docs/backends/cuda.md 第 8 章)上与
// elementwise_store 使用的逐元素 __float2half/__float2bfloat16 同为
// round-to-nearest-even 的同一条 cvt 指令语义,数值逐位一致(见 CUDA SDK
// cuda_fp16.hpp/cuda_bf16.hpp 中 __float2half/__floats2half2_rn 等函数的设备侧
// 内联汇编实现,均为 cvt.rn.{f16,bf16}{,x2}.f32)。
template <typename T>
__device__ __forceinline__ typename VecPacked<T>::Type vec_pack(const float (&lanes)[4]) {
  if constexpr (std::is_same_v<T, float>) {
    return make_float4(lanes[0], lanes[1], lanes[2], lanes[3]);
  } else if constexpr (std::is_same_v<T, frame::float16_t>) {
    return __floats2half2_rn(lanes[0], lanes[1]);
  } else {
    static_assert(std::is_same_v<T, frame::bfloat16_t>,
                  "vec_pack only instantiated for float/float16_t/bfloat16_t");
    return __floats2bfloat162_rn(lanes[0], lanes[1]);
  }
}

// ---------------------------------------------------------------------------
// 四个算子各自的 __global__ kernel(命名体现算子与 dtype 模板参数)。
// ---------------------------------------------------------------------------
template <typename T>
__global__ void add_kernel(const T* lhs, const T* rhs, T* out, int64_t numel) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= numel) return;
  elementwise_store(out, i, elementwise_load(lhs, i) + elementwise_load(rhs, i));
}

template <typename T>
__global__ void mul_kernel(const T* lhs, const T* rhs, T* out, int64_t numel) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= numel) return;
  elementwise_store(out, i, elementwise_load(lhs, i) * elementwise_load(rhs, i));
}

template <typename T>
__global__ void relu_kernel(const T* in, T* out, int64_t numel) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= numel) return;
  const float x = elementwise_load(in, i);
  elementwise_store(out, i, x > 0.0F ? x : 0.0F);
}

template <typename T>
__global__ void square_kernel(const T* in, T* out, int64_t numel) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= numel) return;
  const float x = elementwise_load(in, i);
  elementwise_store(out, i, x * x);
}

// sigmoid(x) = 1/(1+e^-x)(M21,批3 T5)。数值稳定式与 cpu 参考同语义(见
// src/backends/cpu/kernels/elementwise.cpp::sigmoid_cpu_kernel):x>=0 用
// 1/(1+e^-x)(e^-x 不上溢);x<0 改用 e^x/(1+e^x)(e^x 不上溢)。expf 为 CUDA
// 设备侧内建函数。
template <typename T>
__global__ void sigmoid_kernel(const T* in, T* out, int64_t numel) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= numel) return;
  const float x = elementwise_load(in, i);
  const float value = x >= 0.0F ? 1.0F / (1.0F + expf(-x)) : expf(x) / (1.0F + expf(x));
  elementwise_store(out, i, value);
}

// 代理阶跃前向固定 x>=0 ? 1 : 0;alpha 只控制注册的平滑代理梯度。
template <typename T>
__global__ void heaviside_surrogate_kernel(const T* in, T* out, int64_t numel) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= numel) return;
  const float x = elementwise_load(in, i);
  elementwise_store(out, i, x >= 0.0F ? 1.0F : 0.0F);
}

// tanh(x)(M22,批4 T4)。数值语义与 cpu 参考一致(std::tanh),设备侧用 CUDA
// 内建 tanhf。
template <typename T>
__global__ void tanh_kernel(const T* in, T* out, int64_t numel) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= numel) return;
  const float x = elementwise_load(in, i);
  elementwise_store(out, i, tanhf(x));
}

// rsqrt(x) = 1/sqrt(x)(M22,批4 T4,spec 外增项,见
// src/ops/schemas/elementwise.cpp::rsqrt_gradient 头注释)。与 cpu 参考同一
// 数值式(1.0F/std::sqrt(x)),不用 CUDA 近似内建 rsqrtf(该内建为快速近似,
// 精度低于 BUILD-011 容差假设的"与 cpu 参考同式"基准)。
template <typename T>
__global__ void rsqrt_kernel(const T* in, T* out, int64_t numel) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= numel) return;
  const float x = elementwise_load(in, i);
  elementwise_store(out, i, 1.0F / sqrtf(x));
}

// add 的整数直加 kernel(M22,批4 T4,§1.1 决议点A):int32_t/int64_t 精确整数
// 相加,不经 elementwise_load/elementwise_store 的 float 桥接(避免大整数精度
// 损失,与 cpu 侧 apply_add_elements 的 int32_t/int64_t 分支同语义)。仅在
// add_cuda_kernel 内经 if constexpr(T 为 int32_t/int64_t)分支调用,其余 dtype
// 不会实例化本模板。
template <typename T>
__global__ void add_kernel_int(const T* lhs, const T* rhs, T* out, int64_t numel) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= numel) return;
  out[i] = lhs[i] + rhs[i];
}

// ---------------------------------------------------------------------------
// 上面四个 kernel 的向量化访存版本:每线程一次处理 kVectorWidth<T> 个元素
// (一次宽访存代替 kVectorWidth<T> 次窄访存),内层逐元素计算与对应标量 kernel
// 完全一致,只改访存宽度,不改运算。
// ---------------------------------------------------------------------------
template <typename T>
__global__ void add_kernel_vec(const T* lhs, const T* rhs, T* out, int64_t vec_count) {
  using Packed = typename VecPacked<T>::Type;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= vec_count) return;
  float lhs_lanes[4];
  float rhs_lanes[4];
  vec_unpack<T>(reinterpret_cast<const Packed*>(lhs)[i], lhs_lanes);
  vec_unpack<T>(reinterpret_cast<const Packed*>(rhs)[i], rhs_lanes);
  float out_lanes[4];
#pragma unroll
  for (int lane = 0; lane < kVectorWidth<T>; ++lane) {
    out_lanes[lane] = lhs_lanes[lane] + rhs_lanes[lane];
  }
  reinterpret_cast<Packed*>(out)[i] = vec_pack<T>(out_lanes);
}

template <typename T>
__global__ void mul_kernel_vec(const T* lhs, const T* rhs, T* out, int64_t vec_count) {
  using Packed = typename VecPacked<T>::Type;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= vec_count) return;
  float lhs_lanes[4];
  float rhs_lanes[4];
  vec_unpack<T>(reinterpret_cast<const Packed*>(lhs)[i], lhs_lanes);
  vec_unpack<T>(reinterpret_cast<const Packed*>(rhs)[i], rhs_lanes);
  float out_lanes[4];
#pragma unroll
  for (int lane = 0; lane < kVectorWidth<T>; ++lane) {
    out_lanes[lane] = lhs_lanes[lane] * rhs_lanes[lane];
  }
  reinterpret_cast<Packed*>(out)[i] = vec_pack<T>(out_lanes);
}

template <typename T>
__global__ void relu_kernel_vec(const T* in, T* out, int64_t vec_count) {
  using Packed = typename VecPacked<T>::Type;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= vec_count) return;
  float lanes[4];
  vec_unpack<T>(reinterpret_cast<const Packed*>(in)[i], lanes);
#pragma unroll
  for (int lane = 0; lane < kVectorWidth<T>; ++lane) {
    lanes[lane] = lanes[lane] > 0.0F ? lanes[lane] : 0.0F;
  }
  reinterpret_cast<Packed*>(out)[i] = vec_pack<T>(lanes);
}

template <typename T>
__global__ void square_kernel_vec(const T* in, T* out, int64_t vec_count) {
  using Packed = typename VecPacked<T>::Type;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= vec_count) return;
  float lanes[4];
  vec_unpack<T>(reinterpret_cast<const Packed*>(in)[i], lanes);
#pragma unroll
  for (int lane = 0; lane < kVectorWidth<T>; ++lane) {
    lanes[lane] = lanes[lane] * lanes[lane];
  }
  reinterpret_cast<Packed*>(out)[i] = vec_pack<T>(lanes);
}

template <typename T>
__global__ void sigmoid_kernel_vec(const T* in, T* out, int64_t vec_count) {
  using Packed = typename VecPacked<T>::Type;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= vec_count) return;
  float lanes[4];
  vec_unpack<T>(reinterpret_cast<const Packed*>(in)[i], lanes);
#pragma unroll
  for (int lane = 0; lane < kVectorWidth<T>; ++lane) {
    const float x = lanes[lane];
    lanes[lane] = x >= 0.0F ? 1.0F / (1.0F + expf(-x)) : expf(x) / (1.0F + expf(x));
  }
  reinterpret_cast<Packed*>(out)[i] = vec_pack<T>(lanes);
}

// tanh(M22,批4 T4)向量化访存版本,同上四个 kernel 的宽访存改写手法。
template <typename T>
__global__ void tanh_kernel_vec(const T* in, T* out, int64_t vec_count) {
  using Packed = typename VecPacked<T>::Type;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= vec_count) return;
  float lanes[4];
  vec_unpack<T>(reinterpret_cast<const Packed*>(in)[i], lanes);
#pragma unroll
  for (int lane = 0; lane < kVectorWidth<T>; ++lane) {
    lanes[lane] = tanhf(lanes[lane]);
  }
  reinterpret_cast<Packed*>(out)[i] = vec_pack<T>(lanes);
}

// rsqrt(M22,批4 T4)向量化访存版本,同上。
template <typename T>
__global__ void rsqrt_kernel_vec(const T* in, T* out, int64_t vec_count) {
  using Packed = typename VecPacked<T>::Type;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= vec_count) return;
  float lanes[4];
  vec_unpack<T>(reinterpret_cast<const Packed*>(in)[i], lanes);
#pragma unroll
  for (int lane = 0; lane < kVectorWidth<T>; ++lane) {
    lanes[lane] = 1.0F / sqrtf(lanes[lane]);
  }
  reinterpret_cast<Packed*>(out)[i] = vec_pack<T>(lanes);
}

// launch 配置计算:同目录共享工具(铁律 5 收敛,M22 批4 判重,见
// launch_config.cuh 头注释)。
using frame::backends::cuda::compute_launch_config;
using frame::backends::cuda::LaunchConfig;

// 向量化访存路径可用性判定(共享 helper,供 add/mul/relu/square 四个 launcher
// 复用,铁律5:不逐算子复制)。width==0(dtype 不支持向量化)与 numel 不整除
// width 时直接回退标量;否则要求全部相关指针 16 字节对齐(判据依据见上方
// "向量化访存路径"注释块)。numel/对齐判断均与 dtype 取值本身无关,仅 width
// 常量随 T 在编译期确定,不构成 CPP-012 意义上的运行时 dtype 分支。
template <typename T>
bool can_use_vectorized_path(std::initializer_list<const void*> ptrs, int64_t numel) {
  constexpr int width = kVectorWidth<T>;
  if constexpr (width == 0) {
    return false;
  } else {
    if (numel % width != 0) return false;
    constexpr std::uintptr_t kAlignmentBytes = 16;
    for (const void* ptr : ptrs) {
      if (reinterpret_cast<std::uintptr_t>(ptr) % kAlignmentBytes != 0) return false;
    }
    return true;
  }
}

// ctx.stream 可空(eager 单算子路径的防御性兜底);非空时经 ARCH-030 白名单方法
// native_handle() 下沉到原生 cudaStream_t(本文件属 src/backends/,合法调用点)。
cudaStream_t native_stream(frame::hal::Stream* stream) {
  return stream != nullptr ? static_cast<cudaStream_t>(stream->native_handle()) : nullptr;
}

// dtype 白名单(v0 三档浮点,与 cpu 参考一致):fp32/fp16/bf16。
bool is_supported_elementwise_dtype(frame::DTypeCode code) {
  return code == frame::DTypeCode::kFloat32 || code == frame::DTypeCode::kFloat16 ||
         code == frame::DTypeCode::kBFloat16;
}

// add 专用 dtype 白名单(M22,批4 T4,§1.1 决议点A):v0 三档浮点之上扩
// int32/int64(梯度累加安全网)。mul/relu/square/sigmoid/tanh/rsqrt 仍只认
// is_supported_elementwise_dtype 三档,理由见文件头注释与
// validate_add_elementwise 头注释。
bool is_supported_add_dtype(frame::DTypeCode code) {
  return is_supported_elementwise_dtype(code) || code == frame::DTypeCode::kInt32 ||
         code == frame::DTypeCode::kInt64;
}

// ---------------------------------------------------------------------------
// 二元/一元算子共用的校验骨架(REUSE-002,与 src/backends/cpu/kernels/
// elementwise.cpp 的 binary_elementwise_cpu_kernel/unary_elementwise_cpu_kernel
// 同一份校验逻辑的 cuda 版本):输入/输出个数、dtype/shape 一致、dtype 白名单。
// ---------------------------------------------------------------------------
struct BinaryArgs {
  const void* lhs = nullptr;
  const void* rhs = nullptr;
  void* out = nullptr;
  int64_t numel = 0;
  frame::DTypeCode code = frame::DTypeCode::kFloat32;
};

frame::Result<BinaryArgs> validate_binary_elementwise(std::string_view op_name,
                                                      frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel expects 2 inputs, got " +
                                   std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel expects 1 output, got " +
                                   std::to_string(ctx.outputs.size()));
  }
  const frame::Tensor& lhs = ctx.inputs[0];
  const frame::Tensor& rhs = ctx.inputs[1];
  const frame::Tensor& out = ctx.outputs[0];
  // 布尔先落地为具名变量再判断(与 src/backends/cpu/kernels/elementwise.cpp
  // 同一惯例):check_iron_rules.sh 对 kernels/ 目录的 CPP-012 文本扫描按
  // `if (...dtype...)` 形态识别疑似运行时 dtype 分支,提前拆出具名变量可与
  // dispatch_dtype 编译期分派清晰区分,避免误判。
  const bool elem_type_mismatch = !(lhs.dtype() == rhs.dtype()) || !(lhs.dtype() == out.dtype());
  if (elem_type_mismatch) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel requires lhs/rhs/out of the same dtype, got '" +
                                   std::string(lhs.dtype().name()) + "', '" +
                                   std::string(rhs.dtype().name()) + "', '" +
                                   std::string(out.dtype().name()) + "'");
  }
  const bool shape_mismatch = !(lhs.shape() == rhs.shape()) || !(lhs.shape() == out.shape());
  if (shape_mismatch) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel requires lhs/rhs/out of the same shape, got " +
                                   lhs.shape().to_string() + ", " + rhs.shape().to_string() + ", " +
                                   out.shape().to_string());
  }
  const frame::DTypeCode code = lhs.dtype().code();
  const bool supported = is_supported_elementwise_dtype(code);
  if (!supported) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) + "' cuda kernel does not support dtype '" +
            std::string(lhs.dtype().name()) + "' (v0 supports float32/float16/bfloat16 only)");
  }
  const int64_t numel = lhs.numel();
  if (numel == 0) return BinaryArgs{nullptr, nullptr, nullptr, 0, code};
  return BinaryArgs{lhs.raw_data(), rhs.raw_data(), ctx.outputs[0].raw_data(), numel, code};
}

// add 专用校验(M22,批4 T4,§1.1 决议点A):不复用上面 validate_binary_elementwise
// ——二者的 dtype 白名单与错误消息不同(add 扩 int32/int64,mul 仍限三档浮点),
// 理由与 cpu 侧 add_cpu_kernel 独立成一份实现同款(见
// src/backends/cpu/kernels/elementwise.cpp 头注释:强行合并会让 mul 专属的
// dtype 契约被 add 的扩容掩盖)。其余校验逐条与 validate_binary_elementwise
// 同构(输入/输出个数、lhs/rhs/out 的 shape/dtype 一致)。
frame::Result<BinaryArgs> validate_add_elementwise(std::string_view op_name,
                                                   frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel expects 2 inputs, got " +
                                   std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel expects 1 output, got " +
                                   std::to_string(ctx.outputs.size()));
  }
  const frame::Tensor& lhs = ctx.inputs[0];
  const frame::Tensor& rhs = ctx.inputs[1];
  const frame::Tensor& out = ctx.outputs[0];
  const bool elem_type_mismatch = !(lhs.dtype() == rhs.dtype()) || !(lhs.dtype() == out.dtype());
  if (elem_type_mismatch) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel requires lhs/rhs/out of the same dtype, got '" +
                                   std::string(lhs.dtype().name()) + "', '" +
                                   std::string(rhs.dtype().name()) + "', '" +
                                   std::string(out.dtype().name()) + "'");
  }
  const bool shape_mismatch = !(lhs.shape() == rhs.shape()) || !(lhs.shape() == out.shape());
  if (shape_mismatch) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel requires lhs/rhs/out of the same shape, got " +
                                   lhs.shape().to_string() + ", " + rhs.shape().to_string() + ", " +
                                   out.shape().to_string());
  }
  const frame::DTypeCode code = lhs.dtype().code();
  const bool supported = is_supported_add_dtype(code);
  if (!supported) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel does not support dtype '" +
                                   std::string(lhs.dtype().name()) +
                                   "' (v0 supports float32/float16/bfloat16/int32/int64 only)");
  }
  const int64_t numel = lhs.numel();
  if (numel == 0) return BinaryArgs{nullptr, nullptr, nullptr, 0, code};
  return BinaryArgs{lhs.raw_data(), rhs.raw_data(), ctx.outputs[0].raw_data(), numel, code};
}

struct UnaryArgs {
  const void* in = nullptr;
  void* out = nullptr;
  int64_t numel = 0;
  frame::DTypeCode code = frame::DTypeCode::kFloat32;
};

frame::Result<UnaryArgs> validate_unary_elementwise(std::string_view op_name,
                                                    frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel expects 1 input, got " +
                                   std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel expects 1 output, got " +
                                   std::to_string(ctx.outputs.size()));
  }
  const frame::Tensor& in = ctx.inputs[0];
  const frame::Tensor& out = ctx.outputs[0];
  const bool elem_type_mismatch = !(in.dtype() == out.dtype());
  if (elem_type_mismatch) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) + "' cuda kernel requires x/out of the same dtype, got '" +
            std::string(in.dtype().name()) + "', '" + std::string(out.dtype().name()) + "'");
  }
  const bool shape_mismatch = !(in.shape() == out.shape());
  if (shape_mismatch) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel requires x/out of the same shape, got " +
                                   in.shape().to_string() + ", " + out.shape().to_string());
  }
  const frame::DTypeCode code = in.dtype().code();
  const bool supported = is_supported_elementwise_dtype(code);
  if (!supported) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) + "' cuda kernel does not support dtype '" +
            std::string(in.dtype().name()) + "' (v0 supports float32/float16/bfloat16 only)");
  }
  const int64_t numel = in.numel();
  if (numel == 0) return UnaryArgs{nullptr, nullptr, 0, code};
  return UnaryArgs{in.raw_data(), ctx.outputs[0].raw_data(), numel, code};
}

// CUDA kernel 边界独立校验代理阶跃的必需正有限 alpha。
frame::Status validate_heaviside_alpha(const frame::ops::KernelContext& ctx) {
  if (ctx.attrs == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'heaviside_surrogate' cuda kernel is missing required attribute 'alpha': no attrs "
        "provided");
  }
  const auto alpha_it = ctx.attrs->find("alpha");
  if (alpha_it == ctx.attrs->end()) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'heaviside_surrogate' cuda kernel is missing required attribute 'alpha'");
  }
  const double* alpha = std::get_if<double>(&alpha_it->second);
  if (alpha == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'heaviside_surrogate' cuda kernel attribute 'alpha' has the wrong type, expected "
        "double");
  }
  if (!std::isfinite(*alpha) || !(*alpha > 0.0)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'heaviside_surrogate' cuda kernel attribute 'alpha' must be finite and positive, "
        "got " +
            std::to_string(*alpha));
  }
  return frame::Status::ok();
}

frame::Status add_cuda_kernel(frame::ops::KernelContext& ctx) {
  const frame::Result<BinaryArgs> args = validate_add_elementwise("add", ctx);
  if (!args.is_ok()) return args.status();
  if (args.value().numel == 0) return frame::Status::ok();
  const LaunchConfig cfg = compute_launch_config(args.value().numel);
  const cudaStream_t stream = native_stream(ctx.stream);
  const BinaryArgs& a = args.value();
  return frame::dispatch_dtype(a.code, [&]<typename T>() -> frame::Status {
    // int32_t/int64_t(M22,批4 T4,§1.1 决议点A):精确整数直加,不入下方
    // float 向量化/标量分支(add_kernel/add_kernel_vec 经 elementwise_load/
    // elementwise_store 升 float,对大整数会精度损失)。
    if constexpr (std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::int64_t>) {
      add_kernel_int<T><<<cfg.grid, cfg.block, 0, stream>>>(static_cast<const T*>(a.lhs),
                                                            static_cast<const T*>(a.rhs),
                                                            static_cast<T*>(a.out), a.numel);
      return frame::backends::cuda::cuda_status(cudaGetLastError(),
                                                "add cuda kernel launch (integer)");
    } else {
      if constexpr (kVectorWidth<T> > 0) {
        if (can_use_vectorized_path<T>({a.lhs, a.rhs, a.out}, a.numel)) {
          const int64_t vec_count = a.numel / kVectorWidth<T>;
          const LaunchConfig vec_cfg = compute_launch_config(vec_count);
          add_kernel_vec<T><<<vec_cfg.grid, vec_cfg.block, 0, stream>>>(
              static_cast<const T*>(a.lhs), static_cast<const T*>(a.rhs), static_cast<T*>(a.out),
              vec_count);
          return frame::backends::cuda::cuda_status(cudaGetLastError(),
                                                    "add cuda kernel launch (vectorized)");
        }
      }
      add_kernel<T><<<cfg.grid, cfg.block, 0, stream>>>(static_cast<const T*>(a.lhs),
                                                        static_cast<const T*>(a.rhs),
                                                        static_cast<T*>(a.out), a.numel);
      return frame::backends::cuda::cuda_status(cudaGetLastError(), "add cuda kernel launch");
    }
  });
}

frame::Status mul_cuda_kernel(frame::ops::KernelContext& ctx) {
  const frame::Result<BinaryArgs> args = validate_binary_elementwise("mul", ctx);
  if (!args.is_ok()) return args.status();
  if (args.value().numel == 0) return frame::Status::ok();
  const LaunchConfig cfg = compute_launch_config(args.value().numel);
  const cudaStream_t stream = native_stream(ctx.stream);
  const BinaryArgs& a = args.value();
  return frame::dispatch_dtype(a.code, [&]<typename T>() -> frame::Status {
    if constexpr (kVectorWidth<T> > 0) {
      if (can_use_vectorized_path<T>({a.lhs, a.rhs, a.out}, a.numel)) {
        const int64_t vec_count = a.numel / kVectorWidth<T>;
        const LaunchConfig vec_cfg = compute_launch_config(vec_count);
        mul_kernel_vec<T><<<vec_cfg.grid, vec_cfg.block, 0, stream>>>(
            static_cast<const T*>(a.lhs), static_cast<const T*>(a.rhs), static_cast<T*>(a.out),
            vec_count);
        return frame::backends::cuda::cuda_status(cudaGetLastError(),
                                                  "mul cuda kernel launch (vectorized)");
      }
    }
    mul_kernel<T><<<cfg.grid, cfg.block, 0, stream>>>(static_cast<const T*>(a.lhs),
                                                      static_cast<const T*>(a.rhs),
                                                      static_cast<T*>(a.out), a.numel);
    return frame::backends::cuda::cuda_status(cudaGetLastError(), "mul cuda kernel launch");
  });
}

frame::Status relu_cuda_kernel(frame::ops::KernelContext& ctx) {
  const frame::Result<UnaryArgs> args = validate_unary_elementwise("relu", ctx);
  if (!args.is_ok()) return args.status();
  if (args.value().numel == 0) return frame::Status::ok();
  const LaunchConfig cfg = compute_launch_config(args.value().numel);
  const cudaStream_t stream = native_stream(ctx.stream);
  const UnaryArgs& a = args.value();
  return frame::dispatch_dtype(a.code, [&]<typename T>() -> frame::Status {
    if constexpr (kVectorWidth<T> > 0) {
      if (can_use_vectorized_path<T>({a.in, a.out}, a.numel)) {
        const int64_t vec_count = a.numel / kVectorWidth<T>;
        const LaunchConfig vec_cfg = compute_launch_config(vec_count);
        relu_kernel_vec<T><<<vec_cfg.grid, vec_cfg.block, 0, stream>>>(
            static_cast<const T*>(a.in), static_cast<T*>(a.out), vec_count);
        return frame::backends::cuda::cuda_status(cudaGetLastError(),
                                                  "relu cuda kernel launch (vectorized)");
      }
    }
    relu_kernel<T><<<cfg.grid, cfg.block, 0, stream>>>(static_cast<const T*>(a.in),
                                                       static_cast<T*>(a.out), a.numel);
    return frame::backends::cuda::cuda_status(cudaGetLastError(), "relu cuda kernel launch");
  });
}

frame::Status square_cuda_kernel(frame::ops::KernelContext& ctx) {
  const frame::Result<UnaryArgs> args = validate_unary_elementwise("square", ctx);
  if (!args.is_ok()) return args.status();
  if (args.value().numel == 0) return frame::Status::ok();
  const LaunchConfig cfg = compute_launch_config(args.value().numel);
  const cudaStream_t stream = native_stream(ctx.stream);
  const UnaryArgs& a = args.value();
  return frame::dispatch_dtype(a.code, [&]<typename T>() -> frame::Status {
    if constexpr (kVectorWidth<T> > 0) {
      if (can_use_vectorized_path<T>({a.in, a.out}, a.numel)) {
        const int64_t vec_count = a.numel / kVectorWidth<T>;
        const LaunchConfig vec_cfg = compute_launch_config(vec_count);
        square_kernel_vec<T><<<vec_cfg.grid, vec_cfg.block, 0, stream>>>(
            static_cast<const T*>(a.in), static_cast<T*>(a.out), vec_count);
        return frame::backends::cuda::cuda_status(cudaGetLastError(),
                                                  "square cuda kernel launch (vectorized)");
      }
    }
    square_kernel<T><<<cfg.grid, cfg.block, 0, stream>>>(static_cast<const T*>(a.in),
                                                         static_cast<T*>(a.out), a.numel);
    return frame::backends::cuda::cuda_status(cudaGetLastError(), "square cuda kernel launch");
  });
}

frame::Status sigmoid_cuda_kernel(frame::ops::KernelContext& ctx) {
  const frame::Result<UnaryArgs> args = validate_unary_elementwise("sigmoid", ctx);
  if (!args.is_ok()) return args.status();
  if (args.value().numel == 0) return frame::Status::ok();
  const LaunchConfig cfg = compute_launch_config(args.value().numel);
  const cudaStream_t stream = native_stream(ctx.stream);
  const UnaryArgs& a = args.value();
  return frame::dispatch_dtype(a.code, [&]<typename T>() -> frame::Status {
    if constexpr (kVectorWidth<T> > 0) {
      if (can_use_vectorized_path<T>({a.in, a.out}, a.numel)) {
        const int64_t vec_count = a.numel / kVectorWidth<T>;
        const LaunchConfig vec_cfg = compute_launch_config(vec_count);
        sigmoid_kernel_vec<T><<<vec_cfg.grid, vec_cfg.block, 0, stream>>>(
            static_cast<const T*>(a.in), static_cast<T*>(a.out), vec_count);
        return frame::backends::cuda::cuda_status(cudaGetLastError(),
                                                  "sigmoid cuda kernel launch (vectorized)");
      }
    }
    sigmoid_kernel<T><<<cfg.grid, cfg.block, 0, stream>>>(static_cast<const T*>(a.in),
                                                          static_cast<T*>(a.out), a.numel);
    return frame::backends::cuda::cuda_status(cudaGetLastError(), "sigmoid cuda kernel launch");
  });
}

frame::Status heaviside_surrogate_cuda_kernel(frame::ops::KernelContext& ctx) {
  const frame::Result<UnaryArgs> args = validate_unary_elementwise("heaviside_surrogate", ctx);
  if (!args.is_ok()) return args.status();
  const frame::Status alpha_status = validate_heaviside_alpha(ctx);
  if (!alpha_status.is_ok()) return alpha_status;
  if (args.value().numel == 0) return frame::Status::ok();

  const LaunchConfig cfg = compute_launch_config(args.value().numel);
  const cudaStream_t stream = native_stream(ctx.stream);
  const UnaryArgs& a = args.value();
  return frame::dispatch_dtype(a.code, [&]<typename T>() -> frame::Status {
    heaviside_surrogate_kernel<T><<<cfg.grid, cfg.block, 0, stream>>>(
        static_cast<const T*>(a.in), static_cast<T*>(a.out), a.numel);
    return frame::backends::cuda::cuda_status(cudaGetLastError(),
                                              "heaviside_surrogate cuda kernel launch");
  });
}

// tanh(M22,批4 T4):照抄 sigmoid_cuda_kernel 结构。
frame::Status tanh_cuda_kernel(frame::ops::KernelContext& ctx) {
  const frame::Result<UnaryArgs> args = validate_unary_elementwise("tanh", ctx);
  if (!args.is_ok()) return args.status();
  if (args.value().numel == 0) return frame::Status::ok();
  const LaunchConfig cfg = compute_launch_config(args.value().numel);
  const cudaStream_t stream = native_stream(ctx.stream);
  const UnaryArgs& a = args.value();
  return frame::dispatch_dtype(a.code, [&]<typename T>() -> frame::Status {
    if constexpr (kVectorWidth<T> > 0) {
      if (can_use_vectorized_path<T>({a.in, a.out}, a.numel)) {
        const int64_t vec_count = a.numel / kVectorWidth<T>;
        const LaunchConfig vec_cfg = compute_launch_config(vec_count);
        tanh_kernel_vec<T><<<vec_cfg.grid, vec_cfg.block, 0, stream>>>(
            static_cast<const T*>(a.in), static_cast<T*>(a.out), vec_count);
        return frame::backends::cuda::cuda_status(cudaGetLastError(),
                                                  "tanh cuda kernel launch (vectorized)");
      }
    }
    tanh_kernel<T><<<cfg.grid, cfg.block, 0, stream>>>(static_cast<const T*>(a.in),
                                                       static_cast<T*>(a.out), a.numel);
    return frame::backends::cuda::cuda_status(cudaGetLastError(), "tanh cuda kernel launch");
  });
}

// rsqrt(M22,批4 T4):照抄 sigmoid_cuda_kernel 结构。
frame::Status rsqrt_cuda_kernel(frame::ops::KernelContext& ctx) {
  const frame::Result<UnaryArgs> args = validate_unary_elementwise("rsqrt", ctx);
  if (!args.is_ok()) return args.status();
  if (args.value().numel == 0) return frame::Status::ok();
  const LaunchConfig cfg = compute_launch_config(args.value().numel);
  const cudaStream_t stream = native_stream(ctx.stream);
  const UnaryArgs& a = args.value();
  return frame::dispatch_dtype(a.code, [&]<typename T>() -> frame::Status {
    if constexpr (kVectorWidth<T> > 0) {
      if (can_use_vectorized_path<T>({a.in, a.out}, a.numel)) {
        const int64_t vec_count = a.numel / kVectorWidth<T>;
        const LaunchConfig vec_cfg = compute_launch_config(vec_count);
        rsqrt_kernel_vec<T><<<vec_cfg.grid, vec_cfg.block, 0, stream>>>(
            static_cast<const T*>(a.in), static_cast<T*>(a.out), vec_count);
        return frame::backends::cuda::cuda_status(cudaGetLastError(),
                                                  "rsqrt cuda kernel launch (vectorized)");
      }
    }
    rsqrt_kernel<T><<<cfg.grid, cfg.block, 0, stream>>>(static_cast<const T*>(a.in),
                                                        static_cast<T*>(a.out), a.numel);
    return frame::backends::cuda::cuda_status(cudaGetLastError(), "rsqrt cuda kernel launch");
  });
}

}  // namespace

FRAME_REGISTER_KERNEL("add", frame::kCudaBackendName, add_cuda_kernel);
FRAME_REGISTER_KERNEL("mul", frame::kCudaBackendName, mul_cuda_kernel);
FRAME_REGISTER_KERNEL("relu", frame::kCudaBackendName, relu_cuda_kernel);
FRAME_REGISTER_KERNEL("square", frame::kCudaBackendName, square_cuda_kernel);
FRAME_REGISTER_KERNEL("sigmoid", frame::kCudaBackendName, sigmoid_cuda_kernel);
FRAME_REGISTER_KERNEL("heaviside_surrogate", frame::kCudaBackendName,
                      heaviside_surrogate_cuda_kernel);
FRAME_REGISTER_KERNEL("tanh", frame::kCudaBackendName, tanh_cuda_kernel);
FRAME_REGISTER_KERNEL("rsqrt", frame::kCudaBackendName, rsqrt_cuda_kernel);
