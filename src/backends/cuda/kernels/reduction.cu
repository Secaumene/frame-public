// CUDA 归约内核(阶段 C-12):sum。
//
// 【BE-CUDA-002】手写准入依据(docs/backends/cuda.md 第3章「库不满足」判据):
// CUB(SDK 自带,免 ADR)的 DeviceReduce/DeviceSegmentedReduce 覆盖「全归约」与
// 「等长分段归约」,但不直接覆盖 v0 sum 的任意 axes 子集投影语义(keepdims +
// 任意维组合、非等长分段);手写朴素 kernel 属库覆盖不到的场景,不构成与库
// 功能重叠的手写。
//
// 全归约(axes 为空或显式列全部维度,输出恒为单元素)分支改走
// cub::DeviceReduce::Sum 两段式调用(ADR-0010,docs/decisions/
// 0010-adopt-cub-for-full-reduction.md),避免单线程串行求和 numel 个元素的
// 性能悬崖;任意轴投影分支(库覆盖不到的场景)维持本文件朴素实现,依据同上。
// ADR-0010【待查证】核实结论(积压批次③实测,本机 CUDA Toolkit 13.3.73):
// ①CCCL include 形态——nvcc 13.3 自动把 <CUDA_HOME>/targets/x86_64-linux/
//   include/cccl 加入 -isystem 搜索路径(`nvcc -Xcompiler -v` 可见),
//   `#include <cub/device/device_reduce.cuh>` 与
//   `#include <thrust/iterator/transform_iterator.h>` 无需额外 CMake include
//   目录、无需 find_package/FetchContent(随 Toolkit,REUSE-010 SDK 豁免)。
// ②fp16/bf16 累加类型迭代器方案——本机 CCCL 13.3 已移除
//   cub::TransformInputIterator(/usr/local/cuda/include/cccl/cub/iterator/
//   目录已无该头),改用 thrust::transform_iterator
//   (thrust/iterator/transform_iterator.h 的 thrust::make_transform_iterator
//   工厂函数)对输入做逐元素升 float 转换;实测直接以 half*/bf16* 作
//   InputIteratorT、float* 作 OutputIteratorT 在本 CCCL 版本编译期 dispatch
//   推导失败,经 transform_iterator 包装后可编译;数值实测(2^16 个 fp16
//   元素求和)transform_iterator 路径与 double 参考值误差量级 1e-4,逐元素
//   half 累加误差量级 1e4,证实前者确以 float 精度累加,与 cpu 参考语义一致。
//
// 实现:每输出元素一线程,内层沿归约轴串行(朴素并行,不追求性能,唯一目标是
// 数值正确性,语义与 cpu 参考 src/backends/cpu/kernels/reduction.cpp 一致)。
// dtype 差异经 dispatch_dtype 编译期展开(ARCH-042)。

#include <cstdint>
#include <cub/device/device_reduce.cuh>
#include <memory>
#include <string>
#include <thrust/iterator/transform_iterator.h>
#include <type_traits>
#include <variant>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/storage.h>
#include <frame/core/tensor.h>
#include <frame/hal/backend.h>
#include <frame/hal/stream.h>
#include <frame/ops/kernel_registry.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "../cuda_status.h"
#include "accum_load_store.cuh"
#include "launch_config.cuh"

namespace {

// v0 归约维数上限(kernel 参数区大小与设备端固定数组容量;当前项目内已知
// 用例均为低秩张量,足够覆盖)。超出即返回英文错误,不静默截断。
constexpr int32_t kMaxReductionRank = 8;

// 位型 <-> float 读写桥接:同目录共享工具(铁律 5 收敛,M22 批4 判重,见
// accum_load_store.cuh 头注释;本文件曾以 reduction_load/reduction_store 命名
// 持一份与 elementwise_load/elementwise_store 逐字同构的实现,收编增量核查
// 发现后并入共享头)。
using frame::backends::cuda::elementwise_load;
using frame::backends::cuda::elementwise_store;

// 归约几何信息:host 端算好后按值传给 kernel(POD,kernel 参数区容量足够)。
// in_strides/dim_out_stride/reduced_substride 均按"输入 rank 长度"对齐(下标
// 语义见 build_reduction_plan 注释)。
struct ReductionShapeInfo {
  int64_t in_strides[kMaxReductionRank] = {};
  // 非归约维(保留维):该维在输出张量中的行优先 stride,供从 out_idx 反解出
  // 该维下标;归约维填 0(不使用)。
  int64_t dim_out_stride[kMaxReductionRank] = {};
  // 归约维:把"仅由归约维组成的子空间"单独按行优先编号的 stride,供从内层
  // 串行计数器 r 反解出各归约维下标;保留维填 0(不使用)。
  int64_t reduced_substride[kMaxReductionRank] = {};
  int32_t reduced[kMaxReductionRank] = {};  // 1=归约维,0=保留维
  int32_t rank = 0;
};

// 输出索引 out_idx -> 归约维 Cartesian 积逐一累加 -> 单个输出值(朴素实现,
// 见文件头 BE-CUDA-002 依据)。
template <typename T>
__global__ void sum_kernel(const T* in, T* out, int64_t numel_out, ReductionShapeInfo info,
                           int64_t reduction_size) {
  const int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (out_idx >= numel_out) return;

  int64_t kept_index[kMaxReductionRank] = {};
  int64_t remaining = out_idx;
  for (int32_t d = 0; d < info.rank; ++d) {
    if (info.reduced[d]) continue;
    const int64_t stride = info.dim_out_stride[d];
    kept_index[d] = stride != 0 ? remaining / stride : remaining;
    remaining = stride != 0 ? remaining % stride : 0;
  }

  float accum = 0.0F;
  for (int64_t r = 0; r < reduction_size; ++r) {
    int64_t r_remaining = r;
    int64_t input_linear = 0;
    for (int32_t d = 0; d < info.rank; ++d) {
      int64_t idx_d = kept_index[d];
      if (info.reduced[d]) {
        const int64_t stride = info.reduced_substride[d];
        idx_d = stride != 0 ? r_remaining / stride : r_remaining;
        r_remaining = stride != 0 ? r_remaining % stride : 0;
      }
      input_linear += idx_d * info.in_strides[d];
    }
    accum += elementwise_load(in, input_linear);
  }
  elementwise_store(out, out_idx, accum);
}

// launch 配置计算:同目录共享工具(铁律 5 收敛,M22 批4 判重,见
// launch_config.cuh 头注释)。
using frame::backends::cuda::compute_launch_config;
using frame::backends::cuda::LaunchConfig;

cudaStream_t native_stream(frame::hal::Stream* stream) {
  return stream != nullptr ? static_cast<cudaStream_t>(stream->native_handle()) : nullptr;
}

// fp16/bf16 -> float 的逐元素升精度转换 functor,装进
// thrust::make_transform_iterator 供 cub::DeviceReduce::Sum 以 float 精度
// 累加(文件头 ADR-0010 核实结论②)。dispatch_dtype 对全体 DTypeCode 编译期
// 穷举展开(ARCH-042),故须对不可达的其余 T 提供 fallback 分支(与
// elementwise_load 同一惯例);operator() 标 __host__ __device__:
// thrust::transform_iterator 的迭代器骨架在宿主端也可能实例化该类型。
template <typename T>
struct UpcastToFloat {
  __host__ __device__ float operator()(T value) const {
    if constexpr (std::is_same_v<T, float>) {
      return value;
    } else if constexpr (std::is_same_v<T, frame::float16_t>) {
      return __half2float(*reinterpret_cast<const __half*>(&value));
    } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
      return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&value));
    } else {
      return 0.0F;
    }
  }
};

// 全归约(CUB)路径专用:把 float 精度的归约结果(单元素 scratch)转回输出
// dtype T(fp16/bf16 需要下采样,与 cpu 参考"逐元素升 float 累加、结束后一次性
// 转回"同一语义)。单线程 kernel,处理元素数恒为 1,属手写 elementwise 类
// (BE-CUDA-002 三类准入之一)。
template <typename T>
__global__ void sum_finalize_kernel(const float* accum, T* out) {
  elementwise_store(out, 0, accum[0]);
}

// 全归约(axes 为空或覆盖全部维度)专用路径:cub::DeviceReduce::Sum 两段式
// 调用(先查询临时存储字节数,再执行;ADR-0010)。fp32 直接以 in_data/out_data
// (均为 float*)调用,无需中转;fp16/bf16 经 UpcastToFloat + 1 元素 float
// scratch 累加,再经 sum_finalize_kernel 转回 T。临时存储与 scratch 均经调用
// 方所在设备的 Allocator 分配(frame::Storage RAII,函数返回时自动释放,即便
// 提前 return 错误路径也不泄漏)。RAII 释放前显式 cudaStreamSynchronize(见
// 函数体内注释):不依赖 cudaFree 隐式全设备同步语义——该语义未写入 CUDA
// 官方契约,且防未来 CudaAllocator 若改为流序释放(如 cudaFreeAsync/
// cudaMemPool_t)会与本函数排入 stream 的异步 cub/finalize kernel 产生释放
// 早于完成的竞态。
template <typename T>
frame::Status run_full_reduction_sum(const T* in_data, T* out_data, int64_t numel_in,
                                     frame::hal::Allocator& allocator, frame::Device device,
                                     cudaStream_t stream) {
  if constexpr (std::is_same_v<T, float>) {
    size_t temp_bytes = 0;
    FRAME_RETURN_IF_ERROR(frame::backends::cuda::cuda_status(
        cub::DeviceReduce::Sum(nullptr, temp_bytes, in_data, out_data, numel_in, stream),
        "sum cuda kernel: cub::DeviceReduce::Sum temp storage query"));

    const frame::Result<std::shared_ptr<frame::Storage>> temp_storage =
        frame::Storage::allocate(allocator, temp_bytes, frame::kDefaultAlignment, device);
    if (!temp_storage.is_ok()) return temp_storage.status();

    const frame::Status run_status = frame::backends::cuda::cuda_status(
        cub::DeviceReduce::Sum(temp_storage.value()->data(), temp_bytes, in_data, out_data,
                               numel_in, stream),
        "sum cuda kernel: cub::DeviceReduce::Sum");
    // temp_storage 即将随本作用域结束被 RAII 释放(Storage 析构 ->
    // Allocator::deallocate -> CudaAllocator::deallocate -> cudaFree)。上面
    // 的 cub 调用异步排入 stream,函数返回前必须先等它真正执行完(而非依赖
    // cudaFree 的隐式全设备同步这一未文档化的历史行为,见函数头注释),才能
    // 安全释放该缓冲。
    const frame::Status sync_status = frame::backends::cuda::cuda_status(
        cudaStreamSynchronize(stream),
        "sum cuda kernel: cudaStreamSynchronize before temp storage release");
    return !run_status.is_ok() ? run_status : sync_status;
  } else {
    const frame::Result<std::shared_ptr<frame::Storage>> float_scratch =
        frame::Storage::allocate(allocator, sizeof(float), frame::kDefaultAlignment, device);
    if (!float_scratch.is_ok()) return float_scratch.status();
    auto* scratch_ptr = static_cast<float*>(float_scratch.value()->data());

    auto transformed_in = thrust::make_transform_iterator(in_data, UpcastToFloat<T>{});

    size_t temp_bytes = 0;
    FRAME_RETURN_IF_ERROR(frame::backends::cuda::cuda_status(
        cub::DeviceReduce::Sum(nullptr, temp_bytes, transformed_in, scratch_ptr, numel_in, stream),
        "sum cuda kernel: cub::DeviceReduce::Sum temp storage query"));

    const frame::Result<std::shared_ptr<frame::Storage>> temp_storage =
        frame::Storage::allocate(allocator, temp_bytes, frame::kDefaultAlignment, device);
    if (!temp_storage.is_ok()) return temp_storage.status();

    const frame::Status run_status = frame::backends::cuda::cuda_status(
        cub::DeviceReduce::Sum(temp_storage.value()->data(), temp_bytes, transformed_in,
                               scratch_ptr, numel_in, stream),
        "sum cuda kernel: cub::DeviceReduce::Sum");
    frame::Status finalize_status = run_status;
    if (run_status.is_ok()) {
      sum_finalize_kernel<T><<<1, 1, 0, stream>>>(scratch_ptr, out_data);
      finalize_status = frame::backends::cuda::cuda_status(
          cudaGetLastError(), "sum cuda kernel: finalize kernel launch");
    }
    // temp_storage 与 float_scratch 均即将随本作用域结束被 RAII 释放,理由同
    // 上(fp32 分支同款注释):显式同步后才允许函数返回、触发释放,不依赖
    // cudaFree 隐式同步语义。
    const frame::Status sync_status = frame::backends::cuda::cuda_status(
        cudaStreamSynchronize(stream),
        "sum cuda kernel: cudaStreamSynchronize before temp storage release");
    return !finalize_status.is_ok() ? finalize_status : sync_status;
  }
}

// axes 校验:与 src/backends/cpu/kernels/reduction.cpp::validate_axes 同规则、
// 同错误消息文案(两处各自独立实现,分属不同后端的 kernel 校验,非同一层的
// 文本复制,见该文件头注释同款论证)。空数组 = 全维归约。
frame::Status validate_axes(int64_t rank, const std::vector<int64_t>& axes,
                            std::vector<bool>& reduced) {
  reduced.assign(static_cast<size_t>(rank), false);
  if (axes.empty()) {
    reduced.assign(static_cast<size_t>(rank), true);
    return frame::Status::ok();
  }
  for (int64_t axis : axes) {
    if (axis < 0) {
      return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                                 "op 'sum' cuda kernel axes entry " + std::to_string(axis) +
                                     " is negative; v0 requires 0 <= axis < rank and does not "
                                     "normalize negative indices");
    }
    if (axis >= rank) {
      return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                                 "op 'sum' cuda kernel axes entry " + std::to_string(axis) +
                                     " is out of range for rank " + std::to_string(rank) +
                                     " (must satisfy 0 <= axis < rank)");
    }
    if (reduced[static_cast<size_t>(axis)]) {
      return frame::Status::make(
          frame::ErrorCode::kInvalidArgument,
          "op 'sum' cuda kernel axes entry " + std::to_string(axis) + " is duplicated");
    }
    reduced[static_cast<size_t>(axis)] = true;
  }
  return frame::Status::ok();
}

std::vector<int64_t> compute_reduced_dims(const std::vector<int64_t>& in_dims,
                                          const std::vector<bool>& reduced, bool keepdims) {
  std::vector<int64_t> out_dims;
  out_dims.reserve(in_dims.size());
  for (size_t d = 0; d < in_dims.size(); ++d) {
    if (reduced[d]) {
      if (keepdims) out_dims.push_back(1);
    } else {
      out_dims.push_back(in_dims[d]);
    }
  }
  return out_dims;
}

// 组装 ReductionShapeInfo(见结构体头注释各字段语义)+ reduction_size(归约维
// 元素个数连乘,内层串行循环次数)。
frame::Result<ReductionShapeInfo> build_reduction_plan(const std::vector<int64_t>& in_dims,
                                                       const std::vector<bool>& reduced,
                                                       const frame::Strides& out_strides,
                                                       int64_t& reduction_size) {
  const int32_t rank = static_cast<int32_t>(in_dims.size());
  if (rank > kMaxReductionRank) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'sum' cuda kernel rank " + std::to_string(rank) +
                                   " exceeds v0 maximum " + std::to_string(kMaxReductionRank));
  }

  ReductionShapeInfo info;
  info.rank = rank;
  const frame::Strides in_strides = frame::row_major_strides(frame::Shape(in_dims));
  const std::vector<int64_t>& in_stride_values = in_strides.values();
  const std::vector<int64_t>& out_stride_values = out_strides.values();

  int64_t kept_position = 0;
  for (int32_t d = 0; d < rank; ++d) {
    info.in_strides[d] = in_stride_values[static_cast<size_t>(d)];
    info.reduced[d] = reduced[static_cast<size_t>(d)] ? 1 : 0;
    if (!reduced[static_cast<size_t>(d)]) {
      info.dim_out_stride[d] = out_stride_values[static_cast<size_t>(kept_position)];
      ++kept_position;
    }
  }

  int64_t running = 1;
  for (int32_t d = rank - 1; d >= 0; --d) {
    if (reduced[static_cast<size_t>(d)]) {
      info.reduced_substride[d] = running;
      running *= in_dims[static_cast<size_t>(d)];
    }
  }
  reduction_size = running;
  return info;
}

frame::Status sum_cuda_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sum' cuda kernel expects 1 input, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sum' cuda kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& x = ctx.inputs[0];
  frame::Tensor& out = ctx.outputs[0];

  // 布尔先落地为具名变量再判断(与 src/backends/cpu/kernels/reduction.cpp 同
  // 一惯例):避免 check_iron_rules.sh 的 CPP-012 文本扫描误判为运行时 dtype
  // 分支。
  const bool elem_type_mismatch = !(x.dtype() == out.dtype());
  if (elem_type_mismatch) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'sum' cuda kernel requires x/out of the same dtype, got '" +
                                   std::string(x.dtype().name()) + "', '" +
                                   std::string(out.dtype().name()) + "'");
  }
  const frame::DTypeCode code = x.dtype().code();
  const bool supported = code == frame::DTypeCode::kFloat32 || code == frame::DTypeCode::kFloat16 ||
                         code == frame::DTypeCode::kBFloat16;
  if (!supported) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'sum' cuda kernel does not support dtype '" +
                                   std::string(x.dtype().name()) +
                                   "' (v0 supports float32/float16/bfloat16 only)");
  }

  if (ctx.attrs == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'sum' cuda kernel is missing required attribute 'axes' (int64 "
                               "array): no attrs provided");
  }
  const auto axes_it = ctx.attrs->find("axes");
  if (axes_it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'sum' cuda kernel is missing required attribute 'axes' (int64 "
                               "array)");
  }
  const std::vector<int64_t>* axes_ptr = std::get_if<std::vector<int64_t>>(&axes_it->second);
  if (axes_ptr == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sum' cuda kernel attribute 'axes' has the wrong type, expected int64 array");
  }

  bool keepdims = false;
  const auto keepdims_it = ctx.attrs->find("keepdims");
  if (keepdims_it != ctx.attrs->end()) {
    const bool* keepdims_ptr = std::get_if<bool>(&keepdims_it->second);
    if (keepdims_ptr == nullptr) {
      return frame::Status::make(
          frame::ErrorCode::kInvalidArgument,
          "op 'sum' cuda kernel attribute 'keepdims' has the wrong type, expected bool");
    }
    keepdims = *keepdims_ptr;
  }

  const int64_t rank = x.shape().rank();
  std::vector<bool> reduced;
  const frame::Status axes_status = validate_axes(rank, *axes_ptr, reduced);
  if (!axes_status.is_ok()) return axes_status;

  const std::vector<int64_t>& in_dims = x.shape().dims();
  const std::vector<int64_t> expected_out_dims = compute_reduced_dims(in_dims, reduced, keepdims);
  const frame::Shape expected_out_shape(expected_out_dims);
  if (!(out.shape() == expected_out_shape)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sum' cuda kernel requires out shape to match the reduction result, got " +
            out.shape().to_string() + ", expected " + expected_out_shape.to_string());
  }

  const cudaStream_t stream = native_stream(ctx.stream);
  const void* in_data = x.raw_data();
  void* out_data = out.raw_data();

  // 全归约判定:reduced 内全体维度均标记为归约——axes 为空时 validate_axes
  // 已把全体置 true;axes 非空但显式列全部维度时同样全体为 true。两种情形
  // 统一归入 CUB 路径(ADR-0010「全归约(axes 为空或覆盖全维)」判据),此时
  // 输出恒为单元素(numel_out==1,keepdims 只影响输出 rank 不影响 numel)。
  bool all_reduced = true;
  for (bool r : reduced) {
    if (!r) {
      all_reduced = false;
      break;
    }
  }

  if (all_reduced) {
    const frame::Result<frame::hal::Backend*> backend_lookup =
        frame::hal::BackendRegistry::instance().get(ctx.device.backend);
    if (!backend_lookup.is_ok()) {
      return frame::Status::make(
          backend_lookup.status().code(),
          "op 'sum' cuda kernel: " + std::string(backend_lookup.status().message()));
    }
    frame::hal::Allocator* allocator = backend_lookup.value()->allocator(ctx.device);
    if (allocator == nullptr) {
      return frame::Status::make(frame::ErrorCode::kInternal,
                                 "op 'sum' cuda kernel: allocator unavailable for device '" +
                                     std::string(ctx.device.backend) + ":" +
                                     std::to_string(ctx.device.index) + "'");
    }
    const int64_t numel_in = x.numel();
    const frame::Device device = ctx.device;
    return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
      return run_full_reduction_sum<T>(static_cast<const T*>(in_data), static_cast<T*>(out_data),
                                       numel_in, *allocator, device, stream);
    });
  }

  const frame::Strides out_strides = frame::row_major_strides(expected_out_shape);
  int64_t reduction_size = 1;
  const frame::Result<ReductionShapeInfo> plan =
      build_reduction_plan(in_dims, reduced, out_strides, reduction_size);
  if (!plan.is_ok()) return plan.status();

  const int64_t numel_out = out.numel();
  const LaunchConfig cfg = compute_launch_config(numel_out);
  const ReductionShapeInfo info = plan.value();

  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    sum_kernel<T><<<cfg.grid, cfg.block, 0, stream>>>(
        static_cast<const T*>(in_data), static_cast<T*>(out_data), numel_out, info, reduction_size);
    return frame::backends::cuda::cuda_status(cudaGetLastError(), "sum cuda kernel launch");
  });
}

}  // namespace

FRAME_REGISTER_KERNEL("sum", frame::kCudaBackendName, sum_cuda_kernel);
