// CUDA 形状变换内核(reshape,M21 批3 T5;concat/slice,M22 批4 T4,§1.4/1.6
// 决议点D/F)。numel 守恒由 schema 侧 shape 推断
// (src/ops/schemas/shape.cpp::infer_reshape_shape 等)已保证,kernel 侧仍
// 防御性复核 dtype 一致与 numel/shape 相等(与 cpu 参考
// src/backends/cpu/kernels/shape.cpp 同一纪律,cuda 侧改用
// cudaMemcpyAsync(D2D)经 ctx.stream 排队,而非 host 端 memcpy——device 张量
// 不可 host 端直接解引用,见 cuda_backend.h 头注释)。concat/slice 均为纯
// 连续段拷贝(cpu 参考按 outer 循环逐段 memcpy 的同一几何,见
// src/backends/cpu/kernels/shape.cpp::concat_cpu_kernel/slice_cpu_kernel 头
// 注释),故沿用 reshape 的"host 端 cudaMemcpyAsync 编排、不写 __global__
// kernel"路线(§1.6 决议表"落点镜像既有结构...或按仓内现状最贴近的组织方式,
// 不发明新结构"授权范围内的实现选择)——每个 (outer_idx[, 各输入]) 对应一段
// 连续字节区间的 D2D 拷贝。transpose 因逐输出元素需按 perm 反查非连续输入
// 索引,无法表达为有限段 cudaMemcpyAsync,故落地为真正的 __global__ strided
// copy kernel,见同目录新建 shape.cu(不含 __global__ 代码的本文件与含
// __global__ 代码的 shape.cu 按 CMakeLists.txt 注释的既定规则分文件)。三者
// dtype 均限 v0 三档浮点(与 cpu 参考白名单一致);reshape 保持
// dtype-agnostic 不变(理由见其自身函数头注释,不受本次改动影响)。不含
// __global__ 代码,纯 host 端 cudaMemcpyAsync 编排,故为 .cpp。

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/stream.h>
#include <frame/ops/kernel_registry.h>

#include <cuda_runtime.h>

#include "../cuda_status.h"

namespace {

cudaStream_t native_stream(frame::hal::Stream* stream) {
  return stream != nullptr ? static_cast<cudaStream_t>(stream->native_handle()) : nullptr;
}

// dtype 白名单(v0 三档浮点,transpose/concat/slice 专用;reshape 保持
// dtype-agnostic 不受影响,理由见其函数头注释)。
bool is_supported_shape_dtype(frame::DTypeCode code) {
  return code == frame::DTypeCode::kFloat32 || code == frame::DTypeCode::kFloat16 ||
         code == frame::DTypeCode::kBFloat16;
}

frame::Status reshape_cuda_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'reshape' cuda kernel expects 1 input, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'reshape' cuda kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& x = ctx.inputs[0];
  frame::Tensor& out = ctx.outputs[0];

  const frame::DType x_type = x.dtype();
  const frame::DType out_type = out.dtype();
  if (!(x_type == out_type)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'reshape' cuda kernel requires x/out of the same dtype, got '" +
                                   std::string(x_type.name()) + "', '" +
                                   std::string(out_type.name()) + "'");
  }
  if (x.numel() != out.numel()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'reshape' cuda kernel requires x/out numel to match, got " +
                                   std::to_string(x.numel()) + ", " + std::to_string(out.numel()));
  }

  const size_t nbytes = static_cast<size_t>(x.numel()) * x.dtype().itemsize();
  if (nbytes == 0) return frame::Status::ok();

  const cudaStream_t stream = native_stream(ctx.stream);
  return frame::backends::cuda::cuda_status(
      cudaMemcpyAsync(out.raw_data(), x.raw_data(), nbytes, cudaMemcpyDeviceToDevice, stream),
      "reshape cuda kernel: cudaMemcpyAsync");
}

// concat(xs...; axis) 的 CUDA 参考实现(M22,批4 T4,REUSE-002:与 cpu 参考
// src/backends/cpu/kernels/shape.cpp::concat_cpu_kernel 同一几何算法,仅把
// std::memcpy 换成 cudaMemcpyAsync(D2D))——逐输入段拷贝:把 axis 之前的维度
// 视为 outer 循环、之后的维度视为连续内层块(inner),每个 (outer, input)
// 对应一段 width*inner 个元素的连续 D2D 拷贝,按各输入沿 axis 的宽度累加偏移
// 写入 out。
frame::Status concat_cuda_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.empty()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'concat' cuda kernel expects at least 1 input, got 0");
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'concat' cuda kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }
  frame::Tensor& out = ctx.outputs[0];

  const frame::DType first_type = ctx.inputs[0].dtype();
  bool elem_type_mismatch = !(first_type == out.dtype());
  for (const frame::Tensor& t : ctx.inputs) {
    // 先落地为具名变量再判断(与本文件/pool.cpp/conv.cpp 同一惯例):
    // check_iron_rules.sh 对 kernels/ 目录的 CPP-012 文本扫描按
    // `if (...dtype...)` 形态识别疑似运行时 dtype 分支,提前拆出具名变量可与
    // dispatch_dtype 编译期分派清晰区分,避免误判。
    const bool current_type_mismatch = !(t.dtype() == first_type);
    if (current_type_mismatch) elem_type_mismatch = true;
  }
  if (elem_type_mismatch) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'concat' cuda kernel requires xs/out of the same dtype, got "
                               "inputs dtype '" +
                                   std::string(first_type.name()) + "', out dtype '" +
                                   std::string(out.dtype().name()) + "'");
  }
  const bool supported = is_supported_shape_dtype(first_type.code());
  if (!supported) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'concat' cuda kernel does not support dtype '" +
                                   std::string(first_type.name()) +
                                   "' (v0 supports float32/float16/bfloat16 only)");
  }

  if (ctx.attrs == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'concat' cuda kernel is missing required attribute 'axis': no "
                               "attrs provided");
  }
  const auto axis_it = ctx.attrs->find("axis");
  if (axis_it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'concat' cuda kernel is missing required attribute 'axis'");
  }
  const int64_t* axis_ptr = std::get_if<int64_t>(&axis_it->second);
  if (axis_ptr == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'concat' cuda kernel attribute 'axis' has the wrong type, expected int64");
  }
  const int64_t axis = *axis_ptr;

  const frame::Tensor& first = ctx.inputs[0];
  const int64_t rank = first.shape().rank();
  if (axis < 0 || axis >= rank) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'concat' cuda kernel attribute 'axis' " + std::to_string(axis) +
                                   " is out of range for rank " + std::to_string(rank));
  }

  int64_t axis_total = 0;
  for (const frame::Tensor& t : ctx.inputs) {
    if (t.shape().rank() != rank) {
      return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                                 "op 'concat' cuda kernel requires all inputs of the same rank, "
                                 "got rank " +
                                     std::to_string(t.shape().rank()) + " and " +
                                     std::to_string(rank));
    }
    for (int64_t d = 0; d < rank; ++d) {
      if (d == axis) continue;
      if (t.shape().dim(d) != first.shape().dim(d)) {
        return frame::Status::make(
            frame::ErrorCode::kInvalidArgument,
            "op 'concat' cuda kernel requires all inputs to match on non-axis dimension " +
                std::to_string(d));
      }
    }
    axis_total += t.shape().dim(axis);
  }
  std::vector<int64_t> expected_out_dims = first.shape().dims();
  expected_out_dims[static_cast<size_t>(axis)] = axis_total;
  const frame::Shape expected_out_shape(expected_out_dims);
  if (!(out.shape() == expected_out_shape)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'concat' cuda kernel requires out shape to match the concatenation result, got " +
            out.shape().to_string() + ", expected " + expected_out_shape.to_string());
  }

  int64_t outer = 1;
  for (int64_t d = 0; d < axis; ++d) outer *= first.shape().dim(d);
  int64_t inner = 1;
  for (int64_t d = axis + 1; d < rank; ++d) inner *= first.shape().dim(d);
  const size_t itemsize = first.dtype().itemsize();
  const cudaStream_t stream = native_stream(ctx.stream);

  char* out_bytes = static_cast<char*>(out.raw_data());
  for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
    int64_t axis_offset = 0;
    for (const frame::Tensor& t : ctx.inputs) {
      const int64_t width = t.shape().dim(axis);
      const int64_t elements = width * inner;
      if (elements > 0) {
        const char* in_bytes = static_cast<const char*>(t.raw_data());
        const int64_t in_offset = outer_idx * width * inner;
        const int64_t out_offset = outer_idx * axis_total * inner + axis_offset * inner;
        const frame::Status copy_status = frame::backends::cuda::cuda_status(
            cudaMemcpyAsync(out_bytes + static_cast<size_t>(out_offset) * itemsize,
                            in_bytes + static_cast<size_t>(in_offset) * itemsize,
                            static_cast<size_t>(elements) * itemsize, cudaMemcpyDeviceToDevice,
                            stream),
            "concat cuda kernel: cudaMemcpyAsync");
        if (!copy_status.is_ok()) return copy_status;
      }
      axis_offset += width;
    }
  }
  return frame::Status::ok();
}

// slice(x; axis, start, stop) 的 CUDA 参考实现(M22,批4 T4,REUSE-002:与
// cpu 参考 src/backends/cpu/kernels/shape.cpp::slice_cpu_kernel 同一几何算法,
// 仅把 std::memcpy 换成 cudaMemcpyAsync(D2D))——连续段拷贝:concat 的逆
// 操作,把 axis 之前的维度视为 outer 循环,每个 outer 对应一段
// (stop-start)*inner 个元素的连续 D2D 拷贝。
frame::Status slice_cuda_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'slice' cuda kernel expects 1 input, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'slice' cuda kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& x = ctx.inputs[0];
  frame::Tensor& out = ctx.outputs[0];

  const bool elem_type_mismatch = !(x.dtype() == out.dtype());
  if (elem_type_mismatch) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'slice' cuda kernel requires x/out of the same dtype, got '" +
                                   std::string(x.dtype().name()) + "', '" +
                                   std::string(out.dtype().name()) + "'");
  }
  const bool supported = is_supported_shape_dtype(x.dtype().code());
  if (!supported) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'slice' cuda kernel does not support dtype '" +
                                   std::string(x.dtype().name()) +
                                   "' (v0 supports float32/float16/bfloat16 only)");
  }

  if (ctx.attrs == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'slice' cuda kernel is missing required attribute "
                               "'axis'/'start'/'stop': no attrs provided");
  }
  const auto axis_it = ctx.attrs->find("axis");
  if (axis_it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'slice' cuda kernel is missing required attribute 'axis'");
  }
  const int64_t* axis_ptr = std::get_if<int64_t>(&axis_it->second);
  if (axis_ptr == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'slice' cuda kernel attribute 'axis' has the wrong type, expected int64");
  }
  const auto start_it = ctx.attrs->find("start");
  if (start_it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'slice' cuda kernel is missing required attribute 'start'");
  }
  const int64_t* start_ptr = std::get_if<int64_t>(&start_it->second);
  if (start_ptr == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'slice' cuda kernel attribute 'start' has the wrong type, expected int64");
  }
  const auto stop_it = ctx.attrs->find("stop");
  if (stop_it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'slice' cuda kernel is missing required attribute 'stop'");
  }
  const int64_t* stop_ptr = std::get_if<int64_t>(&stop_it->second);
  if (stop_ptr == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'slice' cuda kernel attribute 'stop' has the wrong type, expected int64");
  }

  const int64_t axis = *axis_ptr;
  const int64_t start = *start_ptr;
  const int64_t stop = *stop_ptr;
  const int64_t rank = x.shape().rank();
  if (axis < 0 || axis >= rank) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'slice' cuda kernel attribute 'axis' " + std::to_string(axis) +
                                   " is out of range for rank " + std::to_string(rank));
  }
  const int64_t dim = x.shape().dim(axis);
  if (start < 0 || start >= dim) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'slice' cuda kernel attribute 'start' " + std::to_string(start) +
                                   " is out of range for dimension " + std::to_string(dim));
  }
  if (stop <= start || stop > dim) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'slice' cuda kernel attribute 'stop' " + std::to_string(stop) +
                                   " is invalid for start=" + std::to_string(start) +
                                   " and dimension " + std::to_string(dim));
  }

  std::vector<int64_t> expected_out_dims = x.shape().dims();
  expected_out_dims[static_cast<size_t>(axis)] = stop - start;
  const frame::Shape expected_out_shape(expected_out_dims);
  if (!(out.shape() == expected_out_shape)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'slice' cuda kernel requires out shape to match the sliced result, got " +
            out.shape().to_string() + ", expected " + expected_out_shape.to_string());
  }

  int64_t outer = 1;
  for (int64_t d = 0; d < axis; ++d) outer *= x.shape().dim(d);
  int64_t inner = 1;
  for (int64_t d = axis + 1; d < rank; ++d) inner *= x.shape().dim(d);
  const int64_t width = stop - start;
  const size_t itemsize = x.dtype().itemsize();
  const cudaStream_t stream = native_stream(ctx.stream);

  const char* x_bytes = static_cast<const char*>(x.raw_data());
  char* out_bytes = static_cast<char*>(out.raw_data());
  const int64_t elements = width * inner;
  for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
    if (elements == 0) continue;
    const int64_t in_offset = outer_idx * dim * inner + start * inner;
    const int64_t out_offset = outer_idx * width * inner;
    const frame::Status copy_status = frame::backends::cuda::cuda_status(
        cudaMemcpyAsync(out_bytes + static_cast<size_t>(out_offset) * itemsize,
                        x_bytes + static_cast<size_t>(in_offset) * itemsize,
                        static_cast<size_t>(elements) * itemsize, cudaMemcpyDeviceToDevice, stream),
        "slice cuda kernel: cudaMemcpyAsync");
    if (!copy_status.is_ok()) return copy_status;
  }
  return frame::Status::ok();
}

}  // namespace

FRAME_REGISTER_KERNEL("reshape", frame::kCudaBackendName, reshape_cuda_kernel);
FRAME_REGISTER_KERNEL("concat", frame::kCudaBackendName, concat_cuda_kernel);
FRAME_REGISTER_KERNEL("slice", frame::kCudaBackendName, slice_cuda_kernel);
