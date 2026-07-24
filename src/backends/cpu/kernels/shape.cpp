// CPU 形状变换内核:reshape(M21 批3 T4)+ transpose/concat/slice(M22 批4
// T3,§1.4)。内核形如 Status kernel(ops::KernelContext&),再经
// FRAME_REGISTER_KERNEL 注册到 (op, kCpuBackendName)。transpose/concat/slice
// 三者均为纯搬运(无数值语义),dtype 支持浮点三档,逐段/逐元素 memcpy——不经
// dispatch_dtype(与 reshape 同一理由,见其头注释:字节拷贝天然与 dtype 无关,
// ARCH-042 豁免),仅经 kernel_dtype_checks.h::require_matching_supported_dtype
// 做白名单校验(取 itemsize 供拷贝用)。

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <variant>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/ops/kernel_registry.h>

#include "kernel_dtype_checks.h"

namespace {

// reshape 的 CPU 参考实现:逐字节拷贝——numel 守恒由 schema 侧 shape 推断
// (src/ops/schemas/shape.cpp::infer_reshape_shape)已保证,kernel 侧仍防御性
// 复核 dtype 一致与 numel 相等。不经 dispatch_dtype:本算子不依赖具体数值
// 类型的算术语义,按字节拷贝天然与 dtype 无关(kernels/ 目录 dtype 分支检查
// 针对"按 dtype 做不同数值处理"的运行时分支,ARCH-042;本 kernel 无该类
// 分支,合规)。
frame::Status reshape_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'reshape' cpu kernel expects 1 input, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'reshape' cpu kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& x = ctx.inputs[0];
  frame::Tensor& out = ctx.outputs[0];

  // 比较前先拆具名变量(CPP-012 惯例:与 dispatch_dtype 编译期分派区分,
  // 见 kernel_dtype_checks.h 头注释)。
  const frame::DType x_type = x.dtype();
  const frame::DType out_type = out.dtype();
  if (!(x_type == out_type)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'reshape' cpu kernel requires x/out of the same dtype, got '" +
                                   std::string(x_type.name()) + "', '" +
                                   std::string(out_type.name()) + "'");
  }
  if (x.numel() != out.numel()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'reshape' cpu kernel requires x/out numel to match, got " +
                                   std::to_string(x.numel()) + ", " + std::to_string(out.numel()));
  }

  const size_t nbytes = static_cast<size_t>(x.numel()) * x.dtype().itemsize();
  std::memcpy(out.raw_data(), x.raw_data(), nbytes);
  return frame::Status::ok();
}

// transpose(x; perm) 的 CPU 参考实现(M22,批4 T3,REUSE-011:参考实现,数值
// 校验用,禁作性能路径)——逐输出元素按 perm 反查输入索引:对每个输出线性
// 下标解码出多维下标(行优先),第 i 维对应输入的第 perm[i] 维,据此按输入
// strides 求出输入线性下标后逐元素(itemsize 宽)memcpy。
frame::Status transpose_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'transpose' cpu kernel expects 1 input, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'transpose' cpu kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& x = ctx.inputs[0];
  frame::Tensor& out = ctx.outputs[0];

  const frame::Result<frame::DTypeCode> code_result =
      frame::backends::cpu::require_matching_supported_dtype("transpose", "x/out", {&x, &out});
  if (!code_result.is_ok()) return code_result.status();

  if (ctx.attrs == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'transpose' cpu kernel is missing required attribute 'perm' (int64 array): no attrs "
        "provided");
  }
  const auto perm_it = ctx.attrs->find("perm");
  if (perm_it == ctx.attrs->end()) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'transpose' cpu kernel is missing required attribute 'perm' (int64 array)");
  }
  const std::vector<int64_t>* perm = std::get_if<std::vector<int64_t>>(&perm_it->second);
  if (perm == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'transpose' cpu kernel attribute 'perm' has the wrong type, expected int64 array");
  }

  const int64_t rank = x.shape().rank();
  if (static_cast<int64_t>(perm->size()) != rank) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'transpose' cpu kernel attribute 'perm' must have rank=" + std::to_string(rank) +
            " element(s), got " + std::to_string(perm->size()));
  }
  std::vector<int64_t> expected_out_dims(static_cast<size_t>(rank));
  for (int64_t i = 0; i < rank; ++i) {
    const int64_t p = (*perm)[static_cast<size_t>(i)];
    if (p < 0 || p >= rank) {
      return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                                 "op 'transpose' cpu kernel attribute 'perm' entry " +
                                     std::to_string(p) + " is out of range for rank " +
                                     std::to_string(rank));
    }
    expected_out_dims[static_cast<size_t>(i)] = x.shape().dim(p);
  }
  const frame::Shape expected_out_shape(expected_out_dims);
  if (!(out.shape() == expected_out_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'transpose' cpu kernel requires out shape to match the "
                               "permuted result, got " +
                                   out.shape().to_string() + ", expected " +
                                   expected_out_shape.to_string());
  }

  const frame::Strides x_strides = frame::row_major_strides(x.shape());
  const std::vector<int64_t>& x_stride_values = x_strides.values();
  const std::vector<int64_t>& out_dims = out.shape().dims();
  const int64_t numel = out.numel();
  const size_t itemsize = x.dtype().itemsize();

  const char* x_bytes = static_cast<const char*>(x.raw_data());
  char* out_bytes = static_cast<char*>(out.raw_data());

  std::vector<int64_t> out_index(static_cast<size_t>(rank));
  for (int64_t linear = 0; linear < numel; ++linear) {
    int64_t remaining = linear;
    for (int64_t d = rank - 1; d >= 0; --d) {
      const int64_t dim_size = out_dims[static_cast<size_t>(d)];
      const int64_t safe_dim = dim_size > 0 ? dim_size : 1;
      out_index[static_cast<size_t>(d)] = remaining % safe_dim;
      remaining /= safe_dim;
    }
    int64_t x_linear = 0;
    for (int64_t d = 0; d < rank; ++d) {
      const int64_t p = (*perm)[static_cast<size_t>(d)];
      x_linear += out_index[static_cast<size_t>(d)] * x_stride_values[static_cast<size_t>(p)];
    }
    std::memcpy(out_bytes + static_cast<size_t>(linear) * itemsize,
                x_bytes + static_cast<size_t>(x_linear) * itemsize, itemsize);
  }
  return frame::Status::ok();
}

// concat(xs...; axis) 的 CPU 参考实现(M22,批4 T3,REUSE-011:参考实现,数值
// 校验用,禁作性能路径)——逐输入段拷贝:把 axis 之前的维度视为 outer 循环、
// 之后的维度视为连续内层块(inner),每个 (outer, input) 对应一段
// width*inner 个元素的连续 memcpy,按各输入沿 axis 的宽度累加偏移写入
// out。
frame::Status concat_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.empty()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'concat' cpu kernel expects at least 1 input, got 0");
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'concat' cpu kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }
  frame::Tensor& out = ctx.outputs[0];

  std::vector<const frame::Tensor*> tensors;
  tensors.reserve(ctx.inputs.size() + 1);
  for (const frame::Tensor& t : ctx.inputs) tensors.push_back(&t);
  tensors.push_back(&out);
  const frame::Result<frame::DTypeCode> code_result =
      frame::backends::cpu::require_matching_supported_dtype("concat", "xs/out", tensors);
  if (!code_result.is_ok()) return code_result.status();

  if (ctx.attrs == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'concat' cpu kernel is missing required attribute 'axis': no attrs provided");
  }
  const auto axis_it = ctx.attrs->find("axis");
  if (axis_it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'concat' cpu kernel is missing required attribute 'axis'");
  }
  const int64_t* axis_ptr = std::get_if<int64_t>(&axis_it->second);
  if (axis_ptr == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'concat' cpu kernel attribute 'axis' has the wrong type, expected int64");
  }
  const int64_t axis = *axis_ptr;

  const frame::Tensor& first = ctx.inputs[0];
  const int64_t rank = first.shape().rank();
  if (axis < 0 || axis >= rank) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'concat' cpu kernel attribute 'axis' " + std::to_string(axis) +
                                   " is out of range for rank " + std::to_string(rank));
  }

  int64_t axis_total = 0;
  for (const frame::Tensor& t : ctx.inputs) {
    if (t.shape().rank() != rank) {
      return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                                 "op 'concat' cpu kernel requires all inputs of the same rank, "
                                 "got rank " +
                                     std::to_string(t.shape().rank()) + " and " +
                                     std::to_string(rank));
    }
    for (int64_t d = 0; d < rank; ++d) {
      if (d == axis) continue;
      if (t.shape().dim(d) != first.shape().dim(d)) {
        return frame::Status::make(
            frame::ErrorCode::kInvalidArgument,
            "op 'concat' cpu kernel requires all inputs to match on non-axis dimension " +
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
        "op 'concat' cpu kernel requires out shape to match the concatenation result, got " +
            out.shape().to_string() + ", expected " + expected_out_shape.to_string());
  }

  int64_t outer = 1;
  for (int64_t d = 0; d < axis; ++d) outer *= first.shape().dim(d);
  int64_t inner = 1;
  for (int64_t d = axis + 1; d < rank; ++d) inner *= first.shape().dim(d);
  const size_t itemsize = first.dtype().itemsize();

  char* out_bytes = static_cast<char*>(out.raw_data());
  for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
    int64_t axis_offset = 0;
    for (const frame::Tensor& t : ctx.inputs) {
      const int64_t width = t.shape().dim(axis);
      const int64_t elements = width * inner;
      const char* in_bytes = static_cast<const char*>(t.raw_data());
      const int64_t in_offset = outer_idx * width * inner;
      const int64_t out_offset = outer_idx * axis_total * inner + axis_offset * inner;
      std::memcpy(out_bytes + static_cast<size_t>(out_offset) * itemsize,
                  in_bytes + static_cast<size_t>(in_offset) * itemsize,
                  static_cast<size_t>(elements) * itemsize);
      axis_offset += width;
    }
  }
  return frame::Status::ok();
}

// slice(x; axis, start, stop) 的 CPU 参考实现(M22,批4 T3,REUSE-011:参考
// 实现,数值校验用,禁作性能路径)——连续段拷贝:concat 的逆操作,把 axis
// 之前的维度视为 outer 循环,每个 outer 对应一段 (stop-start)*inner 个元素
// 的连续 memcpy。
frame::Status slice_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'slice' cpu kernel expects 1 input, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'slice' cpu kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& x = ctx.inputs[0];
  frame::Tensor& out = ctx.outputs[0];

  const frame::Result<frame::DTypeCode> code_result =
      frame::backends::cpu::require_matching_supported_dtype("slice", "x/out", {&x, &out});
  if (!code_result.is_ok()) return code_result.status();

  if (ctx.attrs == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'slice' cpu kernel is missing required attribute 'axis'/'start'/'stop': no attrs "
        "provided");
  }
  const auto axis_it = ctx.attrs->find("axis");
  if (axis_it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'slice' cpu kernel is missing required attribute 'axis'");
  }
  const int64_t* axis_ptr = std::get_if<int64_t>(&axis_it->second);
  if (axis_ptr == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'slice' cpu kernel attribute 'axis' has the wrong type, expected int64");
  }
  const auto start_it = ctx.attrs->find("start");
  if (start_it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'slice' cpu kernel is missing required attribute 'start'");
  }
  const int64_t* start_ptr = std::get_if<int64_t>(&start_it->second);
  if (start_ptr == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'slice' cpu kernel attribute 'start' has the wrong type, expected int64");
  }
  const auto stop_it = ctx.attrs->find("stop");
  if (stop_it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'slice' cpu kernel is missing required attribute 'stop'");
  }
  const int64_t* stop_ptr = std::get_if<int64_t>(&stop_it->second);
  if (stop_ptr == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'slice' cpu kernel attribute 'stop' has the wrong type, expected int64");
  }

  const int64_t axis = *axis_ptr;
  const int64_t start = *start_ptr;
  const int64_t stop = *stop_ptr;
  const int64_t rank = x.shape().rank();
  if (axis < 0 || axis >= rank) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'slice' cpu kernel attribute 'axis' " + std::to_string(axis) +
                                   " is out of range for rank " + std::to_string(rank));
  }
  const int64_t dim = x.shape().dim(axis);
  if (start < 0 || start >= dim) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'slice' cpu kernel attribute 'start' " + std::to_string(start) +
                                   " is out of range for dimension " + std::to_string(dim));
  }
  if (stop <= start || stop > dim) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'slice' cpu kernel attribute 'stop' " + std::to_string(stop) +
                                   " is invalid for start=" + std::to_string(start) +
                                   " and dimension " + std::to_string(dim));
  }

  std::vector<int64_t> expected_out_dims = x.shape().dims();
  expected_out_dims[static_cast<size_t>(axis)] = stop - start;
  const frame::Shape expected_out_shape(expected_out_dims);
  if (!(out.shape() == expected_out_shape)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'slice' cpu kernel requires out shape to match the sliced result, got " +
            out.shape().to_string() + ", expected " + expected_out_shape.to_string());
  }

  int64_t outer = 1;
  for (int64_t d = 0; d < axis; ++d) outer *= x.shape().dim(d);
  int64_t inner = 1;
  for (int64_t d = axis + 1; d < rank; ++d) inner *= x.shape().dim(d);
  const int64_t width = stop - start;
  const size_t itemsize = x.dtype().itemsize();

  const char* x_bytes = static_cast<const char*>(x.raw_data());
  char* out_bytes = static_cast<char*>(out.raw_data());
  const int64_t elements = width * inner;
  for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
    const int64_t in_offset = outer_idx * dim * inner + start * inner;
    const int64_t out_offset = outer_idx * width * inner;
    std::memcpy(out_bytes + static_cast<size_t>(out_offset) * itemsize,
                x_bytes + static_cast<size_t>(in_offset) * itemsize,
                static_cast<size_t>(elements) * itemsize);
  }
  return frame::Status::ok();
}

}  // namespace

FRAME_REGISTER_KERNEL("reshape", frame::kCpuBackendName, reshape_cpu_kernel);
FRAME_REGISTER_KERNEL("transpose", frame::kCpuBackendName, transpose_cpu_kernel);
FRAME_REGISTER_KERNEL("concat", frame::kCpuBackendName, concat_cpu_kernel);
FRAME_REGISTER_KERNEL("slice", frame::kCpuBackendName, slice_cpu_kernel);
