#pragma once
// cuda pool kernel 侧共享的几何/属性读取工具(M21,批3 T5)。收敛动机
// (铁律 5):pool.cpp(cudnnPoolingForward/Backward 编排)与 pool.cu(自写
// max_pool2d_select_internal/avg_pool2d_grad_internal kernel)同属 pool 算子族
// 且在同一 commit 交付,若各自独立持有一份 Pool2dRuntimeParams/属性读取实现,
// 将构成 REUSE-002 意义上的同批次重复(与 src/backends/cpu/kernels/
// kernel_dtype_checks.h 同一收敛动机,该头供 conv.cpp/pool.cpp 共用 dtype
// 校验)。仅供 src/backends/cuda/kernels/ 内部包含,不入公开 API。

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/ops/kernel_registry.h>

namespace frame::backends::cuda {

// 二维池化的几何参数,供 max_pool2d/avg_pool2d/max_pool2d_grad_internal
// (pool.cpp)与 max_pool2d_select_internal/avg_pool2d_grad_internal(pool.cu)
// 共用。
struct Pool2dRuntimeParams {
  int64_t n = 0;
  int64_t c = 0;
  int64_t h = 0;
  int64_t w = 0;
  int64_t kh = 0;
  int64_t kw = 0;
  int64_t stride_h = 0;
  int64_t stride_w = 0;
  int64_t pad_h = 0;
  int64_t pad_w = 0;
  int64_t out_h = 0;
  int64_t out_w = 0;
};

inline frame::Result<std::vector<int64_t>> read_int64_array_attr(
    const frame::ops::KernelContext& ctx, std::string_view op_name, std::string_view attr_name) {
  if (ctx.attrs == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel is missing required attribute '" +
                                   std::string(attr_name) + "': no attrs provided");
  }
  const auto it = ctx.attrs->find(std::string(attr_name));
  if (it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel is missing required attribute '" +
                                   std::string(attr_name) + "'");
  }
  const std::vector<int64_t>* value = std::get_if<std::vector<int64_t>>(&it->second);
  if (value == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' cuda kernel attribute '" +
                                   std::string(attr_name) +
                                   "' has the wrong type, expected int64 array");
  }
  return *value;
}

inline frame::Result<frame::Shape> read_shape_attr(const frame::ops::KernelContext& ctx,
                                                   std::string_view op_name,
                                                   std::string_view attr_name) {
  if (ctx.attrs == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel is missing required attribute '" +
                                   std::string(attr_name) + "': no attrs provided");
  }
  const auto it = ctx.attrs->find(std::string(attr_name));
  if (it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel is missing required attribute '" +
                                   std::string(attr_name) + "'");
  }
  const frame::Shape* value = std::get_if<frame::Shape>(&it->second);
  if (value == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' cuda kernel attribute '" +
                                   std::string(attr_name) + "' has the wrong type, expected shape");
  }
  return *value;
}

// 读取 kernel/stride/padding 三个必需 kInt64Array 属性,组装进
// Pool2dRuntimeParams。x_shape 提供 [N,C,H,W]。
inline frame::Result<Pool2dRuntimeParams> read_pool2d_runtime_params(
    const frame::ops::KernelContext& ctx, std::string_view op_name, const frame::Shape& x_shape) {
  const frame::Result<std::vector<int64_t>> kernel_result =
      read_int64_array_attr(ctx, op_name, "kernel");
  if (!kernel_result.is_ok()) return kernel_result.status();
  const frame::Result<std::vector<int64_t>> stride_result =
      read_int64_array_attr(ctx, op_name, "stride");
  if (!stride_result.is_ok()) return stride_result.status();
  const frame::Result<std::vector<int64_t>> padding_result =
      read_int64_array_attr(ctx, op_name, "padding");
  if (!padding_result.is_ok()) return padding_result.status();

  const std::vector<int64_t>& kernel = kernel_result.value();
  const std::vector<int64_t>& stride = stride_result.value();
  const std::vector<int64_t>& padding = padding_result.value();
  if (kernel.size() != 2 || stride.size() != 2 || padding.size() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel requires 'kernel'/'stride'/'padding' to each "
                                   "have 2 elements");
  }

  Pool2dRuntimeParams params;
  params.n = x_shape.dim(0);
  params.c = x_shape.dim(1);
  params.h = x_shape.dim(2);
  params.w = x_shape.dim(3);
  params.kh = kernel[0];
  params.kw = kernel[1];
  params.stride_h = stride[0];
  params.stride_w = stride[1];
  params.pad_h = padding[0];
  params.pad_w = padding[1];
  params.out_h = (params.h + 2 * params.pad_h - params.kh) / params.stride_h + 1;
  params.out_w = (params.w + 2 * params.pad_w - params.kw) / params.stride_w + 1;
  return params;
}

}  // namespace frame::backends::cuda
