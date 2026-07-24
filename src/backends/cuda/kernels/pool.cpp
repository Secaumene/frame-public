// CUDA 二维池化内核(M21,批3 T5,ADR-0021):max_pool2d/avg_pool2d 前向经
// cudnnPoolingForward;max_pool2d_grad_internal(dy,x)->dx 经"cudnnPoolingForward
// 重算 y 至临时缓冲 + cudnnPoolingBackward"两步(裁决点③,计划 1.3 节:复用
// cuDNN 优先于自写 scatter,铁律 5)。max_pool2d_select_internal/
// avg_pool2d_grad_internal 无 cuDNN 对应物,见同目录 pool.cu 自写 kernel。
// 不含 __global__ 代码,纯 host 端 cuDNN 调用编排,故为 .cpp(同 conv.cpp/
// matmul.cpp 先例)。
//
// avg_pool2d 分母口径:CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING(恒
// KH*KW,与 CPU 参考 include-padding 口径一致,计划 1.1 节)。max_pool2d 的
// padding 语义 = -inf,由 cuDNN pooling 内部对越界位置的固定处理承担(不显式
// 传 -inf 值,cuDNN legacy pooling API 本身即按此语义实现)。

#include <cstddef>
#include <cstdint>
#include <cudnn.h>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/storage.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/stream.h>
#include <frame/ops/kernel_registry.h>

#include "../cuda_backend.h"
#include "../cuda_status.h"
#include "cudnn_utils.h"
#include "pool_geometry.h"

namespace {

using frame::backends::cuda::CudaBackend;
using frame::backends::cuda::CudnnHandleGuard;
using frame::backends::cuda::Pool2dRuntimeParams;
using frame::backends::cuda::read_pool2d_runtime_params;
using frame::backends::cuda::read_shape_attr;

// cudnnStatus_t -> Status 翻译、TensorDescGuard(RAII):同目录共享工具(铁律
// 5 收敛,M22 批4 判重,见 cudnn_utils.h 头注释)。
using frame::backends::cuda::cudnn_status;
using frame::backends::cuda::TensorDescGuard;

cudnnDataType_t cudnn_data_type(frame::DTypeCode code) {
  if (code == frame::DTypeCode::kFloat16) return CUDNN_DATA_HALF;
  if (code == frame::DTypeCode::kBFloat16) return CUDNN_DATA_BFLOAT16;
  return CUDNN_DATA_FLOAT;
}

struct PoolingDescGuard {
  cudnnPoolingDescriptor_t desc = nullptr;
  PoolingDescGuard() = default;
  PoolingDescGuard(const PoolingDescGuard&) = delete;
  PoolingDescGuard& operator=(const PoolingDescGuard&) = delete;
  ~PoolingDescGuard() {
    if (desc != nullptr) cudnnDestroyPoolingDescriptor(desc);
  }
};

frame::Result<frame::DTypeCode> require_matching_supported_dtype(
    std::string_view op_name, std::string_view role_phrase,
    const std::vector<const frame::Tensor*>& tensors) {
  const frame::DType first_type = tensors.front()->dtype();
  bool mismatch = false;
  for (const frame::Tensor* tensor : tensors) {
    // 先落地为具名变量再判断:理由同 conv.cpp::require_matching_supported_dtype
    // 同名函数头注释(CPP-012 文本扫描规避)。
    const frame::DType current_type = tensor->dtype();
    if (!(current_type == first_type)) {
      mismatch = true;
      break;
    }
  }
  if (mismatch) {
    std::string type_list_text;
    for (const frame::Tensor* tensor : tensors) {
      if (!type_list_text.empty()) type_list_text += ", ";
      type_list_text += "'" + std::string(tensor->dtype().name()) + "'";
    }
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' cuda kernel requires " +
                                   std::string(role_phrase) + " of the same dtype, got " +
                                   type_list_text);
  }
  const frame::DTypeCode code = first_type.code();
  const bool supported = code == frame::DTypeCode::kFloat32 || code == frame::DTypeCode::kFloat16 ||
                         code == frame::DTypeCode::kBFloat16;
  if (!supported) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) + "' cuda kernel does not support dtype '" +
            std::string(first_type.name()) + "' (v0 supports float32/float16/bfloat16 only)");
  }
  return code;
}

frame::Status require_rank(std::string_view op_name, std::string_view operand_label,
                           int64_t expected_rank, const frame::Tensor& tensor) {
  if (tensor.shape().rank() != expected_rank) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' cuda kernel requires " +
                                   std::string(operand_label) + " to be rank-" +
                                   std::to_string(expected_rank) + ", got rank " +
                                   std::to_string(tensor.shape().rank()));
  }
  return frame::Status::ok();
}

// 构建 x/y/pooling 三个描述符(REUSE-002:max_pool2d/avg_pool2d/
// max_pool2d_grad_internal 三个 kernel 共用同一套描述符构建逻辑,仅 mode 不
// 同)。out 形参:调用方在栈上持有 guard 局部变量,理由同
// conv.cpp::setup_conv_descriptors。
frame::Status setup_pool_descriptors(std::string_view op_name, const Pool2dRuntimeParams& p,
                                     cudnnDataType_t data_type, cudnnPoolingMode_t mode,
                                     TensorDescGuard& x_desc, TensorDescGuard& y_desc,
                                     PoolingDescGuard& pooling_desc) {
  FRAME_RETURN_IF_ERROR(
      cudnn_status(cudnnCreateTensorDescriptor(&x_desc.desc),
                   std::string(op_name) + " cuda kernel: cudnnCreateTensorDescriptor(x)"));
  FRAME_RETURN_IF_ERROR(
      cudnn_status(cudnnSetTensor4dDescriptor(x_desc.desc, CUDNN_TENSOR_NCHW, data_type,
                                              static_cast<int>(p.n), static_cast<int>(p.c),
                                              static_cast<int>(p.h), static_cast<int>(p.w)),
                   std::string(op_name) + " cuda kernel: cudnnSetTensor4dDescriptor(x)"));

  FRAME_RETURN_IF_ERROR(
      cudnn_status(cudnnCreateTensorDescriptor(&y_desc.desc),
                   std::string(op_name) + " cuda kernel: cudnnCreateTensorDescriptor(y)"));
  FRAME_RETURN_IF_ERROR(
      cudnn_status(cudnnSetTensor4dDescriptor(y_desc.desc, CUDNN_TENSOR_NCHW, data_type,
                                              static_cast<int>(p.n), static_cast<int>(p.c),
                                              static_cast<int>(p.out_h), static_cast<int>(p.out_w)),
                   std::string(op_name) + " cuda kernel: cudnnSetTensor4dDescriptor(y)"));

  FRAME_RETURN_IF_ERROR(
      cudnn_status(cudnnCreatePoolingDescriptor(&pooling_desc.desc),
                   std::string(op_name) + " cuda kernel: cudnnCreatePoolingDescriptor"));
  FRAME_RETURN_IF_ERROR(cudnn_status(
      cudnnSetPooling2dDescriptor(pooling_desc.desc, mode, CUDNN_NOT_PROPAGATE_NAN,
                                  static_cast<int>(p.kh), static_cast<int>(p.kw),
                                  static_cast<int>(p.pad_h), static_cast<int>(p.pad_w),
                                  static_cast<int>(p.stride_h), static_cast<int>(p.stride_w)),
      std::string(op_name) + " cuda kernel: cudnnSetPooling2dDescriptor"));
  return frame::Status::ok();
}

frame::Result<CudaBackend*> lookup_cuda_backend(std::string_view op_name, frame::Device device) {
  const frame::Result<frame::hal::Backend*> backend_lookup =
      frame::hal::BackendRegistry::instance().get(device.backend);
  if (!backend_lookup.is_ok()) {
    return frame::Status::make(backend_lookup.status().code(),
                               "op '" + std::string(op_name) + "' cuda kernel: " +
                                   std::string(backend_lookup.status().message()));
  }
  // static_cast 而非 dynamic_cast(CPP-011):理由同 conv.cpp::lookup_cuda_backend。
  return static_cast<CudaBackend*>(backend_lookup.value());
}

cudaStream_t native_stream(frame::hal::Stream* stream) {
  return stream != nullptr ? static_cast<cudaStream_t>(stream->native_handle()) : nullptr;
}

// pool 前向共用驱动(max_pool2d/avg_pool2d 除 mode 外完全同构,REUSE-002):
// 1 输入 x,attrs=kernel/stride/padding,1 输出;cudnnPoolingForward。
frame::Status pool2d_forward_cuda_kernel(std::string_view op_name, cudnnPoolingMode_t mode,
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

  const frame::Tensor& x = ctx.inputs[0];
  frame::Tensor& out = ctx.outputs[0];

  frame::Status rank_status = require_rank(op_name, "x", 4, x);
  if (!rank_status.is_ok()) return rank_status;

  const frame::Result<frame::DTypeCode> code_result =
      require_matching_supported_dtype(op_name, "x/out", {&x, &out});
  if (!code_result.is_ok()) return code_result.status();
  const frame::DTypeCode code = code_result.value();

  const frame::Result<Pool2dRuntimeParams> params_result =
      read_pool2d_runtime_params(ctx, op_name, x.shape());
  if (!params_result.is_ok()) return params_result.status();
  const Pool2dRuntimeParams& params = params_result.value();

  const frame::Shape expected_out_shape({params.n, params.c, params.out_h, params.out_w});
  if (!(out.shape() == expected_out_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel requires out shape to match the pooling "
                                   "result, got " +
                                   out.shape().to_string() + ", expected " +
                                   expected_out_shape.to_string());
  }

  const frame::Result<CudaBackend*> backend_result = lookup_cuda_backend(op_name, ctx.device);
  if (!backend_result.is_ok()) return backend_result.status();
  CudaBackend* cuda_backend = backend_result.value();
  const cudaStream_t stream = native_stream(ctx.stream);
  const frame::Result<CudnnHandleGuard> handle_guard = cuda_backend->acquire_cudnn_handle(stream);
  if (!handle_guard.is_ok()) return handle_guard.status();
  const cudnnHandle_t handle = handle_guard.value().handle;

  const cudnnDataType_t data_type = cudnn_data_type(code);
  TensorDescGuard x_desc;
  TensorDescGuard y_desc;
  PoolingDescGuard pooling_desc;
  const frame::Status setup_status =
      setup_pool_descriptors(op_name, params, data_type, mode, x_desc, y_desc, pooling_desc);
  if (!setup_status.is_ok()) return setup_status;

  const float alpha = 1.0F;
  const float beta = 0.0F;
  return cudnn_status(cudnnPoolingForward(handle, pooling_desc.desc, &alpha, x_desc.desc,
                                          x.raw_data(), &beta, y_desc.desc, out.raw_data()),
                      std::string(op_name) + " cuda kernel: cudnnPoolingForward");
}

frame::Status max_pool2d_cuda_kernel(frame::ops::KernelContext& ctx) {
  return pool2d_forward_cuda_kernel("max_pool2d", CUDNN_POOLING_MAX, ctx);
}

frame::Status avg_pool2d_cuda_kernel(frame::ops::KernelContext& ctx) {
  return pool2d_forward_cuda_kernel("avg_pool2d", CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, ctx);
}

// max_pool2d_grad_internal(dy,x)->dx(裁决点③):先 cudnnPoolingForward 在临时
// 缓冲重算 y,再 cudnnPoolingBackward(y,dy,x)->dx。attrs=
// kernel/stride/padding/input_shape。
frame::Status max_pool2d_grad_internal_cuda_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'max_pool2d_grad_internal' cuda kernel expects 2 inputs, got " +
                                   std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'max_pool2d_grad_internal' cuda kernel expects 1 output, got " +
                                   std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& dy = ctx.inputs[0];
  const frame::Tensor& x = ctx.inputs[1];
  frame::Tensor& dx = ctx.outputs[0];

  frame::Status rank_status = require_rank("max_pool2d_grad_internal", "dy", 4, dy);
  if (!rank_status.is_ok()) return rank_status;
  rank_status = require_rank("max_pool2d_grad_internal", "x", 4, x);
  if (!rank_status.is_ok()) return rank_status;

  const frame::Result<frame::DTypeCode> code_result =
      require_matching_supported_dtype("max_pool2d_grad_internal", "dy/x/dx", {&dy, &x, &dx});
  if (!code_result.is_ok()) return code_result.status();
  const frame::DTypeCode code = code_result.value();

  const frame::Result<frame::Shape> input_shape_result =
      read_shape_attr(ctx, "max_pool2d_grad_internal", "input_shape");
  if (!input_shape_result.is_ok()) return input_shape_result.status();
  const frame::Shape& input_shape = input_shape_result.value();
  if (!(x.shape() == input_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'max_pool2d_grad_internal' cuda kernel requires x shape to "
                               "match attribute 'input_shape', got " +
                                   x.shape().to_string() + ", expected " + input_shape.to_string());
  }
  if (!(dx.shape() == input_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'max_pool2d_grad_internal' cuda kernel requires dx(out) shape "
                               "to match attribute 'input_shape', got " +
                                   dx.shape().to_string() + ", expected " +
                                   input_shape.to_string());
  }

  const frame::Result<Pool2dRuntimeParams> params_result =
      read_pool2d_runtime_params(ctx, "max_pool2d_grad_internal", input_shape);
  if (!params_result.is_ok()) return params_result.status();
  const Pool2dRuntimeParams& params = params_result.value();

  const frame::Shape expected_dy_shape({params.n, params.c, params.out_h, params.out_w});
  if (!(dy.shape() == expected_dy_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'max_pool2d_grad_internal' cuda kernel requires dy shape to be "
                               "consistent with input_shape/kernel/stride/padding, got " +
                                   dy.shape().to_string() + ", expected " +
                                   expected_dy_shape.to_string());
  }

  const frame::Result<CudaBackend*> backend_result =
      lookup_cuda_backend("max_pool2d_grad_internal", ctx.device);
  if (!backend_result.is_ok()) return backend_result.status();
  CudaBackend* cuda_backend = backend_result.value();
  const cudaStream_t stream = native_stream(ctx.stream);
  const frame::Result<CudnnHandleGuard> handle_guard = cuda_backend->acquire_cudnn_handle(stream);
  if (!handle_guard.is_ok()) return handle_guard.status();
  const cudnnHandle_t handle = handle_guard.value().handle;

  const cudnnDataType_t data_type = cudnn_data_type(code);
  TensorDescGuard x_desc;
  TensorDescGuard y_desc;
  PoolingDescGuard pooling_desc;
  const frame::Status setup_status =
      setup_pool_descriptors("max_pool2d_grad_internal", params, data_type, CUDNN_POOLING_MAX,
                             x_desc, y_desc, pooling_desc);
  if (!setup_status.is_ok()) return setup_status;

  frame::hal::Allocator* allocator = cuda_backend->allocator(ctx.device);
  if (allocator == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInternal,
                               "op 'max_pool2d_grad_internal' cuda kernel: allocator unavailable "
                               "for device '" +
                                   std::string(ctx.device.backend) + "'");
  }
  // 临时缓冲重算 y(裁决点③):字节数恒 > 0(out_h/out_w 已由 shape 推断保证
  // >=1),经既有分配器分配,RAII 释放前显式同步(理由同 conv.cpp 头注释)。
  const size_t y_bytes =
      static_cast<size_t>(params.n * params.c * params.out_h * params.out_w) * x.dtype().itemsize();
  const frame::Result<std::shared_ptr<frame::Storage>> y_storage_result =
      frame::Storage::allocate(*allocator, y_bytes, frame::kDefaultAlignment, ctx.device);
  if (!y_storage_result.is_ok()) return y_storage_result.status();
  const std::shared_ptr<frame::Storage>& y_storage = y_storage_result.value();

  const float alpha = 1.0F;
  const float beta = 0.0F;
  frame::Status run_status =
      cudnn_status(cudnnPoolingForward(handle, pooling_desc.desc, &alpha, x_desc.desc, x.raw_data(),
                                       &beta, y_desc.desc, y_storage->data()),
                   "max_pool2d_grad_internal cuda kernel: cudnnPoolingForward (recompute y)");
  if (run_status.is_ok()) {
    run_status = cudnn_status(
        cudnnPoolingBackward(handle, pooling_desc.desc, &alpha, y_desc.desc, y_storage->data(),
                             y_desc.desc, dy.raw_data(), x_desc.desc, x.raw_data(), &beta,
                             x_desc.desc, dx.raw_data()),
        "max_pool2d_grad_internal cuda kernel: cudnnPoolingBackward");
  }

  const frame::Status sync_status = frame::backends::cuda::cuda_status(
      cudaStreamSynchronize(stream),
      "max_pool2d_grad_internal cuda kernel: cudaStreamSynchronize before scratch release");
  return !run_status.is_ok() ? run_status : sync_status;
}

}  // namespace

FRAME_REGISTER_KERNEL("max_pool2d", frame::kCudaBackendName, max_pool2d_cuda_kernel);
FRAME_REGISTER_KERNEL("avg_pool2d", frame::kCudaBackendName, avg_pool2d_cuda_kernel);
FRAME_REGISTER_KERNEL("max_pool2d_grad_internal", frame::kCudaBackendName,
                      max_pool2d_grad_internal_cuda_kernel);
