// CUDA softmax 内核(M22,批4 T4,§1.2/1.6 决议点B/F,ADR-0021 决策 2 增补):
// softmax 前向经 cudnnSoftmaxForward(CUDNN_SOFTMAX_ACCURATE +
// CUDNN_SOFTMAX_MODE_INSTANCE;[N,D] -> 4D desc N=N,C=D,H=W=1;dataType 按
// fp32/fp16/bf16 映射,同 conv.cpp/pool.cpp 的映射口径)。**硬约束**(ADR-0021
// 决策 2 增补的圈禁清单):cuDNN 调用只准出现在本文件,layer_norm 无 cuDNN
// legacy 对应物,自写 kernel 见同目录 sequence.cu(镜像 pool.cpp/pool.cu 的
// host 包装/自写 kernel 分工先例)。softmax 反向无 kernel——梯度 = 公开算子
// 微图(§1.2 表),由既有 mul/sum/reshape/matmul/add cuda kernel 承载,构图侧
// 见 src/ops/schemas/sequence.cpp::softmax_gradient。不含 __global__ 代码,
// 纯 host 端 cuDNN 调用编排,故为 .cpp(同 conv.cpp/pool.cpp 先例)。

#include <cstddef>
#include <cstdint>
#include <cudnn.h>
#include <string>
#include <string_view>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/backend.h>
#include <frame/hal/stream.h>
#include <frame/ops/kernel_registry.h>

#include "../cuda_backend.h"
#include "../cuda_status.h"
#include "cudnn_utils.h"

namespace {

using frame::backends::cuda::CudaBackend;
using frame::backends::cuda::CudnnHandleGuard;

// cudnnStatus_t -> Status 翻译、TensorDescGuard(RAII):同目录共享工具(铁律
// 5 收敛,M22 批4 判重,见 cudnn_utils.h 头注释)。
using frame::backends::cuda::cudnn_status;
using frame::backends::cuda::TensorDescGuard;

// dtype -> cudnnDataType_t 映射(ADR-0021 决策 4,与 conv.cpp/pool.cpp 同一
// 映射口径);调用前已校验 dtype 属 v0 三档浮点。
cudnnDataType_t cudnn_data_type(frame::DTypeCode code) {
  if (code == frame::DTypeCode::kFloat16) return CUDNN_DATA_HALF;
  if (code == frame::DTypeCode::kBFloat16) return CUDNN_DATA_BFLOAT16;
  return CUDNN_DATA_FLOAT;
}

// dtype 一致性 + v0 浮点三档校验(REUSE-002:与
// src/backends/cuda/kernels/pool.cpp::require_matching_supported_dtype 同一
// 动机,cuda 侧各文件独立持有一份实现,不跨文件借用匿名命名空间符号)。
frame::Result<frame::DTypeCode> require_matching_supported_dtype(
    std::string_view op_name, std::string_view role_phrase,
    const std::vector<const frame::Tensor*>& tensors) {
  const frame::DType first_type = tensors.front()->dtype();
  bool mismatch = false;
  for (const frame::Tensor* tensor : tensors) {
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

frame::Result<CudaBackend*> lookup_cuda_backend(std::string_view op_name, frame::Device device) {
  const frame::Result<frame::hal::Backend*> backend_lookup =
      frame::hal::BackendRegistry::instance().get(device.backend);
  if (!backend_lookup.is_ok()) {
    return frame::Status::make(backend_lookup.status().code(),
                               "op '" + std::string(op_name) + "' cuda kernel: " +
                                   std::string(backend_lookup.status().message()));
  }
  // static_cast 而非 dynamic_cast(CPP-011):理由同 conv.cpp/pool.cpp
  // ::lookup_cuda_backend。
  return static_cast<CudaBackend*>(backend_lookup.value());
}

cudaStream_t native_stream(frame::hal::Stream* stream) {
  return stream != nullptr ? static_cast<cudaStream_t>(stream->native_handle()) : nullptr;
}

// softmax(x[N,D]) -> out[N,D]:末轴 softmax,经 cudnnSoftmaxForward(4D desc
// N=N,C=D,H=W=1,CUDNN_SOFTMAX_MODE_INSTANCE 即对每个 N 独立在 C 维做
// softmax,恰对应"末轴"语义)。
frame::Status softmax_cuda_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'softmax' cuda kernel expects 1 input, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'softmax' cuda kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& x = ctx.inputs[0];
  frame::Tensor& out = ctx.outputs[0];

  if (x.shape().rank() != 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'softmax' cuda kernel requires x to be rank-2 [N, D], got rank " +
            std::to_string(x.shape().rank()));
  }
  const frame::Result<frame::DTypeCode> code_result =
      require_matching_supported_dtype("softmax", "x/out", {&x, &out});
  if (!code_result.is_ok()) return code_result.status();
  const frame::DTypeCode code = code_result.value();

  if (!(out.shape() == x.shape())) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'softmax' cuda kernel requires out shape to match x, got " +
                                   out.shape().to_string() + ", expected " + x.shape().to_string());
  }

  const int64_t n = x.shape().dim(0);
  const int64_t d = x.shape().dim(1);

  const frame::Result<CudaBackend*> backend_result = lookup_cuda_backend("softmax", ctx.device);
  if (!backend_result.is_ok()) return backend_result.status();
  CudaBackend* cuda_backend = backend_result.value();
  const cudaStream_t stream = native_stream(ctx.stream);
  const frame::Result<CudnnHandleGuard> handle_guard = cuda_backend->acquire_cudnn_handle(stream);
  if (!handle_guard.is_ok()) return handle_guard.status();
  const cudnnHandle_t handle = handle_guard.value().handle;

  const cudnnDataType_t data_type = cudnn_data_type(code);
  TensorDescGuard x_desc;
  FRAME_RETURN_IF_ERROR(cudnn_status(cudnnCreateTensorDescriptor(&x_desc.desc),
                                     "softmax cuda kernel: cudnnCreateTensorDescriptor(x)"));
  FRAME_RETURN_IF_ERROR(
      cudnn_status(cudnnSetTensor4dDescriptor(x_desc.desc, CUDNN_TENSOR_NCHW, data_type,
                                              static_cast<int>(n), static_cast<int>(d), 1, 1),
                   "softmax cuda kernel: cudnnSetTensor4dDescriptor(x)"));

  TensorDescGuard y_desc;
  FRAME_RETURN_IF_ERROR(cudnn_status(cudnnCreateTensorDescriptor(&y_desc.desc),
                                     "softmax cuda kernel: cudnnCreateTensorDescriptor(y)"));
  FRAME_RETURN_IF_ERROR(
      cudnn_status(cudnnSetTensor4dDescriptor(y_desc.desc, CUDNN_TENSOR_NCHW, data_type,
                                              static_cast<int>(n), static_cast<int>(d), 1, 1),
                   "softmax cuda kernel: cudnnSetTensor4dDescriptor(y)"));

  const float alpha = 1.0F;
  const float beta = 0.0F;
  return cudnn_status(
      cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
                          x_desc.desc, x.raw_data(), &beta, y_desc.desc, out.raw_data()),
      "softmax cuda kernel: cudnnSoftmaxForward");
}

}  // namespace

FRAME_REGISTER_KERNEL("softmax", frame::kCudaBackendName, softmax_cuda_kernel);
