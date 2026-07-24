#pragma once
// cuda kernel 层共享的 cuDNN 工具(M22,批4 T4,code-reviewer 建议①判重
// 收敛):cudnn_status(cudnnStatus_t -> Status 翻译)与 TensorDescGuard
// (cudnnTensorDescriptor_t RAII guard)曾在 conv.cpp(M21)、pool.cpp(M21)、
// sequence.cpp(M22 批4 T4)三个文件各持一份逐字相同实现,同目录可共享,收敛
// 为单份(铁律 5)。conv.cpp/pool.cpp 各自专有的其他 cuDNN 描述符 guard
// (FilterDescGuard/ConvDescGuard/PoolingDescGuard 等)不在本次收编范围,继续
// 各自持有(其余文件不需要这些类型,不构成重复)。仅供
// src/backends/cuda/kernels/ 内部包含,不入公开 API。

#include <cudnn.h>
#include <string>
#include <string_view>

#include <frame/core/status.h>

namespace frame::backends::cuda {

// cudnnStatus_t -> Status 翻译:SUCCESS 映射 ok(),其余一律 kInternal,消息
// 含调用方传入的 context(定位用,如函数名/调用点摘要)与
// cudnnGetErrorString 文本。
inline frame::Status cudnn_status(cudnnStatus_t status, std::string_view context) {
  if (status == CUDNN_STATUS_SUCCESS) return frame::Status::ok();
  return frame::Status::make(
      frame::ErrorCode::kInternal,
      std::string(context) + ": cudnn call failed: " + cudnnGetErrorString(status));
}

// cudnnTensorDescriptor_t 的 RAII guard:禁止拷贝,析构时若已创建则销毁。
struct TensorDescGuard {
  cudnnTensorDescriptor_t desc = nullptr;
  TensorDescGuard() = default;
  TensorDescGuard(const TensorDescGuard&) = delete;
  TensorDescGuard& operator=(const TensorDescGuard&) = delete;
  ~TensorDescGuard() {
    if (desc != nullptr) cudnnDestroyTensorDescriptor(desc);
  }
};

}  // namespace frame::backends::cuda
