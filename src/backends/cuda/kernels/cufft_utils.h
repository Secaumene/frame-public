#pragma once
// cuda kernel 层共享的 cuFFT 工具(M23,批5 T4,ADR-0022 决策 2):cufft_status
// (cufftResult -> Status 翻译)与 CufftPlanGuard(cufftHandle RAII guard),
// cudnn_utils.h 同款口径(单份共享工具,供同目录 kernels/fft.cpp 使用)。
// cuFFT 调用面圈禁:仅 kernels/fft.cpp 与本文件(ADR-0022 决策 2:
// `grep -rln "cufft" src/ include/` 命中须 ⊆ 这两个文件 + CMakeLists.txt
// 链接行)。仅供 src/backends/cuda/kernels/ 内部包含,不入公开 API。

#include <cufft.h>
#include <string>
#include <string_view>

#include <frame/core/status.h>

namespace frame::backends::cuda {

// cufftResult -> Status 翻译:CUFFT_SUCCESS 映射 ok(),其余一律 kInternal,
// 消息含调用方传入的 context(定位用,如函数名/调用点摘要)与错误码数值——
// 不同于 cudnnGetErrorString/cudaGetErrorString,cuFFT 公开头文件
// (cufft.h)未提供错误码转字符串的公共 API,故直接嵌入 cufftResult 数值
// (LANG-005:消息本身仍为英文)。
inline frame::Status cufft_status(cufftResult status, std::string_view context) {
  if (status == CUFFT_SUCCESS) return frame::Status::ok();
  return frame::Status::make(frame::ErrorCode::kInternal,
                             std::string(context) + ": cufft call failed with error code " +
                                 std::to_string(static_cast<int>(status)));
}

// cufftHandle 的 RAII guard:禁止拷贝,析构时若已成功创建(created==true)则
// 销毁。ADR-0022 决策 6(v0 不缓存 plan):每次 kernel 调用现建现毁,plan
// 生命周期与本 guard 的作用域一致——调用方须在析构前对绑定的 stream 显式
// cudaStreamSynchronize,cufftDestroy 是否具备 stream 顺序语义未见 cuFFT
// 官方文档明文承诺(同 conv.cpp::allocate_workspace_if_needed 头注释同一
// 纪律:cudaFree 类释放操作不假设与异步在途工作天然有序)。
struct CufftPlanGuard {
  cufftHandle plan = 0;
  bool created = false;
  CufftPlanGuard() = default;
  CufftPlanGuard(const CufftPlanGuard&) = delete;
  CufftPlanGuard& operator=(const CufftPlanGuard&) = delete;
  ~CufftPlanGuard() {
    if (created) cufftDestroy(plan);
  }
};

}  // namespace frame::backends::cuda
