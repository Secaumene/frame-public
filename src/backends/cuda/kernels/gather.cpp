// cuda gather 内核族共享的 indices D2H 预检工具的实现单元(见
// kernels/gather.h)。不含 __global__ 代码,故为 .cpp;kernels/gather.cu 的
// gather/scatter_add/gather_grad_internal 三个 host 包装函数均调用本文件唯一的
// 导出函数(REUSE-002:预检逻辑单份,避免各算子复制)。

#include "gather.h"

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include <frame/core/dtype.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>

#include "../cuda_status.h"

namespace frame::backends::cuda {

frame::Status validate_gather_indices_range(std::string_view op_name, const frame::Tensor& indices,
                                            int64_t bound, std::string_view bound_description,
                                            cudaStream_t stream) {
  const frame::DTypeCode code = indices.dtype().code();
  const bool supported = code == frame::DTypeCode::kInt32 || code == frame::DTypeCode::kInt64;
  if (!supported) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel requires indices dtype to be int32 or int64, "
                                   "got '" +
                                   std::string(indices.dtype().name()) + "'");
  }

  const int64_t count = indices.numel();
  if (count == 0) return frame::Status::ok();

  return frame::dispatch_dtype(code, [&]<typename IdxT>() -> frame::Status {
    if constexpr (std::is_same_v<IdxT, std::int32_t> || std::is_same_v<IdxT, std::int64_t>) {
      // typed vector 直接建立 IdxT 对象生命周期，避免 char 缓冲 reinterpret
      // 后解引用的严格别名/对象生命周期未定义行为。
      std::vector<IdxT> host_indices(static_cast<size_t>(count));
      const size_t bytes = host_indices.size() * sizeof(IdxT);
      FRAME_RETURN_IF_ERROR(cuda_status(
          cudaMemcpyAsync(host_indices.data(), indices.raw_data(), bytes, cudaMemcpyDeviceToHost,
                          stream),
          std::string(op_name) + " cuda kernel: indices D2H predcheck cudaMemcpyAsync"));
      FRAME_RETURN_IF_ERROR(cuda_status(
          cudaStreamSynchronize(stream),
          std::string(op_name) + " cuda kernel: indices D2H predcheck cudaStreamSynchronize"));
      for (int64_t i = 0; i < count; ++i) {
        const int64_t idx = static_cast<int64_t>(host_indices[static_cast<size_t>(i)]);
        if (idx < 0 || idx >= bound) {
          return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                                     "op '" + std::string(op_name) + "' cuda kernel indices[" +
                                         std::to_string(i) + "]=" + std::to_string(idx) +
                                         " is out of range for " + std::string(bound_description) +
                                         "=" + std::to_string(bound));
        }
      }
    }
    return frame::Status::ok();
  });
}

}  // namespace frame::backends::cuda
