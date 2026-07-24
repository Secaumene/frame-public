// cuda_status 翻译 helper 的实现单元(声明见 cuda_status.h)。

#include "cuda_status.h"

#include <string>

namespace frame::backends::cuda {

Status cuda_status(cudaError_t error, std::string_view context) {
  if (error == cudaSuccess) return Status::ok();

  const ErrorCode code =
      (error == cudaErrorMemoryAllocation) ? ErrorCode::kOutOfMemory : ErrorCode::kInternal;
  return Status::make(code, std::string(context) + ": " + cudaGetErrorString(error));
}

}  // namespace frame::backends::cuda
