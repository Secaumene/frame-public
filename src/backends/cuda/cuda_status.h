#pragma once
// cuda_status:单份 cudaError_t -> Status 翻译 helper(M11,阶段 B-5)。全部 CUDA
// Runtime API 返回值经本函数翻译,禁止各调用点各自 if/else 拼错误码
// (REUSE-002:单份翻译逻辑)。分流规则:cudaErrorMemoryAllocation ->
// kOutOfMemory;其余非 cudaSuccess 一律 kInternal。消息英文、含
// cudaGetErrorString(error) 与调用方传入的 context(定位用,如函数名/参数摘要,
// LANG-005)。

#include <string_view>

#include <frame/core/status.h>

#include <cuda_runtime.h>

namespace frame::backends::cuda {

// error == cudaSuccess 时返回 Status::ok();否则按上述规则翻译。
Status cuda_status(cudaError_t error, std::string_view context);

}  // namespace frame::backends::cuda
