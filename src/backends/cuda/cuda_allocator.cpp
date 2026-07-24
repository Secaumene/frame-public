// CUDA 设备内存分配器实现单元(声明与实现选择理由见 cuda_backend.h 头注释)。

#include <cstddef>
#include <cstdio>
#include <string>

#include "cuda_backend.h"
#include "cuda_status.h"

namespace frame::backends::cuda {

namespace {

// 判定 value 是否为 2 的幂(0 不算),与 src/backends/cpu/cpu_allocator.cpp 同一份判据。
bool is_power_of_two(size_t value) { return value != 0 && (value & (value - 1)) == 0; }

// cudaMalloc 对齐保证的保守观测口径【待查证】(来源:NVIDIA CUDA C++
// Programming Guide「Device Memory」章节未给出显式数值承诺,本机构建环境无
// 网络核实,维持积压批次③前的 256 字节保守值,见 cuda_backend.h 头注释)。
// 请求 alignment 超出该值时拒绝而非静默忽略(design-reviewer REVISE 闭环
// 采纳建议②)。
constexpr size_t kCudaAllocationAlignmentGuarantee = 256;

}  // namespace

Result<void*> CudaAllocator::allocate(size_t bytes, size_t alignment) {
  if (!is_power_of_two(alignment)) {
    return Status::make(ErrorCode::kInvalidArgument, "CudaAllocator::allocate: alignment " +
                                                         std::to_string(alignment) +
                                                         " must be a nonzero power of two");
  }
  if (alignment > kCudaAllocationAlignmentGuarantee) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "CudaAllocator::allocate: requested alignment " +
                            std::to_string(alignment) +
                            " exceeds the observed cudaMalloc guarantee of " +
                            std::to_string(kCudaAllocationAlignmentGuarantee) + " byte(s)");
  }
  if (bytes == 0) {
    return Status::make(ErrorCode::kInvalidArgument, "CudaAllocator::allocate: bytes must be > 0");
  }

  const cudaError_t set_device_error = cudaSetDevice(device_index_);
  if (set_device_error != cudaSuccess) {
    return cuda_status(set_device_error, "CudaAllocator::allocate: cudaSetDevice");
  }

  void* ptr = nullptr;
  const cudaError_t error = cudaMalloc(&ptr, bytes);
  if (error != cudaSuccess) {
    return cuda_status(error, "CudaAllocator::allocate: cudaMalloc");
  }
  return ptr;
}

void CudaAllocator::deallocate(void* ptr) {
  if (ptr == nullptr) return;
  const cudaError_t error = cudaFree(ptr);
  if (error != cudaSuccess) {
    // deallocate() 签名不带 Status(见 include/frame/hal/allocator.h),违例经
    // stderr 输出英文诊断(同 CpuBackend::allocator 对无 Status 出口的既有处理
    // 方式)。
    std::fprintf(stderr, "CudaAllocator::deallocate: cudaFree failed: %s\n",
                 cudaGetErrorString(error));
  }
}

}  // namespace frame::backends::cuda
