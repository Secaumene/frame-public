// CPU 内存分配器实现单元(必须真编译进 cpu-only 构建)。

#include <cstddef>
#include <cstdlib>
#include <string>

#include "cpu_backend.h"

namespace frame::backends::cpu {

namespace {

// 判定 value 是否为 2 的幂(0 不算)。
bool is_power_of_two(size_t value) { return value != 0 && (value & (value - 1)) == 0; }

}  // namespace

Result<void*> CpuAllocator::allocate(size_t bytes, size_t alignment) {
  if (!is_power_of_two(alignment)) {
    return Status::make(ErrorCode::kInvalidArgument, "CpuAllocator::allocate: alignment " +
                                                         std::to_string(alignment) +
                                                         " must be a nonzero power of two");
  }
  if (bytes == 0) {
    return Status::make(ErrorCode::kInvalidArgument, "CpuAllocator::allocate: bytes must be > 0");
  }
  // std::aligned_alloc 要求 size 是 alignment 的整数倍,上取整满足该约束;
  // 上取整前先判溢出,防 bytes 接近 SIZE_MAX 时回绕导致静默欠分配。
  if (bytes > SIZE_MAX - (alignment - 1)) {
    return Status::make(ErrorCode::kOutOfMemory, "CpuAllocator::allocate: byte count " +
                                                     std::to_string(bytes) +
                                                     " overflows when rounded up to alignment");
  }
  const size_t rounded_bytes = ((bytes + alignment - 1) / alignment) * alignment;
  void* ptr = std::aligned_alloc(alignment, rounded_bytes);
  if (ptr == nullptr) {
    return Status::make(ErrorCode::kOutOfMemory, "CpuAllocator::allocate: out of memory for " +
                                                     std::to_string(bytes) + " byte(s)");
  }
  return ptr;
}

void CpuAllocator::deallocate(void* ptr) {
  // std::free(nullptr) 为合法空操作,容忍调用方传入 nullptr。
  std::free(ptr);
}

}  // namespace frame::backends::cpu
