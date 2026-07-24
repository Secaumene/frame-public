// Intel GPU 设备内存分配器实现单元(骨架;cpu-only 下不参与构建)。

#include <cstddef>

#include "intel_gpu_backend.h"

namespace frame::backends::intel_gpu {

Result<void*> IntelGpuAllocator::allocate(size_t /*bytes*/, size_t /*alignment*/) {
  // TODO(FRAME-IMPL): 经 USM sycl::malloc_device 分配设备内存。
  //   参考:docs/backends/intel-gpu.md。完成判据:intel-gpu preset 下分配/释放冒烟用例通过。
  return FRAME_UNIMPLEMENTED();
}

void IntelGpuAllocator::deallocate(void* /*ptr*/) {
  // TODO(FRAME-IMPL): 经 sycl::free 释放。参考:docs/backends/intel-gpu.md。
  //   完成判据:与 allocate 配对无泄漏。
}

}  // namespace frame::backends::intel_gpu
