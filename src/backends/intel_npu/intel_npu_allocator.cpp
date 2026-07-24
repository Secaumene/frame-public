// Intel NPU 分配器实现单元(骨架;cpu-only 下不参与构建)。
// 注意:NPU 侧内存由 OpenVINO 托管,本分配器承担宿主内存 + import 语义
// (见 docs/backends/intel-npu.md 第 4 章 HAL 映射表)。

#include <cstddef>

#include "intel_npu_backend.h"

namespace frame::backends::intel_npu {

Result<void*> IntelNpuAllocator::allocate(size_t /*bytes*/, size_t /*alignment*/) {
  // TODO(FRAME-IMPL): 分配宿主内存并按 OpenVINO tensor import 语义对接。
  //   参考:docs/backends/intel-npu.md。完成判据:intel-npu preset 下分配/释放冒烟用例通过。
  return FRAME_UNIMPLEMENTED();
}

void IntelNpuAllocator::deallocate(void* /*ptr*/) {
  // TODO(FRAME-IMPL): 释放宿主内存。参考:docs/backends/intel-npu.md。
  //   完成判据:与 allocate 配对无泄漏。
}

}  // namespace frame::backends::intel_npu
