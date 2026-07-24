// 昇腾设备内存分配器实现单元(骨架;cpu-only 下不参与构建)。

#include <cstddef>

#include "ascend_backend.h"

namespace frame::backends::ascend {

Result<void*> AscendAllocator::allocate(size_t /*bytes*/, size_t /*alignment*/) {
  // TODO(FRAME-IMPL): 经 aclrtMalloc 分配设备内存。
  //   参考:docs/backends/ascend.md。完成判据:ascend preset 下分配/释放冒烟用例通过。
  return FRAME_UNIMPLEMENTED();
}

void AscendAllocator::deallocate(void* /*ptr*/) {
  // TODO(FRAME-IMPL): 经 aclrtFree 释放。参考:docs/backends/ascend.md。
  //   完成判据:与 allocate 配对无泄漏。
}

}  // namespace frame::backends::ascend
