#pragma once
// Allocator:后端内存分配器(HAL 白名单虚函数,判定规则见 include/frame/hal/backend.h 头部)。
// 是否池化由后端自行决定,接口不变。

#include <cstddef>

#include <frame/core/macros.h>
#include <frame/core/status.h>

namespace frame::hal {

// Allocator:设备内存分配器抽象。
class FRAME_API Allocator {
 public:
  virtual ~Allocator() = default;

  // 分配 bytes 字节且满足 alignment 对齐;失败返回 Status。
  virtual Result<void*> allocate(size_t bytes, size_t alignment) = 0;
  // 释放先前由本分配器分配的内存。
  virtual void deallocate(void* ptr) = 0;
};

}  // namespace frame::hal
