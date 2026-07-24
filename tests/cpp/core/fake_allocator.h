#pragma once
// core 层测试专用假分配器:host 堆内存 + 分配/释放计数,供 storage/tensor 单测复用。
// cpu 参考后端的真实 Allocator 要到 M4 才落地(见 src/backends/cpu/cpu_backend.cpp
// 中尚未实现的待办事项),本文件不依赖任何已注册后端,仅依赖 frame::hal::Allocator
// 这一 HAL 接口。

#include <cstddef>
#include <new>
#include <unordered_map>

#include <frame/core/status.h>
#include <frame/hal/allocator.h>

namespace frame::testing {

// 简单堆内存分配器:用带对齐版本的 operator new/delete 满足调用方的 alignment
// 要求;按指针记录各自的分配对齐值,释放时取回并调用匹配的带对齐 delete 重载
// (避免 new/delete 版本不匹配导致的未定义行为)。分配一律走 nothrow 重载,
// 失败(返回 nullptr)时转换为 Status(kOutOfMemory),不抛异常——示范
// include/frame/hal/allocator.h::allocate() 的"失败返回 Status"契约(CPP-020,
// 核心库/HAL 边界禁用异常)。
class FakeAllocator final : public frame::hal::Allocator {
 public:
  frame::Result<void*> allocate(size_t bytes, size_t alignment) override {
    ++allocate_count_;
    void* ptr = ::operator new(bytes, std::align_val_t(alignment), std::nothrow);
    if (ptr == nullptr) {
      return frame::Status::make(frame::ErrorCode::kOutOfMemory,
                                 "FakeAllocator::allocate: out of memory");
    }
    alignment_by_ptr_[ptr] = alignment;
    last_allocated_ = ptr;
    return ptr;
  }

  void deallocate(void* ptr) override {
    ++deallocate_count_;
    const auto it = alignment_by_ptr_.find(ptr);
    const size_t alignment =
        (it != alignment_by_ptr_.end()) ? it->second : alignof(std::max_align_t);
    if (it != alignment_by_ptr_.end()) alignment_by_ptr_.erase(it);
    ::operator delete(ptr, std::align_val_t(alignment));
  }

  int allocate_count() const { return allocate_count_; }
  int deallocate_count() const { return deallocate_count_; }
  void* last_allocated() const { return last_allocated_; }

 private:
  int allocate_count_ = 0;
  int deallocate_count_ = 0;
  void* last_allocated_ = nullptr;
  std::unordered_map<void*, size_t> alignment_by_ptr_;
};

}  // namespace frame::testing
