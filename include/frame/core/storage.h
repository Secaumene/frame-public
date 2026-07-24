#pragma once
// Storage:设备内存块 —— 持有一段设备内存及其归属设备。
// 引用计数由持有者(Tensor)经 std::shared_ptr 管理;设备差异下沉到 HAL Allocator。
// 层次纪律:core 公共头对 hal 类型仅允许前向声明,不得 include hal 头
// (分配/回收的具体实现下沉到 src/core/storage.cpp,该实现单元允许 include
// include/frame/hal/allocator.h,详见 docs/architecture/overview.md 第 3 章)。
//
// 契约(与 Storage::allocate 配套,调用方与实现方须共同遵守):
// ①Allocator 由 Backend 持有、进程级长寿,其生命周期必须长于一切由它分配的 Storage;
// ②device 参数仅作元数据记录,allocator 与 device 的一致性由调用方保证(Storage 不校验);
// ③core 公共头对 hal 类型仅前向声明,不得触达 BackendRegistry 或任何具体后端实现。

#include <cstddef>
#include <memory>

#include <frame/core/device.h>
#include <frame/core/macros.h>
#include <frame/core/status.h>

namespace frame::hal {
class Allocator;  // 前向声明:core 公共头不 include hal 头(见上方层次纪律注释)
}  // namespace frame::hal

namespace frame {

// 设备内存块。值语义 API,内部持有非拥有的 Allocator 指针用于析构时回收;
// 不可拷贝(内存所有权唯一),仅可移动;跨持有者共享经 shared_ptr<Storage>。
class FRAME_API Storage {
 public:
  Storage() = default;
  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  Storage(Storage&&) noexcept;
  Storage& operator=(Storage&&) noexcept;
  ~Storage();

  // 经 allocator 分配 bytes 字节(满足 alignment 对齐)并绑定 device 元数据。
  // allocator 生命周期契约见本文件头部注释①。
  static Result<std::shared_ptr<Storage>> allocate(hal::Allocator& allocator, size_t bytes,
                                                   size_t alignment, Device device);

  void* data() { return data_; }
  const void* data() const { return data_; }
  size_t nbytes() const { return nbytes_; }
  Device device() const { return device_; }

 private:
  void* data_ = nullptr;                 // 设备内存首地址
  size_t nbytes_ = 0;                    // 字节数
  Device device_{};                      // 归属设备
  hal::Allocator* allocator_ = nullptr;  // 非拥有:用于析构时回收 data_
};

// 张量内存默认对齐(字节)。
inline constexpr size_t kDefaultAlignment = 64;

}  // namespace frame
