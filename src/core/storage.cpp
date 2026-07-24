// Storage 设备内存块的实现单元。
// 本单元允许 include include/frame/hal/allocator.h(hal 纯接口头随 frame_core
// 发布,无链接依赖,详见 docs/architecture/overview.md 第 3 章 ARCH-001 处的
// 分层细化语句);core 公共头 storage.h 对 hal::Allocator 仅前向声明。

#include <utility>

#include <frame/core/storage.h>
#include <frame/hal/allocator.h>

namespace frame {

Storage::Storage(Storage&& other) noexcept
    : data_(other.data_),
      nbytes_(other.nbytes_),
      device_(other.device_),
      allocator_(other.allocator_) {
  // 移后源状态契约:整体重置为等价于默认构造(data_/nbytes_/device_/allocator_
  // 全部清零),避免出现 data()==nullptr 但 nbytes()>0 的不一致状态,
  // 也避免其析构时重复释放已被窃取的内存。
  other.data_ = nullptr;
  other.nbytes_ = 0;
  other.device_ = Device{};
  other.allocator_ = nullptr;
}

Storage& Storage::operator=(Storage&& other) noexcept {
  if (this == &other) return *this;
  // 先释放自有内存,再窃取 other 的资源,防止双重释放/泄漏。
  if (data_ != nullptr && allocator_ != nullptr) {
    allocator_->deallocate(data_);
  }
  data_ = other.data_;
  nbytes_ = other.nbytes_;
  device_ = other.device_;
  allocator_ = other.allocator_;
  // 移后源状态契约同移动构造:整体重置为等价于默认构造。
  other.data_ = nullptr;
  other.nbytes_ = 0;
  other.device_ = Device{};
  other.allocator_ = nullptr;
  return *this;
}

Storage::~Storage() {
  if (data_ != nullptr && allocator_ != nullptr) {
    allocator_->deallocate(data_);
  }
}

Result<std::shared_ptr<Storage>> Storage::allocate(hal::Allocator& allocator, size_t bytes,
                                                   size_t alignment, Device device) {
  auto storage = std::make_shared<Storage>();
  if (bytes == 0) {
    // 0 字节分配:仍构造合法 Storage,data_ 保持 nullptr,不下沉到 allocator
    // (避免依赖具体后端对 0 字节请求的行为)。
    storage->nbytes_ = 0;
    storage->device_ = device;
    return storage;
  }
  Result<void*> allocated = allocator.allocate(bytes, alignment);
  if (!allocated.is_ok()) {
    return allocated.status();
  }
  storage->data_ = allocated.value();
  storage->nbytes_ = bytes;
  storage->device_ = device;
  storage->allocator_ = &allocator;
  return storage;
}

}  // namespace frame
