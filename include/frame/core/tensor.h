#pragma once
// Tensor:轻量句柄(共享 Storage + 视图元数据),按值拷贝为浅拷贝。
// 判定规则:本类禁止虚函数;设备差异全部下沉到 Storage/Allocator(HAL)。

#include <cstdint>
#include <memory>
#include <type_traits>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/macros.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/storage.h>

namespace frame::hal {
class Allocator;  // 前向声明:core 公共头不 include hal 头(同 storage.h 契约)
}  // namespace frame::hal

namespace frame {

// Tensor:值语义张量句柄。多个 Tensor 可共享同一 Storage(视图)。
class FRAME_API Tensor {
 public:
  Tensor() = default;

  // 在指定设备上分配未初始化张量(经调用方显式传入的 allocator 分配 Storage)。
  // 注:按 device 查注册表自动取 allocator 的三参便捷版留待 M4 另行决议
  // (BackendRegistry 尚未落地,当前不预登接口先例)。
  static Result<Tensor> empty(const Shape& shape, DType dtype, Device device,
                              hal::Allocator& allocator);

  // 共享既有 Storage 的字节偏移切片(M9,决议点 D 覆盖版/裁决修订4):供 arena
  // 场景构造中间张量——多个 Tensor 共享同一 storage 的不同区间,经
  // shared_ptr<Storage> 引用计数自动托管生命周期,本函数不额外分配/拥有内存
  // (storage 参数按值传入即完成共享,调用方仍可继续持有自己的 shared_ptr
  // 副本)。byte_offset 是相对 storage 起始地址的字节偏移,内部按
  // dtype.itemsize() 换算为既有 offset_(元素个数)表示,要求整除
  // (FRAME_CHECK,不整除属调用方违反不变量,fatal 而非 Status,与 data<T>()
  // 既定纪律一致)。越界(byte_offset + shape.numel()*dtype.itemsize() 超出
  // storage->nbytes())同样由调用方保证,本函数以 FRAME_CHECK 兜底校验、不做
  // 静默截断。device 须与 storage->device() 一致(FRAME_CHECK,仅作调用方
  // 意图的显式声明与防御性核验,不改变 storage 归属设备)。
  static Tensor from_storage_slice(std::shared_ptr<Storage> storage, size_t byte_offset,
                                   const Shape& shape, DType dtype, Device device);

  const Shape& shape() const { return shape_; }
  const Strides& strides() const { return strides_; }
  DType dtype() const { return dtype_; }
  Device device() const;  // 返回归属设备(取自 Storage)
  int64_t numel() const { return shape_.numel(); }

  void* raw_data();
  const void* raw_data() const;

  // 强类型数据指针。内部校验 dtype_traits<T>::code == dtype().code(),
  // 不符即 FRAME_CHECK fatal(违反调用方不变量,不是可恢复错误,故不经 Status)。
  template <ScalarType T>
  T* data() {
    FRAME_CHECK(dtype_traits<std::remove_cv_t<T>>::code == dtype_.code());
    return static_cast<T*>(raw_data());
  }

  // 共享 Storage 的视图(不拷贝数据)。v0 仅支持 contiguous 视图:new_shape 必须
  // 与原 shape 元素总数相同,strides 按新 shape 行优先重推。
  Tensor view(const Shape& new_shape) const;

 private:
  std::shared_ptr<Storage> storage_;
  Shape shape_;
  Strides strides_;
  int64_t offset_ = 0;  // storage_ 内的元素偏移量(单位:元素个数,非字节)
  DType dtype_{DTypeCode::kFloat32};
};

}  // namespace frame
