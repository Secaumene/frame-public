// Tensor 值语义张量句柄的实现单元。
// 头文件声明了若干非内联成员,此处提供实现。

#include <cstddef>
#include <limits>
#include <utility>

#include <frame/core/tensor.h>

namespace frame {

Result<Tensor> Tensor::empty(const Shape& shape, DType dtype, Device device,
                             hal::Allocator& allocator) {
  Status verify_status = shape.verify();
  if (!verify_status.is_ok()) {
    return verify_status;
  }

  // verify() 通过后各维均非负,numel 理应非负;此处仍防御性重校验一次
  // (调用方不变量之外的最后一道防线,而非信任 verify() 的唯一实现)。
  const int64_t numel_signed = shape.numel();
  if (numel_signed < 0) {
    return Status::make(ErrorCode::kInvalidArgument, "Tensor.empty: numel must be non-negative");
  }
  const size_t numel = static_cast<size_t>(numel_signed);
  const size_t itemsize = dtype.itemsize();

  // numel == 0 允许(空张量);nbytes 随之为 0,Storage::allocate 对 0 字节
  // 请求仍构造合法 Storage(data 为 nullptr),不下沉到具体 allocator 实现。
  // 乘法前先做溢出检查,避免 numel * itemsize 环绕成小值而低估实际分配需求。
  if (itemsize != 0 && numel > std::numeric_limits<size_t>::max() / itemsize) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "Tensor.empty: numel * itemsize overflows size_t");
  }
  const size_t nbytes = numel * itemsize;
  Result<std::shared_ptr<Storage>> storage_result =
      Storage::allocate(allocator, nbytes, kDefaultAlignment, device);
  if (!storage_result.is_ok()) {
    return storage_result.status();
  }

  Tensor tensor;
  tensor.storage_ = storage_result.value();
  tensor.shape_ = shape;
  tensor.strides_ = row_major_strides(shape);
  tensor.offset_ = 0;
  tensor.dtype_ = dtype;
  return tensor;
}

Tensor Tensor::from_storage_slice(std::shared_ptr<Storage> storage, size_t byte_offset,
                                  const Shape& shape, DType dtype, Device device) {
  FRAME_CHECK(storage != nullptr);
  FRAME_CHECK(device == storage->device());

  const size_t itemsize = dtype.itemsize();
  // 先断言 itemsize 非 0 再做整除校验,防止理论边界上的对 0 取模。
  FRAME_CHECK(itemsize > 0);
  FRAME_CHECK(byte_offset % itemsize == 0);

  const int64_t numel_signed = shape.numel();
  FRAME_CHECK(numel_signed >= 0);  // v0 静态 shape(ARCH-013),调用方保证
  const size_t numel = static_cast<size_t>(numel_signed);

  // 溢出防线(同 Tensor::empty 既有惯例):乘法前先检查 numel*itemsize 是否
  // 环绕,避免低估实际所需字节数从而让越界校验失真。
  FRAME_CHECK(itemsize == 0 || numel <= std::numeric_limits<size_t>::max() / itemsize);
  const size_t nbytes = numel * itemsize;
  // 越界防线(调用方保证,头文件声明处已述契约):切片不得超出 storage 容量。
  // 用减法而非 byte_offset + nbytes 比较,避免加法本身环绕(byte_offset 若为
  // 调用方传入的非法巨大值)造成越界校验失真。
  FRAME_CHECK(byte_offset <= storage->nbytes());
  FRAME_CHECK(nbytes <= storage->nbytes() - byte_offset);

  Tensor tensor;
  tensor.shape_ = shape;
  tensor.strides_ = row_major_strides(shape);
  tensor.offset_ = static_cast<int64_t>(byte_offset / itemsize);
  tensor.dtype_ = dtype;
  tensor.storage_ = std::move(storage);
  return tensor;
}

Device Tensor::device() const {
  // 归属设备取自 Storage;storage_ 为空时返回默认设备(空后端键)。
  return storage_ ? storage_->device() : Device{};
}

void* Tensor::raw_data() {
  if (storage_ == nullptr) return nullptr;
  void* base = storage_->data();
  if (base == nullptr) return nullptr;
  // offset_ 单位为元素个数,乘 itemsize() 换算为字节偏移(见 tensor.h 成员注释)。
  return static_cast<char*>(base) + static_cast<size_t>(offset_) * dtype_.itemsize();
}

const void* Tensor::raw_data() const {
  if (storage_ == nullptr) return nullptr;
  const void* base = storage_->data();
  if (base == nullptr) return nullptr;
  return static_cast<const char*>(base) + static_cast<size_t>(offset_) * dtype_.itemsize();
}

Tensor Tensor::view(const Shape& new_shape) const {
  // v0 仅支持 contiguous 视图:仅校验元素总数一致,不支持改变内存布局语义的
  // 非 contiguous 视图(如任意 stride 重排)。
  FRAME_CHECK(new_shape.verify().is_ok());
  FRAME_CHECK(new_shape.numel() == shape_.numel());

  Tensor result;
  result.storage_ = storage_;
  result.shape_ = new_shape;
  result.strides_ = row_major_strides(new_shape);
  result.offset_ = offset_;
  result.dtype_ = dtype_;
  return result;
}

}  // namespace frame
