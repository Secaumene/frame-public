#pragma once
// 形状与步幅:Shape / Strides,均为值类型,无虚函数。
// v0 仅支持静态 shape:数据结构保留动态维哨兵位以便未来扩展,但 verify() 一律
// 拒绝动态维(ARCH-013 / ARCH-044,见 docs/architecture/ir-design.md)。

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include <frame/core/status.h>

namespace frame {

// 动态维哨兵:v0 下含此值的 Shape 会被 verify() 拒绝。
inline constexpr int64_t kDynamicDim = -1;

// Shape:各维尺寸的有序列表。
class Shape {
 public:
  Shape() = default;
  Shape(std::initializer_list<int64_t> dims) : dims_(dims) {}
  explicit Shape(std::vector<int64_t> dims) : dims_(std::move(dims)) {}

  int64_t rank() const { return static_cast<int64_t>(dims_.size()); }
  int64_t dim(int64_t i) const { return dims_[static_cast<size_t>(i)]; }
  const std::vector<int64_t>& dims() const { return dims_; }

  // 元素总数(各维连乘)。含动态维时返回 kDynamicDim。
  int64_t numel() const {
    int64_t total = 1;
    for (int64_t d : dims_) {
      if (d == kDynamicDim) return kDynamicDim;
      total *= d;
    }
    return total;
  }

  bool has_dynamic_dim() const {
    for (int64_t d : dims_) {
      if (d == kDynamicDim) return true;
    }
    return false;
  }

  // 校验:v0 仅支持静态 shape,含动态维(kDynamicDim)一律判定为错误
  // (拒绝动态维,ARCH-013/ARCH-044 口径);其余负维同样判定为错误
  // (非法维度值)。错误消息英文(LANG-005)。
  Status verify() const;

  // 调试/错误消息用的文本表示,格式 "[d0, d1, ...]"(空 shape 为 "[]")。
  // 仅供人读(错误消息、调试输出),不是序列化格式——文本序列化的规范 shape
  // 格式唯一权威见 include/frame/ir/serialization.h,与本函数无关、互不复用。
  std::string to_string() const;

  bool operator==(const Shape&) const = default;

 private:
  std::vector<int64_t> dims_;
};

// Strides:与 Shape 同秩的步幅(单位为元素个数)。
class Strides {
 public:
  Strides() = default;
  Strides(std::initializer_list<int64_t> strides) : strides_(strides) {}
  explicit Strides(std::vector<int64_t> strides) : strides_(std::move(strides)) {}

  int64_t rank() const { return static_cast<int64_t>(strides_.size()); }
  const std::vector<int64_t>& values() const { return strides_; }

  bool operator==(const Strides&) const = default;

 private:
  std::vector<int64_t> strides_;
};

// 按行优先(C 序,row-major)由 shape 推导默认 strides:最末维步幅为 1,
// 其余各维步幅为其右侧全部维尺寸的连乘(单位:元素个数,非字节)。
Strides row_major_strides(const Shape& shape);

}  // namespace frame
