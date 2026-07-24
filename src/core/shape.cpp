// Shape / Strides 的实现单元。

#include <string>

#include <frame/core/shape.h>

namespace frame {

Status Shape::verify() const {
  for (int64_t d : dims_) {
    if (d == kDynamicDim) {
      // v0 仅支持静态 shape:含动态维一律拒绝(ARCH-013/ARCH-044 口径)。
      return Status::make(ErrorCode::kInvalidArgument,
                          "Shape.verify: dynamic dimension (kDynamicDim) is not supported "
                          "in v0 static-shape mode (ARCH-013/ARCH-044)");
    }
    if (d < 0) {
      // kDynamicDim(-1)以外的其余负维一律非法,消息含具体维度值便于定位。
      return Status::make(ErrorCode::kInvalidArgument,
                          "Shape.verify: invalid negative dimension " + std::to_string(d));
    }
  }
  return Status::ok();
}

std::string Shape::to_string() const {
  std::string text = "[";
  for (size_t i = 0; i < dims_.size(); ++i) {
    if (i != 0) text += ", ";
    text += std::to_string(dims_[i]);
  }
  text += ']';
  return text;
}

Strides row_major_strides(const Shape& shape) {
  const int64_t rank = shape.rank();
  std::vector<int64_t> strides(static_cast<size_t>(rank));
  int64_t running = 1;
  for (int64_t i = rank - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = running;
    running *= shape.dim(i);
  }
  return Strides(std::move(strides));
}

}  // namespace frame
