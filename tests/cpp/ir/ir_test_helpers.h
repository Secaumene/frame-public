#pragma once
// ir 层测试专用小工具:构造 TensorType 的辅助函数,供 tests/cpp/ir/ 下各测试文件
// 复用,避免每个文件重复相同的样板代码(先搜后写:本目录仅此一处需要该构造)。

#include <cstdint>
#include <initializer_list>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/ir/node.h>

namespace frame::ir::testing {

// 构造一个 float32、行主序、给定 shape/device 的 TensorType。device 缺省为
// cpu:0——TensorType::device 只是数据字段,与 HAL 后端是否已注册无关,测试
// 全程不依赖任何已注册后端(纯主机内存)。
inline TensorType MakeFloat32Type(std::initializer_list<int64_t> dims,
                                  Device device = cpu_device()) {
  TensorType type;
  type.dtype = DType::of<float>();
  type.shape = Shape(dims);
  type.layout = Layout::kRowMajor;
  type.device = device;
  return type;
}

}  // namespace frame::ir::testing
