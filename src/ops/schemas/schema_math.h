#pragma once
// schema 侧共享的小型几何算术工具(M21,批3 T4;M22 批4 T3 新增
// make_constant_splat)。收敛动机(铁律 5):conv.cpp 与 pool.cpp 的 geometry
// 求解同需 floor 语义;elementwise.cpp(tanh/rsqrt 梯度)/sequence.cpp
// (softmax/layer_norm 梯度)/shape.cpp(slice 零段梯度)/gather.cpp(零梯度)
// 同需"构造 shape 全形常量节点"这一基元(pool.cpp 已有的文件局部版本先于本次
// 扩容存在,形态相同但因非本批触及文件、不改动之,新增调用点一律改用本文件
// 单份实现,不再各自复制)。仅供 src/ops/schemas/ 内部包含,不入公开 API。

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/constant_utils.h>
#include <frame/ops/graph_builder.h>

namespace frame::ops::schemas {

// 已经通过 Shape::verify() 的静态 shape 是否能用 int64_t 安全表示元素总数。
// Shape::numel() 的公开签名没有错误通道，公开 schema 在构造常量或进入分配
// 路径前先用本函数拒绝极端乘积，避免有符号整数连乘溢出。
inline bool static_shape_numel_fits_int64(const Shape& shape) {
  int64_t total = 1;
  for (int64_t dimension : shape.dims()) {
    if (dimension < 0) return false;
    if (dimension != 0 && total > std::numeric_limits<int64_t>::max() / dimension) return false;
    total *= dimension;
  }
  return true;
}

// floor 除法(除数恒正,stride/KH/KW 经 schema 校验后必 >= 1):C++ 内置 `/`
// 对负被除数是向零截断而非数学 floor,而分子(H+2p-KH 等)在 kernel 大于已
// 填充输入时可为负;此处显式改判 floor 语义,否则「输出维必须 >= 1」的校验
// 在分子为负且不整除时会被截断语义误判为非负。
inline int64_t floor_div_positive_denominator(int64_t numerator, int64_t denominator) {
  const int64_t quotient = numerator / denominator;
  const int64_t remainder = numerator % denominator;
  return (remainder < 0) ? quotient - 1 : quotient;
}

// 构造 shape 全形、值全为 fill_value 的 constant 节点(square_gradient 的
// constant(2)先例同机制)。dtype 既可传浮点档也可传整数档(v0 constant 白名单
// 见 include/frame/ops/constant_utils.h::is_constant_dtype_supported)。
inline Result<ir::Node*> make_constant_splat(ir::Graph& graph, const Shape& shape, DType dtype,
                                             Device device, double fill_value) {
  const int64_t numel = shape.numel();
  const std::vector<double> values(static_cast<size_t>(numel), fill_value);
  const AttrMap attrs{
      {"value", values},
      {"shape", shape},
      {"dtype", dtype},
  };
  return create_node_with_inferred_types(graph, kConstantOpName, device, attrs);
}

}  // namespace frame::ops::schemas
