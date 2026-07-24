// bind_autograd:训练 API(反向图 / SGD 更新图构建)的 pybind11 绑定(积压批次
// ②)。薄壳纪律(PY-001):函数体只做参数转换 + 调
// compiler::build_backward_graph / compiler::build_sgd_update_graph + 错误
// 转换(经 translate_status)。二者均不在 PY-010 五类
// (compile/run/synchronize/memcpy/allocate)之列,不释放 GIL(PY-011)。

#include <cstdint>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <utility>
#include <vector>

#include <frame/compiler/autograd.h>
#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/hal/backend.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>

#include "translate_status.h"

namespace frame::python_bindings {

namespace py = pybind11;

namespace {

// 返回值取 unique_ptr<Graph> 而非按值返回 Graph 本身的说明(二者对 Python
// 调用方完全等价:pybind11 对 unique_ptr 返回值直接把指针过户给新建的 Graph
// Python 实例,与"按值返回、移动构造进新实例"观感一致,均产出一个持有独立
// Graph 的 Python 对象;bind_graph.cpp 已按值类型绑定 Graph,持有语义不变)。
// 取此形态是规避 GCC 13 + libstdc++ 13 + pybind11 的一处工具链级 SFINAE 陷阱
// (实测确认,不改动 include/frame/ir/graph.h 任何公开声明):Graph 含
// vector<unique_ptr<Node>> 成员,其隐式拷贝构造函数按语言规则不会在"声明层"
// 被判定为 deleted(vector 的拷贝构造函数本身总是声明存在,只在函数体实例化
// 时才因 unique_ptr 不可拷贝而失败),致使 std::is_copy_constructible<Graph>
// 被判定为 true;pybind11 按值返回类型时据此试探性实例化 Graph 拷贝构造函数体
// 以生成 make_copy_constructor,触发 libstdc++ 13 新增的 constexpr vector
// 拷贝路径内部 static_assert 硬性失败(非 SFINAE 可吸收的替换失败,编译期
// 报错)。unique_ptr<Graph> 返回值只需 Graph 的移动构造(隐式生成,合法),
// 规避该拷贝构造探测路径。
using GraphPtr = std::unique_ptr<ir::Graph>;

// build_backward_graph 薄封装:list[int] -> vector<int32_t>(经 span 隐式转换
// 传给 C++ 入口)+ 错误转换。
GraphPtr py_build_backward_graph(const ir::Graph& forward, int32_t loss_output_index,
                                 std::vector<int32_t> wrt_input_indices) {
  return std::make_unique<ir::Graph>(translate_result(
      compiler::build_backward_graph(forward, loss_output_index, wrt_input_indices)));
}

// build_sgd_update_graph 薄封装:逐 (shape, dtype) tuple 构造
// ir::TensorType{dtype, Shape(shape), Layout::kUnknown, Device{backend,0}}
// (device 由 backend 参数统一指派,与 bind_ops.cpp::constant 的 backend 字符
// 串处理同构)。
GraphPtr py_build_sgd_update_graph(
    const std::vector<std::pair<std::vector<int64_t>, DTypeCode>>& param_types,
    double learning_rate, const std::string& backend) {
  hal::Backend* found = translate_result(hal::BackendRegistry::instance().get(backend));
  const Device device{found->name(), 0};
  std::vector<ir::TensorType> types;
  types.reserve(param_types.size());
  for (const auto& [shape, dtype] : param_types) {
    types.push_back(ir::TensorType{DType(dtype), Shape(shape), ir::Layout::kUnknown, device});
  }
  return std::make_unique<ir::Graph>(
      translate_result(compiler::build_sgd_update_graph(types, learning_rate)));
}

}  // namespace

void bind_autograd(py::module_& m) {
  m.def("build_backward_graph", &py_build_backward_graph, py::arg("forward"),
        py::arg("loss_output_index"), py::arg("wrt_input_indices"),
        "由前向图生成反向训练图:loss_output_index 选择一个标量输出;图输出依次为 "
        "[forward_outputs..., grad(wrt_input_indices[0]), "
        "grad(wrt_input_indices[1]), ...](原输出前缀保持原序,梯度按给定顺序追加);"
        "不在 wrt_input_indices 中的 forward 输入按停止梯度处理,不引入任何 IR 标记;"
        "wrt_input_indices 中不在 loss 依赖链上的输入按惯例补零梯度(均见 "
        "docs/architecture/autograd.md 第2章)。");
  m.def("build_sgd_update_graph", &py_build_sgd_update_graph, py::arg("param_types"),
        py::arg("learning_rate"), py::arg("backend") = "cpu",
        "构建 SGD 参数更新图:param_types 为 (shape, dtype) 二元组列表,图输入按位 = "
        "[param_0..param_{n-1}, grad_0..grad_{n-1}],图输出按位 = "
        "[new_param_0..new_param_{n-1}],与 param 顺序一一对应;learning_rate 经 constant "
        "烘焙进图(v0 固定学习率,docs/architecture/autograd.md 第6章)。");
}

}  // namespace frame::python_bindings
