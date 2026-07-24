// bind_graph:Graph 构图入口的 pybind11 绑定(M12 决议点 B)。Value/Node 以
// 不透明句柄导出,仅供构图接线;全部返回句柄的绑定均用
// py::return_value_policy::reference_internal(等价于 reference +
// keep_alive<0,1>,裁决修订 2)绑定到所属 Graph,防止 Graph 析构后句柄悬垂,
// 也防止 pybind11 默认的 unique_ptr 持有策略误删 Graph 内部拥有的 Value。

#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

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

// add_graph_input 薄封装:list[int] shape + DType 枚举 + backend 字符串 ->
// 构造 TensorType(layout=kUnknown,布局由 layout_assignment pass 事后指派)
// 后调用 ir::Graph::add_graph_input。
ir::Value* add_graph_input(ir::Graph& graph, std::vector<int64_t> shape, DTypeCode dtype,
                           const std::string& backend) {
  hal::Backend* found = translate_result(hal::BackendRegistry::instance().get(backend));
  ir::TensorType type;
  type.dtype = DType(dtype);
  type.shape = Shape(std::move(shape));
  type.layout = ir::Layout::kUnknown;
  type.device = Device{found->name(), 0};
  return translate_result(graph.add_graph_input(type));
}

}  // namespace

void bind_graph(py::module_& m) {
  // 具名局部变量(而非匿名临时对象):规避 clang-tidy bugprone-unused-raii 误报
  // ——py::class_ 的注册效果发生在构造函数内部(向 m 登记类型),本身即为
  // "已使用",无需链式 .def() 调用。
  [[maybe_unused]] const py::class_<ir::Value> value_type(m, "Value",
                                                          "SSA 值句柄(不透明,仅供构图接线)。");
  [[maybe_unused]] const py::class_<ir::Node> node_type(m, "Node",
                                                        "算子节点句柄(不透明,仅供构图接线)。");

  py::class_<ir::Graph>(m, "Graph", "静态计算图句柄(构图入口,铁律 #1①)。")
      .def(py::init<std::string>(), py::arg("name") = std::string(), "按名字构造一个空图。")
      .def_property_readonly(
          "name", [](const ir::Graph& graph) { return std::string(graph.name()); }, "图名字。")
      .def("add_graph_input", &add_graph_input, py::arg("shape"), py::arg("dtype"),
           py::arg("backend") = "cpu", py::return_value_policy::reference_internal,
           "新增一个图输入,返回其唯一输出 Value 句柄(生命周期绑定本图)。")
      .def(
          "mark_output",
          [](ir::Graph& graph, ir::Value* value) { translate_status(graph.mark_output(value)); },
          py::arg("value"), "登记图输出(value 须属于本图,否则报错)。");
}

}  // namespace frame::python_bindings
