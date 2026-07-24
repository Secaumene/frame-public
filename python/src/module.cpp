// Frame Python 绑定模块入口(M12:构图→编译→执行最小闭环)。
// 铁律 #2:python/ 仅做绑定,严禁核心逻辑(计算/调度/内存管理)。
// 模块名固定为 _core;纯 Python 包 frame 经 __init__.py 转发(见
// python/frame/__init__.py)。绑定源码按 PY-002 预判超 500 行直接模块化:
// bind_core.cpp(DType/Shape/Device/Tensor/numpy 互操作)、bind_graph.cpp
// (Graph/Value/Node)、bind_ops.cpp(七个面向用户算子)、bind_runtime.cpp
// (compile/Executable)、bind_interop.cpp(ONNX 权重导入/导出,ADR-0013)、
// bind_autograd.cpp(训练 API:反向图/SGD 更新图构建,积压批次②)、
// bind_nn.cpp(frame::nn Module 与 Linear/Relu/Sequential/MseLoss 工厂,
// M20 批2 Task5)、bind_data.cpp(frame::data TensorDataset/DataLoader,
// M20 批2 Task5),错误转换收敛于 translate_status.{h,cpp}(PY-030)。

#include <pybind11/pybind11.h>

namespace frame::python_bindings {

namespace py = pybind11;

// 各子绑定单元的入口,定义分别位于同名 .cpp(bind_core.cpp/bind_graph.cpp/
// bind_ops.cpp/bind_runtime.cpp/bind_interop.cpp/bind_autograd.cpp/
// bind_nn.cpp/bind_data.cpp)。
void bind_core(py::module_& m);
void bind_graph(py::module_& m);
void bind_ops(py::module_& m);
void bind_runtime(py::module_& m);
void bind_interop(py::module_& m);
void bind_autograd(py::module_& m);
void bind_nn(py::module_& m);
void bind_data(py::module_& m);

}  // namespace frame::python_bindings

PYBIND11_MODULE(_core, m) {
  m.doc() = "Frame C++ 核心的 pybind11 绑定(M12:构图→编译→执行最小闭环)。";

  frame::python_bindings::bind_core(m);
  frame::python_bindings::bind_graph(m);
  frame::python_bindings::bind_ops(m);
  frame::python_bindings::bind_runtime(m);
  frame::python_bindings::bind_interop(m);
  frame::python_bindings::bind_autograd(m);
  frame::python_bindings::bind_nn(m);
  frame::python_bindings::bind_data(m);
}
