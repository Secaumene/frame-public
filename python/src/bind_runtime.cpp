// bind_runtime:compile()/Executable.run() 的 pybind11 绑定(M12 决议点 B/E)。
// Stream/Event 不导出(v0,决议点 E):Executable.run 内部经
// runtime::run_with_allocated_outputs 编排 create_stream/run/synchronize,
// 对 Python 调用方隐藏。BoundExecutable 是绑定层内部小包装(非 HAL 公开
// 类型):hal::Executable 本身不携带其编译目标的后端名,而
// run_with_allocated_outputs 需要该名字才能在正确设备上预分配输出——本包装
// 把 compile() 时已知的 backend 名字与 Executable 一并存留,供 run() 使用。

#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include <frame/core/tensor.h>
#include <frame/hal/backend.h>
#include <frame/hal/executable.h>
#include <frame/ir/graph.h>
#include <frame/runtime/compile.h>

#include "translate_status.h"

namespace frame::python_bindings {

namespace py = pybind11;

namespace {

struct BoundExecutable {
  std::shared_ptr<hal::Executable> executable;
  std::string backend_name;
};

BoundExecutable compile_graph(const ir::Graph& graph, const std::string& backend) {
  std::shared_ptr<hal::Executable> executable =
      translate_result(runtime::compile(graph, backend, hal::CompileOptions{}));
  return BoundExecutable{std::move(executable), backend};
}

std::vector<Tensor> run_executable(BoundExecutable& self, const std::vector<Tensor>& inputs) {
  return translate_result(
      runtime::run_with_allocated_outputs(*self.executable, self.backend_name, inputs));
}

}  // namespace

void bind_runtime(py::module_& m) {
  py::class_<BoundExecutable>(m, "Executable", "整图编译产物的执行句柄(compile() 的返回值)。")
      .def("run", &run_executable, py::arg("inputs"), py::call_guard<py::gil_scoped_release>(),
           "按编译期签名执行整图,预分配并返回输出张量列表。");

  m.def("compile", &compile_graph, py::arg("graph"), py::arg("backend"),
        py::call_guard<py::gil_scoped_release>(), "把图编译为可执行句柄(整图编译路径,铁律 #1①)。");
}

}  // namespace frame::python_bindings
