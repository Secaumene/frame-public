// bind_ops:面向用户算子薄函数绑定(add/mul/relu/square/sum/matmul/constant,
// PY-021 暂缓期清单全量销项,M12 起全量执法;mse_loss 于 M17 随
// docs/architecture/autograd.md ARCH-064 新增注册,同一执法条款要求随注册
// 同变更绑定)。各函数 = 参数转换 + 调 ops::create_node_with_inferred_types +
// 错误转换(PY-001);均返回单个输出 Value(采纳建议⑥,便于链式
// mark_output),绑定处一律 py::return_value_policy::reference_internal 绑到
// graph(裁决修订 2)。不释放 GIL(PY-011:这些算子不在 PY-010 五类清单内)。

#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/hal/backend.h>
#include <frame/ir/attribute.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/constant_utils.h>
#include <frame/ops/graph_builder.h>

#include "translate_status.h"

namespace frame::python_bindings {

namespace py = pybind11;

namespace {

ir::Value* add(ir::Graph& graph, ir::Value* lhs, ir::Value* rhs) {
  ir::Node* node = translate_result(ops::create_node_with_inferred_types(graph, "add", {lhs, rhs}));
  return node->output(0);
}

ir::Value* mul(ir::Graph& graph, ir::Value* lhs, ir::Value* rhs) {
  ir::Node* node = translate_result(ops::create_node_with_inferred_types(graph, "mul", {lhs, rhs}));
  return node->output(0);
}

// selective_scan 五输入薄绑定:参数原样转交通用 C++ 构图入口。
ir::Value* selective_scan(ir::Graph& graph, ir::Value* x, ir::Value* a, ir::Value* b, ir::Value* c,
                          ir::Value* d) {
  ir::Node* node = translate_result(
      ops::create_node_with_inferred_types(graph, "selective_scan", {x, a, b, c, d}));
  return node->output(0);
}

// 代理阶跃薄绑定:alpha 作为必需 kDouble 属性原样转交通用构图入口。
ir::Value* heaviside_surrogate(ir::Graph& graph, ir::Value* x, double alpha) {
  const ops::AttrMap attrs{{"alpha", alpha}};
  ir::Node* node = translate_result(
      ops::create_node_with_inferred_types(graph, "heaviside_surrogate", {x}, attrs));
  return node->output(0);
}

// 公共 scatter_add 薄绑定:output_shape 转为 kShape 属性后转交通用构图入口。
ir::Value* scatter_add(ir::Graph& graph, ir::Value* updates, ir::Value* indices,
                       std::vector<int64_t> output_shape) {
  const ops::AttrMap attrs{{"output_shape", Shape(std::move(output_shape))}};
  ir::Node* node = translate_result(
      ops::create_node_with_inferred_types(graph, "scatter_add", {updates, indices}, attrs));
  return node->output(0);
}

ir::Value* relu(ir::Graph& graph, ir::Value* x) {
  ir::Node* node = translate_result(ops::create_node_with_inferred_types(graph, "relu", {x}));
  return node->output(0);
}

ir::Value* square(ir::Graph& graph, ir::Value* x) {
  ir::Node* node = translate_result(ops::create_node_with_inferred_types(graph, "square", {x}));
  return node->output(0);
}

ir::Value* matmul(ir::Graph& graph, ir::Value* lhs, ir::Value* rhs) {
  ir::Node* node =
      translate_result(ops::create_node_with_inferred_types(graph, "matmul", {lhs, rhs}));
  return node->output(0);
}

ir::Value* sum(ir::Graph& graph, ir::Value* x, std::vector<int64_t> axes) {
  const ops::AttrMap attrs{{"axes", ir::AttrValue{std::move(axes)}}};
  ir::Node* node = translate_result(ops::create_node_with_inferred_types(graph, "sum", {x}, attrs));
  return node->output(0);
}

// mse_loss(M17,ARCH-064):均方误差损失,标量(rank-0)输出;pred/target 两
// 输入,与 add/mul 等二元算子同构,复用同一绑定骨架。
ir::Value* mse_loss(ir::Graph& graph, ir::Value* pred, ir::Value* target) {
  ir::Node* node =
      translate_result(ops::create_node_with_inferred_types(graph, "mse_loss", {pred, target}));
  return node->output(0);
}

// constant 五参(graph/values/shape/dtype/backend)是其 schema 本身的属性面
// (value/shape/dtype 三个必需属性 + 0 输入无从推导的 device)决定的下限,非
// 可再精简的参数打包,PY-001 允许的例外(裁决修订/采纳建议③)。
ir::Value* constant(ir::Graph& graph, std::vector<double> values, std::vector<int64_t> shape,
                    DTypeCode dtype, const std::string& backend) {
  hal::Backend* found = translate_result(hal::BackendRegistry::instance().get(backend));
  const ops::AttrMap attrs{
      {"value", ir::AttrValue{std::move(values)}},
      {"shape", ir::AttrValue{Shape(std::move(shape))}},
      {"dtype", ir::AttrValue{DType(dtype)}},
  };
  ir::Node* node = translate_result(ops::create_node_with_inferred_types(
      graph, ops::kConstantOpName, Device{found->name(), 0}, attrs));
  return node->output(0);
}

}  // namespace

void bind_ops(py::module_& m) {
  const py::return_value_policy kToGraph = py::return_value_policy::reference_internal;

  m.def("add", &add, py::arg("graph"), py::arg("lhs"), py::arg("rhs"), kToGraph, "逐元素加法。");
  m.def("mul", &mul, py::arg("graph"), py::arg("lhs"), py::arg("rhs"), kToGraph, "逐元素乘法。");
  m.def("selective_scan", &selective_scan, py::arg("graph"), py::arg("x"), py::arg("a"),
        py::arg("b"), py::arg("c"), py::arg("d"), kToGraph, "沿最后一轴执行选择性状态扫描。");
  m.def("heaviside_surrogate", &heaviside_surrogate, py::arg("graph"), py::arg("x"),
        py::arg("alpha"), kToGraph, "逐元素代理阶跃:x>=0 输出 1,否则输出 0。");
  m.def("scatter_add", &scatter_add, py::arg("graph"), py::arg("updates"), py::arg("indices"),
        py::arg("output_shape"), kToGraph, "按 indices 将 updates 行累加到指定输出形状。");
  m.def("relu", &relu, py::arg("graph"), py::arg("x"), kToGraph, "逐元素 ReLU。");
  m.def("square", &square, py::arg("graph"), py::arg("x"), kToGraph, "逐元素平方。");
  m.def("matmul", &matmul, py::arg("graph"), py::arg("lhs"), py::arg("rhs"), kToGraph,
        "矩阵乘法(rank-2)。");
  m.def("sum", &sum, py::arg("graph"), py::arg("x"), py::arg("axes"), kToGraph,
        "沿 axes 求和归约(axes 为空表示全维归约)。");
  m.def("mse_loss", &mse_loss, py::arg("graph"), py::arg("pred"), py::arg("target"), kToGraph,
        "均方误差损失 mean((pred-target)^2),标量(rank-0)输出。");
  m.def("constant", &constant, py::arg("graph"), py::arg("values"), py::arg("shape"),
        py::arg("dtype"), py::arg("backend") = "cpu", kToGraph,
        "把常量值物化为图中一个 0 输入节点。");
}

}  // namespace frame::python_bindings
