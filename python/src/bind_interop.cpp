// bind_interop:ONNX 权重导入/导出的 pybind11 绑定(ADR-0013,决议③用户可用性)。
// 薄壳纪律(PY-001):函数体只做参数转换 + 调用 C++ 公开 API(interop::
// save_onnx_weights/load_onnx_weights)+ 错误转换。interop 自身仅依赖 core
// (include/frame/interop/onnx_weights.h 头注释);本绑定单元随 module.cpp 链接
// frame::frame 聚合库,故在 hal 依赖面内,经 BackendRegistry 取得 cpu 分配器
// 后按维护者裁决(方案 b)注入 load_onnx_weights——解析设备/分配器属"参数转换"
// 范畴,不是业务逻辑。save_onnx_weights/load_onnx_weights 不在 PY-010 五类
// (compile/run/synchronize/memcpy/allocate)内,不释放 GIL(PY-011)。

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <utility>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/interop/onnx_weights.h>

#include "translate_status.h"

namespace frame::python_bindings {

namespace py = pybind11;

namespace {

// dict[str, Tensor] -> vector<NamedTensor>,按插入序(py::dict 迭代顺序即
// Python dict 的插入顺序,CPython 3.7+ 语言保证)。
std::vector<interop::NamedTensor> dict_to_named_tensors(const py::dict& weights) {
  std::vector<interop::NamedTensor> named_tensors;
  named_tensors.reserve(weights.size());
  for (const auto& item : weights) {
    named_tensors.push_back(
        interop::NamedTensor{item.first.cast<std::string>(), item.second.cast<Tensor>()});
  }
  return named_tensors;
}

void py_save_onnx_weights(const std::string& path, const py::dict& weights) {
  translate_status(interop::save_onnx_weights(path, dict_to_named_tensors(weights)));
}

py::dict py_load_onnx_weights(const std::string& path) {
  hal::Backend* backend = translate_result(hal::BackendRegistry::instance().get(kCpuBackendName));
  hal::Allocator* allocator = backend->allocator(cpu_device());
  if (allocator == nullptr) {
    translate_status(Status::make(ErrorCode::kInternal,
                                  "load_onnx_weights: cpu backend returned a null allocator"));
  }
  std::vector<interop::NamedTensor> named_tensors =
      translate_result(interop::load_onnx_weights(path, *allocator));

  py::dict result;
  for (interop::NamedTensor& item : named_tensors) {
    result[py::str(item.name)] = py::cast(std::move(item.tensor));
  }
  return result;
}

}  // namespace

void bind_interop(py::module_& m) {
  m.def("save_onnx_weights", &py_save_onnx_weights, py::arg("path"), py::arg("weights"),
        "把 dict[str, Tensor] 写为一份最小合法 ONNX ModelProto(仅权重 initializer 子集,"
        "ADR-0013;每张量须 device='cpu' 且 dtype 属 float32/float16/bfloat16 白名单)。");
  m.def("load_onnx_weights", &py_load_onnx_weights, py::arg("path"),
        "从 path 读取 ONNX 权重 initializer 子集,返回 dict[str, Tensor](顺序与文件中出现"
        "顺序一致,ADR-0013)。");
}

}  // namespace frame::python_bindings
