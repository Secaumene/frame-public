// translate_status 实现单元(声明与契约见 translate_status.h)。
//
// GIL 安全性说明:本函数可能在 py::call_guard<py::gil_scoped_release> 释放
// GIL 期间被调用(compile/Executable.run/Tensor.to 等函数体内,PY-010 五类)。
// 失败路径需要调用 Python C API(py::set_error/throw py::error_already_set),
// 这些调用必须在持锁状态下进行,故显式经 py::gil_scoped_acquire 确保——该
// guard 可重入(即便调用方本就持锁,构造/析构仍安全),因此本函数在持锁或
// 释放两种上下文中调用结果一致。成功路径(status.is_ok())不触碰任何 Python
// C API,直接返回,不产生额外 GIL 开销。

#include "translate_status.h"

#include <pybind11/pybind11.h>
#include <string>

namespace frame::python_bindings {

namespace py = pybind11;

void translate_status(const Status& status) {
  if (status.is_ok()) return;

  const py::gil_scoped_acquire gil;
  const std::string message(status.message());

  // PY-030 映射表(唯一权威:docs/standards/python-binding.md 第5章);未列出
  // 的错误码(含 kAlreadyExists/kDeviceError)一律 RuntimeError。异常消息取
  // status.message() 原文,不翻译/不二次拼接(LANG-005)。
  switch (status.code()) {
    case ErrorCode::kInvalidArgument:
      py::set_error(PyExc_ValueError, message.c_str());
      break;
    case ErrorCode::kNotFound:
      py::set_error(PyExc_KeyError, message.c_str());
      break;
    case ErrorCode::kUnimplemented:
      py::set_error(PyExc_NotImplementedError, message.c_str());
      break;
    case ErrorCode::kOutOfMemory:
      py::set_error(PyExc_MemoryError, message.c_str());
      break;
    case ErrorCode::kOk:
    case ErrorCode::kAlreadyExists:
    case ErrorCode::kInternal:
    case ErrorCode::kDeviceError:
      py::set_error(PyExc_RuntimeError, message.c_str());
      break;
  }
  throw py::error_already_set();
}

// GIL 安全性同 translate_status():可重入的 gil_scoped_acquire,使本函数在
// 持锁或释放两种上下文中调用结果一致(当前唯一调用点 DataLoader.__next__
// 不释放 GIL,此处按同一惯例防御式处理,不假设调用方状态)。
void raise_stop_iteration() {
  const py::gil_scoped_acquire gil;
  throw py::stop_iteration();
}

}  // namespace frame::python_bindings
