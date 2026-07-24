#pragma once
// Status/Result<T> -> Python 异常转换的唯一入口(PY-030)。绑定层其余代码禁止
// 自行 throw 或另立转换逻辑:`grep -n 'throw' python/src/*.cpp` 的命中应仅落在
// translate_status.cpp(CPP-020 的绑定层例外)。

#include <utility>

#include <frame/core/status.h>

namespace frame::python_bindings {

// status 非 ok 时依 PY-030 映射表 throw 对应 Python 异常(消息取
// status.message() 原文,不翻译/不二次拼接,LANG-005);status.is_ok() 时
// 直接返回,不触碰任何 Python C API。失败路径内部经 py::gil_scoped_acquire
// 重新确保持锁(实现见 translate_status.cpp),故本函数在 GIL 已释放的上下文
// 中调用同样安全——五类 GIL 释放函数(compile/run/to 等,PY-010)的函数体
// 均可直接调用本函数做错误转换,无需调用方自行处理 GIL。
void translate_status(const Status& status);

// Result<T> 版本:status 非 ok 时 throw(同上,内部调用 translate_status),
// 否则移动返回其持有值。
template <typename T>
T translate_result(Result<T> result) {
  translate_status(result.status());
  return std::move(result.value());
}

// Python 迭代协议控制流信号(非 Status 错误,如 DataLoader.__next__ 的 epoch
// 尽头哨兵)的唯一 throw 入口——同 translate_status() 一并把绑定层全部 throw
// 收拢到本翻译单元(PY-030 判定方法:`grep -n 'throw' python/src/*.cpp` 命中
// 仅落在 translate_status.cpp)。实现见 translate_status.cpp。
[[noreturn]] void raise_stop_iteration();

}  // namespace frame::python_bindings
