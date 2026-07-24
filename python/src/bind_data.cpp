// bind_data:frame::data 的 TensorDataset/DataLoaderOptions/DataLoader 的
// pybind11 绑定(M20 批2 Task5,docs/architecture/nn-design.md §6)。薄壳纪律
// (PY-001):函数体只做参数转换 + 调用 C++ 公开 API(data::TensorDataset::create/
// data::DataLoader::create/next)+ 错误转换(经 translate_status,PY-030)。
// DataLoader 迭代面实现 Python 迭代协议(__iter__/__next__):epoch 尽头按 C++
// next() 契约(include/frame/data/dataloader.h 头注释①,推进到下一 epoch 的
// 哨兵)转换为 StopIteration——该异常经 translate_status.h 新增的唯一入口
// raise_stop_iteration() 触发(该入口是本仓库唯一持有异常关键字的位置,
// PY-030)。allocator 由绑定层内部经 BackendRegistry 取
// cpu 后端注入(先例:bind_interop.cpp::py_load_onnx_weights、
// bind_core.cpp::from_numpy)。构造函数与 __next__ 均不在 PY-010 五类
// (compile/run/synchronize/memcpy/allocate)之列,不释放 GIL(PY-011)。

#include <cstdint>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/data/dataloader.h>
#include <frame/data/dataset.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>

#include "translate_status.h"

namespace frame::python_bindings {

namespace py = pybind11;

namespace {

// TensorDataset 构造薄封装:校验交由 data::TensorDataset::create,失败经
// translate_result 转换。
data::TensorDataset py_make_tensor_dataset(std::vector<Tensor> columns) {
  return translate_result(data::TensorDataset::create(std::move(columns)));
}

// DataLoaderOptions 构造薄封装:四字段聚合初始化,字段序与
// include/frame/data/dataloader.h 声明序一致。
data::DataLoaderOptions make_dataloader_options(int64_t batch_size, bool shuffle, uint64_t seed,
                                                bool drop_last) {
  return data::DataLoaderOptions{batch_size, shuffle, seed, drop_last};
}

// DataLoader 的 Python 侧包装:持有值语义 data::DataLoader,实现迭代协议
// (非 C++ 公开 API 类型,仅供本绑定单元内部使用,同 bind_runtime.cpp::
// BoundExecutable 先例)。
class PyDataLoader {
 public:
  PyDataLoader(data::TensorDataset dataset, data::DataLoaderOptions options)
      : loader_(translate_result(data::DataLoader::create(std::move(dataset), options))) {}

  PyDataLoader& iter() { return *this; }

  // 产出下一批;当前 epoch 批序耗尽时抛 StopIteration(dataloader.h 头注释①
  // 哨兵语义:C++ next() 已在返回 nullopt 的同时把内部状态推进到下一 epoch)。
  std::vector<Tensor> next() {
    hal::Backend* backend = translate_result(hal::BackendRegistry::instance().get(kCpuBackendName));
    hal::Allocator* allocator = backend->allocator(cpu_device());
    if (allocator == nullptr) {
      translate_status(Status::make(ErrorCode::kInternal,
                                    "DataLoader.__next__: cpu backend returned a null allocator"));
    }
    std::optional<std::vector<Tensor>> batch = translate_result(loader_.next(*allocator));
    if (!batch.has_value()) {
      raise_stop_iteration();
    }
    return std::move(*batch);
  }

 private:
  data::DataLoader loader_;
};

}  // namespace

void bind_data(py::module_& m) {
  py::class_<data::TensorDataset>(m, "TensorDataset",
                                  "一组等长「列」张量(样本维为 axis0),各列须驻 cpu 后端"
                                  "(docs/architecture/nn-design.md ARCH-076)。")
      .def(py::init(&py_make_tensor_dataset), py::arg("columns"),
           "按列张量列表构造(逐条校验:非空/各列 rank>=1/各列驻 cpu 后端/各列 "
           "axis0 尺寸一致)。");

  py::class_<data::DataLoaderOptions>(m, "DataLoaderOptions", "DataLoader 构造选项。")
      .def(py::init(&make_dataloader_options), py::arg("batch_size") = 1,
           py::arg("shuffle") = false, py::arg("seed") = 0, py::arg("drop_last") = false,
           "构造批迭代选项。");

  py::class_<PyDataLoader>(m, "DataLoader",
                           "绑定一个 TensorDataset + DataLoaderOptions 的批迭代器(Python 迭代协议,"
                           "__next__ 产出 list[Tensor])。")
      .def(py::init<data::TensorDataset, data::DataLoaderOptions>(), py::arg("dataset"),
           py::arg("options"), "校验 options.batch_size >= 1 后就绪 epoch 0。")
      .def("__iter__", &PyDataLoader::iter, py::return_value_policy::reference_internal,
           "返回自身。")
      .def("__next__", &PyDataLoader::next,
           "产出下一批 list[Tensor];当前 epoch 批序耗尽时抛 StopIteration 并推进到"
           "下一 epoch(dataloader.h 头注释①)。");
}

}  // namespace frame::python_bindings
