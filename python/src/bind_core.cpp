// bind_core:DType/Shape/Device/Tensor 与 numpy 互操作(from_numpy/numpy/to)
// 的 pybind11 绑定(M12 决议点 B/D)。薄壳纪律(PY-001):函数体只做参数转换 +
// 调用 C++ 公开 API(Tensor::empty/BackendRegistry/Backend::copy)+ 错误转换,
// 不实现任何计算/调度逻辑。

#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>

#include "translate_status.h"

namespace frame::python_bindings {

namespace py = pybind11;

namespace {

// frame DType -> numpy dtype 格式名。bfloat16 无 numpy 原生表示,报
// NotImplementedError(v0 边界,决议点 D;PY-013 DLPack 后续跟进)。int32/int64
// 是 M22 批4 决议点A的最小接触面扩项(docs/plan/2026-07-19-batch4-m22-seq.md
// §1.1):'i' kind,itemsize 4/8。不可达分支收尾风格与
// include/frame/core/dtype.h::DType::itemsize()/name() 一致(FRAME_CHECK 换成
// 本文件的 translate_status,同为"非法值必定不返回"契约)。
std::string dtype_to_npy_format(DType dtype) {
  switch (dtype.code()) {
    case DTypeCode::kFloat32:
      return "float32";
    case DTypeCode::kFloat16:
      return "float16";
    case DTypeCode::kInt32:
      return "int32";
    case DTypeCode::kInt64:
      return "int64";
    default:
      break;
  }
  translate_status(
      Status::make(ErrorCode::kUnimplemented, "Tensor.numpy: dtype '" + std::string(dtype.name()) +
                                                  "' has no numpy representation (v0 supports "
                                                  "float32/float16/int32/int64 only)"));
  return {};  // 不可达
}

// numpy dtype -> frame DTypeCode。接受 float32/float16(kind 'f',itemsize
// 4/2)与 int32/int64(kind 'i',itemsize 4/8,M22 批4 §1.1 扩项);其余一律
// InvalidArgument(含 from_numpy 收到 bfloat16 等无法表达场景——numpy 本身无
// bfloat16 原生 dtype,故落入本分支而非专门的 bf16 判断)。
DTypeCode npy_format_to_dtype(const py::dtype& dtype) {
  const char kind = dtype.kind();
  const py::ssize_t itemsize = dtype.itemsize();
  if (kind == 'f' && itemsize == 4) return DTypeCode::kFloat32;
  if (kind == 'f' && itemsize == 2) return DTypeCode::kFloat16;
  if (kind == 'i' && itemsize == 4) return DTypeCode::kInt32;
  if (kind == 'i' && itemsize == 8) return DTypeCode::kInt64;
  translate_status(Status::make(
      ErrorCode::kInvalidArgument,
      "from_numpy: unsupported numpy dtype (kind '" + std::string(1, kind) + "', itemsize " +
          std::to_string(itemsize) + "); v0 supports float32/float16/int32/int64 only"));
  return DTypeCode::kFloat32;  // 不可达
}

// 按注册键字符串构造 Device:backend 须已在 BackendRegistry 注册(不存在报
// KeyError,经 translate_status);Device::backend 是不持有数据的
// string_view,借用已注册 Backend::name() 的进程级长寿命存储,避免绑定层
// 引入悬垂 string_view(Python str 临时对象析构后其数据不再有效)。
Device make_device(const std::string& backend, int32_t index) {
  hal::Backend* found = translate_result(hal::BackendRegistry::instance().get(backend));
  return Device{found->name(), index};
}

// Tensor.numpy():D2H 拷出为一份新 numpy 数组(v0 拷贝语义,决议点 D)。
// 前后两段各自触碰 Python C API(构造 py::dtype/py::array),故不用
// py::call_guard 整函数释放 GIL,改手动局部化仅覆盖中间的纯 C++ 拷贝段
// (PY-010 memcpy 类,表注记见 docs/standards/python-binding.md)——超 15 行
// 属该类原因,PY-001 允许的例外。
py::array tensor_to_numpy(const Tensor& tensor) {
  const std::string npy_format = dtype_to_npy_format(tensor.dtype());

  const std::vector<int64_t>& dims = tensor.shape().dims();
  const std::vector<py::ssize_t> shape(dims.begin(), dims.end());
  py::array out(py::dtype(npy_format), shape);
  void* dst = out.mutable_data();

  const size_t bytes = static_cast<size_t>(tensor.numel()) * tensor.dtype().itemsize();
  hal::Backend* backend =
      translate_result(hal::BackendRegistry::instance().get(tensor.device().backend));
  {
    const py::gil_scoped_release release;
    translate_status(
        backend->copy(dst, cpu_device(), tensor.raw_data(), tensor.device(), bytes, nullptr));
  }
  return out;
}

// from_numpy(array):拷入一份 cpu Tensor(v0 拷贝语义,决议点 D)。持锁段先
// 取 buffer_info(其内部持有对底层缓冲的引用,保证 info.ptr 在本函数结束前
// 保持存活)与 dtype 元数据;GIL 局部释放段只访问已取得的纯 C++ 数据
// (info.ptr/dims),不再触碰任何 Python 对象——info 本身在函数末尾(晚于
// 释放段)才析构,届时 GIL 已重新持有。超 15 行属该 GIL 编排原因,PY-001
// 允许的例外。
Tensor from_numpy(const py::array& array) {
  const py::buffer_info info = array.request();
  if ((array.flags() & py::array::c_style) == 0) {
    translate_status(
        Status::make(ErrorCode::kInvalidArgument, "from_numpy: array must be C-contiguous"));
  }
  const DTypeCode code = npy_format_to_dtype(array.dtype());
  const std::vector<int64_t> dims(info.shape.begin(), info.shape.end());

  hal::Backend* backend = translate_result(hal::BackendRegistry::instance().get(kCpuBackendName));
  hal::Allocator* allocator = backend->allocator(cpu_device());
  if (allocator == nullptr) {
    translate_status(
        Status::make(ErrorCode::kInternal, "from_numpy: cpu backend returned a null allocator"));
  }

  Tensor out;
  {
    const py::gil_scoped_release release;
    out = translate_result(Tensor::empty(Shape(dims), DType(code), cpu_device(), *allocator));
    const size_t bytes = static_cast<size_t>(out.numel()) * out.dtype().itemsize();
    translate_status(
        backend->copy(out.raw_data(), cpu_device(), info.ptr, cpu_device(), bytes, nullptr));
  }
  return out;
}

// Tensor.to(backend):显式设备搬运(H2D/D2H/D2D,PY-012 动词形式)。选两端中
// "非 cpu"一侧的 Backend 执行拷贝——cpu 侧 Backend::copy 仅接受 cpu<->cpu
// (见 CpuBackend::copy 校验),H2D/D2H 须由设备侧后端实现(M11 决议点 D)。
// 函数体超 PY-001 的 15 行上限:后端解析 + 目标分配 + 方向拷贝的 HAL 编排
// 不可再拆(与 numpy()/from_numpy 同款豁免口径,原因随收口提交说明)。
Tensor tensor_to_device(const Tensor& tensor, const std::string& backend) {
  hal::Backend* dst_backend = translate_result(hal::BackendRegistry::instance().get(backend));
  const Device dst_device{dst_backend->name(), 0};
  hal::Allocator* allocator = dst_backend->allocator(dst_device);
  if (allocator == nullptr) {
    translate_status(Status::make(
        ErrorCode::kInternal,
        "Tensor.to: backend '" + backend + "' returned a null allocator for device index 0"));
  }
  Tensor out =
      translate_result(Tensor::empty(tensor.shape(), tensor.dtype(), dst_device, *allocator));

  const size_t bytes = static_cast<size_t>(tensor.numel()) * tensor.dtype().itemsize();
  const std::string_view copy_backend_name =
      dst_device.backend != kCpuBackendName ? dst_device.backend : tensor.device().backend;
  hal::Backend* copy_backend =
      translate_result(hal::BackendRegistry::instance().get(copy_backend_name));
  translate_status(copy_backend->copy(out.raw_data(), dst_device, tensor.raw_data(),
                                      tensor.device(), bytes, nullptr));
  return out;
}

}  // namespace

void bind_core(py::module_& m) {
  py::enum_<DTypeCode>(m, "DType",
                       "张量元素类型(v0 白名单:float32/float16/bfloat16 三浮点档 + "
                       "M22 批4 决议点A新增的 int32/int64 两整数档,docs/plan/"
                       "2026-07-19-batch4-m22-seq.md §1.1)。")
      .value("float32", DTypeCode::kFloat32)
      .value("float16", DTypeCode::kFloat16)
      .value("bfloat16", DTypeCode::kBFloat16)
      .value("int32", DTypeCode::kInt32)
      .value("int64", DTypeCode::kInt64);

  py::class_<Shape>(m, "Shape", "张量形状,与 list[int] 互转。")
      .def(py::init<std::vector<int64_t>>(), py::arg("dims"), "按各维尺寸列表构造。")
      .def_property_readonly("dims", &Shape::dims, "各维尺寸列表。")
      .def("__repr__", [](const Shape& shape) { return "Shape(" + shape.to_string() + ")"; });

  py::class_<Device>(m, "Device", "设备寻址句柄:后端注册键字符串 + 设备序号。")
      .def(py::init(&make_device), py::arg("backend"), py::arg("index") = 0, "构造设备句柄。")
      .def_property_readonly(
          "backend", [](const Device& device) { return std::string(device.backend); },
          "后端注册键。")
      .def_readonly("index", &Device::index, "设备序号。");

  py::class_<Tensor>(m, "Tensor", "张量句柄(值语义,实际数据由 C++ 核心持有,铁律 #2)。")
      .def_property_readonly("shape", &Tensor::shape, "张量形状。")
      .def_property_readonly(
          "dtype", [](const Tensor& tensor) { return tensor.dtype().code(); }, "张量元素类型。")
      .def_property_readonly("device", &Tensor::device, "张量归属设备。")
      .def("numpy", &tensor_to_numpy, "拷出为一份新 numpy 数组(D2H,v0 拷贝语义)。")
      .def("to", &tensor_to_device, py::arg("backend"), py::call_guard<py::gil_scoped_release>(),
           "显式搬运到指定后端设备(H2D/D2H/D2D,v0 拷贝语义)。");

  m.def("from_numpy", &from_numpy, py::arg("array"),
        "从 numpy 数组拷入一份 cpu Tensor(v0 拷贝语义,仅支持 "
        "float32/float16/int32/int64 且须 C-contiguous)。");
}

}  // namespace frame::python_bindings
