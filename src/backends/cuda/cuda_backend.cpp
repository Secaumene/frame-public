// CUDA 后端实现单元(阶段 B)。虚函数依据见 include/frame/hal/backend.h 头部;
// 五接口映射表见 docs/backends/cuda.md 第 4 章。

#include "cuda_backend.h"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include <frame/ir/graph.h>
#include <frame/ops/kernel_registry.h>

#include "cuda_executable.h"
#include "cuda_status.h"

namespace frame::backends::cuda {

namespace {

// device.backend/index 校验:cuda 后端服务 [0, device_count) 内的逻辑设备。
// 违例返回带实际值的英文 kInvalidArgument(ARCH-031 口径:不静默降级),与
// src/backends/cpu/cpu_backend.cpp::validate_cpu_device 同一惯例。
Status validate_cuda_device(Device device, std::string_view caller) {
  if (device.backend != kCudaBackendName) {
    return Status::make(ErrorCode::kInvalidArgument,
                        std::string(caller) + ": unsupported device backend '" +
                            std::string(device.backend) + "', expected '" +
                            std::string(kCudaBackendName) + "'");
  }
  int32_t count = 0;
  const cudaError_t error = cudaGetDeviceCount(&count);
  if (error != cudaSuccess) {
    return cuda_status(error, std::string(caller) + ": cudaGetDeviceCount");
  }
  if (device.index < 0 || device.index >= count) {
    return Status::make(ErrorCode::kInvalidArgument,
                        std::string(caller) + ": device index " + std::to_string(device.index) +
                            " out of range, expected [0, " + std::to_string(count) + ")");
  }
  return Status::ok();
}

// 两端 Device.backend 显式推导 cudaMemcpyKind(design-reviewer REVISE 闭环
// 采纳建议③:弃 cudaMemcpyDefault 依赖,契约措辞与 backend-hal.md copy 条目
// 一致——「方向由两端 Device 推导,主机内存以 cpu 后端的 Device 表示」)。两端
// 均非本后端可识别域(cpu/cuda 之外)→ kInvalidArgument。
Result<cudaMemcpyKind> infer_memcpy_kind(Device dst_device, Device src_device) {
  const bool dst_is_host = dst_device.backend == kCpuBackendName;
  const bool dst_is_cuda = dst_device.backend == kCudaBackendName;
  const bool src_is_host = src_device.backend == kCpuBackendName;
  const bool src_is_cuda = src_device.backend == kCudaBackendName;
  if (!(dst_is_host || dst_is_cuda) || !(src_is_host || src_is_cuda)) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "CudaBackend::copy: dst/src device backend must be '" + std::string(kCpuBackendName) +
            "' or '" + std::string(kCudaBackendName) + "', got dst='" +
            std::string(dst_device.backend) + "' src='" + std::string(src_device.backend) + "'");
  }
  if (src_is_host && dst_is_cuda) return cudaMemcpyHostToDevice;
  if (src_is_cuda && dst_is_host) return cudaMemcpyDeviceToHost;
  if (src_is_cuda && dst_is_cuda) return cudaMemcpyDeviceToDevice;
  return cudaMemcpyHostToHost;  // 双 host(hal_conformance H2H 用例)
}

}  // namespace

CudaStream::~CudaStream() {
  if (stream_ != nullptr) cudaStreamDestroy(stream_);
}

Result<std::unique_ptr<CudaStream>> CudaStream::create(int32_t device_index) {
  const cudaError_t set_device_error = cudaSetDevice(device_index);
  if (set_device_error != cudaSuccess) {
    return cuda_status(set_device_error, "CudaStream::create: cudaSetDevice");
  }
  auto stream = std::make_unique<CudaStream>();
  const cudaError_t error = cudaStreamCreate(&stream->stream_);
  if (error != cudaSuccess) return cuda_status(error, "CudaStream::create: cudaStreamCreate");
  return stream;
}

Status CudaStream::synchronize() {
  return cuda_status(cudaStreamSynchronize(stream_), "CudaStream::synchronize");
}

Status CudaStream::record(hal::Event& event) {
  auto& cuda_event = static_cast<CudaEvent&>(event);
  const cudaError_t error = cudaEventRecord(cuda_event.native(), stream_);
  if (error != cudaSuccess) return cuda_status(error, "CudaStream::record: cudaEventRecord");
  cuda_event.mark_recorded();
  return Status::ok();
}

Status CudaStream::wait(const hal::Event& event) {
  const auto& cuda_event = static_cast<const CudaEvent&>(event);
  return cuda_status(cudaStreamWaitEvent(stream_, cuda_event.native(), 0),
                     "CudaStream::wait: cudaStreamWaitEvent");
}

void* CudaStream::native_handle() { return static_cast<void*>(stream_); }

CudaEvent::~CudaEvent() {
  if (event_ != nullptr) cudaEventDestroy(event_);
}

Result<std::unique_ptr<CudaEvent>> CudaEvent::create(int32_t device_index) {
  const cudaError_t set_device_error = cudaSetDevice(device_index);
  if (set_device_error != cudaSuccess) {
    return cuda_status(set_device_error, "CudaEvent::create: cudaSetDevice");
  }
  auto event = std::make_unique<CudaEvent>();
  const cudaError_t error = cudaEventCreate(&event->event_);
  if (error != cudaSuccess) return cuda_status(error, "CudaEvent::create: cudaEventCreate");
  return event;
}

bool CudaEvent::query() const {
  // 未 record 契约(backend-hal.md 2.3):显式标志位保证,不依赖 CUDA 对未记录
  // 事件的实际行为(见 cuda_backend.h 头注释)。
  if (!recorded_) return true;
  return cudaEventQuery(event_) == cudaSuccess;
}

Status CudaEvent::synchronize() {
  if (!recorded_) return Status::ok();
  return cuda_status(cudaEventSynchronize(event_), "CudaEvent::synchronize");
}

CudaBackend::~CudaBackend() {
  if (cublas_handle_ != nullptr) cublasDestroy(cublas_handle_);
  if (cublaslt_handle_ != nullptr) cublasLtDestroy(cublaslt_handle_);
  if (cudnn_handle_ != nullptr) cudnnDestroy(cudnn_handle_);
}

std::string_view CudaBackend::name() const { return kCudaBackendName; }

Result<int32_t> CudaBackend::device_count() const {
  int32_t count = 0;
  const cudaError_t error = cudaGetDeviceCount(&count);
  if (error != cudaSuccess)
    return cuda_status(error, "CudaBackend::device_count: cudaGetDeviceCount");
  return count;
}

Result<std::unique_ptr<hal::Stream>> CudaBackend::create_stream(Device device) {
  const Status validation = validate_cuda_device(device, "CudaBackend::create_stream");
  if (!validation.is_ok()) return validation;
  Result<std::unique_ptr<CudaStream>> stream = CudaStream::create(device.index);
  if (!stream.is_ok()) return stream.status();
  return std::unique_ptr<hal::Stream>(std::move(stream.value()));
}

Result<std::unique_ptr<hal::Event>> CudaBackend::create_event(Device device) {
  const Status validation = validate_cuda_device(device, "CudaBackend::create_event");
  if (!validation.is_ok()) return validation;
  Result<std::unique_ptr<CudaEvent>> event = CudaEvent::create(device.index);
  if (!event.is_ok()) return event.status();
  return std::unique_ptr<hal::Event>(std::move(event.value()));
}

hal::Allocator* CudaBackend::allocator(Device device) {
  const Status validation = validate_cuda_device(device, "CudaBackend::allocator");
  if (!validation.is_ok()) {
    std::fprintf(stderr, "%s\n", std::string(validation.message()).c_str());
    return nullptr;
  }
  // 进程级单例表:每设备序号一个 CudaAllocator(函数内 static,magic statics
  // 线程安全初始化;首次调用时按当前 device_count 建表,与
  // src/backends/cpu/cpu_backend.cpp::CpuBackend::allocator 的单例惯例同构,
  // 差异仅在于按设备数建表而非固定单实例)。
  static const std::vector<std::unique_ptr<CudaAllocator>> allocators = [] {
    std::vector<std::unique_ptr<CudaAllocator>> result;
    int32_t count = 0;
    cudaGetDeviceCount(&count);
    result.reserve(static_cast<size_t>(count));
    for (int32_t i = 0; i < count; ++i) {
      result.push_back(std::make_unique<CudaAllocator>(i));
    }
    return result;
  }();
  return allocators[static_cast<size_t>(device.index)].get();
}

Status CudaBackend::copy(void* dst, Device dst_device, const void* src, Device src_device,
                         size_t bytes, hal::Stream* stream) {
  const Result<cudaMemcpyKind> kind = infer_memcpy_kind(dst_device, src_device);
  if (!kind.is_ok()) return kind.status();
  if (bytes == 0) return Status::ok();
  if (dst == nullptr || src == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "CudaBackend::copy: dst and src must be non-null when bytes > 0");
  }
  // stream 非空 -> cudaMemcpyAsync(调用方负责同步);stream 为空 ->
  // cudaMemcpy 同步(对齐 backend-hal.md copy 契约,design-reviewer REVISE
  // 闭环采纳建议③)。native_handle() 是 ARCH-030 白名单方法,仅供 src/backends/
  // 内部下沉使用。
  if (stream != nullptr) {
    auto native_stream = static_cast<cudaStream_t>(stream->native_handle());
    return cuda_status(cudaMemcpyAsync(dst, src, bytes, kind.value(), native_stream),
                       "CudaBackend::copy: cudaMemcpyAsync");
  }
  return cuda_status(cudaMemcpy(dst, src, bytes, kind.value()), "CudaBackend::copy: cudaMemcpy");
}

Result<std::unique_ptr<hal::Executable>> CudaBackend::compile(const ir::Graph& graph,
                                                              const hal::CompileOptions& options) {
  if (graph.topological_order().empty()) {
    return Status::make(ErrorCode::kInvalidArgument, "CudaBackend::compile: graph has no nodes");
  }

  // 取图 device:V6 保证全图所有 Value 的 device 一致
  // (docs/architecture/ir-design.md 第4章),与
  // src/backends/cpu/cpu_backend.cpp::CpuBackend::compile 的取法一致。
  Device graph_device{};
  bool found_device = false;
  for (const ir::Node* node : graph.topological_order()) {
    if (!node->outputs().empty()) {
      graph_device = node->outputs()[0].type().device;
      found_device = true;
      break;
    }
  }
  if (!found_device) {
    return Status::make(ErrorCode::kInternal,
                        "CudaBackend::compile: graph has node(s) but no Value carries a device");
  }
  FRAME_RETURN_IF_ERROR(validate_cuda_device(graph_device, "CudaBackend::compile"));

  hal::Allocator* alloc = allocator(graph_device);
  if (alloc == nullptr) {
    return Status::make(ErrorCode::kInternal,
                        "CudaBackend::compile: allocator unavailable for cuda device");
  }

  Result<std::unique_ptr<CudaExecutable>> executable =
      CudaExecutable::compile(graph, *alloc, graph_device, options);
  if (!executable.is_ok()) return executable.status();
  return std::unique_ptr<hal::Executable>(std::move(executable.value()));
}

Status CudaBackend::launch(const hal::KernelInvocation& invocation, hal::Stream* stream) {
  FRAME_RETURN_IF_ERROR(validate_cuda_device(invocation.device, "CudaBackend::launch"));
  const Result<ops::KernelFn> kernel = ops::KernelRegistry::instance().find(invocation.op, name());
  if (!kernel.is_ok()) return kernel.status();
  ops::KernelContext context{invocation.inputs, invocation.outputs, invocation.attrs,
                             invocation.device, stream};
  return kernel.value()(context);
}

Result<CublasHandleGuard> CudaBackend::acquire_cublas_handle(cudaStream_t stream) {
  std::unique_lock<std::mutex> lock(cublas_mutex_);
  if (cublas_handle_ == nullptr) {
    const cublasStatus_t create_status = cublasCreate(&cublas_handle_);
    if (create_status != CUBLAS_STATUS_SUCCESS) {
      return Status::make(ErrorCode::kInternal,
                          "CudaBackend::acquire_cublas_handle: cublasCreate failed with status " +
                              std::to_string(static_cast<int>(create_status)));
    }
  }
  const cublasStatus_t set_stream_status = cublasSetStream(cublas_handle_, stream);
  if (set_stream_status != CUBLAS_STATUS_SUCCESS) {
    return Status::make(ErrorCode::kInternal,
                        "CudaBackend::acquire_cublas_handle: cublasSetStream failed with status " +
                            std::to_string(static_cast<int>(set_stream_status)));
  }
  CublasHandleGuard guard;
  guard.lock = std::move(lock);
  guard.handle = cublas_handle_;
  return guard;
}

Result<CublasLtHandleGuard> CudaBackend::acquire_cublaslt_handle() {
  std::unique_lock<std::mutex> lock(cublaslt_mutex_);
  if (cublaslt_handle_ == nullptr) {
    const cublasStatus_t create_status = cublasLtCreate(&cublaslt_handle_);
    if (create_status != CUBLAS_STATUS_SUCCESS) {
      return Status::make(
          ErrorCode::kInternal,
          "CudaBackend::acquire_cublaslt_handle: cublasLtCreate failed with status " +
              std::to_string(static_cast<int>(create_status)));
    }
  }
  CublasLtHandleGuard guard;
  guard.lock = std::move(lock);
  guard.handle = cublaslt_handle_;
  return guard;
}

Result<CudnnHandleGuard> CudaBackend::acquire_cudnn_handle(cudaStream_t stream) {
  std::unique_lock<std::mutex> lock(cudnn_mutex_);
  if (cudnn_handle_ == nullptr) {
    const cudnnStatus_t create_status = cudnnCreate(&cudnn_handle_);
    if (create_status != CUDNN_STATUS_SUCCESS) {
      return Status::make(ErrorCode::kInternal,
                          "CudaBackend::acquire_cudnn_handle: cudnnCreate failed: " +
                              std::string(cudnnGetErrorString(create_status)));
    }
  }
  // legacy 即时 API 不接收 stream 实参(不同于 cublasLtMatmul),必须经
  // cudnnSetStream 绑定,见 CudnnHandleGuard 头注释。
  const cudnnStatus_t set_stream_status = cudnnSetStream(cudnn_handle_, stream);
  if (set_stream_status != CUDNN_STATUS_SUCCESS) {
    return Status::make(ErrorCode::kInternal,
                        "CudaBackend::acquire_cudnn_handle: cudnnSetStream failed: " +
                            std::string(cudnnGetErrorString(set_stream_status)));
  }
  CudnnHandleGuard guard;
  guard.lock = std::move(lock);
  guard.handle = cudnn_handle_;
  return guard;
}

}  // namespace frame::backends::cuda

// CUDA 后端静态注册:仅当 FRAME_ENABLE_CUDA 且 CUDA Toolkit 探测成功时本翻译
// 单元才参与构建(见 src/backends/cuda/CMakeLists.txt / cmake/
// frame_dependencies.cmake)。聚合库须以 WHOLE_ARCHIVE 链接本静态库,见
// cmake/frame_backend.cmake。
FRAME_REGISTER_BACKEND(frame::kCudaBackendName, frame::backends::cuda::CudaBackend);
