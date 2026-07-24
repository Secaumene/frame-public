#pragma once
// CUDA(NVIDIA GPU)后端内部头(不进 include/;仅供 src/backends/cuda/ 使用)。
// 虚函数依据见 include/frame/hal/backend.h 头部。
// 后端隔离要求见 docs/architecture/overview.md 的 ARCH-001。

#include <cstddef>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <memory>
#include <mutex>
#include <string_view>

#include <frame/core/device.h>
#include <frame/core/status.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/event.h>
#include <frame/hal/executable.h>
#include <frame/hal/stream.h>

#include <cuda_runtime.h>

namespace frame::backends::cuda {

// CUDA 设备内存分配器:一个实例对应一个设备序号(阶段 B-6)。
// 实现:底层经 cudaMalloc/cudaFree(纯设备驻留内存,不可被 host 直接解引用)。
// 积压批次③前曾临时改用 cudaMallocManaged 规避 tests/cpp/hal_conformance/ 的
// 通用一致性套件对分配指针做 host 端 std::memset/std::memcmp 的隐含假设(该
// 偏离登记见 git 历史);该套件已泛化为「host 侧准备 pattern -> backend->copy
// 写入分配指针 -> backend->copy 读回 -> synchronize -> host 侧 memcmp」
// (tests/cpp/hal_conformance/test_hal_conformance.cpp /
// test_hal_conformance_stub.cpp),不再假设分配指针可 host 解引用,依据消失,
// 回切 cudaMalloc(完成判据见 docs/backends/cuda.md 第 4 章 Allocator 行待办
// 标注)。真正需要 stream 顺序保证的路径(CudaExecutable/kernel 间的搬运)
// 一律经 Backend::copy(cudaMemcpy[Async])完成。
// 对齐保证:cudaMalloc 官方未在 CUDA C++ Programming Guide 给出显式数值承诺,
// 以 256 字节作为保守观测口径(与业界惯例一致)——【待查证】来源:NVIDIA
// CUDA C++ Programming Guide「Device Memory」章节,本机构建环境无网络核实,
// 维持该保守值(见 cuda_allocator.cpp 同款注释)。请求 alignment 超出该保守值
// 时拒绝(kInvalidArgument),不静默忽略。
class CudaAllocator final : public hal::Allocator {
 public:
  explicit CudaAllocator(int32_t device_index) : device_index_(device_index) {}

  Result<void*> allocate(size_t bytes, size_t alignment) override;
  void deallocate(void* ptr) override;

 private:
  int32_t device_index_ = 0;
};

// CUDA 执行流:cudaStream_t 的 RAII 包装(阶段 B-7)。
class CudaStream final : public hal::Stream {
 public:
  ~CudaStream() override;

  static Result<std::unique_ptr<CudaStream>> create(int32_t device_index);

  Status synchronize() override;
  Status record(hal::Event& event) override;
  Status wait(const hal::Event& event) override;
  void* native_handle() override;

  cudaStream_t native() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

// CUDA 事件:cudaEvent_t 的 RAII 包装(阶段 B-7)。「未 record」契约
// (query()==true、synchronize()==Ok)以显式标志位 recorded_ 保证,不依赖/不
// 假设 CUDA 对未记录事件的实际行为(backend-hal.md 2.3 契约优先于巧合)。
class CudaEvent final : public hal::Event {
 public:
  ~CudaEvent() override;

  static Result<std::unique_ptr<CudaEvent>> create(int32_t device_index);

  bool query() const override;
  Status synchronize() override;

  cudaEvent_t native() const { return event_; }
  // 仅供 CudaStream::record 调用,标记本事件已被记录过一次。
  void mark_recorded() { recorded_ = true; }

 private:
  cudaEvent_t event_ = nullptr;
  bool recorded_ = false;
};

// acquire_cublas_handle() 的 RAII 返回值:持锁期间可安全使用 handle(cuBLAS
// handle 非线程安全,见 CudaBackend::acquire_cublas_handle 头注释);析构时
// 自动解锁。
struct CublasHandleGuard {
  std::unique_lock<std::mutex> lock;
  cublasHandle_t handle = nullptr;
};

// acquire_cublaslt_handle() 的 RAII 返回值:持锁期间可安全使用 handle;析构时
// 自动解锁(ADR-0019,与 CublasHandleGuard 同款模式)。cublasLt handle 不绑定
// stream(无 cublasLtSetStream 这类 API),stream 由调用点直接传给
// cublasLtMatmul,故本 guard 不持有/不设置 stream。
struct CublasLtHandleGuard {
  std::unique_lock<std::mutex> lock;
  cublasLtHandle_t handle = nullptr;
};

// acquire_cudnn_handle() 的 RAII 返回值:持锁期间可安全使用 handle;析构时
// 自动解锁(M21,ADR-0021,惰性单例+mutex guard 模式镜像
// CublasHandleGuard/CublasLtHandleGuard)。与 CublasLtHandleGuard 不同——cuDNN
// legacy 即时 API(cudnnConvolutionForward 等)不像 cublasLtMatmul 那样接收
// stream 实参,只能经 cudnnSetStream 把 stream 绑定到 handle 上;故本 guard
// 镜像的是 CublasHandleGuard/acquire_cublas_handle 那一支(取 stream 形参、
// 每次 acquire 时重新绑定),而非"不持有 stream"的 cublasLt 那一支。
struct CudnnHandleGuard {
  std::unique_lock<std::mutex> lock;
  cudnnHandle_t handle = nullptr;
};

// CUDA 后端:实现 Backend HAL 全部接口(阶段 B-9)。
class CudaBackend final : public hal::Backend {
 public:
  ~CudaBackend() override;

  std::string_view name() const override;
  Result<int32_t> device_count() const override;
  Result<std::unique_ptr<hal::Stream>> create_stream(Device device) override;
  Result<std::unique_ptr<hal::Event>> create_event(Device device) override;
  hal::Allocator* allocator(Device device) override;
  Status copy(void* dst, Device dst_device, const void* src, Device src_device, size_t bytes,
              hal::Stream* stream) override;
  Result<std::unique_ptr<hal::Executable>> compile(const ir::Graph& graph,
                                                   const hal::CompileOptions& options) override;
  Status launch(const hal::KernelInvocation& invocation, hal::Stream* stream) override;

  // 供 src/backends/cuda/kernels/matmul.cpp 使用(阶段 C-13):惰性创建
  // cublasHandle_t(每 CudaBackend 实例一个,进程生命周期内复用)、绑定
  // stream、加锁返回——cuBLAS handle 并发调用非线程安全,mutex 序列化跨线程
  // 复用同一 handle 的场景。stream 可为 nullptr(等价 cublasSetStream 默认流)。
  Result<CublasHandleGuard> acquire_cublas_handle(cudaStream_t stream);

  // 供 src/backends/cuda/kernels/matmul.cpp 使用(ADR-0019):惰性创建
  // cublasLtHandle_t(每 CudaBackend 实例一个,进程生命周期内复用)、加锁
  // 返回——与 acquire_cublas_handle 同款模式,差异在于 cublasLt handle 不绑
  // stream,故不接收 stream 参数(调用点自行把 stream 传给 cublasLtMatmul)。
  Result<CublasLtHandleGuard> acquire_cublaslt_handle();

  // 供 src/backends/cuda/kernels/conv.cpp、kernels/pool.cpp 使用(M21,
  // ADR-0021):惰性创建 cudnnHandle_t(每 CudaBackend 实例一个,进程生命周期
  // 内复用)、经 cudnnSetStream 绑定 stream、加锁返回——cuDNN handle 并发调用
  // 非线程安全,mutex 序列化跨线程复用同一 handle 的场景,与
  // acquire_cublas_handle 同款模式(见该方法与 CudnnHandleGuard 头注释)。
  Result<CudnnHandleGuard> acquire_cudnn_handle(cudaStream_t stream);

 private:
  std::mutex cublas_mutex_;
  cublasHandle_t cublas_handle_ = nullptr;  // 惰性创建,首次 acquire_cublas_handle 时初始化
  std::mutex cublaslt_mutex_;
  cublasLtHandle_t cublaslt_handle_ = nullptr;  // 惰性创建,首次 acquire_cublaslt_handle 时初始化
  std::mutex cudnn_mutex_;
  cudnnHandle_t cudnn_handle_ = nullptr;  // 惰性创建,首次 acquire_cudnn_handle 时初始化
};

}  // namespace frame::backends::cuda
