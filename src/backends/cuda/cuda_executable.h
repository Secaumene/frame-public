#pragma once
// CudaExecutable:CUDA 后端的整图编译产物(后端专有,铁律 #3 合规,不进
// include/)。结构与 src/backends/cpu/cpu_executable.h 的 CpuExecutable 同源
// (M11,均消费 ops::build_executable_plan,见 include/frame/ops/
// executable_plan.h);核心差异:
//   ①run() 把真正的 hal::Stream& 传给 KernelContext(cuda kernel 需要在该流上
//     异步 launch,不同于 cpu 恒同步执行、不消费 stream);
//   ②图输出交还经 Backend::copy(D2D,cudaMemcpyAsync 排入同一流)而非
//     memmove——src(内部 slot 张量)与 dst(调用方 outputs)均校验为 cuda 设备
//     (hal::validate_tensor_devices),raw_data() 是设备指针,host memmove 不
//     可安全解引用(CudaAllocator 内部经 cudaMalloc 分配,纯设备驻留内存不可
//     host 解引用,须走 CUDA API 以维持流内顺序,不能绕过 stream 排队直接
//     host 端搬运——见 cuda_executable.cpp 头注释)。

#include <memory>
#include <span>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/executable.h>
#include <frame/hal/stream.h>
#include <frame/ir/graph.h>
#include <frame/ops/executable_plan.h>

namespace frame::backends::cuda {

// CudaExecutable:见文件头注释。虚函数依据同 include/frame/hal/backend.h 头部
// R1∧R2∧R3 判定(Executable 属 HAL 白名单六类之一)。
class CudaExecutable final : public hal::Executable {
 public:
  // 编译入口:由 CudaBackend::compile 调用(该处已校验图非空且 device 为
  // cuda,并把校验通过的 device 直接传入)。options 按值存入实例(ADR-0019);
  // run() 期经 KernelContext 借用视图下传 kernel(首例消费者 allow_tf32)。
  static Result<std::unique_ptr<CudaExecutable>> compile(const ir::Graph& graph,
                                                         hal::Allocator& allocator, Device device,
                                                         const hal::CompileOptions& options);

  Status run(std::span<const Tensor> inputs, std::span<Tensor> outputs,
             hal::Stream& stream) override;

  std::vector<hal::IoSpec> input_signature() const override { return input_signature_; }
  std::vector<hal::IoSpec> output_signature() const override { return output_signature_; }

  // 默认构造仅供 compile() 内部经 std::make_unique 使用(CPP-060)。
  CudaExecutable() = default;

 private:
  ops::ExecutablePlan plan_;
  std::vector<hal::IoSpec> input_signature_;
  std::vector<hal::IoSpec> output_signature_;
  Device device_{};
  hal::Allocator* allocator_ = nullptr;  // 借用指针,生命周期贯穿进程(CudaBackend 持有)
  // 编译期传入的选项按值存(ADR-0019,非借用指针):run() 构造 KernelContext 时
  // 取本成员地址下传 kernel,借用期贯穿本 Executable 存活期。
  hal::CompileOptions options_{};
};

}  // namespace frame::backends::cuda
