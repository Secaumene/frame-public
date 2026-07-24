// Intel GPU(SYCL)逐元素内核实现单元(骨架;cpu-only 下不参与构建,由 icpx 编译)。
// 骨架期不引 SDK 头、不写 SYCL 调用;仅留占位与实现指引。

#include <frame/ops/kernel_registry.h>

// TODO(FRAME-IMPL): 落地 SYCL 逐元素 kernel(parallel_for + USM),内部经 dispatch_dtype
//   按 dtype 展开,再经 FRAME_REGISTER_KERNEL(op, kIntelGpuBackendName, fn) 注册。参考:
//   docs/backends/intel-gpu.md;docs/architecture/operator-system.md 第4章。完成判据:
//   intel-gpu preset 下 KernelRegistry::find("add", kIntelGpuBackendName) 可取到内核并计算正确。
