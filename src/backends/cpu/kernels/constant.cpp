// CPU 参考 kernel:constant(M8)——0 输入 1 输出,直接按 attrs 填充调用方
// 预分配的输出张量。物化逻辑单份(REUSE-002):调
// ops::fill_tensor_from_constant_attrs(见 include/frame/ops/constant_utils.h),
// constant_folding pass 的编译期求值同样调用该函数,禁止本文件另写一份。
// 不触碰 ctx.stream(cpu kernel 不解引用 stream,M4 既有契约)。

#include <string>

#include <frame/core/device.h>
#include <frame/core/status.h>
#include <frame/ops/constant_utils.h>
#include <frame/ops/kernel_registry.h>

namespace {

frame::Status constant_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (!ctx.inputs.empty()) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'constant' cpu kernel expects 0 inputs, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'constant' cpu kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }
  if (ctx.attrs == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'constant' cpu kernel requires non-null attrs");
  }
  return frame::ops::fill_tensor_from_constant_attrs(*ctx.attrs, ctx.outputs[0]);
}

}  // namespace

FRAME_REGISTER_KERNEL(frame::ops::kConstantOpName, frame::kCpuBackendName, constant_cpu_kernel);
