// CpuExecutable 实现单元(声明见 cpu_executable.h 头注释)。

#include "cpu_executable.h"

#include <cstddef>
#include <cstring>
#include <optional>
#include <utility>

#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/storage.h>

namespace frame::backends::cpu {

Result<std::unique_ptr<CpuExecutable>> CpuExecutable::compile(const ir::Graph& graph,
                                                              hal::Allocator& allocator,
                                                              Device device) {
  // 共享组件一次性算好 slot 表 + KernelFn 解析 + arena 偏移(M11,REUSE-002)。
  Result<ops::ExecutablePlan> plan_result = ops::build_executable_plan(graph, kCpuBackendName);
  if (!plan_result.is_ok()) return plan_result.status();

  auto executable = std::make_unique<CpuExecutable>();
  executable->allocator_ = &allocator;
  executable->device_ = device;
  executable->plan_ = std::move(plan_result.value());

  // hal::IoSpec 转换(消费方职责,共享组件禁用 hal include):仅取 dtype/shape,
  // 与此前 CpuExecutable 自建 input_signature_/output_signature_ 的内容等价。
  executable->input_signature_.reserve(executable->plan_.input_types.size());
  for (const ir::TensorType& type : executable->plan_.input_types) {
    executable->input_signature_.push_back(hal::IoSpec{type.dtype, type.shape});
  }
  executable->output_signature_.reserve(executable->plan_.output_types.size());
  for (const ir::TensorType& type : executable->plan_.output_types) {
    executable->output_signature_.push_back(hal::IoSpec{type.dtype, type.shape});
  }

  return executable;
}

Status CpuExecutable::run(std::span<const Tensor> inputs, std::span<Tensor> outputs,
                          hal::Stream& stream) {
  // 签名校验(M10 抽取):见 include/frame/hal/executable.h::validate_io_signature。
  FRAME_RETURN_IF_ERROR(
      hal::validate_io_signature(inputs, input_signature_, "CpuExecutable::run: input"));
  FRAME_RETURN_IF_ERROR(
      hal::validate_io_signature(outputs, output_signature_, "CpuExecutable::run: output"));
  // device 独立校验(M11,design-reviewer REVISE 闭环裁决修订3):对照本类自持
  // device_ 逐张量比对,与上方 IoSpec 校验(dtype/shape)彼此独立、互不替代。
  FRAME_RETURN_IF_ERROR(hal::validate_tensor_devices(inputs, device_, "CpuExecutable::run: input"));
  FRAME_RETURN_IF_ERROR(
      hal::validate_tensor_devices(outputs, device_, "CpuExecutable::run: output"));

  // arena 一次性分配(M9,决议点 D 覆盖版):total_bytes==0 时跳过分配(既无
  // 需要,也避免 Storage::allocate 对 0 字节的空请求做多余的下沉判断)。
  // arena_storage 生命周期覆盖本次 run() 调用,经 shared_ptr 引用计数托管
  // ——run() 结束、全部切片 Tensor 析构后自动回收。
  std::shared_ptr<Storage> arena_storage;
  if (plan_.arena_total_bytes > 0) {
    const Result<std::shared_ptr<Storage>> allocated_arena =
        Storage::allocate(*allocator_, plan_.arena_total_bytes, kDefaultAlignment, device_);
    if (!allocated_arena.is_ok()) return allocated_arena.status();
    arena_storage = allocated_arena.value();
  }

  // slots 前 input_signature_.size() 项直接复用调用方 inputs(浅拷贝句柄,
  // 共享同一 Storage,无需分配)。
  std::vector<Tensor> slots(static_cast<size_t>(plan_.slot_count));
  for (size_t i = 0; i < inputs.size(); ++i) {
    slots[i] = inputs[i];
  }

  for (const ops::ExecutablePlanStep& step : plan_.steps) {
    std::vector<Tensor> step_inputs;
    step_inputs.reserve(step.input_slots.size());
    for (const int32_t slot : step.input_slots) {
      step_inputs.push_back(slots[static_cast<size_t>(slot)]);
    }

    std::vector<Tensor> step_outputs;
    step_outputs.reserve(step.output_types.size());
    for (size_t i = 0; i < step.output_types.size(); ++i) {
      const ir::TensorType& type = step.output_types[i];
      const std::optional<size_t>& arena_offset = step.output_arena_offsets[i];
      if (arena_offset.has_value()) {
        // 零元素中间 Value 仍有 memory-plan entry,但整图 arena 可为 0 字节而
        // 不分配；此时经 Tensor::empty 保留合法 Storage/device 归属。非空
        // 中间 Value 继续在 arena 内切片,保持既有复用路径。
        if (type.shape.numel() == 0) {
          Result<Tensor> allocated = Tensor::empty(type.shape, type.dtype, device_, *allocator_);
          if (!allocated.is_ok()) return allocated.status();
          step_outputs.push_back(allocated.value());
        } else {
          step_outputs.push_back(Tensor::from_storage_slice(arena_storage, *arena_offset,
                                                            type.shape, type.dtype, device_));
        }
      } else {
        // 图输出 Value:独立分配,生命周期覆盖整个 run(),交还调用方前
        // 逐字节拷出(见下方图输出拷贝段)。
        Result<Tensor> allocated = Tensor::empty(type.shape, type.dtype, device_, *allocator_);
        if (!allocated.is_ok()) return allocated.status();
        step_outputs.push_back(allocated.value());
      }
    }

    ops::KernelContext ctx{step_inputs, step_outputs, &step.attrs, device_, &stream};
    const Status status = step.kernel(ctx);
    if (!status.is_ok()) {
      return Status::make(status.code(), "CpuExecutable::run: op '" + step.op +
                                             "' kernel failed: " + std::string(status.message()));
    }

    for (size_t i = 0; i < step.output_slots.size(); ++i) {
      slots[static_cast<size_t>(step.output_slots[i])] = step_outputs[i];
    }
  }

  // 图输出从对应 slot 拷贝进调用方 outputs(逐字节拷贝——调用方 outputs 是
  // 预分配的独立 Storage,须真正搬运数据,不能只做句柄赋值;签名校验已保证
  // dtype/shape 一致,字节数相同)。
  for (size_t i = 0; i < plan_.output_slots.size(); ++i) {
    const Tensor& src = slots[static_cast<size_t>(plan_.output_slots[i])];
    Tensor& dst = outputs[i];
    const size_t bytes = static_cast<size_t>(src.numel()) * src.dtype().itemsize();
    if (bytes != 0) {
      std::memmove(dst.raw_data(), src.raw_data(), bytes);
    }
  }
  return Status::ok();
}

}  // namespace frame::backends::cpu
