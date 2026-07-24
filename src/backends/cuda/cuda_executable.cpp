// CudaExecutable 实现单元(声明见 cuda_executable.h 头注释)。

#include "cuda_executable.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/storage.h>
#include <frame/hal/backend.h>

namespace frame::backends::cuda {

Result<std::unique_ptr<CudaExecutable>> CudaExecutable::compile(
    const ir::Graph& graph, hal::Allocator& allocator, Device device,
    const hal::CompileOptions& options) {
  Result<ops::ExecutablePlan> plan_result = ops::build_executable_plan(graph, kCudaBackendName);
  if (!plan_result.is_ok()) return plan_result.status();

  auto executable = std::make_unique<CudaExecutable>();
  executable->allocator_ = &allocator;
  executable->device_ = device;
  executable->options_ = options;
  executable->plan_ = std::move(plan_result.value());

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

Status CudaExecutable::run(std::span<const Tensor> inputs, std::span<Tensor> outputs,
                           hal::Stream& stream) {
  FRAME_RETURN_IF_ERROR(
      hal::validate_io_signature(inputs, input_signature_, "CudaExecutable::run: input"));
  FRAME_RETURN_IF_ERROR(
      hal::validate_io_signature(outputs, output_signature_, "CudaExecutable::run: output"));
  // device 独立校验(M11,design-reviewer REVISE 闭环裁决修订3):调用方须提供
  // 与本图编译目标一致的 cuda 设备张量(execution-model.md「cuda 图喂 device
  // 张量,调用方负责 H2D/D2H」契约),不静默接受其他设备指针。
  FRAME_RETURN_IF_ERROR(
      hal::validate_tensor_devices(inputs, device_, "CudaExecutable::run: input"));
  FRAME_RETURN_IF_ERROR(
      hal::validate_tensor_devices(outputs, device_, "CudaExecutable::run: output"));

  std::shared_ptr<Storage> arena_storage;
  if (plan_.arena_total_bytes > 0) {
    const Result<std::shared_ptr<Storage>> allocated_arena =
        Storage::allocate(*allocator_, plan_.arena_total_bytes, kDefaultAlignment, device_);
    if (!allocated_arena.is_ok()) return allocated_arena.status();
    arena_storage = allocated_arena.value();
  }

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
        // 零元素中间 Value 的 arena entry 不应丢失,但 total_bytes 可为 0；
        // 单独构造零字节 Tensor 以维持 cuda device 归属,非空值仍走 arena。
        if (type.shape.numel() == 0) {
          Result<Tensor> allocated = Tensor::empty(type.shape, type.dtype, device_, *allocator_);
          if (!allocated.is_ok()) return allocated.status();
          step_outputs.push_back(allocated.value());
        } else {
          step_outputs.push_back(Tensor::from_storage_slice(arena_storage, *arena_offset,
                                                            type.shape, type.dtype, device_));
        }
      } else {
        Result<Tensor> allocated = Tensor::empty(type.shape, type.dtype, device_, *allocator_);
        if (!allocated.is_ok()) return allocated.status();
        step_outputs.push_back(allocated.value());
      }
    }

    // ①cuda kernel 需要在调用方指定的流上异步 launch(与 cpu 恒同步执行、不
    // 消费 stream 不同,见类头注释)。
    ops::KernelContext ctx{step_inputs, step_outputs, &step.attrs, device_, &stream, &options_};
    const Status status = step.kernel(ctx);
    if (!status.is_ok()) {
      return Status::make(status.code(), "CudaExecutable::run: op '" + step.op +
                                             "' kernel failed: " + std::string(status.message()));
    }

    for (size_t i = 0; i < step.output_slots.size(); ++i) {
      slots[static_cast<size_t>(step.output_slots[i])] = step_outputs[i];
    }
  }

  // 图输出交还:src(内部 slot 张量)与 dst(调用方 outputs)均已校验为 device_
  // (cuda),经 Backend::copy(D2D)排入同一流,而非 host memmove(类头注释②)。
  if (!plan_.output_slots.empty()) {
    const Result<hal::Backend*> backend = hal::BackendRegistry::instance().get(device_.backend);
    if (!backend.is_ok()) return backend.status();
    for (size_t i = 0; i < plan_.output_slots.size(); ++i) {
      const Tensor& src = slots[static_cast<size_t>(plan_.output_slots[i])];
      Tensor& dst = outputs[i];
      const size_t bytes = static_cast<size_t>(src.numel()) * src.dtype().itemsize();
      if (bytes == 0) continue;
      const Status copy_status =
          backend.value()->copy(dst.raw_data(), device_, src.raw_data(), device_, bytes, &stream);
      if (!copy_status.is_ok()) return copy_status;
    }
  }
  return Status::ok();
}

}  // namespace frame::backends::cuda
