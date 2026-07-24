// FallbackExecutable 实现单元(声明见同目录 fallback_executable.h)。
//
// 前提(由调用方 runtime::compile 保证,本单元不重复校验):target_backend 已
// 在 BackendRegistry 注册——runtime::compile 仅在管线/Backend::compile 返回
// kUnimplemented 时才调用 build(),而该哨兵码的两处翻译落点
// (src/compiler/passes/backend_lowering.cpp、src/backends/cpu/
// cpu_executable.cpp)均晚于 BackendRegistry::get(backend_name) 的成功查找
// (不改 BackendRegistry::get 的 kNotFound,design-reviewer REVISE 闭环修订
// 1)。同理,由于 backend_lowering 是标准管线的最后一 pass、shape_inference
// 是第二 pass(见 src/compiler/pipeline.cpp 固定顺序 ARCH-053),能触发本类
// build() 的图必然已通过 shape_inference 对 OpRegistry::find(op)!=nullptr
// 的校验——原图(未融合)的每个节点 op 名理论上均已注册,故下方 OpRegistry
// 查找失败被当作理论不可达的防御分支处理(kInternal 而非 kUnimplemented)。

#include "fallback_executable.h"

#include <cstddef>
#include <cstring>
#include <string>
#include <utility>

#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/op_registry.h>
#include <frame/runtime/fallback_stats.h>

#include "warn_log.h"

namespace frame::runtime {

namespace {

// 回退目标标签(m10-design-brief 决议点 B/C,WARN 消息与 execution-model.md
// 第5章用词一致)。"eager-kernel"(① 直接命中)不出现在下方 log_fallback 的
// 调用点:本类仅在某节点的 ① 查找失败、需要 ② 或 ③ 才能解决时才计一次"降级"
// (呼应"每发生一跳回退,便于发现性能悬崖"的统计目的——① 直接命中不是性能
// 悬崖,不计入)。
constexpr std::string_view kDecompositionDestination = "decomposition";
constexpr std::string_view kCpuReferenceDestination = "cpu-reference";

// 打一条 WARN + 记一次回退决策(build 期一次性,execution-model.md 第5章
// "v0 实现口径")。op/backend 是发生降级的那个节点自身的 op 名与目标后端名
// (顶层节点或 decomposition 展开的微图节点均适用同一份实现,REUSE-002)。
void log_fallback(std::string_view op, std::string_view backend, std::string_view destination,
                  std::string_view reason) {
  warn_log("eager fallback: op='" + std::string(op) + "', backend='" + std::string(backend) +
           "', destination='" + std::string(destination) + "', reason: " + std::string(reason));
  FallbackStats::instance().record(op, backend);
}

// 朴素分配一个步骤输出张量(回退是逃生舱非性能路径,REUSE-011 精神,不做
// arena/复用规划,逐步各自分配)。device 决定去哪个后端取 Allocator——① 方案
// 用目标后端,③ 方案用 cpu 参考后端(见 FallbackExecutable::run)。
Result<Tensor> allocate_step_output(const hal::IoSpec& spec, Device device) {
  const Result<hal::Backend*> backend = hal::BackendRegistry::instance().get(device.backend);
  if (!backend.is_ok()) return backend.status();
  hal::Allocator* allocator = backend.value()->allocator(device);
  if (allocator == nullptr) {
    return Status::make(ErrorCode::kInternal,
                        "FallbackExecutable::run: allocator unavailable for device backend '" +
                            std::string(device.backend) + "'");
  }
  return Tensor::empty(spec.shape, spec.dtype, device, *allocator);
}

// 校验张量归属设备是否属于本类可安全处理的"host 内存或目标后端"集合(v0 边界:
// cpu 参考后端 + 目标后端本身)。M11 决议点 D:①/③ 混跑内部的跨设备搬运已经
// Backend::copy 编排(见下方 stage_tensor_to_device/图输出 copyout 边),但外部
// 调用方仍只被允许提供 cpu 或目标后端两者之一的张量——"cpu 与目标后端之外"
// 仍防御式硬失败,而非静默按 host 处理导致数据损坏(设计裁决 4b:该判据本身不
// 放开,放开的是内部实现)。
Status validate_host_capable_device(Device device, std::string_view target_backend,
                                    std::string_view label) {
  if (device.backend == kCpuBackendName || device.backend == target_backend) {
    return Status::ok();
  }
  return Status::make(ErrorCode::kUnimplemented,
                      "FallbackExecutable::run: " + std::string(label) +
                          " tensor device backend '" + std::string(device.backend) +
                          "' is neither the cpu reference backend nor the target backend '" +
                          std::string(target_backend) + "'");
}

// 经 target_backend 对应的 Backend::copy 搬运 bytes 字节(M11,决议点 D):方向
// (H2D/D2H/D2D)由两端 Device 推导;跨设备编排是逃生舱非性能路径,统一走同步
// copy(stream=nullptr,对齐 backend-hal.md copy 契约"stream 为空则同步"口径),
// 调用方无需额外 synchronize。若 target_backend 的 Backend::copy 本身返回
// kUnimplemented(如测试替身/整图模式后端未实现该方法),按 ARCH-031 同一哨兵
// 码语义降级为 host memmove——本类既有 v0 内存边界(validate_host_capable_
// device)保证参与拷贝的两端 device 只能是 cpu 或 target_backend,若
// target_backend 连 copy 都未实现,其后端在该边界下只可能是纯 host 内存的
// 替身后端(真实设备后端不会对合法 H2D/D2H/D2D 组合返回 kUnimplemented),故
// 该退化是安全兜底,不掩盖真实拷贝失败(kUnimplemented 以外的错误码原样传播)。
Status copy_via_target_backend(void* dst, Device dst_device, const void* src, Device src_device,
                               size_t bytes, std::string_view target_backend) {
  const Result<hal::Backend*> backend = hal::BackendRegistry::instance().get(target_backend);
  if (!backend.is_ok()) return backend.status();
  // 非 const:允许 return 时自动移动(performance-no-automatic-move)。
  Status status = backend.value()->copy(dst, dst_device, src, src_device, bytes, nullptr);
  if (status.is_ok()) return status;
  if (status.code() == ErrorCode::kUnimplemented) {
    std::memmove(dst, src, bytes);
    return Status::ok();
  }
  return status;
}

// 把 src 张量搬运到 dst_device 上的一份新 staging 张量(M11,决议点 D:③步
// 输入若在目标 device → host staging + copy D2H;对称地,①步输入若在 cpu →
// device staging + copy H2D)。仅供 run() 对"由更早步骤产出的 slot"调用——
// 图输入 slot 不适用本函数(见 run() 内调用点注释,保持 v0 既有"信任调用方
// 提供的图输入张量"边界不变,避免触碰 validate_host_capable_device 未覆盖的
// 组合)。
Result<Tensor> stage_tensor_to_device(const Tensor& src, Device dst_device,
                                      std::string_view target_backend) {
  const Result<Tensor> allocated =
      allocate_step_output(hal::IoSpec{src.dtype(), src.shape()}, dst_device);
  if (!allocated.is_ok()) return allocated.status();
  Tensor staged = allocated.value();
  const size_t bytes = static_cast<size_t>(src.numel()) * src.dtype().itemsize();
  if (bytes > 0) {
    const Status copy_status = copy_via_target_backend(
        staged.raw_data(), dst_device, src.raw_data(), src.device(), bytes, target_backend);
    if (!copy_status.is_ok()) return copy_status;
  }
  return staged;
}

}  // namespace

void FallbackExecutable::append_step(std::string op, PlanKind kind, ops::KernelFn cpu_kernel,
                                     std::unordered_map<std::string, ir::AttrValue> attrs,
                                     const std::vector<int32_t>& input_slots,
                                     const std::vector<ir::TensorType>& output_types,
                                     int32_t& next_slot, std::vector<int32_t>& out_output_slots) {
  Step step;
  step.op = std::move(op);
  step.kind = kind;
  step.cpu_kernel = cpu_kernel;
  step.attrs = std::move(attrs);
  step.input_slots = input_slots;
  step.output_slots.reserve(output_types.size());
  step.output_specs.reserve(output_types.size());

  out_output_slots.clear();
  out_output_slots.reserve(output_types.size());
  for (const ir::TensorType& type : output_types) {
    step.output_slots.push_back(next_slot);
    step.output_specs.push_back(hal::IoSpec{type.dtype, type.shape});
    out_output_slots.push_back(next_slot);
    ++next_slot;
  }

  steps_.push_back(std::move(step));
}

Status FallbackExecutable::resolve_node(FallbackExecutable& executable, std::string_view op,
                                        const std::unordered_map<std::string, ir::AttrValue>& attrs,
                                        const std::vector<ir::TensorType>& input_types,
                                        const std::vector<int32_t>& input_slots,
                                        const std::vector<ir::TensorType>& output_types,
                                        std::string_view target_backend, bool allow_decomposition,
                                        int32_t& next_slot,
                                        std::vector<int32_t>& out_output_slots) {
  // ① 目标后端 eager kernel。
  const Result<ops::KernelFn> target_lookup =
      ops::KernelRegistry::instance().find(op, target_backend);
  if (target_lookup.is_ok()) {
    executable.append_step(std::string(op), PlanKind::kEagerLaunch, nullptr, attrs, input_slots,
                           output_types, next_slot, out_output_slots);
    return Status::ok();
  }
  const std::string target_fail_reason(target_lookup.status().message());

  // ② decomposition:仅顶层节点允许(单层分解——微图节点不再走②,防递归失控)。
  if (allow_decomposition) {
    const ops::OpSchema* schema = ops::OpRegistry::instance().find(op);
    if (schema == nullptr) {
      // 理论不可达:见本文件头注释(能触发 build() 的图必然已通过
      // shape_inference 的 op 已注册校验)。
      return Status::make(ErrorCode::kInternal, "FallbackExecutable::build: op '" +
                                                    std::string(op) + "' is not registered");
    }

    const ops::DecomposeFn decompose = schema->decomposition();
    if (decompose != nullptr) {
      ops::NodeContext ctx;
      ctx.op = op;
      ctx.input_types = input_types;
      ctx.attrs = &attrs;
      const Result<ir::Graph> micro = decompose(ctx);
      if (!micro.is_ok()) return micro.status();
      const ir::Graph& micro_graph = micro.value();

      if (micro_graph.inputs().size() != input_slots.size()) {
        return Status::make(
            ErrorCode::kInternal,
            "FallbackExecutable::build: op '" + std::string(op) + "' decomposition produced " +
                std::to_string(micro_graph.inputs().size()) + " graph input(s), expected " +
                std::to_string(input_slots.size()) +
                " (violates positional decomposition contract, operator-system.md)");
      }
      // 红旗要求:微图输出个数与本算子输出个数不一致即报错(operator-system.md
      // 「图输出按位对应本算子输出」)。
      if (micro_graph.outputs().size() != output_types.size()) {
        return Status::make(
            ErrorCode::kInvalidArgument,
            "FallbackExecutable::build: op '" + std::string(op) + "' decomposition produced " +
                std::to_string(micro_graph.outputs().size()) + " graph output(s), expected " +
                std::to_string(output_types.size()) +
                " (violates positional decomposition contract, operator-system.md)");
      }

      // 微图 slot 映射:graph_input 直接复用父节点的输入 slot(同一份数据,不
      // 新分配、不拷贝)。
      std::unordered_map<const ir::Value*, int32_t> micro_slot_of;
      micro_slot_of.reserve(micro_graph.inputs().size());
      for (size_t i = 0; i < micro_graph.inputs().size(); ++i) {
        micro_slot_of.emplace(micro_graph.inputs()[i], input_slots[i]);
      }

      for (const ir::Node* micro_node : micro_graph.topological_order()) {
        if (micro_node->op() == ir::kGraphInputOp) continue;

        std::vector<ir::TensorType> micro_input_types;
        std::vector<int32_t> micro_input_slots;
        micro_input_types.reserve(micro_node->inputs().size());
        micro_input_slots.reserve(micro_node->inputs().size());
        for (const ir::Value* micro_input : micro_node->inputs()) {
          micro_input_types.push_back(micro_input->type());
          const auto it = micro_slot_of.find(micro_input);
          if (it == micro_slot_of.end()) {
            // 理论不可达:拓扑序 + SSA 保证 producer 先于消费者处理。
            return Status::make(
                ErrorCode::kInternal,
                "FallbackExecutable::build: op '" + std::string(op) + "' decomposition sub-op '" +
                    std::string(micro_node->op()) +
                    "' input value has no assigned slot (violates topological order invariant)");
          }
          micro_input_slots.push_back(it->second);
        }

        std::vector<ir::TensorType> micro_output_types;
        micro_output_types.reserve(micro_node->outputs().size());
        for (const ir::Value& micro_output : micro_node->outputs()) {
          micro_output_types.push_back(micro_output.type());
        }

        std::vector<int32_t> micro_output_slots;
        // 单层分解:allow_decomposition=false,微图节点不再走②。
        const Status micro_status =
            resolve_node(executable, micro_node->op(), micro_node->attrs(), micro_input_types,
                         micro_input_slots, micro_output_types, target_backend,
                         /*allow_decomposition=*/false, next_slot, micro_output_slots);
        if (!micro_status.is_ok()) {
          // 红旗要求:微图节点 target 与 cpu 双缺 → 硬失败,消息含内外算子名。
          return Status::make(micro_status.code(),
                              "FallbackExecutable::build: op '" + std::string(op) +
                                  "' decomposition sub-op '" + std::string(micro_node->op()) +
                                  "': " + std::string(micro_status.message()));
        }

        for (size_t j = 0; j < micro_node->outputs().size(); ++j) {
          micro_slot_of.emplace(&micro_node->outputs()[j], micro_output_slots[j]);
        }
      }

      // 微图输出 slot 直接复用为本算子输出 slot(按位对应,无需额外拷贝步骤)。
      out_output_slots.clear();
      out_output_slots.reserve(micro_graph.outputs().size());
      for (const ir::Value* micro_output : micro_graph.outputs()) {
        const auto it = micro_slot_of.find(micro_output);
        if (it == micro_slot_of.end()) {
          return Status::make(ErrorCode::kInternal,
                              "FallbackExecutable::build: op '" + std::string(op) +
                                  "' decomposition graph output value has no assigned slot");
        }
        out_output_slots.push_back(it->second);
      }

      log_fallback(op, target_backend, kDecompositionDestination, target_fail_reason);
      return Status::ok();
    }
  }

  // ③ cpu 参考实现(ARCH-041 保证存在;防御式查失败仍报错)。
  const Result<ops::KernelFn> cpu_lookup =
      ops::KernelRegistry::instance().find(op, kCpuBackendName);
  if (!cpu_lookup.is_ok()) {
    return Status::make(ErrorCode::kUnimplemented,
                        "FallbackExecutable::build: op '" + std::string(op) +
                            "' has no kernel for target backend '" + std::string(target_backend) +
                            "' (reason: " + target_fail_reason +
                            ") and no kernel for cpu reference backend (reason: " +
                            std::string(cpu_lookup.status().message()) + ")");
  }

  executable.append_step(std::string(op), PlanKind::kCpuReference, cpu_lookup.value(), attrs,
                         input_slots, output_types, next_slot, out_output_slots);
  log_fallback(op, target_backend, kCpuReferenceDestination, target_fail_reason);
  return Status::ok();
}

Result<std::unique_ptr<FallbackExecutable>> FallbackExecutable::build(
    const ir::Graph& graph, std::string_view target_backend) {
  auto executable = std::make_unique<FallbackExecutable>();
  executable->target_backend_ = std::string(target_backend);

  // 取图 device:V6 保证全图所有 Value 的 device 一致
  // (docs/architecture/ir-design.md 第4章),任取拓扑序中第一个带输出的节点
  // (含 graph_input,其恰有 1 输出)即可代表整图,与
  // src/compiler/passes/backend_lowering.cpp、
  // src/backends/cpu/cpu_backend.cpp 的取法一致。
  bool has_op_node = false;
  bool found_device = false;
  for (const ir::Node* node : graph.topological_order()) {
    if (node->op() != ir::kGraphInputOp) has_op_node = true;
    if (!found_device && !node->outputs().empty()) {
      executable->device_ = node->outputs()[0].type().device;
      found_device = true;
    }
  }
  if (has_op_node && !found_device) {
    return Status::make(
        ErrorCode::kInternal,
        "FallbackExecutable::build: graph has operator node(s) but no Value carries a device");
  }

  // ①slot 表第一段:图输入,按 inputs() 序(同 CpuExecutable::compile 口径)。
  std::unordered_map<const ir::Value*, int32_t> slot_of;
  int32_t next_slot = 0;
  for (const ir::Value* input : graph.inputs()) {
    slot_of.emplace(input, next_slot);
    executable->input_signature_.push_back(hal::IoSpec{input->type().dtype, input->type().shape});
    ++next_slot;
  }

  // ②slot 表第二段:逐非 graph_input 节点解析执行方案(拓扑序)。decomposition
  // 展开的微图节点在 resolve_node 内联入 steps_,next_slot 是贯穿整个 build()
  // 的单一计数器(顶层节点与微图节点共享同一份 slot 编号空间)。
  for (const ir::Node* node : graph.topological_order()) {
    if (node->op() == ir::kGraphInputOp) continue;

    std::vector<ir::TensorType> input_types;
    std::vector<int32_t> input_slots;
    input_types.reserve(node->inputs().size());
    input_slots.reserve(node->inputs().size());
    for (const ir::Value* input : node->inputs()) {
      input_types.push_back(input->type());
      const auto it = slot_of.find(input);
      if (it == slot_of.end()) {
        return Status::make(
            ErrorCode::kInternal,
            "FallbackExecutable::build: input value has no assigned slot for node '" +
                std::string(node->op()) + "' (violates topological order invariant)");
      }
      input_slots.push_back(it->second);
    }

    std::vector<ir::TensorType> output_types;
    output_types.reserve(node->outputs().size());
    for (const ir::Value& output : node->outputs()) output_types.push_back(output.type());

    std::vector<int32_t> output_slots;
    const Status resolve_status =
        resolve_node(*executable, node->op(), node->attrs(), input_types, input_slots, output_types,
                     target_backend, /*allow_decomposition=*/true, next_slot, output_slots);
    if (!resolve_status.is_ok()) return resolve_status;

    for (size_t j = 0; j < node->outputs().size(); ++j) {
      slot_of.emplace(&node->outputs()[j], output_slots[j]);
    }
  }

  executable->slot_count_ = next_slot;

  // ③图输出签名与其对应 slot。
  executable->output_signature_.reserve(graph.outputs().size());
  executable->output_slots_.reserve(graph.outputs().size());
  for (const ir::Value* output : graph.outputs()) {
    executable->output_signature_.push_back(
        hal::IoSpec{output->type().dtype, output->type().shape});
    const auto it = slot_of.find(output);
    if (it == slot_of.end()) {
      return Status::make(ErrorCode::kInternal,
                          "FallbackExecutable::build: graph output value has no assigned slot");
    }
    executable->output_slots_.push_back(it->second);
  }

  return executable;
}

Status FallbackExecutable::run(std::span<const Tensor> inputs, std::span<Tensor> outputs,
                               hal::Stream& stream) {
  FRAME_RETURN_IF_ERROR(
      hal::validate_io_signature(inputs, input_signature_, "FallbackExecutable::run: input"));
  FRAME_RETURN_IF_ERROR(
      hal::validate_io_signature(outputs, output_signature_, "FallbackExecutable::run: output"));

  // v0 内存边界防御(见本文件匿名命名空间 validate_host_capable_device 头注释):
  // 调用方提供的 inputs/outputs 张量归属设备须是 cpu 或本产物的目标后端,否则
  // 防御式硬失败而非静默按 host 内存处理。
  for (const Tensor& input : inputs) {
    FRAME_RETURN_IF_ERROR(validate_host_capable_device(input.device(), target_backend_, "input"));
  }
  for (const Tensor& output : outputs) {
    FRAME_RETURN_IF_ERROR(validate_host_capable_device(output.device(), target_backend_, "output"));
  }

  // 执行循环与 CpuExecutable::run 表面同构度不足以套用 REUSE-002"同构复制"
  // 判据:kind 分派(Backend::launch vs 直调 KernelFn)、双 Device(目标后端/
  // cpu)各自解析 Allocator、无 arena(朴素逐步分配替代按偏移复用),均是本类
  // 与 CpuExecutable 的核心差异点,故不抽取公共循环体,仅共享
  // hal::validate_io_signature 这一底层设施(已抽取,见上方两条校验)。
  std::vector<Tensor> slots(static_cast<size_t>(slot_count_));
  for (size_t i = 0; i < inputs.size(); ++i) {
    slots[i] = inputs[i];
  }

  for (const Step& step : steps_) {
    const Device step_device = (step.kind == PlanKind::kCpuReference) ? cpu_device() : device_;
    const auto graph_input_slot_count = static_cast<int32_t>(input_signature_.size());

    // 跨设备编排(M11,决议点D):某 slot 若由更早的步骤产出(非图输入 slot)
    // 且所在 device 与本步骤需求 device 不同,插入一次 Backend::copy 搬运
    // (D2H/H2D,视两端而定)。图输入 slot 不做本判断——v0 既有边界仍信任调用
    // 方按 validate_host_capable_device 提供的张量(cpu 或目标后端),不额外
    // 搬运,避免与该校验放行的组合冲突。
    std::vector<Tensor> step_inputs;
    step_inputs.reserve(step.input_slots.size());
    for (const int32_t slot : step.input_slots) {
      const Tensor& slot_tensor = slots[static_cast<size_t>(slot)];
      const bool is_step_produced_slot = slot >= graph_input_slot_count;
      if (is_step_produced_slot && !(slot_tensor.device() == step_device)) {
        const Result<Tensor> staged =
            stage_tensor_to_device(slot_tensor, step_device, target_backend_);
        if (!staged.is_ok()) return staged.status();
        step_inputs.push_back(staged.value());
      } else {
        step_inputs.push_back(slot_tensor);
      }
    }

    std::vector<Tensor> step_outputs;
    step_outputs.reserve(step.output_specs.size());
    for (const hal::IoSpec& spec : step.output_specs) {
      const Result<Tensor> allocated = allocate_step_output(spec, step_device);
      if (!allocated.is_ok()) return allocated.status();
      step_outputs.push_back(allocated.value());
    }

    Status status;
    if (step.kind == PlanKind::kEagerLaunch) {
      const Result<hal::Backend*> backend = hal::BackendRegistry::instance().get(target_backend_);
      if (!backend.is_ok()) return backend.status();
      hal::KernelInvocation invocation;
      invocation.op = step.op;
      invocation.inputs = step_inputs;
      invocation.outputs = step_outputs;
      invocation.attrs = &step.attrs;
      invocation.device = step_device;
      status = backend.value()->launch(invocation, &stream);
    } else {
      ops::KernelContext ctx{step_inputs, step_outputs, &step.attrs, step_device, nullptr};
      status = step.cpu_kernel(ctx);
    }
    if (!status.is_ok()) {
      return Status::make(status.code(),
                          "FallbackExecutable::run: op '" + step.op +
                              "' execution failed: " + std::string(status.message()));
    }

    for (size_t i = 0; i < step.output_slots.size(); ++i) {
      slots[static_cast<size_t>(step.output_slots[i])] = step_outputs[i];
    }
  }

  // 最终图输出 copyout 边(M11,design-reviewer REVISE 闭环裁决修订2):
  // fallback_executable.cpp 曾经的无条件 memmove 在真实 device 指针上会
  // 损坏/崩溃(memmove 是纯 host 操作,不能解引用设备内存,且不遵守 stream
  // 顺序——即便设备内存恰好对 host 可见,①步产出的异步 kernel 结果也可能
  // 尚未就绪)。此处按 slot 的实际 device 与调用方 outputs 的实际 device 决定
  // 路径:两端均为 cpu(host 内存,含③步产出与恒等直通图输入)直接 memmove
  // (快路径,避免为纯 cpu 回退场景引入不必要的 Backend::copy 间接开销);
  // 其余组合(任一侧非 cpu,含①步产出的真实 device 输出)一律经
  // copy_via_target_backend(D2D/H2D 排入正确的执行顺序;该函数对
  // kUnimplemented 目标后端——纯 host 替身——自动退化为 memmove,兼容 v0 既有
  // 边界)。
  for (size_t i = 0; i < output_slots_.size(); ++i) {
    const Tensor& src = slots[static_cast<size_t>(output_slots_[i])];
    Tensor& dst = outputs[i];
    const size_t bytes = static_cast<size_t>(src.numel()) * src.dtype().itemsize();
    if (bytes == 0) continue;
    if (src.device().backend == kCpuBackendName && dst.device().backend == kCpuBackendName) {
      std::memmove(dst.raw_data(), src.raw_data(), bytes);
      continue;
    }
    const Status copy_status = copy_via_target_backend(dst.raw_data(), dst.device(), src.raw_data(),
                                                       src.device(), bytes, target_backend_);
    if (!copy_status.is_ok()) return copy_status;
  }
  return Status::ok();
}

}  // namespace frame::runtime
