// ExecutablePlan 构建的实现单元(声明见 include/frame/ops/executable_plan.h)。
// 算法与边界与此前 src/backends/cpu/cpu_executable.cpp::CpuExecutable::compile
// 的对应段落逐一同源(M11 抽取,REUSE-002)。

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include <frame/ir/graph.h>
#include <frame/ir/memory_plan.h>
#include <frame/ops/executable_plan.h>

namespace frame::ops {

Result<ExecutablePlan> build_executable_plan(const ir::Graph& graph, std::string_view backend) {
  ExecutablePlan plan;

  // ir::compute_memory_plan 落地:计划在 pass 内只验证可计算性,真正落地在
  // 本函数(REUSE-002,与此前 CpuExecutable::compile 同一份算法)。
  const Result<ir::MemoryPlan> memory_plan_result = ir::compute_memory_plan(graph);
  if (!memory_plan_result.is_ok()) {
    return Status::make(
        memory_plan_result.status().code(),
        "build_executable_plan: " + std::string(memory_plan_result.status().message()));
  }
  const ir::MemoryPlan& memory_plan = memory_plan_result.value();
  plan.arena_total_bytes = memory_plan.total_bytes;

  // ①slot 表第一段:图输入,按 inputs() 序。
  std::unordered_map<const ir::Value*, int32_t> slot_of;
  int32_t next_slot = 0;
  for (const ir::Value* input : graph.inputs()) {
    slot_of.emplace(input, next_slot);
    plan.input_types.push_back(input->type());
    ++next_slot;
  }

  // ②slot 表第二段:各非 graph_input 节点的输出,按拓扑序;同一遍历顺带解析
  // KernelFn 并组装执行步骤(此时该节点全部输入的 slot 已在①或更早的②迭代
  // 中登记完毕,拓扑序 + SSA 保证生产者先于消费者)。topo_index 与
  // ir::compute_memory_plan 内部编号口径一致(遍历 topological_order() 时对
  // 每个节点递增,含 graph_input 节点),用于按 (topo_index, output_index)
  // 查询 memory_plan.entries。
  int64_t topo_index = 0;
  for (const ir::Node* node : graph.topological_order()) {
    if (node->op() == ir::kGraphInputOp) {
      ++topo_index;
      continue;
    }

    const Result<KernelFn> kernel = KernelRegistry::instance().find(node->op(), backend);
    if (!kernel.is_ok()) {
      // 哨兵码翻译(ARCH-031 同源,与 backend_lowering pass 的落点一致):
      // KernelFn 缺失统一翻译为 kUnimplemented,供 runtime::compile 据此触发
      // 回退链;消息原样透传(仍含算子名与后端名)。
      return Status::make(ErrorCode::kUnimplemented,
                          "build_executable_plan: " + std::string(kernel.status().message()));
    }

    ExecutablePlanStep step;
    step.op = std::string(node->op());
    step.kernel = kernel.value();
    step.attrs = node->attrs();

    step.input_slots.reserve(node->inputs().size());
    for (const ir::Value* input : node->inputs()) {
      const auto it = slot_of.find(input);
      if (it == slot_of.end()) {
        // 理论不可达:拓扑序 + V1(SSA)已保证输入的 producer 先于本节点处理,
        // 其 slot 必已登记;fail-fast 而非静默产出错误的执行计划。
        return Status::make(ErrorCode::kInternal,
                            "build_executable_plan: input value has no assigned slot for node '" +
                                step.op + "' (violates topological order invariant)");
      }
      step.input_slots.push_back(it->second);
    }

    step.output_slots.reserve(node->outputs().size());
    step.output_types.reserve(node->outputs().size());
    step.output_arena_offsets.reserve(node->outputs().size());
    for (int32_t j = 0; j < static_cast<int32_t>(node->outputs().size()); ++j) {
      const ir::Value& output = node->outputs()[static_cast<size_t>(j)];
      slot_of.emplace(&output, next_slot);
      step.output_slots.push_back(next_slot);
      step.output_types.push_back(output.type());
      const auto plan_it = memory_plan.entries.find(ir::MemoryPlanKey{topo_index, j});
      if (plan_it != memory_plan.entries.end()) {
        step.output_arena_offsets.push_back(plan_it->second.byte_offset);
      } else {
        // 未入 plan:图输出 Value(compute_memory_plan 已排除,见
        // include/frame/ir/memory_plan.h 头注释边界),消费方走独立分配。
        step.output_arena_offsets.push_back(std::nullopt);
      }
      ++next_slot;
    }

    plan.steps.push_back(std::move(step));
    ++topo_index;
  }

  plan.slot_count = next_slot;

  // ③图输出签名与其对应 slot(恒等/仅输入图:图输出直接引用某个图输入的
  // Value*,slot_of 对该 Value* 的登记同样来自①,无需特判)。
  plan.output_types.reserve(graph.outputs().size());
  plan.output_slots.reserve(graph.outputs().size());
  for (const ir::Value* output : graph.outputs()) {
    plan.output_types.push_back(output->type());
    const auto it = slot_of.find(output);
    if (it == slot_of.end()) {
      // 理论不可达:graph.outputs() 中每个 Value 均已由本函数上方两段逻辑
      // 登记过 slot(V1 保证 producer 属于本图)。
      return Status::make(ErrorCode::kInternal,
                          "build_executable_plan: graph output value has no assigned slot");
    }
    plan.output_slots.push_back(it->second);
  }

  return plan;
}

}  // namespace frame::ops
