// 内置 pass:内存规划(memory_planning)——静态 shape 下 AOT 规划缓冲区复用与
// 生命周期(铁律 #1① 的关键收益),产出供 runtime 落地的分配计划。
// pass 名 = 全词文件名(见 include/frame/compiler/pipeline.h)。
//
// v0 口径(M9,决议点 D 覆盖版,§3.8 随件修订,照 M7 backend_lowering pass
// 先例):Pass::run(Graph&) 无产物通道,本 pass 职责收窄为"计划可计算性
// 验证"——调 ir::compute_memory_plan(graph) 复核图可规划(无动态维等),
// 失败即报错;成功则不改图直接放行。真正落地(把朴素逐步分配替换为 arena
// 分配 + 偏移切片)在 CpuExecutable::compile/run(同一函数 REUSE-002 复用,
// 见 src/backends/cpu/cpu_executable.cpp)。§3.8 后置条件("每个 Value 有
// 确定的缓冲区偏移与生命周期区间")解释为"计划可由图确定性推导"。

#include <string_view>

#include <frame/compiler/pass.h>
#include <frame/ir/graph.h>
#include <frame/ir/memory_plan.h>

namespace frame::compiler {
namespace {

// CRTP 接入:只需提供 static constexpr kName 与 run_impl(ir::Graph&)。
class MemoryPlanningPass final : public PassBase<MemoryPlanningPass> {
 public:
  static constexpr std::string_view kName = "memory_planning";

  Status run_impl(ir::Graph& graph) {
    const Result<ir::MemoryPlan> plan = ir::compute_memory_plan(graph);
    if (!plan.is_ok()) return plan.status();
    return Status::ok();
  }
};

}  // namespace

FRAME_REGISTER_PASS(MemoryPlanningPass);

}  // namespace frame::compiler
