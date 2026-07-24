// 内置 pass:后端降低(backend_lowering)——v0 为支持性校验(m7-design-brief 决议
// 点 2):目标后端名取自图 device(V6 单 device 保证唯一),经 BackendRegistry::get
// 取后端存在性,再逐非 graph_input 节点查 KernelRegistry::find(op, backend) 判定
// 是否有可用内核,缺失即返回带算子名与后端名的英文错误(ARCH-031 编译期报错
// 落点)。本 pass 不产出 Executable——Backend::compile 由 runtime 编排入口在
// 标准管线全绿后调用(见 include/frame/runtime/compile.h)。
// 限定语(修订节 5-③):上述 KernelRegistry 支持性判定仅适用于"逐 kernel 模式"
// 后端(cpu/cuda/intel_gpu);"整图模式"后端(intel_npu,交厂商编译器整图处理)
// 的支持性判定不经 KernelRegistry,而是内嵌在其自身 Backend::compile 内部按
// ARCH-031 报错——按执行模式分流本 pass 的判定逻辑是后续议题(前置登记见
// docs/plan/milestones.md M15 节)。
// pass 名 = 全词文件名(见 include/frame/compiler/pipeline.h)。

#include <string>
#include <string_view>

#include <frame/compiler/pass.h>
#include <frame/hal/backend.h>
#include <frame/ir/graph.h>
#include <frame/ops/kernel_registry.h>

namespace frame::compiler {
namespace {

// CRTP 接入:只需提供 static constexpr kName 与 run_impl(ir::Graph&)。
class BackendLoweringPass final : public PassBase<BackendLoweringPass> {
 public:
  static constexpr std::string_view kName = "backend_lowering";

  Status run_impl(ir::Graph& graph) {
    // 判定"仅输入图"(含空图):拓扑序中不存在非 graph_input 节点 ⇒ 无算子需要
    // 判定支持性,直接放行(决议点 2 修订:空图/仅输入图跳过)。
    bool has_op_node = false;
    for (const ir::Node* node : graph.topological_order()) {
      if (node->op() != ir::kGraphInputOp) {
        has_op_node = true;
        break;
      }
    }
    if (!has_op_node) return Status::ok();

    // 目标后端名取自图 device:V6 保证全图所有 Value 的 device 一致
    // (docs/architecture/ir-design.md 第4章),任取拓扑序中第一个带输出的
    // 节点(含 graph_input,其恰有 1 输出,见同文档 V3 结构检查)即可代表整图。
    std::string_view backend_name;
    bool found_device = false;
    for (const ir::Node* node : graph.topological_order()) {
      if (!node->outputs().empty()) {
        backend_name = node->outputs()[0].type().device.backend;
        found_device = true;
        break;
      }
    }
    if (!found_device) {
      return Status::make(ErrorCode::kInternal,
                          "backend_lowering: graph has operator node(s) but no Value carries a "
                          "device, cannot determine target backend");
    }

    const Result<hal::Backend*> backend = hal::BackendRegistry::instance().get(backend_name);
    if (!backend.is_ok()) {
      return Status::make(backend.status().code(),
                          "backend_lowering: " + std::string(backend.status().message()));
    }

    for (const ir::Node* node : graph.topological_order()) {
      if (node->op() == ir::kGraphInputOp) continue;
      const Result<ops::KernelFn> kernel =
          ops::KernelRegistry::instance().find(node->op(), backend_name);
      if (!kernel.is_ok()) {
        // 哨兵码翻译(design-reviewer REVISE 闭环修订 1):逐节点 kernel 缺失是
        // ARCH-031 的"不支持"落点之一,统一翻译为 kUnimplemented,供
        // runtime::compile 据此触发回退链(execution-model.md 第5章);消息
        // 原样透传(仍含算子名与后端名),不改 KernelRegistry::find 自身的
        // kNotFound(该码继续服务 ①③ 与 CpuExecutable 的其余查找场景)。
        return Status::make(ErrorCode::kUnimplemented,
                            "backend_lowering: " + std::string(kernel.status().message()));
      }
    }
    return Status::ok();
  }
};

}  // namespace

FRAME_REGISTER_PASS(BackendLoweringPass);

}  // namespace frame::compiler
