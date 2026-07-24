// 内置 pass:死节点消除(dead_node_elimination)——删除无输出可达路径的节点。
// pass 名 = 全词文件名(见 include/frame/compiler/pipeline.h)。
//
// 反向可达标记(M8,决议点 F):worklist 种子 = 图输出各 Value 的 producer ∪
// 全部带 OpTrait::kHasSideEffect 的节点;沿输入边回溯标记
// (unordered_set<const Node*>)。
// 保留条款:已标记(可达图输出或带副作用)∪ graph_input 节点(op ==
// ir::kGraphInputOp,无条件保留——删除未被使用的 graph_input 会改变图输入
// 签名,v0 禁止,呼应 §3.5 后置条件的图签名不变式豁免)。
// 删除:未保留节点按逆拓扑序(topological_order() 快照反向,先消费者后生产者)
// erase_node——erase_node 要求节点输出未被引用,逆拓扑序保证删除某节点前,
// 其全部消费者(必定排在其后)均已先被处理:若消费者可达则该节点自身也已被
// 标记可达(回溯标记沿输入边传递,矛盾),若消费者不可达则已在本次逆序遍历
// 中被先一步删除;二者恰好覆盖全部情形,故 erase_node 在本 pass 内恒不因
// "输出被引用"报错。错误透传不吞(any erase_node 失败即视为 pass 失败)。

#include <string_view>
#include <unordered_set>
#include <vector>

#include <frame/compiler/pass.h>
#include <frame/ir/graph.h>
#include <frame/ops/op_registry.h>

namespace frame::compiler {
namespace {

// CRTP 接入:只需提供 static constexpr kName 与 run_impl(ir::Graph&)。
class DeadNodeEliminationPass final : public PassBase<DeadNodeEliminationPass> {
 public:
  static constexpr std::string_view kName = "dead_node_elimination";

  Status run_impl(ir::Graph& graph) {
    std::unordered_set<const ir::Node*> reachable;
    std::vector<const ir::Node*> worklist;

    auto seed = [&](const ir::Node* node) {
      if (node != nullptr && reachable.insert(node).second) {
        worklist.push_back(node);
      }
    };

    for (const ir::Value* output : graph.outputs()) {
      seed(output->producer());
    }
    for (const ir::Node* node : graph.topological_order()) {
      const ops::OpSchema* schema = ops::OpRegistry::instance().find(node->op());
      if (schema != nullptr && schema->has_trait(ops::OpTrait::kHasSideEffect)) {
        seed(node);
      }
    }

    while (!worklist.empty()) {
      const ir::Node* current = worklist.back();
      worklist.pop_back();
      for (const ir::Value* input : current->inputs()) {
        seed(input->producer());
      }
    }

    const std::vector<ir::Node*> snapshot = graph.topological_order();
    for (auto it = snapshot.rbegin(); it != snapshot.rend(); ++it) {
      ir::Node* node = *it;
      if (node->op() == ir::kGraphInputOp) continue;  // 图签名不变式,无条件保留
      if (reachable.find(node) != reachable.end()) continue;
      FRAME_RETURN_IF_ERROR(graph.erase_node(node));
    }
    return Status::ok();
  }
};

}  // namespace

FRAME_REGISTER_PASS(DeadNodeEliminationPass);

}  // namespace frame::compiler
