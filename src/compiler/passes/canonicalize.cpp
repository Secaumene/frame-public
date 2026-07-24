// 内置 pass:归一化(canonicalize)——把等价图形归一化(常量输入位置、别名 op
// 重写),为后续 pass 减少模式数量。
// pass 名 = 全词文件名(见 include/frame/compiler/pipeline.h)。
//
// v0 规则集(M8,仅一条,design-reviewer 已批):
//   规则①「常量居右」——对带 OpTrait::kCommutative 且恰 2 输入的节点,若输入 0
//   的 producer 是 constant(ops::kConstantOpName)而输入 1 的不是,则经
//   graph.swap_node_inputs(node, 0, 1) 把常量交换到右侧;两输入同为/同非常量
//   均不动(全常量子图交给 constant_folding 处理,非常量对无归一化空间)。
//   "constant" 字符串锚是 IR 级概念判定,允许作为本 pass 唯一的字面量锚点
//   (brief 决议点 C 唯一豁免)。
//   规则②「别名 op 重写」——v0 内置算子集无别名 op(add/mul/relu/square/
//   matmul/sum/constant 互不重名互不重叠语义),本里程碑零规则实例;本节留作
//   未来扩展点,不发明尚不存在的算子。
// 幂等性(MUST,ARCH-051/§3.1 后置条件):规则①结构幂等——命中一次后输入 0
// 变为非常量(或双侧均常量,不再满足"输入 1 非常量"条件),再次运行不会
// 重复交换,连跑两次结果逐字节相同。
// 已知局限:constant_folding 折叠子图产生的新常量节点不会回跑本 pass(标准
// 管线单遍线性执行,§3.3/§3.4 之间无回边),故"常量居右"并非管线末端的全图
// 不变式,仅在 canonicalize 运行的那一刻成立;下游若需要该不变式,应重新调用
// 本 pass。

#include <string_view>

#include <frame/compiler/pass.h>
#include <frame/ir/graph.h>
#include <frame/ops/constant_utils.h>
#include <frame/ops/op_registry.h>

namespace frame::compiler {
namespace {

// CRTP 接入:只需提供 static constexpr kName 与 run_impl(ir::Graph&)。
class CanonicalizePass final : public PassBase<CanonicalizePass> {
 public:
  static constexpr std::string_view kName = "canonicalize";

  Status run_impl(ir::Graph& graph) {
    for (ir::Node* node : graph.topological_order()) {
      if (node->inputs().size() != 2) continue;

      // 查无 schema 的节点跳过(防御式):graph.verify() 已保证每个非
      // graph_input 节点均已注册(V3),理论不可达,此处只做兜底不报错。
      const ops::OpSchema* schema = ops::OpRegistry::instance().find(node->op());
      if (schema == nullptr) continue;
      if (!schema->has_trait(ops::OpTrait::kCommutative)) continue;

      const ir::Node* lhs_producer = node->inputs()[0]->producer();
      const ir::Node* rhs_producer = node->inputs()[1]->producer();
      const bool lhs_is_constant =
          lhs_producer != nullptr && lhs_producer->op() == ops::kConstantOpName;
      const bool rhs_is_constant =
          rhs_producer != nullptr && rhs_producer->op() == ops::kConstantOpName;
      if (lhs_is_constant && !rhs_is_constant) {
        FRAME_RETURN_IF_ERROR(graph.swap_node_inputs(node, 0, 1));
      }
    }
    return Status::ok();
  }
};

}  // namespace

FRAME_REGISTER_PASS(CanonicalizePass);

}  // namespace frame::compiler
