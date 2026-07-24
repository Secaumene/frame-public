// 内置 pass:算子融合(operator_fusion)——按 OpTrait::kFusable 合并相邻算子。
// pass 名 = 全词文件名(见 include/frame/compiler/pipeline.h)。
//
// v0 算法(M9,决议点 C 覆盖版,线性链):
//   候选节点:schema 已注册、同时带 kElementwise + kFusable trait、无 attrs、
//   恰 1 输出。链条件:前驱输出仅被后继消费(单消费者,且不在图输出列表)
//   且后继首输入(inputs()[0])== 该输出。沿拓扑序贪心生长最长链,长度 >= 2
//   才融合。
//   算法安全性论证(为何不会重复处理/越界访问已删除节点):consumers 映射与
//   graph_output_set 均基于运行前拍下的拓扑序快照一次性算好,不随融合过程
//   重算;某节点是否可作为某条链的"后继"仅由其自身"第 0 输入是否等于某个
//   候选前驱的输出"这一静态属性决定,故每个节点至多属于一条链(链与链之间
//   不重叠,forward-only 生长天然无歧义);visited 集合防止把已被早前完成的
//   链吸收(并因而已被 erase_node 删除)的节点当作新链起点重新处理。
//   融合时显式断言链内各 sub-op 具备 cpu kernel
//   (KernelRegistry::find(op, kCpuBackendName)——此处 "cpu" 特指 ARCH-041
//   参考基准,与图的实际目标后端无关,目标后端自身的支持性判定归
//   backend_lowering pass);缺失则该链整体不融合(跳过,不报错)。
//   改写:收集外部输入序(第 i 段 i==0 时其全部输入均为外部输入,i>0 时仅
//   inputs()[1:] 为外部输入——inputs()[0] 是链内部连接,不计入)→
//   create_node(fused_elementwise_internal, 外部输入, {链尾输出类型}) +
//   encode_fused_chain 设 attrs → replace_all_uses(链尾输出, 融合输出)
//   (A2 第二豁免生效:融合节点的全部输入的 producer 拓扑下标均严格 <
//   链尾节点下标,见 include/frame/ir/graph.h 头注释安全论证)→ 逆拓扑序
//   erase_node 链内节点。
//   幂等:融合产物 fused_elementwise_internal 不标 kFusable trait ⇒ 二跑无
//   候选链(候选判定即排除),幂等自然满足。

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <frame/compiler/pass.h>
#include <frame/core/device.h>
#include <frame/ir/graph.h>
#include <frame/ops/fused_elementwise_utils.h>
#include <frame/ops/kernel_registry.h>
#include <frame/ops/op_registry.h>

namespace frame::compiler {
namespace {

// 候选判定:schema 已注册(graph_input 天然被 find()==nullptr 排除,无需
// 显式判空,同 constant_folding.cpp::is_foldable_node 既有理由)、同时带
// kElementwise + kFusable、无 attrs、恰 1 输出。
bool is_fusable_candidate(const ir::Node* node) {
  const ops::OpSchema* schema = ops::OpRegistry::instance().find(node->op());
  if (schema == nullptr) return false;
  if (!schema->has_trait(ops::OpTrait::kElementwise)) return false;
  if (!schema->has_trait(ops::OpTrait::kFusable)) return false;
  if (!node->attrs().empty()) return false;
  if (node->outputs().size() != 1) return false;
  return true;
}

// 改写单条已确定融合的链(size>=2,各 sub-op 均已确认具备 cpu kernel)。
Status fuse_chain(ir::Graph& graph, const std::vector<ir::Node*>& chain) {
  std::vector<ir::Value*> external_inputs;
  std::vector<std::string> sub_ops;
  std::vector<int64_t> arities;
  sub_ops.reserve(chain.size());
  arities.reserve(chain.size());

  for (size_t i = 0; i < chain.size(); ++i) {
    ir::Node* member = chain[i];
    sub_ops.push_back(std::string(member->op()));
    arities.push_back(static_cast<int64_t>(member->inputs().size()));
    // 接线约定(决议点 B):第 i 段(i>0)的第 0 输入 = 前段输出,不计入外部
    // 输入;其余输入按序即外部输入。
    const size_t start_index = (i == 0) ? 0 : 1;
    for (size_t j = start_index; j < member->inputs().size(); ++j) {
      external_inputs.push_back(member->inputs()[j]);
    }
  }

  ir::Node* tail = chain.back();
  // 类型须在任何 erase 之前提取(erase_node(tail) 后其 Value 随节点销毁)。
  const ir::TensorType tail_type = tail->output(0)->type();

  const Result<ir::Node*> fused_node_result =
      graph.create_node(std::string(ops::kFusedElementwiseOpName), external_inputs, {tail_type});
  if (!fused_node_result.is_ok()) return fused_node_result.status();
  ir::Node* fused_node = fused_node_result.value();

  std::unordered_map<std::string, ir::AttrValue> attrs;
  ops::encode_fused_chain(sub_ops, arities, attrs);
  for (auto& [name, value] : attrs) {
    fused_node->set_attr(name, std::move(value));
  }

  // replace_all_uses 触发 A2 第二豁免重定位:fused_node 由 create_node 追加
  // 在拓扑序尾部(晚于 tail),但其全部输入(external_inputs)的 producer 拓扑
  // 下标均严格 < tail 的下标(链内每个成员的输入,按 SSA 定义均须先于该成员
  // 产出,而链内全部成员的拓扑下标均 <= tail 的下标),满足重定位条件。
  FRAME_RETURN_IF_ERROR(graph.replace_all_uses(tail->output(0), fused_node->output(0)));

  // 逆拓扑序 erase 链内原节点(链尾先删,链头后删):erase_node 要求节点输出
  // 未被引用——链尾的唯一外部引用已被上一步 replace_all_uses 重定向;链内
  // 每个非尾节点的输出按链条件只被其后继(链内下一节点)消费,一旦后继已被
  // 删除,该节点自身也不再被引用。
  for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
    FRAME_RETURN_IF_ERROR(graph.erase_node(*it));
  }
  return Status::ok();
}

// CRTP 接入:只需提供 static constexpr kName 与 run_impl(ir::Graph&)。
class OperatorFusionPass final : public PassBase<OperatorFusionPass> {
 public:
  static constexpr std::string_view kName = "operator_fusion";

  Status run_impl(ir::Graph& graph) {
    // 快照(与 constant_folding 同模式,REUSE-002):遍历对象是运行前拍下的
    // 拓扑序快照;融合过程中会 create_node/erase_node 就地改写活跃图。
    const std::vector<ir::Node*> snapshot = graph.topological_order();

    const std::unordered_set<const ir::Value*> graph_output_set(graph.outputs().begin(),
                                                                graph.outputs().end());
    // Value -> 消费该 Value 的节点列表(用于判定"单消费者"),基于运行前的
    // 原始结构一次性算好,不随融合过程重算(算法安全性论证见文件头注释)。
    std::unordered_map<const ir::Value*, std::vector<ir::Node*>> consumers;
    for (ir::Node* node : snapshot) {
      for (ir::Value* input : node->inputs()) {
        consumers[input].push_back(node);
      }
    }

    std::unordered_set<const ir::Node*> visited;

    for (ir::Node* start : snapshot) {
      if (visited.find(start) != visited.end()) continue;
      if (!is_fusable_candidate(start)) {
        visited.insert(start);
        continue;
      }

      std::vector<ir::Node*> chain{start};
      visited.insert(start);
      while (true) {
        ir::Node* last = chain.back();
        const ir::Value* last_output = last->output(0);
        if (graph_output_set.find(last_output) != graph_output_set.end()) break;
        const auto consumers_it = consumers.find(last_output);
        if (consumers_it == consumers.end() || consumers_it->second.size() != 1) break;
        ir::Node* next = consumers_it->second[0];
        if (visited.find(next) != visited.end()) break;  // 理论不可达,防御性(见文件头注释)
        if (!is_fusable_candidate(next)) break;
        if (next->inputs().empty() || next->inputs()[0] != last_output) break;
        chain.push_back(next);
        visited.insert(next);
      }

      if (chain.size() < 2) continue;

      // 显式断言链内各 sub-op 具备 cpu kernel(此处 "cpu" 特指 ARCH-041 参考
      // 基准,见文件头注释);缺失则整条链不融合(跳过,不报错)。
      bool all_have_cpu_kernel = true;
      for (const ir::Node* member : chain) {
        if (!ops::KernelRegistry::instance().find(member->op(), kCpuBackendName).is_ok()) {
          all_have_cpu_kernel = false;
          break;
        }
      }
      if (!all_have_cpu_kernel) continue;

      FRAME_RETURN_IF_ERROR(fuse_chain(graph, chain));
    }
    return Status::ok();
  }
};

}  // namespace

FRAME_REGISTER_PASS(OperatorFusionPass);

}  // namespace frame::compiler
