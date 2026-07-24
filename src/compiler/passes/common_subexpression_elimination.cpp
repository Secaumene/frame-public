// 内置 pass:公共子表达式消除(common_subexpression_elimination)——合并等价
// 节点(同 op、同输入、同属性;commutative trait 参与输入归一化),带
// has_side_effect trait 的节点除外。
// pass 名 = 全词文件名(见 include/frame/compiler/pipeline.h)。
//
// 等价键(M8,决议点 E):op 名 + 各输入的 "(producer 拓扑序下标,
// output_index)" 数字对序列(kCommutative 且恰 2 输入时对两个数字对排序
// 归一)+ attrs 按键排序后逐个用 ir::format_attr_value(A2 公开版,
// REUSE-002:与 dump_text 共用同一份属性文本化,禁止第二份复制)文本化。
// 用拓扑序下标而非裸指针值:裸指针值运行间不确定,会破坏确定性纪律
// (golden 测试要求跨进程/跨运行同图产出同结果)。
// 排除条款:graph_input 节点不进表也不被并(图签名不变式——两个同型
// graph_input 若被误判等价并合并,会经 erase_node 改变图输入签名,v0 禁止,
// design-reviewer 必须修复项②);带 has_side_effect trait 的节点同样不进表
// 不被并。constant 节点自然参与去重(键含 attrs,同值常量天然合并到同一
// 表项)。
// 快照下标论证:node -> 拓扑序下标 的映射基于运行前拍下的
// topological_order() 快照一次性建立,CSE 全程只删除节点(不新建节点,与
// constant_folding 不同),故该映射覆盖的节点集合在本 pass 运行期间不会扩大;
// 被合并(erase_node)的节点此后不再作为任何节点输入的 producer 出现——
// replace_all_uses 已把全部指向它的引用改写到保留节点——因此其下标不会再被
// 后续任何 node_key() 调用查询到,映射失效不构成问题。
// 前置依赖:canonicalize 已完成(常量输入位置已归一化,commutative 节点的
// "常量居右"减少不必要的模式数量),该依赖由标准管线顺序保证(§3.4 前置
// 条件),本 pass 内部不重复查验;脱离标准管线单独调用本 pass 是调用方责任。

#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <frame/compiler/pass.h>
#include <frame/ir/graph.h>
#include <frame/ir/serialization.h>
#include <frame/ops/op_registry.h>

namespace frame::compiler {
namespace {

// 单个节点的等价键(细节见文件头注释)。topo_index 覆盖 CSE 运行前快照内的
// 全部节点(node 的每个输入 producer 必属其中,论证同文件头)。
std::string node_key(const ir::Node* node,
                     const std::unordered_map<const ir::Node*, int64_t>& topo_index) {
  std::vector<std::pair<int64_t, int32_t>> input_keys;
  input_keys.reserve(node->inputs().size());
  for (const ir::Value* input : node->inputs()) {
    const auto it = topo_index.find(input->producer());
    input_keys.emplace_back(it->second, input->output_index());
  }

  const ops::OpSchema* schema = ops::OpRegistry::instance().find(node->op());
  const bool normalize_pair =
      schema != nullptr && schema->has_trait(ops::OpTrait::kCommutative) && input_keys.size() == 2;
  if (normalize_pair && input_keys[1] < input_keys[0]) {
    std::swap(input_keys[0], input_keys[1]);
  }

  std::string key(node->op());
  key += '|';
  for (const auto& [producer_index, output_index] : input_keys) {
    key += std::to_string(producer_index);
    key += ':';
    key += std::to_string(output_index);
    key += ',';
  }
  key += '|';

  // attrs 按名字典序输出(node->attrs() 枚举顺序不确定,与序列化层同款
  // std::map 排序契约,见 include/frame/ir/serialization.h 头注释)。
  std::map<std::string, const ir::AttrValue*> sorted_attrs;
  for (const auto& [name, value] : node->attrs()) {
    sorted_attrs.emplace(name, &value);
  }
  for (const auto& [name, value] : sorted_attrs) {
    key += name;
    key += '=';
    key += ir::format_attr_value(*value);
    key += ';';
  }
  return key;
}

// CRTP 接入:只需提供 static constexpr kName 与 run_impl(ir::Graph&)。
class CommonSubexpressionEliminationPass final
    : public PassBase<CommonSubexpressionEliminationPass> {
 public:
  static constexpr std::string_view kName = "common_subexpression_elimination";

  Status run_impl(ir::Graph& graph) {
    const std::vector<ir::Node*> snapshot = graph.topological_order();
    std::unordered_map<const ir::Node*, int64_t> topo_index;
    topo_index.reserve(snapshot.size());
    for (size_t i = 0; i < snapshot.size(); ++i) {
      topo_index.emplace(snapshot[i], static_cast<int64_t>(i));
    }

    std::unordered_map<std::string, ir::Node*> seen;
    for (ir::Node* node : snapshot) {
      if (node->op() == ir::kGraphInputOp) continue;  // 图签名不变式,不进表不被并

      const ops::OpSchema* schema = ops::OpRegistry::instance().find(node->op());
      if (schema == nullptr) continue;  // 防御式跳过:verify 已保证注册,理论不可达
      if (schema->has_trait(ops::OpTrait::kHasSideEffect)) continue;

      const std::string key = node_key(node, topo_index);
      const auto [it, inserted] = seen.try_emplace(key, node);
      if (inserted) continue;

      ir::Node* earlier = it->second;
      // v0 全部算子单输出,循环写法兼容未来多输出扩展;earlier 与 node 同 op
      // 同 schema,输出个数恒等。
      for (int32_t i = 0; i < static_cast<int32_t>(node->outputs().size()); ++i) {
        FRAME_RETURN_IF_ERROR(graph.replace_all_uses(node->output(i), earlier->output(i)));
      }
      FRAME_RETURN_IF_ERROR(graph.erase_node(node));
    }
    return Status::ok();
  }
};

}  // namespace

FRAME_REGISTER_PASS(CommonSubexpressionEliminationPass);

}  // namespace frame::compiler
