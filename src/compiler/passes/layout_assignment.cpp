// 内置 pass:布局指派(layout_assignment)——为张量选择/传播数据布局。
// pass 名 = 全词文件名(见 include/frame/compiler/pipeline.h)。
//
// v0 口径(M9,决议点 A):唯一具体 layout 是 Layout::kRowMajor(core 的
// row_major_strides 工具即行优先),cpu 后端别无偏好;全图统一指派——遍历
// 全部 Value(含 graph_input 与图输出,凡是某节点的输出即被遍历到,图输出
// 天然覆盖)经 Graph::assign_layout 写入 kRowMajor:已是 kUnknown → 首次
// 指派;已是 kRowMajor → 幂等跳过(assign_layout 内部处理,§3.1 幂等契约天然
// 满足)。v0 单一 layout ⇒ §3.6"冲突处插入转换节点"在单 layout 下恒为空集,
// 不发明转换算子。

#include <cstdint>
#include <string_view>
#include <vector>

#include <frame/compiler/pass.h>
#include <frame/ir/graph.h>

namespace frame::compiler {
namespace {

// CRTP 接入:只需提供 static constexpr kName 与 run_impl(ir::Graph&)。
class LayoutAssignmentPass final : public PassBase<LayoutAssignmentPass> {
 public:
  static constexpr std::string_view kName = "layout_assignment";

  Status run_impl(ir::Graph& graph) {
    // TODO(FRAME-DESIGN): 依后端偏好为各 Value 指派 Layout(CompileOptions
    //   扩展/Backend 查询方法)并在冲突处插入显式布局转换节点;v0 单一
    //   layout 下该分支恒为空集,不预先发明转换算子。参考:
    //   docs/architecture/compiler-passes.md §3.6;docs/plan/milestones.md
    //   M11/M14 节。完成判据:后端偏好查询面落地后,本 pass 按偏好指派
    //   而非恒定 kRowMajor。
    for (ir::Node* node : graph.topological_order()) {
      const std::vector<ir::Value>& outputs = node->outputs();
      for (int32_t i = 0; i < static_cast<int32_t>(outputs.size()); ++i) {
        FRAME_RETURN_IF_ERROR(graph.assign_layout(node->output(i), ir::Layout::kRowMajor));
      }
    }
    return Status::ok();
  }
};

}  // namespace

FRAME_REGISTER_PASS(LayoutAssignmentPass);

}  // namespace frame::compiler
