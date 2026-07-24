// 标准编译管线命名入口 standard_pipeline 的实现单元。

#include <array>

#include <frame/compiler/pipeline.h>

namespace frame::compiler {

Result<PassManager> standard_pipeline(std::string_view /*backend*/) {
  // backend 参数本里程碑忽略:M6 九段管线全部为直通桩,尚无需要区分后端的
  // 行为;待 M7 backend_lowering 实化后按后端选择 Backend::compile 时再接线
  // (实化待办见 src/compiler/passes/backend_lowering.cpp)。

  // 单一来源:与 pipeline.h 头注释的固定全序逐字一致(ARCH-053)。
  static constexpr std::array<std::string_view, 9> kStandardPipelineOrder = {
      "canonicalize",          "shape_inference",
      "constant_folding",      "common_subexpression_elimination",
      "dead_node_elimination", "layout_assignment",
      "operator_fusion",       "memory_planning",
      "backend_lowering",
  };

  PassManager manager;
  for (const std::string_view pass_name : kStandardPipelineOrder) {
    manager.add(pass_name);
  }
  return manager;
}

}  // namespace frame::compiler
