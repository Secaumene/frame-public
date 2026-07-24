#pragma once
// 标准编译管线的命名入口。
//
// 【标准管线固定全序(改序或增删标准 pass 必须先有已接受的 ADR,ARCH-053)】:
//   canonicalize
//     → shape_inference
//     → constant_folding
//     → common_subexpression_elimination
//     → dead_node_elimination
//     → layout_assignment
//     → operator_fusion
//     → memory_planning
//     → backend_lowering
// pass 名 = 全词文件名。依据:docs/architecture/compiler-passes.md 第3章。

#include <string_view>

#include <frame/compiler/pass_manager.h>
#include <frame/core/macros.h>
#include <frame/core/status.h>

namespace frame::compiler {

// 构建面向指定后端(注册键字符串,如 "cpu")的标准管线:依上述固定顺序装配
// PassManager,九段全部经 PassRegistry::create(name) 接线(见
// src/compiler/pipeline.cpp)。backend 本里程碑(M6)未使用——九段管线现皆为
// 直通桩,尚无需要区分后端的行为;待 backend_lowering pass 实化(M7)后按后端
// 选择 Backend::compile 时再接线该参数。
FRAME_API Result<PassManager> standard_pipeline(std::string_view backend);

}  // namespace frame::compiler
