#pragma once
// json_loader:JSON 模型描述文件 -> frame::frontend::ModelSpec 的结构化载入
// (ADR-0017 工具层职责;ADR-0018 nlohmann/json 准入)。本文件是 nlohmann/json
// 在仓库内的消费点之一(限定 tools/ 与 tests/,REUSE-012);frame_frontend 库
// 不依赖 nlohmann,JSON 语法解析与结构性校验(含 FE-001 schema_version、必填
// 字段存在性与类型)全部收敛于此。语义校验(frontend-dsl.md 第 3 节
// FE-002~005)由 frame::frontend::validate 执行,本函数载入成功后不重复校验。

#include <string>

#include <frame/core/status.h>
#include <frame/frontend/model_spec.h>

namespace frame_dslc {

// 从 path 读取 JSON 文件并解析为 ModelSpec。失败场景(文件打开失败、JSON 语法
// 错误、schema_version 非 0、必填字段缺失或类型不符、无法识别的枚举取值)均
// 返回英文错误消息(含违例字段名,LANG-005)的 Status;成功返回不代表语义合法
// ——调用方仍需经 frame::frontend::validate(或 lower_to_graph/run_training/
// emit_cpp 内部隐含调用)完成第 3 节 FE-002~005 校验。
frame::Result<frame::frontend::ModelSpec> load_model_spec_from_json_file(const std::string& path);

}  // namespace frame_dslc
