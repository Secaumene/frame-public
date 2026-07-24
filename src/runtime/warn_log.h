#pragma once
// warn_log:eager 回退链 WARN 级日志的极薄内部实现。src/runtime/ 内部头,不进
// include/frame/ ——非公共 API(m10-design-brief 决议点 C)。
//
// 设计裁决:不引入 spdlog(REUSE-010:接入即新增 FetchContent,须先 ADR;回退
// WARN 一处需求不足以立项,design-reviewer REVISE 闭环已确认该 REUSE-010 解读)。
// 本函数**永不生长为日志框架**——禁止新增 level/sink/格式化 DSL 等能力,后续如
// 需要更完整的日志设施,走下方标注的独立 ADR 议题,不在本函数上叠加。
//
// TODO(FRAME-DEP): spdlog 接入已裁决推迟(ADR-0012),按其重评条件触发。
//   参考:docs/decisions/0012-defer-spdlog-keep-thin-warn-log.md。完成判据:
//   ADR-0012 重评为已接受且全仓 fprintf 日志点收敛,或该 ADR 被取代。

#include <string_view>

namespace frame::runtime {

// 打印一条 WARN 级诊断到 stderr,格式固定为 "[frame][WARN] <message>\n"。
// message 须为英文(LANG-005)。
void warn_log(std::string_view message);

}  // namespace frame::runtime
