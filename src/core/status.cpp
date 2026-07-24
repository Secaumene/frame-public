// Status / Result<T> 错误模型的实现单元。
// 现状:status.h 中 Status 与 Result 全部为内联定义,本翻译单元暂无需落地符号,
// 保留占位以承接未来的非内联成员。

#include <frame/core/status.h>

// 当前 Status/Result<T> 无任何非内联成员需求,本单元预留(留待未来如错误链/
// 结构化诊断上下文等扩展需要非内联符号时使用)。
