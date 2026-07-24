#pragma once
// ops 内部注册/声明期 fatal 诊断共享工具(仅供 src/ops/ 内部翻译单元 include,
// 不进 include/frame/,公开度最小)。OpRegistry::register_op(启动期
// fail-fast,ARCH-040)与 OpSchema::input/variadic_input(M9 起新增的 builder
// 期 fail-fast,变长输入两条硬约束校验,note A)共用同一份 fprintf+abort 实现
// (REUSE-002),禁止各自复制第二份。

#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace frame::ops {

// 注册/声明期违例一律 fatal(fail-fast,启动期或链式 builder 调用期):先向
// stderr 输出含算子名与违例原因的英文诊断,再终止进程。macros.h 的
// FRAME_CHECK 不带自定义消息且不得修改 macros.h,故调用方自行输出诊断行后
// 调用 std::abort()。inline 函数(C++17 起 header-only 多译文件包含无 ODR
// 冲突),无需拆分单独 .cpp 翻译单元。
[[noreturn]] inline void fatal_registration_error(std::string_view name, std::string_view reason) {
  std::fprintf(stderr, "frame::ops registration fatal: op '%.*s' rejected: %.*s\n",
               static_cast<int>(name.size()), name.data(), static_cast<int>(reason.size()),
               reason.data());
  std::abort();
}

}  // namespace frame::ops
