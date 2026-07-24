#pragma once
// OpRegistry:算子 schema 注册表(全局唯一)+ 自定义算子第 1 步入口。
// 见 docs/architecture/operator-system.md 第3章。

#include <string>
#include <string_view>
#include <unordered_map>

#include <frame/core/macros.h>
#include <frame/core/status.h>
#include <frame/ir/graph.h>
#include <frame/ops/op_schema.h>

namespace frame::ops {

// OpRegistry:name -> OpSchema 的全局注册表。
class FRAME_API OpRegistry {
 public:
  // instance()/register_op 均标 noexcept:项目禁用异常(CPP-020),二者位于
  // FRAME_REGISTER_OP 静态初始化链路上;标准容器扩容失败(bad_alloc)经
  // noexcept 转为 std::terminate 即 fail-fast,如实反映启动期注册"不可恢复"
  // 的既定契约(register_op 校验失败同样 fatal,见下)。
  static OpRegistry& instance() noexcept;

  // 注册算子并返回其 OpSchema& 供链式声明。
  // 【ARCH-040】算子名须匹配 ^[a-z][a-z0-9_]*$、全局唯一、且非 ir 层保留名
  // (frame::ir::kGraphInputOp / frame::ir::kGraphOutputMarker);任一违例在
  // 启动期 fatal(fail-fast——注册发生在静态初始化期,以 Status 返回也无人能
  // 处理),fatal 前向 stderr 输出含算子名与违例原因的英文诊断,使死亡测试可
  // 区分「非法名」/「重名」/「保留名」三类。
  OpSchema& register_op(std::string_view name) noexcept;

  // 查找算子 schema;不存在返回 nullptr。
  const OpSchema* find(std::string_view name) const;

 private:
  // Meyer's singleton(instance())内的算子表;FRAME_REGISTER_OP 宏生成的初始化
  // 器只经 instance() 访问该单例,不直接触碰 schemas_,无跨 TU 静态初始化顺序
  // 问题。
  std::unordered_map<std::string, OpSchema> schemas_;
};

// 构造接线 OpRegistry 的 ir::OpQuery(Graph::verify V3/V4 用):
// op_registered → find(name) != nullptr;
// check_schema → 满足 schema 输入约束(定长恒等,或 schema 声明变长输入组时
//   改判"至少 min_input_count() 个",M9 前置设计:OpSchema 变长输入支持)、
//   输出数量一致、必需属性存在、属性类型匹配(ir::attr_type_of)、未知属性拒绝
//   (枚举 node.attrs())。错误消息英文。
// 前缀纪律:check_schema 返回的消息不带 "V4: " 前缀——Graph::verify(见
// src/ir/graph.cpp)统一在外层加前缀后原样透传,这里若也加前缀会造成双重前缀,
// 破坏 golden 文本对齐。
ir::OpQuery make_op_query();

}  // namespace frame::ops

// ---------------------------------------------------------------------------
// FRAME_REGISTER_OP(name):自定义算子第 1 步 —— 声明 schema(第 2 步是注册 kernel,
// 见 include/frame/ops/kernel_registry.h)。展开为一条内部链接(static)的初始化
// 语句;初始化表达式即 register_op(name) 的返回值(OpSchema&),调用方在同一条
// 语句内继续链式追加 .input(...).trait(...) 等 builder 调用,最终分号由调用方
// 书写(宏本身不带尾随分号)。记号拼接用 include/frame/core/macros.h 的
// FRAME_CONCAT(全部注册宏共用同一份两层拼接宏,REUSE-002)保证 __COUNTER__
// 先展开为具体数值。
// ---------------------------------------------------------------------------
// NOLINTBEGIN(bugprone-macro-parentheses) —— 本宏展开为一条声明语句(而非表达式),
// 不可整体加括号(那会破坏 `static 引用 = 初始化式` 声明语法);已知误报,理由见
// 上方注释(需要调用方在同一条语句内继续链式追加 builder 方法)。
#define FRAME_REGISTER_OP(name)                                                                 \
  [[maybe_unused]] static ::frame::ops::OpSchema& FRAME_CONCAT(frame_op_schema_, __COUNTER__) = \
      ::frame::ops::OpRegistry::instance().register_op(name)
// NOLINTEND(bugprone-macro-parentheses)
