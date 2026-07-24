#pragma once
// 编译器 pass 扩展接口(铁律 #5:为自定义 pass 提供可扩展接口)。
// Pass 属虚函数白名单六类之一(R1 整图/pass 粒度 / R2 插件 pass / R3 白名单,
// 判定规则全文见 include/frame/hal/backend.h 头部)。子类经 CRTP 基类接入:
// 名字与工厂在编译期生成,注册合法性由 concept 编译期校验。

#include <concepts>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>

#include <frame/core/macros.h>
#include <frame/core/status.h>

namespace frame::ir {
class Graph;  // 前向声明:pass 就地变换的对象
}

namespace frame::compiler {

// Pass:图变换接口。
class FRAME_API Pass {
 public:
  virtual ~Pass() = default;
  virtual std::string_view name() const = 0;
  // 就地变换;失败返回 Status,禁止留下半变换状态。
  virtual Status run(ir::Graph& graph) = 0;
};

// CRTP 基类:自定义 pass 只需提供 static constexpr kName 与 run_impl(ir::Graph&)。
// 构造函数私有 + friend Derived:防止以错误的 Derived 实参实例化基类
// (bugprone-crtp-constructor-accessibility)。
template <typename Derived>
class PassBase : public Pass {
 public:
  std::string_view name() const final { return Derived::kName; }
  Status run(ir::Graph& graph) final { return static_cast<Derived*>(this)->run_impl(graph); }

 private:
  PassBase() = default;
  friend Derived;
};

// concept:注册时不满足直接编译错误(而非运行时崩溃)。
template <typename P>
concept PassType = std::derived_from<P, Pass> && std::is_default_constructible_v<P> && requires {
  { P::kName } -> std::convertible_to<std::string_view>;
};

// 全局 pass 注册表:工厂用函数指针而非 std::function(零开销)。
// instance()/register_pass/register_pass_or_die 均标 noexcept:项目禁用异常
// (CPP-020),三者均位于 FRAME_REGISTER_PASS 静态初始化链路上(与 KernelRegistry
// 同型,见 include/frame/ops/kernel_registry.h)。
class FRAME_API PassRegistry {
 public:
  static PassRegistry& instance() noexcept;

  using Factory = std::unique_ptr<Pass> (*)();

  // 以 name 为键登记工厂函数指针;重复注册返回错误(英文消息,含名)。
  Status register_pass(std::string_view name, Factory factory) noexcept;

  template <PassType P>
  Status register_pass() {
    return register_pass(P::kName, []() -> std::unique_ptr<Pass> { return std::make_unique<P>(); });
  }

  // 启动期注册宏专用的 fail-fast 包装(CPP-020):调用 register_pass,失败时向
  // stderr 输出可区分的英文诊断后 std::abort(),成功返回 true(与
  // KernelRegistry::register_kernel_or_die 同型,见 src/ops/kernel_registry.cpp)。
  bool register_pass_or_die(std::string_view name, Factory factory) noexcept;

  // 模板薄壳:以 P::kName 与默认构造工厂调用非模板 register_pass_or_die;
  // FRAME_REGISTER_PASS 宏直接调用本方法。
  template <PassType P>
  bool register_pass_or_die() noexcept {
    return register_pass_or_die(P::kName,
                                []() -> std::unique_ptr<Pass> { return std::make_unique<P>(); });
  }

  // 按名查找并经工厂构造;不存在返回 kNotFound(消息含名)。
  Result<std::unique_ptr<Pass>> create(std::string_view name) const;

 private:
  // Meyer's singleton(instance())内的 pass 表;FRAME_REGISTER_PASS 宏生成的
  // 初始化器只经 instance() 访问该单例,不直接触碰 passes_,无跨 TU 静态初始化
  // 顺序问题(同 OpRegistry/KernelRegistry)。
  std::unordered_map<std::string, Factory> passes_;
};

}  // namespace frame::compiler

// ---------------------------------------------------------------------------
// FRAME_REGISTER_PASS(PassClass):自定义 pass 三步之末的一行注册宏(v0 单参数)。
// v0 语义:注册使 pass 可按名查找,但**不自动进入标准管线**,须由调用方经
// PassManager 显式插入(见 docs/architecture/compiler-passes.md 第4章)。
// 展开为内部链接(static)的静态初始化器:对 noexcept 函数
// register_pass_or_die<PassClass>() 的单次直接调用(与 FRAME_REGISTER_KERNEL
// 同型,见 include/frame/ops/kernel_registry.h:96-99)——fprintf/abort 的诊断
// 逻辑收敛在 src/compiler/pass_registry.cpp 的普通函数体内。记号拼接用
// include/frame/core/macros.h 的 FRAME_CONCAT(全部注册宏共用同一份两层拼接宏,
// REUSE-002)保证 __COUNTER__ 先展开为具体数值。
// ---------------------------------------------------------------------------
#define FRAME_REGISTER_PASS(PassClass)                                            \
  [[maybe_unused]] static const bool FRAME_CONCAT(frame_pass_reg_, __COUNTER__) = \
      ::frame::compiler::PassRegistry::instance().register_pass_or_die<PassClass>()
