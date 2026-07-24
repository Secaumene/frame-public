// PassRegistry 全局 pass 注册表的实现单元。
// FRAME_REGISTER_PASS 宏(见 include/frame/compiler/pass.h)展开为内部链接
// (static)的静态初始化器,对 noexcept 的 register_pass_or_die<PassClass>()
// 单次直接调用;fprintf/abort 的诊断逻辑收敛在本文件的普通函数体内,不再处于
// 任何静态初始化式之中(与 FRAME_REGISTER_KERNEL 同型,见
// src/ops/kernel_registry.cpp)。

#include <cstdio>
#include <cstdlib>
#include <string>

#include <frame/compiler/pass.h>

namespace frame::compiler {

PassRegistry& PassRegistry::instance() noexcept {
  static PassRegistry registry;
  return registry;
}

Status PassRegistry::register_pass(std::string_view name, Factory factory) noexcept {
  const std::string key(name);
  if (passes_.find(key) != passes_.end()) {
    return Status::make(ErrorCode::kAlreadyExists,
                        "PassRegistry.register_pass: duplicate pass '" + key + "'");
  }
  passes_.emplace(key, factory);
  return Status::ok();
}

bool PassRegistry::register_pass_or_die(std::string_view name, Factory factory) noexcept {
  const Status status = register_pass(name, factory);
  if (!status.is_ok()) {
    std::fprintf(stderr, "FRAME_REGISTER_PASS fatal: %.*s\n",
                 static_cast<int>(status.message().size()), status.message().data());
    std::abort();
  }
  return true;
}

Result<std::unique_ptr<Pass>> PassRegistry::create(std::string_view name) const {
  const auto it = passes_.find(std::string(name));
  if (it == passes_.end()) {
    return Status::make(
        ErrorCode::kNotFound,
        "PassRegistry.create: no pass registered with name '" + std::string(name) + "'");
  }
  return it->second();
}

}  // namespace frame::compiler
