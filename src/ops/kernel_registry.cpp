// KernelRegistry 内核注册表的实现单元。
// 注册键 =(op 名字符串, backend 名字符串),不含 dtype —— dtype 差异在 kernel 内部
// 经 dispatch_dtype 编译期展开(见 include/frame/core/dtype.h)。
// FRAME_REGISTER_KERNEL 宏(见 include/frame/ops/kernel_registry.h)展开为内部
// 链接静态初始化器,对 noexcept 的 register_kernel_or_die 单次直接调用;该方法
// 在本文件内以普通函数体完成"调用 register_kernel + 失败 fatal",fprintf/abort
// 不再处于任何静态初始化式之中(与 FRAME_REGISTER_OP 同型,见 op_registry.cpp)。

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

#include <frame/ops/kernel_registry.h>

namespace frame::ops {

KernelRegistry& KernelRegistry::instance() noexcept {
  static KernelRegistry registry;
  return registry;
}

Status KernelRegistry::register_kernel(std::string_view op, std::string_view backend,
                                       KernelFn fn) noexcept {
  const std::pair<std::string, std::string> key{std::string(op), std::string(backend)};
  if (kernels_.find(key) != kernels_.end()) {
    return Status::make(ErrorCode::kAlreadyExists,
                        "KernelRegistry.register_kernel: duplicate kernel for (op='" +
                            std::string(op) + "', backend='" + std::string(backend) + "')");
  }
  kernels_.emplace(key, fn);
  return Status::ok();
}

bool KernelRegistry::register_kernel_or_die(std::string_view op, std::string_view backend,
                                            KernelFn fn) noexcept {
  const Status status = register_kernel(op, backend, fn);
  if (!status.is_ok()) {
    std::fprintf(stderr, "FRAME_REGISTER_KERNEL fatal: %.*s\n",
                 static_cast<int>(status.message().size()), status.message().data());
    std::abort();
  }
  return true;
}

Result<KernelFn> KernelRegistry::find(std::string_view op, std::string_view backend) const {
  const std::pair<std::string, std::string> key{std::string(op), std::string(backend)};
  const auto it = kernels_.find(key);
  if (it == kernels_.end()) {
    return Status::make(ErrorCode::kNotFound,
                        "KernelRegistry.find: no kernel registered for (op='" + std::string(op) +
                            "', backend='" + std::string(backend) + "')");
  }
  return it->second;
}

}  // namespace frame::ops
