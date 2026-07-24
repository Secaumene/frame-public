// BackendRegistry 全局后端注册表的实现单元。
// FRAME_REGISTER_BACKEND 宏(见 include/frame/hal/backend.h)展开为匿名命名空间
// 静态初始化器,经模板薄壳调用本文件提供的非模板 register_backend_or_die。

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

#include <frame/hal/backend.h>

namespace frame::hal {

BackendRegistry& BackendRegistry::instance() noexcept {
  static BackendRegistry registry;
  return registry;
}

Status BackendRegistry::register_backend(std::string_view name,
                                         std::unique_ptr<Backend> backend) noexcept {
  if (name.empty()) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "BackendRegistry.register_backend: backend name must not be empty");
  }
  if (backend == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "BackendRegistry.register_backend: backend must not be null");
  }
  const std::string key(name);
  if (backends_.find(key) != backends_.end()) {
    return Status::make(ErrorCode::kAlreadyExists,
                        "BackendRegistry.register_backend: duplicate backend name '" + key + "'");
  }
  backends_.emplace(key, std::move(backend));
  return Status::ok();
}

bool BackendRegistry::register_backend_or_die(std::string_view name,
                                              std::unique_ptr<Backend> backend) noexcept {
  const Status status = register_backend(name, std::move(backend));
  if (!status.is_ok()) {
    std::fprintf(stderr, "FRAME_REGISTER_BACKEND fatal: %.*s\n",
                 static_cast<int>(status.message().size()), status.message().data());
    std::abort();
  }
  return true;
}

Result<Backend*> BackendRegistry::get(std::string_view name) {
  const auto it = backends_.find(std::string(name));
  if (it == backends_.end()) {
    return Status::make(
        ErrorCode::kNotFound,
        "BackendRegistry.get: no backend registered for name '" + std::string(name) + "'");
  }
  return it->second.get();
}

std::vector<std::string_view> BackendRegistry::available() const {
  std::vector<std::string_view> names;
  names.reserve(backends_.size());
  for (const auto& [key, backend] : backends_) {
    (void)backend;
    names.push_back(key);
  }
  std::sort(names.begin(), names.end());
  return names;
}

}  // namespace frame::hal
