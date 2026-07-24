#include <cstdint>
#include <iostream>

#include <frame/frame.h>

int main() {
  const frame::Result<frame::hal::Backend*> backend_result =
      frame::hal::BackendRegistry::instance().get(frame::kCpuBackendName);
  if (!backend_result.is_ok()) {
    std::cerr << "CPU backend lookup failed: " << backend_result.status().message() << '\n';
    return 1;
  }

  frame::hal::Backend* const backend = backend_result.value();
  const frame::Result<int32_t> device_count = backend->device_count();
  if (!device_count.is_ok()) {
    std::cerr << "CPU device count query failed: " << device_count.status().message() << '\n';
    return 1;
  }
  if (device_count.value() <= 0) {
    std::cerr << "CPU backend has no available devices" << '\n';
    return 1;
  }
  if (backend->allocator(frame::cpu_device()) == nullptr) {
    std::cerr << "CPU backend returned a null allocator" << '\n';
    return 1;
  }

  return 0;
}
