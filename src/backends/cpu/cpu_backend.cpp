// CPU 参考后端实现单元(必须真编译进 cpu-only 构建)。

#include "cpu_backend.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <utility>

#include <frame/ir/graph.h>
#include <frame/ops/kernel_registry.h>

#include "cpu_executable.h"

namespace frame::backends::cpu {

namespace {

// device.backend/index 校验:cpu 后端只服务单一逻辑设备 {kCpuBackendName, 0}。
// 违例返回带实际值的英文 kInvalidArgument(ARCH-031 口径:不静默降级)。
Status validate_cpu_device(Device device, std::string_view caller) {
  if (device.backend != kCpuBackendName) {
    return Status::make(ErrorCode::kInvalidArgument,
                        std::string(caller) + ": unsupported device backend '" +
                            std::string(device.backend) + "', expected '" +
                            std::string(kCpuBackendName) + "'");
  }
  if (device.index != 0) {
    return Status::make(ErrorCode::kInvalidArgument,
                        std::string(caller) + ": unsupported device index " +
                            std::to_string(device.index) + ", expected 0");
  }
  return Status::ok();
}

}  // namespace

Status CpuStream::synchronize() { return Status::ok(); }

Status CpuStream::record(hal::Event& /*event*/) {
  // 同步后端:record 时事件已天然完成(见 event.h 未 record 语义,本处等价)。
  return Status::ok();
}

Status CpuStream::wait(const hal::Event& /*event*/) { return Status::ok(); }

void* CpuStream::native_handle() { return nullptr; }

bool CpuEvent::query() const { return true; }

Status CpuEvent::synchronize() { return Status::ok(); }

std::string_view CpuBackend::name() const { return kCpuBackendName; }

Result<int32_t> CpuBackend::device_count() const { return int32_t{1}; }

Result<std::unique_ptr<hal::Stream>> CpuBackend::create_stream(Device device) {
  const Status validation = validate_cpu_device(device, "CpuBackend::create_stream");
  if (!validation.is_ok()) return validation;
  std::unique_ptr<hal::Stream> stream = std::make_unique<CpuStream>();
  return stream;
}

Result<std::unique_ptr<hal::Event>> CpuBackend::create_event(Device device) {
  const Status validation = validate_cpu_device(device, "CpuBackend::create_event");
  if (!validation.is_ok()) return validation;
  std::unique_ptr<hal::Event> event = std::make_unique<CpuEvent>();
  return event;
}

hal::Allocator* CpuBackend::allocator(Device device) {
  const Status validation = validate_cpu_device(device, "CpuBackend::allocator");
  if (!validation.is_ok()) {
    // allocator() 签名不带 Status/Result,违例经 stderr 输出英文诊断后返回
    // nullptr(该接口下"无可用分配器"的唯一表达方式)。
    std::fprintf(stderr, "%s\n", std::string(validation.message()).c_str());
    return nullptr;
  }
  // 进程级单例(函数内 static):CPU 分配器无状态,天然线程安全、生命周期贯穿进程。
  static CpuAllocator allocator_instance;
  return &allocator_instance;
}

Status CpuBackend::copy(void* dst, Device dst_device, const void* src, Device src_device,
                        size_t bytes, hal::Stream* /*stream*/) {
  FRAME_RETURN_IF_ERROR(validate_cpu_device(dst_device, "CpuBackend::copy"));
  FRAME_RETURN_IF_ERROR(validate_cpu_device(src_device, "CpuBackend::copy"));
  if (bytes == 0) return Status::ok();
  if (dst == nullptr || src == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "CpuBackend::copy: dst and src must be non-null when bytes > 0");
  }
  // memmove 容忍源/目的区间重叠(一致性套件固定此语义,见 backend-hal.md)。
  std::memmove(dst, src, bytes);
  return Status::ok();
}

Result<std::unique_ptr<hal::Executable>> CpuBackend::compile(
    const ir::Graph& graph, const hal::CompileOptions& /*options*/) {
  // options 本里程碑(M7)未使用:cpu Executable 逐 kernel 拼装不含后端内部
  // lowering/codegen 优化档位选择,M9 memory_planning 落地后再评估是否消费
  // opt_level(如切换朴素分配 vs 复用计划)。
  if (graph.topological_order().empty()) {
    return Status::make(ErrorCode::kInvalidArgument, "CpuBackend::compile: graph has no nodes");
  }

  // 取图 device:V6 保证全图所有 Value 的 device 一致
  // (docs/architecture/ir-design.md 第4章),任取拓扑序中第一个带输出的节点
  // (含 graph_input,其恰有 1 输出)即可代表整图,与
  // src/compiler/passes/backend_lowering.cpp 的取法一致。
  Device graph_device{};
  bool found_device = false;
  for (const ir::Node* node : graph.topological_order()) {
    if (!node->outputs().empty()) {
      graph_device = node->outputs()[0].type().device;
      found_device = true;
      break;
    }
  }
  if (!found_device) {
    return Status::make(ErrorCode::kInternal,
                        "CpuBackend::compile: graph has node(s) but no Value carries a device");
  }
  FRAME_RETURN_IF_ERROR(validate_cpu_device(graph_device, "CpuBackend::compile"));

  hal::Allocator* alloc = allocator(graph_device);
  if (alloc == nullptr) {
    return Status::make(ErrorCode::kInternal,
                        "CpuBackend::compile: allocator unavailable for cpu device");
  }

  Result<std::unique_ptr<CpuExecutable>> executable =
      CpuExecutable::compile(graph, *alloc, graph_device);
  if (!executable.is_ok()) return executable.status();
  return std::unique_ptr<hal::Executable>(std::move(executable.value()));
}

Status CpuBackend::launch(const hal::KernelInvocation& invocation, hal::Stream* stream) {
  FRAME_RETURN_IF_ERROR(validate_cpu_device(invocation.device, "CpuBackend::launch"));
  // 查 KernelRegistry:未注册返回含算子名与后端名的英文错误(ARCH-031,禁静默降级)。
  const Result<ops::KernelFn> kernel = ops::KernelRegistry::instance().find(invocation.op, name());
  if (!kernel.is_ok()) return kernel.status();
  ops::KernelContext context{invocation.inputs, invocation.outputs, invocation.attrs,
                             invocation.device, stream};
  return kernel.value()(context);
}

}  // namespace frame::backends::cpu

// CPU 后端静态注册:CPU 永远启用。FRAME_REGISTER_BACKEND 展开为静态初始化器,
// 使 BackendRegistry::get(kCpuBackendName) 可取到 CpuBackend 实例(聚合库须以
// WHOLE_ARCHIVE 链接本静态库,见 cmake/frame_backend.cmake)。
FRAME_REGISTER_BACKEND(frame::kCpuBackendName, frame::backends::cpu::CpuBackend);
