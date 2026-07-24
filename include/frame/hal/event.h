#pragma once
// Event:后端同步事件(HAL 白名单虚函数,判定规则见 include/frame/hal/backend.h 头部)。
// 用于在流之间建立依赖与查询异步工作的完成状态。

#include <frame/core/macros.h>
#include <frame/core/status.h>

namespace frame::hal {

// Event:同步事件抽象。创建入口是 Backend::create_event(Device)(设备域生命周期,
// 见 include/frame/hal/backend.h),而非 Stream——同一 event 可在流 A record、
// 流 B wait,创建入口绑定 Stream 会误示流亲和。
// 未 record 的 Event 语义定案:query() 恒为 true、synchronize() 恒返回 Ok
// ——「视为已完成」,与 CUDA cudaEventCreate 后未 record 即成功的语义一致
// (backend-hal.md 2.3,HAL 一致性套件据此断言)。
class FRAME_API Event {
 public:
  virtual ~Event() = default;

  // 非阻塞查询:事件是否已完成。
  virtual bool query() const = 0;
  // 阻塞直至事件完成。
  virtual Status synchronize() = 0;
};

}  // namespace frame::hal
