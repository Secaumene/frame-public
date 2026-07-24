#pragma once
// Stream:后端执行流(HAL 白名单虚函数,判定规则见 include/frame/hal/backend.h 头部)。
// 异步操作在 Stream 上排队,经 Event 建立跨流依赖。

#include <frame/core/macros.h>
#include <frame/core/status.h>

namespace frame::hal {

class Event;  // 前向声明:record/wait 的同步对象

// Stream:执行流抽象。
class FRAME_API Stream {
 public:
  virtual ~Stream() = default;

  // 阻塞直至流上全部工作完成。
  virtual Status synchronize() = 0;
  // 在本流当前位置记录事件。
  virtual Status record(Event& event) = 0;
  // 令本流等待事件完成(跨流依赖)。
  virtual Status wait(const Event& event) = 0;

  // 后端原生流句柄(如 cudaStream_t)。
  // 【ARCH-030】仅供 src/backends/ 与测试下沉使用;core/ops/ir/compiler/runtime
  // 与 python 禁止调用。机械校验:scripts/check_iron_rules.sh 对 native_handle
  // 出现位置做白名单比对。
  virtual void* native_handle() = 0;
};

}  // namespace frame::hal
