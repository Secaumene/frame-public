#pragma once
// FallbackStats:eager 回退链决策次数统计(docs/architecture/execution-model.md
// 第5章"统计"要求)。进程级 Meyer's 单例 + std::mutex(照 CompilationCache 模式,
// src/runtime/compile.cpp,粒度粗即可 v0)。
//
// 计数语义(m10-design-brief 决议点 A/C,design-reviewer REVISE 闭环采纳建议②):
// 记的是回退**决策**发生次数(FallbackExecutable::build 期一次性),不是每次
// run() 的执行次数——编译产物入编译缓存后命中路径不重跑 build(),故不重复计数。

#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

#include <frame/core/macros.h>

namespace frame::runtime {

// 全局回退统计表:键 = 算子名 + 目标后端名(与编译缓存键同款 '\x1f' 分隔纪律)。
class FRAME_API FallbackStats {
 public:
  static FallbackStats& instance();

  // 查询累计计数;未记录过返回 0。
  int64_t count(std::string_view op, std::string_view backend) const;

  // 记一次回退决策。供 FallbackExecutable::build 内部调用,不面向外部直接调用
  // (公开是实现取舍——记录发生在 src/runtime/fallback_executable.cpp 另一翻译
  // 单元,无需 friend 声明的额外复杂度)。
  void record(std::string_view op, std::string_view backend);

  // 清空全部计数。仅供测试隔离使用,非产品语义——不承诺跨进程/跨用例的累计口径。
  void reset();

 private:
  FallbackStats() = default;

  mutable std::mutex mutex_;
  std::unordered_map<std::string, int64_t> counts_;
};

}  // namespace frame::runtime
