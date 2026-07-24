// FallbackStats 实现单元(声明见 include/frame/runtime/fallback_stats.h)。

#include <string>
#include <utility>

#include <frame/runtime/fallback_stats.h>

namespace frame::runtime {

namespace {

// 键拼接:与编译缓存键(src/runtime/compile.cpp::make_cache_key)同款
// '\x1f'(ASCII Unit Separator)分隔纪律,规避算子名/后端名内部可能出现的
// 常见分隔符。
std::string make_stats_key(std::string_view op, std::string_view backend) {
  std::string key;
  key.reserve(op.size() + backend.size() + 1);
  key += op;
  key += '\x1f';
  key += backend;
  return key;
}

}  // namespace

FallbackStats& FallbackStats::instance() {
  static FallbackStats stats;
  return stats;
}

int64_t FallbackStats::count(std::string_view op, std::string_view backend) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it = counts_.find(make_stats_key(op, backend));
  if (it == counts_.end()) return 0;
  return it->second;
}

void FallbackStats::record(std::string_view op, std::string_view backend) {
  const std::lock_guard<std::mutex> lock(mutex_);
  ++counts_[make_stats_key(op, backend)];
}

void FallbackStats::reset() {
  const std::lock_guard<std::mutex> lock(mutex_);
  counts_.clear();
}

}  // namespace frame::runtime
