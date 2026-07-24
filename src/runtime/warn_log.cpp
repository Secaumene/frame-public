// warn_log 实现单元(声明见同目录 warn_log.h)。

#include "warn_log.h"

#include <cstdio>

namespace frame::runtime {

void warn_log(std::string_view message) {
  std::fprintf(stderr, "[frame][WARN] %.*s\n", static_cast<int>(message.size()), message.data());
}

}  // namespace frame::runtime
