#pragma once
// 错误模型:Status / Result<T>。
// 纪律:核心库禁用异常,HAL 边界与算子实现一律以 Status/Result 返回(CPP-020);
// throw 仅允许出现在 python/ 绑定层。所有 Status/错误消息一律英文(LANG-005)。

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <frame/core/macros.h>

namespace frame {

// 错误码封闭枚举。新增错误码追加在末尾,不复用既有数值;
// 底型 uint8_t 足以承载封闭取值集(performance-enum-size)。
enum class ErrorCode : uint8_t {
  kOk = 0,
  kInvalidArgument,
  kNotFound,
  kAlreadyExists,
  kUnimplemented,
  kOutOfMemory,
  kInternal,
  kDeviceError,
};

// Status:成功或带英文消息的失败。值语义,无虚函数。
class FRAME_API Status {
 public:
  // 默认构造为成功。
  Status() = default;

  // 成功状态。
  static Status ok() { return Status(); }

  // 失败状态:错误码 + 英文消息(LANG-005)。
  static Status make(ErrorCode code, std::string_view message) {
    Status status;
    status.code_ = code;
    status.message_ = std::string(message);
    return status;
  }

  bool is_ok() const { return code_ == ErrorCode::kOk; }
  ErrorCode code() const { return code_; }
  std::string_view message() const { return message_; }

 private:
  ErrorCode code_ = ErrorCode::kOk;
  std::string message_;
};

// Result<T>:承载值(成功)或 Status(失败)。禁止异常跨边界的返回载体。
template <typename T>
class Result {
 public:
  // 由值构造:视为成功。
  Result(T value) : status_(Status::ok()), value_(std::move(value)) {}

  // 由 Status 构造:通常为失败状态(成功但无值属调用方误用)。
  Result(Status status) : status_(std::move(status)) {}

  bool is_ok() const { return status_.is_ok() && value_.has_value(); }
  const Status& status() const { return status_; }

  // 前置条件:is_ok() 为真;违反前置条件立即 fatal(不可恢复)。
  T& value() {
    FRAME_CHECK(value_.has_value());
    return *value_;
  }
  const T& value() const {
    FRAME_CHECK(value_.has_value());
    return *value_;
  }

 private:
  Status status_;
  std::optional<T> value_;
};

}  // namespace frame

// ---------------------------------------------------------------------------
// FRAME_RETURN_IF_ERROR(expr):求值 expr(须为 Status),失败即提前返回该 Status。
// ---------------------------------------------------------------------------
#define FRAME_RETURN_IF_ERROR(expr)                   \
  do {                                                \
    ::frame::Status _frame_status = (expr);           \
    if (!_frame_status.is_ok()) return _frame_status; \
  } while (0)

// ---------------------------------------------------------------------------
// FRAME_UNIMPLEMENTED():返回 kUnimplemented + 当前函数名(英文)。
// 骨架期需返回 Status 的桩函数统一用它。
// ---------------------------------------------------------------------------
#define FRAME_UNIMPLEMENTED() (::frame::Status::make(::frame::ErrorCode::kUnimplemented, __func__))
