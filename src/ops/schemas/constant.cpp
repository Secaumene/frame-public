// 内置 constant 算子 schema 注册桩(M8):常量进入 IR 的唯一形态——0 输入
// 1 输出,数据以三个属性编码:value(kDoubleArray,行优先展平)/
// shape(kShape)/dtype(kDType)。不扩展 AttrType(ARCH-020),v0 dtype 白名单
// float32/float16/bfloat16/int32/int64(M22 批4 决议点A 扩容,
// docs/plan/2026-07-19-batch4-m22-seq.md §1.1)。float32/float16/bfloat16
// 均可被 double 精确表示(53 位尾数 ⊇ fp32 24 位尾数;fp16/bf16 是 fp32 值域
// 子集),编码零精度损失(与 include/frame/ops/constant_utils.h 精度论证
// 一致);int32/int64 逐元素校验整值(v==trunc(v))、落入目标 dtype 值域、且
// |v| <= 2^53(kMaxDoubleExactInteger,double 尾数精确整数界)——诚实校验而非
// 默扩,零 splat 天然通过。无副作用、非 elementwise、不可交换,不标任何
// trait。cpu kernel 见 src/backends/cpu/kernels/constant.cpp;编译期折叠
// 产物同样落地为本算子(src/compiler/passes/constant_folding.cpp)。

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/ir/node.h>
#include <frame/ops/constant_utils.h>
#include <frame/ops/op_registry.h>

namespace {

// 整数 dtype(int32/int64)的逐元素校验:v==trunc(v)(整值)、|v|<=2^53
// (kMaxDoubleExactInteger,double 尾数精确整数界)、落入目标 dtype 值域
// (仅 int32 有效——int64 的原生值域远大于 2^53,前一条已是更紧的约束)。
// index 是该值在 value 数组中的位置(供错误消息定位)。
frame::Status validate_integer_constant_value(frame::DType dtype, double v, size_t index) {
  if (v != std::trunc(v)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'constant' attribute 'value' entry " + std::to_string(index) +
                                   " = " + std::to_string(v) +
                                   " is not an integral value for dtype '" +
                                   std::string(dtype.name()) + "'");
  }
  const double abs_v = std::fabs(v);
  if (abs_v > static_cast<double>(frame::ops::kMaxDoubleExactInteger)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'constant' attribute 'value' entry " + std::to_string(index) +
                                   " = " + std::to_string(v) +
                                   " exceeds the double-exact integer bound 2^53=" +
                                   std::to_string(frame::ops::kMaxDoubleExactInteger));
  }
  if (dtype.code() == frame::DTypeCode::kInt32) {
    constexpr double kInt32Min = static_cast<double>(std::numeric_limits<std::int32_t>::min());
    constexpr double kInt32Max = static_cast<double>(std::numeric_limits<std::int32_t>::max());
    if (v < kInt32Min || v > kInt32Max) {
      return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                                 "op 'constant' attribute 'value' entry " + std::to_string(index) +
                                     " = " + std::to_string(v) +
                                     " is out of range for dtype 'int32'");
    }
  }
  return frame::Status::ok();
}

// constant 的 shape 推断:0 输入;value/shape/dtype 三属性必须存在且类型
// 正确、value 元素数与 shape.numel() 一致、shape 无动态维(ARCH-013)、dtype
// 属 v0 白名单;整数 dtype 额外逐元素校验(validate_integer_constant_value);
// 任一违例返回英文错误(消息含实际值,ARCH-031 口径:不静默降级)。输出
// shape 即 shape 属性本身(dtype 一致性由 attrs 的 dtype 属性保证,
// shape_inference pass 的③dtype 复核对 0 输入节点天然豁免,见
// src/compiler/passes/shape_inference.cpp)。
frame::Result<std::vector<frame::Shape>> infer_constant_shape(const frame::ops::NodeContext& ctx) {
  if (!ctx.input_types.empty()) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'constant' expects 0 inputs, got " + std::to_string(ctx.input_types.size()));
  }

  const std::vector<double>* value = ctx.attr<std::vector<double>>("value");
  if (value == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'constant' is missing required attribute 'value' (double array)");
  }
  const frame::Shape* shape = ctx.attr<frame::Shape>("shape");
  if (shape == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'constant' is missing required attribute 'shape'");
  }
  const frame::DType* dtype = ctx.attr<frame::DType>("dtype");
  if (dtype == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'constant' is missing required attribute 'dtype'");
  }

  if (shape->has_dynamic_dim()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'constant' attribute 'shape' " + shape->to_string() +
                                   " has a dynamic dimension, static shape required (ARCH-013)");
  }

  const int64_t expected_numel = shape->numel();
  if (static_cast<int64_t>(value->size()) != expected_numel) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'constant' attribute 'value' has " +
                                   std::to_string(value->size()) +
                                   " element(s), attribute 'shape' " + shape->to_string() +
                                   " expects " + std::to_string(expected_numel));
  }

  if (!frame::ops::is_constant_dtype_supported(*dtype)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'constant' does not support dtype '" +
                                   std::string(dtype->name()) +
                                   "' (v0 supports float32/float16/bfloat16/int32/int64 only)");
  }

  if (dtype->code() == frame::DTypeCode::kInt32 || dtype->code() == frame::DTypeCode::kInt64) {
    for (size_t i = 0; i < value->size(); ++i) {
      const frame::Status entry_status = validate_integer_constant_value(*dtype, (*value)[i], i);
      if (!entry_status.is_ok()) return entry_status;
    }
  }

  return std::vector<frame::Shape>{*shape};
}

}  // namespace

FRAME_REGISTER_OP(frame::ops::kConstantOpName)
    .attr("value", frame::ir::AttrType::kDoubleArray, /*required=*/true)
    .attr("shape", frame::ir::AttrType::kShape, /*required=*/true)
    .attr("dtype", frame::ir::AttrType::kDType, /*required=*/true)
    .output("out", "materialized constant tensor")
    .shape_infer(&infer_constant_shape);
