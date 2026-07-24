// CPU 损失算子内核(M17):mse_loss + mse_loss_grad_internal。
// 每个内核形如 Status kernel(ops::KernelContext&),内部经 dispatch_dtype 按 dtype
// 编译期展开(见 include/frame/core/dtype.h),再经 FRAME_REGISTER_KERNEL 注册到
// (op, kCpuBackendName)。

#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/ops/kernel_registry.h>

namespace {

// mse_loss/mse_loss_grad_internal 共用:校验 pred/target 的 dtype/shape 完全
// 一致(REUSE-002:二者共用本函数,避免同构校验各自复制)。
frame::Status require_matching_pred_target(std::string_view op_name, const frame::Tensor& pred,
                                           const frame::Tensor& target) {
  const frame::DType pred_type = pred.dtype();
  const frame::DType target_type = target.dtype();
  // 布尔先落地为具名变量再判断(与 src/backends/cpu/kernels/elementwise.cpp
  // 同一惯例):check_iron_rules.sh 对 kernels/ 目录的 CPP-012 文本扫描按
  // `if (...dtype...)` 形态识别疑似运行时 dtype 分支,提前拆出具名变量可与
  // dispatch_dtype 编译期分派清晰区分,避免误判。
  const bool elem_type_mismatch = !(pred_type == target_type);
  if (elem_type_mismatch) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) +
            "' cpu kernel requires pred and target of the same dtype, got '" +
            std::string(pred_type.name()) + "', '" + std::string(target_type.name()) + "'");
  }
  if (!(pred.shape() == target.shape())) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cpu kernel requires pred and target of the same shape, got " +
                                   pred.shape().to_string() + ", " + target.shape().to_string());
  }
  return frame::Status::ok();
}

// mse_loss/mse_loss_grad_internal 共用:校验 dtype 属 v0 浮点三档,返回其
// DTypeCode(供 dispatch_dtype 使用)。REUSE-002:二者共用本函数。
frame::Result<frame::DTypeCode> require_supported_float_dtype(std::string_view op_name,
                                                              frame::DType dtype) {
  const frame::DTypeCode code = dtype.code();
  const bool supported = (code == frame::DTypeCode::kFloat32) ||
                         (code == frame::DTypeCode::kFloat16) ||
                         (code == frame::DTypeCode::kBFloat16);
  if (!supported) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) + "' cpu kernel does not support dtype '" +
            std::string(dtype.name()) + "' (v0 supports float32/float16/bfloat16 only)");
  }
  return code;
}

// mse_loss(pred,target)=mean((pred-target)^2) 的 CPU 参考实现(REUSE-011:
// 参考实现,数值校验用,禁作性能路径)——朴素逐元素累加平方差、除以元素总数,
// 累加以 float 精度进行(fp16/bf16 逐元素升 float 累加,遍历结束后一次性
// 转回,与 src/backends/cpu/kernels/reduction.cpp 的 sum 累加同一约定)。
frame::Status mse_loss_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'mse_loss' cpu kernel expects 2 inputs, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'mse_loss' cpu kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& pred = ctx.inputs[0];
  const frame::Tensor& target = ctx.inputs[1];
  frame::Tensor& out = ctx.outputs[0];

  frame::Status match_status = require_matching_pred_target("mse_loss", pred, target);
  if (!match_status.is_ok()) return match_status;

  const frame::DType pred_type = pred.dtype();
  const frame::DType out_type = out.dtype();
  const bool out_elem_type_mismatch = !(out_type == pred_type);
  if (out_elem_type_mismatch) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'mse_loss' cpu kernel requires out dtype to match pred/target dtype, got out='" +
            std::string(out_type.name()) + "' pred='" + std::string(pred_type.name()) + "'");
  }
  const frame::Shape kScalarShape;
  const bool out_shape_mismatch = !(out.shape() == kScalarShape);
  if (out_shape_mismatch) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'mse_loss' cpu kernel requires out shape to be rank-0 (scalar), "
                               "got " +
                                   out.shape().to_string());
  }

  const frame::Result<frame::DTypeCode> code_result =
      require_supported_float_dtype("mse_loss", pred_type);
  if (!code_result.is_ok()) return code_result.status();
  const frame::DTypeCode code = code_result.value();

  const int64_t numel = pred.numel();
  if (numel == 0) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'mse_loss' cpu kernel requires pred/target to have at least 1 "
                               "element, got shape " +
                                   pred.shape().to_string() + " (numel 0)");
  }

  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    const T* pred_data = static_cast<const T*>(pred.raw_data());
    const T* target_data = static_cast<const T*>(target.raw_data());
    T* out_data = out.data<T>();

    float accum = 0.0F;
    for (int64_t i = 0; i < numel; ++i) {
      float p = 0.0F;
      float t = 0.0F;
      if constexpr (std::is_same_v<T, float>) {
        p = pred_data[i];
        t = target_data[i];
      } else if constexpr (std::is_same_v<T, frame::float16_t>) {
        p = frame::float16_to_float(pred_data[i]);
        t = frame::float16_to_float(target_data[i]);
      } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
        p = frame::bfloat16_to_float(pred_data[i]);
        t = frame::bfloat16_to_float(target_data[i]);
      }
      const float diff = p - t;
      accum += diff * diff;
    }
    const float mean = accum / static_cast<float>(numel);
    if constexpr (std::is_same_v<T, float>) {
      out_data[0] = mean;
    } else if constexpr (std::is_same_v<T, frame::float16_t>) {
      out_data[0] = frame::float_to_float16(mean);
    } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
      out_data[0] = frame::float_to_bfloat16(mean);
    }
    return frame::Status::ok();
  });
}

// mse_loss_grad_internal(pred,target,gy) = 2*(pred-target)/N*gy 的 CPU 参考
// 实现(M17,REUSE-011)。gy 是标量(numel==1),kernel 内直接读取其首元素
// (升 float)乘进系数,不引广播语义(与 add/mul 等逐元素算子的"同 shape 才能
// 相乘"约束不同,gy 标量在此处按数学标量常数处理)。
frame::Status mse_loss_grad_internal_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 3) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'mse_loss_grad_internal' cpu kernel expects 3 inputs, got " +
                                   std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'mse_loss_grad_internal' cpu kernel expects 1 output, got " +
                                   std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& pred = ctx.inputs[0];
  const frame::Tensor& target = ctx.inputs[1];
  const frame::Tensor& gy = ctx.inputs[2];
  frame::Tensor& gpred = ctx.outputs[0];

  frame::Status match_status = require_matching_pred_target("mse_loss_grad_internal", pred, target);
  if (!match_status.is_ok()) return match_status;

  const frame::DType pred_type = pred.dtype();
  const frame::DType gy_type = gy.dtype();
  const bool gy_elem_type_mismatch = !(gy_type == pred_type);
  if (gy_elem_type_mismatch) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'mse_loss_grad_internal' cpu kernel requires gy dtype to match pred/target dtype, got "
        "gy='" +
            std::string(gy_type.name()) + "' pred='" + std::string(pred_type.name()) + "'");
  }
  const bool gy_not_scalar = gy.shape().rank() != 0 && gy.shape().numel() != 1;
  if (gy_not_scalar) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'mse_loss_grad_internal' cpu kernel requires gy to be scalar (rank 0 or numel==1), "
        "got shape " +
            gy.shape().to_string());
  }
  const frame::DType gpred_type = gpred.dtype();
  const bool gpred_elem_type_mismatch = !(gpred_type == pred_type);
  if (gpred_elem_type_mismatch) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'mse_loss_grad_internal' cpu kernel requires gpred(out) dtype to match pred dtype, "
        "got gpred='" +
            std::string(gpred_type.name()) + "' pred='" + std::string(pred_type.name()) + "'");
  }
  const bool gpred_shape_mismatch = !(gpred.shape() == pred.shape());
  if (gpred_shape_mismatch) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'mse_loss_grad_internal' cpu kernel requires gpred(out) shape to match pred shape, "
        "got " +
            gpred.shape().to_string() + ", expected " + pred.shape().to_string());
  }

  const frame::Result<frame::DTypeCode> code_result =
      require_supported_float_dtype("mse_loss_grad_internal", pred_type);
  if (!code_result.is_ok()) return code_result.status();
  const frame::DTypeCode code = code_result.value();

  const int64_t numel = pred.numel();
  if (numel == 0) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'mse_loss_grad_internal' cpu kernel requires pred/target to have at least 1 element, "
        "got shape " +
            pred.shape().to_string() + " (numel 0)");
  }

  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    const T* pred_data = static_cast<const T*>(pred.raw_data());
    const T* target_data = static_cast<const T*>(target.raw_data());
    const T* gy_data = static_cast<const T*>(gy.raw_data());
    T* gpred_data = gpred.data<T>();

    float gy_scalar = 0.0F;
    if constexpr (std::is_same_v<T, float>) {
      gy_scalar = gy_data[0];
    } else if constexpr (std::is_same_v<T, frame::float16_t>) {
      gy_scalar = frame::float16_to_float(gy_data[0]);
    } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
      gy_scalar = frame::bfloat16_to_float(gy_data[0]);
    }
    const float coefficient = 2.0F * gy_scalar / static_cast<float>(numel);

    for (int64_t i = 0; i < numel; ++i) {
      float p = 0.0F;
      float t = 0.0F;
      if constexpr (std::is_same_v<T, float>) {
        p = pred_data[i];
        t = target_data[i];
      } else if constexpr (std::is_same_v<T, frame::float16_t>) {
        p = frame::float16_to_float(pred_data[i]);
        t = frame::float16_to_float(target_data[i]);
      } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
        p = frame::bfloat16_to_float(pred_data[i]);
        t = frame::bfloat16_to_float(target_data[i]);
      }
      const float value = coefficient * (p - t);
      if constexpr (std::is_same_v<T, float>) {
        gpred_data[i] = value;
      } else if constexpr (std::is_same_v<T, frame::float16_t>) {
        gpred_data[i] = frame::float_to_float16(value);
      } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
        gpred_data[i] = frame::float_to_bfloat16(value);
      }
    }
    return frame::Status::ok();
  });
}

}  // namespace

FRAME_REGISTER_KERNEL("mse_loss", frame::kCpuBackendName, mse_loss_cpu_kernel);
FRAME_REGISTER_KERNEL("mse_loss_grad_internal", frame::kCpuBackendName,
                      mse_loss_grad_internal_cpu_kernel);
