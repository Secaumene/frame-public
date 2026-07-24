// CPU 序列/归一化内核:M22 softmax/layer_norm 与 M25 selective_scan。
// 内核形如 Status kernel(ops::KernelContext&),内部经 dispatch_dtype 按 dtype
// 编译期展开(见 include/frame/core/dtype.h),再经 FRAME_REGISTER_KERNEL 注册到
// (op, kCpuBackendName)。两者均限 rank-2 [N,D]、作用于末轴,dtype 支持浮点
// 三档(kernel_dtype_checks.h::require_matching_supported_dtype)。

#include <cmath>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/ops/kernel_registry.h>

#include "accum_cast.h"
#include "kernel_dtype_checks.h"

namespace {

// float 累加转换:同目录共享工具(铁律 5 收敛,M22 批4 判重,见
// accum_cast.h 头注释)。
using frame::backends::cpu::from_accum;
using frame::backends::cpu::to_accum;

// selective_scan(x,a,b,c,d) 的 CPU 参考实现(REUSE-011:参考实现,数值校验
// 用,禁作性能路径)。除最后一轴外的每个前导位置是一条独立序列,状态 h
// 以 float 精度累计,fp16/bf16 仅在输入读取与输出写回时转换。
frame::Status selective_scan_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 5) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'selective_scan' cpu kernel expects 5 inputs (x, a, b, c, d), got " +
            std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'selective_scan' cpu kernel expects 1 output, got " +
                                   std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& x = ctx.inputs[0];
  const frame::Tensor& a = ctx.inputs[1];
  const frame::Tensor& b = ctx.inputs[2];
  const frame::Tensor& c = ctx.inputs[3];
  const frame::Tensor& d = ctx.inputs[4];
  frame::Tensor& out = ctx.outputs[0];

  const frame::Result<frame::DTypeCode> code_result =
      frame::backends::cpu::require_matching_supported_dtype("selective_scan", "x/a/b/c/d/out",
                                                             {&x, &a, &b, &c, &d, &out});
  if (!code_result.is_ok()) return code_result.status();
  const frame::DTypeCode code = code_result.value();

  if (x.shape().rank() < 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'selective_scan' cpu kernel requires rank >= 1, got rank 0");
  }
  for (const frame::Tensor* input : {&a, &b, &c, &d}) {
    if (!(input->shape() == x.shape())) {
      return frame::Status::make(
          frame::ErrorCode::kInvalidArgument,
          "op 'selective_scan' cpu kernel requires all input shapes to match x " +
              x.shape().to_string() + ", got " + input->shape().to_string());
    }
  }
  if (!(out.shape() == x.shape())) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'selective_scan' cpu kernel requires out shape to match x, got " +
            out.shape().to_string() + ", expected " + x.shape().to_string());
  }
  const int64_t steps = x.shape().dim(x.shape().rank() - 1);
  if (steps < 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'selective_scan' cpu kernel requires the last dimension (steps) to be >= 1, got " +
            std::to_string(steps));
  }
  const int64_t rows = x.numel() / steps;
  if (rows == 0) return frame::Status::ok();

  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    const T* x_data = static_cast<const T*>(x.raw_data());
    const T* a_data = static_cast<const T*>(a.raw_data());
    const T* b_data = static_cast<const T*>(b.raw_data());
    const T* c_data = static_cast<const T*>(c.raw_data());
    const T* d_data = static_cast<const T*>(d.raw_data());
    T* out_data = out.data<T>();

    for (int64_t row = 0; row < rows; ++row) {
      const int64_t row_base = row * steps;
      float state = 0.0F;
      for (int64_t step = 0; step < steps; ++step) {
        const int64_t index = row_base + step;
        const float x_value = to_accum<T>(x_data[index]);
        state = to_accum<T>(a_data[index]) * state + to_accum<T>(b_data[index]) * x_value;
        const float output =
            to_accum<T>(c_data[index]) * state + to_accum<T>(d_data[index]) * x_value;
        out_data[index] = from_accum<T>(output);
      }
    }
    return frame::Status::ok();
  });
}

// softmax(x[N,D]) 的 CPU 参考实现(REUSE-011:参考实现,数值校验用,禁作
// 性能路径)——逐行:先减行内最大值(数值稳定,kernel 层语义)、逐元素 exp、
// 求和、逐元素除以和;累加与中间量以 float 精度进行。
frame::Status softmax_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'softmax' cpu kernel expects 1 input, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'softmax' cpu kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& x = ctx.inputs[0];
  frame::Tensor& out = ctx.outputs[0];

  const frame::Result<frame::DTypeCode> code_result =
      frame::backends::cpu::require_matching_supported_dtype("softmax", "x/out", {&x, &out});
  if (!code_result.is_ok()) return code_result.status();
  const frame::DTypeCode code = code_result.value();

  if (x.shape().rank() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'softmax' cpu kernel requires x to be rank-2 [N, D], got rank " +
                                   std::to_string(x.shape().rank()));
  }
  if (!(out.shape() == x.shape())) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'softmax' cpu kernel requires out shape to match x, got " +
                                   out.shape().to_string() + ", expected " + x.shape().to_string());
  }

  const int64_t n = x.shape().dim(0);
  const int64_t d = x.shape().dim(1);

  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    const T* x_data = static_cast<const T*>(x.raw_data());
    T* out_data = out.data<T>();
    std::vector<float> row(static_cast<size_t>(d));

    for (int64_t i = 0; i < n; ++i) {
      const int64_t row_base = i * d;
      float row_max = to_accum<T>(x_data[row_base]);
      row[0] = row_max;
      for (int64_t j = 1; j < d; ++j) {
        const float v = to_accum<T>(x_data[row_base + j]);
        row[static_cast<size_t>(j)] = v;
        if (v > row_max) row_max = v;
      }
      float sum = 0.0F;
      for (int64_t j = 0; j < d; ++j) {
        const float e = std::exp(row[static_cast<size_t>(j)] - row_max);
        row[static_cast<size_t>(j)] = e;
        sum += e;
      }
      for (int64_t j = 0; j < d; ++j) {
        out_data[row_base + j] = from_accum<T>(row[static_cast<size_t>(j)] / sum);
      }
    }
    return frame::Status::ok();
  });
}

// layer_norm(x[N,D], gamma[D], beta[D]; eps) 的 CPU 参考实现(REUSE-011:参考
// 实现,数值校验用,禁作性能路径)——逐行:均值/方差以 float 精度累加,
// r=1/sqrt(var+eps),out=gamma*(x-mean)*r+beta。eps 已由构图期 shape_infer
// 校验为正(src/ops/schemas/sequence.cpp::infer_layer_norm_shape),kernel 侧
// 不重复校验(与 avg_pool2d 等既有 kernel 同一"不重验已由 shape_infer 把关的
// 深层数值约束"惯例)。
frame::Status layer_norm_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 3) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'layer_norm' cpu kernel expects 3 inputs (x, gamma, beta), "
                               "got " +
                                   std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'layer_norm' cpu kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& x = ctx.inputs[0];
  const frame::Tensor& gamma = ctx.inputs[1];
  const frame::Tensor& beta = ctx.inputs[2];
  frame::Tensor& out = ctx.outputs[0];

  const frame::Result<frame::DTypeCode> code_result =
      frame::backends::cpu::require_matching_supported_dtype("layer_norm", "x/gamma/beta/out",
                                                             {&x, &gamma, &beta, &out});
  if (!code_result.is_ok()) return code_result.status();
  const frame::DTypeCode code = code_result.value();

  if (x.shape().rank() != 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'layer_norm' cpu kernel requires x to be rank-2 [N, D], got rank " +
            std::to_string(x.shape().rank()));
  }
  const int64_t n = x.shape().dim(0);
  const int64_t d = x.shape().dim(1);
  if (gamma.shape().rank() != 1 || gamma.shape().dim(0) != d) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'layer_norm' cpu kernel requires gamma to be rank-1 [D=" +
                                   std::to_string(d) + "], got " + gamma.shape().to_string());
  }
  if (beta.shape().rank() != 1 || beta.shape().dim(0) != d) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'layer_norm' cpu kernel requires beta to be rank-1 [D=" +
                                   std::to_string(d) + "], got " + beta.shape().to_string());
  }
  if (!(out.shape() == x.shape())) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'layer_norm' cpu kernel requires out shape to match x, got " +
                                   out.shape().to_string() + ", expected " + x.shape().to_string());
  }

  if (ctx.attrs == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'layer_norm' cpu kernel is missing required attribute 'eps': no attrs provided");
  }
  const auto eps_it = ctx.attrs->find("eps");
  if (eps_it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'layer_norm' cpu kernel is missing required attribute 'eps'");
  }
  const double* eps_ptr = std::get_if<double>(&eps_it->second);
  if (eps_ptr == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'layer_norm' cpu kernel attribute 'eps' has the wrong type, expected double");
  }
  const float eps = static_cast<float>(*eps_ptr);

  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    const T* x_data = static_cast<const T*>(x.raw_data());
    const T* gamma_data = static_cast<const T*>(gamma.raw_data());
    const T* beta_data = static_cast<const T*>(beta.raw_data());
    T* out_data = out.data<T>();

    for (int64_t i = 0; i < n; ++i) {
      const int64_t row_base = i * d;
      float sum = 0.0F;
      for (int64_t j = 0; j < d; ++j) {
        sum += to_accum<T>(x_data[row_base + j]);
      }
      const float mean = sum / static_cast<float>(d);

      float var_sum = 0.0F;
      for (int64_t j = 0; j < d; ++j) {
        const float centered = to_accum<T>(x_data[row_base + j]) - mean;
        var_sum += centered * centered;
      }
      const float variance = var_sum / static_cast<float>(d);
      const float r = 1.0F / std::sqrt(variance + eps);

      for (int64_t j = 0; j < d; ++j) {
        const float xhat = (to_accum<T>(x_data[row_base + j]) - mean) * r;
        const float g = to_accum<T>(gamma_data[j]);
        const float b = to_accum<T>(beta_data[j]);
        out_data[row_base + j] = from_accum<T>(g * xhat + b);
      }
    }
    return frame::Status::ok();
  });
}

}  // namespace

FRAME_REGISTER_KERNEL("selective_scan", frame::kCpuBackendName, selective_scan_cpu_kernel);
FRAME_REGISTER_KERNEL("softmax", frame::kCpuBackendName, softmax_cpu_kernel);
FRAME_REGISTER_KERNEL("layer_norm", frame::kCpuBackendName, layer_norm_cpu_kernel);
