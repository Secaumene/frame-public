// CPU 二维池化内核(M21,批3 T4):max_pool2d/avg_pool2d(直接循环,ARCH-041
// 参考实现)+ max_pool2d_grad_internal/max_pool2d_select_internal/
// avg_pool2d_grad_internal(梯度专用)。内核形如 Status kernel(ops::
// KernelContext&),内部经 dispatch_dtype 按 dtype 编译期展开,再经
// FRAME_REGISTER_KERNEL 注册到 (op, kCpuBackendName)。
// 【REUSE-011:参考实现,数值校验用,禁作性能路径】。
//
// argmax 平局约定(与 src/ops/schemas/pool.cpp 头注释、cuda kernel 注释一致,
// 严格伴随性依赖此约定):窗口内以 kh 外层、kw 内层的行优先顺序遍历,严格
// `>` 比较——先出现的最大值获胜,即取窗口内最低线性索引 kh*KW+kw。
// max_pool2d 的 padding 语义 = -inf(只在有效、非 padding 位置参与取最大);
// avg_pool2d 的 padding 语义 = include-padding(分母恒 KH*KW,padding 位置
// 贡献 0)。

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
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

// NCHW 行优先线性下标,与 src/backends/cpu/kernels/conv.cpp::nchw_index 同一
// 动机各自持有一份实现(两文件互不可见,REUSE-002 自查)。相邻同型 int64_t
// 形参的论证同该文件头注释。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
int64_t nchw_index(int64_t n, int64_t c, int64_t h, int64_t w, int64_t c_dim, int64_t h_dim,
                   int64_t w_dim) {
  return ((n * c_dim + c) * h_dim + h) * w_dim + w;
}

frame::Status require_rank(std::string_view op_name, std::string_view operand_label,
                           int64_t expected_rank, const frame::Tensor& tensor) {
  if (tensor.shape().rank() != expected_rank) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' cpu kernel requires " +
                                   std::string(operand_label) + " to be rank-" +
                                   std::to_string(expected_rank) + ", got rank " +
                                   std::to_string(tensor.shape().rank()));
  }
  return frame::Status::ok();
}

// dtype 一致性 + v0 浮点三档校验:同目录共享工具(铁律 5 收敛,见
// kernel_dtype_checks.h 头注释)。
using frame::backends::cpu::require_matching_supported_dtype;

// 二维池化的几何参数,与 src/ops/schemas/pool.cpp::Pool2dGeometry 同构但各自
// 独立持有(kernel 侧从实际 Tensor::shape()/KernelContext::attrs 取值,
// REUSE-002 自查:两文件互不可见)。
struct Pool2dRuntimeParams {
  int64_t n = 0;
  int64_t c = 0;
  int64_t h = 0;
  int64_t w = 0;
  int64_t kh = 0;
  int64_t kw = 0;
  int64_t stride_h = 0;
  int64_t stride_w = 0;
  int64_t pad_h = 0;
  int64_t pad_w = 0;
  int64_t out_h = 0;
  int64_t out_w = 0;
};

frame::Result<std::vector<int64_t>> read_int64_array_attr(const frame::ops::KernelContext& ctx,
                                                          std::string_view op_name,
                                                          std::string_view attr_name) {
  if (ctx.attrs == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cpu kernel is missing required "
                                   "attribute '" +
                                   std::string(attr_name) + "': no attrs provided");
  }
  const auto it = ctx.attrs->find(std::string(attr_name));
  if (it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cpu kernel is missing required "
                                   "attribute '" +
                                   std::string(attr_name) + "'");
  }
  const std::vector<int64_t>* value = std::get_if<std::vector<int64_t>>(&it->second);
  if (value == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' cpu kernel attribute '" +
                                   std::string(attr_name) +
                                   "' has the wrong type, expected int64 array");
  }
  return *value;
}

frame::Result<frame::Shape> read_shape_attr(const frame::ops::KernelContext& ctx,
                                            std::string_view op_name, std::string_view attr_name) {
  if (ctx.attrs == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cpu kernel is missing required "
                                   "attribute '" +
                                   std::string(attr_name) + "': no attrs provided");
  }
  const auto it = ctx.attrs->find(std::string(attr_name));
  if (it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cpu kernel is missing required "
                                   "attribute '" +
                                   std::string(attr_name) + "'");
  }
  const frame::Shape* value = std::get_if<frame::Shape>(&it->second);
  if (value == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' cpu kernel attribute '" +
                                   std::string(attr_name) + "' has the wrong type, expected shape");
  }
  return *value;
}

// 读取 kernel/stride/padding 三个必需 kInt64Array 属性,组装进
// Pool2dRuntimeParams(REUSE-002:本文件全部 kernel 共用)。x_shape 提供
// [N,C,H,W]。
frame::Result<Pool2dRuntimeParams> read_pool2d_runtime_params(const frame::ops::KernelContext& ctx,
                                                              std::string_view op_name,
                                                              const frame::Shape& x_shape) {
  const frame::Result<std::vector<int64_t>> kernel_result =
      read_int64_array_attr(ctx, op_name, "kernel");
  if (!kernel_result.is_ok()) return kernel_result.status();
  const frame::Result<std::vector<int64_t>> stride_result =
      read_int64_array_attr(ctx, op_name, "stride");
  if (!stride_result.is_ok()) return stride_result.status();
  const frame::Result<std::vector<int64_t>> padding_result =
      read_int64_array_attr(ctx, op_name, "padding");
  if (!padding_result.is_ok()) return padding_result.status();

  const std::vector<int64_t>& kernel = kernel_result.value();
  const std::vector<int64_t>& stride = stride_result.value();
  const std::vector<int64_t>& padding = padding_result.value();
  if (kernel.size() != 2 || stride.size() != 2 || padding.size() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cpu kernel requires "
                                   "'kernel'/'stride'/'padding' to "
                                   "each have 2 elements");
  }

  Pool2dRuntimeParams params;
  params.n = x_shape.dim(0);
  params.c = x_shape.dim(1);
  params.h = x_shape.dim(2);
  params.w = x_shape.dim(3);
  params.kh = kernel[0];
  params.kw = kernel[1];
  params.stride_h = stride[0];
  params.stride_w = stride[1];
  params.pad_h = padding[0];
  params.pad_w = padding[1];
  params.out_h = (params.h + 2 * params.pad_h - params.kh) / params.stride_h + 1;
  params.out_w = (params.w + 2 * params.pad_w - params.kw) / params.stride_w + 1;
  return params;
}

// 在 (n,c,oh,ow) 对应的窗口内以 x_data 为参照求 argmax(平局取窗口内最低
// 线性索引 kh*KW+kw,由遍历顺序 + 严格 `>` 比较天然实现)。返回 true 且写入
// (best_ih,best_iw) 当窗口至少含一个有效(非 padding)位置,否则返回 false
// (理论上不应发生——padding*2<=kernel 的 schema 校验保证每个窗口至少覆盖
// 一个有效位置,此处仍防御性返回错误而非假设)。
// 相邻同型 int64_t 形参 (n,c,oh,ow) 是 NCHW 输出坐标的固定契约序;调用点均
// 以同名循环变量传入,误置换会立即越界或数值失真(同 nchw_index 先例论证)。
template <typename T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
bool find_window_argmax(const T* x_data, const Pool2dRuntimeParams& p, int64_t n, int64_t c,
                        int64_t oh, int64_t ow, int64_t& best_ih, int64_t& best_iw) {
  bool found = false;
  float best = 0.0F;
  for (int64_t kh_i = 0; kh_i < p.kh; ++kh_i) {
    const int64_t ih = oh * p.stride_h - p.pad_h + kh_i;
    if (ih < 0 || ih >= p.h) continue;
    for (int64_t kw_i = 0; kw_i < p.kw; ++kw_i) {
      const int64_t iw = ow * p.stride_w - p.pad_w + kw_i;
      if (iw < 0 || iw >= p.w) continue;
      const float value = to_accum(x_data[nchw_index(n, c, ih, iw, p.c, p.h, p.w)]);
      if (!found || value > best) {
        found = true;
        best = value;
        best_ih = ih;
        best_iw = iw;
      }
    }
  }
  return found;
}

// max_pool2d 前向计算核:padding 视为 -inf(仅在有效位置参与取最大)。
template <typename T>
frame::Status max_pool2d_forward_compute(const Pool2dRuntimeParams& p, const T* x_data,
                                         T* out_data) {
  for (int64_t n = 0; n < p.n; ++n) {
    for (int64_t c = 0; c < p.c; ++c) {
      for (int64_t oh = 0; oh < p.out_h; ++oh) {
        for (int64_t ow = 0; ow < p.out_w; ++ow) {
          int64_t best_ih = 0;
          int64_t best_iw = 0;
          const bool found = find_window_argmax(x_data, p, n, c, oh, ow, best_ih, best_iw);
          if (!found) {
            return frame::Status::make(
                frame::ErrorCode::kInternal,
                "op 'max_pool2d' cpu kernel window (n=" + std::to_string(n) +
                    ", c=" + std::to_string(c) + ", oh=" + std::to_string(oh) +
                    ", ow=" + std::to_string(ow) + ") has no valid (non-padding) position");
          }
          const float value = to_accum(x_data[nchw_index(n, c, best_ih, best_iw, p.c, p.h, p.w)]);
          out_data[nchw_index(n, c, oh, ow, p.c, p.out_h, p.out_w)] = from_accum<T>(value);
        }
      }
    }
  }
  return frame::Status::ok();
}

frame::Status max_pool2d_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'max_pool2d' cpu kernel expects 1 input, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'max_pool2d' cpu kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& x = ctx.inputs[0];
  frame::Tensor& out = ctx.outputs[0];

  frame::Status rank_status = require_rank("max_pool2d", "x", 4, x);
  if (!rank_status.is_ok()) return rank_status;

  const frame::Result<frame::DTypeCode> code_result =
      require_matching_supported_dtype("max_pool2d", "x/out", {&x, &out});
  if (!code_result.is_ok()) return code_result.status();
  const frame::DTypeCode code = code_result.value();

  const frame::Result<Pool2dRuntimeParams> params_result =
      read_pool2d_runtime_params(ctx, "max_pool2d", x.shape());
  if (!params_result.is_ok()) return params_result.status();
  const Pool2dRuntimeParams& params = params_result.value();

  const frame::Shape expected_out_shape({params.n, params.c, params.out_h, params.out_w});
  if (!(out.shape() == expected_out_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'max_pool2d' cpu kernel requires out shape to match the "
                               "pooling result, got " +
                                   out.shape().to_string() + ", expected " +
                                   expected_out_shape.to_string());
  }

  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    const T* x_data = static_cast<const T*>(x.raw_data());
    T* out_data = out.data<T>();
    return max_pool2d_forward_compute<T>(params, x_data, out_data);
  });
}

// avg_pool2d 前向计算核:分母恒 KH*KW(include padding),padding 位置贡献 0。
template <typename T>
void avg_pool2d_forward_compute(const Pool2dRuntimeParams& p, const T* x_data, T* out_data) {
  const float denom = static_cast<float>(p.kh * p.kw);
  for (int64_t n = 0; n < p.n; ++n) {
    for (int64_t c = 0; c < p.c; ++c) {
      for (int64_t oh = 0; oh < p.out_h; ++oh) {
        for (int64_t ow = 0; ow < p.out_w; ++ow) {
          float accum = 0.0F;
          for (int64_t kh_i = 0; kh_i < p.kh; ++kh_i) {
            const int64_t ih = oh * p.stride_h - p.pad_h + kh_i;
            if (ih < 0 || ih >= p.h) continue;
            for (int64_t kw_i = 0; kw_i < p.kw; ++kw_i) {
              const int64_t iw = ow * p.stride_w - p.pad_w + kw_i;
              if (iw < 0 || iw >= p.w) continue;
              accum += to_accum(x_data[nchw_index(n, c, ih, iw, p.c, p.h, p.w)]);
            }
          }
          out_data[nchw_index(n, c, oh, ow, p.c, p.out_h, p.out_w)] = from_accum<T>(accum / denom);
        }
      }
    }
  }
}

frame::Status avg_pool2d_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'avg_pool2d' cpu kernel expects 1 input, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'avg_pool2d' cpu kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& x = ctx.inputs[0];
  frame::Tensor& out = ctx.outputs[0];

  frame::Status rank_status = require_rank("avg_pool2d", "x", 4, x);
  if (!rank_status.is_ok()) return rank_status;

  const frame::Result<frame::DTypeCode> code_result =
      require_matching_supported_dtype("avg_pool2d", "x/out", {&x, &out});
  if (!code_result.is_ok()) return code_result.status();
  const frame::DTypeCode code = code_result.value();

  const frame::Result<Pool2dRuntimeParams> params_result =
      read_pool2d_runtime_params(ctx, "avg_pool2d", x.shape());
  if (!params_result.is_ok()) return params_result.status();
  const Pool2dRuntimeParams& params = params_result.value();

  const frame::Shape expected_out_shape({params.n, params.c, params.out_h, params.out_w});
  if (!(out.shape() == expected_out_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'avg_pool2d' cpu kernel requires out shape to match the "
                               "pooling result, got " +
                                   out.shape().to_string() + ", expected " +
                                   expected_out_shape.to_string());
  }

  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    const T* x_data = static_cast<const T*>(x.raw_data());
    T* out_data = out.data<T>();
    avg_pool2d_forward_compute<T>(params, x_data, out_data);
    return frame::Status::ok();
  });
}

// max_pool2d_grad_internal(dy,x)->dx:零初始化 dx,对每个 dy 位置以 x 重算
// argmax(平局取最低线性索引),把 dy 累加进该 argmax 位置(不同输出窗口在
// stride<kernel 时可能共享同一 argmax 位置,故用 += 累加;CPU 单线程顺序
// 执行,无需原子操作)。
// 相邻同型 const T* 形参 (dy,x) 与 schema 输入序一致;调用点以具名局部指针
// 传入,误置换会因形状寻址立即越界(同 conv2d_forward_compute 先例论证)。
template <typename T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
frame::Status max_pool2d_grad_compute(const Pool2dRuntimeParams& p, const T* dy_data,
                                      const T* x_data, T* dx_data) {
  const int64_t dx_numel = p.n * p.c * p.h * p.w;
  std::vector<float> accum(static_cast<size_t>(dx_numel), 0.0F);

  for (int64_t n = 0; n < p.n; ++n) {
    for (int64_t c = 0; c < p.c; ++c) {
      for (int64_t oh = 0; oh < p.out_h; ++oh) {
        for (int64_t ow = 0; ow < p.out_w; ++ow) {
          int64_t best_ih = 0;
          int64_t best_iw = 0;
          const bool found = find_window_argmax(x_data, p, n, c, oh, ow, best_ih, best_iw);
          if (!found) {
            return frame::Status::make(
                frame::ErrorCode::kInternal,
                "op 'max_pool2d_grad_internal' cpu kernel window (n=" + std::to_string(n) +
                    ", c=" + std::to_string(c) + ", oh=" + std::to_string(oh) +
                    ", ow=" + std::to_string(ow) + ") has no valid (non-padding) position");
          }
          const int64_t dx_idx = nchw_index(n, c, best_ih, best_iw, p.c, p.h, p.w);
          const int64_t dy_idx = nchw_index(n, c, oh, ow, p.c, p.out_h, p.out_w);
          accum[static_cast<size_t>(dx_idx)] += to_accum(dy_data[dy_idx]);
        }
      }
    }
  }

  for (int64_t i = 0; i < dx_numel; ++i) {
    dx_data[i] = from_accum<T>(accum[static_cast<size_t>(i)]);
  }
  return frame::Status::ok();
}

frame::Status max_pool2d_grad_internal_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'max_pool2d_grad_internal' cpu kernel expects 2 inputs, got " +
                                   std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'max_pool2d_grad_internal' cpu kernel expects 1 output, got " +
                                   std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& dy = ctx.inputs[0];
  const frame::Tensor& x = ctx.inputs[1];
  frame::Tensor& dx = ctx.outputs[0];

  frame::Status rank_status = require_rank("max_pool2d_grad_internal", "dy", 4, dy);
  if (!rank_status.is_ok()) return rank_status;
  rank_status = require_rank("max_pool2d_grad_internal", "x", 4, x);
  if (!rank_status.is_ok()) return rank_status;

  const frame::Result<frame::DTypeCode> code_result =
      require_matching_supported_dtype("max_pool2d_grad_internal", "dy/x/dx", {&dy, &x, &dx});
  if (!code_result.is_ok()) return code_result.status();
  const frame::DTypeCode code = code_result.value();

  const frame::Result<frame::Shape> input_shape_result =
      read_shape_attr(ctx, "max_pool2d_grad_internal", "input_shape");
  if (!input_shape_result.is_ok()) return input_shape_result.status();
  const frame::Shape& input_shape = input_shape_result.value();
  if (!(x.shape() == input_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'max_pool2d_grad_internal' cpu kernel requires x shape to "
                               "match attribute 'input_shape', got " +
                                   x.shape().to_string() + ", expected " + input_shape.to_string());
  }
  if (!(dx.shape() == input_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'max_pool2d_grad_internal' cpu kernel requires dx(out) shape "
                               "to match attribute 'input_shape', got " +
                                   dx.shape().to_string() + ", expected " +
                                   input_shape.to_string());
  }

  const frame::Result<Pool2dRuntimeParams> params_result =
      read_pool2d_runtime_params(ctx, "max_pool2d_grad_internal", input_shape);
  if (!params_result.is_ok()) return params_result.status();
  const Pool2dRuntimeParams& params = params_result.value();

  const frame::Shape expected_dy_shape({params.n, params.c, params.out_h, params.out_w});
  if (!(dy.shape() == expected_dy_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'max_pool2d_grad_internal' cpu kernel requires dy shape to be "
                               "consistent with input_shape/kernel/stride/padding, got " +
                                   dy.shape().to_string() + ", expected " +
                                   expected_dy_shape.to_string());
  }

  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    const T* dy_data = static_cast<const T*>(dy.raw_data());
    const T* x_data = static_cast<const T*>(x.raw_data());
    T* dx_data = dx.data<T>();
    return max_pool2d_grad_compute<T>(params, dy_data, x_data, dx_data);
  });
}

// max_pool2d_select_internal(g,x)->out:对每个输出窗口以 x 重算 argmax(同一
// 平局约定),取 g 在该位置的值(纯 gather,一对一,无需累加)。
template <typename T>
frame::Status max_pool2d_select_compute(const Pool2dRuntimeParams& p, const T* g_data,
                                        const T* x_data, T* out_data) {
  for (int64_t n = 0; n < p.n; ++n) {
    for (int64_t c = 0; c < p.c; ++c) {
      for (int64_t oh = 0; oh < p.out_h; ++oh) {
        for (int64_t ow = 0; ow < p.out_w; ++ow) {
          int64_t best_ih = 0;
          int64_t best_iw = 0;
          const bool found = find_window_argmax(x_data, p, n, c, oh, ow, best_ih, best_iw);
          if (!found) {
            return frame::Status::make(
                frame::ErrorCode::kInternal,
                "op 'max_pool2d_select_internal' cpu kernel window (n=" + std::to_string(n) +
                    ", c=" + std::to_string(c) + ", oh=" + std::to_string(oh) +
                    ", ow=" + std::to_string(ow) + ") has no valid (non-padding) position");
          }
          const float value = to_accum(g_data[nchw_index(n, c, best_ih, best_iw, p.c, p.h, p.w)]);
          out_data[nchw_index(n, c, oh, ow, p.c, p.out_h, p.out_w)] = from_accum<T>(value);
        }
      }
    }
  }
  return frame::Status::ok();
}

frame::Status max_pool2d_select_internal_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'max_pool2d_select_internal' cpu kernel expects 2 inputs, got " +
                                   std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'max_pool2d_select_internal' cpu kernel expects 1 output, got " +
                                   std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& g = ctx.inputs[0];
  const frame::Tensor& x = ctx.inputs[1];
  frame::Tensor& out = ctx.outputs[0];

  frame::Status rank_status = require_rank("max_pool2d_select_internal", "g", 4, g);
  if (!rank_status.is_ok()) return rank_status;
  rank_status = require_rank("max_pool2d_select_internal", "x", 4, x);
  if (!rank_status.is_ok()) return rank_status;
  if (!(g.shape() == x.shape())) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'max_pool2d_select_internal' cpu kernel requires g and x of "
                               "the same shape, got " +
                                   g.shape().to_string() + " and " + x.shape().to_string());
  }

  const frame::Result<frame::DTypeCode> code_result =
      require_matching_supported_dtype("max_pool2d_select_internal", "g/x/out", {&g, &x, &out});
  if (!code_result.is_ok()) return code_result.status();
  const frame::DTypeCode code = code_result.value();

  const frame::Result<Pool2dRuntimeParams> params_result =
      read_pool2d_runtime_params(ctx, "max_pool2d_select_internal", x.shape());
  if (!params_result.is_ok()) return params_result.status();
  const Pool2dRuntimeParams& params = params_result.value();

  const frame::Shape expected_out_shape({params.n, params.c, params.out_h, params.out_w});
  if (!(out.shape() == expected_out_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'max_pool2d_select_internal' cpu kernel requires out shape to "
                               "match the pooling result, got " +
                                   out.shape().to_string() + ", expected " +
                                   expected_out_shape.to_string());
  }

  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    const T* g_data = static_cast<const T*>(g.raw_data());
    const T* x_data = static_cast<const T*>(x.raw_data());
    T* out_data = out.data<T>();
    return max_pool2d_select_compute<T>(params, g_data, x_data, out_data);
  });
}

// avg_pool2d_grad_internal(dy)->dx:逐输入位聚合覆盖窗口(gather 风格,不
// 物化中间 scatter 缓冲,无原子操作)——对每个 dx[n,c,ih,iw],枚举全部满足
// (ih+pad-kh)%stride==0 的 (oh,ow) 组合求和 dy/denom。
template <typename T>
void avg_pool2d_grad_compute(const Pool2dRuntimeParams& p, const T* dy_data, T* dx_data) {
  const float denom = static_cast<float>(p.kh * p.kw);
  for (int64_t n = 0; n < p.n; ++n) {
    for (int64_t c = 0; c < p.c; ++c) {
      for (int64_t ih = 0; ih < p.h; ++ih) {
        for (int64_t iw = 0; iw < p.w; ++iw) {
          float accum = 0.0F;
          for (int64_t kh_i = 0; kh_i < p.kh; ++kh_i) {
            const int64_t num_h = ih + p.pad_h - kh_i;
            if (num_h % p.stride_h != 0) continue;
            const int64_t oh = num_h / p.stride_h;
            if (oh < 0 || oh >= p.out_h) continue;
            for (int64_t kw_i = 0; kw_i < p.kw; ++kw_i) {
              const int64_t num_w = iw + p.pad_w - kw_i;
              if (num_w % p.stride_w != 0) continue;
              const int64_t ow = num_w / p.stride_w;
              if (ow < 0 || ow >= p.out_w) continue;
              accum += to_accum(dy_data[nchw_index(n, c, oh, ow, p.c, p.out_h, p.out_w)]);
            }
          }
          dx_data[nchw_index(n, c, ih, iw, p.c, p.h, p.w)] = from_accum<T>(accum / denom);
        }
      }
    }
  }
}

frame::Status avg_pool2d_grad_internal_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'avg_pool2d_grad_internal' cpu kernel expects 1 input, got " +
                                   std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'avg_pool2d_grad_internal' cpu kernel expects 1 output, got " +
                                   std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& dy = ctx.inputs[0];
  frame::Tensor& dx = ctx.outputs[0];

  frame::Status rank_status = require_rank("avg_pool2d_grad_internal", "dy", 4, dy);
  if (!rank_status.is_ok()) return rank_status;

  const frame::Result<frame::DTypeCode> code_result =
      require_matching_supported_dtype("avg_pool2d_grad_internal", "dy/dx", {&dy, &dx});
  if (!code_result.is_ok()) return code_result.status();
  const frame::DTypeCode code = code_result.value();

  const frame::Result<frame::Shape> input_shape_result =
      read_shape_attr(ctx, "avg_pool2d_grad_internal", "input_shape");
  if (!input_shape_result.is_ok()) return input_shape_result.status();
  const frame::Shape& input_shape = input_shape_result.value();
  if (!(dx.shape() == input_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'avg_pool2d_grad_internal' cpu kernel requires dx(out) shape "
                               "to match attribute 'input_shape', got " +
                                   dx.shape().to_string() + ", expected " +
                                   input_shape.to_string());
  }

  const frame::Result<Pool2dRuntimeParams> params_result =
      read_pool2d_runtime_params(ctx, "avg_pool2d_grad_internal", input_shape);
  if (!params_result.is_ok()) return params_result.status();
  const Pool2dRuntimeParams& params = params_result.value();

  const frame::Shape expected_dy_shape({params.n, params.c, params.out_h, params.out_w});
  if (!(dy.shape() == expected_dy_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'avg_pool2d_grad_internal' cpu kernel requires dy shape to be "
                               "consistent with input_shape/kernel/stride/padding, got " +
                                   dy.shape().to_string() + ", expected " +
                                   expected_dy_shape.to_string());
  }

  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    const T* dy_data = static_cast<const T*>(dy.raw_data());
    T* dx_data = dx.data<T>();
    avg_pool2d_grad_compute<T>(params, dy_data, dx_data);
    return frame::Status::ok();
  });
}

}  // namespace

FRAME_REGISTER_KERNEL("max_pool2d", frame::kCpuBackendName, max_pool2d_cpu_kernel);
FRAME_REGISTER_KERNEL("avg_pool2d", frame::kCpuBackendName, avg_pool2d_cpu_kernel);
FRAME_REGISTER_KERNEL("max_pool2d_grad_internal", frame::kCpuBackendName,
                      max_pool2d_grad_internal_cpu_kernel);
FRAME_REGISTER_KERNEL("max_pool2d_select_internal", frame::kCpuBackendName,
                      max_pool2d_select_internal_cpu_kernel);
FRAME_REGISTER_KERNEL("avg_pool2d_grad_internal", frame::kCpuBackendName,
                      avg_pool2d_grad_internal_cpu_kernel);
