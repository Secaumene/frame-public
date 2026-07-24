// CPU 二维/一维卷积内核(M21,批3 T4):conv2d/conv1d(直接循环,ARCH-041 参考
// 实现)+ conv2d_grad_input_internal/conv2d_grad_filter_internal(梯度专用)。
// 内核形如 Status kernel(ops::KernelContext&),内部经 dispatch_dtype 按 dtype
// 编译期展开,再经 FRAME_REGISTER_KERNEL 注册到 (op, kCpuBackendName)。
// 【REUSE-011:参考实现,数值校验用,禁作性能路径】——朴素多重循环,不追求
// 性能,唯一目标是数值正确性(与 src/backends/cpu/kernels/matmul.cpp 同一
// 豁免口径:cuDNN 等加速库留给 cuda 等性能后端,T5 交付)。

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

// NCHW 行优先线性下标 ((n*C+c)*H+h)*W+w,供本文件全部 kernel 共用
// (REUSE-002)。7 个相邻同型 int64_t 形参是 NCHW 张量寻址的固定契约形态
// (下标四元组 + 对应维度大小三元组恒为该顺序);调用点均以具名局部变量
// (循环变量/维度大小)传入,误置换会导致越界访问或参考实现数值立即失真,
// 不会静默产出难以察觉的错误结果(同 src/nn/layers.cpp::Linear 的
// batch/in_dim/out_dim 先例论证)。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
int64_t nchw_index(int64_t n, int64_t c, int64_t h, int64_t w, int64_t c_dim, int64_t h_dim,
                   int64_t w_dim) {
  return ((n * c_dim + c) * h_dim + h) * w_dim + w;
}

// NCL(一维,N/C/L)行优先线性下标,供 conv1d 使用。
int64_t ncl_index(int64_t n, int64_t c, int64_t l, int64_t c_dim, int64_t l_dim) {
  return (n * c_dim + c) * l_dim + l;
}

// 校验张量为指定秩,不满足返回英文错误(消息含 op 名/操作数标签/期望秩/
// 实际秩)。REUSE-002:本文件全部 kernel 共用。
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

// 二维卷积的几何参数,与 src/ops/schemas/conv.cpp::Conv2dGeometry 同构但各自
// 独立持有(kernel 侧从实际 Tensor::shape()/KernelContext::attrs 取值,与
// schema 侧从 NodeContext 取值来源不同,REUSE-002 自查:两文件互不可见)。
struct Conv2dRuntimeParams {
  int64_t n = 0;
  int64_t cin = 0;
  int64_t h = 0;
  int64_t w = 0;
  int64_t cout = 0;
  int64_t cin_per_group = 0;
  int64_t kh = 0;
  int64_t kw = 0;
  int64_t groups = 0;
  int64_t stride_h = 0;
  int64_t stride_w = 0;
  int64_t pad_h = 0;
  int64_t pad_w = 0;
  int64_t out_h = 0;
  int64_t out_w = 0;
};

// 从 ctx.attrs 读取 kInt64Array 属性(2 元)或 kInt64 属性,供本文件 conv2d
// 系 kernel 共用(REUSE-002)。
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

frame::Result<int64_t> read_int64_attr(const frame::ops::KernelContext& ctx,
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
  const int64_t* value = std::get_if<int64_t>(&it->second);
  if (value == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' cpu kernel attribute '" +
                                   std::string(attr_name) + "' has the wrong type, expected int64");
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

// 读取 conv2d 系 kernel 共用的 stride/padding/groups 三属性,组装进
// Conv2dRuntimeParams(REUSE-002:conv2d/conv2d_grad_input_internal/
// conv2d_grad_filter_internal 三处共用)。x_shape/w_shape 分别提供
// [N,Cin,H,W]/[Cout,Cin/groups,KH,KW]。
frame::Result<Conv2dRuntimeParams> read_conv2d_runtime_params(const frame::ops::KernelContext& ctx,
                                                              std::string_view op_name,
                                                              const frame::Shape& x_shape,
                                                              const frame::Shape& w_shape) {
  const frame::Result<std::vector<int64_t>> stride_result =
      read_int64_array_attr(ctx, op_name, "stride");
  if (!stride_result.is_ok()) return stride_result.status();
  const frame::Result<std::vector<int64_t>> padding_result =
      read_int64_array_attr(ctx, op_name, "padding");
  if (!padding_result.is_ok()) return padding_result.status();
  const frame::Result<int64_t> groups_result = read_int64_attr(ctx, op_name, "groups");
  if (!groups_result.is_ok()) return groups_result.status();

  const std::vector<int64_t>& stride = stride_result.value();
  const std::vector<int64_t>& padding = padding_result.value();
  if (stride.size() != 2 || padding.size() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cpu kernel requires "
                                   "'stride'/'padding' to each have "
                                   "2 elements");
  }

  Conv2dRuntimeParams params;
  params.n = x_shape.dim(0);
  params.cin = x_shape.dim(1);
  params.h = x_shape.dim(2);
  params.w = x_shape.dim(3);
  params.cout = w_shape.dim(0);
  params.cin_per_group = w_shape.dim(1);
  params.kh = w_shape.dim(2);
  params.kw = w_shape.dim(3);
  params.groups = groups_result.value();
  params.stride_h = stride[0];
  params.stride_w = stride[1];
  params.pad_h = padding[0];
  params.pad_w = padding[1];
  params.out_h = (params.h + 2 * params.pad_h - params.kh) / params.stride_h + 1;
  params.out_w = (params.w + 2 * params.pad_w - params.kw) / params.stride_w + 1;
  return params;
}

// conv2d 前向计算核:直接六重循环(N,Cout,out_h,out_w,Cin_per_group,KH,KW,
// 实际七重——groups 经通道分组隐含体现,不额外计数)。bias_data 为 nullptr
// 表示无 bias。
// 相邻同型 const T* 形参(x/w/bias)是卷积算子的固定契约形态(x/w/bias/out
// 顺序与 schema 输入序一致);调用点均以具名局部指针传入,误置换会因形状
// 寻址立即越界或数值失真,不会静默产出难以察觉的错误(同 nchw_index 先例
// 论证)。
template <typename T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void conv2d_forward_compute(const Conv2dRuntimeParams& p, const T* x_data, const T* w_data,
                            const T* bias_data, T* out_data) {
  const int64_t cout_per_group = p.cout / p.groups;
  for (int64_t n = 0; n < p.n; ++n) {
    for (int64_t co = 0; co < p.cout; ++co) {
      const int64_t g = co / cout_per_group;
      for (int64_t oh = 0; oh < p.out_h; ++oh) {
        for (int64_t ow = 0; ow < p.out_w; ++ow) {
          float accum = 0.0F;
          for (int64_t ci = 0; ci < p.cin_per_group; ++ci) {
            const int64_t ci_full = g * p.cin_per_group + ci;
            for (int64_t kh_i = 0; kh_i < p.kh; ++kh_i) {
              const int64_t ih = oh * p.stride_h - p.pad_h + kh_i;
              if (ih < 0 || ih >= p.h) continue;
              for (int64_t kw_i = 0; kw_i < p.kw; ++kw_i) {
                const int64_t iw = ow * p.stride_w - p.pad_w + kw_i;
                if (iw < 0 || iw >= p.w) continue;
                const int64_t x_idx = nchw_index(n, ci_full, ih, iw, p.cin, p.h, p.w);
                const int64_t w_idx = nchw_index(co, ci, kh_i, kw_i, p.cin_per_group, p.kh, p.kw);
                accum += to_accum(x_data[x_idx]) * to_accum(w_data[w_idx]);
              }
            }
          }
          if (bias_data != nullptr) {
            accum += to_accum(bias_data[co]);
          }
          out_data[nchw_index(n, co, oh, ow, p.cout, p.out_h, p.out_w)] = from_accum<T>(accum);
        }
      }
    }
  }
}

// conv2d 的 CPU 参考实现:2 或 3 输入(x,w[,bias]),1 输出。
frame::Status conv2d_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 2 && ctx.inputs.size() != 3) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'conv2d' cpu kernel expects 2 or 3 inputs, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'conv2d' cpu kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }
  const bool has_bias = ctx.inputs.size() == 3;

  const frame::Tensor& x = ctx.inputs[0];
  const frame::Tensor& w = ctx.inputs[1];
  frame::Tensor& out = ctx.outputs[0];

  frame::Status rank_status = require_rank("conv2d", "x", 4, x);
  if (!rank_status.is_ok()) return rank_status;
  rank_status = require_rank("conv2d", "w", 4, w);
  if (!rank_status.is_ok()) return rank_status;
  if (has_bias) {
    rank_status = require_rank("conv2d", "bias", 1, ctx.inputs[2]);
    if (!rank_status.is_ok()) return rank_status;
  }

  std::vector<const frame::Tensor*> checked_tensors{&x, &w, &out};
  if (has_bias) checked_tensors.insert(checked_tensors.begin() + 2, &ctx.inputs[2]);
  const frame::Result<frame::DTypeCode> code_result = require_matching_supported_dtype(
      "conv2d", has_bias ? "x/w/bias/out" : "x/w/out", checked_tensors);
  if (!code_result.is_ok()) return code_result.status();
  const frame::DTypeCode code = code_result.value();

  const frame::Result<Conv2dRuntimeParams> params_result =
      read_conv2d_runtime_params(ctx, "conv2d", x.shape(), w.shape());
  if (!params_result.is_ok()) return params_result.status();
  const Conv2dRuntimeParams& params = params_result.value();

  const frame::Shape expected_out_shape({params.n, params.cout, params.out_h, params.out_w});
  if (!(out.shape() == expected_out_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'conv2d' cpu kernel requires out shape to match the "
                               "convolution result, got " +
                                   out.shape().to_string() + ", expected " +
                                   expected_out_shape.to_string());
  }

  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    const T* x_data = static_cast<const T*>(x.raw_data());
    const T* w_data = static_cast<const T*>(w.raw_data());
    const T* bias_data = has_bias ? static_cast<const T*>(ctx.inputs[2].raw_data()) : nullptr;
    T* out_data = out.data<T>();
    conv2d_forward_compute<T>(params, x_data, w_data, bias_data, out_data);
    return frame::Status::ok();
  });
}

// conv2d_grad_input_internal(dy,w)->dx 的计算核(BackwardData):对每个输出
// 位置 dx[n,ci,ih,iw],按“输出索引=输入索引投影”的反向(gather,非 scatter)
// 方式,枚举全部满足 (ih+pad-kh)%stride==0 的 (co,kh,kw) 组合求和(与
// src/ops/schemas/conv.cpp 头注释一致,不物化中间 scatter 缓冲)。
template <typename T>
void conv2d_grad_input_compute(const Conv2dRuntimeParams& p, const T* dy_data, const T* w_data,
                               T* dx_data) {
  const int64_t cout_per_group = p.cout / p.groups;
  for (int64_t n = 0; n < p.n; ++n) {
    for (int64_t ci_full = 0; ci_full < p.cin; ++ci_full) {
      const int64_t g = ci_full / p.cin_per_group;
      const int64_t ci = ci_full - g * p.cin_per_group;
      for (int64_t ih = 0; ih < p.h; ++ih) {
        for (int64_t iw = 0; iw < p.w; ++iw) {
          float accum = 0.0F;
          for (int64_t co = g * cout_per_group; co < (g + 1) * cout_per_group; ++co) {
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
                const int64_t dy_idx = nchw_index(n, co, oh, ow, p.cout, p.out_h, p.out_w);
                const int64_t w_idx = nchw_index(co, ci, kh_i, kw_i, p.cin_per_group, p.kh, p.kw);
                accum += to_accum(dy_data[dy_idx]) * to_accum(w_data[w_idx]);
              }
            }
          }
          dx_data[nchw_index(n, ci_full, ih, iw, p.cin, p.h, p.w)] = from_accum<T>(accum);
        }
      }
    }
  }
}

frame::Status conv2d_grad_input_internal_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'conv2d_grad_input_internal' cpu kernel expects 2 inputs, "
                               "got " +
                                   std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'conv2d_grad_input_internal' cpu kernel expects 1 output, "
                               "got " +
                                   std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& dy = ctx.inputs[0];
  const frame::Tensor& w = ctx.inputs[1];
  frame::Tensor& dx = ctx.outputs[0];

  frame::Status rank_status = require_rank("conv2d_grad_input_internal", "dy", 4, dy);
  if (!rank_status.is_ok()) return rank_status;
  rank_status = require_rank("conv2d_grad_input_internal", "w", 4, w);
  if (!rank_status.is_ok()) return rank_status;

  const frame::Result<frame::DTypeCode> code_result =
      require_matching_supported_dtype("conv2d_grad_input_internal", "dy/w/dx", {&dy, &w, &dx});
  if (!code_result.is_ok()) return code_result.status();
  const frame::DTypeCode code = code_result.value();

  // input_shape 是权威几何来源(与 src/ops/schemas/conv.cpp 的 shape 推断
  // 同一份契约);dx(out)的实际 shape 另行与之交叉核对,不隐式信任调用方
  // 预分配的 out 一定正确(同 sum_grad_internal cpu kernel 先例)。
  const frame::Result<frame::Shape> input_shape_result =
      read_shape_attr(ctx, "conv2d_grad_input_internal", "input_shape");
  if (!input_shape_result.is_ok()) return input_shape_result.status();
  const frame::Shape& input_shape = input_shape_result.value();
  if (!(dx.shape() == input_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'conv2d_grad_input_internal' cpu kernel requires dx(out) "
                               "shape to match attribute 'input_shape', got " +
                                   dx.shape().to_string() + ", expected " +
                                   input_shape.to_string());
  }

  const frame::Result<Conv2dRuntimeParams> params_result =
      read_conv2d_runtime_params(ctx, "conv2d_grad_input_internal", input_shape, w.shape());
  if (!params_result.is_ok()) return params_result.status();
  const Conv2dRuntimeParams& params = params_result.value();

  const frame::Shape expected_dy_shape({params.n, params.cout, params.out_h, params.out_w});
  if (!(dy.shape() == expected_dy_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'conv2d_grad_input_internal' cpu kernel requires dy shape to "
                               "match [N, Cout, out_h, out_w], got " +
                                   dy.shape().to_string() + ", expected " +
                                   expected_dy_shape.to_string());
  }

  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    const T* dy_data = static_cast<const T*>(dy.raw_data());
    const T* w_data = static_cast<const T*>(w.raw_data());
    T* dx_data = dx.data<T>();
    conv2d_grad_input_compute<T>(params, dy_data, w_data, dx_data);
    return frame::Status::ok();
  });
}

// conv2d_grad_filter_internal(x,dy)->dw 的计算核(BackwardFilter):对每个
// 滤波器位置 dw[co,ci,kh,kw],枚举全部 (n,oh,ow) 求和(gather 风格,同上)。
template <typename T>
void conv2d_grad_filter_compute(const Conv2dRuntimeParams& p, const T* x_data, const T* dy_data,
                                T* dw_data) {
  const int64_t cout_per_group = p.cout / p.groups;
  for (int64_t co = 0; co < p.cout; ++co) {
    const int64_t g = co / cout_per_group;
    for (int64_t ci = 0; ci < p.cin_per_group; ++ci) {
      const int64_t ci_full = g * p.cin_per_group + ci;
      for (int64_t kh_i = 0; kh_i < p.kh; ++kh_i) {
        for (int64_t kw_i = 0; kw_i < p.kw; ++kw_i) {
          float accum = 0.0F;
          for (int64_t n = 0; n < p.n; ++n) {
            for (int64_t oh = 0; oh < p.out_h; ++oh) {
              const int64_t ih = oh * p.stride_h - p.pad_h + kh_i;
              if (ih < 0 || ih >= p.h) continue;
              for (int64_t ow = 0; ow < p.out_w; ++ow) {
                const int64_t iw = ow * p.stride_w - p.pad_w + kw_i;
                if (iw < 0 || iw >= p.w) continue;
                const int64_t x_idx = nchw_index(n, ci_full, ih, iw, p.cin, p.h, p.w);
                const int64_t dy_idx = nchw_index(n, co, oh, ow, p.cout, p.out_h, p.out_w);
                accum += to_accum(x_data[x_idx]) * to_accum(dy_data[dy_idx]);
              }
            }
          }
          dw_data[nchw_index(co, ci, kh_i, kw_i, p.cin_per_group, p.kh, p.kw)] =
              from_accum<T>(accum);
        }
      }
    }
  }
}

frame::Status conv2d_grad_filter_internal_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'conv2d_grad_filter_internal' cpu kernel expects 2 inputs, "
                               "got " +
                                   std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'conv2d_grad_filter_internal' cpu kernel expects 1 output, "
                               "got " +
                                   std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& x = ctx.inputs[0];
  const frame::Tensor& dy = ctx.inputs[1];
  frame::Tensor& dw = ctx.outputs[0];

  frame::Status rank_status = require_rank("conv2d_grad_filter_internal", "x", 4, x);
  if (!rank_status.is_ok()) return rank_status;
  rank_status = require_rank("conv2d_grad_filter_internal", "dy", 4, dy);
  if (!rank_status.is_ok()) return rank_status;

  const frame::Result<frame::DTypeCode> code_result =
      require_matching_supported_dtype("conv2d_grad_filter_internal", "x/dy/dw", {&x, &dy, &dw});
  if (!code_result.is_ok()) return code_result.status();
  const frame::DTypeCode code = code_result.value();

  // filter_shape 是权威几何来源;dw(out)的实际 shape 另行交叉核对(同上,
  // sum_grad_internal cpu kernel 先例)。
  const frame::Result<frame::Shape> filter_shape_result =
      read_shape_attr(ctx, "conv2d_grad_filter_internal", "filter_shape");
  if (!filter_shape_result.is_ok()) return filter_shape_result.status();
  const frame::Shape& filter_shape = filter_shape_result.value();
  if (!(dw.shape() == filter_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'conv2d_grad_filter_internal' cpu kernel requires dw(out) "
                               "shape to match attribute 'filter_shape', got " +
                                   dw.shape().to_string() + ", expected " +
                                   filter_shape.to_string());
  }

  const frame::Result<Conv2dRuntimeParams> params_result =
      read_conv2d_runtime_params(ctx, "conv2d_grad_filter_internal", x.shape(), filter_shape);
  if (!params_result.is_ok()) return params_result.status();
  const Conv2dRuntimeParams& params = params_result.value();

  const frame::Shape expected_dy_shape({params.n, params.cout, params.out_h, params.out_w});
  if (!(dy.shape() == expected_dy_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'conv2d_grad_filter_internal' cpu kernel requires dy shape to "
                               "match [N, Cout, out_h, out_w], got " +
                                   dy.shape().to_string() + ", expected " +
                                   expected_dy_shape.to_string());
  }

  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    const T* x_data = static_cast<const T*>(x.raw_data());
    const T* dy_data = static_cast<const T*>(dy.raw_data());
    T* dw_data = dw.data<T>();
    conv2d_grad_filter_compute<T>(params, x_data, dy_data, dw_data);
    return frame::Status::ok();
  });
}

// ---------------------------------------------------------------------------
// conv1d:一维卷积直接参考 kernel(裁决点②,不经 decomposition 落地——CUDA
// 侧才走 decomposition 复用 conv2d,CPU 侧仍有直循环参考 kernel,ARCH-041)。
// ---------------------------------------------------------------------------

struct Conv1dRuntimeParams {
  int64_t n = 0;
  int64_t cin = 0;
  int64_t l = 0;
  int64_t cout = 0;
  int64_t cin_per_group = 0;
  int64_t k = 0;
  int64_t groups = 0;
  int64_t stride = 0;
  int64_t padding = 0;
  int64_t out_l = 0;
};

frame::Result<Conv1dRuntimeParams> read_conv1d_runtime_params(const frame::ops::KernelContext& ctx,
                                                              const frame::Shape& x_shape,
                                                              const frame::Shape& w_shape) {
  const frame::Result<int64_t> stride_result = read_int64_attr(ctx, "conv1d", "stride");
  if (!stride_result.is_ok()) return stride_result.status();
  const frame::Result<int64_t> padding_result = read_int64_attr(ctx, "conv1d", "padding");
  if (!padding_result.is_ok()) return padding_result.status();
  const frame::Result<int64_t> groups_result = read_int64_attr(ctx, "conv1d", "groups");
  if (!groups_result.is_ok()) return groups_result.status();

  Conv1dRuntimeParams params;
  params.n = x_shape.dim(0);
  params.cin = x_shape.dim(1);
  params.l = x_shape.dim(2);
  params.cout = w_shape.dim(0);
  params.cin_per_group = w_shape.dim(1);
  params.k = w_shape.dim(2);
  params.groups = groups_result.value();
  params.stride = stride_result.value();
  params.padding = padding_result.value();
  params.out_l = (params.l + 2 * params.padding - params.k) / params.stride + 1;
  return params;
}

// 相邻同型 const T* 形参:论证同 conv2d_forward_compute(x/w/bias/out 契约序)。
template <typename T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void conv1d_forward_compute(const Conv1dRuntimeParams& p, const T* x_data, const T* w_data,
                            const T* bias_data, T* out_data) {
  const int64_t cout_per_group = p.cout / p.groups;
  for (int64_t n = 0; n < p.n; ++n) {
    for (int64_t co = 0; co < p.cout; ++co) {
      const int64_t g = co / cout_per_group;
      for (int64_t ol = 0; ol < p.out_l; ++ol) {
        float accum = 0.0F;
        for (int64_t ci = 0; ci < p.cin_per_group; ++ci) {
          const int64_t ci_full = g * p.cin_per_group + ci;
          for (int64_t k_i = 0; k_i < p.k; ++k_i) {
            const int64_t il = ol * p.stride - p.padding + k_i;
            if (il < 0 || il >= p.l) continue;
            const int64_t x_idx = ncl_index(n, ci_full, il, p.cin, p.l);
            const int64_t w_idx = ncl_index(co, ci, k_i, p.cin_per_group, p.k);
            accum += to_accum(x_data[x_idx]) * to_accum(w_data[w_idx]);
          }
        }
        if (bias_data != nullptr) {
          accum += to_accum(bias_data[co]);
        }
        out_data[ncl_index(n, co, ol, p.cout, p.out_l)] = from_accum<T>(accum);
      }
    }
  }
}

frame::Status conv1d_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 2 && ctx.inputs.size() != 3) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'conv1d' cpu kernel expects 2 or 3 inputs, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'conv1d' cpu kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }
  const bool has_bias = ctx.inputs.size() == 3;

  const frame::Tensor& x = ctx.inputs[0];
  const frame::Tensor& w = ctx.inputs[1];
  frame::Tensor& out = ctx.outputs[0];

  frame::Status rank_status = require_rank("conv1d", "x", 3, x);
  if (!rank_status.is_ok()) return rank_status;
  rank_status = require_rank("conv1d", "w", 3, w);
  if (!rank_status.is_ok()) return rank_status;
  if (has_bias) {
    rank_status = require_rank("conv1d", "bias", 1, ctx.inputs[2]);
    if (!rank_status.is_ok()) return rank_status;
  }

  std::vector<const frame::Tensor*> checked_tensors{&x, &w, &out};
  if (has_bias) checked_tensors.insert(checked_tensors.begin() + 2, &ctx.inputs[2]);
  const frame::Result<frame::DTypeCode> code_result = require_matching_supported_dtype(
      "conv1d", has_bias ? "x/w/bias/out" : "x/w/out", checked_tensors);
  if (!code_result.is_ok()) return code_result.status();
  const frame::DTypeCode code = code_result.value();

  const frame::Result<Conv1dRuntimeParams> params_result =
      read_conv1d_runtime_params(ctx, x.shape(), w.shape());
  if (!params_result.is_ok()) return params_result.status();
  const Conv1dRuntimeParams& params = params_result.value();

  const frame::Shape expected_out_shape({params.n, params.cout, params.out_l});
  if (!(out.shape() == expected_out_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'conv1d' cpu kernel requires out shape to match the "
                               "convolution result, got " +
                                   out.shape().to_string() + ", expected " +
                                   expected_out_shape.to_string());
  }

  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    const T* x_data = static_cast<const T*>(x.raw_data());
    const T* w_data = static_cast<const T*>(w.raw_data());
    const T* bias_data = has_bias ? static_cast<const T*>(ctx.inputs[2].raw_data()) : nullptr;
    T* out_data = out.data<T>();
    conv1d_forward_compute<T>(params, x_data, w_data, bias_data, out_data);
    return frame::Status::ok();
  });
}

}  // namespace

FRAME_REGISTER_KERNEL("conv2d", frame::kCpuBackendName, conv2d_cpu_kernel);
FRAME_REGISTER_KERNEL("conv2d_grad_input_internal", frame::kCpuBackendName,
                      conv2d_grad_input_internal_cpu_kernel);
FRAME_REGISTER_KERNEL("conv2d_grad_filter_internal", frame::kCpuBackendName,
                      conv2d_grad_filter_internal_cpu_kernel);
FRAME_REGISTER_KERNEL("conv1d", frame::kCpuBackendName, conv1d_cpu_kernel);
