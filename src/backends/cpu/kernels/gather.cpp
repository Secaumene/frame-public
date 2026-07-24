// CPU gather/scatter-add 内核(M22 gather,M28 scatter_add):gather、
// scatter_add、gather_grad_internal。内核形如 Status kernel(ops::KernelContext&),内部经嵌套
// dispatch_dtype 按 value dtype(浮点三档)与 indices dtype(int32/int64)分别
// 编译期展开(CPP-012:入口分派,内层循环零 dtype 分支),再经
// FRAME_REGISTER_KERNEL 注册到 (op, kCpuBackendName)。索引值域属运行时,
// 静态图无法静态校验张量值;本文件逐元素校验 0<=idx<N,越界返回
// kInvalidArgument(消息含越界值与界,ARCH-031:拒绝 device 侧静默 clamp)。

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
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

// gather/gather_grad_internal 共用:校验 indices 张量 dtype 属 int32/int64
// 二档(唯一整数张量消费者,§1.1 决议点A),返回其 DTypeCode。
frame::Result<frame::DTypeCode> require_index_dtype(std::string_view op_name,
                                                    const frame::Tensor& indices) {
  const frame::DTypeCode code = indices.dtype().code();
  const bool supported = code == frame::DTypeCode::kInt32 || code == frame::DTypeCode::kInt64;
  if (!supported) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cpu kernel requires indices dtype to be int32 or int64, "
                                   "got '" +
                                   std::string(indices.dtype().name()) + "'");
  }
  return code;
}

// gather(x[N,F], indices[K]) 的 CPU 参考实现(REUSE-011:参考实现,数值校验
// 用,禁作性能路径)——逐行拷贝:out[k,:] = x[indices[k],:]。逐元素校验
// 0<=idx<N,越界返回 kInvalidArgument(消息含越界值与界)。
frame::Status gather_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'gather' cpu kernel expects 2 inputs, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'gather' cpu kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& x = ctx.inputs[0];
  const frame::Tensor& indices = ctx.inputs[1];
  frame::Tensor& out = ctx.outputs[0];

  if (x.shape().rank() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'gather' cpu kernel requires x to be rank-2 [N, F], got rank " +
                                   std::to_string(x.shape().rank()));
  }
  if (indices.shape().rank() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'gather' cpu kernel requires indices to be rank-1 [K], got rank " +
            std::to_string(indices.shape().rank()));
  }

  const frame::Result<frame::DTypeCode> value_code_result =
      frame::backends::cpu::require_matching_supported_dtype("gather", "x/out", {&x, &out});
  if (!value_code_result.is_ok()) return value_code_result.status();
  const frame::DTypeCode value_code = value_code_result.value();

  const frame::Result<frame::DTypeCode> index_code_result = require_index_dtype("gather", indices);
  if (!index_code_result.is_ok()) return index_code_result.status();
  const frame::DTypeCode index_code = index_code_result.value();

  const int64_t n = x.shape().dim(0);
  const int64_t f = x.shape().dim(1);
  const int64_t k = indices.shape().dim(0);
  if (n < 0 || f < 0 || k < 0 || (f != 0 && k > std::numeric_limits<int64_t>::max() / f)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'gather' cpu kernel shape element count overflows int64 or "
                               "contains a negative dimension");
  }
  const frame::Shape expected_out_shape({k, f});
  if (!(out.shape() == expected_out_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'gather' cpu kernel requires out shape to match [K, F], got " +
                                   out.shape().to_string() + ", expected " +
                                   expected_out_shape.to_string());
  }

  return frame::dispatch_dtype(value_code, [&]<typename T>() -> frame::Status {
    const T* x_data = static_cast<const T*>(x.raw_data());
    T* out_data = out.data<T>();
    return frame::dispatch_dtype(index_code, [&]<typename IdxT>() -> frame::Status {
      // 仅 int32_t/int64_t 有意义(indices dtype 已在上方校验为该二档之一);
      // 其余 IdxT(如 float16_t,无到 int64_t 的转换算子)分支本体留空,同款
      // 理由见 kernel_dtype_checks.h 头注释——dispatch_dtype 对全体 DTypeCode
      // 编译期穷举,运行时不可达。
      if constexpr (std::is_same_v<IdxT, std::int32_t> || std::is_same_v<IdxT, std::int64_t>) {
        const IdxT* idx_data = static_cast<const IdxT*>(indices.raw_data());
        for (int64_t row = 0; row < k; ++row) {
          const int64_t idx = static_cast<int64_t>(idx_data[row]);
          if (idx < 0 || idx >= n) {
            return frame::Status::make(
                frame::ErrorCode::kInvalidArgument,
                "op 'gather' cpu kernel indices[" + std::to_string(row) +
                    "]=" + std::to_string(idx) +
                    " is out of range for x's first dimension N=" + std::to_string(n));
          }
          for (int64_t col = 0; col < f; ++col) {
            out_data[row * f + col] = x_data[idx * f + col];
          }
        }
      }
      return frame::Status::ok();
    });
  });
}

// scatter-add 族 CPU 参考实现的共享核心(REUSE-011):先用 float 缓冲清零并
// 累加,再一次性转回输出 dtype。公共 scatter_add 与旧 internal 仅参数化名称、
// 输入输出角色、shape 属性及首维符号,散加循环保持单份。
struct ScatterAddCpuContract {
  std::string_view op_name;
  std::string_view values_role;
  std::string_view output_role;
  std::string_view dtype_role;
  std::string_view shape_attr_name;
  std::string_view consistency_role;
  std::string_view range_output_role;
  std::string_view output_dim_name;
};

frame::Status scatter_add_cpu_kernel_impl(frame::ops::KernelContext& ctx,
                                          const ScatterAddCpuContract& contract) {
  const std::string_view op_name = contract.op_name;
  const std::string_view values_role = contract.values_role;
  const std::string_view output_role = contract.output_role;
  const std::string_view dtype_role = contract.dtype_role;
  const std::string_view shape_attr_name = contract.shape_attr_name;
  const std::string_view consistency_role = contract.consistency_role;
  const std::string_view range_output_role = contract.range_output_role;
  const std::string_view output_dim_name = contract.output_dim_name;
  if (ctx.inputs.size() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cpu kernel expects 2 inputs, got " +
                                   std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cpu kernel expects 1 output, got " +
                                   std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& values = ctx.inputs[0];
  const frame::Tensor& indices = ctx.inputs[1];
  frame::Tensor& output = ctx.outputs[0];

  if (values.shape().rank() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' cpu kernel requires " +
                                   std::string(values_role) + " to be rank-2 [K, F], got rank " +
                                   std::to_string(values.shape().rank()));
  }
  if (indices.shape().rank() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cpu kernel requires indices to be rank-1 [K], got rank " +
                                   std::to_string(indices.shape().rank()));
  }
  if (output.shape().rank() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' cpu kernel requires " +
                                   std::string(output_role) + " to be rank-2 [" +
                                   std::string(output_dim_name) + ", F], got rank " +
                                   std::to_string(output.shape().rank()));
  }

  const frame::Result<frame::DTypeCode> value_code_result =
      frame::backends::cpu::require_matching_supported_dtype(op_name, dtype_role,
                                                             {&values, &output});
  if (!value_code_result.is_ok()) return value_code_result.status();
  const frame::DTypeCode value_code = value_code_result.value();

  const frame::Result<frame::DTypeCode> index_code_result = require_index_dtype(op_name, indices);
  if (!index_code_result.is_ok()) return index_code_result.status();
  const frame::DTypeCode index_code = index_code_result.value();

  // shape 属性与输出实际 shape 的一致性防御性复核。
  if (ctx.attrs == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cpu kernel is missing required attribute '" +
                                   std::string(shape_attr_name) + "': no attrs provided");
  }
  const auto output_shape_it = ctx.attrs->find(std::string(shape_attr_name));
  if (output_shape_it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cpu kernel is missing required attribute '" +
                                   std::string(shape_attr_name) + "'");
  }
  const frame::Shape* output_shape_ptr = std::get_if<frame::Shape>(&output_shape_it->second);
  if (output_shape_ptr == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' cpu kernel attribute '" +
                                   std::string(shape_attr_name) +
                                   "' has the wrong type, expected shape");
  }
  if (!(output.shape() == *output_shape_ptr)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) + "' cpu kernel requires " + std::string(output_role) +
            " shape to match attribute '" + std::string(shape_attr_name) + "', got " +
            output.shape().to_string() + ", expected " + output_shape_ptr->to_string());
  }

  const int64_t n = output.shape().dim(0);
  const int64_t f = output.shape().dim(1);
  const int64_t k = indices.shape().dim(0);
  if (n < 0 || f < 0 || k < 0 ||
      (f != 0 && (n > std::numeric_limits<int64_t>::max() / f ||
                  k > std::numeric_limits<int64_t>::max() / f))) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cpu kernel shape element count overflows int64 or contains "
                                   "a negative dimension");
  }
  const frame::Shape expected_values_shape({k, f});
  if (!(values.shape() == expected_values_shape)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) + "' cpu kernel requires " + std::string(values_role) +
            " shape to be consistent with " + std::string(consistency_role) + ", got " +
            values.shape().to_string() + ", expected " + expected_values_shape.to_string());
  }

  const int64_t output_numel = output.numel();
  // K=0 时无需取得空 values/indices 的数据指针；V/F 非空则只写零输出，
  // V=0 或 F=0 则直接成功。该分支同时服务公共 scatter_add 与旧梯度 wrapper。
  if (k == 0) {
    if (output_numel == 0) return frame::Status::ok();
    return frame::dispatch_dtype(value_code, [&]<typename T>() -> frame::Status {
      T* output_data = output.data<T>();
      for (int64_t i = 0; i < output_numel; ++i) {
        output_data[i] = from_accum<T>(0.0F);
      }
      return frame::Status::ok();
    });
  }
  return frame::dispatch_dtype(value_code, [&]<typename T>() -> frame::Status {
    const T* values_data = static_cast<const T*>(values.raw_data());
    T* output_data = output.data<T>();
    std::vector<float> accum(static_cast<size_t>(output_numel), 0.0F);

    // 非 const:允许 return 时自动移动(performance-no-automatic-move,与
    // src/backends/cpu/kernels/reduction.cpp::sum_grad_internal_cpu_kernel
    // 同款理由)。
    frame::Status scatter_status =
        frame::dispatch_dtype(index_code, [&]<typename IdxT>() -> frame::Status {
          if constexpr (std::is_same_v<IdxT, std::int32_t> || std::is_same_v<IdxT, std::int64_t>) {
            const IdxT* idx_data = static_cast<const IdxT*>(indices.raw_data());
            for (int64_t row = 0; row < k; ++row) {
              const int64_t idx = static_cast<int64_t>(idx_data[row]);
              if (idx < 0 || idx >= n) {
                return frame::Status::make(
                    frame::ErrorCode::kInvalidArgument,
                    "op '" + std::string(op_name) + "' cpu kernel indices[" + std::to_string(row) +
                        "]=" + std::to_string(idx) + " is out of range for " +
                        std::string(range_output_role) + "'s first dimension " +
                        std::string(output_dim_name) + "=" + std::to_string(n));
              }
              for (int64_t col = 0; col < f; ++col) {
                accum[static_cast<size_t>(idx * f + col)] +=
                    to_accum<T>(values_data[row * f + col]);
              }
            }
          }
          return frame::Status::ok();
        });
    if (!scatter_status.is_ok()) return scatter_status;

    for (int64_t i = 0; i < output_numel; ++i) {
      output_data[i] = from_accum<T>(accum[static_cast<size_t>(i)]);
    }
    return frame::Status::ok();
  });
}

// 旧 internal CPU kernel 的兼容薄 wrapper:input_shape、诊断和行为保持不变。
frame::Status gather_grad_internal_cpu_kernel(frame::ops::KernelContext& ctx) {
  constexpr ScatterAddCpuContract kContract{.op_name = "gather_grad_internal",
                                            .values_role = "gy",
                                            .output_role = "gx(out)",
                                            .dtype_role = "gy/gx(out)",
                                            .shape_attr_name = "input_shape",
                                            .consistency_role = "indices/gx(out)",
                                            .range_output_role = "gx",
                                            .output_dim_name = "N"};
  return scatter_add_cpu_kernel_impl(ctx, kContract);
}

// 公共 scatter_add CPU kernel 的薄 wrapper。
frame::Status public_scatter_add_cpu_kernel(frame::ops::KernelContext& ctx) {
  constexpr ScatterAddCpuContract kContract{.op_name = "scatter_add",
                                            .values_role = "updates",
                                            .output_role = "out",
                                            .dtype_role = "updates/out",
                                            .shape_attr_name = "output_shape",
                                            .consistency_role = "indices/out",
                                            .range_output_role = "out",
                                            .output_dim_name = "V"};
  return scatter_add_cpu_kernel_impl(ctx, kContract);
}

}  // namespace

FRAME_REGISTER_KERNEL("gather", frame::kCpuBackendName, gather_cpu_kernel);
FRAME_REGISTER_KERNEL("scatter_add", frame::kCpuBackendName, public_scatter_add_cpu_kernel);
FRAME_REGISTER_KERNEL("gather_grad_internal", frame::kCpuBackendName,
                      gather_grad_internal_cpu_kernel);
