// CPU 逐元素内核注册桩(add/sub/mul/div/relu/square 等)。
// 每个内核形如 Status kernel(ops::KernelContext&),内部经 dispatch_dtype 按 dtype
// 编译期展开(见 include/frame/core/dtype.h),再经 FRAME_REGISTER_KERNEL 注册到
// (op, kCpuBackendName)。add/mul/relu/square 已在本文件下方注册(M5 首批
// elementwise 四算子齐);sub/div 见下方待办标注。

// TODO(FRAME-IMPL): sub/div 属未来批次,不在 M5 首批范围(elementwise v0 四
//   算子 add/mul/relu/square 已全部注册,见 PLAN.md「M5 内置算子 v0 批次」行)。
//   参考:docs/architecture/operator-system.md 第4章;include/frame/ops/kernel_registry.h。
//   完成判据:sub/div 各自分支落地后 KernelRegistry::find("sub"/"div",
//   frame::kCpuBackendName) 均可取到内核,tests/cpp/ops/ 用例通过。

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/ops/kernel_registry.h>

namespace {

// 二元逐元素算子(add/mul)共用的计算体:v0 允许的三档浮点(fp32/fp16/bf16)
// 各自经 if constexpr 特化(编译期展开,CPP-012/ARCH-042,内层循环无运行时
// dtype 分支)。fp16/bf16 是无算术运算符的存储型 POD,借用既有位级转换
// (float16_to_float/float_to_float16 等,见 include/frame/core/dtype.h)升
// float 计算后转回;具体运算(加/乘)经 combine 参数化(REUSE-002:add/mul
// 共用本函数,禁止逐算子复制第二份同构循环骨架)。dispatch_dtype 对
// DTypeCode 全体成员做编译期穷举,故本函数也会为其余 dtype(如 int32_t/bool)
// 实例化;这些分支本体留空——binary_elementwise_cpu_kernel 在调用
// dispatch_dtype 前已拒绝这些 dtype,运行时不可达。
template <typename T, typename BinaryOp>
void apply_binary_elements(const T* lhs, const T* rhs, T* out, int64_t numel, BinaryOp combine) {
  if constexpr (std::is_same_v<T, float>) {
    for (int64_t i = 0; i < numel; ++i) out[i] = combine(lhs[i], rhs[i]);
  } else if constexpr (std::is_same_v<T, frame::float16_t>) {
    for (int64_t i = 0; i < numel; ++i) {
      const float result =
          combine(frame::float16_to_float(lhs[i]), frame::float16_to_float(rhs[i]));
      out[i] = frame::float_to_float16(result);
    }
  } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
    for (int64_t i = 0; i < numel; ++i) {
      const float result =
          combine(frame::bfloat16_to_float(lhs[i]), frame::bfloat16_to_float(rhs[i]));
      out[i] = frame::float_to_bfloat16(result);
    }
  }
}

// 二元逐元素算子(add/mul)共用的 CPU 参考实现骨架(REUSE-011:参考实现,数值
// 校验用,禁作性能路径)——朴素逐元素循环,不追求性能,唯一目标是数值正确性。
// 防御性校验:输入/输出个数、in/out 的 shape/dtype 一致、dtype 限 v0 浮点三档;
// 任一违例返回英文错误(ARCH-031 口径:不静默降级)。op_name 仅用于拼错误
// 消息,具体运算经 combine 参数化(REUSE-002:add/mul 共用本函数,禁止逐算子
// 复制第二份同构校验+分发骨架)。
template <typename BinaryOp>
frame::Status binary_elementwise_cpu_kernel(std::string_view op_name,
                                            frame::ops::KernelContext& ctx, BinaryOp combine) {
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

  // inputs 借用契约为 span<const Tensor>:只读访问经 raw_data() const 取址后
  // static_cast(与 tests/cpp/common/tolerance.h 的既有读取模式一致,REUSE-002);
  // Tensor::data<T>() 只有非 const 重载,专供 outputs(span<Tensor>)写入使用。
  const frame::Tensor& lhs = ctx.inputs[0];
  const frame::Tensor& rhs = ctx.inputs[1];
  frame::Tensor& out = ctx.outputs[0];

  const frame::DType lhs_elem_type = lhs.dtype();
  const frame::DType rhs_elem_type = rhs.dtype();
  const frame::DType out_elem_type = out.dtype();
  const bool elem_type_mismatch =
      !(lhs_elem_type == rhs_elem_type) || !(lhs_elem_type == out_elem_type);
  if (elem_type_mismatch) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cpu kernel requires lhs/rhs/out of the same dtype, got '" +
                                   std::string(lhs_elem_type.name()) + "', '" +
                                   std::string(rhs_elem_type.name()) + "', '" +
                                   std::string(out_elem_type.name()) + "'");
  }

  const bool shape_mismatch = !(lhs.shape() == rhs.shape()) || !(lhs.shape() == out.shape());
  if (shape_mismatch) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cpu kernel requires lhs/rhs/out of the same shape, got " +
                                   lhs.shape().to_string() + ", " + rhs.shape().to_string() + ", " +
                                   out.shape().to_string());
  }

  const frame::DTypeCode code = lhs_elem_type.code();
  const bool supported = (code == frame::DTypeCode::kFloat32) ||
                         (code == frame::DTypeCode::kFloat16) ||
                         (code == frame::DTypeCode::kBFloat16);
  if (!supported) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) + "' cpu kernel does not support dtype '" +
            std::string(lhs_elem_type.name()) + "' (v0 supports float32/float16/bfloat16 only)");
  }

  const int64_t numel = lhs.numel();
  if (numel == 0) return frame::Status::ok();
  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    const T* lhs_data = static_cast<const T*>(lhs.raw_data());
    const T* rhs_data = static_cast<const T*>(rhs.raw_data());
    T* out_data = out.data<T>();
    apply_binary_elements<T>(lhs_data, rhs_data, out_data, numel, combine);
    return frame::Status::ok();
  });
}

// add 的计算体(M22,批4 T3,决议点A):不复用上方 apply_binary_elements/
// binary_elementwise_cpu_kernel——二者的 dtype 白名单硬编码为 v0 浮点三档,
// mul/relu_grad_internal 仍只认这三档;而 add 是"梯度累加安全网"(同一
// indices 被多节点消费时反向引擎以 add 累加其形式零梯度),额外要支持
// int32/int64 精确整数相加(不经 float 桥接,避免大整数精度损失)。若强行给
// apply_binary_elements 加一个整数分支,dispatch_dtype 对 DTypeCode 全体成员
// 的编译期穷举机制会导致该分支对 mul/relu_grad_internal 的 float 专属 combine
// 回调也被实例化(即使运行时不可达),徒增维护负担且模糊 mul/relu_grad_internal
// "只认浮点三档"的既有契约,故 add 单独一份实现(与
// kernel_dtype_checks.h 头注释"matmul.cpp 三具名形参版本形态不同暂不合并"
// 同一先例)。float 三档经既有位级转换升 float 计算后转回(与
// apply_binary_elements 同一约定);int32_t/int64_t 直接整数相加。
// dispatch_dtype 对 DTypeCode 全体成员编译期穷举,其余 dtype(如 bool)分支
// 本体留空,同款理由见 apply_binary_elements 头注释。
template <typename T>
void apply_add_elements(const T* lhs, const T* rhs, T* out, int64_t numel) {
  if constexpr (std::is_same_v<T, float>) {
    for (int64_t i = 0; i < numel; ++i) out[i] = lhs[i] + rhs[i];
  } else if constexpr (std::is_same_v<T, frame::float16_t>) {
    for (int64_t i = 0; i < numel; ++i) {
      const float result = frame::float16_to_float(lhs[i]) + frame::float16_to_float(rhs[i]);
      out[i] = frame::float_to_float16(result);
    }
  } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
    for (int64_t i = 0; i < numel; ++i) {
      const float result = frame::bfloat16_to_float(lhs[i]) + frame::bfloat16_to_float(rhs[i]);
      out[i] = frame::float_to_bfloat16(result);
    }
  } else if constexpr (std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::int64_t>) {
    for (int64_t i = 0; i < numel; ++i) out[i] = lhs[i] + rhs[i];
  }
}

// add 的 CPU 参考实现:防御性校验逐条与 binary_elementwise_cpu_kernel 同构
// (输入/输出个数、lhs/rhs/out 的 shape/dtype 一致),仅 dtype 白名单扩至
// float32/float16/bfloat16/int32/int64(REUSE-011:参考实现,数值校验用,禁作
// 性能路径)。
frame::Status add_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'add' cpu kernel expects 2 inputs, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'add' cpu kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& lhs = ctx.inputs[0];
  const frame::Tensor& rhs = ctx.inputs[1];
  frame::Tensor& out = ctx.outputs[0];

  const frame::DType lhs_elem_type = lhs.dtype();
  const frame::DType rhs_elem_type = rhs.dtype();
  const frame::DType out_elem_type = out.dtype();
  const bool elem_type_mismatch =
      !(lhs_elem_type == rhs_elem_type) || !(lhs_elem_type == out_elem_type);
  if (elem_type_mismatch) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'add' cpu kernel requires lhs/rhs/out of the same dtype, got '" +
                                   std::string(lhs_elem_type.name()) + "', '" +
                                   std::string(rhs_elem_type.name()) + "', '" +
                                   std::string(out_elem_type.name()) + "'");
  }

  const bool shape_mismatch = !(lhs.shape() == rhs.shape()) || !(lhs.shape() == out.shape());
  if (shape_mismatch) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'add' cpu kernel requires lhs/rhs/out of the same shape, got " +
                                   lhs.shape().to_string() + ", " + rhs.shape().to_string() + ", " +
                                   out.shape().to_string());
  }

  const frame::DTypeCode code = lhs_elem_type.code();
  const bool supported = (code == frame::DTypeCode::kFloat32) ||
                         (code == frame::DTypeCode::kFloat16) ||
                         (code == frame::DTypeCode::kBFloat16) ||
                         (code == frame::DTypeCode::kInt32) || (code == frame::DTypeCode::kInt64);
  if (!supported) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'add' cpu kernel does not support dtype '" +
                                   std::string(lhs_elem_type.name()) +
                                   "' (v0 supports float32/float16/bfloat16/int32/int64 only)");
  }

  const int64_t numel = lhs.numel();
  if (numel == 0) return frame::Status::ok();
  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    const T* lhs_data = static_cast<const T*>(lhs.raw_data());
    const T* rhs_data = static_cast<const T*>(rhs.raw_data());
    T* out_data = out.data<T>();
    apply_add_elements<T>(lhs_data, rhs_data, out_data, numel);
    return frame::Status::ok();
  });
}

frame::Status mul_cpu_kernel(frame::ops::KernelContext& ctx) {
  return binary_elementwise_cpu_kernel("mul", ctx, [](float lhs, float rhs) { return lhs * rhs; });
}

// 一元逐元素算子(relu 等)共用的计算体:与二元版 apply_binary_elements 并列
// 而非合并——作用对象数量不同(1 个输入张量 vs 2 个),强行合一需要额外的
// 数量判断,反而掩盖循环体各自的语义,不属于 REUSE-002 意图消灭的"同构复制"。
// 三档浮点分支写法与二元版同构(if constexpr 编译期展开,CPP-012/ARCH-042,
// 内层循环无运行时 dtype 分支);fp16/bf16 借用同一套位级转换升 float 计算后
// 转回。dispatch_dtype 对 DTypeCode 全体成员做编译期穷举,故本函数也会为
// 其余 dtype(如 int32_t/bool)实例化;这些分支本体留空——
// unary_elementwise_cpu_kernel 在调用 dispatch_dtype 前已拒绝这些 dtype,
// 运行时不可达。
template <typename T, typename UnaryOp>
void apply_unary_elements(const T* in, T* out, int64_t numel, UnaryOp combine) {
  if constexpr (std::is_same_v<T, float>) {
    for (int64_t i = 0; i < numel; ++i) out[i] = combine(in[i]);
  } else if constexpr (std::is_same_v<T, frame::float16_t>) {
    for (int64_t i = 0; i < numel; ++i) {
      out[i] = frame::float_to_float16(combine(frame::float16_to_float(in[i])));
    }
  } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
    for (int64_t i = 0; i < numel; ++i) {
      out[i] = frame::float_to_bfloat16(combine(frame::bfloat16_to_float(in[i])));
    }
  }
}

// 一元逐元素算子(relu 等)共用的 CPU 参考实现骨架(REUSE-011:参考实现,
// 数值校验用,禁作性能路径)——朴素逐元素循环,不追求性能,唯一目标是数值
// 正确性。防御性校验逐条与二元版 binary_elementwise_cpu_kernel 同构(输入/
// 输出个数、in/out 的 shape/dtype 一致、dtype 限 v0 浮点三档),但作用对象
// 数量不同(1 个输入 tensor vs 2 个)使得输入个数校验、错误消息模板均不同,
// 未强行合并成一份参数化函数(会引入"按输入个数分支"掩盖两条路径各自结构,
// 不属于 REUSE-002 意图消灭的同构复制)。op_name 仅用于拼错误消息,具体运算
// 经 combine 参数化(REUSE-002:relu 等一元算子共用本函数)。
template <typename UnaryOp>
frame::Status unary_elementwise_cpu_kernel(std::string_view op_name, frame::ops::KernelContext& ctx,
                                           UnaryOp combine) {
  if (ctx.inputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cpu kernel expects 1 input, got " +
                                   std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cpu kernel expects 1 output, got " +
                                   std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& in = ctx.inputs[0];
  frame::Tensor& out = ctx.outputs[0];

  const frame::DType in_elem_type = in.dtype();
  const frame::DType out_elem_type = out.dtype();
  const bool elem_type_mismatch = !(in_elem_type == out_elem_type);
  if (elem_type_mismatch) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) + "' cpu kernel requires x/out of the same dtype, got '" +
            std::string(in_elem_type.name()) + "', '" + std::string(out_elem_type.name()) + "'");
  }

  const bool shape_mismatch = !(in.shape() == out.shape());
  if (shape_mismatch) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cpu kernel requires x/out of the same shape, got " +
                                   in.shape().to_string() + ", " + out.shape().to_string());
  }

  const frame::DTypeCode code = in_elem_type.code();
  const bool supported = (code == frame::DTypeCode::kFloat32) ||
                         (code == frame::DTypeCode::kFloat16) ||
                         (code == frame::DTypeCode::kBFloat16);
  if (!supported) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) + "' cpu kernel does not support dtype '" +
            std::string(in_elem_type.name()) + "' (v0 supports float32/float16/bfloat16 only)");
  }

  const int64_t numel = in.numel();
  if (numel == 0) return frame::Status::ok();
  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    const T* in_data = static_cast<const T*>(in.raw_data());
    T* out_data = out.data<T>();
    apply_unary_elements<T>(in_data, out_data, numel, combine);
    return frame::Status::ok();
  });
}

// relu(x) = max(x, 0):直接调用 std::max<float>,与 combine 的行为定义为
// "与 std::max(x, 0.f) 完全一致"——含 NaN 传播语义(std::max 的典型实现为
// `(a < b) ? b : a`,NaN 与任何值比较恒为 false,故 std::max(NaN, 0.f) 走
// else 分支返回第一个参数 NaN;relu(NaN) = NaN,不做额外的 NaN 判断/清洗)
// 与 -0.0 输入的符号位处理(std::max(-0.0f, 0.0f) 同理返回第一个参数 -0.0f,
// 该 IEEE-754 边界值在实现选择上不敏感,产出 +0.0 或 -0.0 均视为合规)。
frame::Status relu_cpu_kernel(frame::ops::KernelContext& ctx) {
  return unary_elementwise_cpu_kernel("relu", ctx, [](float x) { return std::max(x, 0.0F); });
}

// square(x) = x * x。ARCH-041 要求 kernel 与 decomposition 成对存在时二者不
// 互相替代(见 src/ops/schemas/elementwise.cpp::square_decompose 头注释);
// 本 kernel 是 square 的真实执行路径,decomposition 仅是 M10 回退链素材。
// 复用一元骨架(REUSE-011 标注已在 unary_elementwise_cpu_kernel 入口处,无需
// 重复)。
frame::Status square_cpu_kernel(frame::ops::KernelContext& ctx) {
  return unary_elementwise_cpu_kernel("square", ctx, [](float x) { return x * x; });
}

// sigmoid(x) = 1/(1+e^-x)(M21,批3 T4)。数值稳定式:x>=0 用 1/(1+e^-x)
// (e^-x 在 x>=0 时不会上溢);x<0 改用 e^x/(1+e^x)(e^x 在 x<0 时不会上溢,
// 与直接算 1/(1+e^-x) 相比避免 x 很负时 e^-x 上溢为 inf)。复用一元骨架
// (REUSE-011 标注已在 unary_elementwise_cpu_kernel 入口处,无需重复)。
frame::Status sigmoid_cpu_kernel(frame::ops::KernelContext& ctx) {
  return unary_elementwise_cpu_kernel("sigmoid", ctx, [](float x) {
    if (x >= 0.0F) {
      return 1.0F / (1.0F + std::exp(-x));
    }
    const float exp_x = std::exp(x);
    return exp_x / (1.0F + exp_x);
  });
}

// 从 kernel attrs 读取代理阶跃必需的正有限 alpha,供 CPU 边界 fail-loud。
frame::Result<double> read_heaviside_alpha(const frame::ops::KernelContext& ctx,
                                           std::string_view backend) {
  if (ctx.attrs == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'heaviside_surrogate' " + std::string(backend) +
            " kernel is missing required attribute 'alpha': no attrs provided");
  }
  const auto alpha_it = ctx.attrs->find("alpha");
  if (alpha_it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'heaviside_surrogate' " + std::string(backend) +
                                   " kernel is missing required attribute 'alpha'");
  }
  const double* alpha = std::get_if<double>(&alpha_it->second);
  if (alpha == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'heaviside_surrogate' " + std::string(backend) +
                                   " kernel attribute 'alpha' has the wrong type, expected double");
  }
  if (!std::isfinite(*alpha) || !(*alpha > 0.0)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'heaviside_surrogate' " + std::string(backend) +
                                   " kernel attribute 'alpha' must be finite and positive, got " +
                                   std::to_string(*alpha));
  }
  return *alpha;
}

// 代理阶跃 CPU 参考实现:alpha 仅定义反向平滑度,前向固定 x>=0 ? 1 : 0。
frame::Status heaviside_surrogate_cpu_kernel(frame::ops::KernelContext& ctx) {
  const frame::Result<double> alpha = read_heaviside_alpha(ctx, "cpu");
  if (!alpha.is_ok()) return alpha.status();
  return unary_elementwise_cpu_kernel("heaviside_surrogate", ctx,
                                      [](float x) { return x >= 0.0F ? 1.0F : 0.0F; });
}

// tanh(x)(M22,批4 T3)。复用一元骨架(REUSE-011 标注已在
// unary_elementwise_cpu_kernel 入口处,无需重复)。
frame::Status tanh_cpu_kernel(frame::ops::KernelContext& ctx) {
  return unary_elementwise_cpu_kernel("tanh", ctx, [](float x) { return std::tanh(x); });
}

// rsqrt(x) = x^(-1/2) = 1/sqrt(x)(M22,批4 T3,spec 外增项,见
// src/ops/schemas/elementwise.cpp::rsqrt_gradient 头注释)。复用一元骨架
// (REUSE-011 标注已在 unary_elementwise_cpu_kernel 入口处,无需重复)。
frame::Status rsqrt_cpu_kernel(frame::ops::KernelContext& ctx) {
  return unary_elementwise_cpu_kernel("rsqrt", ctx, [](float x) { return 1.0F / std::sqrt(x); });
}

// relu_grad_internal(x, gy) = x>0 处透传 gy,余 0(M17,relu 的梯度)。二元
// 逐元素,复用 binary_elementwise_cpu_kernel 这一份骨架(REUSE-002,同
// mul_cpu_kernel 先例;add_cpu_kernel 因 dtype 白名单扩容已改自成一份实现,
// 见其头注释)。
frame::Status relu_grad_internal_cpu_kernel(frame::ops::KernelContext& ctx) {
  return binary_elementwise_cpu_kernel("relu_grad_internal", ctx,
                                       [](float x, float gy) { return x > 0.0F ? gy : 0.0F; });
}

}  // namespace

FRAME_REGISTER_KERNEL("add", frame::kCpuBackendName, add_cpu_kernel);
FRAME_REGISTER_KERNEL("mul", frame::kCpuBackendName, mul_cpu_kernel);
FRAME_REGISTER_KERNEL("relu", frame::kCpuBackendName, relu_cpu_kernel);
FRAME_REGISTER_KERNEL("square", frame::kCpuBackendName, square_cpu_kernel);
FRAME_REGISTER_KERNEL("sigmoid", frame::kCpuBackendName, sigmoid_cpu_kernel);
FRAME_REGISTER_KERNEL("heaviside_surrogate", frame::kCpuBackendName,
                      heaviside_surrogate_cpu_kernel);
FRAME_REGISTER_KERNEL("tanh", frame::kCpuBackendName, tanh_cpu_kernel);
FRAME_REGISTER_KERNEL("rsqrt", frame::kCpuBackendName, rsqrt_cpu_kernel);
FRAME_REGISTER_KERNEL("relu_grad_internal", frame::kCpuBackendName, relu_grad_internal_cpu_kernel);
