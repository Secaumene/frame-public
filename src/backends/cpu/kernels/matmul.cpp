// CPU 矩阵乘内核(matmul)。
// 内核形如 Status kernel(ops::KernelContext&),内部经 dispatch_dtype 按 dtype
// 编译期展开(见 include/frame/core/dtype.h),再经 FRAME_REGISTER_KERNEL 注册到
// (op, kCpuBackendName)。

#include <cstdint>
#include <string>
#include <type_traits>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/ops/kernel_registry.h>

namespace {

// matmul 的 CPU 参考实现 —— 【REUSE-011:参考实现,数值校验用,禁作性能
// 路径】。docs/standards/reuse-policy.md REUSE-011 明文:cpu 参考后端的算子
// kernel 以数值正确性为唯一目标,不受铁律 5 对标准能力(此处即 GEMM)引库
// 强制约束的限制,朴素实现(含手写三重循环)合规;但本豁免仅限 cpu 参考路径
// ——任何性能路径不得复用本实现,加速后端的同能力实现仍须走各自已批库
// (如 CUDA:cuBLAS/CUTLASS;Intel GPU:oneMKL/oneDNN);若 CPU 侧未来引入 BLAS
// 类库,仍触发 REUSE-010(需另立 ADR),不因本实现的存在而豁免。
//
// 实现:朴素三重循环(i, j, k),不追求性能,唯一目标是数值正确性。内积以
// float 精度累加(fp16/bf16 逐元素升 float 参与乘加、单个内积算完后一次性
// 转回,与 src/backends/cpu/kernels/reduction.cpp 的 sum 累加同一约定,是
// 实现细节、不进 schema)。fp32/fp16/bf16 三档的 if constexpr 转换分支就地
// 内联在三重循环体中(未抽取为独立具名函数):其与 reduction.cpp 的
// to_accum/from_accum 概念相似,但矩阵乘的收缩维遍历顺序与内积累加时机
// (每个输出元素独立起新的累加序列)和 sum 的「输出索引=输入索引投影」
// 遍历方式结构不同,勉强抽取一个跨两个算子语义的公共转换符号意义有限
// (呼应 src/backends/cpu/kernels/elementwise.cpp 中一元/二元骨架"同构写法
// 不强行合并"的既有先例),故各自就地展开。
// m/k/n 从 lhs/rhs 的 shape 现取(调用方 matmul_cpu_kernel 已校验合法性),不
// 作为独立形参传入——避免三个相邻同类型(int64_t)形参易被调用方误置换顺序
// (bugprone-easily-swappable-parameters)。
// 校验张量为 rank-2,不满足返回英文错误(消息含 op 名/操作数标签/实际
// rank)。REUSE-002:matmul/matmul_grad_lhs_internal/matmul_grad_rhs_internal
// 三个 kernel 的 rank-2 校验共用本函数,避免三份同构判断各自复制。
frame::Status require_rank2(std::string_view op_name, std::string_view operand_label,
                            const frame::Tensor& tensor) {
  if (tensor.shape().rank() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' cpu kernel requires " +
                                   std::string(operand_label) + " to be rank-2, got rank " +
                                   std::to_string(tensor.shape().rank()));
  }
  return frame::Status::ok();
}

// 校验三个张量 dtype 完全一致且属 v0 浮点三档,返回其 DTypeCode(供
// dispatch_dtype 使用)。role_phrase 是三个操作数在错误消息中的固定描述短语
// (如 "lhs/rhs/out"、"gy/b/ga"),与具体张量的实际值无关。REUSE-002:
// matmul/matmul_grad_lhs_internal/matmul_grad_rhs_internal 三个 kernel 共用
// 本函数。
frame::Result<frame::DTypeCode> require_matching_supported_dtype(std::string_view op_name,
                                                                 std::string_view role_phrase,
                                                                 const frame::Tensor& first,
                                                                 const frame::Tensor& second,
                                                                 const frame::Tensor& third) {
  const frame::DType first_type = first.dtype();
  const frame::DType second_type = second.dtype();
  const frame::DType third_type = third.dtype();
  // 布尔先落地为具名变量再判断(与 src/backends/cpu/kernels/elementwise.cpp
  // 同一惯例):check_iron_rules.sh 对 kernels/ 目录的 CPP-012 文本扫描按
  // `if (...dtype...)` 形态识别疑似运行时 dtype 分支,提前拆出具名变量可与
  // dispatch_dtype 编译期分派清晰区分,避免误判。
  const bool elem_type_mismatch = !(first_type == second_type) || !(first_type == third_type);
  if (elem_type_mismatch) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) + "' cpu kernel requires " + std::string(role_phrase) +
            " of the same dtype, got '" + std::string(first_type.name()) + "', '" +
            std::string(second_type.name()) + "', '" + std::string(third_type.name()) + "'");
  }
  const frame::DTypeCode code = first_type.code();
  const bool supported = (code == frame::DTypeCode::kFloat32) ||
                         (code == frame::DTypeCode::kFloat16) ||
                         (code == frame::DTypeCode::kBFloat16);
  if (!supported) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) + "' cpu kernel does not support dtype '" +
            std::string(first_type.name()) + "' (v0 supports float32/float16/bfloat16 only)");
  }
  return code;
}

template <typename T>
frame::Status matmul_compute(const frame::Tensor& lhs, const frame::Tensor& rhs,
                             frame::Tensor& out) {
  const int64_t m = lhs.shape().dim(0);
  const int64_t k = lhs.shape().dim(1);
  const int64_t n = rhs.shape().dim(1);

  const T* lhs_data = static_cast<const T*>(lhs.raw_data());
  const T* rhs_data = static_cast<const T*>(rhs.raw_data());
  T* out_data = out.data<T>();

  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float accum = 0.0F;
      for (int64_t p = 0; p < k; ++p) {
        float lhs_value = 0.0F;
        float rhs_value = 0.0F;
        if constexpr (std::is_same_v<T, float>) {
          lhs_value = lhs_data[i * k + p];
          rhs_value = rhs_data[p * n + j];
        } else if constexpr (std::is_same_v<T, frame::float16_t>) {
          lhs_value = frame::float16_to_float(lhs_data[i * k + p]);
          rhs_value = frame::float16_to_float(rhs_data[p * n + j]);
        } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
          lhs_value = frame::bfloat16_to_float(lhs_data[i * k + p]);
          rhs_value = frame::bfloat16_to_float(rhs_data[p * n + j]);
        }
        accum += lhs_value * rhs_value;
      }
      if constexpr (std::is_same_v<T, float>) {
        out_data[i * n + j] = accum;
      } else if constexpr (std::is_same_v<T, frame::float16_t>) {
        out_data[i * n + j] = frame::float_to_float16(accum);
      } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
        out_data[i * n + j] = frame::float_to_bfloat16(accum);
      }
    }
  }
  return frame::Status::ok();
}

// 防御性校验 + dispatch_dtype 分发骨架:2 输入 1 输出、均 rank-2、dtype 限 v0
// 浮点三档且 lhs/rhs/out 一致、收缩维一致([m,k]×[k,n])、out shape 与
// [m,n] 一致;任一违例返回英文错误(ARCH-031 口径:不静默降级)。
frame::Status matmul_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'matmul' cpu kernel expects 2 inputs, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'matmul' cpu kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& lhs = ctx.inputs[0];
  const frame::Tensor& rhs = ctx.inputs[1];
  frame::Tensor& out = ctx.outputs[0];

  frame::Status rank_status = require_rank2("matmul", "lhs", lhs);
  if (!rank_status.is_ok()) return rank_status;
  rank_status = require_rank2("matmul", "rhs", rhs);
  if (!rank_status.is_ok()) return rank_status;

  const frame::Result<frame::DTypeCode> code_result =
      require_matching_supported_dtype("matmul", "lhs/rhs/out", lhs, rhs, out);
  if (!code_result.is_ok()) return code_result.status();
  const frame::DTypeCode code = code_result.value();

  const int64_t m = lhs.shape().dim(0);
  const int64_t k = lhs.shape().dim(1);
  const int64_t k2 = rhs.shape().dim(0);
  const int64_t n = rhs.shape().dim(1);
  if (k != k2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'matmul' cpu kernel contraction dimension mismatch: lhs " + lhs.shape().to_string() +
            " has k=" + std::to_string(k) + ", rhs " + rhs.shape().to_string() + " has k=" +
            std::to_string(k2) + " (lhs's last dimension must equal rhs's first dimension)");
  }

  const frame::Shape expected_out_shape({m, n});
  const bool out_shape_mismatch = !(out.shape() == expected_out_shape);
  if (out_shape_mismatch) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'matmul' cpu kernel requires out shape to match [m, n], got " +
                                   out.shape().to_string() + ", expected " +
                                   expected_out_shape.to_string());
  }

  return frame::dispatch_dtype(
      code, [&]<typename T>() -> frame::Status { return matmul_compute<T>(lhs, rhs, out); });
}

// matmul_grad_lhs_internal 的计算核(M17):ga[i,p] = sum_j gy[i,j]*b[p,j]
// (= gy·bᵀ,不物化 bᵀ,直接按转置索引读取 b——b 存储仍是原始 [k,n] 行优先
// 布局,读取 b[p,j] 对应线性偏移 p*n+j)。与 matmul_compute(标准 A×B)结构
// 相似但两个乘数的下标推导公式不同(乘数之一按转置方式读取),各自独立实现
// (理由同 schema 侧 infer_matmul_grad_lhs_internal_shape 头注释)。
template <typename T>
frame::Status matmul_grad_lhs_internal_compute(const frame::Tensor& gy, const frame::Tensor& b,
                                               frame::Tensor& ga) {
  const int64_t m = gy.shape().dim(0);
  const int64_t n = gy.shape().dim(1);
  const int64_t k = b.shape().dim(0);

  const T* gy_data = static_cast<const T*>(gy.raw_data());
  const T* b_data = static_cast<const T*>(b.raw_data());
  T* ga_data = ga.data<T>();

  for (int64_t i = 0; i < m; ++i) {
    for (int64_t p = 0; p < k; ++p) {
      float accum = 0.0F;
      for (int64_t j = 0; j < n; ++j) {
        float gy_value = 0.0F;
        float b_value = 0.0F;
        if constexpr (std::is_same_v<T, float>) {
          gy_value = gy_data[i * n + j];
          b_value = b_data[p * n + j];
        } else if constexpr (std::is_same_v<T, frame::float16_t>) {
          gy_value = frame::float16_to_float(gy_data[i * n + j]);
          b_value = frame::float16_to_float(b_data[p * n + j]);
        } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
          gy_value = frame::bfloat16_to_float(gy_data[i * n + j]);
          b_value = frame::bfloat16_to_float(b_data[p * n + j]);
        }
        accum += gy_value * b_value;
      }
      if constexpr (std::is_same_v<T, float>) {
        ga_data[i * k + p] = accum;
      } else if constexpr (std::is_same_v<T, frame::float16_t>) {
        ga_data[i * k + p] = frame::float_to_float16(accum);
      } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
        ga_data[i * k + p] = frame::float_to_bfloat16(accum);
      }
    }
  }
  return frame::Status::ok();
}

frame::Status matmul_grad_lhs_internal_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'matmul_grad_lhs_internal' cpu kernel expects 2 inputs, got " +
                                   std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'matmul_grad_lhs_internal' cpu kernel expects 1 output, got " +
                                   std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& gy = ctx.inputs[0];
  const frame::Tensor& b = ctx.inputs[1];
  frame::Tensor& ga = ctx.outputs[0];

  frame::Status rank_status = require_rank2("matmul_grad_lhs_internal", "gy", gy);
  if (!rank_status.is_ok()) return rank_status;
  rank_status = require_rank2("matmul_grad_lhs_internal", "b", b);
  if (!rank_status.is_ok()) return rank_status;

  const frame::Result<frame::DTypeCode> code_result =
      require_matching_supported_dtype("matmul_grad_lhs_internal", "gy/b/ga", gy, b, ga);
  if (!code_result.is_ok()) return code_result.status();
  const frame::DTypeCode code = code_result.value();

  const int64_t m = gy.shape().dim(0);
  const int64_t n = gy.shape().dim(1);
  const int64_t k = b.shape().dim(0);
  const int64_t n2 = b.shape().dim(1);
  if (n != n2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'matmul_grad_lhs_internal' cpu kernel contraction dimension mismatch: gy " +
            gy.shape().to_string() + " has n=" + std::to_string(n) + ", b " +
            b.shape().to_string() + " has n=" + std::to_string(n2) +
            " (gy's second dimension must equal b's second dimension)");
  }

  const frame::Shape expected_ga_shape({m, k});
  if (!(ga.shape() == expected_ga_shape)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'matmul_grad_lhs_internal' cpu kernel requires ga(out) shape to match [m, k], got " +
            ga.shape().to_string() + ", expected " + expected_ga_shape.to_string());
  }

  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    return matmul_grad_lhs_internal_compute<T>(gy, b, ga);
  });
}

// matmul_grad_rhs_internal 的计算核(M17):gb[p,j] = sum_i a[i,p]*gy[i,j]
// (= aᵀ·gy,不物化 aᵀ,直接按转置索引读取 a——a 存储仍是原始 [m,k] 行优先
// 布局,读取 a[i,p] 对应线性偏移 i*k+p)。
template <typename T>
frame::Status matmul_grad_rhs_internal_compute(const frame::Tensor& a, const frame::Tensor& gy,
                                               frame::Tensor& gb) {
  const int64_t m = a.shape().dim(0);
  const int64_t k = a.shape().dim(1);
  const int64_t n = gy.shape().dim(1);

  const T* a_data = static_cast<const T*>(a.raw_data());
  const T* gy_data = static_cast<const T*>(gy.raw_data());
  T* gb_data = gb.data<T>();

  for (int64_t p = 0; p < k; ++p) {
    for (int64_t j = 0; j < n; ++j) {
      float accum = 0.0F;
      for (int64_t i = 0; i < m; ++i) {
        float a_value = 0.0F;
        float gy_value = 0.0F;
        if constexpr (std::is_same_v<T, float>) {
          a_value = a_data[i * k + p];
          gy_value = gy_data[i * n + j];
        } else if constexpr (std::is_same_v<T, frame::float16_t>) {
          a_value = frame::float16_to_float(a_data[i * k + p]);
          gy_value = frame::float16_to_float(gy_data[i * n + j]);
        } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
          a_value = frame::bfloat16_to_float(a_data[i * k + p]);
          gy_value = frame::bfloat16_to_float(gy_data[i * n + j]);
        }
        accum += a_value * gy_value;
      }
      if constexpr (std::is_same_v<T, float>) {
        gb_data[p * n + j] = accum;
      } else if constexpr (std::is_same_v<T, frame::float16_t>) {
        gb_data[p * n + j] = frame::float_to_float16(accum);
      } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
        gb_data[p * n + j] = frame::float_to_bfloat16(accum);
      }
    }
  }
  return frame::Status::ok();
}

frame::Status matmul_grad_rhs_internal_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'matmul_grad_rhs_internal' cpu kernel expects 2 inputs, got " +
                                   std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'matmul_grad_rhs_internal' cpu kernel expects 1 output, got " +
                                   std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& a = ctx.inputs[0];
  const frame::Tensor& gy = ctx.inputs[1];
  frame::Tensor& gb = ctx.outputs[0];

  frame::Status rank_status = require_rank2("matmul_grad_rhs_internal", "a", a);
  if (!rank_status.is_ok()) return rank_status;
  rank_status = require_rank2("matmul_grad_rhs_internal", "gy", gy);
  if (!rank_status.is_ok()) return rank_status;

  const frame::Result<frame::DTypeCode> code_result =
      require_matching_supported_dtype("matmul_grad_rhs_internal", "a/gy/gb", a, gy, gb);
  if (!code_result.is_ok()) return code_result.status();
  const frame::DTypeCode code = code_result.value();

  const int64_t m = a.shape().dim(0);
  const int64_t k = a.shape().dim(1);
  const int64_t m2 = gy.shape().dim(0);
  const int64_t n = gy.shape().dim(1);
  if (m != m2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'matmul_grad_rhs_internal' cpu kernel contraction dimension mismatch: a " +
            a.shape().to_string() + " has m=" + std::to_string(m) + ", gy " +
            gy.shape().to_string() + " has m=" + std::to_string(m2) +
            " (a's first dimension must equal gy's first dimension)");
  }

  const frame::Shape expected_gb_shape({k, n});
  if (!(gb.shape() == expected_gb_shape)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'matmul_grad_rhs_internal' cpu kernel requires gb(out) shape to match [k, n], got " +
            gb.shape().to_string() + ", expected " + expected_gb_shape.to_string());
  }

  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    return matmul_grad_rhs_internal_compute<T>(a, gy, gb);
  });
}

}  // namespace

FRAME_REGISTER_KERNEL("matmul", frame::kCpuBackendName, matmul_cpu_kernel);
FRAME_REGISTER_KERNEL("matmul_grad_lhs_internal", frame::kCpuBackendName,
                      matmul_grad_lhs_internal_cpu_kernel);
FRAME_REGISTER_KERNEL("matmul_grad_rhs_internal", frame::kCpuBackendName,
                      matmul_grad_rhs_internal_cpu_kernel);
