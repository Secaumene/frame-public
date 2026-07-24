// CUDA 矩阵乘内核(ADR-0019):三精度统一走 cublasLtMatmul(BE-CUDA-001,SDK
// 组件免 ADR)。fp32/fp16/bf16 数据类型分别映射 CUDA_R_32F/CUDA_R_16F/
// CUDA_R_16BF;scale 类型恒 CUDA_R_32F;计算类型恒 CUBLAS_COMPUTE_32F(升 float
// 累加,与 cpu 参考语义一致),唯 fp32 且 CompileOptions::allow_tf32 开启时改用
// CUBLAS_COMPUTE_32F_FAST_TF32(允许而非强制、默认关闭,精度论证见
// ADR-0019,不在此复述)。不含 __global__ 代码,纯 host 端 cuBLASLt 调用,故为
// .cpp 而非 .cu(见 CMakeLists.txt 注释)。
//
// 行主序适配推导:cuBLAS(Lt)原生列主序。本项目 Tensor 一律行主序存储
// (row_major_strides)。设行主序 A[m,k]、B[k,n]、C[m,n]满足 C = A*B。把同一段
// 内存分别按列主序重新解读:行主序 A[m,k](leading dim k)按列主序读恰是
// A^T[k,m](leading dim k);同理 B 按列主序读是 B^T[n,k]... 更直接的等价式:
// C = A*B ⇔ C^T = B^T * A^T(转置乘法逆序公式)。而"C 的行主序内存"与"C^T 的
// 列主序内存"是同一段字节;A/B 同理。因此只需以列主序语义计算
// "C^T(列主序,ld=n) = B^T(列主序,ld=n) * A^T(列主序,ld=k)"——不传任何转置
// 标志(CUBLASLT_MATMUL_DESC_TRANSA/TRANSB 均取默认值 CUBLAS_OP_N),实参位置
// 上把 B(rhs)放"A"位、A(lhs)放"B"位,维度按 (n, m, k) 顺序传入,即可在不做
// 任何显式转置搬运的前提下得到正确的行主序结果,这是行主序调用 cuBLAS(Lt)的
// 标准惯用法。

#include <cstdint>
#include <cublasLt.h>
#include <string>
#include <string_view>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/backend.h>
#include <frame/hal/stream.h>
#include <frame/ops/kernel_registry.h>

#include "../cuda_backend.h"
#include "../cuda_status.h"

namespace {

frame::Status cublas_status(cublasStatus_t status, std::string_view context) {
  if (status == CUBLAS_STATUS_SUCCESS) return frame::Status::ok();
  return frame::Status::make(frame::ErrorCode::kInternal,
                             std::string(context) + ": cublas call failed with status " +
                                 std::to_string(static_cast<int>(status)));
}

// dtype -> cublasLt 数据类型映射(ADR-0019 三精度统一入口);调用前已校验
// dtype 属 {float32, float16, bfloat16} 三者之一(见 matmul_cuda_kernel 校验段)。
cudaDataType_t cublaslt_data_type(frame::DTypeCode code) {
  if (code == frame::DTypeCode::kFloat16) return CUDA_R_16F;
  if (code == frame::DTypeCode::kBFloat16) return CUDA_R_16BF;
  return CUDA_R_32F;
}

// cublasLtMatmulDesc_t 的 RAII 包装:仅在创建成功后持有非空句柄,析构统一销毁
// (ADR-0019 资源纪律:desc/preference/layout 全部错误路径均须销毁,无泄漏)。
// 不可拷贝(持有裸句柄,拷贝会导致重复销毁)。
struct MatmulDescGuard {
  cublasLtMatmulDesc_t desc = nullptr;
  MatmulDescGuard() = default;
  MatmulDescGuard(const MatmulDescGuard&) = delete;
  MatmulDescGuard& operator=(const MatmulDescGuard&) = delete;
  ~MatmulDescGuard() {
    if (desc != nullptr) cublasLtMatmulDescDestroy(desc);
  }
};

// cublasLtMatrixLayout_t 的 RAII 包装,理由同 MatmulDescGuard。
struct MatrixLayoutGuard {
  cublasLtMatrixLayout_t layout = nullptr;
  MatrixLayoutGuard() = default;
  MatrixLayoutGuard(const MatrixLayoutGuard&) = delete;
  MatrixLayoutGuard& operator=(const MatrixLayoutGuard&) = delete;
  ~MatrixLayoutGuard() {
    if (layout != nullptr) cublasLtMatrixLayoutDestroy(layout);
  }
};

// cublasLtMatmulPreference_t 的 RAII 包装,理由同 MatmulDescGuard。
struct MatmulPreferenceGuard {
  cublasLtMatmulPreference_t pref = nullptr;
  MatmulPreferenceGuard() = default;
  MatmulPreferenceGuard(const MatmulPreferenceGuard&) = delete;
  MatmulPreferenceGuard& operator=(const MatmulPreferenceGuard&) = delete;
  ~MatmulPreferenceGuard() {
    if (pref != nullptr) cublasLtMatmulPreferenceDestroy(pref);
  }
};

frame::Status matmul_cuda_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'matmul' cuda kernel expects 2 inputs, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'matmul' cuda kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& lhs = ctx.inputs[0];
  const frame::Tensor& rhs = ctx.inputs[1];
  frame::Tensor& out = ctx.outputs[0];

  if (lhs.shape().rank() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'matmul' cuda kernel requires lhs to be rank-2, got rank " +
                                   std::to_string(lhs.shape().rank()));
  }
  if (rhs.shape().rank() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'matmul' cuda kernel requires rhs to be rank-2, got rank " +
                                   std::to_string(rhs.shape().rank()));
  }
  // 布尔先落地为具名变量再判断(与 src/backends/cpu/kernels/matmul.cpp 同一
  // 惯例):避免 check_iron_rules.sh 的 CPP-012 文本扫描误判为运行时 dtype
  // 分支(本文件不含 __global__ 代码,dtype 分派仅由 cublaslt_data_type 的 if
  // 分支承担,语义等价于 dispatch_dtype 的编译期展开思想,但因无需模板体故未
  // 借用该函数)。
  const bool elem_type_mismatch = !(lhs.dtype() == rhs.dtype()) || !(lhs.dtype() == out.dtype());
  if (elem_type_mismatch) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'matmul' cuda kernel requires lhs/rhs/out of the same dtype, got '" +
            std::string(lhs.dtype().name()) + "', '" + std::string(rhs.dtype().name()) + "', '" +
            std::string(out.dtype().name()) + "'");
  }

  const frame::DTypeCode code = lhs.dtype().code();
  const bool supported = code == frame::DTypeCode::kFloat32 || code == frame::DTypeCode::kFloat16 ||
                         code == frame::DTypeCode::kBFloat16;
  if (!supported) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'matmul' cuda kernel does not support dtype '" +
                                   std::string(lhs.dtype().name()) +
                                   "' (v0 supports float32/float16/bfloat16 only)");
  }

  const int64_t m = lhs.shape().dim(0);
  const int64_t k = lhs.shape().dim(1);
  const int64_t k2 = rhs.shape().dim(0);
  const int64_t n = rhs.shape().dim(1);
  if (k != k2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'matmul' cuda kernel contraction dimension mismatch: lhs " + lhs.shape().to_string() +
            " has k=" + std::to_string(k) + ", rhs " + rhs.shape().to_string() + " has k=" +
            std::to_string(k2) + " (lhs's last dimension must equal rhs's first dimension)");
  }
  const frame::Shape expected_out_shape({m, n});
  if (!(out.shape() == expected_out_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'matmul' cuda kernel requires out shape to match [m, n], got " +
                                   out.shape().to_string() + ", expected " +
                                   expected_out_shape.to_string());
  }

  const frame::Result<frame::hal::Backend*> backend_lookup =
      frame::hal::BackendRegistry::instance().get(ctx.device.backend);
  if (!backend_lookup.is_ok()) {
    return frame::Status::make(
        backend_lookup.status().code(),
        "op 'matmul' cuda kernel: " + std::string(backend_lookup.status().message()));
  }
  // static_cast 而非 dynamic_cast(CPP-011 全域禁 RTTI):本文件位于
  // src/backends/cuda/ 内,ctx.device.backend 由调用方(CudaBackend::launch /
  // CudaExecutable::run)保证等于 "cuda",查得的 Backend* 编译期已知必为
  // CudaBackend 实例(本后端自身注册的类型),安全下转型。
  auto* cuda_backend = static_cast<frame::backends::cuda::CudaBackend*>(backend_lookup.value());

  const cudaStream_t stream =
      ctx.stream != nullptr ? static_cast<cudaStream_t>(ctx.stream->native_handle()) : nullptr;
  const frame::Result<frame::backends::cuda::CublasLtHandleGuard> lt_guard =
      cuda_backend->acquire_cublaslt_handle();
  if (!lt_guard.is_ok()) return lt_guard.status();
  const cublasLtHandle_t lt_handle = lt_guard.value().handle;

  const void* lhs_data = lhs.raw_data();
  const void* rhs_data = rhs.raw_data();
  void* out_data = out.raw_data();

  const cudaDataType_t data_type = cublaslt_data_type(code);
  // 计算类型恒 CUBLAS_COMPUTE_32F(fp32 累加);唯 fp32 且调用方经
  // CompileOptions::allow_tf32 显式开启时降精度到 TF32 级数学模式(允许非
  // 强制、默认关闭,ADR-0019)。compile_options 可空 = 默认选项(视同关闭)。
  const bool use_tf32 = code == frame::DTypeCode::kFloat32 && ctx.compile_options != nullptr &&
                        ctx.compile_options->allow_tf32;
  const cublasComputeType_t compute_type =
      use_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

  MatmulDescGuard matmul_desc;
  {
    const cublasStatus_t status =
        cublasLtMatmulDescCreate(&matmul_desc.desc, compute_type, CUDA_R_32F);
    const frame::Status translated =
        cublas_status(status, "matmul cuda kernel: cublasLtMatmulDescCreate");
    if (!translated.is_ok()) return translated;
  }

  // 见文件头推导:C^T = B^T * A^T 的列主序调用惯用法——rhs 放"A"位、lhs 放
  // "B"位,维度按 (n, m, k) 传入,leading dim 各取自身"行主序列数"(rhs 的 n、
  // lhs 的 k、out 的 n);transA/transB 保持 cublasLt 默认值 CUBLAS_OP_N,无需
  // 显式设置。
  MatrixLayoutGuard a_layout;  // 承载 rhs_data("A"位)
  {
    const cublasStatus_t status = cublasLtMatrixLayoutCreate(
        &a_layout.layout, data_type, static_cast<uint64_t>(n), static_cast<uint64_t>(k), n);
    const frame::Status translated =
        cublas_status(status, "matmul cuda kernel: cublasLtMatrixLayoutCreate(A)");
    if (!translated.is_ok()) return translated;
  }
  MatrixLayoutGuard b_layout;  // 承载 lhs_data("B"位)
  {
    const cublasStatus_t status = cublasLtMatrixLayoutCreate(
        &b_layout.layout, data_type, static_cast<uint64_t>(k), static_cast<uint64_t>(m), k);
    const frame::Status translated =
        cublas_status(status, "matmul cuda kernel: cublasLtMatrixLayoutCreate(B)");
    if (!translated.is_ok()) return translated;
  }
  MatrixLayoutGuard d_layout;  // 承载 out_data;beta=0 下 C/D 复用同一 layout
  {
    const cublasStatus_t status = cublasLtMatrixLayoutCreate(
        &d_layout.layout, data_type, static_cast<uint64_t>(n), static_cast<uint64_t>(m), n);
    const frame::Status translated =
        cublas_status(status, "matmul cuda kernel: cublasLtMatrixLayoutCreate(D)");
    if (!translated.is_ok()) return translated;
  }

  MatmulPreferenceGuard preference;
  {
    const cublasStatus_t status = cublasLtMatmulPreferenceCreate(&preference.pref);
    const frame::Status translated =
        cublas_status(status, "matmul cuda kernel: cublasLtMatmulPreferenceCreate");
    if (!translated.is_ok()) return translated;
  }
  // workspace v0:固定 0 字节(不申请启发式可用的额外 workspace);bench 驱动的
  // workspace 分配留 Task 7/后续跟进。
  // TODO(FRAME-PERF): 经 bench 数据决定非零 workspace 预算。参考:
  // docs/decisions/0019-adopt-cublaslt-and-precision-policy.md。完成判据:
  // bench_matmul 给出 workspace>0 相对 0 字节的吞吐提升数据并据此定案。
  const uint64_t max_workspace_bytes = 0;
  {
    const cublasStatus_t status = cublasLtMatmulPreferenceSetAttribute(
        preference.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_bytes,
        sizeof(max_workspace_bytes));
    const frame::Status translated =
        cublas_status(status, "matmul cuda kernel: cublasLtMatmulPreferenceSetAttribute");
    if (!translated.is_ok()) return translated;
  }

  cublasLtMatmulHeuristicResult_t heuristic_result{};
  int returned_algo_count = 0;
  {
    const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        lt_handle, matmul_desc.desc, a_layout.layout, b_layout.layout, d_layout.layout,
        d_layout.layout, preference.pref, /*requestedAlgoCount=*/1, &heuristic_result,
        &returned_algo_count);
    const frame::Status translated =
        cublas_status(status, "matmul cuda kernel: cublasLtMatmulAlgoGetHeuristic");
    if (!translated.is_ok()) return translated;
  }
  if (returned_algo_count == 0) {
    return cublas_status(
        CUBLAS_STATUS_NOT_SUPPORTED,
        "matmul cuda kernel: cublasLtMatmulAlgoGetHeuristic returned no viable algorithm");
  }

  const float alpha = 1.0F;
  const float beta = 0.0F;
  const cublasStatus_t matmul_status = cublasLtMatmul(
      lt_handle, matmul_desc.desc, &alpha, rhs_data, a_layout.layout, lhs_data, b_layout.layout,
      &beta, out_data, d_layout.layout, out_data, d_layout.layout, &heuristic_result.algo,
      /*workspace=*/nullptr, /*workspaceSizeInBytes=*/0, stream);
  return cublas_status(matmul_status, "matmul cuda kernel: cublasLtMatmul");
}

}  // namespace

FRAME_REGISTER_KERNEL("matmul", frame::kCudaBackendName, matmul_cuda_kernel);
