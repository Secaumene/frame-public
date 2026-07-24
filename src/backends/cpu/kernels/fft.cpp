// CPU 频域内核(M23,批5 T3,ADR-0022):rfft/irfft 经 pocketfft 单头文件库
// 实现(r2c/c2r,末轴变换)。pocketfft include 圈禁本文件(ADR-0022 决策 4:
// `grep -rln "pocketfft" src/ include/` 命中须 ⊆ 本文件 + cmake 声明处)。
//
// 布局约定:rfft 输出/irfft 输入按 [...,k,2] 交错存放,与 std::complex<float>
// 逐字节一致(C++ 标准对 complex<T> 数组与 T[2*N] 数组互相 reinterpret_cast
// 的等价性有明文保证),故直接 reinterpret_cast 复用宿主浮点缓冲、零布局搬运。
// batch(前导维乘积)不手写循环——pocketfft 的 r2c/c2r 接受完整 shape + 变换
// 轴号,内部经 multi_iter 自行遍历其余维度。irfft 的 1/n 归一化经 pocketfft
// fct 参数原生完成(numpy 口径,决议点B)。单线程执行(nthreads=1,
// REUSE-011:参考实现,数值校验用,禁作性能路径,故不追求多线程加速)。
//
// POCKETFFT_NO_MULTITHREADING:关闭该头文件内的 std::thread 线程池实现,
// 只保留单线程 thread_map(f() 直接调用),避免为一份纯数值校验用的参考实现
// 引入线程库依赖。
#define POCKETFFT_NO_MULTITHREADING
#include <complex>
#include <cstddef>
#include <cstdint>
#include <pocketfft_hdronly.h>
#include <string>
#include <variant>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/ops/kernel_registry.h>

namespace {

// 把 frame 的行优先维度列表转为 pocketfft::shape_t(size_t 元素个数数组)。
pocketfft::shape_t to_pocketfft_shape(const std::vector<int64_t>& dims) {
  pocketfft::shape_t shape(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    shape[i] = static_cast<size_t>(dims[i]);
  }
  return shape;
}

// 把行优先元素步幅转为 pocketfft 的字节步幅:pocketfft 的 ndarr/cndarr 对
// `const char*`/`char*` 基址直接加偏移(见 pocketfft_hdronly.h 的
// arr_info/cndarr/ndarr::operator[]),单位是字节而非元素个数,故须乘
// itemsize。
pocketfft::stride_t to_pocketfft_byte_strides(const std::vector<int64_t>& elem_strides,
                                              size_t itemsize) {
  pocketfft::stride_t strides(elem_strides.size());
  for (size_t i = 0; i < elem_strides.size(); ++i) {
    strides[i] = static_cast<ptrdiff_t>(elem_strides[i]) * static_cast<ptrdiff_t>(itemsize);
  }
  return strides;
}

// rfft(x[...,n]) 的 CPU 参考实现(REUSE-011:参考实现,数值校验用,禁作性能
// 路径)——pocketfft::r2c 末轴变换,不归一化(numpy 口径)。dtype 限 fp32
// (pocketfft 无半精度),fail-loud 拒绝其余 dtype。
frame::Status rfft_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'rfft' cpu kernel expects 1 input, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'rfft' cpu kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& x = ctx.inputs[0];
  frame::Tensor& out = ctx.outputs[0];

  const frame::DType x_elem_type = x.dtype();
  const frame::DType out_elem_type = out.dtype();
  const frame::DTypeCode x_code = x_elem_type.code();
  const frame::DTypeCode out_code = out_elem_type.code();
  const bool fp32_only_violation =
      x_code != frame::DTypeCode::kFloat32 || out_code != frame::DTypeCode::kFloat32;
  if (fp32_only_violation) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'rfft' cpu kernel requires x/out to be float32 (pocketfft has no half precision "
        "support), got x='" +
            std::string(x_elem_type.name()) + "', out='" + std::string(out_elem_type.name()) + "'");
  }

  const int64_t rank = x.shape().rank();
  if (rank < 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'rfft' cpu kernel requires x to have rank >= 1, got rank " + std::to_string(rank));
  }
  const int64_t n = x.shape().dim(rank - 1);
  if (n < 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'rfft' cpu kernel requires the last dimension (n) to be >= 2, got n=" +
            std::to_string(n));
  }
  const int64_t k = n / 2 + 1;

  std::vector<int64_t> expected_out_dims = x.shape().dims();
  expected_out_dims[static_cast<size_t>(rank - 1)] = k;
  expected_out_dims.push_back(2);
  const frame::Shape expected_out_shape(expected_out_dims);
  if (!(out.shape() == expected_out_shape)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'rfft' cpu kernel requires out shape to match the rfft result, got " +
            out.shape().to_string() + ", expected " + expected_out_shape.to_string());
  }

  const pocketfft::shape_t shape_in = to_pocketfft_shape(x.shape().dims());
  const pocketfft::stride_t stride_in =
      to_pocketfft_byte_strides(x.strides().values(), sizeof(float));

  // 输出复数数组的形状是 shape_in 把变换轴替换为 k(即 [...,k]);其自然
  // (行优先)步幅对该 [...,k] 形状单独重推,不依赖与 [...,k,2] 实数步幅之间
  // 的除法关系,更直白。
  const std::vector<int64_t> out_complex_dims(expected_out_dims.begin(),
                                              expected_out_dims.end() - 1);
  const frame::Shape out_complex_shape(out_complex_dims);
  const frame::Strides out_complex_strides = frame::row_major_strides(out_complex_shape);
  const pocketfft::stride_t stride_out =
      to_pocketfft_byte_strides(out_complex_strides.values(), sizeof(std::complex<float>));

  const float* x_data = static_cast<const float*>(x.raw_data());
  auto* out_data = reinterpret_cast<std::complex<float>*>(out.data<float>());

  pocketfft::r2c<float>(shape_in, stride_in, stride_out, static_cast<size_t>(rank - 1),
                        pocketfft::FORWARD, x_data, out_data, /*fct=*/1.0F, /*nthreads=*/1);

  return frame::Status::ok();
}

// irfft(z[...,k,2]; n) 的 CPU 参考实现(REUSE-011:参考实现,数值校验用,禁作
// 性能路径)——pocketfft::c2r 末轴逆变换,归一化 1/n(numpy 口径,
// irfft(rfft(x), n)≡x)经 pocketfft 的 fct 参数原生完成。dtype 限 fp32,
// fail-loud 拒绝其余 dtype。
frame::Status irfft_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'irfft' cpu kernel expects 1 input, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'irfft' cpu kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& z = ctx.inputs[0];
  frame::Tensor& out = ctx.outputs[0];

  const frame::DType z_elem_type = z.dtype();
  const frame::DType out_elem_type = out.dtype();
  const frame::DTypeCode z_code = z_elem_type.code();
  const frame::DTypeCode out_code = out_elem_type.code();
  const bool fp32_only_violation =
      z_code != frame::DTypeCode::kFloat32 || out_code != frame::DTypeCode::kFloat32;
  if (fp32_only_violation) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'irfft' cpu kernel requires z/out to be float32 (pocketfft has no half precision "
        "support), got z='" +
            std::string(z_elem_type.name()) + "', out='" + std::string(out_elem_type.name()) + "'");
  }

  const int64_t rank = z.shape().rank();
  if (rank < 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'irfft' cpu kernel requires z to have rank >= 2 (trailing axes are [k, 2]), got "
        "rank " +
            std::to_string(rank));
  }
  const int64_t last_dim = z.shape().dim(rank - 1);
  if (last_dim != 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'irfft' cpu kernel requires the last dimension to be 2 (interleaved re/im), got " +
            std::to_string(last_dim));
  }

  if (ctx.attrs == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'irfft' cpu kernel is missing required attribute 'n' (int64): "
                               "no attrs provided");
  }
  const auto n_it = ctx.attrs->find("n");
  if (n_it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'irfft' cpu kernel is missing required attribute 'n' (int64)");
  }
  const int64_t* n_ptr = std::get_if<int64_t>(&n_it->second);
  if (n_ptr == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'irfft' cpu kernel attribute 'n' has the wrong type, expected int64");
  }
  const int64_t n = *n_ptr;

  const int64_t k = z.shape().dim(rank - 2);
  const int64_t expected_k = n / 2 + 1;
  if (k != expected_k) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'irfft' cpu kernel requires k=n/2+1 for attribute n=" + std::to_string(n) +
            ", expected k=" + std::to_string(expected_k) + ", got k=" + std::to_string(k));
  }

  std::vector<int64_t> expected_out_dims(z.shape().dims().begin(), z.shape().dims().end() - 1);
  expected_out_dims.back() = n;
  const frame::Shape expected_out_shape(expected_out_dims);
  if (!(out.shape() == expected_out_shape)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'irfft' cpu kernel requires out shape to match the irfft result, got " +
            out.shape().to_string() + ", expected " + expected_out_shape.to_string());
  }

  const pocketfft::shape_t shape_out = to_pocketfft_shape(expected_out_dims);
  const pocketfft::stride_t stride_out =
      to_pocketfft_byte_strides(out.strides().values(), sizeof(float));

  // 输入复数数组的形状是 shape_out 把变换轴替换为 k(即 z 去掉末尾交错轴后的
  // [...,k] 形状);其自然(行优先)步幅对该形状单独重推,原因同 rfft 侧。
  const std::vector<int64_t> z_complex_dims(z.shape().dims().begin(), z.shape().dims().end() - 1);
  const frame::Shape z_complex_shape(z_complex_dims);
  const frame::Strides z_complex_strides = frame::row_major_strides(z_complex_shape);
  const pocketfft::stride_t stride_in =
      to_pocketfft_byte_strides(z_complex_strides.values(), sizeof(std::complex<float>));

  const auto* z_data = reinterpret_cast<const std::complex<float>*>(z.raw_data());
  float* out_data = out.data<float>();
  const auto axis = static_cast<size_t>(expected_out_dims.size() - 1);
  const float fct = 1.0F / static_cast<float>(n);

  pocketfft::c2r<float>(shape_out, stride_in, stride_out, axis, pocketfft::BACKWARD, z_data,
                        out_data, fct, /*nthreads=*/1);

  return frame::Status::ok();
}

}  // namespace

FRAME_REGISTER_KERNEL("rfft", frame::kCpuBackendName, rfft_cpu_kernel);
FRAME_REGISTER_KERNEL("irfft", frame::kCpuBackendName, irfft_cpu_kernel);
