// CPU 归约内核注册桩(sum/mean/max 等)。
// 每个内核形如 Status kernel(ops::KernelContext&),内部经 dispatch_dtype 按 dtype
// 编译期展开(见 include/frame/core/dtype.h),再经 FRAME_REGISTER_KERNEL 注册到
// (op, kCpuBackendName)。sum 已在本文件下方注册;其余算子见下方待办标注。

// TODO(FRAME-IMPL): mean/max 待落地,属未来批次,不在 M5 首批范围(M5 首批
//   归约算子仅 sum,见 PLAN.md「M5 内置算子 v0 批次」行)。参考:
//   docs/architecture/operator-system.md 第4章;include/frame/ops/kernel_registry.h。
//   完成判据:mean/max 各自分支落地后 KernelRegistry::find("mean"/"max",
//   frame::kCpuBackendName) 均可取到内核,tests/cpp/ops/ 用例通过。

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/ops/kernel_registry.h>

#include "accum_cast.h"

namespace {

// axes 校验(与 src/ops/schemas/reduction.cpp::infer_sum_shape 同规则、同
// 错误消息文案——两处各自独立实现:一处作用于编译期 NodeContext/TensorType,
// 一处作用于运行期 KernelContext/Tensor,校验对象类型不同,非同一层的文本
// 复制)。空数组 = 全维归约(design-reviewer 决议,m5-design-brief 决议点 3);
// 非空时三类违例(负值/越界/重复)各自返回英文错误(消息含违例值);拒绝
// 负索引的独立论证同 schema 侧头注释。命中的轴写入 reduced[axis]=true。
frame::Status validate_axes(std::string_view op_name, int64_t rank,
                            const std::vector<int64_t>& axes, std::vector<bool>& reduced) {
  reduced.assign(static_cast<size_t>(rank), false);
  if (axes.empty()) {
    reduced.assign(static_cast<size_t>(rank), true);
    return frame::Status::ok();
  }
  for (int64_t axis : axes) {
    if (axis < 0) {
      return frame::Status::make(
          frame::ErrorCode::kInvalidArgument,
          "op '" + std::string(op_name) + "' axes entry " + std::to_string(axis) +
              " is negative; v0 requires 0 <= axis < rank and does not normalize negative "
              "indices");
    }
    if (axis >= rank) {
      return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                                 "op '" + std::string(op_name) + "' axes entry " +
                                     std::to_string(axis) + " is out of range for rank " +
                                     std::to_string(rank) + " (must satisfy 0 <= axis < rank)");
    }
    if (reduced[static_cast<size_t>(axis)]) {
      return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                                 "op '" + std::string(op_name) + "' axes entry " +
                                     std::to_string(axis) + " is duplicated");
    }
    reduced[static_cast<size_t>(axis)] = true;
  }
  return frame::Status::ok();
}

// 按 reduced 标记与 keepdims 计算归约后的输出各维尺寸:keepdims=true 保留
// 尺寸 1 的维,false 直接消维(全归约得空 vector,对应 rank-0 标量)。
std::vector<int64_t> compute_reduced_dims(const std::vector<int64_t>& in_dims,
                                          const std::vector<bool>& reduced, bool keepdims) {
  std::vector<int64_t> out_dims;
  out_dims.reserve(in_dims.size());
  for (size_t d = 0; d < in_dims.size(); ++d) {
    if (reduced[d]) {
      if (keepdims) out_dims.push_back(1);
    } else {
      out_dims.push_back(in_dims[d]);
    }
  }
  return out_dims;
}

// 把 full_dims 形状下的线性下标 full_linear 解码为多维下标(行优先,从最末
// 维起做取模/整除),再按 reduced 标记投影到 projected_strides 对应形状的
// 线性下标(keepdims=true 时归约维保留、恒定下标 0,尺寸 1;false 时该维
// 整体跳过)。REUSE-002 单份实现:sum 正向归约(多对一累加,调用处对每个
// full_linear 求得同一个 projected_linear 后向其累加)与 sum_grad_internal
// 反向广播(一对多复制,调用处对每个 full_linear 求得同一个 projected_linear
// 后从其读值)共用本函数这一份"行优先解码 → 按归约轴投影 → 行优先编码"逻辑,
// 禁止两份同构实现各自复制。
int64_t project_reduced_linear_index(const std::vector<int64_t>& full_dims,
                                     const std::vector<bool>& reduced, bool keepdims,
                                     const std::vector<int64_t>& projected_strides,
                                     int64_t full_linear) {
  const int64_t rank = static_cast<int64_t>(full_dims.size());
  std::vector<int64_t> full_index(static_cast<size_t>(rank));
  int64_t remaining = full_linear;
  for (int64_t d = rank - 1; d >= 0; --d) {
    const int64_t dim_size = full_dims[static_cast<size_t>(d)];
    const int64_t safe_dim_size = dim_size > 0 ? dim_size : 1;
    full_index[static_cast<size_t>(d)] = remaining % safe_dim_size;
    remaining /= safe_dim_size;
  }

  int64_t projected_linear = 0;
  size_t projected_dim_pos = 0;
  for (int64_t d = 0; d < rank; ++d) {
    if (reduced[static_cast<size_t>(d)]) {
      if (keepdims) ++projected_dim_pos;  // 该维保留为尺寸 1,恒定下标 0,对线性偏移贡献为 0
      continue;
    }
    projected_linear += full_index[static_cast<size_t>(d)] * projected_strides[projected_dim_pos];
    ++projected_dim_pos;
  }
  return projected_linear;
}

// float 累加转换:同目录共享工具(铁律 5 收敛,M22 批4 判重,见
// accum_cast.h 头注释)。
using frame::backends::cpu::from_accum;
using frame::backends::cpu::to_accum;

// sum 的 CPU 参考实现(REUSE-011:参考实现,数值校验用,禁作性能路径)——
// 朴素「输出索引 = 输入索引投影」通用实现:任意 axes 子集均适用,不区分
// 全归约/单轴/多轴特例,正确优先、不追求性能。防御性校验:输入/输出个数、
// dtype 限 v0 浮点三档、axes/keepdims 从 ctx.attrs 取并校验、out shape 与
// 按属性推得的期望 shape 一致;任一违例返回英文错误(ARCH-031 口径:不
// 静默降级)。累加以 float 精度进行(fp16/bf16 逐元素升 float 累加,遍历
// 结束后一次性转回;fp32 直接以 float 累加——实现细节,不进 schema)。
frame::Status sum_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sum' cpu kernel expects 1 input, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sum' cpu kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& x = ctx.inputs[0];
  frame::Tensor& out = ctx.outputs[0];

  const frame::DType x_elem_type = x.dtype();
  const frame::DType out_elem_type = out.dtype();
  const bool elem_type_mismatch = !(x_elem_type == out_elem_type);
  if (elem_type_mismatch) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'sum' cpu kernel requires x/out of the same dtype, got '" +
                                   std::string(x_elem_type.name()) + "', '" +
                                   std::string(out_elem_type.name()) + "'");
  }

  const frame::DTypeCode code = x_elem_type.code();
  const bool supported = (code == frame::DTypeCode::kFloat32) ||
                         (code == frame::DTypeCode::kFloat16) ||
                         (code == frame::DTypeCode::kBFloat16);
  if (!supported) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'sum' cpu kernel does not support dtype '" +
                                   std::string(x_elem_type.name()) +
                                   "' (v0 supports float32/float16/bfloat16 only)");
  }

  // axes(必填)/keepdims(可选,缺省 false)从 ctx.attrs 取,与 schema 同规则
  // 校验(借用契约:attrs 指针仅在调用期间有效,见 kernel_registry.h)。
  if (ctx.attrs == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'sum' cpu kernel is missing required attribute 'axes' "
                               "(int64 array): no attrs provided");
  }
  const auto axes_it = ctx.attrs->find("axes");
  if (axes_it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'sum' cpu kernel is missing required attribute 'axes' "
                               "(int64 array)");
  }
  const std::vector<int64_t>* axes_ptr = std::get_if<std::vector<int64_t>>(&axes_it->second);
  if (axes_ptr == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sum' cpu kernel attribute 'axes' has the wrong type, expected int64 array");
  }

  bool keepdims = false;
  const auto keepdims_it = ctx.attrs->find("keepdims");
  if (keepdims_it != ctx.attrs->end()) {
    const bool* keepdims_ptr = std::get_if<bool>(&keepdims_it->second);
    if (keepdims_ptr == nullptr) {
      return frame::Status::make(
          frame::ErrorCode::kInvalidArgument,
          "op 'sum' cpu kernel attribute 'keepdims' has the wrong type, expected bool");
    }
    keepdims = *keepdims_ptr;
  }

  const int64_t rank = x.shape().rank();
  std::vector<bool> reduced;
  frame::Status axes_status = validate_axes("sum", rank, *axes_ptr, reduced);
  if (!axes_status.is_ok()) {
    return axes_status;
  }

  const std::vector<int64_t>& in_dims = x.shape().dims();
  const std::vector<int64_t> expected_out_dims = compute_reduced_dims(in_dims, reduced, keepdims);
  const frame::Shape expected_out_shape(expected_out_dims);
  const bool out_shape_mismatch = !(out.shape() == expected_out_shape);
  if (out_shape_mismatch) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sum' cpu kernel requires out shape to match the reduction result, got " +
            out.shape().to_string() + ", expected " + expected_out_shape.to_string());
  }

  const frame::Strides out_strides = frame::row_major_strides(expected_out_shape);
  const int64_t numel_in = x.numel();
  const int64_t numel_out = out.numel();

  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    const T* in_data = static_cast<const T*>(x.raw_data());
    T* out_data = out.data<T>();

    // 累加缓冲区以 float 精度、初值 0(空归约集合的和为 0,与 sum 定义一致)。
    std::vector<float> accum(static_cast<size_t>(numel_out), 0.0F);

    // 「输出索引 = 输入索引投影」通用实现(project_reduced_linear_index,
    // REUSE-002):任意 axes 子集(全归约/单轴/多轴)均走同一条路径,不做特例
    // 优化。
    const std::vector<int64_t>& out_stride_values = out_strides.values();
    for (int64_t i = 0; i < numel_in; ++i) {
      const int64_t out_linear =
          project_reduced_linear_index(in_dims, reduced, keepdims, out_stride_values, i);
      accum[static_cast<size_t>(out_linear)] += to_accum<T>(in_data[i]);
    }

    for (int64_t j = 0; j < numel_out; ++j) {
      out_data[j] = from_accum<T>(accum[static_cast<size_t>(j)]);
    }
    return frame::Status::ok();
  });
}

// sum_grad_internal(gy) 的 CPU 参考实现(M17,REUSE-011:参考实现,数值校验
// 用,禁作性能路径)——sum 正向归约的逆操作:把 gy 沿 axes 复制展开回
// input_shape(project_reduced_linear_index 同一份投影逻辑,REUSE-002,方向
// 相反:sum 正向多对一累加,本 kernel 一对多复制)。
//
// keepdims 判定(不额外携带 keepdims 属性——sum_grad_internal 的 attrs 只有
// input_shape/axes,见 schema 头注释"歧义论证"):gy 的实际 rank 唯一确定
// sum 原节点构图时是否用了 keepdims——若 gy.rank()==input_shape.rank(),说明
// 归约维在 gy 中保留为尺寸 1(keepdims=true 效果);若
// gy.rank()==input_shape.rank()-归约维数,说明归约维被整体消去
// (keepdims=false 效果);两者以外的 rank 值一律报错(gy 与
// input_shape/axes 不自洽)。sum 的 axes 语义决定归约维数至少为 1(空数组=
// 全维归约,非空数组长度>=1),故上述两个候选 rank 值不会重合、判定不存在
// 歧义。
frame::Status sum_grad_internal_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'sum_grad_internal' cpu kernel expects 1 input, got " +
                                   std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'sum_grad_internal' cpu kernel expects 1 output, got " +
                                   std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& gy = ctx.inputs[0];
  frame::Tensor& gx = ctx.outputs[0];

  const frame::DType gy_elem_type = gy.dtype();
  const frame::DType gx_elem_type = gx.dtype();
  if (!(gy_elem_type == gx_elem_type)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'sum_grad_internal' cpu kernel requires gy/gx(out) of the same "
                               "dtype, got '" +
                                   std::string(gy_elem_type.name()) + "', '" +
                                   std::string(gx_elem_type.name()) + "'");
  }
  const frame::DTypeCode code = gy_elem_type.code();
  const bool supported = (code == frame::DTypeCode::kFloat32) ||
                         (code == frame::DTypeCode::kFloat16) ||
                         (code == frame::DTypeCode::kBFloat16);
  if (!supported) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'sum_grad_internal' cpu kernel does not support dtype '" +
                                   std::string(gy_elem_type.name()) +
                                   "' (v0 supports float32/float16/bfloat16 only)");
  }

  if (ctx.attrs == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'sum_grad_internal' cpu kernel is missing required attribute "
                               "'input_shape': no attrs provided");
  }
  const auto input_shape_it = ctx.attrs->find("input_shape");
  if (input_shape_it == ctx.attrs->end()) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sum_grad_internal' cpu kernel is missing required attribute 'input_shape'");
  }
  const frame::Shape* input_shape_ptr = std::get_if<frame::Shape>(&input_shape_it->second);
  if (input_shape_ptr == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sum_grad_internal' cpu kernel attribute 'input_shape' has the wrong type, expected "
        "shape");
  }
  const auto axes_it = ctx.attrs->find("axes");
  if (axes_it == ctx.attrs->end()) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sum_grad_internal' cpu kernel is missing required attribute 'axes' (int64 array)");
  }
  const std::vector<int64_t>* axes_ptr = std::get_if<std::vector<int64_t>>(&axes_it->second);
  if (axes_ptr == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sum_grad_internal' cpu kernel attribute 'axes' has the wrong type, expected int64 "
        "array");
  }

  const frame::Shape& input_shape = *input_shape_ptr;
  if (!(gx.shape() == input_shape)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sum_grad_internal' cpu kernel requires gx(out) shape to match attribute "
        "'input_shape', got " +
            gx.shape().to_string() + ", expected " + input_shape.to_string());
  }

  const int64_t rank = input_shape.rank();
  std::vector<bool> reduced;
  // 非 const:允许 return 时自动移动(performance-no-automatic-move,与
  // src/runtime/compile.cpp 同款理由)。
  frame::Status axes_status = validate_axes("sum_grad_internal", rank, *axes_ptr, reduced);
  if (!axes_status.is_ok()) {
    return axes_status;
  }
  int64_t reduced_count = 0;
  for (bool r : reduced) {
    if (r) ++reduced_count;
  }

  const int64_t gy_rank = gy.shape().rank();
  bool keepdims_effective = false;
  if (gy_rank == rank) {
    keepdims_effective = true;
  } else if (gy_rank == rank - reduced_count) {
    keepdims_effective = false;
  } else {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sum_grad_internal' cpu kernel gy rank " + std::to_string(gy_rank) +
            " is inconsistent with input_shape rank " + std::to_string(rank) +
            " and axes reducing " + std::to_string(reduced_count) +
            " dimension(s) (expected gy rank " + std::to_string(rank) + " (keepdims) or " +
            std::to_string(rank - reduced_count) + " (no keepdims))");
  }

  const std::vector<int64_t>& in_dims = input_shape.dims();
  const std::vector<int64_t> expected_gy_dims =
      compute_reduced_dims(in_dims, reduced, keepdims_effective);
  const frame::Shape expected_gy_shape(expected_gy_dims);
  if (!(gy.shape() == expected_gy_shape)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sum_grad_internal' cpu kernel requires gy shape to be consistent with input_shape/"
        "axes, got " +
            gy.shape().to_string() + ", expected " + expected_gy_shape.to_string());
  }

  const frame::Strides gy_strides = frame::row_major_strides(expected_gy_shape);
  const int64_t numel_gx = gx.numel();

  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    const T* gy_data = static_cast<const T*>(gy.raw_data());
    T* gx_data = gx.data<T>();

    const std::vector<int64_t>& gy_stride_values = gy_strides.values();
    for (int64_t i = 0; i < numel_gx; ++i) {
      const int64_t gy_linear =
          project_reduced_linear_index(in_dims, reduced, keepdims_effective, gy_stride_values, i);
      gx_data[i] = gy_data[gy_linear];
    }
    return frame::Status::ok();
  });
}

}  // namespace

FRAME_REGISTER_KERNEL("sum", frame::kCpuBackendName, sum_cpu_kernel);
FRAME_REGISTER_KERNEL("sum_grad_internal", frame::kCpuBackendName, sum_grad_internal_cpu_kernel);
