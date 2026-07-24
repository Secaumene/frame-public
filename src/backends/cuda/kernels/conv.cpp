// CUDA 二维卷积内核(M21,批3 T5,ADR-0021):conv2d 前向经
// cudnnConvolutionForward(+bias 时 cudnnAddTensor 通道广播);
// conv2d_grad_input_internal/conv2d_grad_filter_internal 分别经
// cudnnConvolutionBackwardData/cudnnConvolutionBackwardFilter。算法经
// cudnnGetConvolution*Algorithm_v7 启发式首选(取首个 status 为 SUCCESS 的
// 返回项)+ workspace 查询分配(ADR-0021 决策 2)。conv1d 不落地专属 kernel——
// CUDA 侧经 decomposition 转 conv2d(裁决点②,计划 1.1 节)。不含 __global__
// 代码,纯 host 端 cuDNN 调用编排,故为 .cpp 而非 .cu(同 matmul.cpp 先例)。
//
// 精度(ADR-0021 决策 4):dataType 按 fp32/fp16/bf16 映射,computeType 恒
// CUDNN_DATA_FLOAT(fp32 累加);mathType:fp32 严格 = CUDNN_FMA_MATH,
// ctx.compile_options 非空且 allow_tf32=true 时 = CUDNN_DEFAULT_MATH(TF32
// 许可,ADR-0019 单开关纪律沿用),fp16/bf16 恒 CUDNN_TENSOR_OP_MATH。

#include <cstddef>
#include <cstdint>
#include <cudnn.h>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/storage.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/stream.h>
#include <frame/ops/kernel_registry.h>

#include "../cuda_backend.h"
#include "../cuda_status.h"
#include "cudnn_utils.h"

namespace {

using frame::backends::cuda::CudaBackend;
using frame::backends::cuda::CudnnHandleGuard;

// cudnnStatus_t -> Status 翻译、TensorDescGuard(RAII):同目录共享工具(铁律
// 5 收敛,M22 批4 判重,见 cudnn_utils.h 头注释)。
using frame::backends::cuda::cudnn_status;
using frame::backends::cuda::TensorDescGuard;

// dtype -> cudnnDataType_t 映射(ADR-0021 决策 4);调用前已校验 dtype 属 v0
// 三档浮点。
cudnnDataType_t cudnn_data_type(frame::DTypeCode code) {
  if (code == frame::DTypeCode::kFloat16) return CUDNN_DATA_HALF;
  if (code == frame::DTypeCode::kBFloat16) return CUDNN_DATA_BFLOAT16;
  return CUDNN_DATA_FLOAT;
}

// mathType 选择(ADR-0021 决策 4):fp32 严格 FMA;fp32+allow_tf32 降级
// DEFAULT_MATH(TF32 许可);fp16/bf16 恒 TENSOR_OP_MATH。compile_options 可空
// 视同 allow_tf32=false(与 matmul.cpp::matmul_cuda_kernel 的 use_tf32 判据
// 同一惯例)。
cudnnMathType_t select_conv_math_type(frame::DTypeCode code,
                                      const frame::hal::CompileOptions* compile_options) {
  if (code == frame::DTypeCode::kFloat32) {
    const bool allow_tf32 = compile_options != nullptr && compile_options->allow_tf32;
    return allow_tf32 ? CUDNN_DEFAULT_MATH : CUDNN_FMA_MATH;
  }
  return CUDNN_TENSOR_OP_MATH;
}

// ---------------------------------------------------------------------------
// cuDNN 描述符 RAII guard(ADR-0021:描述符全 RAII guard,资源纪律同
// matmul.cpp::MatmulDescGuard 系列——仅在创建成功后持有非空句柄,析构统一
// 销毁,不可拷贝)。TensorDescGuard 见上方共享工具 using 声明;
// FilterDescGuard/ConvDescGuard 为 conv.cpp 专有,不与其余文件重复,不收编。
// ---------------------------------------------------------------------------
struct FilterDescGuard {
  cudnnFilterDescriptor_t desc = nullptr;
  FilterDescGuard() = default;
  FilterDescGuard(const FilterDescGuard&) = delete;
  FilterDescGuard& operator=(const FilterDescGuard&) = delete;
  ~FilterDescGuard() {
    if (desc != nullptr) cudnnDestroyFilterDescriptor(desc);
  }
};

struct ConvDescGuard {
  cudnnConvolutionDescriptor_t desc = nullptr;
  ConvDescGuard() = default;
  ConvDescGuard(const ConvDescGuard&) = delete;
  ConvDescGuard& operator=(const ConvDescGuard&) = delete;
  ~ConvDescGuard() {
    if (desc != nullptr) cudnnDestroyConvolutionDescriptor(desc);
  }
};

// dtype 一致性 + v0 浮点三档校验(REUSE-002:与
// src/backends/cpu/kernels/kernel_dtype_checks.h::require_matching_supported_dtype
// 同一动机,cuda 侧独立持有一份实现,不跨后端目录借用私有头——该头文件自身
// 声明"仅供 src/backends/cpu/kernels/ 内部包含")。
frame::Result<frame::DTypeCode> require_matching_supported_dtype(
    std::string_view op_name, std::string_view role_phrase,
    const std::vector<const frame::Tensor*>& tensors) {
  const frame::DType first_type = tensors.front()->dtype();
  bool mismatch = false;
  for (const frame::Tensor* tensor : tensors) {
    // 先落地为具名变量再判断(与 src/backends/cpu/kernels/kernel_dtype_checks.h
    // 同一惯例):check_iron_rules.sh 的 CPP-012 文本扫描按 `if (...dtype...)`
    // 形态识别疑似运行时 dtype 分支,提前拆出具名变量可与 dispatch_dtype 编译
    // 期分派清晰区分,避免误判。
    const frame::DType current_type = tensor->dtype();
    if (!(current_type == first_type)) {
      mismatch = true;
      break;
    }
  }
  if (mismatch) {
    std::string type_list_text;
    for (const frame::Tensor* tensor : tensors) {
      if (!type_list_text.empty()) type_list_text += ", ";
      type_list_text += "'" + std::string(tensor->dtype().name()) + "'";
    }
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' cuda kernel requires " +
                                   std::string(role_phrase) + " of the same dtype, got " +
                                   type_list_text);
  }
  const frame::DTypeCode code = first_type.code();
  const bool supported = code == frame::DTypeCode::kFloat32 || code == frame::DTypeCode::kFloat16 ||
                         code == frame::DTypeCode::kBFloat16;
  if (!supported) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op '" + std::string(op_name) + "' cuda kernel does not support dtype '" +
            std::string(first_type.name()) + "' (v0 supports float32/float16/bfloat16 only)");
  }
  return code;
}

frame::Status require_rank(std::string_view op_name, std::string_view operand_label,
                           int64_t expected_rank, const frame::Tensor& tensor) {
  if (tensor.shape().rank() != expected_rank) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' cuda kernel requires " +
                                   std::string(operand_label) + " to be rank-" +
                                   std::to_string(expected_rank) + ", got rank " +
                                   std::to_string(tensor.shape().rank()));
  }
  return frame::Status::ok();
}

// 二维卷积的几何参数(REUSE-002:与 src/backends/cpu/kernels/conv.cpp::
// Conv2dRuntimeParams 同一动机,cuda 侧独立持有一份实现,两文件互不可见)。
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

frame::Result<std::vector<int64_t>> read_int64_array_attr(const frame::ops::KernelContext& ctx,
                                                          std::string_view op_name,
                                                          std::string_view attr_name) {
  if (ctx.attrs == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel is missing required attribute '" +
                                   std::string(attr_name) + "': no attrs provided");
  }
  const auto it = ctx.attrs->find(std::string(attr_name));
  if (it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel is missing required attribute '" +
                                   std::string(attr_name) + "'");
  }
  const std::vector<int64_t>* value = std::get_if<std::vector<int64_t>>(&it->second);
  if (value == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' cuda kernel attribute '" +
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
                                   "' cuda kernel is missing required attribute '" +
                                   std::string(attr_name) + "': no attrs provided");
  }
  const auto it = ctx.attrs->find(std::string(attr_name));
  if (it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel is missing required attribute '" +
                                   std::string(attr_name) + "'");
  }
  const int64_t* value = std::get_if<int64_t>(&it->second);
  if (value == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' cuda kernel attribute '" +
                                   std::string(attr_name) + "' has the wrong type, expected int64");
  }
  return *value;
}

frame::Result<frame::Shape> read_shape_attr(const frame::ops::KernelContext& ctx,
                                            std::string_view op_name, std::string_view attr_name) {
  if (ctx.attrs == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel is missing required attribute '" +
                                   std::string(attr_name) + "': no attrs provided");
  }
  const auto it = ctx.attrs->find(std::string(attr_name));
  if (it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) +
                                   "' cuda kernel is missing required attribute '" +
                                   std::string(attr_name) + "'");
  }
  const frame::Shape* value = std::get_if<frame::Shape>(&it->second);
  if (value == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op '" + std::string(op_name) + "' cuda kernel attribute '" +
                                   std::string(attr_name) + "' has the wrong type, expected shape");
  }
  return *value;
}

// 读取 stride/padding/groups 三属性并推出输出空间维(floor 口径),供
// conv2d/conv2d_grad_input_internal/conv2d_grad_filter_internal 三个 kernel
// 共用(REUSE-002)。x_shape/w_shape 分别提供 [N,Cin,H,W]/[Cout,Cin/g,KH,KW]。
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
                                   "' cuda kernel requires 'stride'/'padding' to each have 2 "
                                   "elements");
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

// 构建 x/w/y/conv 四个描述符(out 形参:调用方在栈上持有 guard 局部变量,本
// 函数只负责创建+配置,避免返回值携带不可拷贝的 guard 类型)。角色随调用方
// 语义复用:BackwardData 时 x_desc 承载 dx、y_desc 承载 dy;BackwardFilter 时
// w_desc 承载 dw、y_desc 承载 dy——三个 kernel 的几何/描述符构建逻辑同构,仅
// 调用的 cuDNN 计算函数不同(REUSE-002)。
frame::Status setup_conv_descriptors(std::string_view op_name, const Conv2dRuntimeParams& p,
                                     cudnnDataType_t data_type, cudnnMathType_t math_type,
                                     TensorDescGuard& x_desc, FilterDescGuard& w_desc,
                                     TensorDescGuard& y_desc, ConvDescGuard& conv_desc) {
  FRAME_RETURN_IF_ERROR(
      cudnn_status(cudnnCreateTensorDescriptor(&x_desc.desc),
                   std::string(op_name) + " cuda kernel: cudnnCreateTensorDescriptor(x)"));
  FRAME_RETURN_IF_ERROR(
      cudnn_status(cudnnSetTensor4dDescriptor(x_desc.desc, CUDNN_TENSOR_NCHW, data_type,
                                              static_cast<int>(p.n), static_cast<int>(p.cin),
                                              static_cast<int>(p.h), static_cast<int>(p.w)),
                   std::string(op_name) + " cuda kernel: cudnnSetTensor4dDescriptor(x)"));

  FRAME_RETURN_IF_ERROR(
      cudnn_status(cudnnCreateFilterDescriptor(&w_desc.desc),
                   std::string(op_name) + " cuda kernel: cudnnCreateFilterDescriptor"));
  FRAME_RETURN_IF_ERROR(cudnn_status(
      cudnnSetFilter4dDescriptor(w_desc.desc, data_type, CUDNN_TENSOR_NCHW,
                                 static_cast<int>(p.cout), static_cast<int>(p.cin_per_group),
                                 static_cast<int>(p.kh), static_cast<int>(p.kw)),
      std::string(op_name) + " cuda kernel: cudnnSetFilter4dDescriptor"));

  FRAME_RETURN_IF_ERROR(
      cudnn_status(cudnnCreateTensorDescriptor(&y_desc.desc),
                   std::string(op_name) + " cuda kernel: cudnnCreateTensorDescriptor(y)"));
  FRAME_RETURN_IF_ERROR(
      cudnn_status(cudnnSetTensor4dDescriptor(y_desc.desc, CUDNN_TENSOR_NCHW, data_type,
                                              static_cast<int>(p.n), static_cast<int>(p.cout),
                                              static_cast<int>(p.out_h), static_cast<int>(p.out_w)),
                   std::string(op_name) + " cuda kernel: cudnnSetTensor4dDescriptor(y)"));

  FRAME_RETURN_IF_ERROR(
      cudnn_status(cudnnCreateConvolutionDescriptor(&conv_desc.desc),
                   std::string(op_name) + " cuda kernel: cudnnCreateConvolutionDescriptor"));
  FRAME_RETURN_IF_ERROR(cudnn_status(
      cudnnSetConvolution2dDescriptor(conv_desc.desc, static_cast<int>(p.pad_h),
                                      static_cast<int>(p.pad_w), static_cast<int>(p.stride_h),
                                      static_cast<int>(p.stride_w), /*dilation_h=*/1,
                                      /*dilation_w=*/1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
      std::string(op_name) + " cuda kernel: cudnnSetConvolution2dDescriptor"));
  FRAME_RETURN_IF_ERROR(
      cudnn_status(cudnnSetConvolutionGroupCount(conv_desc.desc, static_cast<int>(p.groups)),
                   std::string(op_name) + " cuda kernel: cudnnSetConvolutionGroupCount"));
  FRAME_RETURN_IF_ERROR(
      cudnn_status(cudnnSetConvolutionMathType(conv_desc.desc, math_type),
                   std::string(op_name) + " cuda kernel: cudnnSetConvolutionMathType"));
  return frame::Status::ok();
}

// 在 cudnnGetConvolution*Algorithm_v7 返回的 perf 数组中取首个 status ==
// CUDNN_STATUS_SUCCESS 的表项下标(ADR-0021"首选"口径);三个方向(Fwd/
// BwdData/BwdFilter)的 Perf 结构体字段同构但类型不同,以模板统一(编译期
// 展开,非运行时分支)。全部候选均不可用时返回 -1。
template <typename PerfT>
int pick_first_successful_algo_index(const PerfT* perf_results, int count) {
  for (int i = 0; i < count; ++i) {
    if (perf_results[i].status == CUDNN_STATUS_SUCCESS) return i;
  }
  return -1;
}

frame::Result<CudaBackend*> lookup_cuda_backend(std::string_view op_name, frame::Device device) {
  const frame::Result<frame::hal::Backend*> backend_lookup =
      frame::hal::BackendRegistry::instance().get(device.backend);
  if (!backend_lookup.is_ok()) {
    return frame::Status::make(backend_lookup.status().code(),
                               "op '" + std::string(op_name) + "' cuda kernel: " +
                                   std::string(backend_lookup.status().message()));
  }
  // static_cast 而非 dynamic_cast(CPP-011 全域禁 RTTI):本文件位于
  // src/backends/cuda/ 内,device.backend 由调用方保证等于 "cuda",查得的
  // Backend* 编译期已知必为 CudaBackend 实例(同 matmul.cpp 先例论证)。
  return static_cast<CudaBackend*>(backend_lookup.value());
}

cudaStream_t native_stream(frame::hal::Stream* stream) {
  return stream != nullptr ? static_cast<cudaStream_t>(stream->native_handle()) : nullptr;
}

// 按 workspace_bytes 经既有分配器分配临时 workspace(bytes==0 时不分配,
// 返回空 shared_ptr)。调用方持有非空返回值时,必须在函数返回前对 stream
// 显式 cudaStreamSynchronize 后才能让本 Storage 随作用域结束析构——cudaFree
// 隐式全设备同步语义未写入 CUDA 官方契约(REUSE-002:同
// src/backends/cuda/kernels/reduction.cu::run_full_reduction_sum 头注释同一
// 纪律,cuda 侧独立持有一份实现)。
frame::Result<std::shared_ptr<frame::Storage>> allocate_workspace_if_needed(
    frame::hal::Allocator& allocator, frame::Device device, size_t workspace_bytes) {
  if (workspace_bytes == 0) return std::shared_ptr<frame::Storage>{};
  return frame::Storage::allocate(allocator, workspace_bytes, frame::kDefaultAlignment, device);
}

// conv2d 前向:2 或 3 输入(x,w[,bias]),1 输出。
frame::Status conv2d_cuda_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 2 && ctx.inputs.size() != 3) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'conv2d' cuda kernel expects 2 or 3 inputs, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'conv2d' cuda kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
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
                               "op 'conv2d' cuda kernel requires out shape to match the "
                               "convolution result, got " +
                                   out.shape().to_string() + ", expected " +
                                   expected_out_shape.to_string());
  }

  const frame::Result<CudaBackend*> backend_result = lookup_cuda_backend("conv2d", ctx.device);
  if (!backend_result.is_ok()) return backend_result.status();
  CudaBackend* cuda_backend = backend_result.value();
  const cudaStream_t stream = native_stream(ctx.stream);
  const frame::Result<CudnnHandleGuard> handle_guard = cuda_backend->acquire_cudnn_handle(stream);
  if (!handle_guard.is_ok()) return handle_guard.status();
  const cudnnHandle_t handle = handle_guard.value().handle;

  const cudnnDataType_t data_type = cudnn_data_type(code);
  const cudnnMathType_t math_type = select_conv_math_type(code, ctx.compile_options);

  TensorDescGuard x_desc;
  FilterDescGuard w_desc;
  TensorDescGuard y_desc;
  ConvDescGuard conv_desc;
  const frame::Status setup_status = setup_conv_descriptors("conv2d", params, data_type, math_type,
                                                            x_desc, w_desc, y_desc, conv_desc);
  if (!setup_status.is_ok()) return setup_status;

  cudnnConvolutionFwdAlgoPerf_t perf_results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
  int returned_count = 0;
  FRAME_RETURN_IF_ERROR(cudnn_status(
      cudnnGetConvolutionForwardAlgorithm_v7(handle, x_desc.desc, w_desc.desc, conv_desc.desc,
                                             y_desc.desc, CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
                                             &returned_count, perf_results),
      "conv2d cuda kernel: cudnnGetConvolutionForwardAlgorithm_v7"));
  const int chosen = pick_first_successful_algo_index(perf_results, returned_count);
  if (chosen < 0) {
    return frame::Status::make(frame::ErrorCode::kInternal,
                               "op 'conv2d' cuda kernel: "
                               "cudnnGetConvolutionForwardAlgorithm_v7 returned no usable "
                               "algorithm");
  }
  const cudnnConvolutionFwdAlgo_t algo = perf_results[chosen].algo;
  const size_t workspace_bytes = perf_results[chosen].memory;

  frame::hal::Allocator* allocator = cuda_backend->allocator(ctx.device);
  if (allocator == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInternal,
                               "op 'conv2d' cuda kernel: allocator unavailable for device '" +
                                   std::string(ctx.device.backend) + "'");
  }
  const frame::Result<std::shared_ptr<frame::Storage>> workspace_result =
      allocate_workspace_if_needed(*allocator, ctx.device, workspace_bytes);
  if (!workspace_result.is_ok()) return workspace_result.status();
  const std::shared_ptr<frame::Storage>& workspace_storage = workspace_result.value();
  void* workspace_ptr = workspace_storage != nullptr ? workspace_storage->data() : nullptr;

  const void* x_data = x.raw_data();
  const void* w_data = w.raw_data();
  void* out_data = out.raw_data();

  const float alpha = 1.0F;
  const float beta = 0.0F;
  frame::Status run_status =
      cudnn_status(cudnnConvolutionForward(handle, &alpha, x_desc.desc, x_data, w_desc.desc, w_data,
                                           conv_desc.desc, algo, workspace_ptr, workspace_bytes,
                                           &beta, y_desc.desc, out_data),
                   "conv2d cuda kernel: cudnnConvolutionForward");

  if (run_status.is_ok() && has_bias) {
    // bias 描述符 [1,Cout,1,1](通道广播,ADR-0021 交付物 C 段);cudnnAddTensor
    // 以 beta=1 累加进已含卷积结果的 out_data(C = alpha*A + beta*C)。
    TensorDescGuard bias_desc;
    run_status = cudnn_status(cudnnCreateTensorDescriptor(&bias_desc.desc),
                              "conv2d cuda kernel: cudnnCreateTensorDescriptor(bias)");
    if (run_status.is_ok()) {
      run_status =
          cudnn_status(cudnnSetTensor4dDescriptor(bias_desc.desc, CUDNN_TENSOR_NCHW, data_type, 1,
                                                  static_cast<int>(params.cout), 1, 1),
                       "conv2d cuda kernel: cudnnSetTensor4dDescriptor(bias)");
    }
    if (run_status.is_ok()) {
      const void* bias_data = ctx.inputs[2].raw_data();
      const float bias_alpha = 1.0F;
      const float bias_beta = 1.0F;
      run_status = cudnn_status(cudnnAddTensor(handle, &bias_alpha, bias_desc.desc, bias_data,
                                               &bias_beta, y_desc.desc, out_data),
                                "conv2d cuda kernel: cudnnAddTensor");
    }
  }

  if (workspace_storage != nullptr) {
    const frame::Status sync_status = frame::backends::cuda::cuda_status(
        cudaStreamSynchronize(stream),
        "conv2d cuda kernel: cudaStreamSynchronize before workspace release");
    return !run_status.is_ok() ? run_status : sync_status;
  }
  return run_status;
}

// conv2d_grad_input_internal(dy,w)->dx(BackwardData);attrs=
// input_shape+stride/padding/groups(与 conv2d 同款几何,取原 x 的 shape)。
frame::Status conv2d_grad_input_internal_cuda_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'conv2d_grad_input_internal' cuda kernel expects 2 inputs, "
                               "got " +
                                   std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'conv2d_grad_input_internal' cuda kernel expects 1 output, "
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

  const frame::Result<frame::Shape> input_shape_result =
      read_shape_attr(ctx, "conv2d_grad_input_internal", "input_shape");
  if (!input_shape_result.is_ok()) return input_shape_result.status();
  const frame::Shape& input_shape = input_shape_result.value();
  if (!(dx.shape() == input_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'conv2d_grad_input_internal' cuda kernel requires dx(out) "
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
                               "op 'conv2d_grad_input_internal' cuda kernel requires dy shape to "
                               "match [N, Cout, out_h, out_w], got " +
                                   dy.shape().to_string() + ", expected " +
                                   expected_dy_shape.to_string());
  }

  const frame::Result<CudaBackend*> backend_result =
      lookup_cuda_backend("conv2d_grad_input_internal", ctx.device);
  if (!backend_result.is_ok()) return backend_result.status();
  CudaBackend* cuda_backend = backend_result.value();
  const cudaStream_t stream = native_stream(ctx.stream);
  const frame::Result<CudnnHandleGuard> handle_guard = cuda_backend->acquire_cudnn_handle(stream);
  if (!handle_guard.is_ok()) return handle_guard.status();
  const cudnnHandle_t handle = handle_guard.value().handle;

  const cudnnDataType_t data_type = cudnn_data_type(code);
  const cudnnMathType_t math_type = select_conv_math_type(code, ctx.compile_options);

  // x_desc 承载 dx(input_shape),y_desc 承载 dy——角色复用见
  // setup_conv_descriptors 头注释。
  TensorDescGuard dx_desc;
  FilterDescGuard w_desc;
  TensorDescGuard dy_desc;
  ConvDescGuard conv_desc;
  const frame::Status setup_status =
      setup_conv_descriptors("conv2d_grad_input_internal", params, data_type, math_type, dx_desc,
                             w_desc, dy_desc, conv_desc);
  if (!setup_status.is_ok()) return setup_status;

  cudnnConvolutionBwdDataAlgoPerf_t perf_results[CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];
  int returned_count = 0;
  FRAME_RETURN_IF_ERROR(cudnn_status(
      cudnnGetConvolutionBackwardDataAlgorithm_v7(
          handle, w_desc.desc, dy_desc.desc, conv_desc.desc, dx_desc.desc,
          CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT, &returned_count, perf_results),
      "conv2d_grad_input_internal cuda kernel: cudnnGetConvolutionBackwardDataAlgorithm_v7"));
  const int chosen = pick_first_successful_algo_index(perf_results, returned_count);
  if (chosen < 0) {
    return frame::Status::make(frame::ErrorCode::kInternal,
                               "op 'conv2d_grad_input_internal' cuda kernel: "
                               "cudnnGetConvolutionBackwardDataAlgorithm_v7 returned no usable "
                               "algorithm");
  }
  const cudnnConvolutionBwdDataAlgo_t algo = perf_results[chosen].algo;
  const size_t workspace_bytes = perf_results[chosen].memory;

  frame::hal::Allocator* allocator = cuda_backend->allocator(ctx.device);
  if (allocator == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInternal,
                               "op 'conv2d_grad_input_internal' cuda kernel: allocator "
                               "unavailable for device '" +
                                   std::string(ctx.device.backend) + "'");
  }
  const frame::Result<std::shared_ptr<frame::Storage>> workspace_result =
      allocate_workspace_if_needed(*allocator, ctx.device, workspace_bytes);
  if (!workspace_result.is_ok()) return workspace_result.status();
  const std::shared_ptr<frame::Storage>& workspace_storage = workspace_result.value();
  void* workspace_ptr = workspace_storage != nullptr ? workspace_storage->data() : nullptr;

  const void* w_data = w.raw_data();
  const void* dy_data = dy.raw_data();
  void* dx_data = dx.raw_data();

  const float alpha = 1.0F;
  const float beta = 0.0F;
  const frame::Status run_status =
      cudnn_status(cudnnConvolutionBackwardData(handle, &alpha, w_desc.desc, w_data, dy_desc.desc,
                                                dy_data, conv_desc.desc, algo, workspace_ptr,
                                                workspace_bytes, &beta, dx_desc.desc, dx_data),
                   "conv2d_grad_input_internal cuda kernel: cudnnConvolutionBackwardData");

  if (workspace_storage != nullptr) {
    const frame::Status sync_status = frame::backends::cuda::cuda_status(
        cudaStreamSynchronize(stream),
        "conv2d_grad_input_internal cuda kernel: cudaStreamSynchronize before workspace release");
    return !run_status.is_ok() ? run_status : sync_status;
  }
  return run_status;
}

// conv2d_grad_filter_internal(x,dy)->dw(BackwardFilter);attrs=
// filter_shape+stride/padding/groups。
frame::Status conv2d_grad_filter_internal_cuda_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'conv2d_grad_filter_internal' cuda kernel expects 2 inputs, "
                               "got " +
                                   std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'conv2d_grad_filter_internal' cuda kernel expects 1 output, "
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

  const frame::Result<frame::Shape> filter_shape_result =
      read_shape_attr(ctx, "conv2d_grad_filter_internal", "filter_shape");
  if (!filter_shape_result.is_ok()) return filter_shape_result.status();
  const frame::Shape& filter_shape = filter_shape_result.value();
  if (!(dw.shape() == filter_shape)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'conv2d_grad_filter_internal' cuda kernel requires dw(out) "
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
                               "op 'conv2d_grad_filter_internal' cuda kernel requires dy shape to "
                               "match [N, Cout, out_h, out_w], got " +
                                   dy.shape().to_string() + ", expected " +
                                   expected_dy_shape.to_string());
  }

  const frame::Result<CudaBackend*> backend_result =
      lookup_cuda_backend("conv2d_grad_filter_internal", ctx.device);
  if (!backend_result.is_ok()) return backend_result.status();
  CudaBackend* cuda_backend = backend_result.value();
  const cudaStream_t stream = native_stream(ctx.stream);
  const frame::Result<CudnnHandleGuard> handle_guard = cuda_backend->acquire_cudnn_handle(stream);
  if (!handle_guard.is_ok()) return handle_guard.status();
  const cudnnHandle_t handle = handle_guard.value().handle;

  const cudnnDataType_t data_type = cudnn_data_type(code);
  const cudnnMathType_t math_type = select_conv_math_type(code, ctx.compile_options);

  // x_desc 承载真实 x,y_desc 承载 dy,w_desc 承载 dw(filter_shape)——角色
  // 复用见 setup_conv_descriptors 头注释。
  TensorDescGuard x_desc;
  FilterDescGuard dw_desc;
  TensorDescGuard dy_desc;
  ConvDescGuard conv_desc;
  const frame::Status setup_status =
      setup_conv_descriptors("conv2d_grad_filter_internal", params, data_type, math_type, x_desc,
                             dw_desc, dy_desc, conv_desc);
  if (!setup_status.is_ok()) return setup_status;

  cudnnConvolutionBwdFilterAlgoPerf_t perf_results[CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT];
  int returned_count = 0;
  FRAME_RETURN_IF_ERROR(cudnn_status(
      cudnnGetConvolutionBackwardFilterAlgorithm_v7(
          handle, x_desc.desc, dy_desc.desc, conv_desc.desc, dw_desc.desc,
          CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT, &returned_count, perf_results),
      "conv2d_grad_filter_internal cuda kernel: cudnnGetConvolutionBackwardFilterAlgorithm_v7"));
  const int chosen = pick_first_successful_algo_index(perf_results, returned_count);
  if (chosen < 0) {
    return frame::Status::make(frame::ErrorCode::kInternal,
                               "op 'conv2d_grad_filter_internal' cuda kernel: "
                               "cudnnGetConvolutionBackwardFilterAlgorithm_v7 returned no usable "
                               "algorithm");
  }
  const cudnnConvolutionBwdFilterAlgo_t algo = perf_results[chosen].algo;
  const size_t workspace_bytes = perf_results[chosen].memory;

  frame::hal::Allocator* allocator = cuda_backend->allocator(ctx.device);
  if (allocator == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInternal,
                               "op 'conv2d_grad_filter_internal' cuda kernel: allocator "
                               "unavailable for device '" +
                                   std::string(ctx.device.backend) + "'");
  }
  const frame::Result<std::shared_ptr<frame::Storage>> workspace_result =
      allocate_workspace_if_needed(*allocator, ctx.device, workspace_bytes);
  if (!workspace_result.is_ok()) return workspace_result.status();
  const std::shared_ptr<frame::Storage>& workspace_storage = workspace_result.value();
  void* workspace_ptr = workspace_storage != nullptr ? workspace_storage->data() : nullptr;

  const void* x_data = x.raw_data();
  const void* dy_data = dy.raw_data();
  void* dw_data = dw.raw_data();

  const float alpha = 1.0F;
  const float beta = 0.0F;
  const frame::Status run_status =
      cudnn_status(cudnnConvolutionBackwardFilter(handle, &alpha, x_desc.desc, x_data, dy_desc.desc,
                                                  dy_data, conv_desc.desc, algo, workspace_ptr,
                                                  workspace_bytes, &beta, dw_desc.desc, dw_data),
                   "conv2d_grad_filter_internal cuda kernel: cudnnConvolutionBackwardFilter");

  if (workspace_storage != nullptr) {
    const frame::Status sync_status = frame::backends::cuda::cuda_status(
        cudaStreamSynchronize(stream),
        "conv2d_grad_filter_internal cuda kernel: cudaStreamSynchronize before workspace "
        "release");
    return !run_status.is_ok() ? run_status : sync_status;
  }
  return run_status;
}

}  // namespace

FRAME_REGISTER_KERNEL("conv2d", frame::kCudaBackendName, conv2d_cuda_kernel);
FRAME_REGISTER_KERNEL("conv2d_grad_input_internal", frame::kCudaBackendName,
                      conv2d_grad_input_internal_cuda_kernel);
FRAME_REGISTER_KERNEL("conv2d_grad_filter_internal", frame::kCudaBackendName,
                      conv2d_grad_filter_internal_cuda_kernel);
