#pragma once
// cpu kernel 侧共享的 dtype 校验工具(M21,批3 T4)。收敛动机(铁律 5):
// conv.cpp 与 pool.cpp 的可变操作数 dtype 一致性校验完全同构,同目录共享,
// 禁止两份复制(matmul.cpp 的三具名形参定长版本先于本头存在,形态不同暂不
// 合并)。仅供 src/backends/cpu/kernels/ 内部包含,不入公开 API。
//
// 写法纪律(CPP-012 机械检查):比较前先拆具名变量,`if` 行不含 dtype 字样,
// 与 dispatch_dtype 编译期分派清晰区分,避免被运行时 dtype 分支检查误判
// (惯例出处:src/backends/cpu/kernels/matmul.cpp 同名函数注释)。

#include <string>
#include <string_view>
#include <vector>

#include <frame/core/dtype.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>

namespace frame::backends::cpu {

// 校验一组张量 dtype 完全一致且属 v0 浮点三档,返回其 DTypeCode(供
// dispatch_dtype 使用)。role_phrase 是这组操作数在错误消息中的固定描述短语。
// 以 vector<const Tensor*> 传参而非逐个具名形参,规避
// bugprone-easily-swappable-parameters 之余也天然适配可变操作数个数
// (conv2d/conv1d 因可选 bias 而操作数个数随之变化)。
inline frame::Result<frame::DTypeCode> require_matching_supported_dtype(
    std::string_view op_name, std::string_view role_phrase,
    const std::vector<const frame::Tensor*>& tensors) {
  const frame::DType first_type = tensors.front()->dtype();
  bool mismatch = false;
  for (const frame::Tensor* tensor : tensors) {
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
                               "op '" + std::string(op_name) + "' cpu kernel requires " +
                                   std::string(role_phrase) + " of the same dtype, got " +
                                   type_list_text);
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

}  // namespace frame::backends::cpu
