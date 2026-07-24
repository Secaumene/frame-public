// Executable 整图编译产物执行句柄的实现单元。
// 现状:Executable 是纯虚 HAL 接口,具体执行句柄由各后端在 src/backends/<name>/ 内
// 实现(为 Backend::compile 的产物);本翻译单元的公共非虚实现见下方
// validate_io_signature。
//
// M10 裁决(design-reviewer REVISE 闭环修订 3,销此前的 FRAME-IMPL 类待办):
// CpuExecutable 的签名校验逻辑(span 尺寸与 dtype/shape 逐位比对,此前落于
// src/backends/cpu/cpu_executable.cpp 匿名命名空间的
// validate_span_matches_signature 模板)在 FallbackExecutable(src/runtime/
// fallback_executable.cpp)成为第二个同构使用者后触发 REUSE-002 抽取;落点
// 裁定为本翻译单元(声明在 include/frame/hal/executable.h),而非新建
// src/hal/ ——后端 target 已 PRIVATE 链接 frame::runtime(见
// cmake/frame_backend.cmake),符号可解析,与 BackendRegistry 抽取先例一致。

#include <cstddef>
#include <string>

#include <frame/hal/executable.h>

namespace frame::hal {

Status validate_io_signature(std::span<const Tensor> values, const std::vector<IoSpec>& signature,
                             std::string_view label) {
  if (values.size() != signature.size()) {
    return Status::make(ErrorCode::kInvalidArgument, std::string(label) + " count mismatch, got " +
                                                         std::to_string(values.size()) +
                                                         ", expected " +
                                                         std::to_string(signature.size()));
  }
  for (size_t i = 0; i < signature.size(); ++i) {
    const Tensor& tensor = values[i];
    if (!(tensor.dtype() == signature[i].dtype) || !(tensor.shape() == signature[i].shape)) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          std::string(label) + " " + std::to_string(i) + " signature mismatch, got dtype '" +
              std::string(tensor.dtype().name()) + "' shape " + tensor.shape().to_string() +
              ", expected dtype '" + std::string(signature[i].dtype.name()) + "' shape " +
              signature[i].shape.to_string());
    }
  }
  return Status::ok();
}

Status validate_tensor_devices(std::span<const Tensor> values, Device expected,
                               std::string_view label) {
  for (size_t i = 0; i < values.size(); ++i) {
    const Device actual = values[i].device();
    if (!(actual == expected)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          std::string(label) + " " + std::to_string(i) +
                              " device mismatch, got backend '" + std::string(actual.backend) +
                              "' index " + std::to_string(actual.index) + ", expected backend '" +
                              std::string(expected.backend) + "' index " +
                              std::to_string(expected.index));
    }
  }
  return Status::ok();
}

}  // namespace frame::hal
