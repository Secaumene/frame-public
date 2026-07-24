#pragma once
// Executable:整图编译产物的执行句柄(HAL 白名单虚函数,判定规则见
// include/frame/hal/backend.h 头部)。run 一次执行整图 —— 这是编译优先(铁律 #1①)
// 的落地产物,由 Backend::compile 产出、可被编译缓存复用。

#include <span>
#include <string_view>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/macros.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>

namespace frame::hal {

class Stream;  // 前向声明

// 输入/输出签名条目:dtype + shape,供编译缓存键与调用校验使用。
struct IoSpec {
  DType dtype{DTypeCode::kFloat32};
  Shape shape;
};

// Executable:整图编译单元。
class FRAME_API Executable {
 public:
  virtual ~Executable() = default;

  // 一次执行整图:inputs/outputs 由调用方按签名准备,stream 上异步排队。
  virtual Status run(std::span<const Tensor> inputs, std::span<Tensor> outputs, Stream& stream) = 0;

  // 编译时确定的输入/输出签名(dtype/shape 列表)。
  virtual std::vector<IoSpec> input_signature() const = 0;
  virtual std::vector<IoSpec> output_signature() const = 0;
};

// 跨后端公共设施(M10 交付物③裁决 = 抽取,design-reviewer REVISE 闭环修订
// 3):校验 span 尺寸与编译期签名(dtype+shape)逐位一致;label 由调用方拼接
// 完整上下文(如 "CpuExecutable::run: input"),供拼错误消息使用(ARCH-031
// 口径:不静默降级)。落点选在本头文件(而非 include/frame/runtime/):backends
// 与 runtime 均可依赖 hal,反之不成立;实现落 src/runtime/executable.cpp(后端
// target 已 PRIVATE 链接 frame::runtime,符号可解析)。outputs 侧调用方传入
// std::span<Tensor> 时隐式转换为 std::span<const Tensor> 成立。
FRAME_API Status validate_io_signature(std::span<const Tensor> values,
                                       const std::vector<IoSpec>& signature,
                                       std::string_view label);

// device 独立校验(M11,design-reviewer REVISE 闭环裁决修订3):对照调用方
// "自持"的期望 device(如某 Executable 编译期确定的目标设备)逐张量比对,
// 与 validate_io_signature 的 dtype/shape 校验彼此独立、互不替代——不给
// IoSpec 加 device 字段(会扰动全部聚合初始化点与编译缓存键)。落点与
// validate_io_signature 同一翻译单元(同一"跨后端公共设施"落点理由)。违例
// 返回英文错误,消息含张量下标、实际 device(backend+index)与期望 device
// (ARCH-031 口径:不静默降级)。
FRAME_API Status validate_tensor_devices(std::span<const Tensor> values, Device expected,
                                         std::string_view label);

}  // namespace frame::hal
