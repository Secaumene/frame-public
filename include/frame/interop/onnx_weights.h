#pragma once
// ONNX 权重交换(ADR-0013,docs/decisions/0013-onnx-weight-exchange-minimal-codec.md):
// 读/写 ONNX ModelProto 中 graph.initializer 的 TensorProto 子集
// (name/dims/data_type/raw_data;dtype 覆盖 FLOAT/FLOAT16/BFLOAT16)+ 合法
// ModelProto 骨架(ir_version/opset_import/graph.name)。不做算子图导入导出。
//
// 依赖纪律(docs/architecture/overview.md §2.9):interop 仅依赖 core。
// hal::Allocator 仅前向声明,以引用形参出现(分配器由调用方注入),循
// include/frame/core/tensor.h 与 include/frame/ops/fused_elementwise_utils.h
// 两处既有先例——本文件不 include hal 头、不派生任何 hal 接口、不产生对
// hal/backends/runtime 的链接依赖(维护者对方案 b 的裁决)。

#include <span>
#include <string>
#include <vector>

#include <frame/core/macros.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>

namespace frame::hal {
class Allocator;  // 前向声明:interop 公共头不 include hal 头(见上方依赖纪律)
}  // namespace frame::hal

namespace frame::interop {

// 具名张量:ONNX initializer 的 name + 数据(host cpu 内存)。
struct NamedTensor {
  std::string name;
  Tensor tensor;
};

// 把 weights 写为一份最小合法 ONNX ModelProto(仅 graph.initializer 子集 +
// ir_version/opset_import/graph.name 骨架,不产生算子图节点)到 path。
// 每张量须 tensor.device().backend == "cpu",dtype 属 {float32, float16,
// bfloat16} 三值白名单,否则返回 kInvalidArgument(消息含张量名与实际值)。
// 打开/写入 path 失败分别返回 kInvalidArgument / kInternal(消息含路径)。
FRAME_API Status save_onnx_weights(const std::string& path, std::span<const NamedTensor> weights);

// 读取 path 处的 ONNX ModelProto,解析 graph.initializer 子集为 NamedTensor
// 列表(顺序与文件中出现顺序一致)。张量以 allocator 分配的 host cpu 内存
// 承载(device 固定为 frame::cpu_device();allocator 生命周期契约同
// Storage::allocate,见 include/frame/core/storage.h 头部注释①,由调用方
// 保证)。文件不存在返回 kNotFound;截断/字段越界/已废弃 groups(wire type
// 3/4)返回相应错误(消息含偏移或 wire type);子集外 data_type 返回
// kInvalidArgument(消息含该 ONNX DataType 枚举值)。
FRAME_API Result<std::vector<NamedTensor>> load_onnx_weights(const std::string& path,
                                                             hal::Allocator& allocator);

}  // namespace frame::interop
