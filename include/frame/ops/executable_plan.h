#pragma once
// ExecutablePlan:整图编译期共享的执行计划组件(M11,design-reviewer REVISE
// 闭环裁决修订1)。从 CpuExecutable::compile 抽取的通用部分:
//   ①slot 表构建(图输入 + 各节点输出,按拓扑序)+
//   ②ir::compute_memory_plan 落地(arena 偏移/总字节数)+
//   ③KernelFn 解析(KernelRegistry::find(op, backend),缺失翻译为
//     kUnimplemented,消息含算子名与后端名,ARCH-031 同源:与
//     src/compiler/passes/backend_lowering.cpp 的翻译落点同一套约定)+
//   ④输入/输出签名(用 ir::TensorType,禁用 hal::IoSpec)+
//   ⑤输出 slot 映射。
// 落点裁决(design-reviewer REVISE 闭环):落 ops 层且**零 hal include**——
// KernelFn 解析属 ops 职责(kernel_registry.h 本身对 hal::Stream 仅前向声明,
// 本头同一先例),规格字段一律用 ir::TensorType 而非 hal::IoSpec,避免 ops 头
// 反向 include hal 破坏既有依赖倒置(ARCH-001)。IoSpec 转换与 run() 执行留给
// 各消费方(CpuExecutable/CudaExecutable)自行处理(FallbackExecutable「共享
// 设施 + run 分离」先例)。
// 复用者:src/backends/cpu/cpu_executable.cpp、src/backends/cuda/
// cuda_executable.cpp(M11)。

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <frame/core/macros.h>
#include <frame/core/status.h>
#include <frame/ir/attribute.h>
#include <frame/ir/node.h>
#include <frame/ops/kernel_registry.h>

namespace frame::ir {
class Graph;  // 前向声明:见 include/frame/ir/graph.h
}  // namespace frame::ir

namespace frame::ops {

// 单个非 graph_input 节点的执行步骤:编译期自持全部所需信息,不回指 Graph
// (与 CpuExecutable::Step/FallbackExecutable::Step 同一惯例)。
struct ExecutablePlanStep {
  std::string op;                                        // 算子名(拼错误消息用)
  KernelFn kernel = nullptr;                             // 编译期解析的内核函数指针
  std::unordered_map<std::string, ir::AttrValue> attrs;  // 属性值拷贝
  std::vector<int32_t> input_slots;                      // 按位输入 slot 索引
  std::vector<int32_t> output_slots;                     // 按位输出 slot 索引
  std::vector<ir::TensorType> output_types;              // 按位输出类型(dtype/shape/layout/device)
  // 按位与 output_slots/output_types 平行:该输出在 arena 内的字节偏移。有值 =
  // 中间 Value,消费方经此偏移从 arena Storage 切片;nullopt = 该输出是图输出
  // Value(compute_memory_plan 已排除,见 include/frame/ir/memory_plan.h 头
  // 注释边界),消费方仍需独立分配。
  std::vector<std::optional<size_t>> output_arena_offsets;
};

// 整图执行计划:纯数据,不含 Allocator/Device 等运行期资源(消费方按自身需要
// 结合这些资源执行)。
struct ExecutablePlan {
  std::vector<ir::TensorType> input_types;   // 按 graph.inputs() 序
  std::vector<ir::TensorType> output_types;  // 按 graph.outputs() 序
  std::vector<int32_t> output_slots;         // 按 graph.outputs() 序,记录各图输出对应的 slot 索引
  std::vector<ExecutablePlanStep> steps;     // 按拓扑序排列的执行步骤
  int32_t slot_count = 0;
  size_t arena_total_bytes = 0;  // 由 ir::compute_memory_plan 算好的 arena 总大小
};

// 构建 graph 针对 backend 的执行计划(见本文件头注释①-⑤)。backend 仅用于
// KernelRegistry 查找键,不做设备/后端合法性校验(该职责留给调用方,如
// CpuBackend::compile 对 device 的校验)。
FRAME_API Result<ExecutablePlan> build_executable_plan(const ir::Graph& graph,
                                                       std::string_view backend);

}  // namespace frame::ops
