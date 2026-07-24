#pragma once
// MemoryPlan:静态 shape 下的中间 Value 缓冲区复用与生命周期规划(M9,决议点
// D)。纯图分析:输入 Graph、输出确定性 MemoryPlan,ir 层仅依赖 core
// (ARCH-001)。分层约束驱动的落点选择:backends 不得 include compiler,而
// compiler(memory_planning pass)与 backends/cpu(CpuExecutable)都需要本算法,
// 落在 ir 层新公共头,二者共用同一份实现(REUSE-002 单份)。
//
// v0 口径(§3.8 随件修订):图输入(graph_input)与凡属图输出列表的 Value 一律
// 排除出规划范围——图输入内存归调用方所有;图输出(含"既是图输出又被内部
// 消费"的情形)统一走独立缓冲、生命周期覆盖整个 run(),内部消费者直接读该
// 独立缓冲,不入 arena。仅中间 Value(非 graph_input 节点的输出、且不在图
// 输出列表)参与规划。v0 无原位/别名:全部 kernel 分配新输出,不复用输入
// 缓冲区(成立依据见本文件 compute_memory_plan 实现注释)。

#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>

#include <frame/core/macros.h>
#include <frame/core/status.h>

namespace frame::ir {

class Graph;  // 前向声明

// 单个中间 Value 的缓冲区分配条目。
struct MemoryPlanEntry {
  size_t byte_offset = 0;      // arena 内字节偏移,按 frame::kDefaultAlignment(64)向上取整
  size_t size_bytes = 0;       // 该 Value 占用字节数(shape numel × dtype itemsize)
  int64_t def_index = 0;       // 生命周期起点:producer 的拓扑序下标
  int64_t last_use_index = 0;  // 生命周期终点:最大消费者拓扑序下标(无消费者时等于 def_index)
};

// 规划条目键 = (producer 拓扑序下标, output_index) 数字对——与 CSE 等价键同
// 纪律(见 docs/architecture/compiler-passes.md §3.4),不用指针,保证跨调用
// 的确定性可比较。
using MemoryPlanKey = std::pair<int64_t, int32_t>;

// MemoryPlan:total_bytes 为 arena 总大小;entries 为按键有序(std::map)的各
// 中间 Value 分配条目映射,便于确定性遍历/断言。
struct MemoryPlan {
  size_t total_bytes = 0;
  std::map<MemoryPlanKey, MemoryPlanEntry> entries;
};

// 规划 graph 内全部中间 Value 的缓冲区偏移与生命周期(算法与边界见本文件头
// 注释);任一待规划 Value 的 shape 含动态维时返回错误(ARCH-013)。确定性:
// 同一图两次调用逐字段相等(纯函数,无隐藏状态)。定义见 src/ir/memory_plan.cpp。
FRAME_API Result<MemoryPlan> compute_memory_plan(const Graph& graph);

}  // namespace frame::ir
