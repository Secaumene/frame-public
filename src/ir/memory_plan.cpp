// MemoryPlan 静态内存规划算法的实现单元(声明见
// include/frame/ir/memory_plan.h 头注释)。

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <frame/core/storage.h>  // frame::kDefaultAlignment
#include <frame/ir/graph.h>
#include <frame/ir/memory_plan.h>

namespace frame::ir {

namespace {

// 64 字节向上取整(kDefaultAlignment,见 include/frame/core/storage.h,命名
// 空间是 frame 而非 frame::core;裁决修订5:废弃 alignof(max_align_t)项)。
size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// 单个待规划 Value 的中间记录(尚未分配 offset,先收集 def/size,再统一按 def
// 顺序贪心分配)。
struct PlannedValue {
  MemoryPlanKey key;
  size_t size_bytes = 0;
  int64_t def_index = 0;
  int64_t last_use_index = 0;
};

}  // namespace

Result<MemoryPlan> compute_memory_plan(const Graph& graph) {
  // 图输出 Value 集合(O(1) 判定,覆盖"既是图输出又被内部消费"情形——凡属
  // 图输出一律排除出规划范围,决议点 D 边界,见头文件注释)。
  const std::unordered_set<const Value*> graph_outputs(graph.outputs().begin(),
                                                       graph.outputs().end());
  const std::vector<Node*>& topo = graph.topological_order();

  // ①收集待规划的中间 Value:非 graph_input 节点的每个输出、且不在图输出
  // 列表内。size_bytes 由 shape numel × dtype itemsize 计算,动态维报错
  // (ARCH-013)。溢出防线同 core/tensor.cpp::Tensor::empty 既有惯例。
  std::vector<PlannedValue> planned;
  std::unordered_map<const Value*, size_t> index_of_planned;

  for (size_t i = 0; i < topo.size(); ++i) {
    const Node* node = topo[i];
    if (node->op() == kGraphInputOp) continue;

    const std::vector<Value>& outputs = node->outputs();
    for (int32_t j = 0; j < static_cast<int32_t>(outputs.size()); ++j) {
      const Value* value = &outputs[static_cast<size_t>(j)];
      if (graph_outputs.find(value) != graph_outputs.end()) continue;  // 图输出排除

      const Shape& shape = value->type().shape;
      if (shape.has_dynamic_dim()) {
        return Status::make(ErrorCode::kInvalidArgument,
                            "compute_memory_plan: node '" + std::string(node->op()) + "' output " +
                                std::to_string(j) +
                                " has a dynamic dimension, static shape required (ARCH-013)");
      }
      const size_t numel = static_cast<size_t>(shape.numel());
      const size_t itemsize = value->type().dtype.itemsize();
      if (itemsize != 0 && numel > std::numeric_limits<size_t>::max() / itemsize) {
        return Status::make(ErrorCode::kInvalidArgument,
                            "compute_memory_plan: node '" + std::string(node->op()) + "' output " +
                                std::to_string(j) + " size overflows size_t");
      }
      const size_t size_bytes = numel * itemsize;

      index_of_planned.emplace(value, planned.size());
      planned.push_back(PlannedValue{MemoryPlanKey{static_cast<int64_t>(i), j}, size_bytes,
                                     static_cast<int64_t>(i), static_cast<int64_t>(i)});
    }
  }

  // ②扫描全部节点输入,登记每个被规划 Value 的 last_use(最大消费者拓扑
  // 下标);无消费者时保持①初值(等于 def_index)。
  for (size_t i = 0; i < topo.size(); ++i) {
    const Node* node = topo[i];
    for (const Value* input : node->inputs()) {
      const auto it = index_of_planned.find(input);
      if (it == index_of_planned.end()) continue;  // 非被规划 Value(图输入/图输出)
      PlannedValue& entry = planned[it->second];
      if (static_cast<int64_t>(i) > entry.last_use_index) {
        entry.last_use_index = static_cast<int64_t>(i);
      }
    }
  }

  // ③按 def 顺序(即①收集顺序,天然按拓扑序+输出序号递增)贪心 first-fit
  // 分配 offset:维护当前存活区间列表,先到期释放(last_use_index < 本值
  // def_index 的区间——严格小于:若恰好相等,说明该区间在本节点执行期间仍
  // 作为输入被读取,不得与本节点的新输出复用同一段内存,即 v0"无原位/别名"
  // 成立依据之一,另一半依据是全部 kernel 恒分配新输出、不写回输入),
  // 再从头扫描存活区间寻找首个能容纳本值的间隙,找不到则接在末尾;offset
  // 起点恒为 0 或某个已存活区间末尾按 64 对齐后的值,故全部产出 offset 天然
  // 是 kDefaultAlignment 的整数倍(裁决修订5)。
  struct ActiveInterval {
    size_t offset = 0;
    size_t end = 0;
    int64_t last_use_index = 0;
  };
  std::vector<ActiveInterval> active;

  MemoryPlan plan;
  for (const PlannedValue& value : planned) {
    active.erase(std::remove_if(active.begin(), active.end(),
                                [&](const ActiveInterval& interval) {
                                  return interval.last_use_index < value.def_index;
                                }),
                 active.end());
    std::sort(active.begin(), active.end(),
              [](const ActiveInterval& a, const ActiveInterval& b) { return a.offset < b.offset; });

    size_t candidate = 0;
    for (const ActiveInterval& interval : active) {
      if (candidate + value.size_bytes <= interval.offset) break;
      candidate = align_up(interval.end, kDefaultAlignment);
    }

    plan.entries.emplace(value.key, MemoryPlanEntry{candidate, value.size_bytes, value.def_index,
                                                    value.last_use_index});
    active.push_back(ActiveInterval{candidate, candidate + value.size_bytes, value.last_use_index});
    plan.total_bytes = std::max(plan.total_bytes, candidate + value.size_bytes);
  }

  return plan;
}

}  // namespace frame::ir
