// ir::compute_memory_plan 单测(M9,决议点 D,include/frame/ir/memory_plan.h)。
// 纯图分析函数,不依赖 OpRegistry/OpQuery(compute_memory_plan 只读 Graph 结构
// 与各 Value 的 TensorType,不调用 graph.verify()),故本文件全程使用未注册的
// 占位 op 名构图,同 test_graph_mutation.cpp 既有先例(全放行场景,不依赖 ops
// 层)。全程纯主机内存,不依赖任何已注册后端。
//
// 覆盖点:
//   1. 生命周期正确性(def_index/last_use_index/size_bytes 精确匹配)。
//   2. 确定性(同一图两次调用逐字段相等)。
//   3. graph_input 与图输出(含"既是图输出又被内部消费")不入 plan。
//   4. 生命周期相交 -> offset 区间不重叠 / 不相交 -> offset 复用 / 峰值
//      total_bytes < 朴素总和(共用一个 4 中间值链式图 fixture,推导见该
//      fixture 上方注释)。
//   5. 零字节中间 Value 保留 entry,但 arena 总字节数为 0。
//   6. 动态维中间 Value 报错(ARCH-013)。
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include <frame/core/dtype.h>
#include <frame/core/macros.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/ir/graph.h>
#include <frame/ir/memory_plan.h>
#include <frame/ir/node.h>

#include "ir_test_helpers.h"

namespace {

using frame::ErrorCode;
using frame::kDynamicDim;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::ir::compute_memory_plan;
using frame::ir::Graph;
using frame::ir::MemoryPlan;
using frame::ir::MemoryPlanKey;
using frame::ir::Node;
using frame::ir::TensorType;
using frame::ir::Value;
using frame::ir::testing::MakeFloat32Type;

// shape{16} float32 = 16 * 4 = 64 字节,恰为 frame::kDefaultAlignment(64)
// 的整数倍,offset 断言无需额外处理取整,聚焦 lifetime/复用本身。
constexpr int64_t kElemCount = 16;
constexpr size_t kValueBytes = 64;

TensorType MakeChainType() { return MakeFloat32Type({kElemCount}); }

// ---------------------------------------------------------------------------
// 1. 生命周期正确性:a(idx0) -> n1(idx1,中间) -> n2(idx2,图输出)。
// ---------------------------------------------------------------------------

TEST(ComputeMemoryPlanTest, IntermediateValueLifetimeMatchesDefAndLastUseIndices) {
  Graph graph;
  Value* a = graph.add_graph_input(MakeChainType()).value();
  Node* n1 = graph.create_node("step1", {a}, {MakeChainType()}).value();
  Node* n2 = graph.create_node("step2", {n1->output(0)}, {MakeChainType()}).value();
  ASSERT_TRUE(graph.mark_output(n2->output(0)).is_ok());

  const Result<MemoryPlan> plan_result = compute_memory_plan(graph);
  ASSERT_TRUE(plan_result.is_ok()) << plan_result.status().message();
  const MemoryPlan& plan = plan_result.value();

  // 仅 n1 的输出是中间 Value(n2 的输出是图输出,被排除)。
  ASSERT_EQ(plan.entries.size(), 1u);
  const auto it = plan.entries.find(MemoryPlanKey{1, 0});
  ASSERT_NE(it, plan.entries.end());
  EXPECT_EQ(it->second.def_index, 1);
  EXPECT_EQ(it->second.last_use_index, 2);  // 被 n2(idx2)消费
  EXPECT_EQ(it->second.size_bytes, kValueBytes);
  EXPECT_EQ(it->second.byte_offset, 0u);  // 首个分配,无在存活区间
  EXPECT_EQ(plan.total_bytes, kValueBytes);
}

// ---------------------------------------------------------------------------
// 2. 确定性:同一图两次调用逐字段相等(纯函数,无隐藏状态)。
// ---------------------------------------------------------------------------

TEST(ComputeMemoryPlanTest, ResultIsDeterministicAcrossRepeatedCallsOnSameGraph) {
  Graph graph;
  Value* a = graph.add_graph_input(MakeChainType()).value();
  Node* n1 = graph.create_node("step1", {a}, {MakeChainType()}).value();
  Node* n2 = graph.create_node("step2", {n1->output(0)}, {MakeChainType()}).value();
  Node* n3 = graph.create_node("step3", {n2->output(0)}, {MakeChainType()}).value();
  ASSERT_TRUE(graph.mark_output(n3->output(0)).is_ok());

  const Result<MemoryPlan> first_result = compute_memory_plan(graph);
  const Result<MemoryPlan> second_result = compute_memory_plan(graph);
  ASSERT_TRUE(first_result.is_ok()) << first_result.status().message();
  ASSERT_TRUE(second_result.is_ok()) << second_result.status().message();
  const MemoryPlan& first = first_result.value();
  const MemoryPlan& second = second_result.value();

  EXPECT_EQ(first.total_bytes, second.total_bytes);
  ASSERT_EQ(first.entries.size(), second.entries.size());
  // MemoryPlanEntry 未定义 operator==(聚合体,无必要为规划结果之外的场景添加
  // 比较运算符),逐字段手工比对(std::map 按键有序,两次调用键集合恒等,
  // 可直接按迭代序配对比较)。
  auto first_it = first.entries.begin();
  auto second_it = second.entries.begin();
  for (; first_it != first.entries.end(); ++first_it, ++second_it) {
    EXPECT_EQ(first_it->first, second_it->first);
    EXPECT_EQ(first_it->second.byte_offset, second_it->second.byte_offset);
    EXPECT_EQ(first_it->second.size_bytes, second_it->second.size_bytes);
    EXPECT_EQ(first_it->second.def_index, second_it->second.def_index);
    EXPECT_EQ(first_it->second.last_use_index, second_it->second.last_use_index);
  }
}

// ---------------------------------------------------------------------------
// 3. graph_input 与图输出不入 plan(含"既是图输出又被内部消费"情形)。
// ---------------------------------------------------------------------------

TEST(ComputeMemoryPlanTest, GraphInputIsNeverPlanned) {
  Graph graph;
  Value* a = graph.add_graph_input(MakeChainType()).value();
  Node* n1 = graph.create_node("step1", {a}, {MakeChainType()}).value();
  ASSERT_TRUE(graph.mark_output(n1->output(0)).is_ok());

  const Result<MemoryPlan> plan_result = compute_memory_plan(graph);
  ASSERT_TRUE(plan_result.is_ok()) << plan_result.status().message();
  // a 是 graph_input(idx0),n1 的输出是图输出(idx1)——两者均被排除,plan
  // 为空。
  EXPECT_TRUE(plan_result.value().entries.empty());
  EXPECT_EQ(plan_result.value().total_bytes, 0u);
}

TEST(ComputeMemoryPlanTest, GraphOutputThatIsAlsoInternallyConsumedIsExcludedFromPlan) {
  // n1 的输出既是图输出、又被 n2 内部消费(边界情形,§3.8 v0 口径:凡属图
  // 输出列表一律排除,不因"仍被内部消费"而入 plan)。
  Graph graph;
  Value* a = graph.add_graph_input(MakeChainType()).value();
  Node* n1 = graph.create_node("step1", {a}, {MakeChainType()}).value();
  Node* n2 = graph.create_node("step2", {n1->output(0)}, {MakeChainType()}).value();
  ASSERT_TRUE(graph.mark_output(n1->output(0)).is_ok());
  ASSERT_TRUE(graph.mark_output(n2->output(0)).is_ok());

  const Result<MemoryPlan> plan_result = compute_memory_plan(graph);
  ASSERT_TRUE(plan_result.is_ok()) << plan_result.status().message();
  // n1 的输出(既是图输出)与 n2 的输出(图输出)均被排除,plan 为空。
  EXPECT_TRUE(plan_result.value().entries.empty());
  EXPECT_EQ(plan_result.value().total_bytes, 0u);
}

// ---------------------------------------------------------------------------
// 4. 相交/不相交/峰值:共用一个 4 中间值链式图。
//
// 图结构:a(idx0) -> n1(idx1,产出 v1) -> n2(idx2,消费 v1、产出 v2) ->
// n3(idx3,消费 v2、产出 v3) -> n4(idx4,消费 v3、产出 v4,图输出)。
// v1..v3 均为中间 Value,各占 kValueBytes(64)字节。
//
// 生命周期:v1=[def1,last_use2](被 n2 消费)、v2=[def2,last_use3](被 n3
// 消费)、v3=[def3,last_use4](被 n4 消费)。逐值按 def 顺序贪心 first-fit
// 手工推演(算法见 src/ir/memory_plan.cpp compute_memory_plan 实现注释):
//   v1(def1): active 空 -> offset=0;active=[0,64)@last_use2。
//   v2(def2): v1 的 last_use(2)==def2,不早于故不释放(v0"无原位"边界依据:
//     n2 执行期仍读 v1 作为输入)-> 与 [0,64) 相交,offset 取 64。
//     active=[0,64)@2,[64,128)@3。
//   v3(def3): v1 的 last_use(2)<def3(3),严格小于 -> 释放 [0,64);
//     [64,128)@3 的 last_use==3,不释放。剩余 active=[64,128)@3;新值 64
//     字节可放入偏移 0(候选 0+64<=64,命中区间前空隙)-> offset=0(复用 v1
//     的偏移)。
// 故 v1/v3 offset 相同(0,验证"不相交则复用"),v1/v2 offset 不同且区间不
// 重叠(验证"相交则不重叠"),total_bytes=128 < 朴素总和 3*64=192(验证
// 峰值优于朴素方案)。
// ---------------------------------------------------------------------------

struct MemoryPlanChainFixture {
  Graph graph;
  Node* n1 = nullptr;  // 产出 v1,topo idx1
  Node* n2 = nullptr;  // 产出 v2,topo idx2
  Node* n3 = nullptr;  // 产出 v3,topo idx3
};

MemoryPlanChainFixture BuildMemoryPlanChainFixture() {
  MemoryPlanChainFixture fixture;
  Value* a = fixture.graph.add_graph_input(MakeChainType()).value();
  fixture.n1 = fixture.graph.create_node("step1", {a}, {MakeChainType()}).value();
  fixture.n2 =
      fixture.graph.create_node("step2", {fixture.n1->output(0)}, {MakeChainType()}).value();
  fixture.n3 =
      fixture.graph.create_node("step3", {fixture.n2->output(0)}, {MakeChainType()}).value();
  Node* n4 = fixture.graph.create_node("step4", {fixture.n3->output(0)}, {MakeChainType()}).value();
  const Status mark_status = fixture.graph.mark_output(n4->output(0));
  FRAME_CHECK(mark_status.is_ok());
  return fixture;
}

TEST(ComputeMemoryPlanTest, OverlappingLifetimesGetNonOverlappingOffsetRanges) {
  MemoryPlanChainFixture fixture = BuildMemoryPlanChainFixture();
  const Result<MemoryPlan> plan_result = compute_memory_plan(fixture.graph);
  ASSERT_TRUE(plan_result.is_ok()) << plan_result.status().message();
  const MemoryPlan& plan = plan_result.value();

  const auto v1_it = plan.entries.find(MemoryPlanKey{1, 0});
  const auto v2_it = plan.entries.find(MemoryPlanKey{2, 0});
  ASSERT_NE(v1_it, plan.entries.end());
  ASSERT_NE(v2_it, plan.entries.end());
  // v1=[def1,last_use2],v2=[def2,last_use3]:在 idx2 处同时存活(n2 既消费 v1
  // 又产出 v2),区间必须不重叠。
  const size_t v1_begin = v1_it->second.byte_offset;
  const size_t v1_end = v1_begin + v1_it->second.size_bytes;
  const size_t v2_begin = v2_it->second.byte_offset;
  const size_t v2_end = v2_begin + v2_it->second.size_bytes;
  EXPECT_TRUE(v1_end <= v2_begin || v2_end <= v1_begin)
      << "v1=[" << v1_begin << "," << v1_end << ") v2=[" << v2_begin << "," << v2_end << ")";
}

TEST(ComputeMemoryPlanTest, DisjointLifetimesReuseTheSameOffset) {
  MemoryPlanChainFixture fixture = BuildMemoryPlanChainFixture();
  const Result<MemoryPlan> plan_result = compute_memory_plan(fixture.graph);
  ASSERT_TRUE(plan_result.is_ok()) << plan_result.status().message();
  const MemoryPlan& plan = plan_result.value();

  const auto v1_it = plan.entries.find(MemoryPlanKey{1, 0});
  const auto v3_it = plan.entries.find(MemoryPlanKey{3, 0});
  ASSERT_NE(v1_it, plan.entries.end());
  ASSERT_NE(v3_it, plan.entries.end());
  // v1 的 last_use(2)严格早于 v3 的 def(3),生命周期不相交,first-fit 应
  // 复用同一 offset(手工推演见本节顶部注释)。
  EXPECT_EQ(v1_it->second.byte_offset, v3_it->second.byte_offset);
}

TEST(ComputeMemoryPlanTest, PeakAllocationIsBelowNaiveSumForNonOverlappingChain) {
  MemoryPlanChainFixture fixture = BuildMemoryPlanChainFixture();
  const Result<MemoryPlan> plan_result = compute_memory_plan(fixture.graph);
  ASSERT_TRUE(plan_result.is_ok()) << plan_result.status().message();
  const MemoryPlan& plan = plan_result.value();

  ASSERT_EQ(plan.entries.size(), 3u);
  size_t naive_sum = 0;
  for (const auto& [key, entry] : plan.entries) {
    (void)key;
    naive_sum += entry.size_bytes;
  }
  ASSERT_EQ(naive_sum, 3 * kValueBytes);  // 3 个中间 Value,各 64 字节
  // 手工推演(本节顶部注释)给出确定性精确值 128,而非仅一个宽松上界——
  // 算法是纯确定性贪心 first-fit,无需类似 BUILD-011 的数值容差。
  EXPECT_EQ(plan.total_bytes, 128u);
  EXPECT_LT(plan.total_bytes, naive_sum);
}

// ---------------------------------------------------------------------------
// 5. 零字节中间 Value 仍须保留 entry,供 ExecutablePlan 区分中间量与图输出。
// ---------------------------------------------------------------------------

TEST(ComputeMemoryPlanTest, ZeroByteIntermediateKeepsEntryWithoutAllocatingArena) {
  TensorType empty_type = MakeFloat32Type({0, 3});
  Graph graph;
  Value* input = graph.add_graph_input(empty_type).value();
  Node* intermediate = graph.create_node("empty_step", {input}, {empty_type}).value();
  Node* output = graph.create_node("empty_output", {intermediate->output(0)}, {empty_type}).value();
  ASSERT_TRUE(graph.mark_output(output->output(0)).is_ok());

  const Result<MemoryPlan> plan_result = compute_memory_plan(graph);
  ASSERT_TRUE(plan_result.is_ok()) << plan_result.status().message();
  const MemoryPlan& plan = plan_result.value();
  ASSERT_EQ(plan.entries.size(), 1U);
  const auto entry = plan.entries.find(MemoryPlanKey{1, 0});
  ASSERT_NE(entry, plan.entries.end());
  EXPECT_EQ(entry->second.byte_offset, 0U);
  EXPECT_EQ(entry->second.size_bytes, 0U);
  EXPECT_EQ(plan.total_bytes, 0U);
}

// ---------------------------------------------------------------------------
// 6. 动态维中间 Value 报错(ARCH-013)。graph_input/图输出的动态维由 pass 之前
//    的 shape_inference(V5)负责拦截,本函数只负责"待规划的中间 Value"这一
//    子集——故本用例把动态维放在中间 Value(非 graph_input、非图输出)上,
//    才能命中 compute_memory_plan 自身的检查分支。
// ---------------------------------------------------------------------------

TEST(ComputeMemoryPlanTest, DynamicDimensionInIntermediateValueIsRejected) {
  Graph graph;
  Value* a = graph.add_graph_input(MakeChainType()).value();
  TensorType dynamic_type = MakeChainType();
  dynamic_type.shape = Shape({kDynamicDim});
  Node* n1 = graph.create_node("step1", {a}, {dynamic_type}).value();
  Node* n2 = graph.create_node("step2", {n1->output(0)}, {MakeChainType()}).value();
  ASSERT_TRUE(graph.mark_output(n2->output(0)).is_ok());

  const Result<MemoryPlan> plan_result = compute_memory_plan(graph);
  ASSERT_FALSE(plan_result.is_ok());
  EXPECT_EQ(plan_result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(plan_result.status().message().find("dynamic dimension"), std::string::npos);
  EXPECT_NE(plan_result.status().message().find("step1"), std::string::npos);
}

}  // namespace
