// memory_planning pass 单测(src/compiler/passes/memory_planning.cpp,
// ARCH-051,M9 决议点 D 覆盖版,§3.8 v0 口径):`Pass::run(Graph&)` 无产物
// 通道,职责收窄为"调 ir::compute_memory_plan 验证计划可计算性,成功不改图
// 放行"(真正落地在 CpuExecutable,见 include/frame/ir/memory_plan.h 与
// tests/cpp/ir/test_memory_plan.cpp 对 compute_memory_plan 本身的算法级覆盖;
// CpuExecutable 的 arena 端到端回归由既有 M7/M8 端到端用例在 arena 路径下
// 自动覆盖,见头注释④)。
//   1. golden:不改图(输入=期望逐字节相同,
//      testdata/memory_planning_no_change_{input,expected}.txt)。
//   2. 对无法规划的图(动态维中间 Value)报错、透传 compute_memory_plan 的
//      错误消息。本用例经直接构图 API + 直调 pass 断言错误,不走
//      run_pass_matches_golden——同 test_shape_inference.cpp
//      ::DynamicDimensionIsExplicitlyRejected 头注释"复核结论"已确认的理由:
//      pass->run() 从未成功过,没有可比对的"跑完后的图"可言,golden 契约
//      (运行成功后 dump_text 逐字比对)不适用于恒失败路径(不"报告后跳过"—
//      —该场景经直接构图 API 可构造,与 shape_inference 既有先例一致)。
//   3. 幂等(SHOULD,推论:①天然满足——不改图的 pass 连跑两次显然产出相同
//      dump_text):同一图连跑两次,dump_text 逐字节相同。
// PassRegistry 是进程级 Meyer's singleton,本文件不新增任何算子注册,golden
// 用例仅消费已由 src/ 静态注册好的 "add"/"relu";②的错误路径用例复用
// tests/cpp/ir/test_graph_mutation.cpp 既有先例(全放行场景,直接构图不依赖
// ops 层,ARCH-001——compute_memory_plan 本身也不依赖 OpQuery)。
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <frame/compiler/pass.h>
#include <frame/core/shape.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ir/serialization.h>

#include "../ir/ir_test_helpers.h"
#include "golden_test_helpers.h"

namespace {

using frame::ErrorCode;
using frame::kDynamicDim;
using frame::Result;
using frame::Status;
using frame::compiler::Pass;
using frame::compiler::PassRegistry;
using frame::compiler::testing::load_ir_text_file;
using frame::compiler::testing::run_pass_matches_golden;
using frame::ir::dump_text;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::TensorType;
using frame::ir::Value;
using frame::ir::testing::MakeFloat32Type;

constexpr std::string_view kNoChangeInputPath =
    "tests/cpp/compiler/testdata/memory_planning_no_change_input.txt";
constexpr std::string_view kNoChangeExpectedPath =
    "tests/cpp/compiler/testdata/memory_planning_no_change_expected.txt";

Result<std::unique_ptr<Pass>> make_memory_planning_pass() {
  return PassRegistry::instance().create("memory_planning");
}

// ---------------------------------------------------------------------------
// 1. golden:不改图。
// ---------------------------------------------------------------------------

TEST(MemoryPlanningTest, SuccessfulPlanDoesNotChangeGraph) {
  EXPECT_TRUE(
      run_pass_matches_golden("memory_planning", kNoChangeInputPath, kNoChangeExpectedPath));
}

// ---------------------------------------------------------------------------
// 2. 无法规划的图:动态维中间 Value 报错。
// ---------------------------------------------------------------------------

TEST(MemoryPlanningTest, DynamicDimensionInIntermediateValueReturnsError) {
  Graph graph;
  Value* a = graph.add_graph_input(MakeFloat32Type({4})).value();
  TensorType dynamic_type = MakeFloat32Type({4});
  dynamic_type.shape = frame::Shape({kDynamicDim});
  // n1 的输出是中间 Value(非 graph_input、非图输出),命中
  // compute_memory_plan 的动态维检查分支(include/frame/ir/memory_plan.h
  // 头注释边界:graph_input/图输出一律排除出规划范围,故动态维必须放在真正
  // 待规划的中间 Value 上才能触达该检查)。
  Node* n1 = graph.create_node("step1", {a}, {dynamic_type}).value();
  Node* n2 = graph.create_node("step2", {n1->output(0)}, {MakeFloat32Type({4})}).value();
  ASSERT_TRUE(graph.mark_output(n2->output(0)).is_ok());

  const Result<std::unique_ptr<Pass>> pass = make_memory_planning_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();
  const Status status = pass.value()->run(graph);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("dynamic dimension"), std::string_view::npos);
  EXPECT_NE(status.message().find("step1"), std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 3. 幂等。
// ---------------------------------------------------------------------------

TEST(MemoryPlanningTest, RunningTwiceIsIdempotent) {
  Result<Graph> loaded = load_ir_text_file(kNoChangeInputPath);
  ASSERT_TRUE(loaded.is_ok()) << loaded.status().message();
  Graph graph = std::move(loaded.value());

  const Result<std::unique_ptr<Pass>> pass = make_memory_planning_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();

  ASSERT_TRUE(pass.value()->run(graph).is_ok());
  const std::string first_run_text = dump_text(graph);

  ASSERT_TRUE(pass.value()->run(graph).is_ok());
  const std::string second_run_text = dump_text(graph);

  EXPECT_EQ(first_run_text, second_run_text);
}

}  // namespace
