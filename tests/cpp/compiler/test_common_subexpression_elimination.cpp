// common_subexpression_elimination pass 单测
// (src/compiler/passes/common_subexpression_elimination.cpp,ARCH-051,M8 决议
// 点 E):合并等价节点(同 op、同输入、同属性;commutative trait 参与输入
// 归一化);带 has_side_effect trait 的节点除外;graph_input 节点不进表不被并
// (design-reviewer 必须修复项②)。
//   1. golden:重复子表达式合并(relu(x) 出现两次,后者合并进前者,
//      testdata/cse_duplicate_subexpr_{input,expected}.txt)。
//   2. golden:commutative 归一化(add(a,b) 与 add(b,a) 视为等价并合并,
//      testdata/cse_commutative_normalization_{input,expected}.txt)。
//   3. kHasSideEffect 节点除外:两个结构完全相同的
//      test_side_effect_op(x) 节点不得合并。
//   4. 两个同型 graph_input 不合并(图签名不变式,design-reviewer 必须修复
//      项②新增测试)。
//   5. 幂等(SHOULD):golden 用例②连跑两次,dump_text 逐字节相同。
// PassRegistry/OpRegistry/KernelRegistry 均为进程级 Meyer's singleton;本文件
// 经 pass_test_common.h 幂等注册 kSideEffectOpName(REUSE-002,与其余三个 pass
// 测试文件共用)。
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <frame/compiler/pass.h>
#include <frame/ir/graph.h>
#include <frame/ir/serialization.h>

#include "../ir/ir_test_helpers.h"
#include "golden_test_helpers.h"
#include "pass_test_common.h"

namespace {

using frame::Result;
using frame::compiler::Pass;
using frame::compiler::PassRegistry;
using frame::compiler::testing::ensure_pass_test_ops_registered;
using frame::compiler::testing::kSideEffectOpName;
using frame::compiler::testing::load_ir_text_file;
using frame::compiler::testing::run_pass_matches_golden;
using frame::ir::dump_text;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::Value;
using frame::ir::testing::MakeFloat32Type;

constexpr std::string_view kDuplicateInputPath =
    "tests/cpp/compiler/testdata/cse_duplicate_subexpr_input.txt";
constexpr std::string_view kDuplicateExpectedPath =
    "tests/cpp/compiler/testdata/cse_duplicate_subexpr_expected.txt";
constexpr std::string_view kCommutativeInputPath =
    "tests/cpp/compiler/testdata/cse_commutative_normalization_input.txt";
constexpr std::string_view kCommutativeExpectedPath =
    "tests/cpp/compiler/testdata/cse_commutative_normalization_expected.txt";

Result<std::unique_ptr<Pass>> make_cse_pass() {
  return PassRegistry::instance().create("common_subexpression_elimination");
}

// ---------------------------------------------------------------------------
// 1/2. golden 用例。
// ---------------------------------------------------------------------------

TEST(CommonSubexpressionEliminationTest, DuplicateSubexpressionIsMerged) {
  EXPECT_TRUE(run_pass_matches_golden("common_subexpression_elimination", kDuplicateInputPath,
                                      kDuplicateExpectedPath));
}

TEST(CommonSubexpressionEliminationTest, CommutativeInputOrderIsNormalizedBeforeMerging) {
  EXPECT_TRUE(run_pass_matches_golden("common_subexpression_elimination", kCommutativeInputPath,
                                      kCommutativeExpectedPath));
}

// ---------------------------------------------------------------------------
// 3. kHasSideEffect 除外。
// ---------------------------------------------------------------------------

TEST(CommonSubexpressionEliminationTest, HasSideEffectNodesAreNeverMergedEvenIfIdentical) {
  ensure_pass_test_ops_registered();

  Graph graph;
  Value* x = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* s1 = graph.create_node(std::string(kSideEffectOpName), {x}, {MakeFloat32Type({4})}).value();
  Node* s2 = graph.create_node(std::string(kSideEffectOpName), {x}, {MakeFloat32Type({4})}).value();
  ASSERT_TRUE(graph.mark_output(s1->output(0)).is_ok());
  ASSERT_TRUE(graph.mark_output(s2->output(0)).is_ok());

  const Result<std::unique_ptr<Pass>> pass = make_cse_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();
  ASSERT_TRUE(pass.value()->run(graph).is_ok());

  int side_effect_node_count = 0;
  for (const Node* node : graph.topological_order()) {
    if (node->op() == kSideEffectOpName) ++side_effect_node_count;
  }
  EXPECT_EQ(side_effect_node_count, 2);
  ASSERT_EQ(graph.outputs().size(), 2u);
  EXPECT_NE(graph.outputs()[0], graph.outputs()[1]);
}

// ---------------------------------------------------------------------------
// 4. 两个同型 graph_input 不合并。
// ---------------------------------------------------------------------------

TEST(CommonSubexpressionEliminationTest, TwoIdenticallyTypedGraphInputsAreNotMerged) {
  Graph graph;
  Value* x1 = graph.add_graph_input(MakeFloat32Type({4})).value();
  Value* x2 = graph.add_graph_input(MakeFloat32Type({4})).value();
  ASSERT_TRUE(graph.mark_output(x1).is_ok());
  ASSERT_TRUE(graph.mark_output(x2).is_ok());
  ASSERT_EQ(graph.inputs().size(), 2u);

  const Result<std::unique_ptr<Pass>> pass = make_cse_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();
  ASSERT_TRUE(pass.value()->run(graph).is_ok());

  // 图签名不变式:图输入个数不得因 CSE 而改变(design-reviewer 必须修复项②)。
  EXPECT_EQ(graph.inputs().size(), 2u);
  ASSERT_EQ(graph.outputs().size(), 2u);
  EXPECT_EQ(graph.outputs()[0], x1);
  EXPECT_EQ(graph.outputs()[1], x2);
}

// ---------------------------------------------------------------------------
// 5. 幂等(SHOULD)。
// ---------------------------------------------------------------------------

TEST(CommonSubexpressionEliminationTest, RunningTwiceIsIdempotent) {
  Result<Graph> loaded = load_ir_text_file(kCommutativeInputPath);
  ASSERT_TRUE(loaded.is_ok()) << loaded.status().message();
  Graph graph = std::move(loaded.value());

  const Result<std::unique_ptr<Pass>> pass = make_cse_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();

  ASSERT_TRUE(pass.value()->run(graph).is_ok());
  const std::string first_run_text = dump_text(graph);

  ASSERT_TRUE(pass.value()->run(graph).is_ok());
  const std::string second_run_text = dump_text(graph);

  EXPECT_EQ(first_run_text, second_run_text);
}

}  // namespace
