// dead_node_elimination pass 单测(src/compiler/passes/dead_node_elimination.cpp,
// ARCH-051,M8 决议点 F):从图输出反向可达标记,删除不可达且无 has_side_effect
// trait 的节点;graph_input 节点无条件保留(图签名不变式)。
//   1. golden:死节点删除 + 未用 graph_input 保留(同一个图形态一并覆盖,
//      testdata/dne_dead_node_and_unused_input_{input,expected}.txt——第二个
//      graph_input 全程未被任何节点引用,验证其被无条件保留)。
//   2. golden:kHasSideEffect 不可达节点保留(即便无消费者、未被 mark_output,
//      见 tests/cpp/compiler/pass_test_common.h::kSideEffectOpName;本用例
//      input==expected,回归含义见用例内注释)。
//   3. 幂等(SHOULD):golden 用例①连跑两次,dump_text 逐字节相同。
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

#include "golden_test_helpers.h"
#include "pass_test_common.h"

namespace {

using frame::Result;
using frame::compiler::Pass;
using frame::compiler::PassRegistry;
using frame::compiler::testing::ensure_pass_test_ops_registered;
using frame::compiler::testing::load_ir_text_file;
using frame::compiler::testing::run_pass_matches_golden;
using frame::ir::dump_text;
using frame::ir::Graph;

constexpr std::string_view kDeadNodeInputPath =
    "tests/cpp/compiler/testdata/dne_dead_node_and_unused_input_input.txt";
constexpr std::string_view kDeadNodeExpectedPath =
    "tests/cpp/compiler/testdata/dne_dead_node_and_unused_input_expected.txt";
constexpr std::string_view kSideEffectInputPath =
    "tests/cpp/compiler/testdata/dne_side_effect_retained_input.txt";
constexpr std::string_view kSideEffectExpectedPath =
    "tests/cpp/compiler/testdata/dne_side_effect_retained_expected.txt";

Result<std::unique_ptr<Pass>> make_dead_node_elimination_pass() {
  return PassRegistry::instance().create("dead_node_elimination");
}

// ---------------------------------------------------------------------------
// 1. golden:死节点删除 + 未用 graph_input 保留。
// ---------------------------------------------------------------------------

TEST(DeadNodeEliminationTest, DeadNodeIsDeletedAndUnusedGraphInputIsRetained) {
  EXPECT_TRUE(
      run_pass_matches_golden("dead_node_elimination", kDeadNodeInputPath, kDeadNodeExpectedPath));
}

// ---------------------------------------------------------------------------
// 2. golden:kHasSideEffect 不可达节点保留。
// ---------------------------------------------------------------------------

TEST(DeadNodeEliminationTest, HasSideEffectUnreachableNodeIsRetained) {
  ensure_pass_test_ops_registered();
  // input == expected:若副作用豁免条款失效,test_side_effect_op 节点(无消费者、
  // 未 mark_output)会被当成死节点删除,dump_text 就不再等于原图,golden 断言
  // 会失败——本用例正是靠这一点捕获该类回归。
  EXPECT_TRUE(run_pass_matches_golden("dead_node_elimination", kSideEffectInputPath,
                                      kSideEffectExpectedPath));
}

// ---------------------------------------------------------------------------
// 3. 幂等(SHOULD)。
// ---------------------------------------------------------------------------

TEST(DeadNodeEliminationTest, RunningTwiceIsIdempotent) {
  Result<Graph> loaded = load_ir_text_file(kDeadNodeInputPath);
  ASSERT_TRUE(loaded.is_ok()) << loaded.status().message();
  Graph graph = std::move(loaded.value());

  const Result<std::unique_ptr<Pass>> pass = make_dead_node_elimination_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();

  ASSERT_TRUE(pass.value()->run(graph).is_ok());
  const std::string first_run_text = dump_text(graph);

  ASSERT_TRUE(pass.value()->run(graph).is_ok());
  const std::string second_run_text = dump_text(graph);

  EXPECT_EQ(first_run_text, second_run_text);
}

}  // namespace
