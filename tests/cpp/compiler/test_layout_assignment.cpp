// layout_assignment pass 单测(src/compiler/passes/layout_assignment.cpp,
// ARCH-051,M9 决议点 A):v0 唯一具体 layout 是 kRowMajor,遍历全部 Value(含
// graph_input 与图输出)经 Graph::assign_layout 统一指派。
//   1. golden:matmul+relu 图,指派后 dump_text 全部 Value 类型后缀带
//      ":row_major"(testdata/layout_assignment_basic_{input,expected}.txt,
//      serialization.h 头注释第3a条)。
//   2. 幂等(assign_layout 对已是 kRowMajor 的 Value 幂等跳过,§3.1 后置
//      条件 MUST):同一图连跑两次,dump_text 逐字节相同。
//   3. graph_input 与图输出也被指派(Graph API 直接断言,而非仅经 dump_text
//      文本旁证):图输入 Value 与图输出 Value 的 TensorType::layout 均变为
//      kRowMajor。
// PassRegistry 是进程级 Meyer's singleton,本文件不新增任何算子注册,仅消费
// 已由 src/ 静态注册好的 "matmul"/"relu"。
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <frame/compiler/pass.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ir/serialization.h>

#include "../ir/ir_test_helpers.h"
#include "golden_test_helpers.h"

namespace {

using frame::Result;
using frame::compiler::Pass;
using frame::compiler::PassRegistry;
using frame::compiler::testing::load_ir_text_file;
using frame::compiler::testing::run_pass_matches_golden;
using frame::ir::dump_text;
using frame::ir::Graph;
using frame::ir::Layout;
using frame::ir::Node;
using frame::ir::Value;
using frame::ir::testing::MakeFloat32Type;

constexpr std::string_view kBasicInputPath =
    "tests/cpp/compiler/testdata/layout_assignment_basic_input.txt";
constexpr std::string_view kBasicExpectedPath =
    "tests/cpp/compiler/testdata/layout_assignment_basic_expected.txt";

Result<std::unique_ptr<Pass>> make_layout_assignment_pass() {
  return PassRegistry::instance().create("layout_assignment");
}

// ---------------------------------------------------------------------------
// 1. golden 基线对照。
// ---------------------------------------------------------------------------

TEST(LayoutAssignmentTest, AllValuesGetRowMajorSuffixInDumpText) {
  EXPECT_TRUE(run_pass_matches_golden("layout_assignment", kBasicInputPath, kBasicExpectedPath));
}

// ---------------------------------------------------------------------------
// 2. 幂等。
// ---------------------------------------------------------------------------

TEST(LayoutAssignmentTest, RunningTwiceIsIdempotent) {
  Result<Graph> loaded = load_ir_text_file(kBasicInputPath);
  ASSERT_TRUE(loaded.is_ok()) << loaded.status().message();
  Graph graph = std::move(loaded.value());

  const Result<std::unique_ptr<Pass>> pass = make_layout_assignment_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();

  ASSERT_TRUE(pass.value()->run(graph).is_ok());
  const std::string first_run_text = dump_text(graph);

  ASSERT_TRUE(pass.value()->run(graph).is_ok());
  const std::string second_run_text = dump_text(graph);

  EXPECT_EQ(first_run_text, second_run_text);
}

// ---------------------------------------------------------------------------
// 3. graph_input/图输出也被指派(Graph API 直接断言)。
// ---------------------------------------------------------------------------

TEST(LayoutAssignmentTest, GraphInputAndGraphOutputValuesAreAssignedRowMajor) {
  Graph graph;
  // 手工构造 kUnknown 起点(MakeFloat32Type 恒置 kRowMajor,layout_assignment
  // 的幂等分支会掩盖"未指派"与"已指派"的区别,故本用例不用它构造图输入)。
  frame::ir::TensorType unknown_type;
  unknown_type.dtype = MakeFloat32Type({4}).dtype;
  unknown_type.shape = MakeFloat32Type({4}).shape;
  unknown_type.device = MakeFloat32Type({4}).device;

  Value* input = graph.add_graph_input(unknown_type).value();
  ASSERT_EQ(input->type().layout, Layout::kUnknown);

  Node* node = graph.create_node("relu", {input}, {unknown_type}).value();
  ASSERT_TRUE(graph.mark_output(node->output(0)).is_ok());
  ASSERT_EQ(node->output(0)->type().layout, Layout::kUnknown);

  const Result<std::unique_ptr<Pass>> pass = make_layout_assignment_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();
  ASSERT_TRUE(pass.value()->run(graph).is_ok());

  EXPECT_EQ(input->type().layout, Layout::kRowMajor);            // graph_input
  EXPECT_EQ(node->output(0)->type().layout, Layout::kRowMajor);  // 图输出
  for (const Value* graph_input : graph.inputs()) {
    EXPECT_EQ(graph_input->type().layout, Layout::kRowMajor);
  }
  for (const Value* graph_output : graph.outputs()) {
    EXPECT_EQ(graph_output->type().layout, Layout::kRowMajor);
  }
}

}  // namespace
