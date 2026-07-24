// constant_folding pass 单测(src/compiler/passes/constant_folding.cpp,
// ARCH-051,M8 决议点 D):单遍拓扑序把仅依赖常量输入的子图折叠为常量节点。
//   1. golden:单节点折叠(两个常量 + add -> 单个折叠后常量,
//      testdata/constant_folding_single_node_{input,expected}.txt)。
//   2. golden:级联折叠(两层常量子图单遍折净——两个 add 节点全部消失,只剩
//      常量节点,testdata/constant_folding_cascade_{input,expected}.txt)。
//   3. kHasSideEffect 节点不折(即便全部输入均为常量,见
//      tests/cpp/compiler/pass_test_common.h::kSideEffectOpName)。
//   4. 无 cpu kernel 的算子跳过不折、不报错(schema 已注册但未注册 cpu
//      kernel,见 pass_test_common.h::kNoKernelOpName)。
//   5. 语义等价:同一图形态折叠前后经 runtime::compile("cpu") 编译执行,
//      数值一致(BUILD-011 容差,ARCH-052 精神——folding 是数值等价变换)。
//   6. 幂等(SHOULD,§3.3 决议点 D:级联单遍即折净,故天然幂等):级联图连跑
//      两次,dump_text 逐字节相同。
// PassRegistry/OpRegistry/KernelRegistry 均为进程级 Meyer's singleton;本文件
// 经 pass_test_common.h 幂等注册两个测试专用算子(kSideEffectOpName /
// kNoKernelOpName,REUSE-002:与其余三个 pass 测试文件共用同一份注册代码)。
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <frame/compiler/pass.h>
#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/executable.h>
#include <frame/hal/stream.h>
#include <frame/ir/attribute.h>
#include <frame/ir/graph.h>
#include <frame/ir/serialization.h>
#include <frame/ops/constant_utils.h>
#include <frame/runtime/compile.h>

#include "../common/tolerance.h"
#include "../ir/ir_test_helpers.h"
#include "golden_test_helpers.h"
#include "pass_test_common.h"

namespace {

using frame::DType;
using frame::DTypeCode;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::Tensor;
using frame::compiler::Pass;
using frame::compiler::PassRegistry;
using frame::compiler::testing::ensure_pass_test_ops_registered;
using frame::compiler::testing::kNoKernelOpName;
using frame::compiler::testing::kSideEffectOpName;
using frame::compiler::testing::load_ir_text_file;
using frame::compiler::testing::run_pass_matches_golden;
using frame::hal::CompileOptions;
using frame::hal::Executable;
using frame::ir::AttrValue;
using frame::ir::dump_text;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::testing::MakeFloat32Type;
using frame::ops::kConstantOpName;
using frame::testing::default_tolerance;
using frame::testing::tensor_all_close;

constexpr std::string_view kSingleNodeInputPath =
    "tests/cpp/compiler/testdata/constant_folding_single_node_input.txt";
constexpr std::string_view kSingleNodeExpectedPath =
    "tests/cpp/compiler/testdata/constant_folding_single_node_expected.txt";
constexpr std::string_view kCascadeInputPath =
    "tests/cpp/compiler/testdata/constant_folding_cascade_input.txt";
constexpr std::string_view kCascadeExpectedPath =
    "tests/cpp/compiler/testdata/constant_folding_cascade_expected.txt";

Result<std::unique_ptr<Pass>> make_constant_folding_pass() {
  return PassRegistry::instance().create("constant_folding");
}

Node* MakeConstant1D(Graph& graph, const std::vector<double>& values) {
  const frame::ir::TensorType type = MakeFloat32Type({static_cast<int64_t>(values.size())});
  Node* node = graph.create_node(std::string(kConstantOpName), {}, {type}).value();
  node->set_attr("value", AttrValue{values});
  node->set_attr("shape", AttrValue{type.shape});
  node->set_attr("dtype", AttrValue{type.dtype});
  return node;
}

// 构造 add(const_a, const_b) 的最小可折叠图,供语义等价用例调用两次(Graph
// move-only 不可拷贝,故每次都独立构造一份结构相同的新图,同
// tests/cpp/runtime/test_runtime_compile.cpp::BuildMatmulAddReluGraph 先例)。
Graph BuildFoldableAddGraph() {
  Graph graph("constant_folding_semantic_equivalence");
  Node* c1 = MakeConstant1D(graph, {1.5, 2.5});
  Node* c2 = MakeConstant1D(graph, {0.5, -1.0});
  Node* add_node =
      graph.create_node("add", {c1->output(0), c2->output(0)}, {MakeFloat32Type({2})}).value();
  graph.mark_output(add_node->output(0));
  return graph;
}

// ---------------------------------------------------------------------------
// 1/2. golden 用例。
// ---------------------------------------------------------------------------

TEST(ConstantFoldingTest, SingleFoldableNodeIsReplacedByConstant) {
  EXPECT_TRUE(
      run_pass_matches_golden("constant_folding", kSingleNodeInputPath, kSingleNodeExpectedPath));
}

TEST(ConstantFoldingTest, CascadedTwoLayerConstantSubgraphFoldsInSinglePass) {
  EXPECT_TRUE(run_pass_matches_golden("constant_folding", kCascadeInputPath, kCascadeExpectedPath));
}

// ---------------------------------------------------------------------------
// 3. kHasSideEffect 除外。
// ---------------------------------------------------------------------------

TEST(ConstantFoldingTest, HasSideEffectNodeIsNotFoldedEvenWithAllConstantInputs) {
  ensure_pass_test_ops_registered();

  Graph graph;
  Node* c1 = MakeConstant1D(graph, {1.0, 2.0});
  Node* side_effect_node =
      graph.create_node(std::string(kSideEffectOpName), {c1->output(0)}, {MakeFloat32Type({2})})
          .value();
  graph.mark_output(side_effect_node->output(0));

  const Result<std::unique_ptr<Pass>> pass = make_constant_folding_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();
  ASSERT_TRUE(pass.value()->run(graph).is_ok());

  bool side_effect_node_still_present = false;
  for (const Node* node : graph.topological_order()) {
    if (node->op() == kSideEffectOpName) side_effect_node_still_present = true;
  }
  EXPECT_TRUE(side_effect_node_still_present);
}

// ---------------------------------------------------------------------------
// 4. 无 cpu kernel 的算子跳过不折、不报错。
// ---------------------------------------------------------------------------

TEST(ConstantFoldingTest, OpWithoutCpuKernelIsSkippedWithoutError) {
  ensure_pass_test_ops_registered();

  Graph graph;
  Node* c1 = MakeConstant1D(graph, {1.0, 2.0});
  Node* no_kernel_node =
      graph.create_node(std::string(kNoKernelOpName), {c1->output(0)}, {MakeFloat32Type({2})})
          .value();
  graph.mark_output(no_kernel_node->output(0));

  const Result<std::unique_ptr<Pass>> pass = make_constant_folding_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();
  const Status status = pass.value()->run(graph);
  ASSERT_TRUE(status.is_ok()) << status.message();

  bool no_kernel_node_still_present = false;
  for (const Node* node : graph.topological_order()) {
    if (node->op() == kNoKernelOpName) no_kernel_node_still_present = true;
  }
  EXPECT_TRUE(no_kernel_node_still_present);
}

// ---------------------------------------------------------------------------
// 5. 语义等价。
// ---------------------------------------------------------------------------

TEST(ConstantFoldingTest, FoldedGraphExecutesToSameValueAsUnfoldedGraph) {
  const Graph before = BuildFoldableAddGraph();
  const Result<std::shared_ptr<Executable>> before_executable =
      frame::runtime::compile(before, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(before_executable.is_ok()) << before_executable.status().message();

  Graph after = BuildFoldableAddGraph();
  const Result<std::unique_ptr<Pass>> pass = make_constant_folding_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();
  ASSERT_TRUE(pass.value()->run(after).is_ok());
  const Result<std::shared_ptr<Executable>> after_executable =
      frame::runtime::compile(after, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(after_executable.is_ok()) << after_executable.status().message();

  const Result<frame::hal::Backend*> backend_result =
      frame::hal::BackendRegistry::instance().get(frame::kCpuBackendName);
  ASSERT_TRUE(backend_result.is_ok());
  frame::hal::Allocator* allocator = backend_result.value()->allocator(frame::cpu_device());
  ASSERT_NE(allocator, nullptr);
  const Result<std::unique_ptr<frame::hal::Stream>> stream_result =
      backend_result.value()->create_stream(frame::cpu_device());
  ASSERT_TRUE(stream_result.is_ok());

  std::vector<Tensor> no_inputs;
  std::vector<Tensor> before_outputs{
      Tensor::empty(Shape({2}), DType::of<float>(), frame::cpu_device(), *allocator).value()};
  std::vector<Tensor> after_outputs{
      Tensor::empty(Shape({2}), DType::of<float>(), frame::cpu_device(), *allocator).value()};

  ASSERT_TRUE(
      before_executable.value()->run(no_inputs, before_outputs, *stream_result.value()).is_ok());
  ASSERT_TRUE(
      after_executable.value()->run(no_inputs, after_outputs, *stream_result.value()).is_ok());

  EXPECT_TRUE(tensor_all_close(after_outputs[0], before_outputs[0],
                               default_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 6. 幂等(SHOULD)。
// ---------------------------------------------------------------------------

TEST(ConstantFoldingTest, RunningTwiceOnCascadeGraphIsIdempotent) {
  Result<Graph> loaded = load_ir_text_file(kCascadeInputPath);
  ASSERT_TRUE(loaded.is_ok()) << loaded.status().message();
  Graph graph = std::move(loaded.value());

  const Result<std::unique_ptr<Pass>> pass = make_constant_folding_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();

  ASSERT_TRUE(pass.value()->run(graph).is_ok());
  const std::string first_run_text = dump_text(graph);

  ASSERT_TRUE(pass.value()->run(graph).is_ok());
  const std::string second_run_text = dump_text(graph);

  EXPECT_EQ(first_run_text, second_run_text);
}

}  // namespace
