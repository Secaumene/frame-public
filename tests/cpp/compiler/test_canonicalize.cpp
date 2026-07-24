// canonicalize pass 单测(src/compiler/passes/canonicalize.cpp,ARCH-051,M8
// 决议点 C):v0 规则集仅一条——「常量居右」:对带 kCommutative trait 的 2
// 输入节点,若输入 0 的 producer 是 constant 而输入 1 的不是,交换两输入。
//   1. golden:常量居左 -> 居右(add 是 kCommutative,见
//      testdata/canonicalize_constant_right_{input,expected}.txt)。
//   2. 幂等(§3.1 后置条件 MUST):同一 pass 实例连跑两次,dump_text 逐字节
//      相同(第二次运行时常量已在右侧,不再满足"输入 0 是常量"条件)。
//   3. 非 commutative 算子(matmul,src/ops/schemas/matmul.cpp 未标注任何
//      trait)不动:即便常量在输入 0,canonicalize 也不得交换。
// PassRegistry 是进程级 Meyer's singleton,本文件不新增任何算子注册(直接
// 复用已由 src/ 静态注册好的 "add"/"matmul"/"constant")。
#include <cstdint>
#include <gtest/gtest.h>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <frame/compiler/pass.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/ir/attribute.h>
#include <frame/ir/graph.h>
#include <frame/ir/serialization.h>
#include <frame/ops/constant_utils.h>

#include "../ir/ir_test_helpers.h"
#include "golden_test_helpers.h"

namespace {

using frame::Result;
using frame::Shape;
using frame::Status;
using frame::compiler::Pass;
using frame::compiler::PassRegistry;
using frame::compiler::testing::load_ir_text_file;
using frame::compiler::testing::run_pass_matches_golden;
using frame::ir::AttrValue;
using frame::ir::dump_text;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::Value;
using frame::ir::testing::MakeFloat32Type;
using frame::ops::kConstantOpName;

constexpr std::string_view kInputPath =
    "tests/cpp/compiler/testdata/canonicalize_constant_right_input.txt";
constexpr std::string_view kExpectedPath =
    "tests/cpp/compiler/testdata/canonicalize_constant_right_expected.txt";

Result<std::unique_ptr<Pass>> make_canonicalize_pass() {
  return PassRegistry::instance().create("canonicalize");
}

// 构造一个 float32 constant 节点(value 全 1.0,满足 fill_tensor_from_
// constant_attrs/shape_infer 的自洽性要求),供本文件直接构图的用例使用。
Node* MakeConstantNode(Graph& graph, std::initializer_list<int64_t> dims) {
  const frame::ir::TensorType type = MakeFloat32Type(dims);
  Node* node = graph.create_node(std::string(kConstantOpName), {}, {type}).value();
  const int64_t numel = type.shape.numel();
  node->set_attr("value", AttrValue{std::vector<double>(static_cast<size_t>(numel), 1.0)});
  node->set_attr("shape", AttrValue{type.shape});
  node->set_attr("dtype", AttrValue{type.dtype});
  return node;
}

TEST(CanonicalizeTest, ConstantOnLeftIsSwappedToRightForCommutativeOp) {
  EXPECT_TRUE(run_pass_matches_golden("canonicalize", kInputPath, kExpectedPath));
}

TEST(CanonicalizeTest, RunningTwiceIsIdempotent) {
  Result<Graph> loaded = load_ir_text_file(kInputPath);
  ASSERT_TRUE(loaded.is_ok()) << loaded.status().message();
  Graph graph = std::move(loaded.value());

  const Result<std::unique_ptr<Pass>> pass = make_canonicalize_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();

  ASSERT_TRUE(pass.value()->run(graph).is_ok());
  const std::string first_run_text = dump_text(graph);

  ASSERT_TRUE(pass.value()->run(graph).is_ok());
  const std::string second_run_text = dump_text(graph);

  EXPECT_EQ(first_run_text, second_run_text);
}

TEST(CanonicalizeTest, NonCommutativeOpWithConstantOnLeftIsNotSwapped) {
  Graph graph;
  Node* constant_lhs = MakeConstantNode(graph, {2, 2});
  Value* rhs = graph.add_graph_input(MakeFloat32Type({2, 2})).value();
  // matmul 未标注任何 trait(非 kCommutative),AB != BA,canonicalize 不得动它。
  Node* matmul_node =
      graph.create_node("matmul", {constant_lhs->output(0), rhs}, {MakeFloat32Type({2, 2})})
          .value();

  const Result<std::unique_ptr<Pass>> pass = make_canonicalize_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();
  ASSERT_TRUE(pass.value()->run(graph).is_ok());

  ASSERT_EQ(matmul_node->inputs().size(), 2u);
  EXPECT_EQ(matmul_node->inputs()[0], constant_lhs->output(0));
  EXPECT_EQ(matmul_node->inputs()[1], rhs);
}

}  // namespace
