// operator_fusion pass 单测(src/compiler/passes/operator_fusion.cpp,
// ARCH-051/ARCH-052,M9 决议点 C 覆盖版):OpTrait::kElementwise+kFusable 的
// 线性链贪心融合为单个 fused_elementwise_internal 节点。
//   1. golden:两节点链(add->relu,
//      testdata/operator_fusion_two_node_chain_{input,expected}.txt)。
//   2. golden:三节点长链(add->mul->relu,
//      testdata/operator_fusion_three_node_chain_{input,expected}.txt)。
//   3. 不融合场景(图不变,expected 与 input 逐字节相同):中间输出双消费者、
//      中间输出是图输出、链长 1(单节点无可扩展邻居)。
//   4. golden:constant 节点自身不参与融合(有 attrs,is_fusable_candidate
//      排除),但其下游的逐元素链仍照常融合,constant 保持原样作为融合节点的
//      外部输入。
//   5. golden:链尾输出多消费者——replace_all_uses 把全部旧消费者的引用一并
//      重定向到融合节点输出(不遗漏任意一个 use)。
//   6. 幂等(§3.7 v0 口径:融合产物不标 kFusable,二跑无候选链):两节点链
//      连跑两次,dump_text 逐字节相同。
//   7. ARCH-052 数值等价:add->relu 与 add->mul->relu 两个图,融合前后经
//      runtime::compile("cpu") 编译执行,数值一致(BUILD-011 容差)——与
//      tests/cpp/compiler/test_constant_folding.cpp「5. 语义等价」同一模式
//      (REUSE-002):runtime::compile 内部恒跑完整标准管线(含
//      operator_fusion 自身),故"before"(未手工预跑融合)与"after"(手工
//      预跑一次融合,融合产物不再是候选链,标准管线内的 operator_fusion 二次
//      运行是空操作)最终都经同一套标准管线编译,该测试验证的是"手工预跑本
//      pass 不改变最终编译数值结果"(与 constant_folding 既有先例同一取舍,
//      两者共享同一限制:runtime::compile 无法产出"跳过某个标准 pass"的
//      编译产物,故无法在本测试层面做到真正意义上的"完全不融合 vs 融合"两条
//      独立执行路径对比;真正验证融合"改变了什么"的是上方 1/2/4/5 的 golden
//      结构断言,本节验证"改变前后数值不受影响")。
// PassRegistry/OpRegistry/KernelRegistry 均为进程级 Meyer's singleton,本文件
// 不新增任何算子/kernel 注册,仅消费已由 src/ 静态注册好的
// add/mul/relu/square/constant/fused_elementwise_internal。
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
#include <frame/ir/graph.h>
#include <frame/ir/serialization.h>
#include <frame/runtime/compile.h>

#include "../common/tolerance.h"
#include "golden_test_helpers.h"

namespace {

using frame::DType;
using frame::DTypeCode;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::Tensor;
using frame::compiler::Pass;
using frame::compiler::PassRegistry;
using frame::compiler::testing::load_ir_text_file;
using frame::compiler::testing::run_pass_matches_golden;
using frame::hal::CompileOptions;
using frame::hal::Executable;
using frame::ir::dump_text;
using frame::ir::Graph;
using frame::testing::default_tolerance;
using frame::testing::tensor_all_close;

constexpr std::string_view kTwoNodeChainInputPath =
    "tests/cpp/compiler/testdata/operator_fusion_two_node_chain_input.txt";
constexpr std::string_view kTwoNodeChainExpectedPath =
    "tests/cpp/compiler/testdata/operator_fusion_two_node_chain_expected.txt";
constexpr std::string_view kThreeNodeChainInputPath =
    "tests/cpp/compiler/testdata/operator_fusion_three_node_chain_input.txt";
constexpr std::string_view kThreeNodeChainExpectedPath =
    "tests/cpp/compiler/testdata/operator_fusion_three_node_chain_expected.txt";
constexpr std::string_view kDualConsumerInputPath =
    "tests/cpp/compiler/testdata/operator_fusion_dual_consumer_not_fused_input.txt";
constexpr std::string_view kDualConsumerExpectedPath =
    "tests/cpp/compiler/testdata/operator_fusion_dual_consumer_not_fused_expected.txt";
constexpr std::string_view kGraphOutputBoundaryInputPath =
    "tests/cpp/compiler/testdata/operator_fusion_graph_output_boundary_not_fused_input.txt";
constexpr std::string_view kGraphOutputBoundaryExpectedPath =
    "tests/cpp/compiler/testdata/operator_fusion_graph_output_boundary_not_fused_expected.txt";
constexpr std::string_view kChainLengthOneInputPath =
    "tests/cpp/compiler/testdata/operator_fusion_chain_length_one_not_fused_input.txt";
constexpr std::string_view kChainLengthOneExpectedPath =
    "tests/cpp/compiler/testdata/operator_fusion_chain_length_one_not_fused_expected.txt";
constexpr std::string_view kConstantNotParticipatingInputPath =
    "tests/cpp/compiler/testdata/operator_fusion_constant_not_participating_input.txt";
constexpr std::string_view kConstantNotParticipatingExpectedPath =
    "tests/cpp/compiler/testdata/operator_fusion_constant_not_participating_expected.txt";
constexpr std::string_view kTailMultiConsumerInputPath =
    "tests/cpp/compiler/testdata/operator_fusion_tail_multi_consumer_input.txt";
constexpr std::string_view kTailMultiConsumerExpectedPath =
    "tests/cpp/compiler/testdata/operator_fusion_tail_multi_consumer_expected.txt";

Result<std::unique_ptr<Pass>> make_operator_fusion_pass() {
  return PassRegistry::instance().create("operator_fusion");
}

// ---------------------------------------------------------------------------
// 1/2. golden:两节点链 / 三节点长链。
// ---------------------------------------------------------------------------

TEST(OperatorFusionTest, TwoNodeElementwiseChainFusesIntoSingleNode) {
  EXPECT_TRUE(run_pass_matches_golden("operator_fusion", kTwoNodeChainInputPath,
                                      kTwoNodeChainExpectedPath));
}

TEST(OperatorFusionTest, ThreeNodeElementwiseChainFusesIntoSingleNode) {
  EXPECT_TRUE(run_pass_matches_golden("operator_fusion", kThreeNodeChainInputPath,
                                      kThreeNodeChainExpectedPath));
}

// ---------------------------------------------------------------------------
// 3. 不融合场景(图不变)。
// ---------------------------------------------------------------------------

TEST(OperatorFusionTest, IntermediateOutputWithTwoConsumersIsNotFused) {
  EXPECT_TRUE(run_pass_matches_golden("operator_fusion", kDualConsumerInputPath,
                                      kDualConsumerExpectedPath));
}

TEST(OperatorFusionTest, IntermediateOutputThatIsAGraphOutputBreaksTheChain) {
  EXPECT_TRUE(run_pass_matches_golden("operator_fusion", kGraphOutputBoundaryInputPath,
                                      kGraphOutputBoundaryExpectedPath));
}

TEST(OperatorFusionTest, SingleNodeWithNoExtendableNeighborIsNotFused) {
  EXPECT_TRUE(run_pass_matches_golden("operator_fusion", kChainLengthOneInputPath,
                                      kChainLengthOneExpectedPath));
}

// ---------------------------------------------------------------------------
// 4. constant 节点自身不参与融合,但下游链仍融合。
// ---------------------------------------------------------------------------

TEST(OperatorFusionTest, ConstantNodeDoesNotParticipateButDownstreamChainStillFuses) {
  EXPECT_TRUE(run_pass_matches_golden("operator_fusion", kConstantNotParticipatingInputPath,
                                      kConstantNotParticipatingExpectedPath));
}

// ---------------------------------------------------------------------------
// 5. 链尾输出多消费者:全部 use 重定向。
// ---------------------------------------------------------------------------

TEST(OperatorFusionTest, ChainTailWithMultipleConsumersRedirectsEveryUse) {
  EXPECT_TRUE(run_pass_matches_golden("operator_fusion", kTailMultiConsumerInputPath,
                                      kTailMultiConsumerExpectedPath));
}

// ---------------------------------------------------------------------------
// 6. 幂等。
// ---------------------------------------------------------------------------

TEST(OperatorFusionTest, RunningTwiceOnTwoNodeChainIsIdempotent) {
  Result<Graph> loaded = load_ir_text_file(kTwoNodeChainInputPath);
  ASSERT_TRUE(loaded.is_ok()) << loaded.status().message();
  Graph graph = std::move(loaded.value());

  const Result<std::unique_ptr<Pass>> pass = make_operator_fusion_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();

  ASSERT_TRUE(pass.value()->run(graph).is_ok());
  const std::string first_run_text = dump_text(graph);

  ASSERT_TRUE(pass.value()->run(graph).is_ok());
  const std::string second_run_text = dump_text(graph);

  EXPECT_EQ(first_run_text, second_run_text);
}

// ---------------------------------------------------------------------------
// 7. ARCH-052 数值等价。
// ---------------------------------------------------------------------------

// path 对应的图必须全部输入为 graph_input(非 constant),各输入 shape 均为
// input_shape,以便本函数用可区分的填充值驱动真实数值比较。"before" 与
// "after" 两份独立加载(Graph move-only 不可拷贝,同
// test_constant_folding.cpp::BuildFoldableAddGraph 先例)。
void ExpectFusionPreservesNumericResult(std::string_view path, int64_t num_inputs,
                                        const Shape& input_shape) {
  Result<Graph> before_loaded = load_ir_text_file(path);
  ASSERT_TRUE(before_loaded.is_ok()) << before_loaded.status().message();
  const Graph before = std::move(before_loaded.value());
  const Result<std::shared_ptr<Executable>> before_executable =
      frame::runtime::compile(before, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(before_executable.is_ok()) << before_executable.status().message();

  Result<Graph> after_loaded = load_ir_text_file(path);
  ASSERT_TRUE(after_loaded.is_ok()) << after_loaded.status().message();
  Graph after = std::move(after_loaded.value());
  const Result<std::unique_ptr<Pass>> pass = make_operator_fusion_pass();
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

  // 非平凡、逐输入可区分的填充值(而非全零/全一),同
  // test_runtime_compile.cpp::MakeFilledTensor 惯例(独立实现,不复用其头
  // 私有辅助函数)。
  std::vector<Tensor> inputs;
  inputs.reserve(static_cast<size_t>(num_inputs));
  for (int64_t i = 0; i < num_inputs; ++i) {
    Tensor tensor =
        Tensor::empty(input_shape, DType::of<float>(), frame::cpu_device(), *allocator).value();
    float* data = tensor.data<float>();
    for (int64_t j = 0; j < tensor.numel(); ++j) {
      data[j] = static_cast<float>(i + 1) + static_cast<float>(j) * 0.25F;
    }
    inputs.push_back(tensor);
  }

  std::vector<Tensor> before_outputs{
      Tensor::empty(input_shape, DType::of<float>(), frame::cpu_device(), *allocator).value()};
  std::vector<Tensor> after_outputs{
      Tensor::empty(input_shape, DType::of<float>(), frame::cpu_device(), *allocator).value()};

  ASSERT_TRUE(
      before_executable.value()->run(inputs, before_outputs, *stream_result.value()).is_ok());
  ASSERT_TRUE(after_executable.value()->run(inputs, after_outputs, *stream_result.value()).is_ok());

  EXPECT_TRUE(tensor_all_close(after_outputs[0], before_outputs[0],
                               default_tolerance(DTypeCode::kFloat32)));
}

TEST(OperatorFusionTest, TwoNodeChainFusionPreservesNumericResult) {
  ExpectFusionPreservesNumericResult(kTwoNodeChainInputPath, /*num_inputs=*/2, Shape({4}));
}

TEST(OperatorFusionTest, ThreeNodeChainFusionPreservesNumericResult) {
  ExpectFusionPreservesNumericResult(kThreeNodeChainInputPath, /*num_inputs=*/3, Shape({4}));
}

}  // namespace
