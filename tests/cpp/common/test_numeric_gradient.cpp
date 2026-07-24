// tests/cpp/common/numeric_gradient.h 自身单测(M17 测试审计补漏):
// docs/architecture/autograd.md ARCH-066 与 docs/standards/build-and-test.md
// BUILD-011「解析梯度 ≡ 数值微分校验」专款要求数值微分侧(中心差分)本身经过
// 验证,独立于 compiler::build_backward_graph 实现是否正确——本文件只测
// numeric_gradient() 这一个函数,不涉及任何 GradientFn/build_backward_graph。
// 与 tests/cpp/compiler/test_autograd.cpp 职责分离:那里验证"解析梯度 ≡
// numeric_gradient() 的输出"(以 numeric_gradient 本身正确为前提);本文件验证
// "numeric_gradient() 的输出 ≡ 已知闭式解"(以 mse_loss(pred,target)=
// mean((pred-target)^2) 为具体标量函数,闭式偏导 d/d(pred_i)=2*(pred_i-
// target_i)/N 已知,不依赖任何 GradientFn 实现)。
//   1. 多元素输入逐分量正确性:扰动经已编译执行的 loss_fn(runtime::compile
//      ("cpu")+run_with_allocated_outputs,非 eager 旁路,与
//      CheckGradientMatchesNumeric 同一执行路径),pred 四分量取值互异,任一
//      分量的差分实现出错即会在该分量处显现;
//   2. 非 fp32 输入拒绝(numeric_gradient.h 头注释契约,ARCH-066);
//   3. loss_fn 出错时立即透传、不继续遍历其余分量,且已扰动分量在返回前复原。
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string_view>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/executable.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/graph_builder.h>
#include <frame/ops/op_registry.h>
#include <frame/runtime/compile.h>

#include "../ops/elementwise_op_test_helpers.h"
#include "numeric_gradient.h"
#include "tolerance.h"

namespace {

using frame::DType;
using frame::DTypeCode;
using frame::ErrorCode;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::Tensor;
using frame::hal::Allocator;
using frame::hal::Backend;
using frame::hal::BackendRegistry;
using frame::hal::CompileOptions;
using frame::hal::Executable;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::OpQuery;
using frame::ir::Value;
using frame::ops::create_node_with_inferred_types;
using frame::ops::make_op_query;
using frame::ops::testing::MakeType;
using frame::testing::default_tolerance;
using frame::testing::numeric_gradient;
using frame::testing::relaxed_tolerance;
using frame::testing::tensor_all_close;

// h(中心差分步长):沿用 tests/cpp/compiler/test_autograd.cpp 实测定案值
// (kCentralDifferenceH),不重新定案——本文件只验证 numeric_gradient 自身对
// 已知闭式解的正确性,不是该值的第二个独立定案来源。
constexpr double kStepH = 1e-2;

class NumericGradientTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const Result<Backend*> result = BackendRegistry::instance().get(frame::kCpuBackendName);
    ASSERT_TRUE(result.is_ok());
    backend_ = result.value();
    device_ = frame::cpu_device();
    allocator_ = backend_->allocator(device_);
    ASSERT_NE(allocator_, nullptr);
  }

  Tensor MakeTensorFromFloats(const std::vector<float>& values, const Shape& shape) {
    Tensor tensor = Tensor::empty(shape, DType::of<float>(), device_, *allocator_).value();
    float* data = tensor.data<float>();
    for (size_t i = 0; i < values.size(); ++i) {
      data[i] = values[i];
    }
    return tensor;
  }

  Backend* backend_ = nullptr;
  frame::Device device_{};
  Allocator* allocator_ = nullptr;
};

// pred/target 两个 graph_input -> mse_loss -> mark_output(唯一输出)。
Graph BuildMseLossForwardGraph() {
  Graph graph("numeric_gradient_mse_loss_forward");
  Value* pred = graph.add_graph_input(MakeType(DType::of<float>(), {4})).value();
  Value* target = graph.add_graph_input(MakeType(DType::of<float>(), {4})).value();
  Node* loss_node = create_node_with_inferred_types(graph, "mse_loss", {pred, target}).value();
  const Status mark_status = graph.mark_output(loss_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

Result<std::shared_ptr<Executable>> CompileMseLossGraph() {
  const Graph forward = BuildMseLossForwardGraph();
  const OpQuery query = make_op_query();
  const Status verify_status = forward.verify(query);
  if (!verify_status.is_ok()) return verify_status;
  return frame::runtime::compile(forward, frame::kCpuBackendName, CompileOptions{});
}

// ---------------------------------------------------------------------------
// 1. 多元素输入逐分量正确性(经已编译执行)。
// ---------------------------------------------------------------------------

TEST_F(NumericGradientTest, MatchesClosedFormForMseLossMultiElementInputViaCompiledExecution) {
  const Result<std::shared_ptr<Executable>> executable = CompileMseLossGraph();
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  Tensor target = MakeTensorFromFloats({1.0F, 1.0F, 1.0F, 1.0F}, Shape({4}));
  auto loss_fn = [&](const Tensor& pred) -> Result<double> {
    std::vector<Tensor> inputs{pred, target};
    const Result<std::vector<Tensor>> run_outputs = frame::runtime::run_with_allocated_outputs(
        *executable.value(), frame::kCpuBackendName, inputs);
    if (!run_outputs.is_ok()) return run_outputs.status();
    return static_cast<double>(*static_cast<const float*>(run_outputs.value()[0].raw_data()));
  };

  Tensor pred = MakeTensorFromFloats({1.0F, 2.0F, 3.0F, 4.0F}, Shape({4}));
  const Result<std::vector<double>> gradient = numeric_gradient(loss_fn, pred, kStepH);
  ASSERT_TRUE(gradient.is_ok()) << gradient.status().message();
  ASSERT_EQ(gradient.value().size(), 4u);

  // 闭式解(mse_loss 定义,docs/architecture/autograd.md 第4章):diff=pred-
  // target=[0,1,2,3],N=4,gpred=2*diff/N=[0,0.5,1.0,1.5]。mse_loss 对 pred 是
  // 纯二次函数,中心差分的 O(h^2) 截断项恒为 0(三阶导数为 0),故用
  // relaxed_tolerance 已留出充分余量(BUILD-011 数值微分专款,放宽一档)。
  Tensor actual = Tensor::empty(Shape({4}), DType::of<float>(), device_, *allocator_).value();
  float* actual_data = actual.data<float>();
  for (size_t i = 0; i < gradient.value().size(); ++i) {
    actual_data[i] = static_cast<float>(gradient.value()[i]);
  }
  Tensor expected = MakeTensorFromFloats({0.0F, 0.5F, 1.0F, 1.5F}, Shape({4}));
  EXPECT_TRUE(tensor_all_close(actual, expected, relaxed_tolerance(DTypeCode::kFloat32)));

  // 全部分量扰动完毕后,pred 应逐位复原为调用前的原始值。
  Tensor expected_original_pred = MakeTensorFromFloats({1.0F, 2.0F, 3.0F, 4.0F}, Shape({4}));
  EXPECT_TRUE(
      tensor_all_close(pred, expected_original_pred, default_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 2. 非 fp32 输入拒绝(numeric_gradient.h 头注释契约,ARCH-066)。
// ---------------------------------------------------------------------------

TEST_F(NumericGradientTest, RejectsNonFloat32Input) {
  Tensor x = Tensor::empty(Shape({2}), DType::of<frame::float16_t>(), device_, *allocator_).value();
  auto loss_fn = [](const Tensor&) -> Result<double> { return 0.0; };

  const Result<std::vector<double>> gradient = numeric_gradient(loss_fn, x, kStepH);
  ASSERT_FALSE(gradient.is_ok());
  EXPECT_EQ(gradient.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(gradient.status().message().find("float32"), std::string_view::npos);
  EXPECT_NE(gradient.status().message().find("ARCH-066"), std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 3. loss_fn 出错时立即透传、不继续遍历其余分量,已扰动分量在返回前复原。
// ---------------------------------------------------------------------------

TEST_F(NumericGradientTest, PropagatesLossFunctionErrorImmediatelyAndRestoresXOnFailure) {
  Tensor x = MakeTensorFromFloats({3.0F, 5.0F}, Shape({2}));
  int32_t call_count = 0;
  auto failing_loss_fn = [&](const Tensor&) -> Result<double> {
    ++call_count;
    return Status::make(ErrorCode::kInternal, "numeric_gradient_test: injected failure");
  };

  const Result<std::vector<double>> gradient = numeric_gradient(failing_loss_fn, x, kStepH);
  ASSERT_FALSE(gradient.is_ok());
  EXPECT_EQ(gradient.status().code(), ErrorCode::kInternal);
  EXPECT_NE(gradient.status().message().find("injected failure"), std::string_view::npos);
  // 第一个分量的 +h 扰动调用即失败,立即返回,不再对第二个分量继续求导。
  EXPECT_EQ(call_count, 1);

  Tensor expected_original = MakeTensorFromFloats({3.0F, 5.0F}, Shape({2}));
  EXPECT_TRUE(tensor_all_close(x, expected_original, default_tolerance(DTypeCode::kFloat32)));
}

}  // namespace
