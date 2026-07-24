// compiler::build_backward_graph 端到端测试(M17,docs/architecture/
// autograd.md 第2/3/4/6章;ARCH-060~ARCH-066)。
//   1. 逐算子(七个:add/mul/square/relu/matmul/sum/mse_loss)解析梯度 ≡
//      数值微分校验(中心差分,tests/cpp/common/numeric_gradient.h;容差经
//      relaxed_tolerance,BUILD-011「解析梯度 ≡ 数值微分校验」专款):非标量
//      输出的算子经 sum(...,axes=[])包一层归约为标量 loss——gy=全1 是
//      "loss=sum(y)"这一具体标量函数的精确上游梯度,数值/解析两侧共用同一个
//      标量函数定义,是充分且标准的梯度校验技术(不因 gy=全1 而降低对
//      Jacobian 各分量正确性的检验力度:数值微分对 x 的每个分量独立求导,
//      任何一个分量的解析公式出错,在一般(非退化)输入下几乎必然导致该分量
//      的数值/解析差值超出容差)。add/mul/matmul 各自对两个操作数分别校验;
//      relu 的输入取值远离 0(避开不可导的 kink 点,含扰动邻域)。
//   2. 组合图(matmul+add+relu+mse_loss 全链)对 wrt 参数(w/bias)的梯度校验;
//      训练图经 verify(V1-V7 无特殊化)与 runtime::compile("cpu") 九段管线
//      编译执行(ARCH-060 判定方法)。
//   3. 错误路径:非标量 loss;loss_output_index/wrt_input_indices 越界;
//      wrt_input_indices 重复;链上含未注册 GradientFn 的算子(消息含算子名,
//      ARCH-062);链上含 kHasSideEffect 算子。
//   4. wrt 输入不在 loss 依赖链上时梯度补零(标准自动微分惯例,非错误——
//      autograd.md 对此边角未显式约定,本实现按此惯例补齐,已在实现报告中
//      提示复核)。
//
// h(中心差分步长)实测定案:见下方 kCentralDifferenceH 常量定义处注释。
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <frame/compiler/autograd.h>
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
#include <frame/ops/fused_elementwise_utils.h>
#include <frame/ops/graph_builder.h>
#include <frame/ops/op_registry.h>
#include <frame/runtime/compile.h>

#include "../common/numeric_gradient.h"
#include "../common/tolerance.h"
#include "../ops/elementwise_op_test_helpers.h"
#include "pass_test_common.h"

namespace {

using frame::bfloat16_t;
using frame::DType;
using frame::DTypeCode;
using frame::ErrorCode;
using frame::float16_t;
using frame::float_to_bfloat16;
using frame::float_to_float16;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::Tensor;
using frame::compiler::build_backward_graph;
using frame::hal::Backend;
using frame::hal::BackendRegistry;
using frame::hal::CompileOptions;
using frame::hal::Executable;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::OpQuery;
using frame::ir::Value;
using frame::ops::AttrMap;
using frame::ops::create_node_with_inferred_types;
using frame::ops::make_op_query;
using frame::ops::testing::MakeType;
using frame::testing::default_tolerance;
using frame::testing::numeric_gradient;
using frame::testing::relaxed_tolerance;
using frame::testing::tensor_all_close;

// 中心差分步长实测定案(fp32,BUILD-011「解析梯度 ≡ 数值微分校验」专款):
// BUILD-011 建议 1e-3 量级起调;实测 1e-3~1e-2 区间对本文件全部用例(add/
// mul/square/relu/matmul/sum/mse_loss/组合图)均稳定通过 relaxed_tolerance
// (fp32 放宽一档 = fp16 档 {rtol=1e-2, atol=1e-3});取 1e-2(该区间内偏大
// 一侧)是为在 O(h²) 截断误差与 float32 舍入误差(h 过小时 (f(x+h)-f(x-h))
// 两个相近浮点数相减损失有效数字)之间留出更充分的安全边际,兼顾组合图
// (matmul+add+relu+mse_loss,数值幅度经四层算子传递)与单算子用例。
constexpr double kCentralDifferenceH = 1e-2;

// 通用 fixture:取 cpu 后端真实 Allocator(经 BackendRegistry)+ 按填充公式/
// 数值向量构造 Tensor 的两个 helper。与 tests/cpp/ops/
// elementwise_op_test_helpers.h::ElementwiseEagerTestBase 同思路,但本文件
// 额外需要 MakeTensorFromDoubles(把 numeric_gradient 产出的 double 向量转为
// Tensor 以复用 tensor_all_close,BUILD-011 容差工具唯一入口),故不直接继承。
class AutogradTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const Result<Backend*> result = BackendRegistry::instance().get(frame::kCpuBackendName);
    ASSERT_TRUE(result.is_ok());
    backend_ = result.value();
    device_ = frame::cpu_device();
    allocator_ = backend_->allocator(device_);
    ASSERT_NE(allocator_, nullptr);
  }

  Tensor MakeFilledTensor(const Shape& shape, float start, float step) {
    Tensor tensor = Tensor::empty(shape, DType::of<float>(), device_, *allocator_).value();
    float* data = tensor.data<float>();
    for (int64_t i = 0; i < tensor.numel(); ++i) {
      data[i] = start + static_cast<float>(i) * step;
    }
    return tensor;
  }

  Tensor MakeTensorFromDoubles(const std::vector<double>& values, const Shape& shape) {
    Tensor tensor = Tensor::empty(shape, DType::of<float>(), device_, *allocator_).value();
    float* data = tensor.data<float>();
    for (size_t i = 0; i < values.size(); ++i) {
      data[i] = static_cast<float>(values[i]);
    }
    return tensor;
  }

  // fp16/bf16 经 fp32 解析参照验证(ARCH-066)专用:按位构造 fp16 Tensor、把
  // 既有 fp32 Tensor 逐元素转换为 fp16 Tensor(供比较参照)。
  Tensor MakeFp16TensorFromFloats(const std::vector<float>& values, const Shape& shape) {
    Tensor tensor = Tensor::empty(shape, DType::of<float16_t>(), device_, *allocator_).value();
    float16_t* data = tensor.data<float16_t>();
    for (size_t i = 0; i < values.size(); ++i) {
      data[i] = float_to_float16(values[i]);
    }
    return tensor;
  }

  Tensor ConvertFloatTensorToFp16(const Tensor& source) {
    Tensor result =
        Tensor::empty(source.shape(), DType::of<float16_t>(), device_, *allocator_).value();
    const float* src_data = static_cast<const float*>(source.raw_data());
    float16_t* dst_data = result.data<float16_t>();
    for (int64_t i = 0; i < source.numel(); ++i) {
      dst_data[i] = float_to_float16(src_data[i]);
    }
    return result;
  }

  // bf16 镜像版本(ARCH-066 同款,ReluGradientBf16MatchesFp32AnalyticReference
  // 专用):按位构造 bf16 Tensor、把既有 fp32 Tensor 逐元素转换为 bf16 Tensor。
  Tensor MakeBf16TensorFromFloats(const std::vector<float>& values, const Shape& shape) {
    Tensor tensor = Tensor::empty(shape, DType::of<bfloat16_t>(), device_, *allocator_).value();
    bfloat16_t* data = tensor.data<bfloat16_t>();
    for (size_t i = 0; i < values.size(); ++i) {
      data[i] = float_to_bfloat16(values[i]);
    }
    return tensor;
  }

  Tensor ConvertFloatTensorToBf16(const Tensor& source) {
    Tensor result =
        Tensor::empty(source.shape(), DType::of<bfloat16_t>(), device_, *allocator_).value();
    const float* src_data = static_cast<const float*>(source.raw_data());
    bfloat16_t* dst_data = result.data<bfloat16_t>();
    for (int64_t i = 0; i < source.numel(); ++i) {
      dst_data[i] = float_to_bfloat16(src_data[i]);
    }
    return result;
  }

  Backend* backend_ = nullptr;
  frame::Device device_{};
  frame::hal::Allocator* allocator_ = nullptr;
};

// 只取解析梯度 Tensor,不做数值微分比较(供跨 dtype 参照测试复用,ARCH-066:
// fp16/bf16 梯度经 fp32 解析参照验证,不直接与数值微分比较——numeric_gradient
// 本身也拒绝非 fp32 输入)。管线与 CheckGradientMatchesNumeric 下方的解析梯度
// 提取部分一致(build_backward_graph → verify → runtime::compile("cpu") →
// run),独立成小函数供 fp32/fp16 两侧各调一次。
Result<Tensor> ComputeAnalyticGradient(const Graph& forward, int32_t loss_output_index,
                                       const std::vector<int32_t>& wrt_indices, size_t wrt_position,
                                       std::vector<Tensor>& inputs) {
  const Result<Graph> training = build_backward_graph(forward, loss_output_index, wrt_indices);
  if (!training.is_ok()) return training.status();

  const OpQuery query = make_op_query();
  const Status verify_status = training.value().verify(query);
  if (!verify_status.is_ok()) return verify_status;

  const Result<std::shared_ptr<Executable>> executable =
      frame::runtime::compile(training.value(), frame::kCpuBackendName, CompileOptions{});
  if (!executable.is_ok()) return executable.status();

  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  if (!outputs.is_ok()) return outputs.status();

  return outputs.value()[1 + wrt_position];
}

// 解析梯度 ≡ 数值微分核心断言(ARCH-066):对 forward 调用 build_backward_graph
// → verify → runtime::compile("cpu") → 编译产物读回 wrt_indices[wrt_position]
// 位置的解析梯度,与 numeric_gradient 对同一编译产物的 loss 输出(outputs[0])
// 做中心差分所得数值梯度比较。inputs 与 forward.inputs() 按位对应,调用方
// 保证其在整个断言期间保持有效(numeric_gradient 原地扰动后会复原,断言结束
// 时 inputs 与调用前一致)。
::testing::AssertionResult CheckGradientMatchesNumeric(const Graph& forward,
                                                       int32_t loss_output_index,
                                                       const std::vector<int32_t>& wrt_indices,
                                                       size_t wrt_position,
                                                       std::vector<Tensor>& inputs, double h) {
  const Result<Graph> training = build_backward_graph(forward, loss_output_index, wrt_indices);
  if (!training.is_ok()) {
    return ::testing::AssertionFailure()
           << "build_backward_graph failed: " << training.status().message();
  }

  const OpQuery query = make_op_query();
  const Status verify_status = training.value().verify(query);
  if (!verify_status.is_ok()) {
    return ::testing::AssertionFailure()
           << "training graph verify() failed: " << verify_status.message();
  }

  const Result<std::shared_ptr<Executable>> executable =
      frame::runtime::compile(training.value(), frame::kCpuBackendName, CompileOptions{});
  if (!executable.is_ok()) {
    return ::testing::AssertionFailure()
           << "runtime::compile failed: " << executable.status().message();
  }

  auto loss_fn = [&](const Tensor&) -> Result<double> {
    const Result<std::vector<Tensor>> run_outputs = frame::runtime::run_with_allocated_outputs(
        *executable.value(), frame::kCpuBackendName, inputs);
    if (!run_outputs.is_ok()) return run_outputs.status();
    return static_cast<double>(*static_cast<const float*>(run_outputs.value()[0].raw_data()));
  };

  const auto wrt_input_index = static_cast<size_t>(wrt_indices[wrt_position]);
  const Result<std::vector<double>> numeric = numeric_gradient(loss_fn, inputs[wrt_input_index], h);
  if (!numeric.is_ok()) {
    return ::testing::AssertionFailure()
           << "numeric_gradient failed: " << numeric.status().message();
  }

  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  if (!outputs.is_ok()) {
    return ::testing::AssertionFailure() << outputs.status().message();
  }
  const Tensor& analytic = outputs.value()[1 + wrt_position];

  const Result<Backend*> backend_result = BackendRegistry::instance().get(frame::kCpuBackendName);
  frame::hal::Allocator* allocator = backend_result.value()->allocator(frame::cpu_device());
  Tensor numeric_tensor =
      Tensor::empty(analytic.shape(), DType::of<float>(), frame::cpu_device(), *allocator).value();
  float* numeric_data = numeric_tensor.data<float>();
  for (size_t i = 0; i < numeric.value().size(); ++i) {
    numeric_data[i] = static_cast<float>(numeric.value()[i]);
  }

  return tensor_all_close(analytic, numeric_tensor, relaxed_tolerance(DTypeCode::kFloat32));
}

Result<std::vector<Tensor>> CompileAndRunCpu(const Graph& graph, std::vector<Tensor>& inputs) {
  const Status verify_status = graph.verify(make_op_query());
  if (!verify_status.is_ok()) return verify_status;
  const Result<std::shared_ptr<Executable>> executable =
      frame::runtime::compile(graph, frame::kCpuBackendName, CompileOptions{});
  if (!executable.is_ok()) return executable.status();
  return frame::runtime::run_with_allocated_outputs(*executable.value(), frame::kCpuBackendName,
                                                    inputs);
}

// ---------------------------------------------------------------------------
// 1. 逐算子解析梯度 ≡ 数值微分(七个)。非标量输出的算子经 sum(...,axes=[])
//    包一层归约为标量 loss(见文件头注释)。
// ---------------------------------------------------------------------------

Graph BuildAddLossGraph() {
  Graph graph("add_loss");
  Value* a = graph.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  Value* b = graph.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  Node* add_node = create_node_with_inferred_types(graph, "add", {a, b}).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {add_node->output(0)}, sum_attrs).value();
  const Status mark_status = graph.mark_output(sum_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

TEST_F(AutogradTest, AddGradientMatchesNumericForBothOperands) {
  const Graph forward = BuildAddLossGraph();
  Tensor a = MakeFilledTensor(Shape({2, 3}), 1.0F, 0.3F);
  Tensor b = MakeFilledTensor(Shape({2, 3}), -0.5F, 0.2F);
  std::vector<Tensor> inputs{a, b};
  const std::vector<int32_t> wrt{0, 1};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kCentralDifferenceH));
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 1, inputs, kCentralDifferenceH));
}

Graph BuildMulLossGraph() {
  Graph graph("mul_loss");
  Value* a = graph.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  Value* b = graph.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  Node* mul_node = create_node_with_inferred_types(graph, "mul", {a, b}).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {mul_node->output(0)}, sum_attrs).value();
  const Status mark_status = graph.mark_output(sum_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

TEST_F(AutogradTest, MulGradientMatchesNumericForBothOperands) {
  const Graph forward = BuildMulLossGraph();
  Tensor a = MakeFilledTensor(Shape({2, 3}), 1.2F, 0.25F);
  Tensor b = MakeFilledTensor(Shape({2, 3}), 0.7F, -0.15F);
  std::vector<Tensor> inputs{a, b};
  const std::vector<int32_t> wrt{0, 1};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kCentralDifferenceH));
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 1, inputs, kCentralDifferenceH));
}

Graph BuildSquareLossGraph() {
  Graph graph("square_loss");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  Node* square_node = create_node_with_inferred_types(graph, "square", {x}).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {square_node->output(0)}, sum_attrs).value();
  const Status mark_status = graph.mark_output(sum_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

TEST_F(AutogradTest, SquareGradientMatchesNumeric) {
  const Graph forward = BuildSquareLossGraph();
  Tensor x = MakeFilledTensor(Shape({2, 3}), -1.0F, 0.3F);
  std::vector<Tensor> inputs{x};
  const std::vector<int32_t> wrt{0};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kCentralDifferenceH));
}

// dtype 参数化(fp16 vs fp32 解析参照测试复用,见下方
// ReluGradientFp16MatchesFp32AnalyticReference);BuildReluLossGraph() 是其
// fp32 特化,行为与重构前逐字一致。
Graph BuildReluLossGraphWithDtype(DType dtype) {
  Graph graph("relu_loss");
  Value* x = graph.add_graph_input(MakeType(dtype, {2, 3})).value();
  Node* relu_node = create_node_with_inferred_types(graph, "relu", {x}).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {relu_node->output(0)}, sum_attrs).value();
  const Status mark_status = graph.mark_output(sum_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

Graph BuildReluLossGraph() { return BuildReluLossGraphWithDtype(DType::of<float>()); }

TEST_F(AutogradTest, ReluGradientMatchesNumeric) {
  const Graph forward = BuildReluLossGraph();
  // 取值远离 0(|x|>=0.5,含 kCentralDifferenceH=1e-2 扰动邻域仍不改变
  // x>0 的符号),避开 relu 在 x=0 处不可导的 kink 点。
  Tensor x = MakeTensorFromDoubles({1.5, -1.5, 2.0, -2.0, 0.5, -0.5}, Shape({2, 3}));
  std::vector<Tensor> inputs{x};
  const std::vector<int32_t> wrt{0};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kCentralDifferenceH));
}

TEST_F(AutogradTest, ReluGradientFp16MatchesFp32AnalyticReference) {
  // ARCH-066/BUILD-011「解析梯度 ≡ 数值微分校验」专款明文:fp16/bf16 的梯度
  // 不直接与数值微分比较,一律经 fp32 解析参照验证(numeric_gradient.h 头
  // 注释同一口径,该函数本身拒绝非 fp32 输入,见
  // tests/cpp/common/test_numeric_gradient.cpp::RejectsNonFloat32Input)。本
  // 用例构造同一 relu_loss 前向图结构的 fp32/fp16 两个实例(数值相同,仅
  // dtype 不同),各自过 build_backward_graph → verify →
  // runtime::compile("cpu") → run 取得解析梯度;fp32 侧结果逐元素转换为 fp16
  // (frame::float_to_float16)作为参照,与 fp16 侧解析梯度按
  // default_tolerance(fp16) 比较——这里两侧都是解析结果(非数值微分),不适用
  // BUILD-011 数值微分专款的放宽一档,套用常规 fp16 容差表。取值远离 relu
  // kink 点(同 ReluGradientMatchesNumeric)且均为 fp16 可精确表示的二进制
  // 小数,避免额外舍入误差混入判定。
  const std::vector<float> x_values{1.5F, -2.5F, 2.0F, -2.0F, 0.5F, -0.5F};
  const std::vector<int32_t> wrt{0};

  const Graph fp32_forward = BuildReluLossGraphWithDtype(DType::of<float>());
  Tensor x_fp32 =
      MakeTensorFromDoubles(std::vector<double>(x_values.begin(), x_values.end()), Shape({2, 3}));
  std::vector<Tensor> fp32_inputs{x_fp32};
  const Result<Tensor> fp32_grad = ComputeAnalyticGradient(fp32_forward, 0, wrt, 0, fp32_inputs);
  ASSERT_TRUE(fp32_grad.is_ok()) << fp32_grad.status().message();

  const Graph fp16_forward = BuildReluLossGraphWithDtype(DType::of<float16_t>());
  Tensor x_fp16 = MakeFp16TensorFromFloats(x_values, Shape({2, 3}));
  std::vector<Tensor> fp16_inputs{x_fp16};
  const Result<Tensor> fp16_grad = ComputeAnalyticGradient(fp16_forward, 0, wrt, 0, fp16_inputs);
  ASSERT_TRUE(fp16_grad.is_ok()) << fp16_grad.status().message();

  const Tensor fp32_grad_as_fp16 = ConvertFloatTensorToFp16(fp32_grad.value());
  EXPECT_TRUE(tensor_all_close(fp16_grad.value(), fp32_grad_as_fp16,
                               default_tolerance(DTypeCode::kFloat16)));
}

TEST_F(AutogradTest, ReluGradientBf16MatchesFp32AnalyticReference) {
  // ARCH-066/BUILD-011「解析梯度 ≡ 数值微分校验」专款明文:fp16/bf16 的梯度
  // 不直接与数值微分比较,一律经 fp32 解析参照验证——本用例是
  // ReluGradientFp16MatchesFp32AnalyticReference 的 bf16 镜像版本,构图/管线/
  // 断言手法逐字一致,仅 dtype 从 float16 换成 bfloat16(容差表相应换成
  // default_tolerance(bf16))。取值同 fp16 版本(远离 relu kink 点、fp16 可
  // 精确表示的二进制小数——这些值同样是 bf16 可精确表示的二进制小数,bf16
  // 尾数位数虽少于 fp16 但对这几个仅需 1~2 位尾数的值仍精确,避免额外舍入
  // 误差混入判定)。
  const std::vector<float> x_values{1.5F, -2.5F, 2.0F, -2.0F, 0.5F, -0.5F};
  const std::vector<int32_t> wrt{0};

  const Graph fp32_forward = BuildReluLossGraphWithDtype(DType::of<float>());
  Tensor x_fp32 =
      MakeTensorFromDoubles(std::vector<double>(x_values.begin(), x_values.end()), Shape({2, 3}));
  std::vector<Tensor> fp32_inputs{x_fp32};
  const Result<Tensor> fp32_grad = ComputeAnalyticGradient(fp32_forward, 0, wrt, 0, fp32_inputs);
  ASSERT_TRUE(fp32_grad.is_ok()) << fp32_grad.status().message();

  const Graph bf16_forward = BuildReluLossGraphWithDtype(DType::of<bfloat16_t>());
  Tensor x_bf16 = MakeBf16TensorFromFloats(x_values, Shape({2, 3}));
  std::vector<Tensor> bf16_inputs{x_bf16};
  const Result<Tensor> bf16_grad = ComputeAnalyticGradient(bf16_forward, 0, wrt, 0, bf16_inputs);
  ASSERT_TRUE(bf16_grad.is_ok()) << bf16_grad.status().message();

  const Tensor fp32_grad_as_bf16 = ConvertFloatTensorToBf16(fp32_grad.value());
  EXPECT_TRUE(tensor_all_close(bf16_grad.value(), fp32_grad_as_bf16,
                               default_tolerance(DTypeCode::kBFloat16)));
}

Graph BuildMatmulLossGraph() {
  Graph graph("matmul_loss");
  Value* a = graph.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  Value* b = graph.add_graph_input(MakeType(DType::of<float>(), {3, 4})).value();
  Node* matmul_node = create_node_with_inferred_types(graph, "matmul", {a, b}).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {matmul_node->output(0)}, sum_attrs).value();
  const Status mark_status = graph.mark_output(sum_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

TEST_F(AutogradTest, MatmulGradientMatchesNumericForBothOperands) {
  const Graph forward = BuildMatmulLossGraph();
  Tensor a = MakeFilledTensor(Shape({2, 3}), 0.3F, 0.15F);
  Tensor b = MakeFilledTensor(Shape({3, 4}), -0.2F, 0.1F);
  std::vector<Tensor> inputs{a, b};
  const std::vector<int32_t> wrt{0, 1};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kCentralDifferenceH));
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 1, inputs, kCentralDifferenceH));
}

Graph BuildSumLossGraph() {
  Graph graph("sum_loss");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node = create_node_with_inferred_types(graph, "sum", {x}, sum_attrs).value();
  const Status mark_status = graph.mark_output(sum_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

TEST_F(AutogradTest, SumGradientMatchesNumeric) {
  const Graph forward = BuildSumLossGraph();
  Tensor x = MakeFilledTensor(Shape({2, 3}), 1.0F, 0.4F);
  std::vector<Tensor> inputs{x};
  const std::vector<int32_t> wrt{0};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kCentralDifferenceH));
}

Graph BuildMseLossGraph() {
  Graph graph("mse_loss_loss");
  Value* pred = graph.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  Value* target = graph.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  Node* loss_node = create_node_with_inferred_types(graph, "mse_loss", {pred, target}).value();
  const Status mark_status = graph.mark_output(loss_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

TEST_F(AutogradTest, MseLossGradientMatchesNumericForPredAndTarget) {
  const Graph forward = BuildMseLossGraph();
  Tensor pred = MakeFilledTensor(Shape({2, 3}), 1.0F, 0.3F);
  Tensor target = MakeFilledTensor(Shape({2, 3}), 0.5F, 0.1F);
  std::vector<Tensor> inputs{pred, target};
  const std::vector<int32_t> wrt{0, 1};
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kCentralDifferenceH));
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 1, inputs, kCentralDifferenceH));
}

// ---------------------------------------------------------------------------
// 2. 组合图(matmul+add+relu+mse_loss 全链)对 wrt 参数(w/bias)的梯度校验。
// ---------------------------------------------------------------------------

// graph_inputs 按序 [x, w, bias, target](下标 0..3)。
Graph BuildCombinedLossGraph() {
  Graph graph("combined_loss");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  Value* w = graph.add_graph_input(MakeType(DType::of<float>(), {3, 4})).value();
  Value* bias = graph.add_graph_input(MakeType(DType::of<float>(), {2, 4})).value();
  Value* target = graph.add_graph_input(MakeType(DType::of<float>(), {2, 4})).value();
  Node* matmul_node = create_node_with_inferred_types(graph, "matmul", {x, w}).value();
  Node* add_node =
      create_node_with_inferred_types(graph, "add", {matmul_node->output(0), bias}).value();
  Node* relu_node = create_node_with_inferred_types(graph, "relu", {add_node->output(0)}).value();
  Node* mse_node =
      create_node_with_inferred_types(graph, "mse_loss", {relu_node->output(0), target}).value();
  const Status mark_status = graph.mark_output(mse_node->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

TEST_F(AutogradTest, CombinedMatmulAddReluMseLossGradientMatchesNumericForWeightAndBias) {
  const Graph forward = BuildCombinedLossGraph();
  // x/w 均取正值、bias 取较大正常数(5.0):matmul(x,w)+bias 在 wrt 扰动邻域
  // (kCentralDifferenceH=1e-2)内稳定保持正值,relu 全程走恒等分支,不触及
  // x=0 kink(见 relu 单算子用例头注释同一考量)。
  Tensor x = MakeFilledTensor(Shape({2, 3}), 0.1F, 0.1F);
  Tensor w = MakeFilledTensor(Shape({3, 4}), 0.1F, 0.1F);
  Tensor bias = MakeFilledTensor(Shape({2, 4}), 5.0F, 0.0F);
  Tensor target = MakeFilledTensor(Shape({2, 4}), 1.0F, 0.0F);
  std::vector<Tensor> inputs{x, w, bias, target};
  const std::vector<int32_t> wrt{1, 2};  // w, bias

  // 训练图经 verify() 与 runtime::compile("cpu") 九段管线编译执行
  // (ARCH-060 判定方法),由 CheckGradientMatchesNumeric 内部完成。
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 0, inputs, kCentralDifferenceH));
  EXPECT_TRUE(CheckGradientMatchesNumeric(forward, 0, wrt, 1, inputs, kCentralDifferenceH));
}

// ---------------------------------------------------------------------------
// 3. 错误路径。
// ---------------------------------------------------------------------------

TEST(BuildBackwardGraphErrorPathTest, RejectsNonScalarLoss) {
  Graph graph("non_scalar_loss");
  Value* a = graph.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  Value* b = graph.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  Node* add_node = create_node_with_inferred_types(graph, "add", {a, b}).value();
  ASSERT_TRUE(graph.mark_output(add_node->output(0)).is_ok());  // 非标量([2,3])

  const std::vector<int32_t> wrt{0};
  const Result<Graph> result = build_backward_graph(graph, 0, wrt);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("loss output must be scalar"), std::string_view::npos);
  EXPECT_NE(result.status().message().find("[2, 3]"), std::string_view::npos);
}

TEST(BuildBackwardGraphErrorPathTest, RejectsOutOfRangeLossOutputIndex) {
  Graph graph("out_of_range_loss_index");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {4})).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node = create_node_with_inferred_types(graph, "sum", {x}, sum_attrs).value();
  ASSERT_TRUE(graph.mark_output(sum_node->output(0)).is_ok());

  const std::vector<int32_t> wrt{0};
  const Result<Graph> result = build_backward_graph(graph, 3, wrt);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("loss_output_index 3 is out of range"),
            std::string_view::npos);
}

TEST(BuildBackwardGraphErrorPathTest, RejectsOutOfRangeWrtIndex) {
  Graph graph("out_of_range_wrt_index");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {4})).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node = create_node_with_inferred_types(graph, "sum", {x}, sum_attrs).value();
  ASSERT_TRUE(graph.mark_output(sum_node->output(0)).is_ok());

  const std::vector<int32_t> wrt{5};
  const Result<Graph> result = build_backward_graph(graph, 0, wrt);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("wrt_input_indices entry 5 is out of range"),
            std::string_view::npos);
}

TEST(BuildBackwardGraphErrorPathTest, RejectsDuplicateWrtIndex) {
  Graph graph("duplicate_wrt_index");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {4})).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node = create_node_with_inferred_types(graph, "sum", {x}, sum_attrs).value();
  ASSERT_TRUE(graph.mark_output(sum_node->output(0)).is_ok());

  const std::vector<int32_t> wrt{0, 0};
  const Result<Graph> result = build_backward_graph(graph, 0, wrt);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("wrt_input_indices entry 0 is duplicated"),
            std::string_view::npos);
}

TEST(BuildBackwardGraphErrorPathTest, RejectsChainWithUnregisteredGradientOpAndNamesTheOp) {
  frame::compiler::testing::ensure_pass_test_ops_registered();
  Graph graph("no_grad_chain");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {4})).value();
  Node* no_grad_node = create_node_with_inferred_types(
                           graph, std::string(frame::compiler::testing::kNoKernelOpName), {x})
                           .value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {no_grad_node->output(0)}, sum_attrs).value();
  ASSERT_TRUE(graph.mark_output(sum_node->output(0)).is_ok());

  const std::vector<int32_t> wrt{0};
  const Result<Graph> result = build_backward_graph(graph, 0, wrt);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kUnimplemented);
  EXPECT_NE(result.status().message().find(frame::compiler::testing::kNoKernelOpName),
            std::string_view::npos);
  EXPECT_NE(result.status().message().find("has no registered GradientFn"), std::string_view::npos);
}

TEST(BuildBackwardGraphErrorPathTest, RejectsChainWithSideEffectOpAndNamesTheOp) {
  frame::compiler::testing::ensure_pass_test_ops_registered();
  Graph graph("side_effect_chain");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {4})).value();
  Node* side_effect_node = create_node_with_inferred_types(
                               graph, std::string(frame::compiler::testing::kSideEffectOpName), {x})
                               .value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {side_effect_node->output(0)}, sum_attrs)
          .value();
  ASSERT_TRUE(graph.mark_output(sum_node->output(0)).is_ok());

  const std::vector<int32_t> wrt{0};
  const Result<Graph> result = build_backward_graph(graph, 0, wrt);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find(frame::compiler::testing::kSideEffectOpName),
            std::string_view::npos);
  EXPECT_NE(result.status().message().find("kHasSideEffect"), std::string_view::npos);
}

TEST(BuildBackwardGraphErrorPathTest, SelectedFusedElementwiseChainFailsLoudly) {
  Graph graph("selected_fused_elementwise");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  const AttrMap fused_attrs{{"ops", std::string("relu;square")},
                            {"arities", std::vector<int64_t>{1, 1}}};
  Node* fused =
      create_node_with_inferred_types(graph, frame::ops::kFusedElementwiseOpName, {x}, fused_attrs)
          .value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* loss = create_node_with_inferred_types(graph, "sum", {fused->output(0)}, sum_attrs).value();
  ASSERT_TRUE(graph.mark_output(loss, 0).is_ok());

  const std::vector<int32_t> wrt{0};
  const Result<Graph> result = build_backward_graph(graph, 0, wrt);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kUnimplemented);
  EXPECT_NE(result.status().message().find("fused_elementwise_internal"), std::string_view::npos);
}

TEST(BuildBackwardGraphMultiOutputTest, UnselectedFusedBranchDoesNotBlockDifferentiation) {
  Graph graph("unselected_fused_branch");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  Node* square = create_node_with_inferred_types(graph, "square", {x}).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* loss =
      create_node_with_inferred_types(graph, "sum", {square->output(0)}, sum_attrs).value();
  const AttrMap fused_attrs{{"ops", std::string("relu;square")},
                            {"arities", std::vector<int64_t>{1, 1}}};
  Node* fused =
      create_node_with_inferred_types(graph, frame::ops::kFusedElementwiseOpName, {x}, fused_attrs)
          .value();
  ASSERT_TRUE(graph.mark_output(loss, 0).is_ok());
  ASSERT_TRUE(graph.mark_output(fused, 0).is_ok());

  const std::vector<int32_t> wrt{0};
  const Result<Graph> result = build_backward_graph(graph, 0, wrt);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  ASSERT_EQ(result.value().outputs().size(), 3U);
  EXPECT_TRUE(result.value().verify(make_op_query()).is_ok());
}

// M26 ARCH-061:多输出 forward 保留全部原输出,loss_output_index 可选择其中
// 任一标量目标,梯度追加在原输出前缀之后。
TEST(BuildBackwardGraphMultiOutputTest, PreservesOriginalOutputsAndAppendsGradient) {
  Graph graph("multi_output_forward");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  Node* relu_node = create_node_with_inferred_types(graph, "relu", {x}).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node =
      create_node_with_inferred_types(graph, "sum", {relu_node->output(0)}, sum_attrs).value();
  ASSERT_TRUE(graph.mark_output(sum_node->output(0)).is_ok());
  ASSERT_TRUE(graph.mark_output(relu_node->output(0)).is_ok());  // 第二个(非 loss)输出

  const std::vector<int32_t> wrt{0};
  const Result<Graph> result = build_backward_graph(graph, 0, wrt);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  ASSERT_EQ(result.value().outputs().size(), 3U);
  EXPECT_EQ(result.value().outputs()[0]->type().shape, Shape());
  EXPECT_EQ(result.value().outputs()[1]->type().shape, Shape({2, 3}));
  EXPECT_EQ(result.value().outputs()[2]->type().shape, Shape({2, 3}));
  EXPECT_TRUE(result.value().verify(make_op_query()).is_ok());
}

// ---------------------------------------------------------------------------
// 4. wrt 输入不在 loss 依赖链上时梯度补零(标准自动微分惯例,非错误;见文件
//    头注释第4条)。
// ---------------------------------------------------------------------------

TEST_F(AutogradTest, WrtInputNotOnDependencyChainGetsZeroGradient) {
  Graph graph("disconnected_wrt");
  Value* x = graph.add_graph_input(MakeType(DType::of<float>(), {3})).value();  // 参与 loss
  // 不参与 loss(仅登记为 graph input,不接入任何节点);Result 经 ASSERT_TRUE
  // 消费,不留具名 Value* 变量(未使用其指针本身)。
  const Result<Value*> unused_result = graph.add_graph_input(MakeType(DType::of<float>(), {2}));
  ASSERT_TRUE(unused_result.is_ok());
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* sum_node = create_node_with_inferred_types(graph, "sum", {x}, sum_attrs).value();
  ASSERT_TRUE(graph.mark_output(sum_node->output(0)).is_ok());

  const std::vector<int32_t> wrt{0, 1};
  const Result<Graph> training = build_backward_graph(graph, 0, wrt);
  ASSERT_TRUE(training.is_ok()) << training.status().message();

  const OpQuery query = make_op_query();
  ASSERT_TRUE(training.value().verify(query).is_ok());

  const Result<std::shared_ptr<Executable>> executable =
      frame::runtime::compile(training.value(), frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  Tensor x_tensor = MakeFilledTensor(Shape({3}), 1.0F, 1.0F);
  Tensor unused_tensor = MakeFilledTensor(Shape({2}), 9.0F, 1.0F);
  std::vector<Tensor> inputs{x_tensor, unused_tensor};
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  // outputs = [loss, grad_x, grad_unused];grad_unused 应全为 0(数值比较统一
  // 走 BUILD-011 容差工具,不手写 EXPECT_FLOAT_EQ)。
  ASSERT_EQ(outputs.value().size(), 3u);
  const Tensor& grad_unused = outputs.value()[2];
  EXPECT_EQ(grad_unused.shape(), Shape({2}));
  const Tensor expected_zero = MakeFilledTensor(Shape({2}), 0.0F, 0.0F);
  EXPECT_TRUE(tensor_all_close(grad_unused, expected_zero, default_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 5. M26 高阶自动微分:同一 build_backward_graph 重复作用。
// ---------------------------------------------------------------------------

TEST_F(AutogradTest, SquareSecondDerivativeEqualsTwo) {
  Graph forward("square_second_derivative");
  Value* x = forward.add_graph_input(MakeType(DType::of<float>(), {3})).value();
  Node* square = create_node_with_inferred_types(forward, "square", {x}).value();
  const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
  Node* loss =
      create_node_with_inferred_types(forward, "sum", {square->output(0)}, sum_attrs).value();
  ASSERT_TRUE(forward.mark_output(loss, 0).is_ok());

  const std::vector<int32_t> wrt{0};
  Result<Graph> first_result = build_backward_graph(forward, 0, wrt);
  ASSERT_TRUE(first_result.is_ok()) << first_result.status().message();
  Graph first = std::move(first_result.value());
  ASSERT_EQ(first.outputs().size(), 2U);
  Node* first_grad_sum =
      create_node_with_inferred_types(first, "sum", {first.outputs()[1]}, sum_attrs).value();
  ASSERT_TRUE(first.mark_output(first_grad_sum, 0).is_ok());

  const Result<Graph> second = build_backward_graph(first, 2, wrt);
  ASSERT_TRUE(second.is_ok()) << second.status().message();
  ASSERT_EQ(second.value().outputs().size(), 4U);
  Tensor x_tensor = MakeTensorFromDoubles({-1.5, 0.25, 2.0}, Shape({3}));
  std::vector<Tensor> inputs{x_tensor};
  const Result<std::vector<Tensor>> outputs = CompileAndRunCpu(second.value(), inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  const Tensor expected = MakeTensorFromDoubles({2.0, 2.0, 2.0}, Shape({3}));
  EXPECT_TRUE(
      tensor_all_close(outputs.value()[3], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(AutogradTest, SumKeepdimsGradientCanBeDifferentiatedAgain) {
  Graph forward("sum_keepdims_second_derivative");
  Value* x = forward.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  const AttrMap inner_attrs{{"axes", std::vector<int64_t>{1}}, {"keepdims", true}};
  Node* inner = create_node_with_inferred_types(forward, "sum", {x}, inner_attrs).value();
  const AttrMap all_axes{{"axes", std::vector<int64_t>{}}};
  Node* loss =
      create_node_with_inferred_types(forward, "sum", {inner->output(0)}, all_axes).value();
  ASSERT_TRUE(forward.mark_output(loss, 0).is_ok());

  const std::vector<int32_t> wrt{0};
  Result<Graph> first_result = build_backward_graph(forward, 0, wrt);
  ASSERT_TRUE(first_result.is_ok()) << first_result.status().message();
  Graph first = std::move(first_result.value());
  Node* first_grad_sum =
      create_node_with_inferred_types(first, "sum", {first.outputs()[1]}, all_axes).value();
  ASSERT_TRUE(first.mark_output(first_grad_sum, 0).is_ok());
  const Result<Graph> second = build_backward_graph(first, 2, wrt);
  ASSERT_TRUE(second.is_ok()) << second.status().message();

  Tensor x_tensor = MakeFilledTensor(Shape({2, 3}), -0.4F, 0.2F);
  std::vector<Tensor> inputs{x_tensor};
  const Result<std::vector<Tensor>> outputs = CompileAndRunCpu(second.value(), inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  const Tensor expected_zero = MakeFilledTensor(Shape({2, 3}), 0.0F, 0.0F);
  EXPECT_TRUE(tensor_all_close(outputs.value().back(), expected_zero,
                               default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(AutogradTest, MseLossSecondDerivativeMatchesAnalyticDiagonal) {
  Graph forward("mse_second_derivative");
  Value* pred = forward.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  Value* target = forward.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  Node* loss = create_node_with_inferred_types(forward, "mse_loss", {pred, target}).value();
  ASSERT_TRUE(forward.mark_output(loss, 0).is_ok());

  const std::vector<int32_t> wrt_pred{0};
  Result<Graph> first_result = build_backward_graph(forward, 0, wrt_pred);
  ASSERT_TRUE(first_result.is_ok()) << first_result.status().message();
  Graph first = std::move(first_result.value());
  const AttrMap all_axes{{"axes", std::vector<int64_t>{}}};
  Node* first_grad_sum =
      create_node_with_inferred_types(first, "sum", {first.outputs()[1]}, all_axes).value();
  ASSERT_TRUE(first.mark_output(first_grad_sum, 0).is_ok());
  const Result<Graph> second = build_backward_graph(first, 2, wrt_pred);
  ASSERT_TRUE(second.is_ok()) << second.status().message();

  Tensor pred_tensor = MakeFilledTensor(Shape({2, 3}), -0.5F, 0.2F);
  Tensor target_tensor = MakeFilledTensor(Shape({2, 3}), 0.25F, -0.1F);
  std::vector<Tensor> inputs{pred_tensor, target_tensor};
  const Result<std::vector<Tensor>> outputs = CompileAndRunCpu(second.value(), inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  const Tensor expected = MakeFilledTensor(Shape({2, 3}), 1.0F / 3.0F, 0.0F);
  EXPECT_TRUE(
      tensor_all_close(outputs.value().back(), expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(AutogradTest, ReluSecondDerivativeUsesZeroConventionIncludingKink) {
  Graph forward("relu_second_derivative");
  Value* x = forward.add_graph_input(MakeType(DType::of<float>(), {3})).value();
  Node* relu = create_node_with_inferred_types(forward, "relu", {x}).value();
  const AttrMap all_axes{{"axes", std::vector<int64_t>{}}};
  Node* loss = create_node_with_inferred_types(forward, "sum", {relu->output(0)}, all_axes).value();
  ASSERT_TRUE(forward.mark_output(loss, 0).is_ok());

  const std::vector<int32_t> wrt{0};
  Result<Graph> first_result = build_backward_graph(forward, 0, wrt);
  ASSERT_TRUE(first_result.is_ok()) << first_result.status().message();
  Graph first = std::move(first_result.value());
  Node* first_grad_sum =
      create_node_with_inferred_types(first, "sum", {first.outputs()[1]}, all_axes).value();
  ASSERT_TRUE(first.mark_output(first_grad_sum, 0).is_ok());
  const Result<Graph> second = build_backward_graph(first, 2, wrt);
  ASSERT_TRUE(second.is_ok()) << second.status().message();

  Tensor x_tensor = MakeTensorFromDoubles({-1.0, 0.0, 2.0}, Shape({3}));
  std::vector<Tensor> inputs{x_tensor};
  const Result<std::vector<Tensor>> outputs = CompileAndRunCpu(second.value(), inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  const Tensor expected_zero = MakeFilledTensor(Shape({3}), 0.0F, 0.0F);
  EXPECT_TRUE(tensor_all_close(outputs.value().back(), expected_zero,
                               default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(AutogradTest, NonSquareMatmulInternalGradientsDifferentiateBothDirections) {
  Graph forward("non_square_matmul_second_derivative");
  Value* a = forward.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  Value* b = forward.add_graph_input(MakeType(DType::of<float>(), {3, 4})).value();
  Node* product = create_node_with_inferred_types(forward, "matmul", {a, b}).value();
  const AttrMap sum_all{{"axes", std::vector<int64_t>{}}};
  Node* loss =
      create_node_with_inferred_types(forward, "sum", {product->output(0)}, sum_all).value();
  ASSERT_TRUE(forward.mark_output(loss, 0).is_ok());

  const std::vector<int32_t> wrt_both{0, 1};
  Result<Graph> first_result = build_backward_graph(forward, 0, wrt_both);
  ASSERT_TRUE(first_result.is_ok()) << first_result.status().message();
  Graph first = std::move(first_result.value());
  Node* grad_a_sum =
      create_node_with_inferred_types(first, "sum", {first.outputs()[1]}, sum_all).value();
  Node* grad_b_sum =
      create_node_with_inferred_types(first, "sum", {first.outputs()[2]}, sum_all).value();
  ASSERT_TRUE(first.mark_output(grad_a_sum, 0).is_ok());
  ASSERT_TRUE(first.mark_output(grad_b_sum, 0).is_ok());

  const std::vector<int32_t> wrt_b{1};
  const Result<Graph> derivative_of_grad_a = build_backward_graph(first, 3, wrt_b);
  ASSERT_TRUE(derivative_of_grad_a.is_ok()) << derivative_of_grad_a.status().message();
  Tensor a_tensor = MakeFilledTensor(Shape({2, 3}), -0.5F, 0.2F);
  Tensor b_tensor = MakeFilledTensor(Shape({3, 4}), 0.3F, -0.05F);
  std::vector<Tensor> inputs{a_tensor, b_tensor};
  const Result<std::vector<Tensor>> b_outputs =
      CompileAndRunCpu(derivative_of_grad_a.value(), inputs);
  ASSERT_TRUE(b_outputs.is_ok()) << b_outputs.status().message();
  const Tensor expected_b = MakeFilledTensor(Shape({3, 4}), 2.0F, 0.0F);
  EXPECT_TRUE(tensor_all_close(b_outputs.value().back(), expected_b,
                               default_tolerance(DTypeCode::kFloat32)));

  const std::vector<int32_t> wrt_a{0};
  const Result<Graph> derivative_of_grad_b = build_backward_graph(first, 4, wrt_a);
  ASSERT_TRUE(derivative_of_grad_b.is_ok()) << derivative_of_grad_b.status().message();
  const Result<std::vector<Tensor>> a_outputs =
      CompileAndRunCpu(derivative_of_grad_b.value(), inputs);
  ASSERT_TRUE(a_outputs.is_ok()) << a_outputs.status().message();
  const Tensor expected_a = MakeFilledTensor(Shape({2, 3}), 4.0F, 0.0F);
  EXPECT_TRUE(tensor_all_close(a_outputs.value().back(), expected_a,
                               default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(AutogradTest, MseGradientInternalDifferentiatesPredTargetAndUpstreamScalar) {
  Graph forward("mse_internal_all_input_derivatives");
  Value* pred = forward.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  Value* target = forward.add_graph_input(MakeType(DType::of<float>(), {2, 3})).value();
  Value* gy = forward.add_graph_input(MakeType(DType::of<float>(), {1})).value();
  Node* gpred =
      create_node_with_inferred_types(forward, "mse_loss_grad_internal", {pred, target, gy})
          .value();
  const AttrMap sum_all{{"axes", std::vector<int64_t>{}}};
  Node* scalar =
      create_node_with_inferred_types(forward, "sum", {gpred->output(0)}, sum_all).value();
  ASSERT_TRUE(forward.mark_output(scalar, 0).is_ok());

  const std::vector<int32_t> wrt{0, 1, 2};
  const Result<Graph> backward = build_backward_graph(forward, 0, wrt);
  ASSERT_TRUE(backward.is_ok()) << backward.status().message();
  Tensor pred_tensor = MakeFilledTensor(Shape({2, 3}), -0.6F, 0.2F);
  Tensor target_tensor = MakeFilledTensor(Shape({2, 3}), 0.3F, -0.1F);
  Tensor gy_tensor = MakeTensorFromDoubles({1.5}, Shape({1}));
  std::vector<Tensor> inputs{pred_tensor, target_tensor, gy_tensor};
  const Result<std::vector<Tensor>> outputs = CompileAndRunCpu(backward.value(), inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  const Tensor expected_pred = MakeFilledTensor(Shape({2, 3}), 0.5F, 0.0F);
  const Tensor expected_target = MakeFilledTensor(Shape({2, 3}), -0.5F, 0.0F);
  EXPECT_TRUE(
      tensor_all_close(outputs.value()[1], expected_pred, default_tolerance(DTypeCode::kFloat32)));
  EXPECT_TRUE(tensor_all_close(outputs.value()[2], expected_target,
                               default_tolerance(DTypeCode::kFloat32)));
  double expected_gy_value = 0.0;
  const float* pred_data = static_cast<const float*>(pred_tensor.raw_data());
  const float* target_data = static_cast<const float*>(target_tensor.raw_data());
  for (int64_t i = 0; i < pred_tensor.numel(); ++i) {
    expected_gy_value += (2.0 / 6.0) * static_cast<double>(pred_data[i] - target_data[i]);
  }
  const Tensor expected_gy = MakeTensorFromDoubles({expected_gy_value}, Shape({1}));
  EXPECT_TRUE(
      tensor_all_close(outputs.value()[3], expected_gy, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(AutogradTest, DisconnectedFirstGradientRemainsZeroUnderSecondTransform) {
  Graph forward("disconnected_second_derivative");
  Value* x = forward.add_graph_input(MakeType(DType::of<float>(), {3})).value();
  forward.add_graph_input(MakeType(DType::of<float>(), {3})).value();
  Node* square = create_node_with_inferred_types(forward, "square", {x}).value();
  const AttrMap sum_all{{"axes", std::vector<int64_t>{}}};
  Node* loss =
      create_node_with_inferred_types(forward, "sum", {square->output(0)}, sum_all).value();
  ASSERT_TRUE(forward.mark_output(loss, 0).is_ok());

  const std::vector<int32_t> wrt_disconnected{1};
  Result<Graph> first_result = build_backward_graph(forward, 0, wrt_disconnected);
  ASSERT_TRUE(first_result.is_ok()) << first_result.status().message();
  Graph first = std::move(first_result.value());
  Node* zero_sum =
      create_node_with_inferred_types(first, "sum", {first.outputs()[1]}, sum_all).value();
  ASSERT_TRUE(first.mark_output(zero_sum, 0).is_ok());
  const std::vector<int32_t> wrt_x{0};
  const Result<Graph> second = build_backward_graph(first, 2, wrt_x);
  ASSERT_TRUE(second.is_ok()) << second.status().message();

  Tensor x_tensor = MakeFilledTensor(Shape({3}), -0.5F, 0.5F);
  Tensor disconnected_tensor = MakeFilledTensor(Shape({3}), 0.25F, 0.25F);
  std::vector<Tensor> inputs{x_tensor, disconnected_tensor};
  const Result<std::vector<Tensor>> outputs = CompileAndRunCpu(second.value(), inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  const Tensor expected_zero = MakeFilledTensor(Shape({3}), 0.0F, 0.0F);
  EXPECT_TRUE(tensor_all_close(outputs.value().back(), expected_zero,
                               default_tolerance(DTypeCode::kFloat32)));
}

}  // namespace
