#pragma once
// 梯度检查测试公共件(M21,批3 T4):"解析梯度 ≡ 数值微分"端到端断言的可复用
// 实现,供 tests/cpp/ops/ 下新增算子(conv2d/conv1d/max_pool2d/avg_pool2d/
// reshape/sigmoid)各自的梯度检查用例共用(REUSE-002:同一份 build_backward_graph
// → verify → runtime::compile("cpu") → run 管线,本目录内只留一份实现,禁止
// 每个新增算子测试文件各自抄一份)。管线与
// tests/cpp/compiler/test_autograd.cpp::CheckGradientMatchesNumeric 同构——该
// 函数是那个文件局部匿名命名空间内的实现、未对外导出,故本文件是同一套已验证
// 手法在 tests/cpp/ops/ 目录下的独立可复用落地(不是也不需要改动该既有文件)。
// 容差经 relaxed_tolerance(BUILD-011「解析梯度 ≡ 数值微分校验」专款:数值
// 微分侧自带 O(h²) 截断误差,明文放宽一档)。

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include <frame/compiler/autograd.h>
#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/executable.h>
#include <frame/ir/graph.h>
#include <frame/ops/op_registry.h>
#include <frame/runtime/compile.h>

#include "../common/numeric_gradient.h"
#include "../common/tolerance.h"

namespace frame::ops::testing {

// 解析梯度 ≡ 数值微分核心断言(ARCH-066):对 forward 调用 build_backward_graph
// → verify → runtime::compile("cpu") → 编译产物读回 wrt_indices[wrt_position]
// 位置的解析梯度,与 numeric_gradient 对同一编译产物的 loss 输出(outputs[0])
// 做中心差分所得数值梯度比较。inputs 与 forward.inputs() 按位对应,调用方须
// 保证其在整个断言期间保持有效(numeric_gradient 原地扰动后会复原,断言结束时
// inputs 与调用前一致)。h(步长)由调用方显式传入(BUILD-011:不同算子/输入
// 量级的数值条件不同,不应共享同一硬编码步长)。
inline ::testing::AssertionResult CheckGradientMatchesNumeric(
    const frame::ir::Graph& forward, int32_t loss_output_index,
    const std::vector<int32_t>& wrt_indices, size_t wrt_position,
    std::vector<frame::Tensor>& inputs, double h) {
  const frame::Result<frame::ir::Graph> training =
      frame::compiler::build_backward_graph(forward, loss_output_index, wrt_indices);
  if (!training.is_ok()) {
    return ::testing::AssertionFailure()
           << "build_backward_graph failed: " << training.status().message();
  }

  const frame::ir::OpQuery query = frame::ops::make_op_query();
  const frame::Status verify_status = training.value().verify(query);
  if (!verify_status.is_ok()) {
    return ::testing::AssertionFailure()
           << "training graph verify() failed: " << verify_status.message();
  }

  const frame::Result<std::shared_ptr<frame::hal::Executable>> executable = frame::runtime::compile(
      training.value(), frame::kCpuBackendName, frame::hal::CompileOptions{});
  if (!executable.is_ok()) {
    return ::testing::AssertionFailure()
           << "runtime::compile failed: " << executable.status().message();
  }

  auto loss_fn = [&](const frame::Tensor&) -> frame::Result<double> {
    const frame::Result<std::vector<frame::Tensor>> run_outputs =
        frame::runtime::run_with_allocated_outputs(*executable.value(), frame::kCpuBackendName,
                                                   inputs);
    if (!run_outputs.is_ok()) return run_outputs.status();
    return static_cast<double>(*static_cast<const float*>(run_outputs.value()[0].raw_data()));
  };

  const auto wrt_input_index = static_cast<size_t>(wrt_indices[wrt_position]);
  const frame::Result<std::vector<double>> numeric =
      frame::testing::numeric_gradient(loss_fn, inputs[wrt_input_index], h);
  if (!numeric.is_ok()) {
    return ::testing::AssertionFailure()
           << "numeric_gradient failed: " << numeric.status().message();
  }

  const frame::Result<std::vector<frame::Tensor>> outputs =
      frame::runtime::run_with_allocated_outputs(*executable.value(), frame::kCpuBackendName,
                                                 inputs);
  if (!outputs.is_ok()) {
    return ::testing::AssertionFailure() << outputs.status().message();
  }
  const frame::Tensor& analytic = outputs.value()[1 + wrt_position];

  const frame::Result<frame::hal::Backend*> backend_result =
      frame::hal::BackendRegistry::instance().get(frame::kCpuBackendName);
  if (!backend_result.is_ok()) {
    return ::testing::AssertionFailure() << backend_result.status().message();
  }
  frame::hal::Allocator* allocator = backend_result.value()->allocator(frame::cpu_device());
  const frame::Result<frame::Tensor> numeric_tensor_result = frame::Tensor::empty(
      analytic.shape(), frame::DType::of<float>(), frame::cpu_device(), *allocator);
  if (!numeric_tensor_result.is_ok()) {
    return ::testing::AssertionFailure() << numeric_tensor_result.status().message();
  }
  frame::Tensor numeric_tensor = numeric_tensor_result.value();
  float* numeric_data = numeric_tensor.data<float>();
  for (size_t i = 0; i < numeric.value().size(); ++i) {
    numeric_data[i] = static_cast<float>(numeric.value()[i]);
  }

  return frame::testing::tensor_all_close(
      analytic, numeric_tensor, frame::testing::relaxed_tolerance(frame::DTypeCode::kFloat32));
}

}  // namespace frame::ops::testing
