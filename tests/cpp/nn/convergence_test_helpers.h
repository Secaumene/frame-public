#pragma once
// M21 批3 T7(docs/plan/2026-07-18-batch3-m21-conv.md 第1.4/3节):CNN/AFF/
// Wavelet 三个固定种子收敛测试(test_cnn_convergence.cpp/test_aff_convergence
// .cpp/test_wavelet_convergence.cpp)共用的训练循环 + 张量构造 + golden 比对
// helper——三文件的"nn 构图 -> add_parameter_inputs -> build_backward_graph ->
// build_sgd_update_graph -> runtime::compile("cpu") -> 训练循环 -> 参数轮换"
// 段落逐字雷同超过一屏,收敛为本地头(REUSE-002)。手法整体照抄
// tests/cpp/nn/test_training_smoke.cpp::NnTrainingSmokeTest 的训练线组装,
// 仅将"单个测试用例内联"的写法泛化为支持任意参数个数网络的可复用函数;golden
// 比对手法照抄 tests/cpp/frontend/test_lowering.cpp::AssertLoweringMatchesGolden
// (复用同一份 tests/cpp/compiler/golden_test_helpers.h::read_file_contents,
// REUSE-002)。本文件不是 nn 源码(include/frame/nn/、src/nn/)的一部分,是
// "调用方"角色的测试基础设施,依赖 compiler/runtime 头不违反 ARCH-070(同
// test_training_smoke.cpp 文件头注释判定)。

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <string>
#include <string_view>
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
#include <frame/ir/serialization.h>
#include <frame/nn/module.h>
#include <frame/ops/op_registry.h>
#include <frame/runtime/compile.h>

#include "../compiler/golden_test_helpers.h"

namespace frame::nn::testing {

// 构造 cpu 设备上的 TensorType(dtype 固定 float32,三网络收敛测试统一数据
// 类型)。与 src/nn/layers.cpp 匿名命名空间同名 helper 同构造口径,不同编译
// 单元各自独立持有一份(REUSE-002 既有取舍,同 test_training_smoke.cpp 头
// 注释先例)。
inline frame::ir::TensorType MakeCpuTensorType(std::vector<int64_t> dims) {
  frame::ir::TensorType type;
  type.dtype = frame::DType::of<float>();
  type.shape = frame::Shape(std::move(dims));
  type.device = frame::cpu_device();
  return type;
}

// 按 values 逐元素填充一个新分配的 cpu 张量(host 内存,可直接经指针写入,
// 手法同 test_training_smoke.cpp::NnTrainingSmokeTest::MakeTensorFromFloats)。
inline frame::Tensor MakeTensorFromFloats(const std::vector<float>& values,
                                          const frame::Shape& shape, frame::Device device,
                                          frame::hal::Allocator& allocator) {
  frame::Tensor tensor =
      frame::Tensor::empty(shape, frame::DType::of<float>(), device, allocator).value();
  float* data = tensor.data<float>();
  for (size_t i = 0; i < values.size(); ++i) data[i] = values[i];
  return tensor;
}

// 全批 SGD 训练循环(图输入序契约,调用方保证):[x, params(wrt 全体)...,
// target];恰 1 个 loss 输出(index 0)。每步:训练图 run 得 loss + 各参数
// 梯度 -> 更新图 run 得新参数 -> 参数指针轮换(Tensor 是共享 Storage 的值语义
// 句柄,重新赋值即完成"轮换到新一步参数",同 test_training_smoke.cpp 手法)。
// 返回逐步 loss 历史;任一环节失败原样透传底层 Status(调用方经
// ASSERT_TRUE(...is_ok())校验)。
// 相邻可转换形参 (learning_rate, num_steps) 与张量组为训练循环固定契约序,
// 调用点均以具名局部变量传入,误置换会使收敛断言立即失败(同 nchw_index
// 先例论证)。
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
inline frame::Result<std::vector<double>> RunFullBatchSgdTraining(
    frame::ir::Graph& forward, const frame::Tensor& x, std::vector<frame::Tensor> params,
    const std::vector<frame::ir::TensorType>& param_types, const frame::Tensor& target,
    double learning_rate, int num_steps) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  const frame::ir::OpQuery query = frame::ops::make_op_query();
  const frame::Status forward_verify_status = forward.verify(query);
  if (!forward_verify_status.is_ok()) return forward_verify_status;

  std::vector<int32_t> wrt(params.size());
  for (size_t i = 0; i < params.size(); ++i) wrt[i] = static_cast<int32_t>(i + 1);

  const frame::Result<frame::ir::Graph> training =
      frame::compiler::build_backward_graph(forward, /*loss_output_index=*/0, wrt);
  if (!training.is_ok()) return training.status();
  const frame::Result<std::shared_ptr<frame::hal::Executable>> train_executable =
      frame::runtime::compile(training.value(), frame::kCpuBackendName,
                              frame::hal::CompileOptions{});
  if (!train_executable.is_ok()) return train_executable.status();

  const frame::Result<frame::ir::Graph> update =
      frame::compiler::build_sgd_update_graph(param_types, learning_rate);
  if (!update.is_ok()) return update.status();
  const frame::Result<std::shared_ptr<frame::hal::Executable>> update_executable =
      frame::runtime::compile(update.value(), frame::kCpuBackendName, frame::hal::CompileOptions{});
  if (!update_executable.is_ok()) return update_executable.status();

  const size_t num_params = params.size();
  std::vector<double> loss_history;
  loss_history.reserve(static_cast<size_t>(num_steps));

  for (int step = 0; step < num_steps; ++step) {
    std::vector<frame::Tensor> train_inputs;
    train_inputs.reserve(num_params + 2);
    train_inputs.push_back(x);
    for (const frame::Tensor& p : params) train_inputs.push_back(p);
    train_inputs.push_back(target);

    const frame::Result<std::vector<frame::Tensor>> train_outputs =
        frame::runtime::run_with_allocated_outputs(*train_executable.value(),
                                                   frame::kCpuBackendName, train_inputs);
    if (!train_outputs.is_ok()) return train_outputs.status();
    if (train_outputs.value().size() != num_params + 1) {
      return frame::Status::make(
          frame::ErrorCode::kInternal,
          "RunFullBatchSgdTraining: unexpected train_outputs size at step " + std::to_string(step));
    }

    const frame::Tensor& loss_tensor = train_outputs.value()[0];
    const float loss_value = *static_cast<const float*>(loss_tensor.raw_data());
    if (!std::isfinite(loss_value)) {
      return frame::Status::make(
          frame::ErrorCode::kInternal,
          "RunFullBatchSgdTraining: loss is not finite at step " + std::to_string(step));
    }
    loss_history.push_back(static_cast<double>(loss_value));

    std::vector<frame::Tensor> update_inputs;
    update_inputs.reserve(num_params * 2);
    for (const frame::Tensor& p : params) update_inputs.push_back(p);
    for (size_t i = 0; i < num_params; ++i) update_inputs.push_back(train_outputs.value()[i + 1]);

    const frame::Result<std::vector<frame::Tensor>> update_outputs =
        frame::runtime::run_with_allocated_outputs(*update_executable.value(),
                                                   frame::kCpuBackendName, update_inputs);
    if (!update_outputs.is_ok()) return update_outputs.status();
    if (update_outputs.value().size() != num_params) {
      return frame::Status::make(
          frame::ErrorCode::kInternal,
          "RunFullBatchSgdTraining: unexpected update_outputs size at step " +
              std::to_string(step));
    }
    params = update_outputs.value();
  }

  return loss_history;
}

// 按 param_specs 声明序,逐个从 [lo, hi) 均匀分布(rng 驱动)抽取初值张量
// (三网络收敛测试的参数初始化统一走此口径,uniform_seeded,同
// test_training_smoke.cpp 头注释先例)。
inline std::vector<frame::Tensor> MakeUniformParamTensors(
    const std::vector<frame::nn::ParamSpec>& param_specs, std::mt19937& rng, float lo, float hi,
    frame::Device device, frame::hal::Allocator& allocator) {
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<frame::Tensor> tensors;
  tensors.reserve(param_specs.size());
  for (const frame::nn::ParamSpec& spec : param_specs) {
    const int64_t numel = spec.type.shape.numel();
    std::vector<float> values(static_cast<size_t>(numel));
    for (float& v : values) v = dist(rng);
    tensors.push_back(MakeTensorFromFloats(values, spec.type.shape, device, allocator));
  }
  return tensors;
}

// 提取 param_specs 各元素的 TensorType(build_sgd_update_graph 所需的
// param_types 形参,顺序与 param_specs/parameters() 一致)。
inline std::vector<frame::ir::TensorType> ParamTypesOf(
    const std::vector<frame::nn::ParamSpec>& param_specs) {
  std::vector<frame::ir::TensorType> types;
  types.reserve(param_specs.size());
  for (const frame::nn::ParamSpec& spec : param_specs) types.push_back(spec.type);
  return types;
}

// 前向图 dump_text 与 testdata golden 文件逐字节比对(手法同
// tests/cpp/frontend/test_lowering.cpp::AssertLoweringMatchesGolden,复用同一
// 份 tests/cpp/compiler/golden_test_helpers.h::read_file_contents,REUSE-002)。
inline ::testing::AssertionResult AssertGraphMatchesGolden(frame::ir::Graph& graph,
                                                           std::string_view expected_path) {
  const frame::ir::OpQuery query = frame::ops::make_op_query();
  const frame::Status verify_status = graph.verify(query);
  if (!verify_status.is_ok()) {
    return ::testing::AssertionFailure()
           << "AssertGraphMatchesGolden: graph.verify() failed: " << verify_status.message();
  }
  const std::string actual_text = frame::ir::dump_text(graph);

  const frame::Result<std::string> expected_result =
      frame::compiler::testing::read_file_contents(expected_path);
  if (!expected_result.is_ok()) {
    return ::testing::AssertionFailure()
           << "AssertGraphMatchesGolden: failed to load expected '" << expected_path
           << "': " << expected_result.status().message();
  }
  const std::string& expected_text = expected_result.value();

  if (actual_text != expected_text) {
    return ::testing::AssertionFailure()
           << "AssertGraphMatchesGolden: dump_text mismatch\n"
           << "--- actual ---\n"
           << actual_text << "--- expected (from " << expected_path << ") ---\n"
           << expected_text;
  }
  return ::testing::AssertionSuccess();
}

}  // namespace frame::nn::testing
