// =============================================================================
// 示例 05:CPU 编译期训练。
//
// 学习目标:构造前向、反向与 SGD 更新图,编译一次并复用 Executable 完成训练。
// 前置章节:help/05-training/README.md。
// 预期 PASS:打印下降的初始/最终 loss 与训练图复用成功的 PASS 行。
// 运行边界:完整训练闭环固定使用 CPU;真实 CUDA 图执行见示例 02。
// =============================================================================

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
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
#include <frame/nn/layers.h>
#include <frame/nn/module.h>
#include <frame/ops/op_registry.h>
#include <frame/runtime/compile.h>

namespace {

constexpr int64_t kBatch = 8;
constexpr int64_t kInputDim = 4;
constexpr int64_t kHiddenDim = 8;
constexpr int64_t kOutputDim = 1;

// 构造训练图统一使用的 float32 CPU 静态类型。
frame::ir::TensorType make_example_05_tensor_type(std::vector<int64_t> dims) {
  frame::ir::TensorType type;
  type.dtype = frame::DType::of<float>();
  type.shape = frame::Shape(std::move(dims));
  type.device = frame::cpu_device();
  return type;
}

// 把确定的 host 浮点数组写入一个新 CPU Tensor。
frame::Result<frame::Tensor> make_example_05_tensor(const std::vector<float>& values,
                                                    const frame::Shape& shape,
                                                    frame::hal::Allocator& allocator) {
  frame::Result<frame::Tensor> tensor_result =
      frame::Tensor::empty(shape, frame::DType::of<float>(), frame::cpu_device(), allocator);
  if (!tensor_result.is_ok()) return tensor_result.status();
  frame::Tensor tensor = tensor_result.value();
  if (tensor.numel() != static_cast<int64_t>(values.size())) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "example tensor value count does not match shape");
  }
  for (size_t i = 0; i < values.size(); ++i) tensor.data<float>()[i] = values[i];
  return tensor;
}

}  // namespace

int main() {
  // ---- 1. 构造前向 MLP 与标量 MSE loss。----
  frame::ir::Graph forward("training_example");
  const frame::Result<frame::ir::Value*> x_result =
      forward.add_graph_input(make_example_05_tensor_type({kBatch, kInputDim}));
  if (!x_result.is_ok()) {
    std::cerr << "failed to add x input: " << x_result.status().message() << "\n";
    return 1;
  }

  // 第二个 Linear 不带 bias,得到三个参数。第一个 Linear 的 v0 bias 与完整
  // 输出同形 [batch,hidden],因此参数输入顺序为 [w0,b0,w2]。
  const frame::nn::Module model = frame::nn::Sequential(
      "mlp", {frame::nn::Linear("0", kBatch, kInputDim, kHiddenDim, /*with_bias=*/true,
                                frame::DType::of<float>()),
              frame::nn::Relu("1"),
              frame::nn::Linear("2", kBatch, kHiddenDim, kOutputDim, /*with_bias=*/false,
                                frame::DType::of<float>())});
  const std::vector<frame::nn::ParamSpec> parameter_specs = model.parameters();
  if (parameter_specs.size() != 3) {
    std::cerr << "model returned an unexpected parameter count\n";
    return 1;
  }
  const frame::Result<std::vector<frame::ir::Value*>> parameter_inputs =
      frame::nn::add_parameter_inputs(forward, parameter_specs);
  if (!parameter_inputs.is_ok()) {
    std::cerr << "failed to add parameter inputs: " << parameter_inputs.status().message() << "\n";
    return 1;
  }
  const frame::Result<std::vector<frame::ir::Value*>> prediction_result = model.build(
      forward, std::vector<frame::ir::Value*>{x_result.value()}, parameter_inputs.value());
  if (!prediction_result.is_ok() || prediction_result.value().size() != 1) {
    std::cerr << "failed to build forward model";
    if (!prediction_result.is_ok()) std::cerr << ": " << prediction_result.status().message();
    std::cerr << "\n";
    return 1;
  }

  // target 最后加入,所以训练可执行体的输入顺序严格为 [x,w0,b0,w2,target]。
  const frame::Result<frame::ir::Value*> target_result =
      forward.add_graph_input(make_example_05_tensor_type({kBatch, kOutputDim}));
  if (!target_result.is_ok()) {
    std::cerr << "failed to add target input: " << target_result.status().message() << "\n";
    return 1;
  }
  const frame::Result<std::vector<frame::ir::Value*>> loss_result =
      frame::nn::MseLoss("loss").build(
          forward,
          std::vector<frame::ir::Value*>{prediction_result.value()[0], target_result.value()},
          std::vector<frame::ir::Value*>{});
  if (!loss_result.is_ok() || loss_result.value().size() != 1) {
    std::cerr << "failed to build mse loss";
    if (!loss_result.is_ok()) std::cerr << ": " << loss_result.status().message();
    std::cerr << "\n";
    return 1;
  }
  const frame::Status mark_status = forward.mark_output(loss_result.value()[0]);
  if (!mark_status.is_ok()) {
    std::cerr << "failed to mark loss output: " << mark_status.message() << "\n";
    return 1;
  }
  const frame::ir::OpQuery query = frame::ops::make_op_query();
  const frame::Status forward_verify_status = forward.verify(query);
  if (!forward_verify_status.is_ok()) {
    std::cerr << "forward graph verification failed: " << forward_verify_status.message() << "\n";
    return 1;
  }

  // ---- 2. 派生并验证反向图。----
  // wrt 使用 forward.inputs() 下标,1/2/3 正好对应 w0/b0/w2。
  const std::vector<int32_t> wrt_indices{1, 2, 3};
  const frame::Result<frame::ir::Graph> training_graph =
      frame::compiler::build_backward_graph(forward, /*loss_output_index=*/0, wrt_indices);
  if (!training_graph.is_ok()) {
    std::cerr << "failed to build backward graph: " << training_graph.status().message() << "\n";
    return 1;
  }
  const frame::Status training_verify_status = training_graph.value().verify(query);
  if (!training_verify_status.is_ok()) {
    std::cerr << "backward graph verification failed: " << training_verify_status.message() << "\n";
    return 1;
  }

  // ---- 3. 派生并验证 SGD 更新图。----
  const std::vector<frame::ir::TensorType> parameter_types{
      parameter_specs[0].type, parameter_specs[1].type, parameter_specs[2].type};
  constexpr double kLearningRate = 0.05;
  const frame::Result<frame::ir::Graph> update_graph =
      frame::compiler::build_sgd_update_graph(parameter_types, kLearningRate);
  if (!update_graph.is_ok()) {
    std::cerr << "failed to build sgd update graph: " << update_graph.status().message() << "\n";
    return 1;
  }
  const frame::Status update_verify_status = update_graph.value().verify(query);
  if (!update_verify_status.is_ok()) {
    std::cerr << "sgd update graph verification failed: " << update_verify_status.message() << "\n";
    return 1;
  }

  // ---- 4. 两张派生图各编译一次。----
  // 下面的循环只复用这两个句柄,不会在每一步重新构图或重新编译。
  const frame::Result<std::shared_ptr<frame::hal::Executable>> training_executable =
      frame::runtime::compile(training_graph.value(), frame::kCpuBackendName,
                              frame::hal::CompileOptions{});
  if (!training_executable.is_ok()) {
    std::cerr << "failed to compile backward graph: " << training_executable.status().message()
              << "\n";
    return 1;
  }
  const frame::Result<std::shared_ptr<frame::hal::Executable>> update_executable =
      frame::runtime::compile(update_graph.value(), frame::kCpuBackendName,
                              frame::hal::CompileOptions{});
  if (!update_executable.is_ok()) {
    std::cerr << "failed to compile sgd update graph: " << update_executable.status().message()
              << "\n";
    return 1;
  }

  // ---- 5. 取得 CPU 分配器并准备确定性训练数据。----
  const frame::Result<frame::hal::Backend*> backend_result =
      frame::hal::BackendRegistry::instance().get(frame::kCpuBackendName);
  if (!backend_result.is_ok()) {
    std::cerr << "failed to get cpu backend: " << backend_result.status().message() << "\n";
    return 1;
  }
  frame::hal::Allocator* allocator = backend_result.value()->allocator(frame::cpu_device());
  if (allocator == nullptr) {
    std::cerr << "cpu backend returned a null allocator\n";
    return 1;
  }

  // 固定种子与小范围初始化来自仓库训练冒烟用例的已校准配方。
  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 random_engine(20260713U);
  std::uniform_real_distribution<float> data_distribution(-1.0F, 1.0F);
  std::uniform_real_distribution<float> weight_distribution(-0.1F, 0.1F);
  std::uniform_real_distribution<float> bias_distribution(0.5F, 1.0F);

  std::vector<float> x_values(static_cast<size_t>(kBatch * kInputDim));
  for (float& value : x_values) value = data_distribution(random_engine);
  std::vector<float> true_weight_values(static_cast<size_t>(kInputDim * kOutputDim));
  for (float& value : true_weight_values) value = data_distribution(random_engine);
  std::vector<float> w0_values(static_cast<size_t>(kInputDim * kHiddenDim));
  for (float& value : w0_values) value = weight_distribution(random_engine);
  std::vector<float> b0_values(static_cast<size_t>(kBatch * kHiddenDim));
  for (float& value : b0_values) value = bias_distribution(random_engine);
  std::vector<float> w2_values(static_cast<size_t>(kHiddenDim * kOutputDim));
  for (float& value : w2_values) value = weight_distribution(random_engine);

  // target=x@true_weight 只用于生成固定监督数据,不参与参数更新实现。
  std::vector<float> target_values(static_cast<size_t>(kBatch * kOutputDim), 0.0F);
  for (int64_t row = 0; row < kBatch; ++row) {
    for (int64_t column = 0; column < kOutputDim; ++column) {
      float value = 0.0F;
      for (int64_t inner = 0; inner < kInputDim; ++inner) {
        value += x_values[static_cast<size_t>(row * kInputDim + inner)] *
                 true_weight_values[static_cast<size_t>(inner * kOutputDim + column)];
      }
      target_values[static_cast<size_t>(row * kOutputDim + column)] = value;
    }
  }

  frame::Result<frame::Tensor> x_tensor_result =
      make_example_05_tensor(x_values, frame::Shape({kBatch, kInputDim}), *allocator);
  frame::Result<frame::Tensor> target_tensor_result =
      make_example_05_tensor(target_values, frame::Shape({kBatch, kOutputDim}), *allocator);
  frame::Result<frame::Tensor> w0_result =
      make_example_05_tensor(w0_values, frame::Shape({kInputDim, kHiddenDim}), *allocator);
  frame::Result<frame::Tensor> b0_result =
      make_example_05_tensor(b0_values, frame::Shape({kBatch, kHiddenDim}), *allocator);
  frame::Result<frame::Tensor> w2_result =
      make_example_05_tensor(w2_values, frame::Shape({kHiddenDim, kOutputDim}), *allocator);
  if (!x_tensor_result.is_ok() || !target_tensor_result.is_ok() || !w0_result.is_ok() ||
      !b0_result.is_ok() || !w2_result.is_ok()) {
    std::cerr << "failed to allocate training tensors\n";
    return 1;
  }
  const frame::Tensor& x_tensor = x_tensor_result.value();
  const frame::Tensor& target_tensor = target_tensor_result.value();
  frame::Tensor w0 = w0_result.value();
  frame::Tensor b0 = b0_result.value();
  frame::Tensor w2 = w2_result.value();

  // ---- 6. 执行固定步数训练。----
  constexpr int kTrainingSteps = 300;
  float initial_loss = 0.0F;
  float final_loss = 0.0F;
  for (int step = 0; step < kTrainingSteps; ++step) {
    // 反向可执行体沿用前向输入顺序,输出顺序为 [loss,grad_w0,grad_b0,grad_w2]。
    const std::vector<frame::Tensor> training_inputs{x_tensor, w0, b0, w2, target_tensor};
    const frame::Result<std::vector<frame::Tensor>> training_outputs =
        frame::runtime::run_with_allocated_outputs(*training_executable.value(),
                                                   frame::kCpuBackendName, training_inputs);
    if (!training_outputs.is_ok() || training_outputs.value().size() != 4) {
      std::cerr << "training execution failed at step " << step;
      if (!training_outputs.is_ok()) {
        std::cerr << ": " << training_outputs.status().message();
      }
      std::cerr << "\n";
      return 1;
    }
    const float loss = *static_cast<const float*>(training_outputs.value()[0].raw_data());
    if (!std::isfinite(loss)) {
      std::cerr << "training produced a non-finite loss at step " << step << "\n";
      return 1;
    }
    if (step == 0) initial_loss = loss;
    final_loss = loss;

    // 更新图输入顺序是 [params...,grads...],输出是同序的新参数。
    // SGD 数学与分配由 build_sgd_update_graph 生成的图负责,示例不做原地更新。
    const std::vector<frame::Tensor> update_inputs{w0,
                                                   b0,
                                                   w2,
                                                   training_outputs.value()[1],
                                                   training_outputs.value()[2],
                                                   training_outputs.value()[3]};
    const frame::Result<std::vector<frame::Tensor>> update_outputs =
        frame::runtime::run_with_allocated_outputs(*update_executable.value(),
                                                   frame::kCpuBackendName, update_inputs);
    if (!update_outputs.is_ok() || update_outputs.value().size() != 3) {
      std::cerr << "sgd update execution failed at step " << step;
      if (!update_outputs.is_ok()) std::cerr << ": " << update_outputs.status().message();
      std::cerr << "\n";
      return 1;
    }

    // Tensor 是共享 Storage 的值语义句柄,赋值即可切换到更新图产出的新参数。
    w0 = update_outputs.value()[0];
    b0 = update_outputs.value()[1];
    w2 = update_outputs.value()[2];
  }

  // 收敛判据只要求最终有限损失严格低于初始损失,不依赖平台逐位一致。
  if (!std::isfinite(final_loss) || !(final_loss < initial_loss)) {
    std::cerr << "training loss did not decrease: initial=" << initial_loss
              << " final=" << final_loss << "\n";
    return 1;
  }
  std::cout << "Training loss: " << initial_loss << " -> " << final_loss << "\n";
  std::cout << "PASS: training graphs compiled once and loss decreased\n";
  return 0;
}
