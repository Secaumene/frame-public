// =============================================================================
// 示例 02:图编译执行。
//
// 学习目标:完成建图、编译、整图执行与确定结果校验。
// 前置章节:help/02-core-concepts/README.md。
// 预期 PASS:打印所选后端及匹配期望值的 PASS 行。
// 运行边界:默认使用 CPU;传入 cuda 时真实执行 H2D、CUDA 整图与 D2H。
// =============================================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/executable.h>
#include <frame/hal/stream.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/runtime/compile.h>

namespace {

// 打印失败 Status 到 stderr 并返回 false;调用方据此立即返回。
bool check_example_02_status(const frame::Status& status, std::string_view operation) {
  if (status.is_ok()) return true;
  std::cerr << operation << " failed: " << status.message() << "\n";
  return false;
}

// TensorType 的 device 必须与本次选择的编译后端一致。
frame::ir::TensorType make_example_02_tensor_type(std::vector<int64_t> dims, frame::Device device) {
  frame::ir::TensorType type;
  type.dtype = frame::DType::of<float>();
  type.shape = frame::Shape(std::move(dims));
  type.device = device;
  return type;
}

}  // namespace

int main(int argc, char** argv) {
  // ---- 1. 解析后端参数。----
  if (argc > 2) {
    std::cerr << "usage: frame_example_02_graph_compile [cpu|cuda]\n";
    return 1;
  }
  const std::string_view requested_backend = argc == 2 ? argv[1] : frame::kCpuBackendName;
  if (requested_backend != frame::kCpuBackendName && requested_backend != frame::kCudaBackendName) {
    std::cerr << "backend must be either 'cpu' or 'cuda'\n";
    return 1;
  }
  const bool use_cuda = requested_backend == frame::kCudaBackendName;
  const std::string_view backend_name = use_cuda ? frame::kCudaBackendName : frame::kCpuBackendName;
  const frame::Device execution_device =
      use_cuda ? frame::Device{frame::kCudaBackendName, 0} : frame::cpu_device();

  // ---- 2. 取得目标后端,并确认目标设备存在。----
  const frame::Result<frame::hal::Backend*> backend_result =
      frame::hal::BackendRegistry::instance().get(backend_name);
  if (!backend_result.is_ok()) {
    std::cerr << "failed to get backend '" << backend_name
              << "': " << backend_result.status().message() << "\n";
    return 1;
  }
  frame::hal::Backend* execution_backend = backend_result.value();
  const frame::Result<int32_t> device_count_result = execution_backend->device_count();
  if (!device_count_result.is_ok()) {
    std::cerr << "failed to query device count: " << device_count_result.status().message() << "\n";
    return 1;
  }
  if (device_count_result.value() <= execution_device.index) {
    std::cerr << "backend '" << backend_name << "' has no device 0\n";
    return 1;
  }
  frame::hal::Allocator* execution_allocator = execution_backend->allocator(execution_device);
  if (execution_allocator == nullptr) {
    std::cerr << "backend '" << backend_name << "' returned a null allocator\n";
    return 1;
  }

  // CPU allocator 始终用于 host staging;CUDA 路径不会直接读写 device 指针。
  const frame::Result<frame::hal::Backend*> cpu_backend_result =
      frame::hal::BackendRegistry::instance().get(frame::kCpuBackendName);
  if (!cpu_backend_result.is_ok()) {
    std::cerr << "failed to get cpu staging backend: " << cpu_backend_result.status().message()
              << "\n";
    return 1;
  }
  frame::hal::Backend* cpu_backend = cpu_backend_result.value();
  const frame::Device cpu_device = frame::cpu_device();
  frame::hal::Allocator* cpu_allocator = cpu_backend->allocator(cpu_device);
  if (cpu_allocator == nullptr) {
    std::cerr << "cpu backend returned a null staging allocator\n";
    return 1;
  }

  // H2D/D2H 是异步 copy,CUDA 路径显式创建传输流并在每个传输阶段同步。
  std::unique_ptr<frame::hal::Stream> transfer_stream;
  if (use_cuda) {
    frame::Result<std::unique_ptr<frame::hal::Stream>> stream_result =
        execution_backend->create_stream(execution_device);
    if (!stream_result.is_ok()) {
      std::cerr << "failed to create cuda transfer stream: " << stream_result.status().message()
                << "\n";
      return 1;
    }
    transfer_stream = std::move(stream_result.value());
    if (transfer_stream == nullptr) {
      std::cerr << "cuda backend returned a null transfer stream\n";
      return 1;
    }
  }

  // ---- 3. 建图:matmul([2,3]x[3,4]) -> add(+bias[2,4]) -> relu。----
  frame::ir::Graph graph("matmul_add_relu_example");

  const frame::Result<frame::ir::Value*> x_result =
      graph.add_graph_input(make_example_02_tensor_type({2, 3}, execution_device));
  if (!check_example_02_status(x_result.status(), "add_graph_input(x)")) return 1;

  const frame::Result<frame::ir::Value*> w_result =
      graph.add_graph_input(make_example_02_tensor_type({3, 4}, execution_device));
  if (!check_example_02_status(w_result.status(), "add_graph_input(w)")) return 1;

  const frame::Result<frame::ir::Value*> bias_result =
      graph.add_graph_input(make_example_02_tensor_type({2, 4}, execution_device));
  if (!check_example_02_status(bias_result.status(), "add_graph_input(bias)")) return 1;

  const frame::Result<frame::ir::Node*> matmul_result =
      graph.create_node("matmul", {x_result.value(), w_result.value()},
                        {make_example_02_tensor_type({2, 4}, execution_device)});
  if (!check_example_02_status(matmul_result.status(), "create_node(matmul)")) return 1;

  const frame::Result<frame::ir::Node*> add_result =
      graph.create_node("add", {matmul_result.value()->output(0), bias_result.value()},
                        {make_example_02_tensor_type({2, 4}, execution_device)});
  if (!check_example_02_status(add_result.status(), "create_node(add)")) return 1;

  const frame::Result<frame::ir::Node*> relu_result =
      graph.create_node("relu", {add_result.value()->output(0)},
                        {make_example_02_tensor_type({2, 4}, execution_device)});
  if (!check_example_02_status(relu_result.status(), "create_node(relu)")) return 1;
  if (!check_example_02_status(graph.mark_output(relu_result.value(), 0), "mark_output")) return 1;

  // runtime::compile 是统一编译入口。CUDA 参数下传入 "cuda",图内 device 也
  // 已是 Device{"cuda",0},因此不会把 CPU 图伪装成 CUDA 执行。
  const frame::Result<std::shared_ptr<frame::hal::Executable>> executable_result =
      frame::runtime::compile(graph, backend_name, frame::hal::CompileOptions{});
  if (!check_example_02_status(executable_result.status(), "runtime::compile")) return 1;

  // ---- 4. 在 CPU staging Tensor 中填写确定输入。----
  const frame::Result<frame::Tensor> x_host_result = frame::Tensor::empty(
      frame::Shape({2, 3}), frame::DType::of<float>(), cpu_device, *cpu_allocator);
  if (!check_example_02_status(x_host_result.status(), "Tensor::empty(x_host)")) return 1;
  frame::Tensor x_host = x_host_result.value();
  const std::vector<float> x_values{1.0F, -2.0F, 3.0F, -4.0F, 5.0F, -6.0F};
  std::copy(x_values.begin(), x_values.end(), x_host.data<float>());

  const frame::Result<frame::Tensor> w_host_result = frame::Tensor::empty(
      frame::Shape({3, 4}), frame::DType::of<float>(), cpu_device, *cpu_allocator);
  if (!check_example_02_status(w_host_result.status(), "Tensor::empty(w_host)")) return 1;
  frame::Tensor w_host = w_host_result.value();
  const std::vector<float> w_values{1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 1.0F,
                                    0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F};
  std::copy(w_values.begin(), w_values.end(), w_host.data<float>());

  const frame::Result<frame::Tensor> bias_host_result = frame::Tensor::empty(
      frame::Shape({2, 4}), frame::DType::of<float>(), cpu_device, *cpu_allocator);
  if (!check_example_02_status(bias_host_result.status(), "Tensor::empty(bias_host)")) return 1;
  frame::Tensor bias_host = bias_host_result.value();
  std::fill(bias_host.data<float>(), bias_host.data<float>() + bias_host.numel(), 0.5F);

  // CPU 路径可直接复用 staging Tensor;CUDA 路径为每个输入分配显存并排入 H2D。
  const auto copy_to_execution_device =
      [&](const frame::Tensor& host_tensor) -> frame::Result<frame::Tensor> {
    if (!use_cuda) return host_tensor;
    frame::Result<frame::Tensor> device_tensor_result = frame::Tensor::empty(
        host_tensor.shape(), host_tensor.dtype(), execution_device, *execution_allocator);
    if (!device_tensor_result.is_ok()) return device_tensor_result.status();
    frame::Tensor device_tensor = device_tensor_result.value();
    const size_t bytes = static_cast<size_t>(host_tensor.numel()) * host_tensor.dtype().itemsize();
    const frame::Status copy_status =
        execution_backend->copy(device_tensor.raw_data(), execution_device, host_tensor.raw_data(),
                                cpu_device, bytes, transfer_stream.get());
    if (!copy_status.is_ok()) return copy_status;
    return device_tensor;
  };

  const frame::Result<frame::Tensor> x_input_result = copy_to_execution_device(x_host);
  const frame::Result<frame::Tensor> w_input_result = copy_to_execution_device(w_host);
  const frame::Result<frame::Tensor> bias_input_result = copy_to_execution_device(bias_host);
  if (!x_input_result.is_ok() || !w_input_result.is_ok() || !bias_input_result.is_ok()) {
    std::cerr << "failed to prepare execution inputs\n";
    return 1;
  }
  if (use_cuda &&
      !check_example_02_status(transfer_stream->synchronize(), "cuda H2D synchronize")) {
    return 1;
  }

  // ---- 5. 执行整图。----
  // run_with_allocated_outputs 在目标后端分配输出、创建执行流、执行并同步。
  const std::vector<frame::Tensor> inputs{x_input_result.value(), w_input_result.value(),
                                          bias_input_result.value()};
  const frame::Result<std::vector<frame::Tensor>> outputs_result =
      frame::runtime::run_with_allocated_outputs(*executable_result.value(), backend_name, inputs);
  if (!check_example_02_status(outputs_result.status(), "run_with_allocated_outputs")) return 1;
  if (outputs_result.value().size() != 1) {
    std::cerr << "execution returned an unexpected output count\n";
    return 1;
  }
  const frame::Tensor& execution_result = outputs_result.value()[0];

  // CUDA 输出先 D2H 到新的 CPU Tensor。只有 result_host 会被 host 解引用。
  frame::Tensor result_host = execution_result;
  if (use_cuda) {
    const frame::Result<frame::Tensor> host_result = frame::Tensor::empty(
        execution_result.shape(), execution_result.dtype(), cpu_device, *cpu_allocator);
    if (!check_example_02_status(host_result.status(), "Tensor::empty(result_host)")) return 1;
    result_host = host_result.value();
    const size_t bytes =
        static_cast<size_t>(execution_result.numel()) * execution_result.dtype().itemsize();
    const frame::Status copy_status =
        execution_backend->copy(result_host.raw_data(), cpu_device, execution_result.raw_data(),
                                execution_device, bytes, transfer_stream.get());
    if (!check_example_02_status(copy_status, "cuda D2H copy")) return 1;
    if (!check_example_02_status(transfer_stream->synchronize(), "cuda D2H synchronize")) {
      return 1;
    }
  }

  // ---- 6. 打印并与手算期望值比对。----
  std::cout << "backend: " << backend_name << "\n";
  std::cout << "relu(matmul(x, w) + bias) =";
  const float* result_data = static_cast<const float*>(result_host.raw_data());
  for (int64_t i = 0; i < result_host.numel(); ++i) {
    std::cout << " " << result_data[i];
  }
  std::cout << "\n";

  const std::vector<float> expected{1.5F, 0.0F, 3.5F, 2.5F, 0.0F, 5.5F, 0.0F, 0.0F};
  for (int64_t i = 0; i < result_host.numel(); ++i) {
    // 本例仅使用整数与二进制可精确表示的 0.5,可直接做精确比较。
    if (result_data[i] != expected[static_cast<size_t>(i)]) {
      std::cerr << "result does not match the hand-computed expected value\n";
      return 1;
    }
  }

  std::cout << "PASS: compiled " << backend_name << " execution matches the expected value\n";
  return 0;
}
