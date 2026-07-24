// =============================================================================
// 示例 06:ONNX initializer 权重。
//
// 学习目标:保存具名 CPU 权重,重新加载并按名称验证 dtype、shape 与数值。
// 前置章节:help/07-cpp-and-tools/README.md。
// 预期 PASS:打印 ONNX initializer 权重按名称往返验证成功的 PASS 行。
// 运行边界:只交换 graph.initializer,不是 ONNX 算子图导入器。
// =============================================================================

#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/interop/onnx_weights.h>

int main() {
  // 文件名固定且位于当前工作目录;CTest 为本示例设置独立二进制目录。
  const std::filesystem::path file_path = "frame_example_06_weights.onnx";

  // 手工运行时拒绝覆盖当前目录中的同名文件。这样即使后续初始化失败,
  // 也不会把用户已有文件误判为本次示例的临时产物。
  std::error_code exists_error;
  const bool file_exists = std::filesystem::exists(file_path, exists_error);
  if (exists_error) {
    std::cerr << "failed to inspect the ONNX output path: " << exists_error.message() << "\n";
    return 1;
  }
  if (file_exists) {
    std::cerr << "refusing to overwrite the existing ONNX output file\n";
    return 1;
  }

  // 只有进入保存阶段后,本次运行才取得该路径的清理所有权。error_code
  // 重载不会抛异常;保存失败留下的部分文件也会被尽力清理。
  bool owns_file = false;
  const auto cleanup_file = [&file_path, &owns_file]() {
    if (!owns_file) {
      return;
    }
    std::error_code remove_error;
    std::filesystem::remove(file_path, remove_error);
  };
  const auto fail = [&cleanup_file](const std::string& message) {
    std::cerr << message << "\n";
    cleanup_file();
    return 1;
  };

  // ---- 1. 取得 CPU 分配器。----
  const frame::Result<frame::hal::Backend*> backend_result =
      frame::hal::BackendRegistry::instance().get(frame::kCpuBackendName);
  if (!backend_result.is_ok()) {
    return fail(std::string("failed to get cpu backend: ") +
                std::string(backend_result.status().message()));
  }
  frame::hal::Allocator* allocator = backend_result.value()->allocator(frame::cpu_device());
  if (allocator == nullptr) return fail("cpu backend returned a null allocator");

  // ---- 2. 创建三份小型 CPU 权重并写入确定值。----
  const frame::Result<frame::Tensor> weight_result = frame::Tensor::empty(
      frame::Shape({2, 3}), frame::DType::of<float>(), frame::cpu_device(), *allocator);
  if (!weight_result.is_ok()) {
    return fail(std::string("failed to allocate encoder.weight: ") +
                std::string(weight_result.status().message()));
  }
  frame::Tensor weight = weight_result.value();
  const std::vector<float> expected_weight{1.0F, -2.0F, 3.5F, 4.0F, 0.25F, -0.5F};
  for (size_t i = 0; i < expected_weight.size(); ++i) {
    weight.data<float>()[i] = expected_weight[i];
  }

  const frame::Result<frame::Tensor> bias_result = frame::Tensor::empty(
      frame::Shape({3}), frame::DType::of<frame::float16_t>(), frame::cpu_device(), *allocator);
  if (!bias_result.is_ok()) {
    return fail(std::string("failed to allocate encoder.bias: ") +
                std::string(bias_result.status().message()));
  }
  frame::Tensor bias = bias_result.value();
  const std::vector<frame::float16_t> expected_bias{
      frame::float_to_float16(0.5F), frame::float_to_float16(-1.0F), frame::float_to_float16(2.0F)};
  for (size_t i = 0; i < expected_bias.size(); ++i) {
    bias.data<frame::float16_t>()[i] = expected_bias[i];
  }

  const frame::Result<frame::Tensor> scale_result = frame::Tensor::empty(
      frame::Shape({1}), frame::DType::of<frame::bfloat16_t>(), frame::cpu_device(), *allocator);
  if (!scale_result.is_ok()) {
    return fail(std::string("failed to allocate projection.scale: ") +
                std::string(scale_result.status().message()));
  }
  frame::Tensor scale = scale_result.value();
  const frame::bfloat16_t expected_scale = frame::float_to_bfloat16(1.25F);
  scale.data<frame::bfloat16_t>()[0] = expected_scale;

  // 名字是交换边界的一部分;加载方应按名字匹配,不能依赖文件内排列顺序。
  const std::vector<frame::interop::NamedTensor> weights{
      {"encoder.weight", weight}, {"projection.scale", scale}, {"encoder.bias", bias}};

  // ---- 3. 保存 initializer-only ONNX 文件并重新加载。----
  owns_file = true;
  const frame::Status save_status = frame::interop::save_onnx_weights(file_path.string(), weights);
  if (!save_status.is_ok()) {
    return fail(std::string("failed to save ONNX weights: ") + std::string(save_status.message()));
  }
  const frame::Result<std::vector<frame::interop::NamedTensor>> loaded_result =
      frame::interop::load_onnx_weights(file_path.string(), *allocator);
  if (!loaded_result.is_ok()) {
    return fail(std::string("failed to load ONNX weights: ") +
                std::string(loaded_result.status().message()));
  }
  if (loaded_result.value().size() != weights.size()) {
    return fail("loaded ONNX weight count does not match");
  }

  // ---- 4. 建立 name -> NamedTensor 映射。----
  std::unordered_map<std::string, const frame::interop::NamedTensor*> loaded_by_name;
  for (const frame::interop::NamedTensor& named_tensor : loaded_result.value()) {
    const bool inserted = loaded_by_name.emplace(named_tensor.name, &named_tensor).second;
    if (!inserted) return fail("loaded ONNX weights contain a duplicate name");
  }
  const auto weight_it = loaded_by_name.find("encoder.weight");
  const auto bias_it = loaded_by_name.find("encoder.bias");
  const auto scale_it = loaded_by_name.find("projection.scale");
  if (weight_it == loaded_by_name.end() || bias_it == loaded_by_name.end() ||
      scale_it == loaded_by_name.end()) {
    return fail("loaded ONNX weights are missing an expected name");
  }

  // ---- 5. 分别验证 dtype、shape 与逐元素数值。----
  const frame::Tensor& loaded_weight = weight_it->second->tensor;
  if (loaded_weight.dtype().code() != frame::DTypeCode::kFloat32 ||
      !(loaded_weight.shape() == frame::Shape({2, 3}))) {
    return fail("encoder.weight metadata does not match");
  }
  const float* loaded_weight_data = static_cast<const float*>(loaded_weight.raw_data());
  for (size_t i = 0; i < expected_weight.size(); ++i) {
    if (loaded_weight_data[i] != expected_weight[i]) {
      return fail("encoder.weight values do not match");
    }
  }

  const frame::Tensor& loaded_bias = bias_it->second->tensor;
  if (loaded_bias.dtype().code() != frame::DTypeCode::kFloat16 ||
      !(loaded_bias.shape() == frame::Shape({3}))) {
    return fail("encoder.bias metadata does not match");
  }
  const frame::float16_t* loaded_bias_data =
      static_cast<const frame::float16_t*>(loaded_bias.raw_data());
  for (size_t i = 0; i < expected_bias.size(); ++i) {
    if (loaded_bias_data[i].bits != expected_bias[i].bits) {
      return fail("encoder.bias values do not match");
    }
  }

  const frame::Tensor& loaded_scale = scale_it->second->tensor;
  if (loaded_scale.dtype().code() != frame::DTypeCode::kBFloat16 ||
      !(loaded_scale.shape() == frame::Shape({1}))) {
    return fail("projection.scale metadata does not match");
  }
  const frame::bfloat16_t* loaded_scale_data =
      static_cast<const frame::bfloat16_t*>(loaded_scale.raw_data());
  if (loaded_scale_data[0].bits != expected_scale.bits) {
    return fail("projection.scale value does not match");
  }

  cleanup_file();
  std::cout << "PASS: ONNX initializer weights saved, loaded, and verified by name\n";
  return 0;
}
