// =============================================================================
// 示例 01:Tensor 基础。
//
// 学习目标:分配 CPU Tensor,查询 shape 与 dtype,并完成强类型数据读写。
// 前置章节:help/01-quickstart/README.md。
// 预期 PASS:程序返回 0,并打印设备、shape、dtype 与六个确定数值。
// 运行边界:本例只讲 Tensor;编译执行整图路径见示例 02。
// =============================================================================

#include <cstdint>
#include <iostream>

#include <frame/frame.h>

int main() {
  // Tensor::empty 需要一个 Allocator&:经 BackendRegistry 取 cpu 参考后端
  // (永远启用,见 include/frame/core/device.h)。
  const frame::Result<frame::hal::Backend*> backend_result =
      frame::hal::BackendRegistry::instance().get(frame::kCpuBackendName);
  if (!backend_result.is_ok()) {
    std::cerr << "failed to get cpu backend: " << backend_result.status().message() << "\n";
    return 1;
  }
  frame::hal::Backend* backend = backend_result.value();
  const frame::Device device = frame::cpu_device();
  frame::hal::Allocator* allocator = backend->allocator(device);

  // ---- 1. empty 分配:在指定设备上分配一个未初始化张量。----
  const frame::Shape shape({2, 3});
  const frame::Result<frame::Tensor> tensor_result =
      frame::Tensor::empty(shape, frame::DType::of<float>(), device, *allocator);
  if (!tensor_result.is_ok()) {
    std::cerr << "failed to allocate tensor: " << tensor_result.status().message() << "\n";
    return 1;
  }
  frame::Tensor tensor = tensor_result.value();
  std::cout << "allocated tensor on device backend '" << device.backend << "'\n";

  // ---- 2. shape 查询:rank / 各维尺寸 / 元素总数。----
  std::cout << "shape: " << tensor.shape().to_string() << "\n";
  std::cout << "rank: " << tensor.shape().rank() << "\n";
  std::cout << "numel: " << tensor.numel() << "\n";

  // ---- 3. dtype 查询:类型英文名与单元素字节数。----
  std::cout << "dtype: " << tensor.dtype().name() << "\n";
  std::cout << "itemsize (bytes): " << tensor.dtype().itemsize() << "\n";

  // ---- 4. 数据读写:data<T>() 取强类型指针,逐元素写入后再读回打印。----
  float* write_data = tensor.data<float>();
  for (int64_t i = 0; i < tensor.numel(); ++i) {
    write_data[i] = static_cast<float>(i) * 1.5F;
  }

  std::cout << "data:";
  const float* read_data = tensor.data<float>();
  for (int64_t i = 0; i < tensor.numel(); ++i) {
    std::cout << " " << read_data[i];
  }
  std::cout << "\n";

  return 0;
}
