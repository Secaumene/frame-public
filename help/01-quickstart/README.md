# 01 仓外 C++ 项目

**适用版本：Frame v1.3 M28，C++20；Python 绑定 0.1.0（可选）**

**最后更新：2026-07-24**

本章是非规范性使用说明；C++ 安装包导出以[构建与测试](../../docs/standards/build-and-test.md)为准。

## 学习目标

在 Frame 仓库之外创建、配置、链接并运行第一个独立 C++ 项目。

## 前置

完成[00 从源码安装](../00-installation/README.md)，并在当前 shell 中保留 `FRAME_PREFIX`。项目目录可以与 `<frame-source>` 平级，也可以位于任意其他位置；它不得引用 Frame 的 `src/`、`examples/` 或源码 `include/` 路径。

## 可运行代码

先在 Frame 仓库之外创建目录：

```bash
mkdir -p "$HOME/frame-tutorial/frame-hello"
cd "$HOME/frame-tutorial/frame-hello"
```

然后创建两个文件：

```text
frame-hello/
├── CMakeLists.txt
└── main.cpp
```

`CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.24)
project(frame_hello LANGUAGES CXX)

find_package(frame CONFIG REQUIRED)

add_executable(frame_hello main.cpp)
target_compile_features(frame_hello PRIVATE cxx_std_20)
target_link_libraries(frame_hello PRIVATE frame::frame)
```

`main.cpp` 复用 Tensor 基础示例的分配和读写模式：

```cpp
#include <cstdint>
#include <iostream>

#include <frame/frame.h>

int main() {
  // 经 CPU 后端取得运行期 Tensor 所需的 allocator。
  const frame::Result<frame::hal::Backend*> backend_result =
      frame::hal::BackendRegistry::instance().get(frame::kCpuBackendName);
  if (!backend_result.is_ok()) {
    std::cerr << "failed to get cpu backend: "
              << backend_result.status().message() << "\n";
    return 1;
  }

  frame::hal::Backend* backend = backend_result.value();
  const frame::Device device = frame::cpu_device();
  frame::hal::Allocator* allocator = backend->allocator(device);
  if (allocator == nullptr) {
    std::cerr << "failed to get cpu allocator\n";
    return 1;
  }

  const frame::Result<frame::Tensor> tensor_result = frame::Tensor::empty(
      frame::Shape({2, 3}), frame::DType::of<float>(), device, *allocator);
  if (!tensor_result.is_ok()) {
    std::cerr << "failed to allocate tensor: "
              << tensor_result.status().message() << "\n";
    return 1;
  }

  frame::Tensor tensor = tensor_result.value();
  float* write_data = tensor.data<float>();
  for (int64_t i = 0; i < tensor.numel(); ++i) {
    write_data[i] = static_cast<float>(i) * 1.5F;
  }

  std::cout << "backend: " << device.backend << "\n";
  std::cout << "shape: " << tensor.shape().to_string() << "\n";
  std::cout << "dtype: " << tensor.dtype().name() << "\n";
  std::cout << "data:";
  const float* read_data = tensor.data<float>();
  for (int64_t i = 0; i < tensor.numel(); ++i) {
    std::cout << " " << read_data[i];
  }
  std::cout << "\n";
  return 0;
}
```

在 `frame-hello/` 中配置、构建并运行：

```bash
CMAKE_PREFIX_PATH="$FRAME_PREFIX" cmake -S . -B build
cmake --build build
./build/frame_hello
```

## 预期输出

```text
backend: cpu
shape: [2, 3]
dtype: float32
data: 0 1.5 3 4.5 6 7.5
```

## PyTorch 对照

```python
import torch

tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3) * 1.5
print(tensor.shape, tensor.dtype, tensor.device, tensor.flatten().tolist())
```

这段 PyTorch 代码是 eager 计算；本例的 Frame `Tensor` 是经显式 allocator 分配的运行期缓冲。两者用于对照数据形状和数值，不代表 API 或执行模型等价。参考：[PyTorch tensor 文档](https://docs.pytorch.org/docs/stable/torch.html)。

## 边界

- 本例不伪造 `zeros`、`ones` 或 Tensor 运算符；它只演示真实的 `Tensor::empty`、分配、写入与读取。
- 所有 `Result` 与 allocator 都必须检查，后续图执行还会加入构图与编译边界。

## 小结

- 外部工程通过 `find_package(frame CONFIG REQUIRED)` 和 `frame::frame` 使用 Frame。
- `FRAME_PREFIX` 让 CMake 找到安装时导出的安装包。

## 练习

1. 在不移动 Frame 源码的前提下，把 `frame-hello` 放到另一目录并重新构建。
2. 将 shape 改为 `{3, 2}`，并相应检查输出的 shape 与数据顺序。

## 下一章

继续[02 核心概念](../02-core-concepts/README.md)。
