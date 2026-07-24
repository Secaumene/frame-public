# 06 后端与设备

**适用版本：Frame v1.3 M28，C++20；Python 绑定 0.1.0（可选）**

**最后更新：2026-07-24**

本章是非规范性使用说明；后端能力以[后端支持矩阵](../../docs/backends/README.md)和[CUDA 后端](../../docs/backends/cuda.md)为准。

## 学习目标

理解 Frame 的 device 契约，选择后端并验证 CUDA 静态图执行路径。

## 前置

完成[02 核心概念](../02-core-concepts/README.md)。CPU 始终可用；NVIDIA GPU 的完整运行入口是仓内[示例 02](../../examples/02_graph_compile/main.cpp)。

## 可运行代码

在具备 CUDA Toolkit 的环境中，可用 CUDA preset 构建并运行示例：

```bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDAToolkit_ROOT=/usr/local/cuda
cd <frame-source>
cmake --preset cuda
cmake --build --preset cuda --target frame_example_02_graph_compile
ctest --preset cuda -R example_02_graph_compile_cuda --output-on-failure
./build/cuda/examples/frame_example_02_graph_compile cuda
```

构图、运行期 Tensor 与编译目标必须一致：

```cpp
const frame::Device device{frame::kCudaBackendName, 0};
auto executable = frame::runtime::compile(graph, frame::kCudaBackendName, {}).value();
auto outputs = frame::runtime::run_with_allocated_outputs(
    *executable, frame::kCudaBackendName, inputs).value();
```

## 预期输出

通过 CUDA 示例时，CTest 会报告该示例通过，程序完成 H2D、GPU 图执行、D2H 和结果校验。没有实际运行时，不应把这些结果写成已验证事实。

## PyTorch 对照

`tensor.to(device)` 可帮助理解“数据需要搬到目标设备”这一概念。Frame 的 device 必须同时贯穿图内 `TensorType`、运行期 `Tensor` 和 `compile` 的 backend；CUDA 场景还要显式处理 H2D/D2H。这个对照不表示存在任意多 GPU API，也不表示两者的设备迁移模型等价。

## 边界

- 图输入 device、编译后端和运行 Tensor device 必须一致。
- CUDA 算子覆盖率以 CUDA 后端文档为准；Intel GPU、Intel NPU 和 Ascend 状态只以后端支持矩阵为准。
- CUDA 图编译示例不等于完整训练可用。
- 若 CMake 缓存保留 `CMAKE_CUDA_COMPILER=NOTFOUND`，可在保留当前 shell 环境后执行：

```bash
cmake --fresh --preset cuda
cmake --build --preset cuda
```

## 小结

- device 是图类型、运行数据和编译后端的共同契约。
- CUDA 数据传输与图执行需要显式处理。
- 示例 02 是 NVIDIA GPU 的验证入口。

## 练习

1. 在有 CUDA 环境的机器上运行示例 02，并保存实际 CTest 输出。
2. 比较 CPU 和 CUDA 两次运行中的 backend 与 Tensor device。

## 下一章

继续[07 C++ 与工具](../07-cpp-and-tools/README.md)。
