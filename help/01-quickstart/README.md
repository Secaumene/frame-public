# 01 仓外 C++ 项目

**适用版本：Frame v1.3 M28，C++20；Python 绑定 0.1.0（可选）**

**最后更新：2026-07-25**

本章是非规范性使用说明；C++ 安装包导出以[构建与测试](../../docs/standards/build-and-test.md)为准。

## 学习目标

在 Frame 仓库之外创建、配置、链接并运行最短的独立 C++ 项目。

## 前置

完成[00 从源码安装](../00-installation/README.md)，并在当前 shell 中保留 `FRAME_PREFIX`。项目目录可以与 `<frame-source>` 平级，也可以位于任意其他位置；它不得引用 Frame 的 `src/`、`examples/` 或源码 `include/` 路径。

## 可运行代码

先在 Frame 仓库之外创建目录：

```bash
mkdir -p "$HOME/frame-tutorial/frame-quickstart"
cd "$HOME/frame-tutorial/frame-quickstart"
```

然后创建两个文件：

```text
frame-quickstart/
├── CMakeLists.txt
└── main.cpp
```

`CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.24)
project(frame_quickstart LANGUAGES CXX)

find_package(frame CONFIG REQUIRED)

add_executable(frame_quickstart main.cpp)
target_compile_features(frame_quickstart PRIVATE cxx_std_20)
target_link_libraries(frame_quickstart PRIVATE frame::frame)
```

`main.cpp`：

```cpp
#include <iostream>

#include <frame/core/shape.h>

int main() { std::cout << frame::Shape({2, 3}).numel() << '\n'; }
```

在 `frame-quickstart/` 中配置、构建并运行：

```bash
CMAKE_PREFIX_PATH="$FRAME_PREFIX" cmake -S . -B build
cmake --build build
./build/frame_quickstart
```

## 预期输出

```text
6
```

## PyTorch 对照

```python
import torch

print(torch.Size([2, 3]).numel())
```

两边都只计算 shape 的元素数，不分配运行期 Tensor。参考：[PyTorch `torch.Size`](https://docs.pytorch.org/docs/stable/size.html)。

## 继续运行 Tensor 示例

[00_quickstart](../../examples/00_quickstart/main.cpp) 由 CTest 构建并运行，章内源码与它保持一致。需要完整的运行期 Tensor 分配、读写示例时，继续阅读[01_tensor_basics](../../examples/01_tensor_basics/main.cpp)。

## 边界

- 本例只验证安装包消费与 `Shape::numel()`；不创建或分配运行期 Tensor。
- 章节中的项目由读者在仓外创建；仓内配套源码仅供学习和 CTest 覆盖，不是仓外工程的依赖。

## 小结

- 外部工程通过 `find_package(frame CONFIG REQUIRED)` 和 `frame::frame` 使用 Frame。
- `FRAME_PREFIX` 让 CMake 找到安装时导出的安装包。

## 练习

1. 在不移动 Frame 源码的前提下，把 `frame-quickstart` 放到另一目录并重新构建。
2. 将 shape 改为 `{3, 2}`，确认输出仍为 `6`。

## 下一章

继续[02 核心概念](../02-core-concepts/README.md)。
