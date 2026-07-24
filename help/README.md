# Frame 动手学教程

**适用版本：Frame v1.3 M28，C++20；Python 绑定 0.1.0（可选）**

**最后更新：2026-07-24**

本手册是非规范性使用说明；实现能力、安装契约与边界以 [docs/](../docs/README.md) 为准。

## 学习目标

从源码安装 Frame，在仓库外建立一个独立 C++ 工程，再逐步学习静态图、nn、训练、设备和工具。

## 主线学习单元

主线章节按同一顺序组织：学习目标、前置、可运行入口或代码、预期输出、
PyTorch 对照、边界、小结、练习和下一章。PyTorch 只作为教学基线，帮助已有
经验的读者建立概念联系；它不是 Frame 的依赖，也不是等价实现。

## 建议路线

1. [00 从源码安装](00-installation/README.md) → [01 仓外 C++ 项目](01-quickstart/README.md) → [02 核心概念](02-core-concepts/README.md)。
2. [04 nn 与数据](04-nn-and-data/README.md) → [05 训练](05-training/README.md) → [06 后端与设备](06-backends/README.md) → [07 C++ 与工具](07-cpp-and-tools/README.md)。
3. [03 PyTorch 对照与 Frame Python 绑定](03-python-api/README.md)是可选参考，可在完成第 02 章后阅读。
4. 修改 Frame 本身时阅读[08 开发者导览](08-development/README.md)和
   [09 添加算子](09-add-operator/README.md)；遇到问题先查[10 排错](10-troubleshooting/README.md)。

## 配套源码

| 章节 | 配套源码 | 作用 |
|---|---|---|
| 01 | 章内完整仓外项目；[01_tensor_basics](../examples/01_tensor_basics/main.cpp) | 安装包消费与 CPU Tensor 基础。 |
| 02、06 | [02_graph_compile](../examples/02_graph_compile/main.cpp) | CPU/CUDA 静态图编译执行。 |
| 04 | [04_nn_and_data](../examples/04_nn_and_data/main.cpp) | nn 构图与 CPU 批数据。 |
| 05 | [05_training](../examples/05_training/main.cpp) | 前向、反向与 SGD 更新图。 |
| 07 | [06_onnx_weights](../examples/06_onnx_weights/main.cpp) | ONNX initializer 权重交换。 |
| 09 | [03_custom_op](../examples/03_custom_op/main.cpp) | 自定义算子注册与编译路径。 |

第 01 章的项目必须由读者创建在 Frame 仓库之外；上表中的示例 01 只是同一
公共 API 的仓内参考，不是仓外项目的源码依赖。

## 教程边界

- 主线使用导出的 C++ 包 `frame::frame`，不依赖仓库内部 `src/`、`examples/` 或源码 `include/` 路径。
- `Graph → compile → run` 是主线；运行期 `Tensor` 的分配需要显式后端与 allocator。
- 各章节的可运行代码是学习入口，完整能力和支持矩阵仍以相应 `docs/` 文档为准。
