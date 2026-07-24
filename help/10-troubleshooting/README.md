# 10 排错与迁移

**适用版本：Frame v1.3 M28，C++20；Python 绑定 0.1.0（可选）**

**最后更新：2026-07-23**

本章是非规范性使用说明；冲突时以仓库 `docs/` 为准。

## 学习目标

沿 C++ configure、build、run 顺序复现问题，并将 eager 习惯迁移为编译图。

## 运行入口

从[示例 02](../../examples/02_graph_compile/main.cpp)的最小图开始，记录 preset、NVIDIA 驱动、Toolkit、backend、dtype、shape 和完整英文错误消息。CUDA 命令与缓存恢复见[06 后端与设备](../06-backends/README.md)。

## 核心概念与分步讲解

1. configure：Toolkit 不可见时按第 06 章设置环境变量；错误缓存时使用 `cmake --fresh --preset cuda`。
2. build：只构建相应示例 target，并以 CTest 验证示例 02 的 CUDA 路径。
3. run：核对图输入顺序，以及 Tensor 数量、shape、dtype、device；任一静态签名变化后重新编译。

| 症状 | 首先检查 |
|---|---|
| CUDA 未生成、构建或运行失败 | 第 06 章的 Toolkit、preset、target 与 CTest 命令。 |
| CUDA device mismatch | 图输入 device、编译后端、运行 Tensor device 是否一致。 |
| 训练更新异常 | 训练图的 forward/gradient 顺序，以及更新图的 param/gradient 顺序。 |
| 期待 ONNX 整图导入 | 当前仅交换 initializer；图仍用 Frame API 或允许前端建立。 |

## 能力边界

训练示例是 CPU 反向链；DataLoader 产出 CPU columns；ONNX 仅 initializer。不能把 CUDA 编译 fallback、ONNX 权重交换或 eager 路径当作完整 CUDA 训练、完整模型导入或主执行路径。

Python 绑定不可用时检查 `frame._core_available` 与[构建与测试](../../docs/standards/build-and-test.md)；`bfloat16.numpy()` 因 NumPy 无原生表示而失败。

## 本章小结

- 从最小编译图开始，而不是从复杂训练程序开始。
- 静态签名和 device 一致性是首要检查项。
- 固定能力边界可避免错误归因。

## 练习

1. 故意传入不匹配 shape 的 Tensor，确认错误指向静态签名问题。
2. 将一条 eager 算子链改写为 `Graph`、`mark_output()`、编译一次、重复运行的流程。

## 下一步

仍无法定位时，带最小 C++ 图复现回到[08 开发者导览](../08-development/README.md)定位实现与测试入口。
