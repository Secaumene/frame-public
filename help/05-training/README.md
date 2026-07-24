# 05 训练

**适用版本：Frame v1.3 M28，C++20；Python 绑定 0.1.0（可选）**

**最后更新：2026-07-24**

本章是非规范性使用说明；自动微分与训练语义以[自动微分与训练架构](../../docs/architecture/autograd.md)为准。

## 学习目标

从 C++ 前向图派生反向图，并以独立 SGD 更新图完成 CPU 训练。

## 前置

完成[04 nn 与数据](../04-nn-and-data/README.md)。完整流程见仓内[示例 05](../../examples/05_training/main.cpp)。

## 可运行代码

在训练循环前分别建立并编译训练图和更新图：

```cpp
auto training = frame::compiler::build_backward_graph(forward, 0, wrt_indices).value();
auto train_exec = frame::runtime::compile(training, frame::kCpuBackendName, {}).value();
auto update = frame::compiler::build_sgd_update_graph(param_types, 0.05).value();
auto update_exec = frame::runtime::compile(update, frame::kCpuBackendName, {}).value();
```

训练图输出是前向输出后接所求参数的梯度；更新图输入为参数后接梯度，输出为新参数。循环只运行两个已编译对象，并将新参数 Tensor 传给下一步。

## 预期输出

示例 05 会在 CPU 反向链上报告训练过程中的数值。学习率、输入和迭代次数改变时输出也会改变；请以实际示例运行结果判断训练是否完成。

## PyTorch 对照

PyTorch 常把 `loss.backward()` 与 `torch.optim.SGD` 放在 eager 训练循环中。Frame 的 `build_backward_graph` 与 `build_sgd_update_graph` 分别构造两张静态图，之后才各自编译与运行。前者是概念基线，后者不是与其逐行或运行时行为等价的 API。

## 边界

- 完整训练示例当前走 CPU 反向链。
- 没有运行时 tape、Adam、momentum、动态学习率、混合精度训练、梯度检查点或完整 Hessian/Jacobian API。
- 固定学习率烘焙在更新图中；改变它应重建更新图。
- CUDA 不完整注册本例反向链所需 kernel，不能把 CUDA fallback 当作训练成功。

## 小结

- 训练和更新是两张独立编译图。
- 固定 shape 时，循环不需要重新编译。
- 每步更新返回新的 Tensor，而不是原地写入。

## 练习

1. 在示例 05 中确认参数、梯度和更新输出的顺序。
2. 将学习率改为另一常量，重建更新图并比较实际 loss 变化。

## 下一章

继续[06 后端与设备](../06-backends/README.md)。
