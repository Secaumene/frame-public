# 04 nn 与数据

**适用版本：Frame v1.3 M28，C++20；Python 绑定 0.1.0（可选）**

**最后更新：2026-07-24**

本章是非规范性使用说明；nn/data 语义以[nn/data 架构](../../docs/architecture/nn-design.md)为准。

## 学习目标

以 C++ 建立固定 batch 的 `frame::nn` 图，并用 `DataLoader` 提供批数据。

## 前置

完成[02 核心概念](../02-core-concepts/README.md)。完整可运行流程见仓内[示例 04](../../examples/04_nn_and_data/main.cpp)。

## 可运行代码

先声明普通图输入，按确定顺序追加参数输入，再调用 Module 的 `build()`：

```cpp
frame::ir::Value* features = graph.add_graph_input(feature_type).value();
auto params = frame::nn::add_parameter_inputs(graph, model.parameters()).value();
frame::ir::Value* output = model.build(graph, {features}, params).value()[0];
graph.mark_output(output);
```

`Linear` 的 weight 是 `[in_dim, out_dim]`。当 `with_bias=true` 时，bias 是 `[batch, out_dim]`，所以 batch 属于静态签名。`TensorDataset` 和 `DataLoader` 按列提供批数据；`drop_last=true` 可保持固定 batch。

## 预期输出

示例 04 会构建模型图并迭代 CPU 批数据。具体批次数和数值取决于示例输入；请以实际运行输出验收，而不是把本章描述当作已执行结果。

## PyTorch 对照

PyTorch 的 `nn.Module` 通常持有参数值，并在调用时 eager 执行；`torch.utils.data.DataLoader` 常与 workers、pinned memory 和预取配置协同。Frame `Module` 只保存模块与参数元信息、只负责构图，不保存运行期参数值。Frame `DataLoader` 仅处理 CPU columns，不承诺 workers、pinned memory 或 prefetch。

## 边界

- 每列必须非空、rank 至少为 1，且 axis 0 样本数相同。
- DataLoader 的 CPU 限制不代表主图必须在 CPU；其他后端的数据迁移由调用方遵循 device 契约完成。

## 小结

- `Module.build()` 产出图内节点，不是即时数值计算。
- 参数输入顺序由模块规格确定。
- 固定 batch 和 `drop_last` 与静态图相配合。

## 练习

1. 将示例 04 的样本数设为不能整除 batch，观察 `drop_last=true` 的影响。
2. 写出 features、参数输入与图输出在一次编译前的各自作用。

## 下一章

继续[05 训练](../05-training/README.md)。
