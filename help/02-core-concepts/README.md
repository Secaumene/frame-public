# 02 核心概念

**适用版本：Frame v1.3 M28，C++20；Python 绑定 0.1.0（可选）**

**最后更新：2026-07-24**

本章是非规范性使用说明；语义以[架构总览](../../docs/architecture/overview.md)、[执行模型](../../docs/architecture/execution-model.md)和[算子系统](../../docs/architecture/operator-system.md)为准。

## 学习目标

区分运行期数据、图内符号和编译结果，并理解 `Graph → compile → run` 的静态图闭环。

## 前置

完成[01 仓外 C++ 项目](../01-quickstart/README.md)。仓内的 [02_graph_compile](../../examples/02_graph_compile/main.cpp) 是完整图编译示例；它用于学习，不是外部工程的依赖。

## 可运行入口与核心代码

先在源码仓库运行完整 CPU 示例：

```bash
cd <frame-source>
cmake --preset dev
cmake --build --preset dev --target frame_example_02_graph_compile
./build/dev/examples/frame_example_02_graph_compile cpu
```

示例先声明输入，再用 `Graph::create_node` 或
`frame::ops::create_node_with_inferred_types` 建立算子节点，最后编译和运行：

```cpp
#include <frame/ops/graph_builder.h>

frame::ir::Graph graph("relu_graph");
// 图输入和算子属性的具体参数以算子 schema 为准。
// 节点输出是 Value，不能把运行期 Tensor 当作图内接线。
auto relu = frame::ops::create_node_with_inferred_types(graph, "relu", {input}).value();
graph.mark_output(relu, 0);

auto executable = frame::runtime::compile(graph, frame::kCpuBackendName, {}).value();
auto outputs = frame::runtime::run_with_allocated_outputs(
    *executable, frame::kCpuBackendName, inputs).value();
```

片段中的 `input` 和 `inputs` 分别来自图输入声明与第 01 章的运行期 Tensor；
完整示例展示了类型声明、输入分配和全部 `Result` 检查。

这里的 `Graph::create_node`/`create_node_with_inferred_types` 只构图；`Value`
表示图内 SSA 接线，而 `Tensor` 持有运行期数值缓冲。不要写 `Tensor + Tensor`
来代替图算子，也不要忽略真实代码中的 `Result` 检查。

```text
Graph → verify / pass → compile → Executable → run
```

## 预期输出

完整示例会打印后端、八个输出值和 `PASS`。外部项目的具体输出取决于你选择的
图和输入，应以实际运行结果验收。

## PyTorch 对照

PyTorch 的 `tensor = torch.relu(x)` 会立即得到 eager 结果，并在需要梯度时由 autograd 记录运算。Frame 先将节点连入 `Graph`，之后才 `compile` 并 `run`。`torch.compile` 也可能编译 PyTorch 程序，但它与 Frame 的 compile 不等价：两者的输入表示、捕获边界、运行时和支持范围不同。

## 边界

- v1 使用静态签名；改变 shape、dtype 或后端应重新构图并编译。
- 相同图、后端和选项可以复用 `Executable`。
- schema 负责约束和 shape 推断，后端 kernel 负责执行；支持检查在编译边界完成。

## 小结

- `Tensor` 是数据，`Value` 是图内连接。
- 图算子通过 `Graph::create_node` 或 `create_node_with_inferred_types` 构建。
- `Executable` 是编译产物，按图输入顺序接收运行期 Tensor。

## 练习

1. 在示例 02 中找到图输入、`Value` 和运行输入的对应关系。
2. 解释为何把输入 shape 从 `[2, 3]` 改为 `[3, 3]` 后不能直接复用原 `Executable`。

## 下一章

继续[04 nn 与数据](../04-nn-and-data/README.md)，或阅读可选的[03 PyTorch 对照与 Frame Python 绑定](../03-python-api/README.md)。
