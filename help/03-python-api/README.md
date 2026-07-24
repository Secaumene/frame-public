# 03 PyTorch 对照与 Frame Python 绑定（可选）

**适用版本：Frame v1.3 M28，C++20；Python 绑定 0.1.0（可选）**

**最后更新：2026-07-24**

本章是非规范性使用说明；绑定实际接口以 [Python 类型存根](../../python/frame/_core.pyi) 和[Python 绑定规范](../../docs/standards/python-binding.md)为准。

## 学习目标

用 PyTorch 的常见概念定位 Frame Python 绑定，并识别不能逐行迁移的 API。

## 前置

完成[02 核心概念](../02-core-concepts/README.md)。Python 绑定是 C++ 核心的薄层入口；PyTorch 只是概念和代码对照基线，不是依赖或等价实现。

## 可运行代码

先确认扩展是否可用，再以 `Graph` 作为每个算子的第一个参数构图：

```python
import frame

print(frame._core_available)
if not frame._core_available:
    raise RuntimeError("frame C++ extension is unavailable")

graph = frame.Graph("python_graph")
x = graph.add_graph_input([2, 3], frame.DType.float32)
y = frame.relu(graph, x)
graph.mark_output(y)
executable = frame.compile(graph, "cpu")
```

## 真实映射总表

| PyTorch 概念或写法 | Frame Python 绑定 | 重要差异 |
|---|---|---|
| `torch.Tensor` | `frame.Tensor`、`frame.from_numpy()` | Frame Tensor 是 C++ 运行期值；`from_numpy` 复制到 CPU。 |
| `torch.add(x, y)`、`torch.relu(x)` | `frame.add(graph, x, y)`、`frame.relu(graph, x)` | 每个 Frame op 都显式接收 `Graph`，并返回图内 `Value`。 |
| Python 程序/`torch.compile` | `frame.Graph`、`frame.compile(graph, backend)`、`Executable.run(inputs)` | Frame 是显式静态图；不能将两种 compile 视为等价。 |
| autograd 记录和反向 | `build_backward_graph(...)` 与 `build_sgd_update_graph(...)` | 生成静态派生图，不存在逐步 tape 对应物。 |
| `torch.nn.Module` | `frame.nn.Module` 和各工厂 | Module 调用 `.build()` 构图，不保存运行期参数值。 |
| `torch.utils.data.DataLoader` | `frame.data.TensorDataset`、`frame.data.DataLoader` | 当前仅 CPU columns。 |
| `tensor.to(device)` | `Tensor.to(backend)` | 显式复制语义，完整图仍需匹配构图和编译后端。 |

`from_numpy()` 只接受连续的 `float32`、`float16`、`int32`、`int64` NumPy 数组，并初始复制到 CPU；`bfloat16` 没有 NumPy 原生表示。Frame 没有 `loss.backward()`、`parameter.grad` 或 `optimizer.step()` 的对应式 API。

## 预期输出

若扩展已构建，示例首先打印 `True`，并能构造 `Graph` 与 `Executable`。若打印
`False`，说明 Python 扩展不可用；这与 C++ 安装包及独立 C++ 教程是两条分发路径。

## PyTorch 对照

以下官方页面用于核对 PyTorch 自身语义，而不是为 Frame 作等价承诺：

- [torch](https://docs.pytorch.org/docs/stable/torch.html)
- [autograd](https://docs.pytorch.org/docs/stable/autograd.html)
- [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [数据工具](https://docs.pytorch.org/docs/stable/data.html)

## 边界

- Python 不改变 Frame 的静态 shape、dtype 与 device 一致性要求。
- C++ 系统安装不包含 Python 扩展；Python 绑定需按项目的 Python 安装入口单独构建。
- `Tensor.to(backend)` 的当前绑定固定使用目标后端的 device 0，不是任意多 GPU API。
- Python `Graph` 未暴露完成通用非标量高阶导所需的全部 C++ 组装接口；此类需求使用 C++ API。

## 小结

- Frame op 的显式 `Graph` 参数是最重要的迁移差别。
- Frame Python 绑定服务于静态图主线，不复制 PyTorch 的 eager/autograd 对象模型。

## 练习

1. 用连续 `float32` NumPy 数组验证 `from_numpy()` 和 `Tensor.numpy()` 的复制方向。
2. 对照类型存根，列出一次 `frame.add(graph, ...)` 与 PyTorch eager `torch.add(...)` 的输入差异。

## 下一章

返回 C++ 主线阅读[04 nn 与数据](../04-nn-and-data/README.md)。
