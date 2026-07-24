# Frame C++ 示例

这里是[官方使用手册](../help/README.md)的配套可运行源码。建议先按编号学习，再回到
对应章节完成练习。

## 学习路线

| 顺序 | 示例 | 学习重点 | 对应章节 |
|---|---|---|---|
| 01 | [Tensor 基础](01_tensor_basics/main.cpp) | CPU Tensor 的分配、元数据与数据读写 | [仓外 C++ 项目](../help/01-quickstart/README.md) |
| 02 | [图编译](02_graph_compile/main.cpp) | 建图、编译、整图执行与结果校验 | [核心概念](../help/02-core-concepts/README.md) |
| 03 | [自定义算子](03_custom_op/main.cpp) | schema、kernel 注册与标准编译路径 | [新增算子](../help/09-add-operator/README.md) |
| 04 | [nn 与数据](04_nn_and_data/main.cpp) | 静态 nn 构图与 CPU 批数据迭代 | [nn 与数据](../help/04-nn-and-data/README.md) |
| 05 | [训练](05_training/main.cpp) | 前向、反向、SGD 更新图与 compile once | [训练](../help/05-training/README.md) |
| 06 | [ONNX 权重](06_onnx_weights/main.cpp) | initializer 权重的保存、加载与验证 | [C++ 与工具](../help/07-cpp-and-tools/README.md) |

## 构建与运行

`dev` preset 会构建六个示例：

```sh
cmake --preset dev
cmake --build --preset dev
ctest --preset dev -L examples --output-on-failure
```

单独运行 CPU 图编译示例：

```sh
./build/dev/examples/frame_example_02_graph_compile cpu
```

成功时程序返回 0，失败时返回非零。示例 02 可用 `cuda` 参数执行真实 CUDA
编译、H2D、整图执行与 D2H；CUDA 构建方法见[后端章节](../help/06-backends/README.md)。
示例 04、05、06 的教学主路径使用 CPU。示例 06 只交换 ONNX initializer，
不导入 ONNX 算子图，并会在校验后删除临时权重文件。
