# 07 C++ 与工具

**适用版本：Frame v1.3 M28，C++20；Python 绑定 0.1.0（可选）**

**最后更新：2026-07-24**

本章是非规范性使用说明；C++ 导出、DSL 和 ONNX 边界分别以
[构建与测试](../../docs/standards/build-and-test.md)、[前端 DSL](../../docs/architecture/frontend-dsl.md)
和 [ONNX 互操作接口](../../include/frame/interop/onnx_weights.h)为准。

## 学习目标

在已完成独立 C++ 项目的基础上，使用导出目标、JSON DSL 和 ONNX initializer 权重交换。

## 前置

安装和独立消费工程的完整步骤见[00 从源码安装](../00-installation/README.md)和[01 仓外 C++ 项目](../01-quickstart/README.md)，本章不重复这些内容。

## 可运行代码

消费工程继续使用导出的安装包和公共聚合目标：

```cmake
find_package(frame CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE frame::frame)
```

`frame/frame.h` 是核心入口，但不包含 `nn`、`data`、`autograd` 或 `interop` 的专用头；使用这些功能时显式包含各自的公共头文件。

`frame_dslc` 的 `--check`、`--run` 和 `--emit` 面向 v0 JSON DSL。ONNX 接口仅交换具名 initializer 权重：

```cpp
const std::vector<frame::interop::NamedTensor> weights{{"weight", weight_tensor}};
frame::interop::save_onnx_weights("weights.onnx", weights);
auto loaded = frame::interop::load_onnx_weights("weights.onnx", *allocator).value();
```

## 预期输出

独立消费工程的验收是 CMake 成功找到 `frame` 并链接 `frame::frame`。DSL 和 ONNX 命令或示例的具体输出取决于输入文件；请运行相应工具或示例后记录实际结果。

## PyTorch 对照

PyTorch 常将模型、权重和执行一起交给 Python 生态工具。Frame 的 v0 JSON DSL 是受限前端，ONNX 接口只交换 initializer；二者都不等于通用模型导入、导出或任意 PyTorch 程序转换。

## 边界

- 只链接 `frame::frame`，不依赖 `src/` 内部头或手工拼接内部库。
- v0 JSON DSL 不表示动态 shape 或通用图导入。
- ONNX 接口不导入或导出计算图；示例 06 当前使用 CPU Tensor。

## 小结

- 安装和 `find_package` 的可复制步骤集中在第 00、01 章。
- 专用功能需要包含专用公共头。
- DSL 与 ONNX initializer 都有明确的 v0 边界。

## 练习

1. 复用第 01 章的外部工程，增加一个显式包含专用公共头的功能模块。
2. 运行仓内[示例 06](../../examples/06_onnx_weights/main.cpp)，确认保存和读取的名称一致。

## 下一章

教程主线到此结束；修改项目本身请读[08 开发者导览](../08-development/README.md)，遇到
运行问题请查[10 排错](../10-troubleshooting/README.md)。
