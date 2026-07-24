# 09 添加算子

**适用版本：Frame v1.3 M28，C++20；Python 绑定 0.1.0（可选）**

**最后更新：2026-07-23**

本章是非规范性使用说明；冲突时以[算子系统](../../docs/architecture/operator-system.md)和相关规范为准。

## 学习目标

通过可运行 C++ 示例理解算子的注册、构图、编译和执行，并区分实验与正式贡献。

## 运行入口

构建并运行 `frame_example_03_custom_op`；完整源码为[03_custom_op](../../examples/03_custom_op/main.cpp)。正式贡献前先完成[08 开发者导览](../08-development/README.md)。

## 核心概念与分步讲解

1. `FRAME_REGISTER_OP("scaled_relu")` 声明 schema、静态 shape 推断与 trait。
2. CPU kernel 通过 `dispatch_dtype` 进行编译期类型分派，并以 `FRAME_REGISTER_KERNEL` 注册。
3. `create_node_with_inferred_types` 构图后，仍由 `runtime::compile` 形成执行计划。

外部可执行文件可保留自己的静态注册以验证应用自用算子；Python 绑定不是该核心实现的一部分。

## 能力边界

外部实验不等同于仓内支持。若要成为正式算子，必须按[算子系统](../../docs/architecture/operator-system.md)与权威规范完成复用搜索、schema、CPU reference、后端路线、梯度、测试、绑定和审查；本章不替代这些判据。

## 本章小结

- schema 与 kernel 注册是不同职责。
- 示例走完整编译路径，不是 eager 替代。
- 正式贡献有额外测试和治理要求。

## 练习

1. 运行示例 03，定位 schema 注册、kernel 注册和 `compile` 三处代码。
2. 为一个假想算子列出需先搜索的同名实现、算子组合和后端能力。

## 下一步

提交仓内改动前回到[08 开发者导览](../08-development/README.md)并执行其权威流程入口。
