# 08 开发者导览

**适用版本：Frame v1.3 M28，C++20；Python 绑定 0.1.0（可选）**

**最后更新：2026-07-23**

本章是非规范性使用说明；冲突时以仓库 `docs/` 为准。本章不修改、解释或替代治理规则。

## 学习目标

按 C++ core-first 路径定位实现、测试和权威文档入口。

## 运行入口

先阅读[架构总览](../../docs/architecture/overview.md)、[执行模型](../../docs/architecture/execution-model.md)和[复用政策](../../docs/standards/reuse-policy.md)，再开始任何仓内改动。

## 核心概念与分步讲解

1. `include/frame/` 提供公共 C++20 契约；`src/` 实现 core、IR、ops、compiler、runtime、nn/data 与后端。
2. `tests/` 是行为证据；`docs/` 是规范、架构、后端与决策的单一事实来源。
3. Python 是 pybind11 薄绑定；后端专有实现位于 `src/backends/` 并经公共 HAL 接入。

```text
任务 → 阅读对应 docs/ → 查找复用 → 必要的设计门 → C++ 实现 → 测试 → 文档 → review
```

## 能力边界

本章只给出导航，不定义流程判据。复用、设计审查、测试、文档和 review 的要求以[复用政策](../../docs/standards/reuse-policy.md)、[C++ 编码规范](../../docs/standards/cpp-coding.md)和[构建与测试规范](../../docs/standards/build-and-test.md)为准。

## 本章小结

- 功能先在 C++ 核心实现。
- 文档规范和测试共同约束改动。
- 路由表决定额外流程门禁。

## 练习

1. 为一个拟改模块找到其 `include/`、`src/`、`tests/` 与权威 `docs/` 入口。
2. 在动手前为新增算子任务列出需要阅读的权威文档。

## 下一步

新算子继续[09 添加算子](../09-add-operator/README.md)；构建或环境失败时阅读[10 排错](../10-troubleshooting/README.md)。
