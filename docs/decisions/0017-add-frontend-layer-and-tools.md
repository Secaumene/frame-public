# ADR-0017:新增 frontend 前端层与 tools 工具目录(JSON DSL v0)

- 状态:已接受
- 日期:2026-07-13
- 关联铁律:#1 编译优先 / #2 C++ 核心 / #5 两级复用
- 关联规则:ARCH-001、ARCH-003、BUILD-002、REUSE-002、REUSE-010

## 背景

M0–M18 收口后,构图/自动微分/优化器图/编译执行均有公开 C++ API,
但缺「模型描述 → 可运行训练/推理」的用户面入口;README 架构图的
C++ Frontend API 仅为概念占位。用户裁决(2026-07-13):JSON 载体、
v0 仅 MLP、工具形态 --emit(生成 C++ 源码)与 --run(直接执行)
兼备。

## 决策

- 新增架构层 `src/frontend/` + `include/frame/frontend/`(库
  `frame_frontend`,别名 `frame::frontend`),依赖链扩为
  core←ir←ops←compiler←runtime←frontend(ARCH-001 同步修订);
  四能力 validate / lower / run / emit,只消费纯 C++ ModelSpec。
- 新增顶层 `tools/` 目录与可执行 `frame_dslc`
  (--check/--run/--emit 三模式);JSON 解析仅存在于工具层
  (nlohmann/json 准入限定「工具与测试代码」,引入经 ADR-0018),
  frontend 库不得 include nlohmann。
- v0 范围:linear 层(matmul + 可选 add bias + 可选 relu)、mse
  损失、sgd 优化器、float32、静态全形状(bias 全形状声明,首维 =
  batch;v0 无广播,ADR-0009 落地后升 schema_version 引入自然
  形状);数据 inline 内联或 seeded 均匀随机。
- 全部复用既有设施:create_node_with_inferred_types /
  build_backward_graph / build_sgd_update_graph / runtime::compile /
  run_with_allocated_outputs;frontend 不新增算子、不新增 pass、
  不触碰 IR 对象模型、不新开执行模式(--run 即编译路径复用,
  铁律 1 合规)。
- --emit 产物 = 自包含 main.cpp + CMakeLists.txt
  (find_package(frame),消费 BUILD-040 安装件)。
- 明确不做:Python 绑定 DSL、CNN/卷积、广播、YAML/TOML、变学习率。
- 判定方法:check_iron_rules.sh(core_dirs 扩 src/frontend)全绿;
  `grep -rn nlohmann include/ src/` 为空;frame_dslc --run 收敛
  冒烟进 cpu-only ctest;BUILD-002 正则覆盖 frame_frontend 与
  frame_dslc 两 target 名。

## 备选方案

- JSON 解析进 frontend 库:违 nlohmann「工具与测试代码」准入
  限定——否决。
- bias 声明 [hidden] 由 lower 物化 [batch,hidden]:训练中各 batch
  行 bias 独立演化,暗改共享 bias 语义——否决。
- 仅 --run 不 --emit:失去 AOT 源码产物形态,用户已裁两者兼备
  ——否决。

## 后果

- 正面:实体化 Frontend 层,打通「描述→训练→源码产物」链路。
- 负面与代价:新增一层维护面;ARCH-001 依赖链变更需同步文档与
  铁律脚本。
- 跟进:overview.md(ARCH-001+新 §2.10)、build-and-test.md、
  language-policy scope 枚举加 frontend/tools 并同步 commit-msg
  钩子中的正则副本(枚举扩展非规则放松)、
  check_iron_rules.sh 目录清单、frontend-dsl.md、README、PLAN.md。
