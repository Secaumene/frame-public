# ADR-0020:新增 frame::nn 构图组合子层与 frame::data 数据加载层

- 状态:已接受
- 日期:2026-07-18
- 关联铁律:#1 #2 #3 #5
- 关联规则:ARCH-001/ARCH-003、ADR-010 触发情形(新增一级子目录)、ADR-0008、ADR-0017、CPP-010

## 背景

v1.2/v1.3 spec(docs/plan/v1_2-v1_3-networks-design.md 第 4 节,用户裁决
2026-07-17:交付形态=完整 nn 高层库 + 自带 DataLoader)要求 M21–M28 十二种
网络以模块化 API 交付;现状网络结构只能经 JSON DSL(仅 MLP)或手工构图,无批迭代设施。

## 决策

1. 新增 `include/frame/nn/` + `src/nn/`(namespace `frame::nn`):编译期构图
   组合子层。Module 契约:`build(ir::Graph&, 输入 Value*) -> Result<输出>`,
   构图一律经既有 `ops::create_node_with_inferred_types`(与 frontend
   lowering 同一份 helper,REUSE-002,不新建 GraphBuilder 包装);Module
   持参数声明(名/形状/dtype/初始化器)与子模块,**零 eager、不触数值**。
   组合机制禁白名单外 virtual(CPP-010):静态组合走模板/CRTP,异构容器
   (Sequential)与 Python 动态组合走类型擦除构建器(std::function/函数
   指针,同 KernelFn/GradientFn 先例);细化在批2设计章定稿。
2. 新增 `include/frame/data/` + `src/data/`(namespace `frame::data`):
   Dataset 接口 + TensorDataset + DataLoader(批迭代/固定种子洗牌/
   drop_last)。**data 仅依赖 core**;批张量组装的分配器由调用方经前向声明
   `hal::Allocator&` 形参注入(design-reviewer 裁定候选 b,同 interop/
   execute_fused_chain 先例);prefetch/磁盘格式记 v2.0(spec 第 11 节)。
3. 分层(ARCH-001 增两分支,类比 interop/hal 分支写法):nn 依赖集 =
   core+ir+ops,**不依赖 compiler/runtime/hal**——nn 只产出前向子图 +
   确定性有序参数清单,build_backward_graph / build_sgd_update_graph 的
   调用留在调用方(frontend runner 等);frontend→nn 单向,无环。
   frontend `lower_to_graph` 重构为经 nn 模块构图(铁律 #5)。
4. Python 面:`frame.nn` / `frame.data` pybind11 薄绑定 + `.pyi`(铁律 #2)。

判定方法:①`bash scripts/check_iron_rules.sh` 通过,且其 core_dirs 已扩
src/nn、src/data(见跟进);②overview.md ARCH-001 含 nn/data 分支;
③lowering 收敛机械判据:`grep -En 'create_node_with_inferred_types\(graph, *"(matmul|add|mul|relu|square|sum|mse_loss|conv1d|conv2d|max_pool2d|avg_pool2d|reshape|sigmoid|tanh|rsqrt|softmax|layer_norm|transpose|concat|slice|gather|rfft|irfft)"' src/frontend/lowering.cpp`
命中数 == 0(网络结构算子一律经 nn 模块构图);④Module 构图纯度与参数清单
确定性测试在仓;⑤`grep -rn "frame/compiler" include/frame/nn/ src/nn/` 零命中。

## 备选方案

- Python 层 Module 体系:核心逻辑落 python/,违铁律 #2——否决。
- 仅扩 JSON DSL:用户裁决已否,且分支/残差/物理损失结构表达不了——否决。
- nn 并入 frontend 目录:API 面与 DSL 解析面耦合,依赖方向颠倒——否决。
- Sequential 用 virtual 多态:CPP-010 白名单外,机械检查即打回——否决。

## 后果

- 正面:十二种网络统一模块表达面;DSL lowering 单一事实来源;示例经
  DataLoader 标准化喂批。
- 负面/代价:新增两分层的文档与检查维护;类型擦除构建器需在设计章写清
  生命周期与注册纪律,防演化成隐式 eager。
- 跟进:①批2 **task #1(硬前置)= docs/architecture/ 的 nn/data 设计章,
  经 design-reviewer APPROVE 后方可编码**(同 M16 先例);②扩
  scripts/check_iron_rules.sh core_dirs 增 src/nn、src/data(同 ADR-0017
  扩 frontend 先例,与实现同 PR);③overview.md 分层图、frontend-dsl.md
  lowering 底座注记;④spec §4 的 `build(GraphBuilder&,…)` 表述以本 ADR
  为准同步修正(INFO 级)。
