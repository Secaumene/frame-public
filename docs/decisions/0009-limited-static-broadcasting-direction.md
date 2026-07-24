# ADR-0009:受限静态广播方向(实施另立里程碑)

- 状态:已接受
- 日期:2026-07-13
- 关联铁律:#1 编译优先 / #5 两级复用
- 关联规则:ARCH-013、ARCH-042、ARCH-044、ARCH-052

## 背景

v0 刻意不支持广播(elementwise 要求同 shape),该约束已两次暴露表达力
痛点:MLP 的 bias 只能建模为与激活同形的 [batch, hidden] 张量(每样本
独立偏置,见 tests/cpp/compiler/test_training_loop.cpp 头注释);SGD 的
标量学习率须按参数 shape 全展开为 constant。2026-07-13 项目所有者将
广播语义列入议题,本 ADR 裁决方向。

## 决策

接受**受限静态广播**方向。本 ADR 为方向裁决,实施前置 = 专项里程碑立项
并通过设计评审;裁决内容:

- 语义子集:NumPy 右对齐规则的静态子集——两输入 shape 自右对齐,逐维
  要求相等或其一为 1;结果维取两者较大值;**仅静态 shape**(ARCH-013
  边界不动,动态 shape 仍属独立 ADR 议题);**仅 elementwise 家族**
  参与(add/mul 等双目;matmul/reduction/loss 不引入广播)。
- 工程影响清单(实施里程碑的设计范围,先行锁定不遗漏):shape_infer
  双目规则与 V4/verify 口径;cpu/cuda kernel 步幅化(0 步幅视图或索引
  映射,内层循环禁 dtype/形态运行时分支,ARCH-042);operator_fusion
  等价键与融合链同 shape 前提;梯度侧新增 sum-to-shape 反向(广播的
  伴随算子);**编译产物缓存键**(src/runtime/compile.cpp::make_cache_key,
  须确认广播变体完整体现于图文本、不产生键碰撞);序列化/golden 全量回归。
  回退执行与 Python 绑定面属传递覆盖(前者复用同批内核,后者形状校验落
  C++ shape_infer),无独立设计工作量。
- 实施前 v0「无广播」约束条文全部继续有效;**禁止任何局部先行实现**
  (零散支持会使上表各面出现不一致的中间态)。
- 判定方法(行为判定,规避文本误伤——注释与「v0 has no broadcasting」类
  错误消息串不算实现):实施里程碑立项前,elementwise 家族「shape 不一致
  被拒」的既有用例(tests/cpp/ops/ 各算子报错路径)保持通过,即证无局部
  先行实现;立项后按该里程碑设计文档验收。

## 备选方案

- 永不支持广播:约束最简;但 bias/标量系数两类高频形态永久走全展开
  workaround,内存与表达力双输——否决。
- 与动态 shape 一并裁决:一次到位;但动态 shape 牵动 ARCH-013/044 与
  全部后端,量级远大于静态广播,捆绑徒增延迟——否决,保持独立议题。

## 后果

- 正面:方向锁定,bias/标量系数的最终形态可预期;实施范围清单先行
  固化,避免届时漏面。
- 负面与代价:实施是横切变更(shape 推断/内核/融合/梯度四面联动),
  须整里程碑投入;在此之前 workaround 继续存在。
- 跟进:实施里程碑立项时以本 ADR 工程影响清单为设计输入;
  docs/architecture/autograd.md 第 8 章「广播语义」条目改指本 ADR。
