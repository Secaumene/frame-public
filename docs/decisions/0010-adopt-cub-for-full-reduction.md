# ADR-0010:cuda 全归约采用 CUB DeviceReduce

- 状态:已接受
- 日期:2026-07-13
- 关联铁律:#5 两级复用
- 关联规则:REUSE-010、BE-CUDA-001、BE-CUDA-002、BUILD-011

## 背景

cuda 后端 sum 内核(src/backends/cuda/kernels/reduction.cu)v0 为朴素
并行(每输出元素一线程、沿归约轴串行),全归约场景(单输出元素累加
全部输入)退化为单线程串行,是已登记的性能悬崖(TODO(FRAME-PERF))。
CUB 属 CUDA Toolkit 自带 CCCL 组件,在 REUSE-010 的 SDK 豁免清单内
——本 ADR 为**决策留档**而非豁免审批。

## 决策

- sum 的**全归约路径**(v0 语义:axes 为空 = 全维归约,输出标量)改用
  `cub::DeviceReduce::Sum` 两段式调用(先查询临时存储字节数,再执行;
  临时存储经后端 Allocator 分配);fp16/bf16 按 cpu 参考同语义升 float
  累加后转回(累加类型口径与 BUILD-011 容差表一致)。【待查证】
  cub::DeviceReduce::Sum 直传 half*/bf16* 时累加类型随输入,须经
  transform/cast 迭代器升 float 才与 cpu 口径一致 —— 来源:CCCL
  DeviceReduce 官方文档;实现时核实并按此落地。
- **任意轴投影归约维持自研朴素内核**:CUB 无「任意轴组合投影」原语
  (DeviceSegmentedReduce 要求连续段),属 BE-CUDA-002「库覆盖不到」
  的合法自研准入,该论证保留在内核注释。
- 实施随积压批次落地,销 reduction.cu 的 TODO(FRAME-PERF)。
- 判定方法:reduction.cu 全归约分支调用 `cub::DeviceReduce` 且
  tests/cpp/backends/ 既有 sum 数值用例(含大规模放宽一档用例)保持
  通过;`cmake` 无新增 find_package/FetchContent(CCCL 头随 Toolkit)。

## 备选方案

- 维持朴素全归约:零改动;但 2^20 级累加单线程串行,与「发现性能
  悬崖」的回退统计初衷相悖——否决。
- 自研树形归约内核:可行;但与 CUB 成熟实现重叠,违反铁律 #5 第一级
  「标准能力优先用成熟库」与 BE-CUDA-002——否决。

## 后果

- 正面:全归约获得厂商优化实现;CCCL 头随 Toolkit,零依赖成本。
- 负面与代价:reduction.cu 出现两条代码路径(全归约/轴投影),路径
  选择逻辑须测试覆盖两侧。
- 跟进:实施时同步核对 cuda.md 第 3 章 CUB 行表述与本 ADR 一致。
