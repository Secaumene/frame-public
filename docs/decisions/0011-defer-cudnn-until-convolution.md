# ADR-0011:cuDNN 推迟至卷积类算子立项

- 状态:被 ADR-0021 取代(2026-07-18;M21 卷积批次触发重评条件)
- 日期:2026-07-13
- 关联铁律:#5 两级复用
- 关联规则:REUSE-010、BE-CUDA-001

## 背景

cuDNN 在复用准入表中为已批准档,但「已批准」仅指选型方向;首次引入
仍须已接受的 REUSE-010 ADR(build-order.md 第 4 节明示)。现算子集
(elementwise/matmul/reduction/loss 及各梯度)中,matmul 已由 cuBLAS
覆盖,其余均无 cuDNN 适用面——引入将是零消费依赖。

## 决策

- 推迟引入 cuDNN。推迟期间:禁止在 cmake 引入 cuDNN 探测/链接,禁止
  实现绑定 cuDNN API 的代码;cuda.md 第 3 章复用优先级表中 cuDNN 行
  维持「已批准、未引入」表述并引用本 ADR。
- 重评条件展开(可判定):v1 算子扩容裁决(PLAN 开放问题 Q1 的卷积类
  遗留议题)通过且清单含 conv/pool 类算子,或出现 cuDNN 独占能力的明确
  需求;重评时按 REUSE-010 全流程
  (版本选型、查找方式、与 cuBLAS 职责边界)更新本 ADR 状态。
- 判定方法:`grep -rin "cudnn" cmake/ src/ CMakeLists.txt` 仅命中文档性
  注释;docs/decisions/ 索引表本条状态为「推迟」。

## 备选方案

- 现在引入并封装占位:提前踩平集成;但零消费依赖违反 REUSE-010 的
  「先有需求后有依赖」次序,且版本配套(cuDNN 对 CUDA 13.x/Blackwell
  的支持矩阵)届时可能已变,现在锁版本反成负担——否决。

## 后果

- 正面:依赖面保持最小;卷积立项时以最新支持矩阵做选型。
- 负面与代价:卷积类算子立项时多一道 ADR 重评前置(约束本就该有)。
- 跟进:v1 算子扩容议题裁决时联动重评本 ADR。
