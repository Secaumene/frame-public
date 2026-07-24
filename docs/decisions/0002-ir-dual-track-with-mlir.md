# ADR-0002:IR 双轨并行(自研主线 + MLIR 预研)

- 状态:已接受
- 日期:2026-07-04
- 关联铁律:#1、#5
- 关联规则:REUSE-010、ARCH-020

## 背景

图 IR 是编译器层的核心数据结构。自研轻量 IR 起步快、完全可控,但 lowering
基础设施需要自建;MLIR 提供成熟的 pass/方言基建与编译器生态,但学习与构建
成本高。四后端接入初期需要尽快跑通主线,同时不希望堵死 MLIR 路线造成返工。

## 决策

采用双轨并行,v0 主线为自研轻量 IR:

- 主线:自研 in-memory 图 IR(Graph/Node/Value,规范见
  docs/architecture/ir-design.md),v0 全部 pass 与 lowering 基于它实现;
- 预研赛道:MLIR 作为并行预研,保留 CMake 三态开关 `FRAME_ENABLE_MLIR`
  (AUTO|ON|OFF,默认 OFF)与 compiler 层 backend_lowering 扩展点;
- 隔离约束:MLIR 依赖只允许受 `FRAME_ENABLE_MLIR` 隔离地出现在未来
  `src/compiler/mlir/` 目录(本期骨架不建立该目录);默认 OFF 构建不得链接
  任何 MLIR/LLVM 库。

判定方法:`grep -rn -i "mlir" CMakeLists.txt cmake/ src/` 的命中处必须位于受
`FRAME_ENABLE_MLIR` 保护的 CMake 分支或 `src/compiler/mlir/` 目录内,否则打回。

## 备选方案

- 纯自研、不留 MLIR 通道:优点是零重量级依赖、构建最轻;缺点是 lowering 基建
  长期重复造轮子,与编译器生态隔绝。否决:保留受开关隔离的接入通道成本极低。
- 直接以 MLIR 为唯一 IR:优点是复用成熟 pass 基建与方言生态;缺点是 v0 阶段
  学习与构建成本高,构建时间与二进制体积大幅上升,拖慢四后端接入。
  否决:v0 不采用;列为预研赛道而非弃案。

## 后果

- 正面:主线轻装快跑;MLIR 若判定为优,可经扩展点切入而不推翻主线代码。
- 负面:双轨并存期间 lowering 语义需两份维护;预研赛道需要持续跟进投入。
- 跟进:TODO(FRAME-DESIGN): 设计 src/compiler/mlir/ 目录骨架与 lowering
  扩展点接口。参考:docs/architecture/compiler-passes.md。完成判据:
  FRAME_ENABLE_MLIR=ON 可编译出含 MLIR lowering 桩的目标,且默认 OFF 构建不受影响。
