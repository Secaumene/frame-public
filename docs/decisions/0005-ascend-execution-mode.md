# ADR-0005:昇腾后端执行模式(aclnn 逐算子拼装 vs GE 整图编译)

- 状态:推迟(重评条件:v2.0 里程碑启动昇腾后端实现时)
- 日期:2026-07-04
- 关联铁律:#3
- 关联规则:BE-ASC-001

## 背景

昇腾 CANN 有两条接入路线:aclnn 系列单算子 API 逐算子拼装,或 GE(Graph
Engine)/ATC 整图编译。两者 API 形态、工程结构与调试方式差异大;当前既无设备
基准数据,也无实现排期压力,过早绑定任一路线会造成返工。

## 决策

推迟至 v2.0 裁决。推迟期间的约束:

- 代码骨架仅建立 `src/backends/ascend/` 目录与 HAL 接口桩(函数体一律
  `return FRAME_UNIMPLEMENTED();`),不含任何绑定单一路线的实现;
- docs/backends/ascend.md 将两条路线并列陈述,均标注为候选,不做承诺;
- HAL 桩接口按两案交集设计(Backend/Stream/Event/Allocator/Executable
  为两案共需),保证裁决后切换不动核心层。

判定方法:本 ADR 状态为「推迟」期间,`src/backends/ascend/` 出现 aclnn 或 GE
的实现代码即打回;docs/backends/ascend.md 出现「已选定路线」类结论性措辞即打回。

## 备选方案(两案并列,均未否决)

- aclnn 逐算子拼装:优点是与自研图编译器和 kernel 注册体系同构,粒度可控、
  问题定位直观;缺点是逐算子下发有调度开销,图级优化全部落回自研编译器层。
  【待查证】aclnn 两段式调用约定(aclnnXxxGetWorkspaceSize / aclnnXxx)细节
  —— 来源:CANN aclnn API 参考(hiascend.com 文档中心)。
- GE 整图编译:优点是整图交厂商编译器,性能上限与 NPU 特性利用更好;缺点是
  需要完整的 IR 转换层,编译过程黑盒、调试与归因困难。
  【待查证】GE/ATC 图接入 API 形态与版本配套 —— 来源:昇腾社区 CANN
  Graph Engine 开发指南。

## 后果

- 正面:避免在信息不足时锁定路线;HAL 交集设计使两案切换均不影响核心层。
- 负面:昇腾后端可用时间推后;两案文档需并行维护至裁决。
- 跟进:v2.0 重评前收集两路线的算子覆盖与性能基准数据;重评时更新本 ADR 状态
  与决策内容(或以新 ADR 取代),并同步 README.md 索引表(ADR-011)。
