# ADR-0014:基准计时设施采用 Google Benchmark

- 状态:已接受
- 日期:2026-07-13
- 关联铁律:#5 两级复用
- 关联规则:REUSE-010、REUSE-012、REUSE-013、BUILD-001

## 背景

开放问题 Q4(性能基准体系)随 v1.1 收口立项:需要微基准计时设施与
benchmarks 规范。计时器/统计/重复次数控制/对比报告属标准能力,铁律 #5
第一级要求优先成熟库;Google Benchmark 未列于复用准入表,首次引入须
本 ADR(REUSE-010)。

## 决策

- 采用 **Google Benchmark** 作为唯一微基准计时设施,禁止自研计时
  harness(重复造轮子)与散落的手写计时代码。
- 版本(REUSE-013 记载):锁定 **v1.9.5**——核实来源
  github.com/google/benchmark/releases,核实日期 2026-07-13,为当时
  最新稳定发行版;许可证 Apache-2.0(仓库 LICENSE,REUSE 许可证白名单
  内)。引入方式:CMake `FetchContent` 锁定版本 tag(REUSE-012 第 2
  档;关闭其自带测试构建)。
- 门控:`FRAME_BUILD_BENCHMARKS`(默认 OFF),不进 cpu-only 门禁口径
  与 ctest;基准以 `release` preset 编译运行(Debug 数字无意义)。
  规范细则见 docs/standards/benchmarks.md(BENCH- 系列,与本 ADR 同批
  交付)。
- 判定方法:`cmake/frame_dependencies.cmake` 存在受
  `FRAME_BUILD_BENCHMARKS` 门控的 FetchContent 且 GIT_TAG 为 v1.9.5;
  `grep -rn "std::chrono" benchmarks/` 无手写计时;cpu-only preset 不
  拉取该依赖(默认 OFF 时 FetchContent 不执行)。

## 备选方案

- 自研微 harness(计时 + 重复 + 统计):实现量看似小,但预热/时钟
  精度/统计稳健性/对比报告全是踩坑面,恰为该库覆盖的标准能力——
  否决(铁律 #5 第一级)。
- nanobench 等替代库:同类能力;Google Benchmark 生态与文档占优,且
  与 GoogleTest 同源风格一致——否决备选,不再并列引入。

## 后果

- 正面:计时/统计/报告零自研;基准数字口径统一。
- 负面与代价:新增一个 FetchContent 依赖(仅 benchmarks 构建时拉取,
  默认 OFF 零成本);版本升级按 REUSE-010/013 流程。
- 跟进:复用准入表(reuse-policy.md 第 2 章)加行;build-order.md
  第 4 节依赖表加行;benchmarks 规范与首个基准同批落地。
