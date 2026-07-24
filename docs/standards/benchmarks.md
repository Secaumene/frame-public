# 性能基准规范(BENCH-)

> **强制等级**:规范(MUST)
> **相关铁律**:#1 编译优先 / #5 两级复用
> **面向读者**:实现代码的工程师
> **最后更新**:2026-07-13(首版,Q4 立项;计时设施依据 ADR-0014)

本文档约束 `benchmarks/` 目录下的全部微基准代码与其运行口径。
规则条文格式与全仓一致:`【编号】【MUST/MUST NOT/SHOULD】正文。判定方法:<可机械执行的检查>。`

## 1. 目录与构建

- 【BENCH-001】【MUST】基准代码一律位于仓库根 `benchmarks/`(文件名
  `bench_<主题>.cpp`),经 `FRAME_BUILD_BENCHMARKS`(默认 OFF)门控构建,
  目标命名 `frame_bench_<主题>`;不注册进 ctest,不参与 cpu-only 门禁。
  判定方法:`tests/` 与 `src/` 下 `grep -rn "benchmark::"` 零命中;
  cpu-only preset 配置不拉取 Google Benchmark。
- 【BENCH-002】【MUST】计时/统计/重复控制一律经 Google Benchmark
  (ADR-0014),禁止手写 `std::chrono` 计时循环。判定方法:
  `grep -rn "std::chrono" benchmarks/` 零命中。

## 2. 运行口径

- 【BENCH-010】【MUST】基准数字仅在 **`bench` preset**(继承 release +
  `FRAME_BUILD_BENCHMARKS=ON`,BUILD-001 preset 纪律)下产生与引用;
  Debug/dev 构建的数字不得写入任何文档或提交说明。引用基准数字处必须
  带固定前缀行 `基准口径: preset=bench; 机器=<概况>`。判定方法:
  文档/提交中出现基准数字而无该前缀行即打回(前缀行可 grep 定位)。
- 【BENCH-011】【MUST】基准主路径必须走编译执行(`runtime::compile` +
  `Executable::run`,ARCH-010);eager 路径只允许作为**对照组**出现且
  须显式命名(如 `BM_matmul_eager_reference`)。判定方法:code-reviewer
  对 benchmarks/ 新增内容核对主路径调用链。
- 【BENCH-012】【MUST NOT】基准不做性能数值断言(不设「必须快于 X」的
  门槛)——数字随机器漂移,断言会把环境差异变成假失败;回归判断由
  人工对比报告完成。判定方法:`grep -rn "ASSERT\|EXPECT" benchmarks/`
  零命中。
- 【BENCH-013】【MUST NOT】代码与文档不得出现无数据支撑的性能声明
  (「更快/高效/性能提升」类措辞而无 BENCH-010 口径数字)——本条为该
  纪律的首次成文来源。判定方法:code-reviewer 对 diff 中新增的性能
  形容词核对其是否伴随「基准口径:」前缀行的数据;无数据即打回。

## 3. 报告与登记

- 【BENCH-020】【SHOULD】每个基准文件头注释登记:测量对象、shape/dtype
  档位、对照组含义;运行产出建议以 `--benchmark_format=json` 存档对比。
  偏离时在提交说明写明。

首个基准:`benchmarks/bench_matmul.cpp`(matmul+add+relu 编译执行,
cpu 后端,多档 shape;cuda 后端可用时同源对比),与本规范同批交付。
