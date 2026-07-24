# 两级复用规范与第三方库准入清单

> **强制等级**:规范(MUST)
> **相关铁律**:#5 两级复用 / #3 后端矩阵
> **面向读者**:实现代码的工程师
> **最后更新**:2026-07-24(GPLv3+ 开源后保持第三方依赖准入边界)

两级复用的定义(铁律 #5):
**第一级**——第三方成熟库能用就用(准入见第 2 章);
**第二级**——项目内先搜索再动手(流程见第 1 章)。
同时,复用不得堵死自定义 pass 与自定义算子的扩展接口(第 5 章)。

规则条文格式:`【编号】【MUST/MUST NOT/SHOULD】正文。判定方法:<可机械执行的检查>。`

---

## 1. 动手前搜索流程(强制五步)

- 【REUSE-001】【MUST】编写任何新组件(函数/类/kernel/pass/工具脚本)之前,依次执行
  以下五步并留痕:
  1. 项目内符号与关键词搜索(命令模板):
     `grep -rn "<关键词>" src/ include/ python/ --include='*.h' --include='*.cpp' --include='*.py'`
     以及 `grep -rn "<拟用符号名>" src/ include/`;
  2. 查 `docs/architecture/` 相关文档,确认该能力归属哪一层、是否已有接口;
  3. 查本文档第 2 章准入表,确认是否有第三方库覆盖该能力;
  4. 查 `docs/decisions/`,确认是否已有相关 ADR;
  5. 以上皆无 → 将搜索命令与结果摘要粘贴进 PR「复用检查」段(LANG-011),之后方可
     动手。

  判定方法:PR 模板「复用检查」为必填段;缺失或空段即打回。

- 【REUSE-002】【MUST NOT】提交与仓库既有代码重复的实现。重复的机械判定 = 满足其一:
  (a) 与既有代码存在**同名或同签名**的符号;(b) 与既有代码存在 **≥20 行连续相同**
  的代码。其余疑似重复(语义等价、仅参数化差异等)归 code-reviewer 人工检查项。
  判定方法:code-reviewer 按 (a) 执行符号名 grep、按 (b) 执行 diff 比对;命中即打回。

- 发现两处以上需要同一工具函数时,沉淀到所属层的公共位置(如
  `include/frame/core/`),并在 PR 中说明;禁止复制粘贴绕过 REUSE-002。

## 2. 第三方库准入表(三档:已批准 / 受限批准 / 禁止)

档位定义:
- **已批准**:选型已预先批准,按第 4 章方式引入;首次接入仍受 REUSE-010 约束(该 ADR
  只需记录版本锁定与引入方式,不再论证选型)。
- **受限批准**:使用附带前提条件——或须先有专门的已接受 ADR,或限定目录/开关范围;
  条件逐行列于表内。
- **禁止**:不得引入。

| 库 | 用途 | 档位 | 条件 / 引入方式 |
|---|---|---|---|
| cuBLAS / cuBLASLt | GEMM | 已批准 | CUDAToolkit 组件,`find_package(CUDAToolkit)` |
| cuDNN | 卷积/归一化 | 已批准 | 仅 CUDA 后端 |
| CUTLASS | 需融合的 GEMM 模板 | 已批准 | 仅 CUDA 后端 |
| CUB / Thrust | reduction/scan | 已批准 | CUDAToolkit 自带 |
| oneDNN | conv/matmul/norm 等 primitive | 已批准 | 仅 Intel GPU 后端 |
| oneMKL | BLAS | 已批准 | 仅 Intel GPU 后端 |
| OpenVINO Runtime | Intel NPU 接入 | 已批准 | `find_package(OpenVINO)`;Intel NPU **一律经此**(BE-INPU-001) |
| CANN / AscendCL | 昇腾接入 | 已批准 | `cmake/find_cann.cmake` |
| pybind11 | Python 绑定 | 已批准 | `FetchContent` 锁定版本 tag |
| GoogleTest / GoogleMock | C++ 测试 | 已批准 | `FetchContent` 锁定版本 tag |
| fmt | 字符串格式化 | 已批准 | `FetchContent` 锁定版本 tag |
| spdlog | 日志 | 已批准 | `FetchContent` 锁定版本 tag |
| Google Benchmark | 微基准计时/统计/报告 | 已批准 | FetchContent 锁 tag,受 FRAME_BUILD_BENCHMARKS 门控(ADR-0014) |
| nlohmann/json | JSON | 已批准 | **仅限工具与测试代码**,不得进核心运行时路径 |
| MLIR | 编译器基础设施(预研赛道) | **受限批准** | 依据 ADR-0002(IR 双轨并行,已接受):仅允许出现在未来 `src/compiler/mlir/`,且必须受 `FRAME_ENABLE_MLIR`(三态,默认 OFF)开关隔离;**当前骨架不引入**该目录与依赖 |
| Level Zero(直连) | Intel 设备底层驱动 | 受限批准 | 直连路线已废弃;仅当 OpenVINO/SYCL 能力不足时,经新的已接受 ADR 引入 |
| abseil | 基础库 | 受限批准 | 须先有已接受 ADR |
| Boost(任意子库) | — | 受限批准 | 须先有已接受 ADR |
| TBB | 宿主并行运行时 | 受限批准 | 须先有已接受 ADR |
| NCCL / oneCCL / HCCL | 分布式通信 | 受限批准 | 分布式暂缓;须先有已接受 ADR |
| GPL/AGPL 等传染性协议库 | — | **禁止** | 见第 3 章 |
| 与已批准库功能重复的同类库 | 例:boost::format(与 fmt 重复) | **禁止** | 先经 ADR 淘汰旧库方可替换 |

- 【REUSE-010】【MUST】新增第三方依赖必须先有**已接受**的 ADR(`docs/decisions/NNNN-*.md`
  且状态为「已接受」)。判定方法:PR diff 中 CMake 出现新增
  `find_package`/`FetchContent`/指向 `third_party/` 的 `add_subdirectory`,而
  `docs/decisions/` 无对应已接受 ADR,即打回;提交前自检与 code-reviewer
  均执行此检查。不追溯适用的豁免清单(仅限骨架期已接线者):后端 SDK(CUDAToolkit、
  IntelSYCL、OpenVINO、CANN)与已以 `FetchContent` 锁定版本 tag 接线的 GoogleTest、
  pybind11(见 `cmake/frame_dependencies.cmake`);其后续版本变更或引入方式变更仍须
  ADR。
- 【REUSE-011】【MUST】CPU 参考 kernel 边界(design-reviewer 裁决,2026-07-11):
  cpu 参考后端的算子 kernel(ARCH-041)以数值正确性为唯一目标,**不受**本表对标准
  能力(GEMM/卷积/归约等)的引库强制约束——朴素实现(含手写循环)合规,但必须在
  kernel 注释标注「参考实现,数值校验用,禁作性能路径」;任何性能路径**不得**复用
  参考实现,加速后端的同能力实现仍须按本表引库;CPU 侧引入 BLAS 类库仍触发
  REUSE-010(需 ADR)。判定方法:code-reviewer 对 src/backends/cpu/kernels/ 缺标注、
  或非 cpu 参考路径手写标准能力的 diff 按本条与铁律 5 打回。

## 3. 许可证白名单

- 【REUSE-014】【MUST】第三方库许可证准入:Apache-2.0 / MIT / BSD-2-Clause /
  BSD-3-Clause 直接可用;LGPL 须 ADR;GPL / AGPL 及其他传染性协议禁止。引入依赖的
  ADR 必须写明许可证。判定方法:code-reviewer 核对 ADR 中的许可证字段与上游仓库
  LICENSE 文件一致。
  (编号勘误 2026-07-13:本条原误编 REUSE-011,与「CPU 参考 kernel 边界」条重号;
  全部按号外部引用均指后者,故后者保留 011、本条改号 014,无引用破坏。)

说明:项目自身已按 `GPL-3.0-or-later` 开源(ADR-0023)，但本条只约束第三方依赖
准入；项目许可证的变化不放宽 LGPL/GPL/AGPL 依赖门槛，REUSE-014 继续执行。

## 4. 引入方式优先级

- 【REUSE-012】【MUST】按以下优先级选择引入方式,且只能在前一级不可行时降级:
  1. 系统/SDK 自带 `find_package`(设备 SDK 一律此法;昇腾用自写
     `cmake/find_cann.cmake`);
  2. CMake `FetchContent` 并锁定版本 tag(禁止跟踪分支/HEAD);
  3. git submodule(仅当前两者不可行,须在引入 ADR 中说明原因)。

  禁止把第三方源码直接拷入仓库(vendoring 须专门 ADR;`third_party/` 默认为空,
  见 third_party/README.md)。判定方法:code review 对照 PR 中的实际引入方式与
  ADR 记载。

- 【REUSE-013】【MUST】版本选型优先**最新稳定版**(项目所有者裁决,2026-07-13):
  选型新依赖或升级既有依赖锁定版本时,存在且可行的前提下必须采用决策时点的
  最新稳定发行版(stable release),而非沿用旧版或机械取「上一个」版本;仅当
  最新稳定版存在可证的兼容性/支持矩阵障碍(如 SDK 配套、平台/编译器支持、
  上游已知回归)时才可退档,退档理由须写入引入/升级说明(ADR 或 PR 描述)。
  预发布版本(alpha/beta/rc)不属「稳定版」,不受本条驱动。本条只约束
  **选哪个版本**,不改变既有变更流程(锁定 tag 纪律见 REUSE-012;版本升级
  须 ADR 见 REUSE-010 末句;排期镜像见 build-order.md 第 4 节)。本条**不
  追溯既有锁定**:存量版本在下一次升级触点按本条 + REUSE-010 一并重评。
  判定方法:引入/升级说明必须记载**核实来源(上游 Releases/Tags 页面)与
  核实日期**;code-reviewer 仅对照该记载判定「所锁版本是否为核实时点最新
  稳定版或附退档理由」,不以 review 当下的活跃版本为基准;缺记载或缺理由
  即打回。

## 5. 扩展点保障(铁律 #5 后半)

- 【REUSE-020】【MUST】复用第三方库时,封装层必须保留自定义扩展路径:
  1. 算子层封装不得堵死经 `FRAME_REGISTER_KERNEL` 注册自定义实现的能力;
  2. 编译层封装不得堵死自定义 pass(`FRAME_REGISTER_PASS`)插入管线的能力;
  3. 整图交厂商编译器的后端(如 Intel NPU)对不支持的子图必须返回带算子名的错误交
     上层回退(ARCH-031),不得以「厂商编译器不支持」为由拒绝整个扩展机制。

  判定方法:`tests/cpp/hal_conformance/` 一致性套件包含「注册自定义算子 kernel 并被
  分发」「注册自定义 pass 并在管线中生效」两个用例;用例失败或被删除即违规。
