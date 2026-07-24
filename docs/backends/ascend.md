# 昇腾(Ascend)后端指南

> **强制等级**:规范(MUST)
> **相关铁律**:#1 编译优先 / #3 后端矩阵 / #5 两级复用
> **面向读者**:实现代码的工程师
> **最后更新**:2026-07-23(M25/M27/M28 算子待实现项回写)

本文档按 `backends/README.md`(BE-001)定义的九章模板撰写。注册名字符串为 `"ascend"`,后端目录 `src/backends/ascend/`。所有不确定信息按 BE-000 用【待查证】标注并在第 8 章集中登记。

---

## 1. 定位与适用范围

- 目标硬件:昇腾 NPU(经 CANN 软件栈)。注册名 `"ascend"`,`FRAME_REGISTER_BACKEND("ascend", AscendBackend)`。
- 执行模式:**未定,ADR-0005 已裁决推迟至 v2.0**。两条路线均列为候选、本文档不做承诺:
  - 候选 A —— aclnn 逐算子:经 `aclnnXxx` 系列 API 逐算子调用拼装。
  - 候选 B —— GE 整图:经 Graph Engine(GE)/ ATC 做整图编译。
  - 【待查证】两路线在性能、算子覆盖、工程复杂度上的取舍数据 —— 来源:昇腾社区 CANN 文档中心(aclnn 与 GE/ATC 章节)。
- 骨架现状:本轮**仅建 `src/backends/ascend/` 目录 + HAL 桩**,不实现任何执行路线;执行模式落地留待 ADR-0005 于 v2.0 裁决后进行。
- 不适用场景:非昇腾硬件;在执行模式未定前对本后端做性能承诺或写入任一路线的生产实现。

---

## 2. 环境要求

- 软件栈:CANN toolkit(含 AscendCL / aclrt 运行时)。
  - 【待查证】CANN toolkit 版本与操作系统、NPU 驱动、固件的配套关系及最低版本 —— 来源:昇腾社区 hiascend.com 文档中心(CANN 安装指南)。
- 宿主编译器:CANN 对宿主 C++ 编译器有版本要求;本项目 C++ 标准为 C++20(ADR-0004)。
  - 【待查证】CANN 支持的宿主编译器及其对 C++20 的兼容性(GCC/Clang 最低版本)—— 来源:昇腾社区 hiascend.com 文档中心(CANN 安装 / 宿主编译器要求)。
- 环境变量:`ASCEND_HOME_PATH`(CANN 安装根,供 CMake find 模块读取)。
- 环境脚本:CANN 的 `set_env.sh`(设置库路径、`ASCEND_HOME_PATH` 等)。
- 设备检测命令(可直接执行):

  ```bash
  source ${ASCEND_HOME_PATH}/set_env.sh   # 路径以本机 CANN 安装为准
  npu-smi info                            # 列出昇腾设备、健康状态与显存
  ```

---

## 3. 复用库清单与复用边界

复用取向:**优先 CANN 内置算子库**;自定义算子用 Ascend C 开发;禁止自研与 CANN 算子库重叠的实现。

| 栏位 | 库 / 路径 | 覆盖范围 |
|---|---|---|
| 必用(MUST) | CANN aclnn 系列(`aclnnXxx`) | CANN 覆盖的标准算子(单算子调用) |
| 必用(MUST) | AscendCL(aclrt 系) | 设备/流/事件/内存等运行时管理 |
| 可用(可选) | Ascend C | 库未覆盖算子的自定义开发 |
| 禁止自研(MUST NOT) | —— | 与 CANN 算子库功能重叠的自研实现 |

- 【BE-ASC-001】【MUST】优先使用 CANN 内置算子库:标准算子经 aclnn 系列 API 调用;库未覆盖的算子才用 Ascend C 自定义开发;禁止自研与 CANN 算子库重叠的实现。判定方法:review 检查 `src/backends/ascend/` 是否存在与 CANN 算子库重叠的自研 kernel,无「库不满足」依据即打回(准入同 CUDA 的 BE-CUDA-002)。
  - 【待查证】`aclnnXxx` 两段式调用约定(先算 workspace,再执行)的具体签名 —— 来源:CANN aclnn API 参考。
  - 【待查证】Ascend C 自定义算子工程脚手架与编译流程 —— 来源:CANN Ascend C 算子开发指南。
- 【BE-ASC-002】【MUST】执行模式未定前,不得在代码中固化 aclnn 或 GE 任一路线为唯一实现;两路线的选型须由 ADR-0005(v2.0)裁决。判定方法:骨架阶段 `src/backends/ascend/` 只含 HAL 桩,不含任一路线的执行实现;新增执行实现的 PR 必须引用已接受的 ADR-0005。
- 引入方式:CANN 走自写 `cmake/find_cann.cmake`(读 `ASCEND_HOME_PATH`);新增其他第三方依赖须先有 ADR(REUSE-010)。

---

## 4. HAL 映射表

HAL 映射以 AscendCL(aclrt 系)运行时 API 为准,与执行模式(aclnn / GE)无关,骨架阶段即可定形为桩。`Device` 为值类型 `{backend, index}`(BE-002)。错误一律转为 `Result<T>` / `Status`(`include/frame/core/status.h`),不跨 HAL 边界抛异常(CPP-020)。

| HAL 接口 / 操作 | CANN(AscendCL / aclrt)原生 API |
|---|---|
| 运行时初始化 / 析构 | `aclInit` / `aclFinalize`(全局单次,统一管理生命周期) |
| Backend 设备枚举 / 选择 | `aclrtGetDeviceCount` / `aclrtSetDevice` |
| Backend 设备属性 | 【待查证】设备属性查询 API(名称/显存)—— 来源:CANN AscendCL API 参考 |
| Stream | `aclrtStream`(`aclrtCreateStream` / `aclrtDestroyStream`) |
| Stream::synchronize | `aclrtSynchronizeStream` |
| Event | `aclrtEvent`(`aclrtCreateEvent` / `aclrtRecordEvent`) |
| Event::synchronize / query | `aclrtSynchronizeEvent` / `aclrtQueryEventStatus` |
| Stream 等待 Event | `aclrtStreamWaitEvent` |
| Allocator::allocate / deallocate | `aclrtMalloc` / `aclrtFree` |
| memcpy(H2D/D2H/D2D) | `aclrtMemcpyAsync` |
| 错误转换 | `ACL_CHECK` 宏:`aclError` → `Status`(消息一律英文,LANG-005) |

- `aclInit`/`aclFinalize` 与多设备/多流的初始化-析构次序须统一在后端入口集中管理,避免各处重复 init。

---

## 5. 编译接入

- CMake 选项:`FRAME_ENABLE_ASCEND`,三态 `AUTO|ON|OFF`,默认 `AUTO`(语义见 BUILD 系列)。
- SDK 探测:自写 `cmake/find_cann.cmake`,读取 `ASCEND_HOME_PATH` 定位 CANN 头文件与库(昇腾无标准 `find_package` 配置包,故自写 find 模块)。
  - 【待查证】CANN 头文件与库在 `ASCEND_HOME_PATH` 下的相对路径布局 —— 来源:昇腾社区 CANN 安装指南目录结构章节。
- 链接目标:后端库目标 `frame_backend_ascend`,链接 AscendCL(aclrt / acl)相关库。
  - 【待查证】需链接的 CANN 库名清单(`libascendcl`、`libacl_op_compiler` 等)—— 来源:CANN AscendCL 链接说明。
- 语言/编译器:本后端主机端代码按 C++20 编译;宿主编译器对 C++20 的兼容性见第 2 章【待查证】。
- Preset:`ascend`。骨架验收口径为 `cpu-only`,不依赖 CANN。
- 后端代码用 CMake 目标隔离,`src/{core,ir,compiler,runtime}` 中禁止出现 `#ifdef FRAME_ENABLE_ASCEND`(BUILD-003)。

---

## 6. 算子/内核开发规范

- 骨架阶段:不开发任何算子实现;执行模式未定(BE-ASC-002),仅保留 HAL 桩。相关实现以 `TODO(FRAME-IMPL)` 标注,并在标注中引用 ADR-0005。
- 未来(v2.0,ADR-0005 裁决后)按选定路线开展:
  - 若选 aclnn 逐算子:注册 `FRAME_REGISTER_KERNEL(op, "ascend", fn)`,kernel 内经 aclnn 两段式调用;dtype 差异经 `dispatch_dtype` 编译期展开,禁止 kernel 内运行时 dtype/设备分支(ARCH-042)。
  - 若选 GE 整图:实现自研 IR → GE 图 → ATC/GE 编译,不逐算子注册 kernel(形态类似 Intel NPU 的整图路线)。
- 目录:aclnn/Ascend C kernel(若采用)放 `src/backends/ascend/kernels/`;GE 图接入(若采用)放 `src/backends/ascend/`。
- 命名:`snake_case`;优先 CANN 算子库(BE-ASC-001)。

### 已支持算子表

执行模式未定(ADR-0005,推迟至 v2.0),骨架期本表保持为空;路线裁决后新增算子维护本表,未实现项按 `TODO(FRAME-IMPL)` 标注,静默缺失即违例。
`fused_elementwise_internal`(M9 融合 pass 产物,非用户构图算子,不入本表;cpu 组合调用参考语义见 docs/architecture/compiler-passes.md §3.7)。

| 算子 | 状态 | 实现方式/备注 |
|---|---|---|
| add | `TODO(FRAME-IMPL)` | cpu 参考实现已注册(`src/backends/cpu/kernels/elementwise.cpp`);ascend 执行路线未定(ADR-0005,推迟至 v2.0)。参考:`docs/decisions/0005-ascend-execution-mode.md`。完成判据:ADR-0005 裁决执行路线后按所选路线接入 add 算子实现。 |
| mul | `TODO(FRAME-IMPL)` | cpu 参考实现已注册(`src/backends/cpu/kernels/elementwise.cpp`);ascend 执行路线未定(ADR-0005,推迟至 v2.0)。参考:`docs/decisions/0005-ascend-execution-mode.md`。完成判据:ADR-0005 裁决执行路线后按所选路线接入 mul 算子实现。 |
| relu | `TODO(FRAME-IMPL)` | cpu 参考实现已注册(`src/backends/cpu/kernels/elementwise.cpp`);ascend 执行路线未定(ADR-0005,推迟至 v2.0)。参考:`docs/decisions/0005-ascend-execution-mode.md`。完成判据:ADR-0005 裁决执行路线后按所选路线接入 relu 算子实现。 |
| sum | `TODO(FRAME-IMPL)` | cpu 参考实现已注册(`src/backends/cpu/kernels/reduction.cpp`);ascend 执行路线未定(ADR-0005,推迟至 v2.0)。参考:`docs/decisions/0005-ascend-execution-mode.md`。完成判据:ADR-0005 裁决执行路线后按所选路线接入 sum 算子实现。 |
| matmul | `TODO(FRAME-IMPL)` | cpu 参考实现已注册(`src/backends/cpu/kernels/matmul.cpp`);ascend 执行路线未定(ADR-0005,推迟至 v2.0)。参考:`docs/decisions/0005-ascend-execution-mode.md`。完成判据:ADR-0005 裁决执行路线后按所选路线接入 matmul 算子实现。 |
| square | `TODO(FRAME-IMPL)` | cpu 参考实现已注册(`src/backends/cpu/kernels/elementwise.cpp`);另有 decomposition(square(x)=mul(x,x),见 `src/ops/schemas/elementwise.cpp::square_decompose`,M10 回退链素材);ascend 执行路线未定(ADR-0005,推迟至 v2.0)。参考:`docs/decisions/0005-ascend-execution-mode.md`。完成判据:ADR-0005 裁决执行路线后按所选路线接入 square 算子实现。 |
| constant | 已支持(cpu) | cpu 参考实现已注册(`src/backends/cpu/kernels/constant.cpp`,M8);0 输入 1 输出,attrs=value/shape/dtype,常量物化。参考:`docs/architecture/operator-system.md` 第4章。ascend 执行路线未定(ADR-0005,推迟至 v2.0)。 |
| selective_scan | TODO(FRAME-IMPL): 说明:接入 ascend 实现。参考:`docs/plan/2026-07-23-batch6-m25-ssm.md`。完成判据:ADR-0005 裁决路线后实现并按 BUILD-011 与 cpu 数值一致。 | cpu 参考已注册；ascend 路线按 ADR-0005 推迟至 v2.0。 |
| heaviside_surrogate | TODO(FRAME-IMPL): 说明:接入 ascend 实现。参考:`docs/plan/2026-07-23-batch8-m27-snn.md`。完成判据:ADR-0005 裁决路线后实现并按 BUILD-011 与 cpu 数值一致。 | cpu 参考已注册；ascend 路线按 ADR-0005 推迟至 v2.0。 |
| scatter_add | TODO(FRAME-IMPL): 说明:接入 ascend 实现。参考:`docs/plan/2026-07-23-batch9-m28-gnn.md`。完成判据:ADR-0005 裁决路线后实现并按 BUILD-011 与 cpu 数值一致。 | cpu 参考已注册；ascend 路线按 ADR-0005 推迟至 v2.0。 |

---

## 7. 测试要求

- 骨架阶段:HAL 桩仅需保证编译通过与 `FRAME_UNIMPLEMENTED()` 行为一致,不含数值测试。
- 未来数值测试:`tests/cpp/backends/test_ascend_*.cpp`;与 CPU 参考实现(ARCH-041)对比,容差用 `standards/build-and-test.md`(BUILD-011,fp16 rtol=1e-2/atol=1e-3 等);后端可加严不可放松。
- HAL 一致性:注册后必须能通过 `tests/cpp/hal_conformance/` 的通用接口行为测试。
- 无设备行为:`npu-smi info` 检测不到昇腾设备时用 `GTEST_SKIP()` 并输出原因(BUILD-010)。
- 有设备不得跳过:在昇腾设备可用的环境中,昇腾测试不得 `GTEST_SKIP`,必须真实执行(M24 政策)。
- CI 现实:若 CI 无昇腾设备,PR 须附本地 `npu-smi` 环境的测试记录,或在 PR 中明确标注「未在昇腾设备验证」。

---

## 8. 已知限制与待查证清单

- 骨架阶段:`src/backends/ascend/` 仅为 HAL 桩,函数体 `return FRAME_UNIMPLEMENTED();`;不含任一执行路线实现。
- 执行模式未定:aclnn 逐算子与 GE 整图两路线均为候选,由 ADR-0005 推迟至 v2.0 裁决,本文档不做承诺(BE-ASC-002)。
- 【待查证】CANN toolkit 与 OS / NPU 驱动 / 固件的配套关系及最低版本 —— 来源:昇腾社区 hiascend.com 文档中心(CANN 安装指南)。
- 【待查证】CANN 支持的宿主 C++ 编译器及其对 C++20 的兼容性(GCC/Clang 最低版本)—— 来源:昇腾社区 CANN 安装 / 宿主编译器要求。
- 【待查证】aclnn 与 GE/ATC 两路线的性能、算子覆盖、工程复杂度取舍数据 —— 来源:昇腾社区 CANN 文档中心。
- 【待查证】`aclnnXxx` 两段式调用约定的具体签名 —— 来源:CANN aclnn API 参考。
- 【待查证】Ascend C 自定义算子工程脚手架与编译流程 —— 来源:CANN Ascend C 算子开发指南。
- 【待查证】设备属性查询 API —— 来源:CANN AscendCL API 参考。
- 【待查证】CANN 头文件与库在 `ASCEND_HOME_PATH` 下的相对路径布局 —— 来源:昇腾社区 CANN 安装指南目录结构章节。
- 【待查证】需链接的 CANN 库名清单 —— 来源:CANN AscendCL 链接说明。
- 【待查证】CANN 软件栈文档各章节稳定 URL(第 9 章链接的时效性)—— 来源:昇腾文档中心。
- 【待查证】Ascend C 算子开发指南稳定 URL(第 9 章链接的时效性)—— 来源:昇腾文档中心。

---

## 9. 权威参考文档

- 昇腾社区(Ascend)门户:https://www.hiascend.com/
- 昇腾文档中心:https://www.hiascend.com/document
- CANN 软件栈文档(AscendCL / aclnn / GE / ATC):https://www.hiascend.com/document 【待查证】各章节稳定 URL —— 来源:昇腾文档中心
- Ascend C 算子开发指南:https://www.hiascend.com/document 【待查证】稳定 URL —— 来源:昇腾文档中心
