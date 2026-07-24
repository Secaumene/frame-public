# Intel NPU 后端指南

> **强制等级**:规范(MUST)
> **相关铁律**:#1 编译优先 / #3 后端矩阵 / #5 两级复用
> **面向读者**:实现代码的工程师
> **最后更新**:2026-07-23(M25/M27/M28 算子待实现项回写)

本文档按 `backends/README.md`(BE-001)定义的九章模板撰写。注册名字符串为 `"intel_npu"`,后端目录 `src/backends/intel_npu/`。所有不确定信息按 BE-000 用【待查证】标注并在第 8 章集中登记。

---

## 1. 定位与适用范围

- 目标硬件:Intel NPU。注册名 `"intel_npu"`,`FRAME_REGISTER_BACKEND("intel_npu", IntelNpuBackend)`。
- 执行模式:**整图编译型后端**。`Backend::compile` 是唯一主路径:自研 IR → `ov::Model` → `ov::Core::compile_model` → `ov::CompiledModel`;`Executable::run` 经 `ov::InferRequest` 执行。
- 关键差异:**不提供逐算子 eager kernel**。不支持的图不做逐算子降级,而是整图回退(策略见 ARCH-011,由上层统一决策)。
- 接入约束:**一律经 OpenVINO Runtime**;禁止绕过 OpenVINO 直连 Level Zero 写计算内核(见第 3 章 BE-INPU-001)。
- 不适用场景:需要逐算子即时执行、需要在 NPU 上做 eager 调试的用法(NPU 无 eager 路径;调试请用 CPU 参考或整图 dump)。

---

## 2. 环境要求

- 运行时:OpenVINO Runtime。
  - 【待查证】开始支持 Intel NPU 设备的 OpenVINO Runtime 最低版本(约 2024.x)—— 来源:OpenVINO Release Notes。
- 驱动:Intel NPU 内核模块与用户态驱动。
  - 【待查证】Linux 下 Intel NPU 内核模块与用户态驱动的安装方式与最低版本 —— 来源:intel/linux-npu-driver GitHub 仓库 README。
- 环境变量:OpenVINO 的运行时环境(部分发行版需 `setupvars.sh`)。【待查证】NPU 插件所需的额外环境变量 —— 来源:OpenVINO NPU 设备文档。
- 设备检测命令(可直接执行):

  ```bash
  # 通过 OpenVINO Python 查询可用设备,输出中应包含 "NPU"
  python -c "import openvino as ov; print(ov.Core().available_devices)"
  ```

  C++ 侧等价检测:`ov::Core::get_available_devices()` 返回列表中包含 `"NPU"`。

---

## 3. 复用库清单与复用边界

复用取向:**全部经 OpenVINO Runtime**;NPU 侧不存在「手写 kernel」这一栏——本后端不写设备计算内核。

| 栏位 | 库 / 路径 | 覆盖范围 |
|---|---|---|
| 必用(MUST) | OpenVINO Runtime(`ov::Core` / `ov::Model` / `ov::CompiledModel` / `ov::InferRequest`) | 图编译与推理执行的唯一通道 |
| 可用(需 ADR) | Level Zero(NPU 扩展) | 仅当 OpenVINO 能力不足时,经 ADR 引入 |
| 禁止自研(MUST NOT) | —— | 绕过 OpenVINO 直连 Level Zero 的计算内核;逐算子 eager kernel |

- 【BE-INPU-001】【MUST】Intel NPU 一律经 OpenVINO Runtime 接入(`ov::Core::compile_model` → `ov::InferRequest`);禁止绕过 OpenVINO 直接写 Level Zero 计算内核。判定方法:`grep -rn 'ze_\|zeInit\|level_zero' src/backends/intel_npu/` 若命中,必须有对应已接受 ADR 引用,否则打回;`src/backends/intel_npu/` 存在计算 kernel 源文件即打回。
- 【BE-INPU-002】【MUST】本后端为整图编译型,不实现逐算子 eager kernel;不支持的图整体回退,不做逐算子静默降级(呼应 ARCH-031)。判定方法:`FRAME_REGISTER_KERNEL(..., "intel_npu", ...)` 不应出现;`grep -rn 'FRAME_REGISTER_KERNEL' src/backends/intel_npu/` 结果为空。
- 【BE-INPU-003】【MUST】Level Zero 直连方案已废弃;仅在 OpenVINO 能力确实不足时经 ADR 重新引入。判定方法:REUSE-010(新增依赖需 ADR)+ 上述 BE-INPU-001 的 grep。
- 引入方式:OpenVINO 走 `find_package(OpenVINO)`(SDK 自带,REUSE 引入方式第一档)。
  - 【待查证】OpenVINO NPU 扩展 / Level Zero NPU 扩展的能力边界 —— 来源:oneAPI Level Zero 规范与 intel/level-zero-npu-extensions 仓库。

---

## 4. HAL 映射表

本后端不提供逐算子 kernel;HAL 接口映射到 OpenVINO 的图编译与异步推理 API。`Device` 为值类型 `{backend, index}`(BE-002)。错误一律转为 `Result<T>` / `Status`(`include/frame/core/status.h`),不跨 HAL 边界抛异常(CPP-020)。

| HAL 接口 / 操作 | OpenVINO Runtime API |
|---|---|
| Backend 设备枚举 | `ov::Core::get_available_devices`(筛选 `"NPU"`) |
| Backend 设备属性 | `ov::Core::get_property("NPU", ...)` |
| Backend::compile | 自研 IR → `ov::Model` → `ov::Core::compile_model("NPU")` → `ov::CompiledModel` |
| Executable::run | `ov::CompiledModel::create_infer_request` → `ov::InferRequest::infer` / `start_async` + `wait` |
| Stream / Event | 【待查证】映射到 `ov::InferRequest` 的异步 API(`start_async` / `wait` / 回调)的对应语义 —— 来源:OpenVINO Runtime C++ API 文档(异步推理) |
| Allocator | NPU 侧内存由 OpenVINO 托管;HAL Allocator 实现为宿主内存 + import。【待查证】能否零拷贝(remote tensor)—— 来源:OpenVINO Remote Tensor / RemoteContext 文档 |
| 错误转换 | `OV_CHECK` 宏:`ov::Exception` → `Status`(消息一律英文,LANG-005) |

- IR → `ov::Model` 的逐算子映射表:首批算子映射需逐条列出,未覆盖的算子标【待查证】并进入整图回退。
  - 【待查证】自研 IR 首批算子(如 add/matmul/conv/relu/softmax 等)到 OpenVINO opset 的逐条映射 —— 来源:OpenVINO Operation Specifications(opset 文档)。

---

## 5. 编译接入

- CMake 选项:`FRAME_ENABLE_INTEL_NPU`,三态 `AUTO|ON|OFF`,默认 `AUTO`(语义见 BUILD 系列)。
- SDK 探测:`find_package(OpenVINO)`。
- 语言/编译器:本后端为主机端 C++ 代码(不含设备 kernel),用系统 C++ 编译器按 C++20 编译即可,无需 icpx。
- 链接目标:后端库目标 `frame_backend_intel_npu`,链接 OpenVINO Runtime 导入目标。
  - 【待查证】OpenVINO 的 CMake 导入目标名(如 `openvino::runtime`)—— 来源:OpenVINO 官方 CMake 集成文档。
- 无 Level Zero:不引入 `find_level_zero.cmake`,不建 `level_zero/` 目录;Level Zero 直连方案已废弃(BE-INPU-003)。
- Preset:`intel-npu`。骨架验收口径为 `cpu-only`,不依赖 OpenVINO。
- 后端代码用 CMake 目标隔离,`src/{core,ir,compiler,runtime}` 中禁止出现 `#ifdef FRAME_ENABLE_INTEL_NPU`(BUILD-003)。

---

## 6. 算子/内核开发规范

- 本后端**无逐算子 kernel 开发**(BE-INPU-002)。「加算子」在本后端等同于「在 IR → `ov::Model` 映射表中补一条算子映射」。
- 目录:IR→`ov::Model` 映射与编译接入代码放 `src/backends/intel_npu/`;无 `kernels/` 目录。
- 不支持的算子:映射表未覆盖时,不得静默降级(ARCH-031),整图回退由上层决策(ARCH-011)。
- 扩展点:映射表必须保留可扩展路径,自定义算子可补映射条目(呼应 REUSE-020,不得堵死扩展)。

### 已支持算子表

本后端「支持一个算子」= IR → `ov::Model` 映射表覆盖该算子;新增映射时维护本表,未覆盖项按 `TODO(FRAME-IMPL)` 标注,静默缺失即违例。
`fused_elementwise_internal`(M9 融合 pass 产物,非用户构图算子,不入本表;cpu 组合调用参考语义见 docs/architecture/compiler-passes.md §3.7)。

| 算子 | 状态 | 实现方式/备注 |
|---|---|---|
| add | `TODO(FRAME-IMPL)` | cpu 参考实现已注册(`src/backends/cpu/kernels/elementwise.cpp`);IR → `ov::Model` 映射待补(拟映射 OpenVINO opset `Add`)。参考:`docs/architecture/operator-system.md` 第4章。完成判据:映射条目落地且整图编译执行数值与 cpu 参考实现一致(BUILD-011 容差)。 |
| mul | `TODO(FRAME-IMPL)` | cpu 参考实现已注册(`src/backends/cpu/kernels/elementwise.cpp`);IR → `ov::Model` 映射待补(拟映射 OpenVINO opset `Multiply`)。参考:`docs/architecture/operator-system.md` 第4章。完成判据:映射条目落地且整图编译执行数值与 cpu 参考实现一致(BUILD-011 容差)。 |
| relu | `TODO(FRAME-IMPL)` | cpu 参考实现已注册(`src/backends/cpu/kernels/elementwise.cpp`);IR → `ov::Model` 映射待补(拟映射 OpenVINO opset `Relu`)。参考:`docs/architecture/operator-system.md` 第4章。完成判据:映射条目落地且整图编译执行数值与 cpu 参考实现一致(BUILD-011 容差)。 |
| sum | `TODO(FRAME-IMPL)` | cpu 参考实现已注册(`src/backends/cpu/kernels/reduction.cpp`);IR → `ov::Model` 映射待补(拟映射 OpenVINO opset `ReduceSum`)。参考:`docs/architecture/operator-system.md` 第4章。完成判据:映射条目落地且整图编译执行数值与 cpu 参考实现一致(BUILD-011 容差)。 |
| matmul | `TODO(FRAME-IMPL)` | cpu 参考实现已注册(`src/backends/cpu/kernels/matmul.cpp`);IR → `ov::Model` 映射待补(拟映射 OpenVINO opset `MatMul`)。参考:`docs/architecture/operator-system.md` 第4章。完成判据:映射条目落地且整图编译执行数值与 cpu 参考实现一致(BUILD-011 容差)。 |
| square | `TODO(FRAME-IMPL)` | cpu 参考实现已注册(`src/backends/cpu/kernels/elementwise.cpp`);另有 decomposition(square(x)=mul(x,x),见 `src/ops/schemas/elementwise.cpp::square_decompose`,M10 回退链素材);IR → `ov::Model` 映射待补(拟映射 OpenVINO opset `Power`,指数常量 2;或等价拆成 `Multiply(x, x)`,二者择一待接入时定案)。参考:`docs/architecture/operator-system.md` 第4章。完成判据:映射条目落地且整图编译执行数值与 cpu 参考实现一致(BUILD-011 容差)。 |
| constant | 已支持(cpu) | cpu 参考实现已注册(`src/backends/cpu/kernels/constant.cpp`,M8);0 输入 1 输出,attrs=value/shape/dtype,常量物化。参考:`docs/architecture/operator-system.md` 第4章。IR → `ov::Model` 映射待补(拟映射 OpenVINO opset `Const`)。 |
| selective_scan | TODO(FRAME-IMPL): 说明:补齐 `ov::Model` 映射。参考:`docs/plan/2026-07-23-batch6-m25-ssm.md`。完成判据:整图执行按 BUILD-011 与 cpu 数值一致。 | cpu 参考实现已注册。 |
| heaviside_surrogate | TODO(FRAME-IMPL): 说明:补齐 `ov::Model` 映射。参考:`docs/plan/2026-07-23-batch8-m27-snn.md`。完成判据:整图执行按 BUILD-011 与 cpu 数值一致。 | cpu 参考实现已注册。 |
| scatter_add | TODO(FRAME-IMPL): 说明:补齐 `ov::Model` 映射。参考:`docs/plan/2026-07-23-batch9-m28-gnn.md`。完成判据:整图执行按 BUILD-011 与 cpu 数值一致。 | cpu 参考实现已注册。 |

---

## 7. 测试要求

- 测试形态:图级端到端测试为主(无逐算子 kernel 测试)。`tests/cpp/backends/test_intel_npu_*.cpp` 覆盖「自研 IR → `ov::Model` → 编译 → 推理」全链路。
- 转换正确性冒烟:可用 OpenVINO 的 CPU 设备(`"CPU"`)做 IR→`ov::Model` 转换正确性冒烟,但必须在测试名/注释中标注「这是转换测试,不是 NPU 设备测试」。
- 数值对比:与 CPU 参考实现(ARCH-041)对比,容差用 `standards/build-and-test.md`(BUILD-011)。
- HAL 一致性:注册后必须能通过 `tests/cpp/hal_conformance/` 的通用接口行为测试(整图编译型后端对应的接口子集)。
- 无 NPU 行为:`available_devices` 不含 `"NPU"` 时用 `GTEST_SKIP()` 并输出原因(BUILD-010)。
- 有 NPU 不得跳过:在 NPU 可用的环境中,NPU 测试不得 `GTEST_SKIP`,必须真实执行(M24 政策);仅当前环境缺设备时允许 SKIP 并在 PR 说明。

---

## 8. 已知限制与待查证清单

- 骨架阶段:`src/backends/intel_npu/` 仅为 HAL 桩,函数体 `return FRAME_UNIMPLEMENTED();`,不含真实映射与编译逻辑。相关实现以 `TODO(FRAME-IMPL)` 标注。
- 结构限制:整图编译型,无 eager kernel;**不支持的图**整体回退,不做逐算子降级(与第 1 章一致)。
- 【待查证】开始支持 Intel NPU 的 OpenVINO Runtime 最低版本 —— 来源:OpenVINO Release Notes。
- 【待查证】Linux 下 Intel NPU 内核模块与用户态驱动的安装方式与最低版本 —— 来源:intel/linux-npu-driver GitHub 仓库 README。
- 【待查证】NPU 插件所需的额外环境变量 —— 来源:OpenVINO NPU 设备文档。
- 【待查证】`ov::InferRequest` 异步 API 到 HAL Stream/Event 的语义对应 —— 来源:OpenVINO Runtime C++ API 文档(异步推理)。
- 【待查证】NPU 侧内存能否零拷贝(remote tensor)—— 来源:OpenVINO Remote Tensor / RemoteContext 文档。
- 【待查证】自研 IR 首批算子到 OpenVINO opset 的逐条映射 —— 来源:OpenVINO Operation Specifications。
- 【待查证】OpenVINO NPU 扩展 / Level Zero NPU 扩展能力边界 —— 来源:oneAPI Level Zero 规范与 intel/level-zero-npu-extensions 仓库。
- 【待查证】OpenVINO 的 CMake 导入目标名 —— 来源:OpenVINO 官方 CMake 集成文档。
- 【待查证】OpenVINO Release Notes 当前版本 URL(第 9 章链接的时效性)—— 来源:OpenVINO 文档门户。
- 【待查证】OpenVINO NPU 设备文档当前版本 URL(第 9 章链接的时效性)—— 来源:OpenVINO 文档门户。

---

## 9. 权威参考文档

- OpenVINO 文档门户:https://docs.openvino.ai/
- OpenVINO Release Notes:https://docs.openvino.ai/2024/about-openvino/release-notes-openvino.html 【待查证】当前版本 URL —— 来源:OpenVINO 文档门户
- OpenVINO NPU 设备文档:https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html 【待查证】当前版本 URL —— 来源:OpenVINO 文档门户
- OpenVINO Operation Specifications(opset):https://docs.openvino.ai/
- Intel Linux NPU Driver(GitHub):https://github.com/intel/linux-npu-driver
- Level Zero NPU Extensions(GitHub):https://github.com/intel/level-zero-npu-extensions
- oneAPI Level Zero 规范:https://spec.oneapi.io/level-zero/latest/index.html
