# 后端矩阵总览与统一文档模板

> **强制等级**:规范(MUST)
> **相关铁律**:#1 编译优先 / #3 后端矩阵 / #5 两级复用
> **面向读者**:实现代码的工程师
> **最后更新**:2026-07-23(M28 后端能力与复用库状态回写)

本文档是 `docs/backends/` 的入口。它定义:①后端矩阵(哪些后端、注册名、执行模式、复用库);②所有单后端文档必须遵守的九章统一模板;③「待查证」标注的统一写法。单后端指南(`cuda.md`、`intel-gpu.md`、`intel-npu.md`、`ascend.md`)一律按本模板撰写。本 README 自身是模板的定义文件,不受九章约束。

---

## 1. 本文档的作用

- 给开发者一张「选后端 / 找指南」的索引表(第 2 章)。
- 冻结后端注册名字符串,防止各处拼写漂移(第 3 章)。
- 规定单后端文档的九章结构,缺章即打回(第 4 章)。
- 规定不确定信息的「待查证」写法,禁止凭记忆编造 API 名与版本号(第 5 章)。

新后端接入的完整流程见 `architecture/backend-hal.md` 的新后端接入 checklist(ARCH 系列);本文档只约束「文档形态」。

---

## 2. 后端矩阵总览

后端目录一律为 `src/backends/<注册名>/`;头文件 HAL 接口在 `include/frame/hal/{backend,stream,event,allocator,executable}.h`。执行模式列区分「逐 kernel 拼装 + 自研图编译」与「整图交厂商编译器」。

| 后端 | 注册名字符串 | 硬件角色 | 执行模式 | 主复用库 | CMake 选项 | Preset | 成熟度 | 指南文档 |
|---|---|---|---|---|---|---|---|---|
| CPU | `"cpu"` | 参考后端(永远启用) | 逐 kernel 标量参考实现 | 无(自研标量参考) | 无开关(始终编译) | `cpu-only` | 可用(M28;全部公开算子均有参考实现) | 无独立指南;参考实现规范见 `architecture/operator-system.md`(ARCH-041) |
| CUDA | `"cuda"` | NVIDIA GPU | 逐 kernel 拼装 + 自研图编译 | cuBLAS/cuBLASLt、cuDNN、cuFFT、CUTLASS、CUB | `FRAME_ENABLE_CUDA` | `cuda` | 可用(M28;完整设备门禁已过) | `cuda.md` |
| Intel GPU | `"intel_gpu"` | Intel GPU(SYCL/oneAPI) | 逐 kernel 拼装 + 自研图编译 | oneDNN、oneMKL | `FRAME_ENABLE_INTEL_GPU` | `intel-gpu` | 骨架(HAL 桩) | `intel-gpu.md` |
| Intel NPU | `"intel_npu"` | Intel NPU | 整图交 OpenVINO Runtime 编译 | OpenVINO Runtime | `FRAME_ENABLE_INTEL_NPU` | `intel-npu` | 骨架(HAL 桩) | `intel-npu.md` |
| Ascend | `"ascend"` | 昇腾 NPU | 待定(aclnn 逐算子 / GE 整图,均为候选,ADR-0005 推迟至 v2.0) | CANN(aclnn/AscendCL) | `FRAME_ENABLE_ASCEND` | `ascend` | 骨架(HAL 桩) | `ascend.md` |

说明:

- CPU 是参考后端,永远编译、不设开关;它承担 ARCH-041 要求的「每个算子必须提供的 CPU 参考实现」,是所有后端数值对比与回退链(ARCH-011)的基准。因此 CPU 无独立后端指南。
- `FRAME_ENABLE_{CUDA,INTEL_GPU,INTEL_NPU,ASCEND}` 均为三态开关 `AUTO|ON|OFF`,默认 `AUTO`(SDK 探测到则启用,探测不到则关闭;`ON` 时 SDK 缺失报 FATAL_ERROR)。语义详见 `standards/build-and-test.md`。
- 执行模式的「整图厂商编译」路线不提供逐算子 eager kernel;其回退是整图回退,见各后端第 1、7 章与 ARCH-011。

---

## 3. 注册名字符串契约

- 【BE-002】【MUST】后端注册键是字符串,取值为封闭枚举:`"cpu"`、`"cuda"`、`"intel_gpu"`、`"intel_npu"`、`"ascend"`。注册用 `FRAME_REGISTER_BACKEND(name_string, BackendClass)`,查询用 `BackendRegistry::instance().get("cuda")`。设备用值类型 `struct Device { std::string_view backend; int32_t index; }` 标识,不设 `DeviceType` 枚举。判定方法:`grep -rn 'FRAME_REGISTER_BACKEND' src/backends/` 的第一个实参必须是 `include/frame/core/device.h` 所列五个注册键常量(`frame::kCpuBackendName` 等)之一,或其对应字符串字面量;`grep -rnE 'enum (class )?DeviceType' include/ src/` 结果为空(注释中提及「不设 DeviceType 枚举」不算违例)。

新增后端字符串即扩展枚举,核心层零改动。字符串必须与目录名 `src/backends/<name>/` 逐字一致(下划线形式),文档文件名用连字符形式(`intel-gpu.md`)。

---

## 4. 单后端文档统一模板(九章)

- 【BE-001】【MUST】`cuda.md`、`intel-gpu.md`、`intel-npu.md`、`ascend.md` 必须逐章对应下列九章,章标题与顺序一致,缺任意一章即在 review 中打回。判定方法:对四篇文档运行 `grep -c '^## ' <file>.md`,九个二级标题必须齐全且与下列清单一一对应(允许 `## N. 标题` 编号前缀)。

| 章 | 标题 | 内容要求 |
|---|---|---|
| 1 | 定位与适用范围 | 该后端的硬件角色、执行模式(逐 kernel / 整图)、不适用场景 |
| 2 | 环境要求 | SDK/驱动版本、环境变量、**设备检测命令**(必须是可直接执行的命令) |
| 3 | 复用库清单与复用边界 | 三栏:必用(MUST)/ 可用(SHOULD/可选)/ 禁止自研(MUST NOT) |
| 4 | HAL 映射表 | HAL 接口 → 后端原生 API,一行一条 |
| 5 | 编译接入 | CMake 选项、find 方式、`enable_language`、链接目标、Preset |
| 6 | 算子/内核开发规范 | 何时手写、放哪个目录、命名、dtype 分发方式;章末必须含三级小节 `### 已支持算子表`(列:算子 \| 状态 \| 实现方式/备注;未实现项按 `TODO(FRAME-IMPL)` 标注;三级标题不计入本条的二级标题检查) |
| 7 | 测试要求 | 必测项 + 无硬件时的行为 + 有硬件时不得 SKIP |
| 8 | 已知限制与待查证清单 | 集中登记所有【待查证】,禁止散落正文 |
| 9 | 权威参考文档 | 官方文档名与 URL |

模板与其他层的对齐:

- 第 4 章 HAL 映射的接口清单以 `architecture/backend-hal.md`(ARCH 系列)为准;HAL 头文件桩为 `include/frame/hal/{backend,stream,event,allocator,executable}.h`,`Device` 为值类型(见 BE-002)。
- 第 3 章复用边界服从 `standards/reuse-policy.md`(REUSE-001 搜索五步、REUSE-010 新依赖需 ADR、REUSE-020 保留扩展点)。
- 第 7 章容差唯一来源为 `standards/build-and-test.md`(BUILD-011);后端可加严不可放松。
- 跨文档引用一律使用规则编号(如 ARCH-011、REUSE-010、BUILD-011),不写「第 N 节」。

---

## 5. 「待查证」写法规范

- 【BE-000】【MUST】后端文档中不确定的 API 名、版本号、驱动安装细节、行为语义,必须用固定格式标注,禁止凭记忆编造:

  ```
  【待查证】<具体问题> —— 来源:<官方文档名或 URL>
  ```

  判定方法:①正文出现具体版本号 / API 名 / 安装步骤而无官方来源支撑的,review 打回;②所有【待查证】条目必须在该文档第 8 章「已知限制与待查证清单」集中登记一份。`grep -n '【待查证】' <file>.md` 的每一条都应能在第 8 章找到对应登记。

- 【BE-003】【SHOULD】能给出稳定官方 URL 的,写 URL;只能给出文档名的,写文档名。不得引用第三方博客、论坛回答作为唯一来源。判定方法:review 检查来源是否为厂商官方域名或官方仓库。

---

## 6. 规则编号与交叉引用约定

- 后端文档规则前缀为 `BE-`;本 README 用 `BE-000`~`BE-003`,单后端文档用带后端码的编号:`BE-CUDA-###`、`BE-IGPU-###`、`BE-INPU-###`、`BE-ASC-###`。
- 规则条文统一格式:`【编号】【MUST/MUST NOT/SHOULD】正文。判定方法:<可机械执行的检查>。`
- 禁止「尽量 / 优雅 / 合理」等不可判定措辞;不可判定的意图必须转写为判定方法或降级为参考(INFO)。
- 引用其他文档一律用规则编号(ARCH-/CPP-/PY-/LANG-/REUSE-/BUILD-/BE-),不写章节号,避免章节调整后引用失效。
