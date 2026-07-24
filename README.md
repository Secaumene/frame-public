# Frame — 编译优先的深度学习框架

以计算图静态编译与 AOT 代码生成为核心、C++20 实现、多后端插件化的深度学习框架。

> 状态声明:**v1.3 M23–M28 已收口**——cpu 与 CUDA 双后端的「构图 → 九段管线 → 编译 → 执行」全链路、Python 薄绑定与编译期训练图可用；Mamba/FourierMamba、PINN/PINO 高阶图变换训练、LIF/SNN、GraphConv/HypergraphConv 公共面已验收。本机最终证据为 dev 1152/1152、CUDA 1152/1152、CUDA 后端直测 142/142 且零跳过、Python 89 通过与 2 项可选 ONNX 依赖跳过；Intel 与昇腾后端仍为桩，不宣称可用。

> **官方使用手册**:[help/README.md](help/README.md)。手册以 C++20 为主线，覆盖
> 从源码安装、仓外第一个 C++ 工程、静态图、训练、后端、开发流程与新增算子，
> 并提供 PyTorch 概念对照和可选 Python 绑定参考；
> 它是非规范性用户指南，架构与规则仍以
> [docs/](docs/README.md) 为准。
>
> **公开源码**：[Secaumene/frame-public](https://github.com/Secaumene/frame-public)。
> Frame 官方只发布纯源码，不提供预编译包；安装从源码在用户本机完成。

## 1. 项目定位

- **是什么**:一个「编译优先」的深度学习框架。默认路径是把模型捕获为静态计算图,经编译器
  pass 优化后做 AOT 代码生成;eager 逐算子解释仅作为调试与回退手段,而非主线。
- **为谁**:面向需要在多种硬件后端(CUDA / Intel GPU / Intel NPU / 昇腾 NPU)上统一一套编译栈、
  且对性能与可维护性要求较高的场景。
- **当前状态**:v1.3(M23–M28)已收口——cpu/cuda、Python 绑定、推理与训练编译路径，以及 Mamba/FourierMamba、PINN/PINO、SNN、GNN/HGNN 公共构图面均已通过最终验收。Intel 双后端按设备/SDK 就绪转入 v2.0，昇腾仍按 ADR-0005 推迟。

## 2. 核心设计原则

Frame 的一切工作由五条铁律约束(完整可机械判定的定义分布于 [docs/](docs/README.md) 各规范):

| 原则 | 一句话 | 深入阅读 |
|---|---|---|
| 编译优先 | 图静态编译 / AOT 优先于 eager;编译期机制优先于运行时机制 | [docs/architecture/execution-model.md](docs/architecture/execution-model.md) |
| C++ 为核心 | 功能逻辑落在 C++;Python 仅做薄绑定 | [docs/standards/python-binding.md](docs/standards/python-binding.md) |
| 后端统一抽象 | 四后端经统一 HAL 插件式接入 | [docs/architecture/backend-hal.md](docs/architecture/backend-hal.md) |
| 中文文档英文代码 | 沟通/注释/文档用中文,标识符/API/日志用英文 | [docs/standards/language-policy.md](docs/standards/language-policy.md) |
| 两级复用 | 优先成熟库 + 项目内先搜索再动手,并保证可扩展 | [docs/standards/reuse-policy.md](docs/standards/reuse-policy.md) |

## 3. 架构总览

每层职责一句话,详细设计见 [docs/architecture/overview.md](docs/architecture/overview.md)。

```
        Python API (pybind11 绑定, 薄封装)
                      │
 ┌────────────────────▼────────────────────┐
 │        C++ Frontend API (Tensor/Module) │
 ├─────────────────────────────────────────┤
 │  Graph IR  ──►  Compiler Passes  ──►    │
 │  (静态计算图)   (PassRegistry 可扩展)     │
 │                       │                 │
 │                  Codegen / AOT          │
 ├─────────────────────────────────────────┤
 │  Runtime (执行器/内存/流)  + OpRegistry   │
 ├─────────────────────────────────────────┤
 │     Backend 统一抽象 (HAL: backend.h)     │
 └──┬─────────┬──────────┬──────────┬──────┘
    │         │          │          │
  CUDA    Intel GPU   Intel NPU   Ascend NPU
 (cuDNN/  (SYCL/      (OpenVINO)   (CANN/
 CUTLASS)  oneDNN)                 AscendCL)
```

> 注:C++ Frontend API 已实体化为 frontend 层与 frame_dslc 工具(ADR-0017)。

## 4. 目录导览

| 目录 | 职责 | 深入阅读入口 |
|---|---|---|
| `include/frame/` | 公共头文件(core / ir / compiler / hal / ops) | [docs/architecture/overview.md](docs/architecture/overview.md) |
| `src/` | C++ 实现(core / ir / compiler / runtime / ops / backends) | 各头文件对应实现单元(core、ir、ops、hal/cpu、CUDA、内置算子、runtime 编译闭环与九段标准管线已实化;Intel 与昇腾后端仍为桩) |
| `python/` | Python 绑定(pybind11 薄壳 + 薄层纯 Python) | [docs/standards/python-binding.md](docs/standards/python-binding.md) |
| `tests/` | C++ 与 Python 测试 | [docs/standards/build-and-test.md](docs/standards/build-and-test.md) |
| `cmake/` | CMake 模块(选项 / 编译标志 / 依赖 / 后端 / find_cann) | [docs/standards/build-and-test.md](docs/standards/build-and-test.md) |
| `scripts/` | 机械执法脚本与 git 钩子 | [docs/standards/build-and-test.md](docs/standards/build-and-test.md) |
| `docs/` | 规范 / 架构 / 后端 / 决策文档(单一事实来源) | [docs/README.md](docs/README.md) |
| `help/` | 官方使用手册(非规范性用户指南) | [help/README.md](help/README.md) |
| `examples/` | 带中文注释的可运行 C++ 示例(`dev` / `cuda` preset 持续构建) | [examples/README.md](examples/README.md) |
| `third_party/` | 第三方依赖(默认为空 + 依赖策略指引) | [docs/standards/reuse-policy.md](docs/standards/reuse-policy.md) |
| `.github/` | CI 工作流与 PR 模板 | `.github/workflows/` |

## 5. 快速开始

**依赖**:CMake ≥ 3.25、g++ ≥ 10 或 icpx(需支持 C++20);Python 3(用于绑定,可选)。本机 CUDA 构建使用已安装的 CUDA Toolkit 13.3、cuDNN、cuBLAS、cuFFT 与 CUB，并显式指定 `/usr/local/cuda` 根路径；Intel oneAPI、OpenVINO Runtime 与昇腾 CANN 仅在对应后端落地时需要。

**构建与测试**(本机 NVIDIA CUDA 主路径):CUDA Toolkit 未加入默认 PATH 时,
显式指定已安装根路径即可；cuBLAS/CUB/cuFFT 随 Toolkit 探测,系统 multiarch
目录下的 cuDNN 由后端 CMake 自动查找:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDAToolkit_ROOT=/usr/local/cuda
cmake --preset cuda && cmake --build --preset cuda && ctest --preset cuda
```

便利封装:`bash scripts/build.sh cuda && bash scripts/test.sh cuda`(原生 preset
命令为权威,脚本清单见 docs/standards/build-and-test.md 第 10 节)。

**模型描述工具**(JSON DSL → 训练/推理,详见 [docs/architecture/frontend-dsl.md](docs/architecture/frontend-dsl.md)):

```bash
./build/cpu-only/tools/frame_dslc --run tools/frame_dslc/testdata/tiny_mlp.json      # 进程内训练
./build/cpu-only/tools/frame_dslc --emit tools/frame_dslc/testdata/tiny_mlp.json --out /tmp/gen  # 生成独立 C++ 工程
```

生成工程消费系统安装件:先 `bash scripts/install.sh --prefix <dir>`,再以 `CMAKE_PREFIX_PATH=<dir>` 配置构建。

其他 preset:`dev`(默认开发入口)、`dev-asan`、`release`、`cuda`、`intel-gpu`、`intel-npu`、
`ascend`、`wheel`。

**后端开关**(三态 `AUTO | ON | OFF`;`AUTO` = 探测到对应 SDK 则启用;CPU 参考后端永远启用):

| CMake 选项 | 说明 | 默认 |
|---|---|---|
| `FRAME_ENABLE_CUDA` | 启用 CUDA 后端(需 CUDAToolkit ≥ 12.0) | `AUTO` |
| `FRAME_ENABLE_INTEL_GPU` | 启用 Intel GPU 后端(SYCL/oneAPI) | `AUTO` |
| `FRAME_ENABLE_INTEL_NPU` | 启用 Intel NPU 后端(OpenVINO) | `AUTO` |
| `FRAME_ENABLE_ASCEND` | 启用昇腾 NPU 后端(CANN) | `AUTO` |
| `FRAME_ENABLE_MLIR` | 启用 MLIR 并行赛道(见 [ADR-0002](docs/decisions/0002-ir-dual-track-with-mlir.md)) | `OFF` |

**Python 绑定安装**(在仓库根执行):

```bash
pip install -e .
```

首次克隆可运行 `bash scripts/setup_dev.sh` 一键完成 git 钩子安装与 `pip install -e .`。

> 当前状态:cpu/cuda 路径全链路可用;Python 绑定已落地(M12,分发名 `frame-ml`,
> `pip install -e .` 本地安装；wheel 仅供本地构建验证，Frame 官方不上传或发布
> wheel);前端层与 frame_dslc 工具已落地(ADR-0017/0018)。

## 6. 后端支持矩阵

| 后端 | 依赖 SDK | 编程模型 | 状态 | 文档 |
|---|---|---|---|---|
| CPU(参考后端) | — | — | 可用(注册/分配/拷贝/eager/编译执行闭环) | — |
| CUDA | CUDA Toolkit 13.3(本机;cuDNN/cuBLAS/cuFFT/CUB 已安装) | CUDA | 可用(M28；完整设备门禁已过) | [docs/backends/cuda.md](docs/backends/cuda.md) |
| Intel GPU | Intel oneAPI(oneDNN) | SYCL | 骨架 | [docs/backends/intel-gpu.md](docs/backends/intel-gpu.md) |
| Intel NPU | OpenVINO Runtime | OpenVINO | 骨架 | [docs/backends/intel-npu.md](docs/backends/intel-npu.md) |
| 昇腾 NPU | CANN(AscendCL) | CANN | 骨架 | [docs/backends/ascend.md](docs/backends/ascend.md) |

后端接入总览见 [docs/backends/README.md](docs/backends/README.md)。

## 7. 贡献指南

单一事实来源,不在此复制内容,只给指针:

1. **先读 [docs/README.md](docs/README.md)**——文档导航与全部规范的入口。
2. **按 docs/README.md 路由表找到任务对应的必读文档**,动手前先读。
3. **遵守 commit 规范与提交前自检清单**(唯一权威来源
   [docs/standards/language-policy.md](docs/standards/language-policy.md))。提交前 `bash scripts/ci_check.sh` 须全绿。

## 8. 许可证与发布

Frame 源码与文档（另有声明者除外）按
[GNU General Public License v3.0 or later](LICENSE) 发布，SPDX 标识符为
`GPL-3.0-or-later`，决策见
[ADR-0023](docs/decisions/0023-adopt-gpl3-source-only-release.md)。

Frame 官方公开出口只包含
[公开仓库](https://github.com/Secaumene/frame-public)的纯源码，以及托管平台从
已验收提交自动生成的源码归档。CI 可以临时编译和测试，但不上传 executable、库、
object、wheel、系统包、容器、厂商 SDK 或与可选专有 SDK 链接的产物。该政策只约束
Frame 官方发布流程，不附加限制到下游 GPL 权利；完整规则见 BUILD-050～052。
