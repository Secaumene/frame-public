# Intel GPU 后端指南

> **强制等级**:规范(MUST)
> **相关铁律**:#1 编译优先 / #3 后端矩阵 / #5 两级复用
> **面向读者**:实现代码的工程师
> **最后更新**:2026-07-23(M25/M27/M28 算子待实现项回写)

本文档按 `backends/README.md`(BE-001)定义的九章模板撰写。注册名字符串为 `"intel_gpu"`,后端目录 `src/backends/intel_gpu/`。所有不确定信息按 BE-000 用【待查证】标注并在第 8 章集中登记。

---

## 1. 定位与适用范围

- 目标硬件:Intel GPU(经 SYCL/oneAPI 编程模型)。注册名 `"intel_gpu"`,`FRAME_REGISTER_BACKEND("intel_gpu", IntelGpuBackend)`。
- 执行模式:逐 kernel 拼装 + 自研图编译。编译路径(`Backend::compile` 产出 `Executable`)是默认路径(ARCH-010);eager 逐算子仅用于调试、回退与单算子测试(ARCH-011)。
- 复用取向:优先 oneDNN primitive 与 oneMKL;手写 SYCL kernel 受与 CUDA 一致的准入约束(见第 3、6 章)。
- 不适用场景:非 Intel GPU;绕过 HAL 直接向核心层暴露 `sycl::queue` 的用法(ARCH-030 禁止上层调用 `native_handle`)。

---

## 2. 环境要求

- 编译器/工具链:Intel oneAPI(DPC++/C++ 编译器 `icpx`)。该后端源码需用 `icpx` 编译。
  - 【待查证】支持 C++20 与目标 Intel GPU 的 oneAPI Base Toolkit 最低版本 —— 来源:Intel oneAPI Base Toolkit 发行说明。
- 运行库:oneDNN、oneMKL(GPU 后端)。【待查证】oneDNN/oneMKL 支持目标 Intel GPU 的最低版本 —— 来源:Intel oneDNN / oneMKL 官方文档。
- 驱动:Intel GPU 计算运行时(Level Zero / OpenCL 后端)。【待查证】目标 Intel GPU 的用户态计算驱动安装方式与最低版本 —— 来源:Intel GPU 驱动安装文档(intel/compute-runtime)。
- 环境脚本:oneAPI 的 `setvars.sh`(设置编译器、oneDNN、oneMKL 与 Level Zero 环境)。
- 设备检测命令(可直接执行):

  ```bash
  source /opt/intel/oneapi/setvars.sh   # 路径以本机 oneAPI 安装为准
  sycl-ls                               # 列出 SYCL 平台与设备,确认存在 Intel GPU
  ```

---

## 3. 复用库清单与复用边界

复用优先级:**oneDNN(神经网络原语)→ oneMKL(BLAS)→ 手写 SYCL kernel(仅受限三类)**。

| 栏位 | 库 | 覆盖范围 |
|---|---|---|
| 必用(MUST) | oneDNN | conv、matmul、norm、pooling、激活等神经网络 primitive |
| 必用(MUST) | oneMKL | BLAS(GEMM)、部分数学函数 |
| 可用(可选) | oneDPL | 工具/测试代码中的并行算法 |
| 禁止自研(MUST NOT) | —— | 与 oneDNN/oneMKL 功能重叠的手写 SYCL kernel |

- 【BE-IGPU-001】【MUST】conv/matmul/norm/pooling 等优先 oneDNN primitive;BLAS 用 oneMKL。手写 SYCL kernel 仅限 elementwise、项目自定义算子、库覆盖不到的融合三类;手写前须在 PR 粘贴「库不满足」的具体依据(准入同 CUDA 的 BE-CUDA-002)。判定方法:review 检查 `src/backends/intel_gpu/` 是否存在与 oneDNN/oneMKL 重叠的手写实现,无准入依据即打回。
- 【BE-IGPU-002】【MUST】队列默认使用 in-order queue(`sycl::property::queue::in_order`);使用乱序队列需先有 ADR。判定方法:`grep -rn 'sycl::queue' src/backends/intel_gpu/` 命中处应带 in-order 属性,或有对应 ADR 引用。
- 引入方式:oneDNN/oneMKL 随 oneAPI 提供,走 `find_package`(SDK 自带,REUSE 引入方式第一档);新增其他第三方依赖须先有 ADR(REUSE-010)。

---

## 4. HAL 映射表

HAL 接口清单以 `architecture/backend-hal.md`(ARCH 系列)为准。`Device` 为值类型 `{backend, index}`(BE-002)。错误一律转为 `Result<T>` / `Status`(`include/frame/core/status.h`),不跨 HAL 边界抛异常(CPP-020)。

| HAL 接口 / 操作 | Intel GPU(SYCL/oneAPI)原生 API |
|---|---|
| Backend 设备枚举 / 选择 | `sycl::device`(GPU selector)/ `sycl::platform::get_devices` |
| Backend 设备属性 | `sycl::device::get_info<...>`(名称、全局内存、compute units) |
| Stream | `sycl::queue`(in-order queue,BE-IGPU-002) |
| Stream::synchronize | `sycl::queue::wait`(或 `wait_and_throw`) |
| Event | `sycl::event` |
| Event::synchronize / query | `sycl::event::wait` / `get_info<command_execution_status>` |
| Allocator::allocate / deallocate | USM:`sycl::malloc_device` / `sycl::free` |
| memcpy(H2D/D2H/D2D) | `sycl::queue::memcpy` |
| 错误转换 | `SYCL_CHECK` 宏:`sycl::exception` → `Status`(消息一律英文,LANG-005) |

- Level Zero 作为底层驱动一般不直接调用;确需与 Level Zero 互操作时经 `sycl::get_native` / backend interop,不绕过 SYCL 自写计算内核。
  - 【待查证】`sycl::get_native` 到 Level Zero 句柄的互操作细节与生命周期约束 —— 来源:SYCL 2020 规范与 oneAPI Level Zero 文档。

---

## 5. 编译接入

- CMake 选项:`FRAME_ENABLE_INTEL_GPU`,三态 `AUTO|ON|OFF`,默认 `AUTO`(语义见 BUILD 系列)。
- SDK 探测:`find_package(IntelSYCL)`(oneAPI 提供)。
- 编译器:该后端源码需用 `icpx` 编译。与主工具链(g++)的混合构建策略——将 Intel GPU 后端独立编译为动态库插件,主体用系统编译器,后端目标用 `icpx`。
  - 【待查证】`icpx` 与系统 g++ 的混合构建在 CMake 中的具体配置(per-target 编译器切换 vs 独立构建)—— 来源:Intel oneAPI DPC++/C++ Compiler Developer Guide 与 IntelSYCL CMake 集成文档。
- 链接目标:后端库目标 `frame_backend_intel_gpu`,链接 SYCL 运行时、oneDNN、oneMKL(GPU)导入目标。
  - 【待查证】oneDNN / oneMKL 的 CMake 导入目标名(`DNNL::dnnl`、`MKL::MKL` 等)—— 来源:oneDNN / oneMKL CMake 集成文档。
- Preset:`intel-gpu`。骨架验收口径为 `cpu-only`,不依赖 oneAPI。
- 后端代码用 CMake 目标隔离,`src/{core,ir,compiler,runtime}` 中禁止出现 `#ifdef FRAME_ENABLE_INTEL_GPU`(BUILD-003)。

---

## 6. 算子/内核开发规范

- 目录:SYCL kernel 源码放 `src/backends/intel_gpu/kernels/`;后端接入/注册代码放 `src/backends/intel_gpu/`。
- 注册:`FRAME_REGISTER_KERNEL(op, "intel_gpu", fn)`(kernel 注册键不含 dtype)。
- dtype 分发:dtype 差异在 kernel 内经 `dispatch_dtype` 编译期展开(模板 + `if constexpr`),禁止 kernel 内层循环出现按 dtype/设备的运行时分支(ARCH-042)。判定方法:`check_iron_rules.sh` 扫描 `src/backends/intel_gpu/kernels/`。
- 命名:kernel 与工具函数 `snake_case`;优先 oneDNN/oneMKL,手写路径仅限受限三类(BE-IGPU-001)。
- 队列:统一使用 in-order queue(BE-IGPU-002),简化事件依赖管理。

### 已支持算子表

新增/补齐算子时维护本表;未实现项按 `TODO(FRAME-IMPL)` 标注,静默缺失即违例。
`fused_elementwise_internal`(M9 融合 pass 产物,非用户构图算子,不入本表;cpu 组合调用参考语义见 docs/architecture/compiler-passes.md §3.7)。

| 算子 | 状态 | 实现方式/备注 |
|---|---|---|
| add | `TODO(FRAME-IMPL)` | cpu 参考实现已注册(`src/backends/cpu/kernels/elementwise.cpp`);intel_gpu kernel 待接入,HAL 五接口实化时机见后端矩阵。参考:`docs/architecture/operator-system.md` 第4章。完成判据:`FRAME_REGISTER_KERNEL("add", "intel_gpu", fn)` 落地且与 cpu 参考实现数值一致(BUILD-011 容差)。 |
| mul | `TODO(FRAME-IMPL)` | cpu 参考实现已注册(`src/backends/cpu/kernels/elementwise.cpp`);intel_gpu kernel 待接入,HAL 五接口实化时机见后端矩阵。参考:`docs/architecture/operator-system.md` 第4章。完成判据:`FRAME_REGISTER_KERNEL("mul", "intel_gpu", fn)` 落地且与 cpu 参考实现数值一致(BUILD-011 容差)。 |
| relu | `TODO(FRAME-IMPL)` | cpu 参考实现已注册(`src/backends/cpu/kernels/elementwise.cpp`);intel_gpu kernel 待接入,HAL 五接口实化时机见后端矩阵。参考:`docs/architecture/operator-system.md` 第4章。完成判据:`FRAME_REGISTER_KERNEL("relu", "intel_gpu", fn)` 落地且与 cpu 参考实现数值一致(BUILD-011 容差)。 |
| sum | `TODO(FRAME-IMPL)` | cpu 参考实现已注册(`src/backends/cpu/kernels/reduction.cpp`);intel_gpu kernel 待接入,HAL 五接口实化时机见后端矩阵。参考:`docs/architecture/operator-system.md` 第4章。完成判据:`FRAME_REGISTER_KERNEL("sum", "intel_gpu", fn)` 落地且与 cpu 参考实现数值一致(BUILD-011 容差)。 |
| matmul | `TODO(FRAME-IMPL)` | cpu 参考实现(朴素三重循环)已注册(`src/backends/cpu/kernels/matmul.cpp`,REUSE-011 仅豁免 cpu 参考路径);**性能路径必须走 oneMKL/oneDNN(REUSE-011:不得复用 cpu 参考实现的手写循环)**,HAL 五接口实化时机见后端矩阵。参考:`docs/architecture/operator-system.md` 第4章;`docs/standards/reuse-policy.md` REUSE-011。完成判据:`FRAME_REGISTER_KERNEL("matmul", "intel_gpu", fn)` 落地(经 oneMKL/oneDNN)且与 cpu 参考实现数值一致(BUILD-011 容差)。 |
| square | `TODO(FRAME-IMPL)` | cpu 参考实现已注册(`src/backends/cpu/kernels/elementwise.cpp`);另有 decomposition(square(x)=mul(x,x),见 `src/ops/schemas/elementwise.cpp::square_decompose`,M10 回退链素材,不替代本行 kernel);intel_gpu kernel 待接入,HAL 五接口实化时机见后端矩阵。参考:`docs/architecture/operator-system.md` 第4章。完成判据:`FRAME_REGISTER_KERNEL("square", "intel_gpu", fn)` 落地且与 cpu 参考实现数值一致(BUILD-011 容差)。 |
| constant | 已支持(cpu) | cpu 参考实现已注册(`src/backends/cpu/kernels/constant.cpp`,M8);0 输入 1 输出,attrs=value/shape/dtype,常量物化。参考:`docs/architecture/operator-system.md` 第4章。intel_gpu 实现待接入。 |
| selective_scan | TODO(FRAME-IMPL): 说明:接入 intel_gpu kernel。参考:`docs/plan/2026-07-23-batch6-m25-ssm.md`。完成判据:注册 kernel 并按 BUILD-011 与 cpu 数值一致。 | cpu 参考实现已注册。 |
| heaviside_surrogate | TODO(FRAME-IMPL): 说明:接入 intel_gpu kernel。参考:`docs/plan/2026-07-23-batch8-m27-snn.md`。完成判据:注册 kernel 并按 BUILD-011 与 cpu 数值一致。 | cpu 参考实现已注册。 |
| scatter_add | TODO(FRAME-IMPL): 说明:接入 intel_gpu kernel。参考:`docs/plan/2026-07-23-batch9-m28-gnn.md`。完成判据:注册 kernel 并按 BUILD-011 与 cpu 数值一致。 | cpu 参考实现已注册。 |

---

## 7. 测试要求

- 冒烟与数值测试:`tests/cpp/backends/test_intel_gpu_*.cpp`;oneDNN/oneMKL 路径与手写路径分别与 CPU 参考实现(ARCH-041)做数值对比,容差用 `standards/build-and-test.md`(BUILD-011);后端可加严不可放松。
- HAL 一致性:Intel GPU 后端注册后必须能通过 `tests/cpp/hal_conformance/` 的通用接口行为测试。
- 无 GPU 行为:`sycl-ls` 检测不到 Intel GPU 时用 `GTEST_SKIP()` 并输出原因(BUILD-010)。
- 有 GPU 不得跳过:在 Intel GPU 可用的环境中,Intel GPU 测试不得 `GTEST_SKIP`,必须真实执行(M24 政策);仅在当前环境缺设备时允许 SKIP 并在 PR 说明。

---

## 8. 已知限制与待查证清单

- 骨架阶段:`src/backends/intel_gpu/` 仅为 HAL 桩,函数体 `return FRAME_UNIMPLEMENTED();`,不含真实 kernel。相关实现以 `TODO(FRAME-IMPL)` 标注。
- 【待查证】支持 C++20 与目标 Intel GPU 的 oneAPI Base Toolkit 最低版本 —— 来源:Intel oneAPI Base Toolkit 发行说明。
- 【待查证】oneDNN/oneMKL 支持目标 Intel GPU 的最低版本 —— 来源:Intel oneDNN / oneMKL 官方文档。
- 【待查证】目标 Intel GPU 用户态计算驱动安装方式与最低版本 —— 来源:Intel GPU 驱动安装文档(intel/compute-runtime)。
- 【待查证】`icpx` 与系统 g++ 的 CMake 混合构建配置 —— 来源:Intel oneAPI DPC++/C++ Compiler Developer Guide 与 IntelSYCL CMake 集成文档。
- 【待查证】oneDNN/oneMKL 的 CMake 导入目标名 —— 来源:oneDNN / oneMKL CMake 集成文档。
- 【待查证】`sycl::get_native` 到 Level Zero 的互操作细节与生命周期约束 —— 来源:SYCL 2020 规范与 oneAPI Level Zero 文档。
- 【待查证】oneDNN 文档当前权威 URL(第 9 章所列链接的时效性)—— 来源:oneDNN 官方仓库 README。

---

## 9. 权威参考文档

- Intel oneAPI 开发者门户:https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html
- Intel oneAPI DPC++/C++ Compiler(icpx)文档:https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html
- SYCL 2020 规范(Khronos):https://www.khronos.org/sycl/
- oneDNN 文档:https://uxlfoundation.github.io/oneDNN/ 【待查证】当前权威 URL —— 来源:oneDNN 官方仓库 README
- oneMKL 文档:https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html
- oneAPI Level Zero 规范:https://spec.oneapi.io/level-zero/latest/index.html
