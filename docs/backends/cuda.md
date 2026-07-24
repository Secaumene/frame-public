# CUDA 后端指南

> **强制等级**:规范(MUST)
> **相关铁律**:#1 编译优先 / #3 后端矩阵 / #5 两级复用
> **面向读者**:实现代码的工程师
> **最后更新**:2026-07-23(M25–M28 已收口；完整 CUDA 设备门禁通过)

本文档按 `backends/README.md`(BE-001)定义的九章模板撰写。注册名字符串为 `"cuda"`,后端目录 `src/backends/cuda/`。所有不确定信息按 BE-000 用【待查证】标注并在第 8 章集中登记。

---

## 1. 定位与适用范围

- 目标硬件:NVIDIA GPU。注册名 `"cuda"`,`FRAME_REGISTER_BACKEND("cuda", CudaBackend)`。
- 执行模式:逐 kernel 拼装 + 自研图编译。编译路径(`Backend::compile` 产出 `Executable`)是默认路径(ARCH-010);eager 逐算子仅用于调试、回退与单算子测试(ARCH-011)。
- 复用取向:优先调用 NVIDIA 成熟库(cuBLAS/cuBLASLt、cuDNN、CUTLASS、CUB),手写 kernel 受严格准入约束(见第 3、6 章)。
- 不适用场景:非 NVIDIA GPU;需要绕过 HAL 直接暴露 `cudaStream_t` 给核心层的用法(ARCH-030 禁止上层调用 `native_handle`)。

---

## 2. 环境要求

- CUDA Toolkit:**≥ 12.0**(基线由 ADR-0004 裁决:C++20 为项目 C++ 标准,nvcc 自 12.0 起支持 C++20;`CMAKE_CUDA_STANDARD` 设为 `20`)。
  - `cudaMallocAsync` / `cudaMemPool_t` 内存池 API 需 CUDA ≥ 11.2,在 ≥ 12.0 基线下自然满足,无需额外条件判断。
  - 实测:CUDA Toolkit 13.3.73(nvcc V13.3.73),位于 `/usr/local/cuda`;nvcc 不在默认 PATH,构建须 `PATH=/usr/local/cuda/bin:$PATH`。
- 驱动:与所用 CUDA Toolkit 配套的 NVIDIA 驱动版本。实测:驱动 610.43.02 与 CUDA Toolkit 13.3.73 兼容。
- 环境变量:`CUDA_HOME` / `CUDA_PATH`(部分环境需要);`CUDA_VISIBLE_DEVICES` 用于测试时限定可见设备。
- 设备检测命令(可直接执行):

  ```bash
  nvidia-smi              # 列出 GPU、驱动版本、显存
  nvcc --version          # 查看 CUDA 编译器版本(确认 ≥ 12.0)
  ```

---

## 3. 复用库清单与复用边界

复用优先级(决策顺序):**cuBLAS / cuBLASLt → cuDNN → CUTLASS → CUB**。即先用 BLAS/GEMM 库,再用神经网络原语库,有融合需求才用 CUTLASS 模板,归约/扫描类用 CUB。

| 栏位 | 库 | 覆盖范围 |
|---|---|---|
| 必用(MUST) | cuBLAS / cuBLASLt | GEMM、matmul、批量矩阵乘 |
| 必用(MUST) | cuDNN | 卷积、pooling、归一化、激活、attention 等神经网络原语 |
| 必用(MUST) | CUB | reduction、scan、sort、histogram 等并行原语 |
| 可用(SHOULD) | CUTLASS | 需要与 GEMM 融合的自定义算子(epilogue 融合、隐式 GEMM) |
| 可用(可选) | Thrust | 仅工具代码 / 测试代码中的容器与算法 |
| 禁止自研(MUST NOT) | —— | 与上述库功能重叠的手写 GEMM / 卷积 / 通用归约 kernel |

- 【BE-CUDA-001】【MUST】GEMM 一律先用 cuBLAS/cuBLASLt;卷积/归一化/pooling/attention 优先 cuDNN;需算子融合的 GEMM 用 CUTLASS 模板;reduction/scan/sort 用 CUB。判定方法:review 检查 `src/backends/cuda/` 中是否存在与上述库功能重叠的手写实现;新增此类实现无第 6 章准入依据即打回。
- 【BE-CUDA-002】【MUST NOT】禁止手写与 cuBLAS/cuBLASLt/cuDNN/CUTLASS/CUB 功能重叠的 kernel。手写 kernel 仅限:elementwise、项目自定义算子、库覆盖不到的融合三类;手写前须在 PR 描述粘贴「库不满足」的具体依据(缺哪个能力 / 性能实测差距)。判定方法:PR 复用检查段(REUSE-001)必须列出对上述库的排查结论。
- 已裁决(M21,ADR-0021 决策 2):cuDNN v1 取 legacy 即时 API(9.24 实测符号全部导出,graph API 并存);迁移触发条件=未来版本 nm 探测不到 legacy 符号,届时新 ADR 迁 graph API。cuDNN 已引入(ADR-0021,销 ADR-0011 推迟)。
- 引入方式:CUDA Toolkit(含 cuBLAS/cuBLASLt/CUB/Thrust)走 `find_package`(SDK 自带,REUSE 引入方式优先级第一档);cuDNN 非 CUDAToolkit find_package 组件,经 `find_path`/`find_library` 探测系统安装(ADR-0021 决策 6,已落 src/backends/cuda/CMakeLists.txt);CUTLASS 为 header-only,经 CMake `FetchContent` 锁定版本 tag 引入——CUTLASS 属 reuse-policy「已批准」档,但**首次接入仍须按 REUSE-010 提交 ADR**(该 ADR 只需记录版本锁定与引入方式);其他新增依赖同样先 ADR。

---

## 4. HAL 映射表

HAL 接口清单以 `architecture/backend-hal.md`(ARCH 系列)为准。`Device` 为值类型 `{backend, index}`(BE-002),设备管理操作落在 `Backend` 上。错误一律转为 `Result<T>` / `Status`(`include/frame/core/status.h`),不跨 HAL 边界抛异常(CPP-020)。

| HAL 接口 / 操作 | CUDA 原生 API |
|---|---|
| Backend 设备枚举 / 选择 | `cudaGetDeviceCount` / `cudaSetDevice` |
| Backend 设备属性 | `cudaGetDeviceProperties` |
| Stream | `cudaStream_t`(`cudaStreamCreate` / `cudaStreamDestroy`) |
| Stream::synchronize | `cudaStreamSynchronize` |
| Event | `cudaEvent_t`(`cudaEventCreate` / `cudaEventRecord`) |
| Event::synchronize / query | `cudaEventSynchronize` / `cudaEventQuery` |
| Stream 等待 Event | `cudaStreamWaitEvent` |
| Allocator::allocate / deallocate | `cudaMalloc` / `cudaFree`(显存驻留分配。历史:v0 曾临时用 `cudaMallocManaged` 迁就一致性套件对分配指针的 host 端直接读写;套件已泛化为经 `Backend::copy` 中转校验后回切——该 FRAME-DESIGN 待办已按完成判据销项,cuda preset 全绿) |
| memcpy(H2D/D2H/D2D) | `cudaMemcpyAsync` |
| 错误转换 | `CUDA_CHECK` 宏:`cudaError_t` → `Status`(消息一律英文,LANG-005) |

- 【待查证】`cudaMemPool_t` 池的 release threshold 与跨 Stream 复用的默认策略 —— 来源:NVIDIA CUDA C++ Programming Guide(Stream Ordered Memory Allocator)。

---

## 5. 编译接入

- CMake 选项:`FRAME_ENABLE_CUDA`,三态 `AUTO|ON|OFF`,默认 `AUTO`(语义见 BUILD 系列)。
- SDK 探测:`find_package(CUDAToolkit)`(要求 ≥ 12.0)。
- 语言启用:`enable_language(CUDA)` 只在顶层 `CMakeLists.txt` 中受 `FRAME_ENABLE_CUDA` 选项保护调用(未启用 CUDA 时不得进入 CUDA 语言启用分支)。设置 `CMAKE_CUDA_STANDARD 20` 与 `CMAKE_CUDA_STANDARD_REQUIRED ON`。
- 架构选择:`CMAKE_CUDA_ARCHITECTURES` 可设为 `native`(自动检测当前机器 GPU 架构)或显式指定如 `"compute_120;compute_121"`。CMakePresets 中 `cuda` preset 设为 `"native"`,该值机器相关;跨机器构建需显式设定目标 compute capability(如 `"70;80;90"`);实测 GPU(RTX 5070 Ti Laptop,Blackwell)的 compute capability 为 12.0,对应 `compute_120`。查询当前机器支持的架构:执行 `nvcc --list-gpu-arch`。
- 链接目标:后端库目标 `frame_backend_cuda`,链接 `CUDA::cudart`、`CUDA::cublas`、`CUDA::cublasLt`;CUB/Thrust 随 Toolkit(CCCL)头文件提供。
  - 已核实(M21,ADR-0021 决策 6):Toolkit 不提供 `find_package(CUDNN)` 查找模块;本仓经 `find_path(cudnn.h)`/`find_library(cudnn)` 探测系统安装并自建 `CUDNN::cudnn` INTERFACE IMPORTED target(src/backends/cuda/CMakeLists.txt)。
- Preset:`cuda`(配套 configure/build/test presets)。骨架验收口径为 `cpu-only`,不依赖 CUDA。
- 后端代码用 CMake 目标隔离,`src/{core,ir,compiler,runtime}` 中禁止出现 `#ifdef FRAME_ENABLE_CUDA`(BUILD-003)。

---

## 6. 算子/内核开发规范

- 目录:`.cu` kernel 源码放 `src/backends/cuda/kernels/`;后端接入/注册代码放 `src/backends/cuda/`。
- 注册:`FRAME_REGISTER_KERNEL(op, "cuda", fn)`(kernel 注册键不含 dtype)。
- dtype 分发:dtype 差异在 kernel 内经 `dispatch_dtype` 编译期展开(模板 + `if constexpr`),禁止在 kernel 内层循环出现按 dtype 或按设备的运行时分支(ARCH-042)。判定方法:`check_iron_rules.sh` 对 `src/backends/cuda/kernels/` 扫描 dtype 运行时分支。
- 命名:kernel 与工具函数 `snake_case`;`__global__` 函数命名体现算子与 dtype 模板参数;launch 配置(grid/block 计算)集中在工具函数,不在每个 kernel 内重复。
- 【BE-CUDA-003】【MUST】CUDA 后端源码按 C++20 + CUDA 20 标准编译;不得为规避编译期机制而回退到虚函数或运行时 dtype 分支(呼应 CPP-010、CPP-014、ARCH-042)。判定方法:`CMAKE_CUDA_STANDARD` 值为 `20`;`grep -rn 'virtual' src/backends/cuda/` 命中项必须属于 HAL 接口实现(白名单)。
- 手写准入:仅 elementwise / 自定义算子 / 库覆盖不到的融合三类(BE-CUDA-002),准入依据写入 PR。

### 已支持算子表

新增/补齐算子时维护本表;未实现项按 `TODO(FRAME-IMPL)` 标注,静默缺失即违例。
`fused_elementwise_internal`(M9 融合 pass 产物,非用户构图算子,不入本表;cpu 与 cuda 组合调用均已支持,参考 docs/architecture/compiler-passes.md §3.7)。

| 算子 | 状态 | 实现方式/备注 |
|---|---|---|
| add | 已支持(M11) | cpu 参考实现与 cuda kernel 已注册(`src/backends/cpu/kernels/elementwise.cpp` / cuda kernel);HAL 五接口实化于 M11。参考:`docs/architecture/operator-system.md` 第4章。 |
| mul | 已支持(M11) | cpu 参考实现与 cuda kernel 已注册(`src/backends/cpu/kernels/elementwise.cpp` / cuda kernel);HAL 五接口实化于 M11。参考:`docs/architecture/operator-system.md` 第4章。 |
| relu | 已支持(M11) | cpu 参考实现与 cuda kernel 已注册(`src/backends/cpu/kernels/elementwise.cpp` / cuda kernel);HAL 五接口实化于 M11。参考:`docs/architecture/operator-system.md` 第4章。 |
| sum | 已支持(M11) | cpu 参考实现与 cuda kernel 已注册(`src/backends/cpu/kernels/reduction.cpp` / cuda kernel);HAL 五接口实化于 M11。参考:`docs/architecture/operator-system.md` 第4章。 |
| matmul | 已支持(M11;M19 升级 cublasLt) | cpu 参考实现(朴素三重循环)与 cuda 性能路径已注册;M19 起三精度(fp32/fp16/bf16)统一经 cublasLtMatmul,fp32 可经 CompileOptions::allow_tf32 启用 TF32(ADR-0019,默认关)。参考:`docs/architecture/operator-system.md` 第4章;`docs/standards/reuse-policy.md` REUSE-011;ADR-0019。 |
| square | 已支持(M11) | cpu 参考实现、decomposition(square(x)=mul(x,x),M10 回退链素材)、cuda kernel 已注册;HAL 五接口实化于 M11。参考:`docs/architecture/operator-system.md` 第4章。 |
| constant | 已支持(M11) | cpu 与 cuda 参考实现已注册(`src/backends/cpu/kernels/constant.cpp` / cuda kernel,M8 起 cpu,M11 起 cuda);0 输入 1 输出,attrs=value/shape/dtype,常量物化。参考:`docs/architecture/operator-system.md` 第4章。 |
| conv2d | 已支持(M21) | cuda 经 cuDNN legacy 即时 API(ConvolutionForward + bias 时 AddTensor 通道广播;反向 conv2d_grad_{input,filter}_internal 经 BackwardData/Filter,算法 Get*Algorithm_v7 启发式首选);三精度 computeType 恒 FLOAT,fp32 可经 CompileOptions::allow_tf32 启用 TF32(实测生效,max_rel 1.75e-4);cpu 参考直循环。参考:ADR-0021;`docs/plan/2026-07-18-batch3-m21-conv.md` 第 1.3 节。 |
| conv1d | 已支持(M21) | cpu 参考直循环;cuda 无专属 kernel,经 decomposition(reshape→conv2d(H=1)→reshape)落 cuDNN(批3裁决点②,回退链②机制)。参考:ADR-0021。 |
| max_pool2d | 已支持(M21) | cuda 前向 cudnnPoolingForward;反向 max_pool2d_grad_internal(kernel 内重算 y 后 PoolingBackward)与 max_pool2d_select_internal(自写 kernel);argmax 平局约定=窗口最低线性索引,与 cpu 参考钉死一致。参考:ADR-0021。 |
| avg_pool2d | 已支持(M21) | cuda 前向 cudnnPoolingForward(COUNT_INCLUDE_PADDING,分母恒 KH·KW 与 cpu 同口径);反向 avg_pool2d_grad_internal 自写均匀回撒 kernel(无原子)。参考:ADR-0021。 |
| reshape | 已支持(M21) | cpu memcpy / cuda cudaMemcpyAsync D2D(经 ctx.stream);attr target_shape(kShape),梯度=reshape 回原形。 |
| sigmoid | 已支持(M21) | cpu 参考(数值稳定式)与 cuda 向量化 elementwise kernel;暂不参与融合(TODO(FRAME-PERF) 记录于 schema 注册处)。 |
| tanh | 已支持(M22) | cpu 参考与 cuda 向量化 elementwise kernel(标量+向量路径);梯度微图 gy·(1−t²) 从 x 重算(sigmoid CSE 前例)。参考:批4计划 §1.2/1.6。 |
| rsqrt | 已支持(M22) | spec 外增项(layer_norm 梯度需 1/√(σ²+ε),设计门核准);cpu 参考与 cuda 向量化 elementwise kernel;梯度 gy·(−0.5)·r³ 从 x 重算。参考:批4计划 §1.2。 |
| softmax | 已支持(M22) | rank-2 末轴;cuda 经 cudnnSoftmaxForward(ACCURATE+MODE_INSTANCE,[N,D]→N,C=D,H=W=1),ADR-0021 决策 2 增补面,圈禁 kernels/sequence.cpp;cpu 参考减行最大值稳定式;梯度=纯公开算子微图(行广播=sum→reshape→matmul(ones) 三连)。 |
| layer_norm | 已支持(M22) | rank-2 末轴,γ/β [D] 图内广播,attr eps;cuda 自写 kernels/sequence.cu(一行一块 256 线程两趟 float 归约);cpu 参考逐式同构;梯度三输出(gx/ggamma/gbeta)纯公开算子微图。 |
| transpose | 已支持(M22) | attr perm(任意秩);cuda 步长拷贝 kernels/shape.cu;梯度=transpose(gy, inverse_perm) 自伴。 |
| concat | 已支持(M22) | variadic_input(min_count=1,单输入退化=恒等,slice 全幅梯度所需);attr axis;cuda 宿主 cudaMemcpyAsync 编排(kernels/shape.cpp);梯度=逐输入 slice(concat↔slice 互逆伴随对)。 |
| slice | 已支持(M22) | 单轴连续 [start,stop)(kInt64 attrs);cuda 宿主 cudaMemcpyAsync 编排(kernels/shape.cpp);梯度=concat(前零,gy,后零)(零宽段省略)。 |
| gather | 已支持(M22) | x[N,F] + indices[K](int32/int64,@input1)沿 axis=0 → [K,F];cuda kernels/gather.cu 行拷贝,越界=启动前 D2H 预检 fail-loud 含实际值(ARCH-031;M28 已知性能项,见 spec §11);cpu 逐元素校验;梯度 wrt x=gather_grad_internal,wrt indices=constant(0) 整数 splat(ARCH-063 全位置)。 |
| gather_grad_internal | 已支持(M22) | scatter-add(重复索引累加,attr input_shape);cuda atomicAdd(fp16/bf16 原生半精度原子,CC≥7.0/8.0,与 cpu float 缓冲参考的累加精度差异由 BUILD-011 容差覆盖);自身可微(wrt gy=gather 回取,wrt indices=int 零 splat,R11 闭包唯一新内部算子)。 |
| rfft | 已支持(M23) | rank≥1 末轴 FFT,输入 [...,n],输出 [...,k,2](k=n/2+1);末轴交错布局(re/im);dtype 限 fp32(fail-loud 非 fp32);cuda 经 cuFFT 12.3.0 cufftPlanMany R2C(圈禁 kernels/fft.cpp + cufft_utils.h);cpu 参考 pocketfft r2c(BSD-3,Max-Planck-Society);梯度纯公开微图=irfft(gy⊙(1/w),n)·n(Hermitian 掩码 w,无专属反向 kernel,R11 闭包与 irfft 互引)。参考:ADR-0022;批5计划 §1.2/1.3。 |
| irfft | 已支持(M23) | 属性 n(kInt64);输入 [...,k,2],输出 [...,n],shape 校验 k=n/2+1 末轴=2;dtype 限 fp32;cuda 经 cuFFT cufftPlanMany C2R(C2R 会破坏输入缓冲故 kernel 内先拷贝,cufftSetStream 绑执行流)+标量缩放 1/n;cpu 参考 pocketfft c2r(含 1/n 归一化);梯度纯公开微图=(w/n)⊙rfft(gx)(R11 闭包,互引用 rfft);irfft(rfft(x))≡x 数学恒等。参考:ADR-0022;批5计划 §1.2/1.3。 |
| selective_scan | 已支持(M25；设备验收通过) | 五输入同 shape/dtype，沿最后一轴的状态递推；cpu 逐行参考，cuda 以 CUB 前缀仿射扫描实现，fp16/bf16 以 fp32 中间量累计。CUDA Toolkit 13.3 位于 `/usr/local/cuda-13.3`，构建使用显式根路径。 |
| heaviside_surrogate | 已支持(M27；设备验收通过) | 前向逐元素阶跃，反向为受限平滑代理微图；cpu 参考与 cuda elementwise kernel 已接线。 |
| scatter_add | 已支持(M28；设备验收通过) | 与 gather_grad_internal 共用散加核心；indices 越界先作 D2H 预检并 fail-loud。cuda 重复索引经 atomicAdd 累加，顺序不保证位级确定，只按 BUILD-011 容差与 CPU 参考比较。 |

---

## 7. 测试要求

- 冒烟与数值测试:`tests/cpp/backends/test_cuda_*.cpp`;每个注册 kernel 与 CPU 参考实现(ARCH-041)做数值对比,容差用 `standards/build-and-test.md`(BUILD-011,fp32 rtol=1e-5/atol=1e-6,fp16 rtol=1e-2/atol=1e-3,bf16 rtol=2e-2/atol=2e-3);后端可加严不可放松。
- HAL 一致性:CUDA 后端注册后必须能通过 `tests/cpp/hal_conformance/` 的通用接口行为测试。
- 无 GPU 行为:检测不到 CUDA 设备时用 `GTEST_SKIP()` 并输出原因(BUILD-010)。
- 有 GPU 不得跳过:在 CUDA 设备可用的环境中,CUDA 测试不得 `GTEST_SKIP`,必须真实执行(M24 政策);仅在当前环境缺设备时允许 SKIP 并在 PR 说明。
- **M25–M28 最终设备证据(2026-07-23，含官方 C++ 示例)**:
  `ctest --preset cuda` 1152/1152 通过；`CudaBackendTest.*` 直测 142/142
  通过且零 SKIP；CUDA 扩展下 Python 89 passed，2 项因可选 `onnx` 模块
  未安装而 SKIP。

---

## 8. 已知限制与待查证清单

- **实现状态(M11)**:HAL 五接口(Backend/Stream/Event/Allocator/Executable)已实现;七个核心算子(add/mul/relu/sum/square/constant)的 cuda kernel 已接入;matmul 已通过 cuBLAS 实现高性能路径。后端基本功能完成,测试覆盖见 `tests/cpp/backends/test_cuda_*.cpp` 与 `tests/cpp/hal_conformance/`。
- **实测环境**:CUDA Toolkit 13.3.73(nvcc V13.3.73,位于 `/usr/local/cuda`,nvcc 不在默认 PATH,构建须 `PATH=/usr/local/cuda/bin:$PATH`);驱动 610.43.02;GPU:NVIDIA GeForce RTX 5070 Ti Laptop GPU(Blackwell,compute capability 12.0,SM 数 46,显存约 11813 MiB);nvcc --list-gpu-arch 含 compute_120/compute_121。
- **M19 三精度 bench 落档(2026-07-18,ADR-0019 收益记录)**:
  基准口径: preset=bench; 机器=NVIDIA GeForce RTX 5070 Ti Laptop GPU(Blackwell,CC 12.0,驱动 610.43.02,CUDA 13.3.73)。
  单 matmul 方阵图,run() 计时含流同步,3 次重复取中位;相对 fp32 严格路径的加速比——TF32:1.253x(N=256)/1.134x(N=512)/1.243x(N=1024);fp16:1.365x/3.821x/1.600x;bf16:1.367x/3.789x/1.642x。绝对值与命令口径见 benchmarks/bench_matmul.cpp 头注释;数字仅代表本机,heuristic 结果设备相关。workspace=0 字节为 v0 口径,加大 workspace 的收益属既有 TODO(FRAME-PERF) 跟进项(src/backends/cuda/kernels/matmul.cpp)。
- **M21 conv2d bench 落档(2026-07-19,批3 T7,docs/plan/2026-07-18-batch3-m21-conv.md 第3节)**:
  基准口径: preset=bench; 机器=NVIDIA GeForce RTX 5070 Ti Laptop GPU(Blackwell,CC 12.0,驱动 610.43.02,CUDA 13.3.73,cuDNN 9.24.0.43)。
  单 conv2d 图(N=8, Cin=Cout=16, H=W=32, K=3, stride=1, padding=1,带 bias),3 次重复取中位。绝对时延(中位)——cpu fp32:9.566ms;cuda fp32 严格:211.2us;cuda fp32 allow_tf32:294.7us;cuda fp16:1067.4us;cuda bf16:870.4us。相对 fp32 严格路径的加速比——TF32:0.72x;fp16:0.20x;bf16:0.24x(本形状下三档均慢于严格 fp32,如实记录——cuDNN `cudnnGetConvolutionForwardAlgorithm_v7` 为 heuristic 而非穷举搜索,该结果设备/形状相关,不代表通用结论,不排除算法选择在该 heuristic 下对该小尺寸未选中最优 Tensor Core 路径)。cuda 全部四档相对 cpu 参考均显著更快(约 9x~45x),不构成"conv cuda 明显慢于 cpu"的异常信号。绝对值与命令口径见 benchmarks/bench_conv2d.cpp 头注释;数字仅代表本机。
- **cublasLt 配套核实(BE-000,2026-07-18,ADR-0019)**:`cublasLt.h` 位于 `/usr/local/cuda/include`(独立头,链接目标 `CUDA::cublasLt`,区别于 cublas_v2.h/CUDA::cublas);cuBLAS 13.3 文档确认 Tensor Core 自 CC 7.0+ 由库自动择优启用;sm_120(CC 12.0,本机 RTX 5070 Ti)各精度组合(TF32/FP16/BF16)**无逐条官方支持矩阵**,以本机一致性套件与 bench 实测为最终判定。来源:cuBLAS Documentation(https://docs.nvidia.com/cuda/cublas/,核实日期 2026-07-18)。
- **cuFFT 配套核实(BE-000,2026-07-21,R8 M23 先行条件)**:`cufft.h` 位于 `/usr/local/cuda/include`(版本 12.3.0,CUFFT_VER_MAJOR=12/MINOR=3/PATCH=0/BUILD=29,CUFFT_VERSION=12300);链接目标 `CUDA::cufft`(libcufft.so.12.3.0.29,约 245MB);sm_120(CC 12.0,本机 RTX 5070 Ti Laptop)运行时验证完全通过:cufftPlan1d(1D R2C,N=64) + cufftExecR2C 成功,输出数值有效(all-ones 输入 DC=64.0,符合预期);无架构兼容问题。来源:本机实测探针(probe_cufft_v2.cu,exit code=0);NVIDIA CUDA Toolkit Release Notes 13.3。
- 【待查证】cudaMalloc 对齐保证(实现按 256B 保守口径,见 `src/backends/cuda/cuda_allocator.cpp` 注释) —— 来源:NVIDIA CUDA C++ Programming Guide。
- 【待查证】cuDNN v9 Graph API 的调用序列与 legacy API 取舍 —— 来源:NVIDIA cuDNN Developer Guide。
- 【待查证】`cudaMemPool_t` 的 release threshold 与跨 Stream 复用默认策略 —— 来源:NVIDIA CUDA C++ Programming Guide(Stream Ordered Memory Allocator)。
- 【待查证】cuDNN 的 CMake 导入目标名与查找方式 —— 来源:NVIDIA cuDNN Installation Guide。

---

## 9. 权威参考文档

- NVIDIA CUDA Toolkit Documentation:https://docs.nvidia.com/cuda/
- CUDA C++ Programming Guide:https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA Toolkit Release Notes:https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
- cuBLAS / cuBLASLt Documentation:https://docs.nvidia.com/cuda/cublas/
- NVIDIA cuDNN Documentation:https://docs.nvidia.com/deeplearning/cudnn/
- CUTLASS(GitHub):https://github.com/NVIDIA/cutlass
- CCCL(CUB / Thrust,GitHub):https://github.com/NVIDIA/cccl
