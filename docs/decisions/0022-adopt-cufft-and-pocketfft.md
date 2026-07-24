# ADR-0022:采纳 cuFFT 承载 CUDA FFT 内核、pocketfft 承载 CPU 参考实现

- 状态:已接受
- 日期:2026-07-21
- 关联铁律:#3 后端隔离、#5 两级复用
- 关联规则:REUSE-010、BE-CUDA-002

## 背景

M23 频域批次交付 rfft/irfft 算子与谱卷积(FNO/PINO 前向)。FFT 属标准能力,
铁律 5 要求优先成熟库。BE-000 已核实(2026-07-21):cuFFT 12.3.0 随 CUDA
Toolkit 13.3.73 同装,Blackwell(CC 12.0)运行时探针(Plan1d R2C + ExecR2C)
全链路成功,R8 不触发。CPU 侧需一个无重依赖的参考实现作数值基准。

## 决策

CUDA 侧 rfft/irfft 内核经 **cuFFT**,CPU 参考实现经 **pocketfft**(单头文件,
BSD-3,numpy/pytorch 同源谱系)。要点:

1. cuFFT 经 `CUDA::cufft`(find_package(CUDAToolkit) 既有 imported target)
   直链,与 CUDA::cublasLt 同类同装先例;不自写 find 模块(cuDNN 的
   find_path 先例源于其独立安装包属性,对 toolkit 同装组件不适用)。
2. cuFFT 调用面圈禁:仅 src/backends/cuda/kernels/fft.cpp 与 cufft_utils.h
   (ADR-0021 决策 2 同款纪律)。判定方法:`grep -rln "cufft" src/ include/`
   命中 ⊆ 上述两文件 + 本批 CMake 链接行。
3. pocketfft 取自官方仓 github.com/mreineck/pocketfft **cpp 分支**,
   FetchContent 锁 immutable commit c90e55b3d529f8efa40ed01a20de22405f45fc65
   (pocketfft_hdronly.h sha256=
   3e9a05318d8e3b1446bda1c4617e6a103cdd23599ae0a776a92a6e8800e92fdc)。
   REUSE-012 合规:该仓无 Releases/Tags(核实来源:上游 Releases/Tags
   页面;核实日期:2026-07-21),第 2 级「锁 tag」不可行,锁 immutable
   commit 保全其「禁止跟踪分支/HEAD」的可复现意图,是无 tag 时的退档引入
   方式(非 GoogleTest/pybind11 锁 tag 之同族)。REUSE-013 退档理由:该
   分支无稳定发行版,所锁 commit(2026-06-30)为核实时点分支头。
   手动 INTERFACE target(不 add_subdirectory),仅链入 cpu kernels target。
4. pocketfft 调用面圈禁:仅 src/backends/cpu/kernels/fft.cpp。判定方法:
   `grep -rln "pocketfft" src/ include/` 命中 ⊆ 该文件 + CMake 声明处。
5. dtype 限 fp32(pocketfft 无半精度;cuFFT 半精度须另开 cufftXt API 面,
   YAGNI);fp16/bf16 FFT 记 v1_2-v1_3 spec §11 沉淀议题。
6. cuFFT plan v0 不缓存:每次调用按 (n, batch) 现建现毁并 cufftSetStream
   绑执行流;plan 缓存留性能批按 bench 证据评估(spec §11 在册)。

## 备选方案

- FFTW/cufftw:性能标杆,但 GPL(商业许可另购),许可面否决。
- KissFFT:BSD 但精度/性能弱于 pocketfft,且 numpy/pytorch 均已收敛到
  pocketfft 谱系,生态证据否决。
- 自研 FFT:违反铁律 5(标准能力必须用成熟库),否决。
- oneMKL FFT 作 CPU 参考:重依赖,且属 v2.0 Intel 后端窗口,本批否决。

## 后果

- 正面:CUDA/CPU 双路零手写 FFT 数值代码;打包布局(末轴 re/im 交错)与
  cufftComplex / std::complex<float> 逐字节一致,kernel 为纯库调用。
- 代价:FetchContent 新增一处网络依赖(锁哈希可复现);plan 现建现毁有
  每调用开销(已记沉淀议题);fp32 限定(已记沉淀议题)。
- 跟进:本 ADR「已接受」+ README 索引行随批5 T2 计划 commit 原子落地
  (ADR-011;设计门复审即 pocketfft 终选),T3 再落 CMake(REUSE-010
  先 ADR 后引入);BE-000 结果登记 docs/backends/cuda.md §8 随批5 T7。
