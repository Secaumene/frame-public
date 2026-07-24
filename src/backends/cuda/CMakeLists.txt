# CUDA(NVIDIA GPU)后端:仅当 FRAME_BACKEND_CUDA_ENABLED 时由 src/CMakeLists.txt 加入。
# CUDA 语言在顶层 CMakeLists.txt 受选项保护 enable_language(CUDA)(评审勘误 8)。
# 后端 target 一律经 frame_add_backend() 创建(BKD-001),禁止手写 add_library。
# matmul/constant/fused_elementwise/conv/pool/shape/sequence/gather/fft 九个
# kernel 文件不含 __global__ 代码(纯 host 端 cuBLAS/cuDNN/cuFFT 调用/
# cudaMemcpy(Async) 编排/组合调用转发/D2H 预检),故为 .cpp 而非 .cu(与
# elementwise.cu/reduction.cu/pool.cu/shape.cu/sequence.cu/gather.cu 六个真正
# 含 __global__ kernel 的文件区分)。M22(批4 T4)新增 sequence.cpp(softmax,
# cudnnSoftmaxForward)/sequence.cu(layer_norm 自写 kernel)/shape.cu(transpose
# 自写 kernel)/gather.cpp(indices D2H 预检共用工具)/gather.cu(gather/
# gather_grad_internal 自写 kernel)。M23(批5 T3 接线 CUDA::cufft、T4 落地)
# 新增 kernels/fft.cpp(rfft/irfft,cufftPlanMany rank=1 R2C/C2R,ADR-0022 决策
# 1/2)。
#
# cuDNN 系统安装探测(M21,ADR-0021 决策 6,REUSE-012):cuDNN 不随 CUDAToolkit
# find_package 分发组件,故不走 cmake/frame_dependencies.cmake 的三态降级
# 通路,改以 find_path/find_library 直接探测 apt cudnn9-cuda-13-3 系统安装
# (本机核实记录见 docs/plan/2026-07-18-batch3-m21-conv.md 第 0 节);找不到即
# FATAL_ERROR(cuda 后端一旦启用即硬依赖 cuDNN,不做「找不到就禁用卷积」这类
# 部分降级)。产出的 CUDNN::cudnn 仅在本文件内消费,不外泄到核心层(铁律 3)。
find_path(FRAME_CUDNN_INCLUDE_DIR
  NAMES cudnn.h
  PATHS /usr/include/x86_64-linux-gnu /usr/include
)
find_library(FRAME_CUDNN_LIBRARY
  NAMES cudnn
  PATHS /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu
)
if(NOT FRAME_CUDNN_INCLUDE_DIR OR NOT FRAME_CUDNN_LIBRARY)
  message(FATAL_ERROR
    "frame cuda backend: cuDNN 9 not found (required by ADR-0021 for conv/pool kernels). "
    "Install the system package matching your CUDA Toolkit version, e.g. on Ubuntu: "
    "apt install cudnn9-cuda-13-3. Searched for header 'cudnn.h' and library 'cudnn' under "
    "/usr/include(/x86_64-linux-gnu) and /usr/lib(/x86_64-linux-gnu) / /lib/x86_64-linux-gnu.")
endif()
add_library(CUDNN::cudnn INTERFACE IMPORTED)
target_include_directories(CUDNN::cudnn INTERFACE ${FRAME_CUDNN_INCLUDE_DIR})
target_link_libraries(CUDNN::cudnn INTERFACE ${FRAME_CUDNN_LIBRARY})

frame_add_backend(
  NAME cuda
  SOURCES
    cuda_status.cpp
    cuda_backend.cpp
    cuda_allocator.cpp
    cuda_executable.cpp
    kernels/elementwise.cu
    kernels/reduction.cu
    kernels/matmul.cpp
    kernels/constant.cpp
    kernels/fused_elementwise.cpp
    kernels/conv.cpp
    kernels/pool.cpp
    kernels/pool.cu
    kernels/shape.cpp
    kernels/shape.cu
    kernels/sequence.cpp
    kernels/sequence.cu
    kernels/gather.cpp
    kernels/gather.cu
    kernels/fft.cpp
  LINK CUDA::cudart CUDA::cublas CUDA::cublasLt CUDNN::cudnn CUDA::cufft
)
