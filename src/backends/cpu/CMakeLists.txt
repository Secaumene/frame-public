# CPU 参考后端:永远启用,是其余后端的样板。
# 后端 target 一律经 frame_add_backend() 创建(BKD-001),禁止手写 add_library。
# frame_add_backend 负责:add_library(STATIC) + 别名、编译宏 FRAME_ENABLE_CPU=1、
# 链接 frame::core 与 frame::compiler_flags、追加到全局属性 FRAME_ENABLED_BACKENDS。
# kernels/fft.cpp(M23,批5 T3,ADR-0022):rfft/irfft 经 pocketfft 单头文件库
# 实现,LINK 该 INTERFACE target(仅 kernels/fft.cpp 消费,调用面圈禁见
# cmake/frame_dependencies.cmake 头注释);必须限于 BUILD_INTERFACE,避免静态
# 后端导出 $<LINK_ONLY:pocketfft> 到 INSTALL_INTERFACE。
frame_add_backend(
  NAME cpu
  SOURCES
    cpu_backend.cpp
    cpu_allocator.cpp
    cpu_executable.cpp
    kernels/elementwise.cpp
    kernels/reduction.cpp
    kernels/matmul.cpp
    kernels/constant.cpp
    kernels/fused_elementwise.cpp
    kernels/loss.cpp
    kernels/conv.cpp
    kernels/pool.cpp
    kernels/shape.cpp
    kernels/sequence.cpp
    kernels/gather.cpp
    kernels/fft.cpp
  LINK $<BUILD_INTERFACE:pocketfft>
)
