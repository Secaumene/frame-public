# Intel GPU(oneAPI/SYCL)后端:仅当 FRAME_BACKEND_INTEL_GPU_ENABLED 时加入(需 icpx)。
# 后端 target 一律经 frame_add_backend() 创建(BKD-001),禁止手写 add_library。
frame_add_backend(
  NAME intel_gpu
  SOURCES
    intel_gpu_backend.cpp
    intel_gpu_allocator.cpp
    kernels/elementwise.cpp
)
# TODO(FRAME-IMPL): SYCL 需对 target 调用 add_sycl_to_target(TARGET frame_backend_intel_gpu ...)
#   (IntelSYCL 提供的函数,而非普通链接目标)以注入 -fsycl 与设备编译。参考:
#   docs/backends/intel-gpu.md。完成判据:intel-gpu preset 构建成功且 .cpp 以 SYCL 设备编译。
