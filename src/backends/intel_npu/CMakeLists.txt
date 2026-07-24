# Intel NPU(OpenVINO Runtime)后端:仅当 FRAME_BACKEND_INTEL_NPU_ENABLED 时加入。
# 后端 target 一律经 frame_add_backend() 创建(BKD-001),禁止手写 add_library。
frame_add_backend(
  NAME intel_npu
  SOURCES
    intel_npu_backend.cpp
    intel_npu_allocator.cpp
  LINK openvino::runtime
)
# TODO(FRAME-IMPL): 校验 OpenVINO 目标名与最低版本,并按 docs/backends/intel-npu.md 接入。
#   参考:docs/backends/intel-npu.md。完成判据:intel-npu preset 构建成功。
