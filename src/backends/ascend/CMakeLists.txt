# 昇腾 NPU(CANN/AscendCL)后端:仅当 FRAME_BACKEND_ASCEND_ENABLED 时加入。
# 后端 target 一律经 frame_add_backend() 创建(BKD-001),禁止手写 add_library。
# CANN::ascendcl 由自写探测模块 cmake/find_cann.cmake 产出。
frame_add_backend(
  NAME ascend
  SOURCES
    ascend_backend.cpp
    ascend_allocator.cpp
  LINK CANN::ascendcl
)
# TODO(FRAME-IMPL): 依昇腾执行模式裁决(ADR-0005)接入 AscendCL。参考:docs/backends/ascend.md。
#   完成判据:ascend preset 构建成功。
