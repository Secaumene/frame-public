# =============================================================================
# frame_compiler_flags.cmake —— 编译告警/Sanitizer 接口 target 与配置摘要
#
# 用法契约(src/、tests/、python/ 各 CMakeLists.txt 必须遵守):
#   所有 Frame 自有 target 一律 target_link_libraries(<target> PRIVATE frame::compiler_flags)。
#
# 规则:
# 【FLAG-001】【MUST】自有 target 全部启用 -Wall -Wextra(MSVC 为 /W4)。
#   判定方法:本文件存在该两项编译选项,且 src/tests/python 各 target 均链接
#   frame::compiler_flags(grep 各子目录 CMakeLists.txt)。
# 【FLAG-002】【MUST】Debug 配置(dev/dev-asan/cpu-only 及继承 dev 的后端 preset)追加
#   -Werror(MSVC 为 /WX)。判定方法:本文件存在 $<CONFIG:Debug> 包裹的 -Werror。
# 【FLAG-003】【MUST】FRAME_ENABLE_ASAN=ON 时编译与链接均加 -fsanitize=address。
#   判定方法:dev-asan preset 配置后 compile_commands.json 含 -fsanitize=address。
# =============================================================================

include_guard(GLOBAL)

add_library(frame_compiler_flags INTERFACE)
add_library(frame::compiler_flags ALIAS frame_compiler_flags)

# 告警只挂 CXX 翻译单元,避免泄漏到 CUDA(nvcc 不识别部分 GCC 告警旗标)
if(MSVC)
  target_compile_options(frame_compiler_flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
  target_compile_options(frame_compiler_flags INTERFACE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/WX>)
else()
  target_compile_options(frame_compiler_flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>)
  target_compile_options(frame_compiler_flags INTERFACE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-Werror>)
endif()

# TODO(FRAME-IMPL): CUDA 翻译单元的宿主端告警透传(-Xcompiler=-Wall,-Wextra)。
#   参考: cmake/frame_backend.cmake。完成判据: cuda preset 配置成功且 .cu 编译命令含 -Xcompiler。

# ---- AddressSanitizer 接线(仅配 dev-asan preset 使用) ----
if(FRAME_ENABLE_ASAN)
  if(MSVC)
    target_compile_options(frame_compiler_flags INTERFACE /fsanitize=address)
  else()
    target_compile_options(frame_compiler_flags INTERFACE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(frame_compiler_flags INTERFACE -fsanitize=address)
  endif()
endif()

# ---- 配置摘要(顶层 CMakeLists.txt 在配置末尾必须调用;格式固定,便于 CI 断言) ----
# 各 FRAME_*_SUMMARY 变量由 frame_dependencies.cmake 在探测时写入顶层作用域。
function(frame_print_configuration_summary)
  message(STATUS "Frame configuration summary")
  message(STATUS "  C++ standard      : ${CMAKE_CXX_STANDARD}")
  message(STATUS "  Backend  CPU      : ON (always)")
  message(STATUS "  Backend  CUDA     : ${FRAME_BACKEND_CUDA_SUMMARY}")
  message(STATUS "  Backend  IntelGPU : ${FRAME_BACKEND_INTEL_GPU_SUMMARY}")
  message(STATUS "  Backend  IntelNPU : ${FRAME_BACKEND_INTEL_NPU_SUMMARY}")
  message(STATUS "  Backend  Ascend   : ${FRAME_BACKEND_ASCEND_SUMMARY}")
  message(STATUS "  MLIR              : ${FRAME_MLIR_SUMMARY}")
  message(STATUS "  Python bindings   : ${FRAME_PYTHON_SUMMARY}")
  message(STATUS "  Tests / Examples  : ${FRAME_BUILD_TESTS} / ${FRAME_BUILD_EXAMPLES}")
  message(STATUS "  ASAN              : ${FRAME_ENABLE_ASAN}")
endfunction()
