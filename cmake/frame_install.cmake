# =============================================================================
# frame_install.cmake —— C++ 库/头文件/CMake 包配置的安装与导出(系统安装)
#
# 用途:
#   `cmake --install build/<preset> [--prefix <dir>]` 安装 Frame 的 C++ 静态库、
#   公共头文件与 CMake 包配置;外部工程经 `find_package(frame REQUIRED)` +
#   `target_link_libraries(... frame::frame)` 消费(消费方需 CMake >= 3.24,
#   缘由见 cmake/frame-config.cmake.in 的版本校验)。
#
# 规则出处(docs/standards/build-and-test.md 第 10 节):
#   【BUILD-040】全部安装/导出规则集中于本文件,受 FRAME_INSTALL_CPP 门控;
#     导出包名 frame、命名空间 frame::、export set 名 frame_targets。
#   【BUILD-041】wheel 与系统安装互不携带对方产物——本文件只装 C++ 产物,
#     Python 扩展 _core 的 wheel 专属安装规则见 python/CMakeLists.txt 的
#     if(SKBUILD) 门控,两者互不重叠。
#
# 调用时机:顶层 CMakeLists.txt 在 add_subdirectory(src) 之后 include 本文件,
# 此时全部 src/ 下 target 已定义,export set 才能收全(见该文件内注释)。
# =============================================================================

include_guard(GLOBAL)

if(NOT FRAME_INSTALL_CPP)
  return()
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ---- export set:随包发布的全部 C++ target ----
# frame_compiler_flags 是纯编译选项 INTERFACE 库,被各静态库 PRIVATE 链接;
# 静态库的 PRIVATE 依赖会以 $<LINK_ONLY:...> 生成表达式进入导出的
# INTERFACE_LINK_LIBRARIES 闭包——若它本身不在 export set 内,
# install(EXPORT) 会在消费方 find_package 时报"导出的 target 引用了未导出
# 的 target"配置期错误,故必须一并导出。
set(_frame_install_targets
  frame_core
  frame_ir
  frame_ops
  frame_compiler
  frame_runtime
  frame_interop
  frame_frontend
  frame_nn
  frame_data
  frame
  frame_compiler_flags
)

# 已启用后端 target(cmake/frame_backend.cmake 写入的全局属性,存的是 target
# 名,如 frame_backend_cpu;cpu 后端无条件启用,故该属性至少含一个元素)。
get_property(_frame_enabled_backend_targets GLOBAL PROPERTY FRAME_ENABLED_BACKENDS)
list(APPEND _frame_install_targets ${_frame_enabled_backend_targets})
unset(_frame_enabled_backend_targets)

install(TARGETS ${_frame_install_targets}
  EXPORT frame_targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
unset(_frame_install_targets)

# ---- 公共头文件(TOP-003:公共头挂在 frame_core,随其一并发布) ----
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/frame
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# ---- 导出 target 描述文件(供 find_package(frame) 加载) ----
install(EXPORT frame_targets
  NAMESPACE frame::
  FILE frame-targets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/frame
)

# ---- 包配置文件(版本校验 + include 上面的导出文件) ----
configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/frame-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/frame-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/frame
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/frame-config-version.cmake
  COMPATIBILITY SameMajorVersion
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/frame-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/frame-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/frame
)
