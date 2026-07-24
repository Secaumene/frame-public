# =============================================================================
# Frame 顶层构建入口
#
# 骨架验收口径:cmake --preset cpu-only && cmake --build --preset cpu-only
#               && ctest --preset cpu-only 零失败。
#
# ---- src/CMakeLists.txt 必须履行的 target 契约(该文件由后续实现,违反即评审打回)----
# 【TOP-001】【MUST】定义模块 target(全部 STATIC):frame_core / frame_ir /
#   frame_compiler / frame_runtime / frame_ops,并提供别名 frame::core、frame::ir、
#   frame::compiler、frame::runtime、frame::ops。
#   判定方法:grep src/CMakeLists.txt 存在上述 5 个 add_library(... STATIC) 与 5 个别名。
# 【TOP-002】【MUST】依赖方向固定为 runtime → compiler → ops → ir → core
#   (与 docs/architecture/overview.md ARCH-001 一致),禁止反向依赖。
#   判定方法:src/CMakeLists.txt 的 target_link_libraries 依赖图与该链一致
#   (仅允许沿链向下依赖)。
# 【TOP-003】【MUST】公共头挂在 frame_core:
#   target_include_directories(frame_core PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)。
#   判定方法:grep src/CMakeLists.txt 命中该语句。
# 【TOP-004】【MUST】定义聚合库 frame(别名 frame::frame):链接全部模块 target,
#   并读取全局属性 FRAME_ENABLED_BACKENDS 链接已启用后端;后端静态库必须用
#   $<LINK_LIBRARY:WHOLE_ARCHIVE,...> 链接(静态注册符号防裁剪,
#   缘由见 cmake/frame_backend.cmake 头部注释)。
#   判定方法:grep src/CMakeLists.txt 命中 LINK_LIBRARY:WHOLE_ARCHIVE。
# 【TOP-005】【MUST】src/backends/cpu/ 无条件 add_subdirectory(参考后端永远启用);
#   src/backends/{cuda,intel_gpu,intel_npu,ascend}/ 仅当对应
#   FRAME_BACKEND_{CUDA,INTEL_GPU,INTEL_NPU,ASCEND}_ENABLED 为真时 add_subdirectory;
#   后端子目录内只调 frame_add_backend(),禁止手写 add_library。
#   判定方法:grep src/CMakeLists.txt 的 add_subdirectory(backends/...) 均有对应 if 保护
#   (cpu 除外);grep -rE '^[^#]*add_library[[:space:]]*\(' src/backends/ 为空
#   (排除注释行——后端 CMakeLists 的注释里出现该词不算违例)。
# 【TOP-006】【MUST】所有自有 target 链接 frame::compiler_flags(告警/ASAN 统一接线)。
#   判定方法:src/、tests/、python/ 各 CMakeLists.txt 每个 target 均有该链接。
#
# ---- 其余子目录契约(简要)----
#   tests/CMakeLists.txt : 使用 GTest::gtest_main + gtest_discover_tests
#                          (GoogleTest 已由 cmake/frame_dependencies.cmake 提供);
#   python/CMakeLists.txt: pybind11_add_module(_core python/src/bindings.cpp) 并链接 frame::frame
#                          (pybind11 已由 cmake/frame_dependencies.cmake 提供)。
# =============================================================================

# 下限 3.25:CMakePresets.json 为 version 6 schema,3.24 无法解析 preset 文件
cmake_minimum_required(VERSION 3.25)

project(frame VERSION 0.1.0 LANGUAGES CXX)

# ---- 全局语言设置(不可变更;C++20 裁决见 docs/decisions/ ADR-0004) ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# 为 Python 扩展与未来插件化做准备
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(frame_options)
include(frame_compiler_flags)
include(frame_dependencies)
include(frame_backend)

# ---- CUDA 语言在顶层受选项保护启用(评审勘误 8:不得下放到后端子目录) ----
# nvcc 自 CUDA 12.0 起支持 C++20,故 CMAKE_CUDA_STANDARD 取 20(ADR-0004)。
if(FRAME_BACKEND_CUDA_ENABLED)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_subdirectory(src)

if(FRAME_BUILD_PYTHON)
  add_subdirectory(python)
endif()

if(FRAME_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# 官方 C++ 示例由 FRAME_BUILD_EXAMPLES 门控；仓库 dev/cuda preset 显式开启，
# 使手册配套源码持续参与构建与测试。
if(FRAME_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# 微基准(BENCH-001,ADR-0014):不进 cpu-only 门禁与 ctest,以独立 bench
# preset 编译运行;EXISTS 保护循 examples 先例。
if(FRAME_BUILD_BENCHMARKS)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/CMakeLists.txt")
    add_subdirectory(benchmarks)
  else()
    message(STATUS "FRAME_BUILD_BENCHMARKS=ON but benchmarks/CMakeLists.txt does not exist yet, skipping benchmarks")
  endif()
endif()

# 命令行工具(tools/frame_dslc,ADR-0017):EXISTS 保护循 examples/benchmarks
# 先例——本阶段 tools/ 尚不存在,故被跳过。
if(FRAME_BUILD_TOOLS)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/CMakeLists.txt")
    add_subdirectory(tools)
  else()
    message(STATUS "FRAME_BUILD_TOOLS=ON but tools/CMakeLists.txt does not exist yet, skipping tools")
  endif()
endif()

# ---- C++ 安装/导出(系统安装,BUILD-040):此时 src/ 下全部 target 已定义,
#   export set 才能收全;受 FRAME_INSTALL_CPP 门控,规则见该文件头部注释。----
include(frame_install)

# ---- 配置摘要(必须最后调用,固定格式供 CI 断言) ----
frame_print_configuration_summary()
