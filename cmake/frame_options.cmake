# =============================================================================
# frame_options.cmake —— 全部构建选项的唯一定义处
#
# 三态开关语义(由 frame_dependencies.cmake 落实,两处注释与实现必须保持一致):
#   ON   → 探测 SDK;找不到 → message(FATAL_ERROR) 并提示环境变量 → 配置失败
#   AUTO → 探测 SDK;找到 → 等同 ON;找不到 → message(STATUS) 降级禁用,绝不失败
#   OFF  → 完全跳过探测与对应子目录
#
# 规则:
# 【OPT-001】【MUST】后端开关与 FRAME_ENABLE_MLIR 一律三态字符串 AUTO|ON|OFF,
#   取值非法立即配置失败。判定方法:cmake -DFRAME_ENABLE_CUDA=BOGUS 配置时报 FATAL_ERROR。
# 【OPT-002】【MUST】全部构建选项只在本文件定义。判定方法:
#   grep -rn "option(" cmake/ CMakeLists.txt 仅命中本文件。
# 【OPT-003】【MUST NOT】开发者与 CI 不得手敲 -D 传选项,一律使用 CMakePresets.json
#   中的 preset。判定方法:CI 脚本与文档中的配置命令均为 cmake --preset <name> 形式。
# =============================================================================

include_guard(GLOBAL)

# 声明一个三态开关(AUTO|ON|OFF)并校验取值合法性
function(frame_declare_tristate_option name default docstring)
  set(${name} "${default}" CACHE STRING "${docstring}")
  set_property(CACHE ${name} PROPERTY STRINGS AUTO ON OFF)
  set(_frame_opt_value "${${name}}")
  if(NOT _frame_opt_value MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "${name} must be one of AUTO|ON|OFF, got: ${_frame_opt_value}")
  endif()
endfunction()

# ---- 后端开关(三态;对应 src/backends/{cuda,intel_gpu,intel_npu,ascend}/) ----
frame_declare_tristate_option(FRAME_ENABLE_CUDA "AUTO" "NVIDIA GPU backend (CUDA Toolkit >= 12.0)")
frame_declare_tristate_option(FRAME_ENABLE_INTEL_GPU "AUTO" "Intel GPU backend (oneAPI SYCL, requires icpx)")
frame_declare_tristate_option(FRAME_ENABLE_INTEL_NPU "AUTO" "Intel NPU backend (OpenVINO Runtime)")
frame_declare_tristate_option(FRAME_ENABLE_ASCEND "AUTO" "Huawei Ascend NPU backend (CANN)")

# ---- MLIR(三态,缺省 OFF;v0 仅保留开关不接线,裁决见 docs/decisions/ ADR-0002) ----
frame_declare_tristate_option(FRAME_ENABLE_MLIR "OFF" "MLIR compiler infrastructure (reserved, not wired)")

# ---- 构建开关(BOOL) ----
# 铁律 2:Python 绑定尽力提供——找不到 Python 开发头时自动降级关闭(见 frame_dependencies.cmake)
option(FRAME_BUILD_PYTHON "Build pybind11 Python bindings" ON)
option(FRAME_BUILD_TESTS "Build C++ tests (GoogleTest)" ON)
# 默认 OFF:消费方不隐式构建示例;仓库 dev/cuda preset 显式启用并纳入 CTest
option(FRAME_BUILD_EXAMPLES "Build examples" OFF)
# 默认 OFF:不进 cpu-only 门禁与 ctest(BENCH-001),基准以独立 bench preset
# 编译运行(ADR-0014、docs/standards/benchmarks.md)
option(FRAME_BUILD_BENCHMARKS "Build microbenchmarks" OFF)
# 骨架期一律静态库;插件化(MODULE)后端属后续演进,见 cmake/frame_backend.cmake 的 FRAME-DESIGN 待办项
option(FRAME_BUILD_SHARED_LIBS "Build backends as runtime-loadable plugins (reserved)" OFF)
option(FRAME_ENABLE_ASAN "Enable AddressSanitizer (use the dev-asan preset)" OFF)
# 默认 ON:系统安装(cmake --install,见 cmake/frame_install.cmake)需要本开关
# 打开才产出 C++ 静态库/头文件/CMake 包配置;wheel 构建(scikit-build-core,
# 见 pyproject.toml 的 cmake.args)显式置 OFF,防止 wheel 前缀混入 C++
# 安装件(BUILD-041,docs/standards/build-and-test.md 第 10 节)。
option(FRAME_INSTALL_CPP "Install C++ libraries, headers and CMake package config" ON)
# 默认 ON:命令行工具(tools/frame_dslc,ADR-0017)。wheel 构建
# (scikit-build-core)显式置 OFF——工具层依赖 nlohmann/json(ADR-0018),不随
# Python 扩展一并打包(与 FRAME_INSTALL_CPP 的 wheel 隔离口径一致)。
option(FRAME_BUILD_TOOLS "Build command-line tools (frame_dslc)" ON)
