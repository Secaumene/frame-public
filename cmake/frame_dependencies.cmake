# =============================================================================
# frame_dependencies.cmake —— 第三方依赖唯一入口 + 硬件 SDK 探测(铁律 3/5)
#
# 依赖决策表(新增依赖必须先在此登记并按类别选择获取方式,禁止自行发挥;
# 新增 find_package/FetchContent 还须有 ADR,见 docs/standards/reuse-policy.md REUSE-010):
# | 类别 | 判定特征 | 获取方式 | 实例 |
# |------|----------|----------|------|
# | A. 硬件 SDK       | 由厂商安装器/驱动栈提供,无法源码构建 | 只用 find_package / 自写 find 模块,严禁 FetchContent | CUDA Toolkit、oneAPI(IntelSYCL)、OpenVINO、CANN |
# | B. 轻量构建期库   | header-only 或官方支持 add_subdirectory,且已列入 reuse-policy 清单 B 档 | FetchContent 锁定 tag(一律不带 FIND_PACKAGE_ARGS,锁定即锁定,见 ADR-0014/0015/0016);URL 模式还须 URL_HASH | GoogleTest v1.17.0、pybind11 v3.0.4、nlohmann/json v3.12.0(ADR-0018)、CUTLASS(仅 CUDA 后端内,尚未引入) |
# | C. 重量级基础设施 | 构建耗时以小时计 | 只用 find_package,配独立开关且缺省 OFF | MLIR/LLVM(FRAME_ENABLE_MLIR,v0 不接线,见 ADR-0002) |
# | D. 后端内数学库   | 随 SDK 或系统包分发 | find_package,找不到按三态规则降级 | cuDNN、oneDNN(均未引入) |
#
# 版本锁定表(当前全部 FetchContent 依赖):
# | 依赖             | tag     | 拉取条件 |
# | GoogleTest       | v1.17.0 | FRAME_BUILD_TESTS=ON |
# | pybind11         | v3.0.4  | FRAME_BUILD_PYTHON=ON 且 Python(>=3.9)开发环境可用 |
# | Google Benchmark | v1.9.5  | FRAME_BUILD_BENCHMARKS=ON(ADR-0014) |
# | nlohmann/json    | v3.12.0 | FRAME_BUILD_TOOLS=ON(ADR-0018) |
# | pocketfft        | commit c90e55b3d529f8efa40ed01a20de22405f45fc65(cpp 分支,无 tag,ADR-0022) | 无条件(cpu 后端永远启用) |
# 【待查证】GoogleTest v1.17.0 / pybind11 v3.0.4 源码包 sha256(改用 URL+URL_HASH 模式时必填)
#   —— 来源:GitHub Releases 页面(google/googletest、pybind/pybind11)
#
# 规则:
# 【DEP-001】【MUST】全部 FetchContent_Declare 集中在本文件,禁止散落子目录。
#   判定方法:grep -rn "FetchContent_Declare" --include=*.cmake --include=CMakeLists.txt .
#   仅命中本文件。
# 【DEP-002】【MUST NOT】类别 A 硬件 SDK 不得使用 FetchContent。
#   判定方法:本文件 CUDA/IntelSYCL/OpenVINO/CANN 各段落内无 FetchContent_Declare。
# 【DEP-003】【MUST】三态开关探测分支语义与 cmake/frame_options.cmake 头部注释逐字一致
#   (ON→FATAL_ERROR 并提示环境变量;AUTO→STATUS 降级;OFF→跳过)。
#   判定方法:人工比对两处注释与本文件各探测分支的 message 级别。
#
# 本文件输出(供顶层 CMakeLists.txt 与 src/CMakeLists.txt 使用):
#   FRAME_BACKEND_{CUDA,INTEL_GPU,INTEL_NPU,ASCEND}_ENABLED : BOOL,后端最终启用与否
#   FRAME_BACKEND_*_SUMMARY / FRAME_MLIR_SUMMARY / FRAME_PYTHON_SUMMARY : 摘要行文本
#   FRAME_BUILD_PYTHON 可能被本文件降级为 OFF(普通变量遮蔽 cache,仅本次配置生效)
# =============================================================================

include_guard(GLOBAL)

include(FetchContent)
include(CheckLanguage)

# ---------------------------------------------------------------------------
# 类别 A:CUDA(NVIDIA GPU)
# 探测 = check_language(CUDA) + find_package(CUDAToolkit 12.0)。
# 注意:enable_language(CUDA) 不在此处调用,而是在顶层 CMakeLists.txt 受选项
# 保护调用(评审勘误 8:enable_language 必须位于顶层目录作用域)。
# ---------------------------------------------------------------------------
set(FRAME_BACKEND_CUDA_ENABLED OFF)
set(FRAME_BACKEND_CUDA_SUMMARY "OFF (reason: disabled)")
if(NOT FRAME_ENABLE_CUDA STREQUAL "OFF")
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit 12.0 QUIET)
  endif()
  if(CMAKE_CUDA_COMPILER AND CUDAToolkit_FOUND)
    set(FRAME_BACKEND_CUDA_ENABLED ON)
    set(FRAME_BACKEND_CUDA_SUMMARY "ON  (CUDA Toolkit ${CUDAToolkit_VERSION})")
  elseif(FRAME_ENABLE_CUDA STREQUAL "ON")
    message(FATAL_ERROR "FRAME_ENABLE_CUDA=ON but CUDA Toolkit >= 12.0 was not found. Install it or set FRAME_ENABLE_CUDA=AUTO/OFF. Search path hint: set CUDAToolkit_ROOT or put nvcc on PATH.")
  else()
    message(STATUS "CUDA Toolkit >= 12.0 not found, CUDA backend auto-disabled (FRAME_ENABLE_CUDA=AUTO)")
    set(FRAME_BACKEND_CUDA_SUMMARY "OFF (reason: not found)")
  endif()
endif()

# ---------------------------------------------------------------------------
# 类别 A:Intel GPU(oneAPI SYCL)
# 前置条件:CXX 编译器必须是 icpx(CMAKE_CXX_COMPILER_ID 为 IntelLLVM,由
# intel-gpu preset 指定);配置中途不允许切换编译器,故编译器非 icpx 且开关
# 为 AUTO 时一律视为"未找到"。
# ---------------------------------------------------------------------------
set(FRAME_BACKEND_INTEL_GPU_ENABLED OFF)
set(FRAME_BACKEND_INTEL_GPU_SUMMARY "OFF (reason: disabled)")
if(NOT FRAME_ENABLE_INTEL_GPU STREQUAL "OFF")
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    if(FRAME_ENABLE_INTEL_GPU STREQUAL "ON")
      message(FATAL_ERROR "FRAME_ENABLE_INTEL_GPU=ON but the CXX compiler is not icpx (IntelLLVM). Use the intel-gpu preset or set CMAKE_CXX_COMPILER=icpx. Search path hint: source <oneapi-root>/setvars.sh (ONEAPI_ROOT).")
    else()
      message(STATUS "CXX compiler is not icpx, Intel GPU backend auto-disabled (FRAME_ENABLE_INTEL_GPU=AUTO)")
      set(FRAME_BACKEND_INTEL_GPU_SUMMARY "OFF (reason: compiler is not icpx)")
    endif()
  else()
    find_package(IntelSYCL QUIET)
    if(IntelSYCL_FOUND)
      set(FRAME_BACKEND_INTEL_GPU_ENABLED ON)
      set(FRAME_BACKEND_INTEL_GPU_SUMMARY "ON  (IntelSYCL, icpx ${CMAKE_CXX_COMPILER_VERSION})")
    elseif(FRAME_ENABLE_INTEL_GPU STREQUAL "ON")
      message(FATAL_ERROR "FRAME_ENABLE_INTEL_GPU=ON but IntelSYCL was not found. Install the Intel oneAPI DPC++/C++ compiler or set FRAME_ENABLE_INTEL_GPU=AUTO/OFF. Search path hint: source <oneapi-root>/setvars.sh (ONEAPI_ROOT).")
    else()
      message(STATUS "IntelSYCL not found, Intel GPU backend auto-disabled (FRAME_ENABLE_INTEL_GPU=AUTO)")
      set(FRAME_BACKEND_INTEL_GPU_SUMMARY "OFF (reason: not found)")
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------
# 类别 A:Intel NPU —— 一律经 OpenVINO Runtime 接入
# (Level Zero 直连路线已废弃;若将来需要直连,须先立 ADR,见命名契约)
# ---------------------------------------------------------------------------
set(FRAME_BACKEND_INTEL_NPU_ENABLED OFF)
set(FRAME_BACKEND_INTEL_NPU_SUMMARY "OFF (reason: disabled)")
if(NOT FRAME_ENABLE_INTEL_NPU STREQUAL "OFF")
  find_package(OpenVINO QUIET)
  if(OpenVINO_FOUND)
    set(FRAME_BACKEND_INTEL_NPU_ENABLED ON)
    if(DEFINED OpenVINO_VERSION)
      set(FRAME_BACKEND_INTEL_NPU_SUMMARY "ON  (OpenVINO ${OpenVINO_VERSION})")
    else()
      set(FRAME_BACKEND_INTEL_NPU_SUMMARY "ON  (OpenVINO)")
    endif()
  elseif(FRAME_ENABLE_INTEL_NPU STREQUAL "ON")
    message(FATAL_ERROR "FRAME_ENABLE_INTEL_NPU=ON but OpenVINO was not found. Install the OpenVINO Runtime or set FRAME_ENABLE_INTEL_NPU=AUTO/OFF. Search path hint: set OpenVINO_DIR or source <openvino-root>/setupvars.sh.")
  else()
    message(STATUS "OpenVINO not found, Intel NPU backend auto-disabled (FRAME_ENABLE_INTEL_NPU=AUTO)")
    set(FRAME_BACKEND_INTEL_NPU_SUMMARY "OFF (reason: not found)")
  endif()
endif()

# ---------------------------------------------------------------------------
# 类别 A:昇腾 NPU(CANN)—— 自写探测模块 cmake/find_cann.cmake
# 产出:CANN_FOUND + imported target CANN::ascendcl
# ---------------------------------------------------------------------------
set(FRAME_BACKEND_ASCEND_ENABLED OFF)
set(FRAME_BACKEND_ASCEND_SUMMARY "OFF (reason: disabled)")
if(NOT FRAME_ENABLE_ASCEND STREQUAL "OFF")
  include(find_cann)
  if(CANN_FOUND)
    set(FRAME_BACKEND_ASCEND_ENABLED ON)
    set(FRAME_BACKEND_ASCEND_SUMMARY "ON  (CANN at ${CANN_ROOT})")
  elseif(FRAME_ENABLE_ASCEND STREQUAL "ON")
    message(FATAL_ERROR "FRAME_ENABLE_ASCEND=ON but CANN was not found. Install the Ascend CANN toolkit or set FRAME_ENABLE_ASCEND=AUTO/OFF. Search path hint: set ASCEND_HOME_PATH (fallback: /usr/local/Ascend/ascend-toolkit/latest).")
  else()
    message(STATUS "CANN not found, Ascend backend auto-disabled (FRAME_ENABLE_ASCEND=AUTO)")
    set(FRAME_BACKEND_ASCEND_SUMMARY "OFF (reason: not found)")
  endif()
endif()

# ---------------------------------------------------------------------------
# 类别 C:MLIR/LLVM —— v0 仅定义开关与摘要打印,不接线
# 裁决:docs/decisions/ ADR-0002(IR 双轨并行:v0 主线自研轻量 IR,MLIR 为并行
# 预研赛道;MLIR 依赖只允许受本开关隔离出现在未来的 src/compiler/mlir/)。
# ---------------------------------------------------------------------------
set(FRAME_MLIR_SUMMARY "OFF (not wired, see ADR-0002)")
if(NOT FRAME_ENABLE_MLIR STREQUAL "OFF")
  message(STATUS "MLIR integration not yet wired, see ADR-0002")
  set(FRAME_MLIR_SUMMARY "${FRAME_ENABLE_MLIR} requested (not wired, see ADR-0002)")
endif()

# ---------------------------------------------------------------------------
# 类别 B:pocketfft(M23,批5 T3,ADR-0022)—— cpu 参考后端 rfft/irfft 内核的
# 数值基准实现,单头文件、BSD-3。无条件拉取(cpu 后端永远启用,TOP-005,
# 不随任何 FRAME_BUILD_* 开关门控,与下方 Python/GoogleTest/Benchmark/
# nlohmann_json 四个条件拉取的类别 B 依赖不同)。cpp 分支无 Releases/Tags
# (核实日期:2026-07-21),按 REUSE-013 退档锁 immutable commit(而非
# GoogleTest/pybind11 的锁 tag 同族做法);sha256 核验记录见 ADR-0022。仓根
# 无 CMakeLists.txt(仅 meson.build),但仍手动声明 INTERFACE target 而非
# FetchContent_MakeAvailable 隐式 add_subdirectory(ADR-0022 决策 3):显式声明
# 不依赖"仓库当前无 CMakeLists.txt"这一事实,避免上游未来新增构建脚本时
# 行为漂移。调用面圈禁:仅 src/backends/cpu/kernels/fft.cpp(ADR-0022 决策 4,
# 判定方法见该 ADR)。
# ---------------------------------------------------------------------------
FetchContent_Declare(
  pocketfft
  GIT_REPOSITORY https://github.com/mreineck/pocketfft.git
  GIT_TAG c90e55b3d529f8efa40ed01a20de22405f45fc65  # cpp 分支头,ADR-0022 锁定
)
# CMP0169(CMake 3.30 新增):FetchContent_Populate() 的"已声明详情"单参数形式
# 被标记 deprecated,建议改用 FetchContent_MakeAvailable 或显式选用 OLD 策略
# 保留直接 Populate 行为。本文件坚持手动 Populate + 手动 INTERFACE target(不
# add_subdirectory,ADR-0022 决策 3),故显式选用 OLD(该策略 3.25 minimum 版本
# 尚不存在,经 if(POLICY ...) 兼容判定,不影响低版本 cmake)。
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()
FetchContent_GetProperties(pocketfft)
if(NOT pocketfft_POPULATED)
  FetchContent_Populate(pocketfft)
endif()
# IMPORTED:pocketfft 是纯外部依赖(不随本项目 install(EXPORT ...) 一并导出,
# 与下方 cuda 后端 CUDNN::cudnn INTERFACE IMPORTED 同一处理口径)——非 IMPORTED
# 的普通 target 一旦被某导出 target PRIVATE 链接,install(EXPORT) 会要求它也在
# 同一导出集合中,IMPORTED 目标豁免该要求。
add_library(pocketfft INTERFACE IMPORTED)
# SYSTEM:第三方头文件以 -isystem 计入(与 GoogleTest/pybind11 经
# FetchContent_MakeAvailable(add_subdirectory)自动获得的系统头待遇一致),
# 避免其内部告警在 -Wall -Wextra -Werror(FLAG-002)下污染
# src/backends/cpu/kernels/fft.cpp 的编译结果。
target_include_directories(pocketfft SYSTEM INTERFACE ${pocketfft_SOURCE_DIR})

# ---------------------------------------------------------------------------
# 类别 B:Python + pybind11(铁律 2:尽力提供,找不到开发头则降级关闭,不报错)
# 本条**不带** FIND_PACKAGE_ARGS(ADR-0015):锁定即锁定,不再允许系统/pip 包
# 静默替代锁定的 GIT_TAG——历史上曾因带 FIND_PACKAGE_ARGS 导致 venv 内 pip
# 安装的更新版本实际替代了锁定版参与构建,锁定语义失效(漂移已实测发现并
# 留档,见 ADR-0015);去除后 FetchContent_MakeAvailable 强制源码构建。
# ---------------------------------------------------------------------------
set(FRAME_PYTHON_SUMMARY "OFF (FRAME_BUILD_PYTHON=OFF)")
if(FRAME_BUILD_PYTHON)
  find_package(Python 3.9 COMPONENTS Interpreter Development.Module)
  if(Python_FOUND)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v3.0.4
    )
    FetchContent_MakeAvailable(pybind11)
    set(FRAME_PYTHON_SUMMARY "ON  (Python ${Python_VERSION})")
  else()
    message(WARNING "Python >= 3.9 with development headers not found, Python bindings auto-disabled (best effort per iron rule 2). Install python3-dev or set FRAME_BUILD_PYTHON=OFF to silence this warning.")
    set(FRAME_BUILD_PYTHON OFF)
    set(FRAME_PYTHON_SUMMARY "OFF (reason: Python >= 3.9 dev not found)")
  endif()
endif()

# ---------------------------------------------------------------------------
# 类别 B:GoogleTest —— 仅 FRAME_BUILD_TESTS=ON 时拉取;本条自 ADR-0016 起**不带**
# FIND_PACKAGE_ARGS(锁定即锁定,禁止系统包静默替代锁定 tag,三个 FetchContent
# 依赖同口径)
# ---------------------------------------------------------------------------
if(FRAME_BUILD_TESTS)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
  )
  # MSVC:与父工程共用运行库,避免 CRT 不匹配(非 MSVC 平台无副作用)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  # 关闭其安装规则,防止 gtest/gmock 头文件与静态库随 cmake --install 进入
  # 系统安装前缀(先例见下方 Google Benchmark 段 BENCHMARK_ENABLE_INSTALL)。
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

# ---------------------------------------------------------------------------
# 类别 B:Google Benchmark —— 仅 FRAME_BUILD_BENCHMARKS=ON 时拉取;微基准
# 计时/统计/报告设施(ADR-0014,docs/standards/benchmarks.md)。
# 本条**不带** FIND_PACKAGE_ARGS(GoogleTest/pybind11 自 ADR-0016/0015 起同此
# 口径):REUSE-013
# 已把"版本锁定"置于"系统包优先"之上——若加 FIND_PACKAGE_ARGS,CMake 会优先
# 复用系统已安装的 Google Benchmark 包,其版本可能早于/晚于本文件锁定的
# v1.9.5,基准数字因而随机器安装的系统包版本漂移而失去可比性,且历史上
# pybind11 曾因系统包版本与锁定 tag 不一致而排查耗时(同类事故不复现)。此处
# 刻意维持"该依赖唯一版本来源 = 本文件锁定的 GIT_TAG"这一不变式。
# ---------------------------------------------------------------------------
if(FRAME_BUILD_BENCHMARKS)
  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.5
  )
  # 关闭其自带测试套件与安装规则(不需要跑 Google Benchmark 自身的单测,也
  # 不希望其产物被装进任何安装前缀)。
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)
endif()

# ---------------------------------------------------------------------------
# 类别 B:nlohmann/json —— 仅 FRAME_BUILD_TOOLS=ON 时拉取;tools/frame_dslc
# 解析 JSON 模型描述所需(ADR-0018)。使用面限定 tools/ 与 tests/,核心库
# (include/frame/ + src/,含本文件即将新增的 frame_frontend)零暴露——
# frontend 库只消费纯 C++ ModelSpec,不 include nlohmann(REUSE-012 准入
# 限定「仅限工具与测试代码」)。本条**不带** FIND_PACKAGE_ARGS(与
# GoogleTest/pybind11/Google Benchmark 同口径,ADR-0014/0015/0016:锁定即
# 锁定,不允许系统/pip 包静默替代锁定 tag)。
# ---------------------------------------------------------------------------
if(FRAME_BUILD_TOOLS)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
  )
  # 关闭其安装规则,防止 nlohmann/json 头文件随 cmake --install 进入系统
  # 安装前缀(先例见上方 Google Benchmark 段 BENCHMARK_ENABLE_INSTALL)。
  set(JSON_Install OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(nlohmann_json)
endif()
