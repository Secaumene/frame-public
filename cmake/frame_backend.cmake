# =============================================================================
# frame_backend.cmake —— frame_add_backend() 统一后端 target 工厂
#
# 用法(每个后端子目录 src/backends/<name>/CMakeLists.txt 只写这一段,消除样板差异):
#   frame_add_backend(
#     NAME cuda                       # → target: frame_backend_cuda,编译宏: FRAME_ENABLE_CUDA=1
#     SOURCES cuda_backend.cpp cuda_allocator.cpp kernels/elementwise.cu
#     LINK CUDA::cudart CUDA::cublas
#   )
#
# 职责:
#   1. add_library(frame_backend_<name> STATIC ...) + 别名 frame::backend_<name>;
#   2. 编译宏 FRAME_ENABLE_<NAME 大写>=1(PUBLIC);
#   3. 链接 frame::core(公共头)、frame::runtime 与 frame::ops(注册宏落地后
#      后端 TU 引用 BackendRegistry/KernelRegistry 符号,声明依赖保证静态库
#      链接顺序正确)、frame::compiler_flags(告警/ASAN)以及 LINK 列表;
#   4. 把 target 追加到全局属性 FRAME_ENABLED_BACKENDS,供聚合库 frame 链接。
#
# ★ WHOLE_ARCHIVE 链接注意(必须遵守,此坑写进注释是命名契约要求):
#   后端通过 FRAME_REGISTER_BACKEND 宏做静态注册,注册符号位于没有被任何外部
#   代码显式引用的翻译单元;链接器处理静态库时默认只取被引用的目标文件,会把
#   注册符号连同其静态初始化器一起丢弃,导致 BackendRegistry::get("<name>")
#   运行期返回 kNotFound。因此聚合库/可执行文件链接后端静态库时必须写:
#     target_link_libraries(frame PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,frame_backend_<name>>)
#   ($<LINK_LIBRARY:WHOLE_ARCHIVE,...> 需 CMake >= 3.24,本项目最低版本 3.25
#   已覆盖——下限由 CMakePresets.json 的 version 6 schema 决定。)
#
# 规则:
# 【BKD-001】【MUST】后端 target 一律经本函数创建,禁止后端子目录手写 add_library。
#   判定方法:grep -rE '^[^#]*add_library[[:space:]]*\(' src/backends/ 输出为空
#   (排除注释行,与顶层 TOP-005 判定一致)。
# 【BKD-002】【MUST】消费 FRAME_ENABLED_BACKENDS 的链接处使用 WHOLE_ARCHIVE
#   生成器表达式。判定方法:grep -rn "FRAME_ENABLED_BACKENDS" src/ 的链接语句
#   均含 LINK_LIBRARY:WHOLE_ARCHIVE。
# =============================================================================

include_guard(GLOBAL)

# 全局属性:已启用后端 target 名单(聚合库 frame 读取)
set_property(GLOBAL PROPERTY FRAME_ENABLED_BACKENDS "")

# frame_add_backend(NAME <name> SOURCES <src...> [LINK <libs...>])
# 前置条件:frame::core、frame::runtime、frame::ops 与 frame::compiler_flags 已存在
# (即 src/CMakeLists.txt 先定义模块 target,再 add_subdirectory 各后端目录)。
function(frame_add_backend)
  set(options "")
  set(one_value_args NAME)
  set(multi_value_args SOURCES LINK)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
  if(NOT ARG_NAME)
    message(FATAL_ERROR "frame_add_backend: NAME is required")
  endif()
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "frame_add_backend: SOURCES is required")
  endif()
  if(NOT TARGET frame::core OR NOT TARGET frame::runtime OR NOT TARGET frame::ops)
    message(FATAL_ERROR "frame_add_backend: targets frame::core/frame::runtime/frame::ops must exist before adding backend '${ARG_NAME}' (define module targets in src/CMakeLists.txt first)")
  endif()
  set(target_name "frame_backend_${ARG_NAME}")
  string(TOUPPER "${ARG_NAME}" name_upper)
  # TODO(FRAME-DESIGN): FRAME_BUILD_SHARED_LIBS=ON 时应改建 MODULE 插件并走运行时加载,
  #   涉及符号可见性与注册机制变更,禁止在无 ADR 时自行决定。参考: docs/architecture/backend-hal.md。
  #   完成判据: 存在插件化 ADR,且 -DFRAME_BUILD_SHARED_LIBS=ON 配置构建成功。
  add_library(${target_name} STATIC ${ARG_SOURCES})
  add_library(frame::backend_${ARG_NAME} ALIAS ${target_name})
  target_compile_definitions(${target_name} PUBLIC FRAME_ENABLE_${name_upper}=1)
  target_link_libraries(${target_name} PUBLIC frame::core)
  # 注册宏(FRAME_REGISTER_BACKEND/FRAME_REGISTER_KERNEL)落地后,后端 TU 引用
  # BackendRegistry(frame_runtime)与 KernelRegistry(frame_ops)符号;声明依赖
  # 使 CMake 正确排序静态库,避免 WHOLE_ARCHIVE 后端库出现未定义引用。
  target_link_libraries(${target_name} PRIVATE frame::runtime frame::ops)
  target_link_libraries(${target_name} PRIVATE frame::compiler_flags)
  if(ARG_LINK)
    target_link_libraries(${target_name} PRIVATE ${ARG_LINK})
  endif()
  set_property(GLOBAL APPEND PROPERTY FRAME_ENABLED_BACKENDS ${target_name})
endfunction()
