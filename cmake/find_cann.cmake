# =============================================================================
# find_cann.cmake —— 昇腾 CANN 探测(类别 A:自写 find 模块,严禁 FetchContent)
#
# 搜索顺序:
#   1. $ENV{ASCEND_HOME_PATH}(CANN 官方 set_env.sh 导出的环境变量)
#   2. /usr/local/Ascend/ascend-toolkit/latest(默认安装路径,次选)
#
# 产出:
#   CANN_FOUND             : BOOL,是否找到 CANN
#   CANN_ROOT              : 找到时的安装根目录(仅用于摘要打印)
#   CANN_INCLUDE_DIR       : acl/acl.h 所在 include 目录(cache)
#   CANN_ASCENDCL_LIBRARY  : libascendcl 路径(cache)
#   imported target CANN::ascendcl(后端经 frame_add_backend(... LINK CANN::ascendcl) 使用)
#
# 【待查证】CANN 安装目录内部布局(include/lib64 之上是否还有 <arch>-linux 层,
#   以及各 CANN 版本间的差异)—— 来源:《CANN 软件安装指南》(华为昇腾官方文档)。
#   下方 PATH_SUFFIXES 同时覆盖两种常见布局,查证后应收敛为单一布局。
# =============================================================================

include_guard(GLOBAL)

set(CANN_FOUND OFF)

# 组装搜索根:环境变量优先,默认安装路径兜底
set(_cann_search_roots "")
if(DEFINED ENV{ASCEND_HOME_PATH})
  list(APPEND _cann_search_roots "$ENV{ASCEND_HOME_PATH}")
endif()
list(APPEND _cann_search_roots "/usr/local/Ascend/ascend-toolkit/latest")

find_path(CANN_INCLUDE_DIR
  NAMES acl/acl.h
  PATHS ${_cann_search_roots}
  PATH_SUFFIXES include x86_64-linux/include aarch64-linux/include
  NO_DEFAULT_PATH
)

find_library(CANN_ASCENDCL_LIBRARY
  NAMES ascendcl
  PATHS ${_cann_search_roots}
  PATH_SUFFIXES lib64 x86_64-linux/lib64 aarch64-linux/lib64 runtime/lib64
  NO_DEFAULT_PATH
)

if(CANN_INCLUDE_DIR AND CANN_ASCENDCL_LIBRARY)
  set(CANN_FOUND ON)
  get_filename_component(CANN_ROOT "${CANN_INCLUDE_DIR}" DIRECTORY)
  if(NOT TARGET CANN::ascendcl)
    add_library(CANN::ascendcl UNKNOWN IMPORTED)
    set_target_properties(CANN::ascendcl PROPERTIES
      IMPORTED_LOCATION "${CANN_ASCENDCL_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CANN_INCLUDE_DIR}"
    )
  endif()
endif()

unset(_cann_search_roots)
