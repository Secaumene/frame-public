#pragma once
// 通用宏定义:导出宏 FRAME_API 与断言宏 FRAME_CHECK。
// 语言纪律(铁律 #4):标识符/宏名纯英文;错误消息一律英文;注释中文。

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// FRAME_API:符号可见性/导出宏。
// 骨架期默认静态库(FRAME_BUILD_SHARED_LIBS=OFF),FRAME_API 展开为空;
// 未来切换为共享库/插件化后端时,由构建系统定义 FRAME_BUILD_SHARED 与
// FRAME_EXPORTS 控制 dllexport/visibility。
// ---------------------------------------------------------------------------
#if defined(_WIN32)
#if defined(FRAME_BUILD_SHARED)
#if defined(FRAME_EXPORTS)
#define FRAME_API __declspec(dllexport)
#else
#define FRAME_API __declspec(dllimport)
#endif
#else
#define FRAME_API
#endif
#else
#if defined(FRAME_BUILD_SHARED)
#define FRAME_API __attribute__((visibility("default")))
#else
#define FRAME_API
#endif
#endif

// ---------------------------------------------------------------------------
// FRAME_CHECK(condition):运行期断言。断言失败时打印英文诊断并终止进程。
// 用于表达"违反即为程序错误(不可恢复)"的前置条件;可恢复错误一律走
// Status/Result(见 include/frame/core/status.h)。
// ---------------------------------------------------------------------------
#define FRAME_CHECK(condition)                                                                   \
  do {                                                                                           \
    if (!(condition)) {                                                                          \
      std::fprintf(stderr, "FRAME_CHECK failed: %s at %s:%d\n", #condition, __FILE__, __LINE__); \
      std::abort();                                                                              \
    }                                                                                            \
  } while (0)

// ---------------------------------------------------------------------------
// FRAME_CONCAT(a, b):两层记号拼接,保证 __COUNTER__/__LINE__ 等宏参数先展开为
// 具体数值再拼接(标准宏替换规则:单层 ## 会拼出字面 "__COUNTER__" 而非其展开
// 值)。供各注册宏共用以生成进程内唯一的静态初始化器变量名(FRAME_REGISTER_OP/
// FRAME_REGISTER_KERNEL 已使用,M4 的 FRAME_REGISTER_BACKEND 亦将使用)。
// ---------------------------------------------------------------------------
#define FRAME_CONCAT_IMPL(a, b) a##b
#define FRAME_CONCAT(a, b) FRAME_CONCAT_IMPL(a, b)
