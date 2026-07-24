# =============================================================================
# benchmarks/CMakeLists.txt —— 微基准(FRAME_BUILD_BENCHMARKS=ON 时接入,见
# 顶层 CMakeLists.txt 的 EXISTS 保护与 cmake/frame_options.cmake)。
#
# 【TOP-006】每个 target 均链接 frame::compiler_flags(告警/ASAN 统一接线)。
# 目标命名 frame_bench_<主题>(BENCH-001),链接聚合库 frame::frame(含全部
# 已启用后端的 WHOLE_ARCHIVE 静态注册,见 src/CMakeLists.txt)与
# benchmark::benchmark(Google Benchmark,ADR-0014)。不链接
# benchmark::benchmark_main:各基准文件自带 main()(运行期探测 cuda 后端是否
# 可用后再决定是否注册对应基准,见 bench_matmul.cpp 尾部)。
# =============================================================================

add_executable(frame_bench_matmul bench_matmul.cpp)
target_link_libraries(frame_bench_matmul PRIVATE frame::frame frame::compiler_flags benchmark::benchmark)

# M21 批3 T7(docs/plan/2026-07-18-batch3-m21-conv.md 第3节):conv2d 编译执行
# 基准,登记手法同上(BENCH-001)。
add_executable(frame_bench_conv2d bench_conv2d.cpp)
target_link_libraries(frame_bench_conv2d PRIVATE frame::frame frame::compiler_flags benchmark::benchmark)
