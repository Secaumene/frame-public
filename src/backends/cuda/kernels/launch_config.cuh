#pragma once
// cuda kernel 层共享的一维 launch 配置计算工具(M22,批4 T4,code-reviewer
// 建议①判重收敛):LaunchConfig/compute_launch_config 曾在 elementwise.cu
// (M19)、reduction.cu(M17)、pool.cu(M21)、shape.cu、gather.cu(均 M22 批4
// T4)五个 .cu 文件各持一份逐字相同实现,同目录可共享,收敛为单份(铁律 5):
// grid/block 计算集中一处,不在每个 kernel 文件内各自重复(docs/backends/
// cuda.md 第6章)。仅供 src/backends/cuda/kernels/ 内部 .cu 翻译单元包含,
// 不入公开 API。

#include <cstdint>

#include <cuda_runtime.h>

namespace frame::backends::cuda {

// 一维 launch 网格/block 配置。
struct LaunchConfig {
  dim3 grid;
  dim3 block;
};

// 固定 block 大小 256,按 numel 向上取整算 block 数,一线程一元素的一维 grid。
inline LaunchConfig compute_launch_config(int64_t numel) {
  constexpr unsigned int kBlockSize = 256;
  const int64_t block_count = (numel + kBlockSize - 1) / static_cast<int64_t>(kBlockSize);
  return LaunchConfig{dim3(static_cast<unsigned int>(block_count)), dim3(kBlockSize)};
}

}  // namespace frame::backends::cuda
