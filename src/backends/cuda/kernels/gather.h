#pragma once
// cuda gather 内核族(gather/scatter_add/gather_grad_internal,kernels/gather.cu)共享的
// indices D2H 预检工具(M22,批4 T4,§1.5 决议点E)。索引值域属运行时,静态图
// 无法静态校验张量值;kernel 启动前把 indices 整体拷回 host 逐元素校验
// 0<=idx<bound,越界返回 kInvalidArgument(消息含越界值与界,ARCH-031:拒绝
// device 侧静默 clamp)。实现见 kernels/gather.cpp(纯 host 端
// cudaMemcpyAsync(D2H)+ 逐元素校验,不含 __global__ 代码),与
// kernels/cuda_status.h/.cpp 同一"头声明 + .cpp 定义 + 跨 TU 复用"结构
// (REUSE-002)。仅供 src/backends/cuda/kernels/ 内部包含,不入公开 API。

#include <cstdint>
#include <string_view>

#include <frame/core/status.h>
#include <frame/core/tensor.h>

#include <cuda_runtime.h>

namespace frame::backends::cuda {

// 把 indices(device 张量,dtype 属 int32/int64)整体 D2H 拷回并逐元素校验
// 0<=value<bound;op_name/bound_description 仅用于拼错误消息。内部经
// cudaMemcpyAsync(D2H)+
// cudaStreamSynchronize 确保数据就绪后再读取(K 级小拷贝,成本诚实记录,
// 见 docs/plan/2026-07-19-batch4-m22-seq.md §1.5)。
frame::Status validate_gather_indices_range(std::string_view op_name, const frame::Tensor& indices,
                                            int64_t bound, std::string_view bound_description,
                                            cudaStream_t stream);

}  // namespace frame::backends::cuda
