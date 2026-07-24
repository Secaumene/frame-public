#pragma once
// build_backward_graph:反向图生成入口(编译期图→图变换,M17 交付)。
// build_sgd_update_graph:SGD 参数更新图生成入口(编译期图→图变换,M18 交付)。
// 权威契约见 docs/architecture/autograd.md 第2/5/6章(ARCH-060~ARCH-069);
// 本文件只声明公开签名。二者均是独立的图→图变换函数(构图期显式
// 调用),不是标准管线 pass(ARCH-060)——不出现在 include/frame/compiler/
// pipeline.h 的九段清单,产出的图经既有 runtime::compile 走完整管线与缓存,
// 对 runtime/后端完全透明。

#include <cstdint>
#include <span>

#include <frame/core/macros.h>
#include <frame/core/status.h>
#include <frame/ir/node.h>  // ir::TensorType 完整定义(std::span<const TensorType> 形参需要)

namespace frame::ir {
class Graph;  // 前向声明:见 include/frame/ir/graph.h
}  // namespace frame::ir

namespace frame::compiler {

// 由 forward(纯函数,不被修改,ARCH-021 精神)生成训练图:训练图 = forward
// 的一份克隆(经 ir::clone_graph),附加反向节点。forward 的全部原输出按原序
// 保留,随后追加
// [grad(wrt_input_indices[0]), grad(wrt_input_indices[1]), ...](按给定顺序)。
// forward 只有 loss 一个输出时,布局仍与 M17 逐字兼容为 [loss, grads...]。
//
// loss_output_index 是 forward.outputs() 中的下标(允许多输出 forward;越界返回
// kInvalidArgument);
// 该输出必须是标量(shape rank 0 或 numel==1,ARCH-061;非标量返回
// kInvalidArgument,消息含实际 shape)。wrt_input_indices 是 forward.inputs()
// 中的下标集合(须界内且不重复,越界或重复均返回 kInvalidArgument;不在此列表
// 中的输入即视为停止梯度——不引入任何 IR 标记,ARCH-061)。
//
// 生成算法(docs/architecture/autograd.md 第2章):①clone_graph 克隆 forward;
// ②loss 输出接种子梯度 constant(1);③按逆拓扑序对每个参与 loss 依赖链的节点
// 查其 OpSchema::gradient(),把梯度微图内联展开进训练图;④同一 Value 被多消费
// 者使用时,各分支梯度经 add 累加;⑤mark_output。链上任一算子未注册
// GradientFn 返回 kUnimplemented(消息含算子名,ARCH-062);带 kHasSideEffect
// trait 的算子出现在链上返回 kInvalidArgument。wrt_input_indices 中不在 loss
// 依赖链上的输入,其梯度按标准自动微分惯例补零(constant 全零张量),不视为
// 错误。
FRAME_API Result<ir::Graph> build_backward_graph(const ir::Graph& forward,
                                                 int32_t loss_output_index,
                                                 std::span<const int32_t> wrt_input_indices);

// 构建 SGD 参数更新图(docs/architecture/autograd.md 第6章,ARCH-065):v0
// 优化器拒绝图内原位更新(M9 memory_planning 口径"所有 kernel 分配新输出",
// 图内更新违 SSA),改用一份独立的更新图,对每个参数位纯用既有算子组合表达
// `new_param_i = add(param_i, mul(constant(-learning_rate), grad_i))`——不
// 注册任何新 kernel(ARCH-065 判定方法)。learning_rate 经 constant 烘焙进图
// (v0;变学习率是后续议题)。
//
// 图输入按位 = [param_0..param_{n-1}, grad_0..grad_{n-1}](param_types 给定
// 顺序;grad_i 类型恒等于 param_i,即 param_types[i]);图输出按位 =
// [new_param_0..new_param_{n-1}],与输入 param 顺序一一对应。
//
// 校验(逐条,违例返回 kInvalidArgument,消息英文):param_types 非空;
// learning_rate 是有限值(非 NaN/非 inf);逐个 param_types[i].dtype 属 v0
// constant 可编码白名单(float32/float16/bfloat16,与
// ops::fill_tensor_from_constant_attrs 同一白名单口径,消息含下标与实际
// dtype 名)。v0 mul 无广播(见 docs/architecture/operator-system.md 逐元素
// 算子约束),故 constant(-learning_rate) 按 param_types[i].shape 展开为
// 逐元素同值张量(与 square 梯度微图内 constant(2) 同构造手法,
// src/ops/schemas/elementwise.cpp::square_gradient),不是 rank-0 标量。
FRAME_API Result<ir::Graph> build_sgd_update_graph(std::span<const ir::TensorType> param_types,
                                                   double learning_rate);

}  // namespace frame::compiler
