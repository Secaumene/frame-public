#pragma once
// lower_to_graph / lower_to_inference_graph:ModelSpec → ir::Graph 的构图入口
// (docs/architecture/frontend-dsl.md 第 4 章)。全部节点经
// ops::create_node_with_inferred_types 创建(shape 推断复用 OpRegistry,
// REUSE-002);构图口径对齐 tests/cpp/compiler/mlp_forward_graph_helper.h。

#include <cstdint>
#include <string>
#include <vector>

#include <frame/core/macros.h>
#include <frame/core/status.h>
#include <frame/frontend/model_spec.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>

namespace frame::frontend {

// lower_to_graph 的产物:训练前向图 + 参数元信息,供
// compiler::build_backward_graph(forward, /*loss_output_index=*/0,
// wrt_input_indices) 与 compiler::build_sgd_update_graph(param_types, lr)
// 直接消费。param_names/param_types/wrt_input_indices 三者等长且逐位对应,
// 顺序 = 逐层 weight 先于 bias(与 frontend-dsl.md 第 2 节随机抽取顺序一致)。
struct LoweredModel {
  ir::Graph forward;
  std::vector<std::string> param_names;
  std::vector<ir::TensorType> param_types;
  std::vector<int32_t> wrt_input_indices;
};

// 训练前向图:图输入序 [数据输入..., 逐层 weight,bias..., target],单一图
// 输出 = loss(mse_loss 节点输出,index 0)。内部先调用 validate(spec),失败
// 原样透传。
FRAME_API Result<LoweredModel> lower_to_graph(const ModelSpec& spec);

// 推理图:图输入序 [数据输入..., 逐层 weight,bias...](无 target),单一图
// 输出 = loss.prediction 引用层的输出。是对已批准草案的最小扩充——训练
// 收敛后需要一张不含 target 的图跑推理(向 code-reviewer 说明)。内部先调用
// validate(spec),失败原样透传。
FRAME_API Result<ir::Graph> lower_to_inference_graph(const ModelSpec& spec);

}  // namespace frame::frontend
