#pragma once
// tiny_mlp ModelSpec 构造 helper(namespace frame::frontend::testing)。供
// test_model_spec.cpp / test_lowering.cpp / test_runner.cpp / test_emitter.cpp
// 四个测试文件复用(REUSE-002,避免每个测试文件各自手写一份同构 ModelSpec)。
//
// 拓扑与 tests/cpp/compiler/mlp_forward_graph_helper.h::BuildMlpForwardGraph
// 完全一致:x[8,4] -> matmul(W1[4,8]) -> add(b1[8,8]) -> relu ->
// matmul(W2[8,1]) -> mse_loss(., target[8,1])(第二层无 bias)。batch ==
// hidden_dim == 8 是刻意取值:第二层 W2 恰有 8 个自由参数,对固定的隐藏层
// 特征矩阵 H([8,8])而言是一个满秩线性方程组解(8 个方程对 8 个未知数),使
// 训练在数百步 SGD 内即可把 loss 压到极低——test_runner.cpp 的收敛断言依据
// 正是这一结构性论证(而非某个精细调参的巧合)。
//
// data/param_init 均取 uniform_seeded(std::mt19937(20260713) 驱动,
// frontend-dsl.md 第 2 节顺序:数据输入 -> target -> 逐层参数[weight 先
// bias]),范围经本机实测校准(见 test_runner.cpp 收敛用例注释)。
// 参数初始化范围取舍:bias 偏正区间 [0.0, 0.5)(而非居中于 0)是与
// tests/cpp/compiler/test_training_loop.cpp 相同的"避免死 relu"手法——
// matmul(x, W1) 项在本数据/权重量级下普遍小于该偏移,使各隐藏单元训练初期
// 大概率落在 relu 恒等分支,减少满秩条件被破坏的概率(本 helper 的 batch ==
// hidden_dim == 8 依赖 H([8,8]) 满秩这一结构性论证收敛,见上方注释)。

#include <vector>

#include <frame/frontend/model_spec.h>

namespace frame::frontend::testing {

inline ModelSpec make_tiny_mlp_spec() {
  ModelSpec spec;
  spec.name = "tiny_mlp";
  spec.batch = 8;

  InputSpec input;
  input.name = "x";
  input.shape = {8, 4};
  spec.inputs = {input};

  LinearLayerSpec layer0;
  layer0.name = "layer0";
  layer0.input = "x";
  layer0.weight_shape = {4, 8};
  layer0.bias_shape = std::vector<int64_t>{8, 8};
  layer0.activation = Activation::kRelu;

  LinearLayerSpec layer1;
  layer1.name = "layer1";
  layer1.input = "layer0";
  layer1.weight_shape = {8, 1};
  // layer1.bias_shape 保持缺省(无 bias)。
  layer1.activation = Activation::kNone;

  spec.layers = {layer0, layer1};

  spec.loss.prediction = "layer1";
  spec.loss.target_shape = {8, 1};

  spec.optimizer.learning_rate = 0.05;

  spec.training.steps = 300;
  spec.training.seed = 20260713U;
  spec.training.log_every = 0;

  TensorDataSpec input_data;
  input_data.kind = InitKind::kUniformSeeded;
  input_data.lo = -1.0F;
  input_data.hi = 1.0F;
  spec.data["x"] = input_data;

  TensorDataSpec target_data;
  target_data.kind = InitKind::kUniformSeeded;
  target_data.lo = -1.0F;
  target_data.hi = 1.0F;
  spec.data["target"] = target_data;

  spec.param_init.weight_lo = -0.3F;
  spec.param_init.weight_hi = 0.3F;
  spec.param_init.bias_lo = 0.0F;
  spec.param_init.bias_hi = 0.5F;

  return spec;
}

}  // namespace frame::frontend::testing
