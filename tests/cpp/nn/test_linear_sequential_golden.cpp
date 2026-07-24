// Linear+Relu 经 Sequential 的 nn 构图与手工构图 dump_text 逐字节相等
// (ARCH-074 判据雏形,docs/architecture/nn-design.md §2/§8)。手工构图节点序
// 照抄 src/frontend/lowering.cpp::AppendLayer:matmul -> add(bias) -> relu。
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ir/serialization.h>
#include <frame/nn/layers.h>
#include <frame/nn/module.h>
#include <frame/ops/graph_builder.h>
#include <frame/ops/op_registry.h>

namespace {

using frame::cpu_device;
using frame::DType;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::ir::dump_text;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::OpQuery;
using frame::ir::TensorType;
using frame::ir::Value;
using frame::nn::add_parameter_inputs;
using frame::nn::Linear;
using frame::nn::Module;
using frame::nn::ParamSpec;
using frame::nn::Relu;
using frame::nn::Sequential;
using frame::ops::create_node_with_inferred_types;
using frame::ops::make_op_query;

// 与 src/nn/layers.cpp(匿名命名空间)::MakeCpuTensorType、
// src/frontend/lowering.cpp::MakeCpuFloat32TensorType 同构造口径(dtype +
// shape + device,layout 保持默认 Layout::kUnknown)——dump_text 把 layout 也
// 序列化进输出文本(src/ir/serialization.cpp::format_tensor_type),两侧
// TensorType 若 layout 取值不同,即便 dtype/shape/device 全同也会导致本文件
// 的逐字节比较失败;因此本 helper 特意不显式设置 layout 字段,与生产代码两处
// 私有 helper 保持一致(而非套用 tests/cpp/ops/ 下会显式置 Layout::kRowMajor
// 的 MakeType——那份 helper 面向 shape_infer 单测场景,语义不同,不能直接
// 复用到本处的逐字节比较场景)。
TensorType MakeCpuTensorType(std::vector<int64_t> dims) {
  TensorType type;
  type.dtype = DType::of<float>();
  type.shape = Shape(std::move(dims));
  type.device = cpu_device();
  return type;
}

TEST(NnLinearSequentialGolden, MatchesManualMatmulAddReluGraphDumpText) {
  constexpr int64_t kBatch = 4;
  constexpr int64_t kInDim = 3;
  constexpr int64_t kOutDim = 5;

  // --- nn 构图:Sequential(Linear("0", with_bias=true), Relu("1")) ---
  Graph nn_graph("linear_relu");
  Value* nn_x = nn_graph.add_graph_input(MakeCpuTensorType({kBatch, kInDim})).value();

  const Module model = Sequential(
      "seq",
      {Linear("0", kBatch, kInDim, kOutDim, /*with_bias=*/true, DType::of<float>()), Relu("1")});
  const std::vector<ParamSpec> param_specs = model.parameters();
  ASSERT_EQ(param_specs.size(), 2u);  // seq.0.weight, seq.0.bias

  const Result<std::vector<Value*>> nn_params = add_parameter_inputs(nn_graph, param_specs);
  ASSERT_TRUE(nn_params.is_ok()) << nn_params.status().message();

  const Result<std::vector<Value*>> nn_outputs =
      model.build(nn_graph, std::vector<Value*>{nn_x}, nn_params.value());
  ASSERT_TRUE(nn_outputs.is_ok()) << nn_outputs.status().message();
  ASSERT_EQ(nn_outputs.value().size(), 1u);
  ASSERT_TRUE(nn_graph.mark_output(nn_outputs.value()[0]).is_ok());

  // --- 手工构图:照 AppendLayer 的节点序 matmul -> add(bias) -> relu ---
  Graph manual_graph("linear_relu");
  Value* manual_x = manual_graph.add_graph_input(MakeCpuTensorType({kBatch, kInDim})).value();
  Value* manual_weight = manual_graph.add_graph_input(MakeCpuTensorType({kInDim, kOutDim})).value();
  Value* manual_bias = manual_graph.add_graph_input(MakeCpuTensorType({kBatch, kOutDim})).value();

  const Result<Node*> matmul_node =
      create_node_with_inferred_types(manual_graph, "matmul", {manual_x, manual_weight});
  ASSERT_TRUE(matmul_node.is_ok()) << matmul_node.status().message();
  const Result<Node*> add_node = create_node_with_inferred_types(
      manual_graph, "add", {matmul_node.value()->output(0), manual_bias});
  ASSERT_TRUE(add_node.is_ok()) << add_node.status().message();
  const Result<Node*> relu_node =
      create_node_with_inferred_types(manual_graph, "relu", {add_node.value()->output(0)});
  ASSERT_TRUE(relu_node.is_ok()) << relu_node.status().message();
  ASSERT_TRUE(manual_graph.mark_output(relu_node.value()->output(0)).is_ok());

  // 两图各自先 verify() 通过(结构合法性前置),再比对 dump_text——同
  // tests/cpp/frontend/test_lowering.cpp::AssertLoweringMatchesGolden 手法
  // (本用例直接比较两份 Graph 的 dump_text,不落 golden 文本文件)。
  const OpQuery query = make_op_query();
  const Status nn_verify_status = nn_graph.verify(query);
  ASSERT_TRUE(nn_verify_status.is_ok()) << nn_verify_status.message();
  const Status manual_verify_status = manual_graph.verify(query);
  ASSERT_TRUE(manual_verify_status.is_ok()) << manual_verify_status.message();

  EXPECT_EQ(dump_text(nn_graph), dump_text(manual_graph));
}

}  // namespace
