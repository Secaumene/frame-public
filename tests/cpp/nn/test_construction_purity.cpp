// Module 构图纯度测试(ARCH-072,docs/architecture/nn-design.md §2/§8)。
//
// 判定要点(设计文档口径,本文件不重复实现层的机械检查、只在此记述判据
// 来源,避免"勿造伪断言"):
//   ①编译期事实——nn API 面(include/frame/nn/module.h、layers.h)与其实现
//     (src/nn/module.cpp、layers.cpp)均不 include <frame/core/tensor.h>,
//     BuildFn 签名(module.h)全程只出现 ir::Graph/ir::Value*,不出现 Tensor
//     类型;`grep -rn "Tensor::empty|raw_data|KernelContext" include/frame/nn/
//     src/nn/` 零命中——该 grep 是 ARCH-072 判定方法原文,机械检查已由
//     scripts/check_iron_rules.sh 覆盖,归实现方/pre-commit 职责,本文件不
//     重复跑该 grep(测试文件不适合断言"某目录 grep 命中数",那是构建期检查
//     的职责),只以本段注释记述判据来源。
//   ②运行期补证——build() 全程只经 ir::Graph 的公开构图 API(create_node/
//     add_graph_input)产出节点与 Value,不触碰任何 hal::Allocator/Tensor;
//     本文件断言 build 前后图节点计数增量恰等于预期新增 op 节点数、返回的
//     Value* 均可被同一 Graph 后续接纳为新节点输入——后者借道
//     Graph::create_node 对"输入 producer 是否属本图"的现成不变量(同
//     tests/cpp/ir/test_graph_construction.cpp::
//     CreateNodeRejectsValueFromAnotherGraph 反向印证的同一机制,非本文件
//     自造),以此代替"build 不触碰 allocator"这一运行期不易直测的断言
//     (spec 允许的口径:「无 hal 依赖(编译期已证)+ 纯 ir 产物」)。
#include <algorithm>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/nn/layers.h>
#include <frame/nn/module.h>

namespace {

using frame::cpu_device;
using frame::DType;
using frame::Result;
using frame::Shape;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::TensorType;
using frame::ir::Value;
using frame::nn::add_parameter_inputs;
using frame::nn::Linear;
using frame::nn::Module;
using frame::nn::ParamSpec;
using frame::nn::Relu;
using frame::nn::Sequential;

TensorType MakeCpuTensorType(std::vector<int64_t> dims) {
  TensorType type;
  type.dtype = DType::of<float>();
  type.shape = Shape(std::move(dims));
  type.device = cpu_device();
  return type;
}

// 核对 value 确属 graph:①直接扫描 topological_order() 核对其 producer 在列;
// ②借道 Graph::create_node 对跨图引用的现成拒绝机制(把 value 作为新建探测
// 节点的输入,创建成功即间接证明 producer 属本图——见
// tests/cpp/ir/test_graph_construction.cpp::CreateNodeRejectsValueFromAnotherGraph
// 的反向印证)。双重印证,不依赖单一机制。
void ExpectValueBelongsToGraph(Graph& graph, Value* value) {
  const std::vector<Node*>& topo = graph.topological_order();
  EXPECT_NE(std::find(topo.begin(), topo.end(), value->producer()), topo.end());

  const Result<Node*> probe = graph.create_node("relu", {value}, {value->type()});
  EXPECT_TRUE(probe.is_ok()) << (probe.is_ok() ? std::string() : probe.status().message());
}

TEST(ModuleConstructionPurity, LinearWithBiasBuildOnlyAddsExpectedIrNodes) {
  Graph graph("purity_linear_with_bias");
  Value* x = graph.add_graph_input(MakeCpuTensorType({4, 3})).value();
  ASSERT_EQ(graph.topological_order().size(), 1u);  // 仅 x 的 graph_input 节点

  const Module linear = Linear("fc", /*batch=*/4, /*in_dim=*/3, /*out_dim=*/5,
                               /*with_bias=*/true, DType::of<float>());
  const std::vector<ParamSpec> param_specs = linear.parameters();
  ASSERT_EQ(param_specs.size(), 2u);  // weight + bias

  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, param_specs);
  ASSERT_TRUE(params.is_ok()) << params.status().message();
  ASSERT_EQ(graph.topological_order().size(), 3u);  // x + weight + bias 三个 graph_input 节点
  ASSERT_EQ(graph.inputs().size(), 3u);

  const size_t pre_build_node_count = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      linear.build(graph, std::vector<Value*>{x}, params.value());
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  ASSERT_EQ(outputs.value().size(), 1u);

  // with_bias=true 的 Linear::build 恰构 2 个新节点(matmul、add),
  // 见 src/nn/layers.cpp::Linear——纯 ir 产物计数断言(ARCH-072①)。
  const size_t post_build_node_count = graph.topological_order().size();
  EXPECT_EQ(post_build_node_count - pre_build_node_count, 2u);
  EXPECT_EQ(graph.inputs().size(),
            3u);  // build 不产生新的图输入(参数已在 add_parameter_inputs 阶段登记)

  ExpectValueBelongsToGraph(graph, outputs.value()[0]);
  EXPECT_EQ(outputs.value()[0]->producer()->op(), "add");  // with_bias 分支末节点是 add
}

TEST(ModuleConstructionPurity, LinearWithoutBiasBuildOnlyAddsOneMatmulNode) {
  Graph graph("purity_linear_no_bias");
  Value* x = graph.add_graph_input(MakeCpuTensorType({2, 6})).value();

  const Module linear = Linear("fc", /*batch=*/2, /*in_dim=*/6, /*out_dim=*/3,
                               /*with_bias=*/false, DType::of<float>());
  const std::vector<ParamSpec> param_specs = linear.parameters();
  ASSERT_EQ(param_specs.size(), 1u);  // weight only

  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, param_specs);
  ASSERT_TRUE(params.is_ok()) << params.status().message();

  const size_t pre_build_node_count = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      linear.build(graph, std::vector<Value*>{x}, params.value());
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  EXPECT_EQ(graph.topological_order().size() - pre_build_node_count, 1u);  // 仅 matmul
  ExpectValueBelongsToGraph(graph, outputs.value()[0]);
  EXPECT_EQ(outputs.value()[0]->producer()->op(), "matmul");
}

TEST(ModuleConstructionPurity, SequentialLinearReluBuildAddsExpectedNodesAndChainsOutput) {
  Graph graph("purity_sequential");
  Value* x = graph.add_graph_input(MakeCpuTensorType({4, 3})).value();

  const Module model = Sequential("seq", {Linear("0", /*batch=*/4, /*in_dim=*/3, /*out_dim=*/5,
                                                 /*with_bias=*/false, DType::of<float>()),
                                          Relu("1")});
  const std::vector<ParamSpec> param_specs = model.parameters();
  ASSERT_EQ(param_specs.size(), 1u);  // 仅 Linear("0") 的 weight(with_bias=false)

  const Result<std::vector<Value*>> params = add_parameter_inputs(graph, param_specs);
  ASSERT_TRUE(params.is_ok()) << params.status().message();

  const size_t pre_build_node_count = graph.topological_order().size();
  const Result<std::vector<Value*>> outputs =
      model.build(graph, std::vector<Value*>{x}, params.value());
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  // matmul(Linear) + relu(Relu) 恰 2 个新节点(Sequential 自身不产生额外节点)。
  EXPECT_EQ(graph.topological_order().size() - pre_build_node_count, 2u);
  ExpectValueBelongsToGraph(graph, outputs.value()[0]);
  EXPECT_EQ(outputs.value()[0]->producer()->op(), "relu");
}

}  // namespace
