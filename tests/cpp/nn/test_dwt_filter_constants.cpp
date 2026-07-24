// Dwt2d/Dwt1d 固定滤波器常量正确性测试(M21 批3 T6,docs/plan/
// 2026-07-18-batch3-m21-conv.md 第1.4节):build() 产出的 "constant" 节点,其
// value/shape/dtype 三属性须与设计文档给定的闭式系数逐一吻合(图内属性值
// 断言,不依赖任何后端执行——纯构图期白盒核对)。系数来源见
// src/nn/layers.cpp::Dwt2d/Dwt1d 头注释:Haar 2D 四子带
// {LL,LH,HL,HH}={[.5,.5,.5,.5],[.5,.5,-.5,-.5],[.5,-.5,.5,-.5],[.5,-.5,-.5,.5]}
// (计划 1.4 节字面量原文);Haar 1D 低通/高通 = ±1/sqrt(2) 组合;Db4(db2,4
// 抽头)标准闭式系数 h0..h3 + QMF 高通镜像关系。
#include <cmath>
#include <cstddef>
#include <cstdint>
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
using frame::nn::Dwt1d;
using frame::nn::Dwt2d;
using frame::nn::Module;
using frame::nn::WaveletKind;

TensorType MakeCpuTensorType(std::vector<int64_t> dims) {
  TensorType type;
  type.dtype = DType::of<float>();
  type.shape = Shape(std::move(dims));
  type.device = cpu_device();
  return type;
}

// 图内查找唯一的 "constant" 节点(Dwt2d/Dwt1d 的 build() 恰产 1 个)。
const Node* FindConstantNode(const Graph& graph) {
  for (const Node* node : graph.topological_order()) {
    if (node->op() == "constant") return node;
  }
  return nullptr;
}

TEST(DwtFilterConstants, Dwt2dHaarFilterValuesShapeAndDtypeMatchClosedForm) {
  constexpr int64_t kChannels = 2;
  Graph graph("dwt2d_haar");
  Value* x = graph.add_graph_input(MakeCpuTensorType({1, kChannels, 4, 4})).value();

  const Module dwt = Dwt2d("w", kChannels, WaveletKind::kHaar);
  const Result<std::vector<Value*>> outputs =
      dwt.build(graph, std::vector<Value*>{x}, std::vector<Value*>{});
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  const Node* constant_node = FindConstantNode(graph);
  ASSERT_NE(constant_node, nullptr);

  const Shape* shape = constant_node->attr<Shape>("shape");
  ASSERT_NE(shape, nullptr);
  EXPECT_EQ(*shape, Shape({4 * kChannels, 1, 2, 2}));

  const DType* dtype = constant_node->attr<DType>("dtype");
  ASSERT_NE(dtype, nullptr);
  EXPECT_EQ(*dtype, DType::of<float>());  // 取自输入 x 的静态 dtype

  const std::vector<double>* value = constant_node->attr<std::vector<double>>("value");
  ASSERT_NE(value, nullptr);

  const std::vector<double> kLL{0.5, 0.5, 0.5, 0.5};
  const std::vector<double> kLH{0.5, 0.5, -0.5, -0.5};
  const std::vector<double> kHL{0.5, -0.5, 0.5, -0.5};
  const std::vector<double> kHH{0.5, -0.5, -0.5, 0.5};
  std::vector<double> expected;
  expected.reserve(static_cast<size_t>(16 * kChannels));
  for (int64_t c = 0; c < kChannels; ++c) {
    for (double v : kLL) expected.push_back(v);
    for (double v : kLH) expected.push_back(v);
    for (double v : kHL) expected.push_back(v);
    for (double v : kHH) expected.push_back(v);
  }
  EXPECT_EQ(*value, expected);
}

TEST(DwtFilterConstants, Dwt1dHaarFilterMatchesClosedFormCoefficients) {
  constexpr int64_t kChannels = 3;
  Graph graph("dwt1d_haar");
  Value* x = graph.add_graph_input(MakeCpuTensorType({1, kChannels, 8})).value();

  const Module dwt = Dwt1d("w", kChannels, WaveletKind::kHaar);
  const Result<std::vector<Value*>> outputs =
      dwt.build(graph, std::vector<Value*>{x}, std::vector<Value*>{});
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  const Node* constant_node = FindConstantNode(graph);
  ASSERT_NE(constant_node, nullptr);
  const Shape* shape = constant_node->attr<Shape>("shape");
  ASSERT_NE(shape, nullptr);
  EXPECT_EQ(*shape, Shape({2 * kChannels, 1, 2}));

  const std::vector<double>* value = constant_node->attr<std::vector<double>>("value");
  ASSERT_NE(value, nullptr);

  const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  std::vector<double> expected;
  expected.reserve(static_cast<size_t>(4 * kChannels));
  for (int64_t c = 0; c < kChannels; ++c) {
    expected.push_back(inv_sqrt2);
    expected.push_back(inv_sqrt2);
    expected.push_back(inv_sqrt2);
    expected.push_back(-inv_sqrt2);
  }
  ASSERT_EQ(value->size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR((*value)[i], expected[i], 1e-12) << "index " << i;
  }
}

TEST(DwtFilterConstants, Dwt1dDb4FilterMatchesClosedFormCoefficients) {
  constexpr int64_t kChannels = 1;
  Graph graph("dwt1d_db4");
  Value* x = graph.add_graph_input(MakeCpuTensorType({1, kChannels, 8})).value();

  const Module dwt = Dwt1d("w", kChannels, WaveletKind::kDb4);
  const Result<std::vector<Value*>> outputs =
      dwt.build(graph, std::vector<Value*>{x}, std::vector<Value*>{});
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  const Node* constant_node = FindConstantNode(graph);
  ASSERT_NE(constant_node, nullptr);
  const Shape* shape = constant_node->attr<Shape>("shape");
  ASSERT_NE(shape, nullptr);
  EXPECT_EQ(*shape, Shape({2 * kChannels, 1, 4}));

  const std::vector<double>* value = constant_node->attr<std::vector<double>>("value");
  ASSERT_NE(value, nullptr);

  // 独立复算闭式系数(与 src/nn/layers.cpp::Dwt1d 同公式,作为测试侧独立
  // oracle):h0=(1+sqrt3)/(4*sqrt2)、h1=(3+sqrt3)/(4*sqrt2)、
  // h2=(3-sqrt3)/(4*sqrt2)、h3=(1-sqrt3)/(4*sqrt2);高通
  // g[k]=(-1)^k*h[3-k] = {h3,-h2,h1,-h0}。
  const double sqrt3 = std::sqrt(3.0);
  const double denom = 4.0 * std::sqrt(2.0);
  const double h0 = (1.0 + sqrt3) / denom;
  const double h1 = (3.0 + sqrt3) / denom;
  const double h2 = (3.0 - sqrt3) / denom;
  const double h3 = (1.0 - sqrt3) / denom;
  const std::vector<double> expected{h0, h1, h2, h3, h3, -h2, h1, -h0};
  ASSERT_EQ(value->size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR((*value)[i], expected[i], 1e-12) << "index " << i;
  }
}

}  // namespace
