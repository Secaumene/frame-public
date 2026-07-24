// R11 机械断言(M21 批3 T4,docs/plan/2026-07-18-batch3-m21-conv.md 第1.2/3
// 节;M22 批4 T3 扩,docs/plan/2026-07-19-batch4-m22-seq.md 验收硬门 §3.3):
// conv/pool/M22 序列批全部新算子的梯度微图必须只用"已注册且自身可微"的算子
// ——对 6+8=14 个公开算子(conv2d/conv1d/max_pool2d/avg_pool2d/reshape/
// sigmoid/tanh/rsqrt/softmax/layer_norm/transpose/concat/slice/gather)+
// 5+1=6 个内部算子(conv2d_grad_input_internal/conv2d_grad_filter_internal/
// max_pool2d_grad_internal/max_pool2d_select_internal/
// avg_pool2d_grad_internal/gather_grad_internal)逐个调用其
// OpSchema::gradient() 生成梯度微图,遍历微图内全部非 graph_input 节点的 op
// 名,断言 OpRegistry 中该 op 名的 schema.gradient() != nullptr,或该节点
// 0 输入(constant,不需要反向传播义务,square_gradient 的 constant(2) 先例
// 同一豁免口径)。另单独断言内部算子自身的 schema.gradient() != nullptr
// (R11 达成路径的直接证据:内部算子自身注册 GradientFn 使梯度微图集合封闭、
// 二阶可展开)。
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/graph_builder.h>
#include <frame/ops/op_registry.h>

#include "elementwise_op_test_helpers.h"

namespace {

using frame::DType;
using frame::Result;
using frame::Shape;
using frame::ir::Graph;
using frame::ir::kGraphInputOp;
using frame::ir::Node;
using frame::ops::AttrMap;
using frame::ops::NodeContext;
using frame::ops::OpRegistry;
using frame::ops::OpSchema;
using frame::ops::testing::MakeType;

// 对梯度微图内每个非 graph_input 节点核对 R11:该 op 已在 OpRegistry 注册,且
// 其 schema.gradient() != nullptr,或该节点 0 输入(constant 豁免)。用普通
// 函数(而非 TEST 内联)复用于 11 个算子各自的用例,ASSERT_/EXPECT_ 宏在
// void 返回类型的被调函数内合法(gtest 惯例)。
void AssertMicrographIsGradientClosed(const Graph& micrograph) {
  for (const Node* node : micrograph.topological_order()) {
    if (node->op() == kGraphInputOp) continue;
    const OpSchema* schema = OpRegistry::instance().find(node->op());
    ASSERT_NE(schema, nullptr) << "op '" << node->op() << "' not found in OpRegistry (R11)";
    const bool has_gradient = schema->gradient() != nullptr;
    const bool is_zero_input = node->inputs().empty();
    EXPECT_TRUE(has_gradient || is_zero_input)
        << "op '" << node->op()
        << "' in gradient micrograph has no registered GradientFn and is not a zero-input "
           "(constant) op (R11 violation)";
  }
}

// ---------------------------------------------------------------------------
// 6 个公开算子。
// ---------------------------------------------------------------------------

TEST(GradientClosureTest, SelectiveScanGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("selective_scan");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "selective_scan";
  const auto type = MakeType(DType::of<float>(), {2, 3});
  ctx.input_types = {type, type, type, type, type};

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

TEST(GradientClosureTest, Conv2dGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("conv2d");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "conv2d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 3, 3}),
                     MakeType(DType::of<float>(), {1, 1, 2, 2}),
                     MakeType(DType::of<float>(), {1})};  // 含 bias,覆盖 "sum" 分支
  const AttrMap attrs{
      {"stride", std::vector<int64_t>{1, 1}},
      {"padding", std::vector<int64_t>{0, 0}},
      {"groups", int64_t{1}},
  };
  ctx.attrs = &attrs;

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

TEST(GradientClosureTest, Conv1dGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("conv1d");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "conv1d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 4}),
                     MakeType(DType::of<float>(), {1, 1, 2}),
                     MakeType(DType::of<float>(), {1})};  // 含 bias
  const AttrMap attrs{
      {"stride", int64_t{1}},
      {"padding", int64_t{0}},
      {"groups", int64_t{1}},
  };
  ctx.attrs = &attrs;

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

TEST(GradientClosureTest, MaxPool2dGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("max_pool2d");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "max_pool2d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 4, 4})};
  const AttrMap attrs{
      {"kernel", std::vector<int64_t>{2, 2}},
      {"stride", std::vector<int64_t>{2, 2}},
      {"padding", std::vector<int64_t>{0, 0}},
  };
  ctx.attrs = &attrs;

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

TEST(GradientClosureTest, AvgPool2dGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("avg_pool2d");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "avg_pool2d";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 4, 4})};
  const AttrMap attrs{
      {"kernel", std::vector<int64_t>{2, 2}},
      {"stride", std::vector<int64_t>{2, 2}},
      {"padding", std::vector<int64_t>{0, 0}},
  };
  ctx.attrs = &attrs;

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

TEST(GradientClosureTest, ReshapeGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("reshape");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "reshape";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};
  const AttrMap attrs{{"target_shape", Shape({3, 2})}};
  ctx.attrs = &attrs;

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

TEST(GradientClosureTest, SigmoidGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("sigmoid");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "sigmoid";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

// ---------------------------------------------------------------------------
// 5 个内部算子。
// ---------------------------------------------------------------------------

TEST(GradientClosureTest, Conv2dGradInputInternalGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("conv2d_grad_input_internal");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "conv2d_grad_input_internal";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 2, 2}),
                     MakeType(DType::of<float>(), {1, 1, 2, 2})};
  const AttrMap attrs{
      {"input_shape", Shape({1, 1, 3, 3})},
      {"stride", std::vector<int64_t>{1, 1}},
      {"padding", std::vector<int64_t>{0, 0}},
      {"groups", int64_t{1}},
  };
  ctx.attrs = &attrs;

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

TEST(GradientClosureTest, Conv2dGradFilterInternalGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("conv2d_grad_filter_internal");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "conv2d_grad_filter_internal";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 3, 3}),
                     MakeType(DType::of<float>(), {1, 1, 2, 2})};
  const AttrMap attrs{
      {"filter_shape", Shape({1, 1, 2, 2})},
      {"stride", std::vector<int64_t>{1, 1}},
      {"padding", std::vector<int64_t>{0, 0}},
      {"groups", int64_t{1}},
  };
  ctx.attrs = &attrs;

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

TEST(GradientClosureTest, MaxPool2dGradInternalGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("max_pool2d_grad_internal");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "max_pool2d_grad_internal";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 2, 2}),
                     MakeType(DType::of<float>(), {1, 1, 4, 4})};
  const AttrMap attrs{
      {"input_shape", Shape({1, 1, 4, 4})},
      {"kernel", std::vector<int64_t>{2, 2}},
      {"stride", std::vector<int64_t>{2, 2}},
      {"padding", std::vector<int64_t>{0, 0}},
  };
  ctx.attrs = &attrs;

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

TEST(GradientClosureTest, MaxPool2dSelectInternalGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("max_pool2d_select_internal");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "max_pool2d_select_internal";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 4, 4}),
                     MakeType(DType::of<float>(), {1, 1, 4, 4})};
  const AttrMap attrs{
      {"kernel", std::vector<int64_t>{2, 2}},
      {"stride", std::vector<int64_t>{2, 2}},
      {"padding", std::vector<int64_t>{0, 0}},
  };
  ctx.attrs = &attrs;

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

TEST(GradientClosureTest, AvgPool2dGradInternalGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("avg_pool2d_grad_internal");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "avg_pool2d_grad_internal";
  ctx.input_types = {MakeType(DType::of<float>(), {1, 1, 2, 2})};
  const AttrMap attrs{
      {"input_shape", Shape({1, 1, 4, 4})},
      {"kernel", std::vector<int64_t>{2, 2}},
      {"stride", std::vector<int64_t>{2, 2}},
      {"padding", std::vector<int64_t>{0, 0}},
  };
  ctx.attrs = &attrs;

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

// ---------------------------------------------------------------------------
// 5 个内部算子自身 gradient() != nullptr 的独立断言(R11 达成路径的直接
// 证据,§1.2 表:全部新内部算子自身注册 GradientFn 使梯度微图集合封闭)。
// ---------------------------------------------------------------------------

TEST(GradientClosureTest, AllFiveInternalOpsHaveOwnGradientFnRegistered) {
  const std::vector<std::string> internal_ops{
      "conv2d_grad_input_internal", "conv2d_grad_filter_internal", "max_pool2d_grad_internal",
      "max_pool2d_select_internal", "avg_pool2d_grad_internal",
  };
  for (const std::string& op_name : internal_ops) {
    const OpSchema* schema = OpRegistry::instance().find(op_name);
    ASSERT_NE(schema, nullptr) << "op '" << op_name << "' not found in OpRegistry";
    EXPECT_NE(schema->gradient(), nullptr)
        << "internal op '" << op_name << "' must self-register a GradientFn (R11)";
  }
}

// ---------------------------------------------------------------------------
// M22(批4 T3,docs/plan/2026-07-19-batch4-m22-seq.md §1.2/1.4/1.5,验收硬门
// §3.3):9 个新算子(含 gather_grad_internal)的梯度微图 R11 机械断言——
// 7 个公开算子(tanh/rsqrt/softmax/layer_norm/transpose/concat/slice/gather,
// 实为 8 个,连同 1 个内部算子 gather_grad_internal 共 9 个)。
// ---------------------------------------------------------------------------

TEST(GradientClosureTest, TanhGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("tanh");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "tanh";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

TEST(GradientClosureTest, RsqrtGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("rsqrt");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "rsqrt";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

TEST(GradientClosureTest, SoftmaxGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("softmax");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "softmax";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

TEST(GradientClosureTest, LayerNormGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("layer_norm");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "layer_norm";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 4}), MakeType(DType::of<float>(), {4}),
                     MakeType(DType::of<float>(), {4})};
  const AttrMap attrs{{"eps", 1e-5}};
  ctx.attrs = &attrs;

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

TEST(GradientClosureTest, TransposeGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("transpose");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "transpose";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3, 4})};
  const AttrMap attrs{{"perm", std::vector<int64_t>{2, 0, 1}}};
  ctx.attrs = &attrs;

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

TEST(GradientClosureTest, ConcatGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("concat");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "concat";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 2}), MakeType(DType::of<float>(), {1, 2})};
  const AttrMap attrs{{"axis", int64_t{0}}};
  ctx.attrs = &attrs;

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

TEST(GradientClosureTest, SliceGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("slice");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "slice";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 4})};
  const AttrMap attrs{{"axis", int64_t{1}}, {"start", int64_t{1}}, {"stop", int64_t{3}}};
  ctx.attrs = &attrs;

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

TEST(GradientClosureTest, GatherGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("gather");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "gather";
  ctx.input_types = {MakeType(DType::of<float>(), {4, 2}),
                     MakeType(DType::of<std::int64_t>(), {3})};

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

TEST(GradientClosureTest, GatherGradInternalGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("gather_grad_internal");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "gather_grad_internal";
  ctx.input_types = {MakeType(DType::of<float>(), {3, 2}),
                     MakeType(DType::of<std::int64_t>(), {3})};
  const AttrMap attrs{{"input_shape", Shape({4, 2})}};
  ctx.attrs = &attrs;

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

// gather_grad_internal 自身 gradient()!=nullptr 的独立断言(R11 达成路径的
// 直接证据,同上方 AllFiveInternalOpsHaveOwnGradientFnRegistered 先例)。
TEST(GradientClosureTest, GatherGradInternalHasOwnGradientFnRegistered) {
  const OpSchema* schema = OpRegistry::instance().find("gather_grad_internal");
  ASSERT_NE(schema, nullptr) << "op 'gather_grad_internal' not found in OpRegistry";
  EXPECT_NE(schema->gradient(), nullptr)
      << "internal op 'gather_grad_internal' must self-register a GradientFn (R11)";
}

// ---------------------------------------------------------------------------
// M23(批5 T3,docs/plan/2026-07-21-batch5-m23-fft.md §1.3/验收硬门 §3):
// rfft/irfft 两个新算子,互引用(rfft 微图含 irfft 节点,irfft 微图含 rfft
// 节点,均已注册且自身可微)+ constant + mul,无新增内部算子。
// ---------------------------------------------------------------------------

TEST(GradientClosureTest, RfftGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("rfft");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "rfft";
  ctx.input_types = {MakeType(DType::of<float>(), {4})};  // n=4(偶)

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

TEST(GradientClosureTest, IrfftGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("irfft");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);

  NodeContext ctx;
  ctx.op = "irfft";
  ctx.input_types = {MakeType(DType::of<float>(), {3, 2})};  // k=3,对应 n=5(奇)
  const AttrMap attrs{{"n", int64_t{5}}};
  ctx.attrs = &attrs;

  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

// ---------------------------------------------------------------------------
// M26(ARCH-068):M17 遗留五个 internal 算子补齐 GradientFn,使一阶派生图
// 可再次执行同一反向变换。
// ---------------------------------------------------------------------------

TEST(GradientClosureTest, M26LegacyInternalGradientMicrographsAreClosed) {
  {
    const OpSchema* schema = OpRegistry::instance().find("sum_grad_internal");
    ASSERT_NE(schema, nullptr);
    ASSERT_NE(schema->gradient(), nullptr);
    NodeContext ctx;
    ctx.op = "sum_grad_internal";
    ctx.input_types = {MakeType(DType::of<float>(), {2, 1})};
    const AttrMap attrs{{"input_shape", Shape({2, 3})}, {"axes", std::vector<int64_t>{1}}};
    ctx.attrs = &attrs;
    const Result<Graph> micrograph = schema->gradient()(ctx);
    ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
    AssertMicrographIsGradientClosed(micrograph.value());
  }
  {
    const OpSchema* schema = OpRegistry::instance().find("matmul_grad_lhs_internal");
    ASSERT_NE(schema, nullptr);
    ASSERT_NE(schema->gradient(), nullptr);
    NodeContext ctx;
    ctx.op = "matmul_grad_lhs_internal";
    ctx.input_types = {MakeType(DType::of<float>(), {2, 4}), MakeType(DType::of<float>(), {3, 4})};
    const Result<Graph> micrograph = schema->gradient()(ctx);
    ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
    AssertMicrographIsGradientClosed(micrograph.value());
  }
  {
    const OpSchema* schema = OpRegistry::instance().find("matmul_grad_rhs_internal");
    ASSERT_NE(schema, nullptr);
    ASSERT_NE(schema->gradient(), nullptr);
    NodeContext ctx;
    ctx.op = "matmul_grad_rhs_internal";
    ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {2, 4})};
    const Result<Graph> micrograph = schema->gradient()(ctx);
    ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
    AssertMicrographIsGradientClosed(micrograph.value());
  }
  {
    const OpSchema* schema = OpRegistry::instance().find("mse_loss_grad_internal");
    ASSERT_NE(schema, nullptr);
    ASSERT_NE(schema->gradient(), nullptr);
    NodeContext ctx;
    ctx.op = "mse_loss_grad_internal";
    ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {2, 3}),
                       MakeType(DType::of<float>(), {})};
    const Result<Graph> micrograph = schema->gradient()(ctx);
    ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
    AssertMicrographIsGradientClosed(micrograph.value());
  }
  {
    const OpSchema* schema = OpRegistry::instance().find("relu_grad_internal");
    ASSERT_NE(schema, nullptr);
    ASSERT_NE(schema->gradient(), nullptr);
    NodeContext ctx;
    ctx.op = "relu_grad_internal";
    ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {2, 3})};
    const Result<Graph> micrograph = schema->gradient()(ctx);
    ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
    AssertMicrographIsGradientClosed(micrograph.value());
  }
}

TEST(GradientClosureTest, HeavisideSurrogateProxyGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("heaviside_surrogate");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);
  NodeContext ctx;
  ctx.op = "heaviside_surrogate";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};
  const AttrMap attrs{{"alpha", 2.0}};
  ctx.attrs = &attrs;
  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

TEST(GradientClosureTest, ScatterAddGradientMicrographIsClosed) {
  const OpSchema* schema = OpRegistry::instance().find("scatter_add");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->gradient(), nullptr);
  NodeContext ctx;
  ctx.op = "scatter_add";
  ctx.input_types = {MakeType(DType::of<float>(), {3, 2}), MakeType(DType::of<int64_t>(), {3})};
  const AttrMap attrs{{"output_shape", Shape({4, 2})}};
  ctx.attrs = &attrs;
  const Result<Graph> micrograph = schema->gradient()(ctx);
  ASSERT_TRUE(micrograph.is_ok()) << micrograph.status().message();
  AssertMicrographIsGradientClosed(micrograph.value());
}

}  // namespace
