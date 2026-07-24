// square 算子测试(schema + decomposition + cpu kernel,
// src/ops/schemas/elementwise.cpp、src/backends/cpu/kernels/elementwise.cpp
// 已实化的行为)。square 是全仓首个注册 decomposition 的算子(m5-design-brief
// 决议点 4):square(x) = mul(x, x),decomposition 是 M10 回退链②的测试素材,
// **不替代** cpu kernel(ARCH-041 二者成对注册)。
//   1. OpRegistry::find("square") 的 schema 字段(1 输入 1 输出)、traits
//      命中/未命中(kElementwise+kFusable 命中,kCommutative+kHasSideEffect
//      未命中)、shape_infer() 非空且 decomposition() 非空(全仓首例);
//   2. infer_square_shape 的恒等合法路径 + 输入数≠1 拒绝路径;
//   3. decomposition 微图测试(本算子的核心交付):结构断言(inputs/outputs
//      各 1、拓扑序恰 1 个 op=="mul" 节点、该节点两个输入引用同一个 Value*、
//      经 make_op_query() 的 graph.verify 全量通过)+ dump_text golden 逐字节
//      比对(内联字符串,不放 compiler/testdata/——本用例非 compiler pass
//      golden,ARCH-051/052 的 testdata 目录规范不适用)+ 输入数≠2 的拒绝
//      路径(消息精确匹配 src 原文);
//   4. eager 数值路径(ARCH-011 第 3 类:单算子单元测试,tests/ 不在
//      Backend::launch 调用点白名单扫描范围,天然放行):fp32 覆盖正/零/负值,
//      fp16/bf16 各一条位级构造已知值;
//   5. square_cpu_kernel 自身的防御性拒绝路径(dtype 不在 v0 支持的三档浮点
//      内 / x·out shape 不一致),经 KernelRegistry::find("square",
//      kCpuBackendName) 直接取 KernelFn 调用驱动(不经 Backend::launch 这层
//      薄壳,聚焦 kernel 自身的校验逻辑)。
//
// 共用设施(MakeType/eager fixture/kernel fixture 模板)复用
// tests/cpp/ops/elementwise_op_test_helpers.h(与 test_op_add.cpp/
// test_op_mul.cpp/test_op_relu.cpp/test_op_sum.cpp/test_op_matmul.cpp 共用,
// REUSE-002)。全程复用 cpu 后端真实 Allocator(经 BackendRegistry 取得,hal
// 已实化),不使用 FakeAllocator。本文件不新增任何 op/kernel 注册,仅消费已由
// src/ 静态注册好的 "square"(decomposition 内部产出的微图会引用已注册的
// "mul",但本文件不重新注册 mul,只读取既有注册状态)。
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ir/serialization.h>
#include <frame/ops/kernel_registry.h>
#include <frame/ops/op_registry.h>

#include "../common/tolerance.h"
#include "elementwise_op_test_helpers.h"

namespace {

using frame::bfloat16_t;
using frame::cpu_device;
using frame::DType;
using frame::DTypeCode;
using frame::ErrorCode;
using frame::float16_t;
using frame::Shape;
using frame::Tensor;
using frame::ir::dump_text;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::OpQuery;
using frame::ops::NodeContext;
using frame::ops::OpRegistry;
using frame::ops::OpSchema;
using frame::ops::OpTrait;
using frame::ops::testing::MakeType;
using frame::testing::default_tolerance;
using frame::testing::tensor_all_close;

// ---------------------------------------------------------------------------
// 1. schema 断言。
// ---------------------------------------------------------------------------

TEST(SquareOpSchemaTest, RegisteredAndFindable) {
  const OpSchema* schema = OpRegistry::instance().find("square");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->name(), "square");
}

TEST(SquareOpSchemaTest, HasOneInputAndOneOutput) {
  const OpSchema* schema = OpRegistry::instance().find("square");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 1u);
  EXPECT_EQ(schema->outputs().size(), 1u);
}

TEST(SquareOpSchemaTest, TraitsMatchElementwiseFusableOnlyNotCommutative) {
  const OpSchema* schema = OpRegistry::instance().find("square");
  ASSERT_NE(schema, nullptr);
  EXPECT_TRUE(schema->has_trait(OpTrait::kElementwise));
  EXPECT_TRUE(schema->has_trait(OpTrait::kFusable));
  // square 是一元算子:kCommutative 语义仅适用于二元可交换运算,不应标注。
  EXPECT_FALSE(schema->has_trait(OpTrait::kCommutative));
  EXPECT_FALSE(schema->has_trait(OpTrait::kHasSideEffect));
}

TEST(SquareOpSchemaTest, HasShapeInferAndDecomposition) {
  const OpSchema* schema = OpRegistry::instance().find("square");
  ASSERT_NE(schema, nullptr);
  EXPECT_NE(schema->shape_infer(), nullptr);
  // square 是全仓首个注册 decomposition 的算子(m5-design-brief 决议点 4)。
  EXPECT_NE(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);  // M17:square_gradient 注册读回(ARCH-063)
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

TEST(SquareShapeInferTest, OutputShapeIsIdenticalToInputShape) {
  const OpSchema* schema = OpRegistry::instance().find("square");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  NodeContext ctx;
  ctx.op = "square";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok());
  ASSERT_EQ(result.value().size(), 1u);
  EXPECT_EQ(result.value()[0], Shape({2, 3}));
}

TEST(SquareShapeInferTest, WrongInputCountIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find("square");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  NodeContext ctx;
  ctx.op = "square";
  // square 是一元算子,schema 要求恰 1 个输入;这里给 2 个触发拒绝路径。
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {2, 3})};

  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_EQ(result.status().message(), "op 'square' expects 1 input, got 2");
}

// ---------------------------------------------------------------------------
// 3. decomposition 微图测试(本算子的核心交付)。
// ---------------------------------------------------------------------------

TEST(SquareDecompositionTest, ProducesExpectedMicrographStructure) {
  const OpSchema* schema = OpRegistry::instance().find("square");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->decomposition(), nullptr);

  NodeContext ctx;
  ctx.op = "square";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};

  const frame::Result<Graph> result = schema->decomposition()(ctx);
  ASSERT_TRUE(result.is_ok());
  const Graph& graph = result.value();

  ASSERT_EQ(graph.inputs().size(), 1u);
  ASSERT_EQ(graph.outputs().size(), 1u);

  // 拓扑序恰 1 个 op=="mul" 节点(topological_order() 还含 add_graph_input
  // 内部创建的 "graph_input" 节点,见 include/frame/ir/graph.h::add_graph_input
  // 实现,故不能直接断言 topological_order().size()==1,须按 op 名过滤计数)。
  const Node* mul_node = nullptr;
  int mul_count = 0;
  for (const Node* node : graph.topological_order()) {
    if (node->op() == "mul") {
      ++mul_count;
      mul_node = node;
    }
  }
  ASSERT_EQ(mul_count, 1);
  ASSERT_NE(mul_node, nullptr);
  ASSERT_EQ(mul_node->inputs().size(), 2u);
  // square(x) = mul(x, x):两个输入引用同一个 Value*(而非两个独立但类型相同
  // 的 Value)。
  EXPECT_EQ(mul_node->inputs()[0], mul_node->inputs()[1]);

  const OpQuery query = frame::ops::make_op_query();
  const frame::Status verify_status = graph.verify(query);
  EXPECT_TRUE(verify_status.is_ok()) << verify_status.message();
}

TEST(SquareDecompositionTest, DumpTextMatchesGolden) {
  const OpSchema* schema = OpRegistry::instance().find("square");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->decomposition(), nullptr);

  NodeContext ctx;
  ctx.op = "square";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3})};

  const frame::Result<Graph> result = schema->decomposition()(ctx);
  ASSERT_TRUE(result.is_ok());

  // golden 逐字节比对(内联字符串,设计定案:非 compiler pass golden,不放
  // tests/cpp/compiler/testdata/)。格式权威见
  // include/frame/ir/serialization.h 头注释;id 0 是 add_graph_input 内部
  // 创建的 "graph_input" 节点的唯一输出,id 1 是 mul 节点的唯一输出。MakeType
  // 恒置 layout=kRowMajor(elementwise_op_test_helpers.h),故 M9 起每个类型
  // 后缀均带 ":row_major" 尾缀(serialization.h 头注释第3a条)。
  const std::string expected =
      "%0 = graph_input() : float32[2,3]@cpu:0:row_major\n"
      "%1 = mul(%0, %0) : float32[2,3]@cpu:0:row_major\n"
      "graph_output(%1)\n";
  EXPECT_EQ(dump_text(result.value()), expected);
}

TEST(SquareDecompositionTest, RejectsWrongInputCount) {
  const OpSchema* schema = OpRegistry::instance().find("square");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->decomposition(), nullptr);

  NodeContext ctx;
  ctx.op = "square";
  ctx.input_types = {MakeType(DType::of<float>(), {2, 3}), MakeType(DType::of<float>(), {2, 3})};

  const frame::Result<Graph> result = schema->decomposition()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_EQ(result.status().message(), "op 'square' decomposition expects 1 input, got 2");
}

// ---------------------------------------------------------------------------
// 4. eager 数值(ARCH-011 第 3 类:单算子单元测试)。经 BackendRegistry 取 cpu
//    后端真实 Allocator 构造输入/输出 Tensor,Backend::launch 执行 "square"。
// ---------------------------------------------------------------------------

class SquareOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(SquareOpEagerTest, Float32SquareComputesElementwiseSquare) {
  // square([-2,0,3]) = [4,0,9]:覆盖负值/零/正值三种输入。
  Tensor x = MakeTensor1D<float>({-2.0f, 0.0f, 3.0f});
  Tensor expected = MakeTensor1D<float>({4.0f, 0.0f, 9.0f});
  Tensor out = MakeTensor1D<float>({0.0f, 0.0f, 0.0f});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "square";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(SquareOpEagerTest, Float16SquareViaBitLevelValues) {
  // 位级构造 fp16 已知值(与 test_op_add.cpp/test_op_matmul.cpp 复用同一组已
  // 交叉验证过的位模式):
  //   square(-2.0(0xC000)) = 4.0(0x4400);square(1.5(0x3E00)) = 2.25(0x4080)。
  Tensor x = MakeTensor1D<float16_t>({float16_t{0xC000u}, float16_t{0x3E00u}});
  Tensor expected = MakeTensor1D<float16_t>({float16_t{0x4400u}, float16_t{0x4080u}});
  Tensor out = MakeTensor1D<float16_t>({float16_t{0x0000u}, float16_t{0x0000u}});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "square";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat16)));
}

TEST_F(SquareOpEagerTest, BFloat16SquareViaBitLevelValues) {
  // 位级构造 bf16 已知值(与上一条 fp16 用例同一组十进制值,位模式不同,均
  // 已在既有测试文件中交叉验证过):
  //   square(-2.0(0xC000)) = 4.0(0x4080);square(1.5(0x3FC0)) = 2.25(0x4010)。
  Tensor x = MakeTensor1D<bfloat16_t>({bfloat16_t{0xC000u}, bfloat16_t{0x3FC0u}});
  Tensor expected = MakeTensor1D<bfloat16_t>({bfloat16_t{0x4080u}, bfloat16_t{0x4010u}});
  Tensor out = MakeTensor1D<bfloat16_t>({bfloat16_t{0x0000u}, bfloat16_t{0x0000u}});

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "square";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kBFloat16)));
}

// ---------------------------------------------------------------------------
// 5. square_cpu_kernel 自身的防御性拒绝路径:经
//    KernelRegistry::find("square", cpu) 直接取 KernelFn 调用,不经
//    Backend::launch(聚焦 kernel 校验逻辑本身)。
// ---------------------------------------------------------------------------

struct SquareOpNameTag {
  static constexpr std::string_view kOpName = "square";
};
using SquareOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<SquareOpNameTag>;

TEST_F(SquareOpKernelTest, RejectsUnsupportedDtypeInt32) {
  Tensor x = MakeTensor<std::int32_t>(Shape({2}));
  Tensor out = MakeTensor<std::int32_t>(Shape({2}));

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("does not support dtype 'int32'"), std::string_view::npos);
}

TEST_F(SquareOpKernelTest, RejectsShapeMismatchBetweenXAndOut) {
  Tensor x = MakeTensor<float>(Shape({2, 3}));
  Tensor out = MakeTensor<float>(Shape({3, 2}));  // 与 x 不一致

  std::vector<Tensor> inputs{x};
  std::vector<Tensor> outputs{out};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("requires x/out of the same shape"), std::string_view::npos);
}

}  // namespace
