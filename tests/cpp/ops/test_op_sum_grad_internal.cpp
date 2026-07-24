// sum_grad_internal 算子测试(M17,sum 的梯度 internal 算子;
// src/ops/schemas/reduction.cpp、src/backends/cpu/kernels/reduction.cpp 已
// 实化的行为)。sum_grad_internal(gy) 沿 axes 把 gy 复制展开回 input_shape
// (project_reduced_linear_index 反向广播,见 kernel 头注释);attrs =
// input_shape(kShape) + axes(kInt64Array),kernel 按 gy 实际 rank 自行判定
// sum 原节点是否用了 keepdims(见 kernel 头注释歧义论证)。
//   1. OpRegistry::find 的 schema 字段(1 输入 1 输出,两个必填属性)、
//      shape_infer()/decomposition()/gradient() 三函数指针状态;
//   2. eager 数值路径(ARCH-011 第 3 类):单轴归约 keepdims=false/true 两种
//      gy 形态(数值应一致,验证"歧义论证"实测成立)+ 全维归约;
//   3. kernel 自身的防御性拒绝路径(dtype 不支持 / gy rank 与
//      input_shape+axes 不自洽 / out shape 不符 input_shape)。
#include <cstdint>
#include <gtest/gtest.h>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/ir/attribute.h>
#include <frame/ir/node.h>
#include <frame/ops/kernel_registry.h>
#include <frame/ops/op_registry.h>

#include "../common/tolerance.h"
#include "elementwise_op_test_helpers.h"

namespace {

using frame::cpu_device;
using frame::DType;
using frame::DTypeCode;
using frame::ErrorCode;
using frame::Shape;
using frame::Tensor;
using frame::ir::AttrType;
using frame::ir::AttrValue;
using frame::ops::NodeContext;
using frame::ops::OpRegistry;
using frame::ops::OpSchema;
using frame::ops::testing::MakeType;
using frame::testing::default_tolerance;
using frame::testing::tensor_all_close;

// ---------------------------------------------------------------------------
// 1. schema 断言。
// ---------------------------------------------------------------------------

TEST(SumGradInternalOpSchemaTest, RegisteredWithOneInputOneOutputTwoAttrs) {
  const OpSchema* schema = OpRegistry::instance().find("sum_grad_internal");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 1u);
  EXPECT_EQ(schema->outputs().size(), 1u);
  ASSERT_EQ(schema->attrs().size(), 2u);
  EXPECT_EQ(schema->attrs()[0].name, "input_shape");
  EXPECT_EQ(schema->attrs()[0].type, AttrType::kShape);
  EXPECT_TRUE(schema->attrs()[0].required);
  EXPECT_EQ(schema->attrs()[1].name, "axes");
  EXPECT_EQ(schema->attrs()[1].type, AttrType::kInt64Array);
  EXPECT_TRUE(schema->attrs()[1].required);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
  EXPECT_NE(schema->gradient(), nullptr);  // M26:二阶图变换闭包(ARCH-068)
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

TEST(SumGradInternalShapeInferTest, OutputShapeEqualsInputShapeAttr) {
  const OpSchema* schema = OpRegistry::instance().find("sum_grad_internal");
  ASSERT_NE(schema, nullptr);
  const std::unordered_map<std::string, AttrValue> attrs{
      {"input_shape", Shape({2, 3})},
      {"axes", std::vector<int64_t>{0}},
  };
  NodeContext ctx;
  ctx.op = "sum_grad_internal";
  ctx.input_types = {MakeType(DType::of<float>(), {3})};
  ctx.attrs = &attrs;
  const frame::Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok());
  EXPECT_EQ(result.value()[0], Shape({2, 3}));
}

// ---------------------------------------------------------------------------
// 3. eager 数值(ARCH-011 第 3 类)。
// ---------------------------------------------------------------------------

class SumGradInternalOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(SumGradInternalOpEagerTest, Axis0KeepdimsFalseBroadcastsGyAcrossReducedAxis) {
  // input_shape=[2,3],axes=[0](归约 axis0,尺寸2);gy=[10,20,30](keepdims=
  // false 形态,shape[3]);期望 gx=[[10,20,30],[10,20,30]]。
  Tensor gy = MakeTensor1D<float>({10.0f, 20.0f, 30.0f});
  Tensor expected =
      MakeTensorWithShape<float>(Shape({2, 3}), {10.0f, 20.0f, 30.0f, 10.0f, 20.0f, 30.0f});
  Tensor out = MakeTensorWithShape<float>(Shape({2, 3}), {0, 0, 0, 0, 0, 0});

  const std::unordered_map<std::string, AttrValue> attrs{
      {"input_shape", Shape({2, 3})},
      {"axes", std::vector<int64_t>{0}},
  };
  std::vector<Tensor> inputs{gy};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "sum_grad_internal";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(SumGradInternalOpEagerTest, Axis0KeepdimsTrueProducesSameBroadcastResult) {
  // 同上 input_shape/axes,但 gy 改为 keepdims=true 形态(shape[1,3],同一组
  // 数值);kernel 应按 gy 实际 rank 自行判定并产出相同结果(歧义论证实测)。
  Tensor gy = MakeTensorWithShape<float>(Shape({1, 3}), {10.0f, 20.0f, 30.0f});
  Tensor expected =
      MakeTensorWithShape<float>(Shape({2, 3}), {10.0f, 20.0f, 30.0f, 10.0f, 20.0f, 30.0f});
  Tensor out = MakeTensorWithShape<float>(Shape({2, 3}), {0, 0, 0, 0, 0, 0});

  const std::unordered_map<std::string, AttrValue> attrs{
      {"input_shape", Shape({2, 3})},
      {"axes", std::vector<int64_t>{0}},
  };
  std::vector<Tensor> inputs{gy};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "sum_grad_internal";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(SumGradInternalOpEagerTest, EmptyAxesFullReductionBroadcastsScalarGyToAllElements) {
  // axes=[](全维归约),gy 为 rank-0 标量 7.0;期望 input_shape=[2,3] 全部
  // 6 个元素均为 7.0。
  Tensor gy = MakeTensorWithShape<float>(Shape(), {7.0f});
  Tensor expected = MakeTensorWithShape<float>(Shape({2, 3}), {7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f});
  Tensor out = MakeTensorWithShape<float>(Shape({2, 3}), {0, 0, 0, 0, 0, 0});

  const std::unordered_map<std::string, AttrValue> attrs{
      {"input_shape", Shape({2, 3})},
      {"axes", std::vector<int64_t>{}},
  };
  std::vector<Tensor> inputs{gy};
  std::vector<Tensor> outputs{out};
  frame::hal::KernelInvocation invocation;
  invocation.op = "sum_grad_internal";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;

  const frame::Status status = backend_->launch(invocation, nullptr);
  ASSERT_TRUE(status.is_ok());
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

// ---------------------------------------------------------------------------
// 4. sum_grad_internal_cpu_kernel 自身的防御性拒绝路径。
// ---------------------------------------------------------------------------

struct SumGradInternalOpNameTag {
  static constexpr std::string_view kOpName = "sum_grad_internal";
};
using SumGradInternalOpKernelTest =
    frame::ops::testing::ElementwiseKernelTestBase<SumGradInternalOpNameTag>;

TEST_F(SumGradInternalOpKernelTest, RejectsUnsupportedDtypeInt32) {
  Tensor gy = MakeTensor<std::int32_t>(Shape({3}));
  Tensor gx = MakeTensor<std::int32_t>(Shape({2, 3}));

  std::vector<Tensor> inputs{gy};
  std::vector<Tensor> outputs{gx};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("does not support dtype 'int32'"), std::string_view::npos);
}

TEST_F(SumGradInternalOpKernelTest, RejectsGyRankInconsistentWithInputShapeAndAxes) {
  // input_shape=[2,3],axes=[0](归约 1 维)=> 仅接受 gy rank 2(keepdims)或
  // rank 1(无 keepdims)两种候选;这里给 gy shape [1,1,3](rank 3),两种都不
  // 是,应拒绝。
  Tensor gy = MakeTensor<float>(Shape({1, 1, 3}));
  Tensor gx = MakeTensor<float>(Shape({2, 3}));

  const std::unordered_map<std::string, AttrValue> attrs{
      {"input_shape", Shape({2, 3})},
      {"axes", std::vector<int64_t>{0}},
  };
  std::vector<Tensor> inputs{gy};
  std::vector<Tensor> outputs{gx};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.attrs = &attrs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("gy rank 3 is inconsistent with input_shape rank 2"),
            std::string_view::npos);
}

TEST_F(SumGradInternalOpKernelTest, RejectsOutShapeMismatchWithInputShapeAttr) {
  Tensor gy = MakeTensor<float>(Shape({3}));
  Tensor gx = MakeTensor<float>(Shape({2, 4}));  // 与 input_shape=[2,3] 不符

  const std::unordered_map<std::string, AttrValue> attrs{
      {"input_shape", Shape({2, 3})},
      {"axes", std::vector<int64_t>{0}},
  };
  std::vector<Tensor> inputs{gy};
  std::vector<Tensor> outputs{gx};
  frame::ops::KernelContext ctx;
  ctx.inputs = inputs;
  ctx.outputs = outputs;
  ctx.attrs = &attrs;
  ctx.device = cpu_device();

  const frame::Status status = kernel_(ctx);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("requires gx(out) shape to match attribute 'input_shape'"),
            std::string_view::npos);
}

}  // namespace
