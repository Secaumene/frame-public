// constant 算子测试(schema + ops 层物化/编码 helper + cpu kernel + 编译路径,
// M8 决议点 A,src/ops/schemas/constant.cpp、
// src/backends/cpu/kernels/constant.cpp、include/frame/ops/constant_utils.h
// 已实化的行为):
//   1. OpRegistry::find("constant") 的 schema 字段(0 输入 1 输出、零 trait、
//      shape_infer() 非空且 decomposition() 为空);
//   2. infer_constant_shape 的合法路径 + 报错路径(缺 value/shape/dtype 属性、
//      value 元素数与 shape.numel() 不符、dtype 白名单外、shape 含动态维、
//      输入数不为 0);
//   3. ops::fill_tensor_from_constant_attrs / ops::encode_tensor_to_attrs 的
//      物化数值:fp32 精确;fp16/bf16 经 encode_tensor_to_attrs ->
//      fill_tensor_from_constant_attrs 位级往返(逐元素比特相等,把
//      include/frame/ops/constant_utils.h 头注释的精度论证机械化——不用容差
//      工具,因为这里断言的是"位模式经双精度中转后精确复原",不是"数值在容差
//      内接近",EXPECT_EQ 位模式才是这个论断本身);
//   4. 编译路径执行:仅含一个 constant 节点的图经 runtime::compile("cpu")
//      编译执行,输出与 attrs 描述的数据一致(BUILD-011 容差);同图经
//      shape_inference pass 单独直接运行同样成功(design-reviewer 必须修复
//      4 的回归测试——0 输入豁免自 M8 起由 constant 实际使用,
//      src/compiler/passes/shape_inference.cpp:85-88)。
//
// 共用设施(MakeType/eager fixture)复用 tests/cpp/ops/elementwise_op_test_helpers.h
// (REUSE-002,与 test_op_add.cpp 等文件共用)。全程复用 cpu 后端真实 Allocator
// (经 BackendRegistry 取得,hal 已实化),不使用 FakeAllocator。本文件不新增
// 任何 op/kernel 注册,仅消费已由 src/ 静态注册好的 "constant"。
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <frame/compiler/pass.h>
#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/executable.h>
#include <frame/hal/stream.h>
#include <frame/ir/attribute.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/constant_utils.h>
#include <frame/ops/op_registry.h>
#include <frame/runtime/compile.h>

#include "../common/tolerance.h"
#include "elementwise_op_test_helpers.h"

namespace {

using frame::bfloat16_t;
using frame::DType;
using frame::DTypeCode;
using frame::ErrorCode;
using frame::float16_t;
using frame::kDynamicDim;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::Tensor;
using frame::compiler::Pass;
using frame::compiler::PassRegistry;
using frame::hal::CompileOptions;
using frame::hal::Executable;
using frame::ir::AttrValue;
using frame::ir::Graph;
using frame::ops::kConstantOpName;
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

TEST(ConstantOpSchemaTest, RegisteredAndFindable) {
  const OpSchema* schema = OpRegistry::instance().find(kConstantOpName);
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->name(), kConstantOpName);
}

TEST(ConstantOpSchemaTest, HasZeroInputsAndOneOutput) {
  const OpSchema* schema = OpRegistry::instance().find(kConstantOpName);
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 0u);
  EXPECT_EQ(schema->outputs().size(), 1u);
}

TEST(ConstantOpSchemaTest, HasNoTraits) {
  const OpSchema* schema = OpRegistry::instance().find(kConstantOpName);
  ASSERT_NE(schema, nullptr);
  // 常量本身无副作用、非逐元素、非可交换(决议点 A:不标注任何 trait)。
  EXPECT_FALSE(schema->has_trait(OpTrait::kElementwise));
  EXPECT_FALSE(schema->has_trait(OpTrait::kFusable));
  EXPECT_FALSE(schema->has_trait(OpTrait::kHasSideEffect));
  EXPECT_FALSE(schema->has_trait(OpTrait::kCommutative));
}

TEST(ConstantOpSchemaTest, HasShapeInferButNoDecomposition) {
  const OpSchema* schema = OpRegistry::instance().find(kConstantOpName);
  ASSERT_NE(schema, nullptr);
  EXPECT_NE(schema->shape_infer(), nullptr);
  EXPECT_EQ(schema->decomposition(), nullptr);
}

// ---------------------------------------------------------------------------
// 2. shape_infer 用例。
// ---------------------------------------------------------------------------

TEST(ConstantShapeInferTest, ValidAttrsProduceShapeAttributeValue) {
  const OpSchema* schema = OpRegistry::instance().find(kConstantOpName);
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->shape_infer(), nullptr);

  std::unordered_map<std::string, AttrValue> attrs;
  attrs["value"] = AttrValue{std::vector<double>{1.0, 2.0, 3.0}};
  attrs["shape"] = AttrValue{Shape({3})};
  attrs["dtype"] = AttrValue{DType::of<float>()};

  NodeContext ctx;
  ctx.op = kConstantOpName;
  ctx.attrs = &attrs;

  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  ASSERT_EQ(result.value().size(), 1u);
  EXPECT_EQ(result.value()[0], Shape({3}));
}

TEST(ConstantShapeInferTest, MissingValueAttributeIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find(kConstantOpName);
  ASSERT_NE(schema, nullptr);

  std::unordered_map<std::string, AttrValue> attrs;
  attrs["shape"] = AttrValue{Shape({2})};
  attrs["dtype"] = AttrValue{DType::of<float>()};

  NodeContext ctx;
  ctx.op = kConstantOpName;
  ctx.attrs = &attrs;

  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("missing required attribute 'value'"),
            std::string_view::npos);
}

TEST(ConstantShapeInferTest, MissingShapeAttributeIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find(kConstantOpName);
  ASSERT_NE(schema, nullptr);

  std::unordered_map<std::string, AttrValue> attrs;
  attrs["value"] = AttrValue{std::vector<double>{1.0, 2.0}};
  attrs["dtype"] = AttrValue{DType::of<float>()};

  NodeContext ctx;
  ctx.op = kConstantOpName;
  ctx.attrs = &attrs;

  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("missing required attribute 'shape'"),
            std::string_view::npos);
}

TEST(ConstantShapeInferTest, MissingDtypeAttributeIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find(kConstantOpName);
  ASSERT_NE(schema, nullptr);

  std::unordered_map<std::string, AttrValue> attrs;
  attrs["value"] = AttrValue{std::vector<double>{1.0, 2.0}};
  attrs["shape"] = AttrValue{Shape({2})};

  NodeContext ctx;
  ctx.op = kConstantOpName;
  ctx.attrs = &attrs;

  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("missing required attribute 'dtype'"),
            std::string_view::npos);
}

TEST(ConstantShapeInferTest, ValueElementCountMismatchWithShapeIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find(kConstantOpName);
  ASSERT_NE(schema, nullptr);

  std::unordered_map<std::string, AttrValue> attrs;
  attrs["value"] = AttrValue{std::vector<double>{1.0, 2.0}};  // 2 个元素
  attrs["shape"] = AttrValue{Shape({3})};                     // 期望 3 个
  attrs["dtype"] = AttrValue{DType::of<float>()};

  NodeContext ctx;
  ctx.op = kConstantOpName;
  ctx.attrs = &attrs;

  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("2 element(s)"), std::string_view::npos);
  EXPECT_NE(result.status().message().find("expects 3"), std::string_view::npos);
}

// int32/int64 已于 M22(批4 T3,决议点A)扩入白名单,不再是拒绝用例;int8
// 仍在白名单外。
TEST(ConstantShapeInferTest, DtypeOutsideWhitelistIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find(kConstantOpName);
  ASSERT_NE(schema, nullptr);

  std::unordered_map<std::string, AttrValue> attrs;
  attrs["value"] = AttrValue{std::vector<double>{1.0}};
  attrs["shape"] = AttrValue{Shape({1})};
  attrs["dtype"] = AttrValue{DType::of<std::int8_t>()};

  NodeContext ctx;
  ctx.op = kConstantOpName;
  ctx.attrs = &attrs;

  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("does not support dtype 'int8'"),
            std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 2b. int32/int64 扩容(M22,批4 T3,决议点A):合法路径 + 三负例(非整值/
//     越域/超 2^53)。
// ---------------------------------------------------------------------------

TEST(ConstantShapeInferTest, Int32ValidIntegerValuesAreAccepted) {
  const OpSchema* schema = OpRegistry::instance().find(kConstantOpName);
  ASSERT_NE(schema, nullptr);

  std::unordered_map<std::string, AttrValue> attrs;
  attrs["value"] = AttrValue{std::vector<double>{1.0, -2.0, 2147483647.0}};
  attrs["shape"] = AttrValue{Shape({3})};
  attrs["dtype"] = AttrValue{DType::of<std::int32_t>()};

  NodeContext ctx;
  ctx.op = kConstantOpName;
  ctx.attrs = &attrs;

  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({3}));
}

TEST(ConstantShapeInferTest, Int64ValidIntegerValuesAreAccepted) {
  const OpSchema* schema = OpRegistry::instance().find(kConstantOpName);
  ASSERT_NE(schema, nullptr);

  std::unordered_map<std::string, AttrValue> attrs;
  attrs["value"] = AttrValue{std::vector<double>{0.0, -9007199254740992.0, 9007199254740992.0}};
  attrs["shape"] = AttrValue{Shape({3})};
  attrs["dtype"] = AttrValue{DType::of<std::int64_t>()};

  NodeContext ctx;
  ctx.op = kConstantOpName;
  ctx.attrs = &attrs;

  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value()[0], Shape({3}));
}

TEST(ConstantShapeInferTest, Int32NonIntegralValueIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find(kConstantOpName);
  ASSERT_NE(schema, nullptr);

  std::unordered_map<std::string, AttrValue> attrs;
  attrs["value"] = AttrValue{std::vector<double>{1.5}};
  attrs["shape"] = AttrValue{Shape({1})};
  attrs["dtype"] = AttrValue{DType::of<std::int32_t>()};

  NodeContext ctx;
  ctx.op = kConstantOpName;
  ctx.attrs = &attrs;

  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("is not an integral value for dtype 'int32'"),
            std::string_view::npos);
}

TEST(ConstantShapeInferTest, Int32OutOfRangeValueIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find(kConstantOpName);
  ASSERT_NE(schema, nullptr);

  std::unordered_map<std::string, AttrValue> attrs;
  attrs["value"] = AttrValue{std::vector<double>{2147483648.0}};  // int32 max + 1
  attrs["shape"] = AttrValue{Shape({1})};
  attrs["dtype"] = AttrValue{DType::of<std::int32_t>()};

  NodeContext ctx;
  ctx.op = kConstantOpName;
  ctx.attrs = &attrs;

  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("is out of range for dtype 'int32'"),
            std::string_view::npos);
}

TEST(ConstantShapeInferTest, Int64ValueExceedingDoubleExactIntegerBoundIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find(kConstantOpName);
  ASSERT_NE(schema, nullptr);

  // 2^53+2=9007199254740994.0(2^53+1 在 double 里不可精确表示,会被舍入到
  // 2^53,反而测不出越界;取 +2 是超出该界且仍可被 double 精确表示的最小偶数)。
  std::unordered_map<std::string, AttrValue> attrs;
  attrs["value"] = AttrValue{std::vector<double>{9007199254740994.0}};
  attrs["shape"] = AttrValue{Shape({1})};
  attrs["dtype"] = AttrValue{DType::of<std::int64_t>()};

  NodeContext ctx;
  ctx.op = kConstantOpName;
  ctx.attrs = &attrs;

  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("exceeds the double-exact integer bound 2^53"),
            std::string_view::npos);
}

TEST(ConstantShapeInferTest, DynamicDimensionIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find(kConstantOpName);
  ASSERT_NE(schema, nullptr);

  std::unordered_map<std::string, AttrValue> attrs;
  attrs["value"] = AttrValue{std::vector<double>{1.0}};
  attrs["shape"] = AttrValue{Shape({kDynamicDim})};
  attrs["dtype"] = AttrValue{DType::of<float>()};

  NodeContext ctx;
  ctx.op = kConstantOpName;
  ctx.attrs = &attrs;

  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("has a dynamic dimension"), std::string_view::npos);
}

TEST(ConstantShapeInferTest, NonZeroInputCountIsRejected) {
  const OpSchema* schema = OpRegistry::instance().find(kConstantOpName);
  ASSERT_NE(schema, nullptr);

  NodeContext ctx;
  ctx.op = kConstantOpName;
  ctx.input_types = {MakeType(DType::of<float>(), {3})};  // constant 须恰 0 输入

  const Result<std::vector<Shape>> result = schema->shape_infer()(ctx);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("expects 0 inputs, got 1"), std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 3. 物化数值(ops::fill_tensor_from_constant_attrs / encode_tensor_to_attrs)。
// ---------------------------------------------------------------------------

class ConstantOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};

TEST_F(ConstantOpEagerTest, Float32MaterializationIsExact) {
  std::unordered_map<std::string, AttrValue> attrs;
  attrs["value"] = AttrValue{std::vector<double>{1.5, -2.25, 0.0, 4.75}};
  attrs["shape"] = AttrValue{Shape({4})};
  attrs["dtype"] = AttrValue{DType::of<float>()};

  Tensor out = MakeTensor1D<float>({0.0f, 0.0f, 0.0f, 0.0f});
  const Status status = frame::ops::fill_tensor_from_constant_attrs(attrs, out);
  ASSERT_TRUE(status.is_ok()) << status.message();

  Tensor expected = MakeTensor1D<float>({1.5f, -2.25f, 0.0f, 4.75f});
  EXPECT_TRUE(tensor_all_close(out, expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(ConstantOpEagerTest, Float16RoundTripThroughEncodeIsBitExact) {
  // 位级已知值(与 tests/cpp/ops/test_op_add.cpp 复用同一组已交叉验证过的位
  // 模式:0x3E00=1.5,0xBC00=-1.0):双精度中转(fp16 -> double -> fp16)论证
  // 机械化——逐元素比特相等,而非数值接近,因此不经 tensor_all_close/容差
  // 工具,直接比较 .bits。
  Tensor in = MakeTensor1D<float16_t>({float16_t{0x3E00u}, float16_t{0xBC00u}, float16_t{0x0000u}});

  std::unordered_map<std::string, AttrValue> attrs;
  const Status encode_status = frame::ops::encode_tensor_to_attrs(in, attrs);
  ASSERT_TRUE(encode_status.is_ok()) << encode_status.message();

  Tensor out =
      MakeTensor1D<float16_t>({float16_t{0x0000u}, float16_t{0x0000u}, float16_t{0x0000u}});
  const Status fill_status = frame::ops::fill_tensor_from_constant_attrs(attrs, out);
  ASSERT_TRUE(fill_status.is_ok()) << fill_status.message();

  ASSERT_EQ(out.numel(), in.numel());
  for (int64_t i = 0; i < in.numel(); ++i) {
    EXPECT_EQ(out.data<float16_t>()[i].bits, in.data<float16_t>()[i].bits) << "index " << i;
  }
}

TEST_F(ConstantOpEagerTest, BFloat16RoundTripThroughEncodeIsBitExact) {
  // 位级已知值(与 test_op_add.cpp 同一组已交叉验证过的位模式:0x3FC0=1.5,
  // 0xBF80=-1.0)。
  Tensor in =
      MakeTensor1D<bfloat16_t>({bfloat16_t{0x3FC0u}, bfloat16_t{0xBF80u}, bfloat16_t{0x0000u}});

  std::unordered_map<std::string, AttrValue> attrs;
  const Status encode_status = frame::ops::encode_tensor_to_attrs(in, attrs);
  ASSERT_TRUE(encode_status.is_ok()) << encode_status.message();

  Tensor out =
      MakeTensor1D<bfloat16_t>({bfloat16_t{0x0000u}, bfloat16_t{0x0000u}, bfloat16_t{0x0000u}});
  const Status fill_status = frame::ops::fill_tensor_from_constant_attrs(attrs, out);
  ASSERT_TRUE(fill_status.is_ok()) << fill_status.message();

  ASSERT_EQ(out.numel(), in.numel());
  for (int64_t i = 0; i < in.numel(); ++i) {
    EXPECT_EQ(out.data<bfloat16_t>()[i].bits, in.data<bfloat16_t>()[i].bits) << "index " << i;
  }
}

TEST_F(ConstantOpEagerTest, Int32MaterializationIsExact) {
  std::unordered_map<std::string, AttrValue> attrs;
  attrs["value"] = AttrValue{std::vector<double>{1.0, -2.0, 2147483647.0}};
  attrs["shape"] = AttrValue{Shape({3})};
  attrs["dtype"] = AttrValue{DType::of<std::int32_t>()};

  Tensor out = MakeTensor1D<std::int32_t>({0, 0, 0});
  const Status status = frame::ops::fill_tensor_from_constant_attrs(attrs, out);
  ASSERT_TRUE(status.is_ok()) << status.message();

  Tensor expected = MakeTensor1D<std::int32_t>({1, -2, 2147483647});
  EXPECT_TRUE(tensor_all_close(out, expected, default_tolerance(DTypeCode::kInt32)));
}

TEST_F(ConstantOpEagerTest, Int64MaterializationIsExact) {
  std::unordered_map<std::string, AttrValue> attrs;
  attrs["value"] = AttrValue{std::vector<double>{1.0, -2.0, 9007199254740992.0}};
  attrs["shape"] = AttrValue{Shape({3})};
  attrs["dtype"] = AttrValue{DType::of<std::int64_t>()};

  Tensor out = MakeTensor1D<std::int64_t>({0, 0, 0});
  const Status status = frame::ops::fill_tensor_from_constant_attrs(attrs, out);
  ASSERT_TRUE(status.is_ok()) << status.message();

  Tensor expected = MakeTensor1D<std::int64_t>({1, -2, 9007199254740992LL});
  EXPECT_TRUE(tensor_all_close(out, expected, default_tolerance(DTypeCode::kInt64)));
}

TEST_F(ConstantOpEagerTest, Int64EncodeRoundTripPreservesValue) {
  Tensor in = MakeTensor1D<std::int64_t>({1, -2, 9007199254740992LL});

  std::unordered_map<std::string, AttrValue> attrs;
  const Status encode_status = frame::ops::encode_tensor_to_attrs(in, attrs);
  ASSERT_TRUE(encode_status.is_ok()) << encode_status.message();

  Tensor out = MakeTensor1D<std::int64_t>({0, 0, 0});
  const Status fill_status = frame::ops::fill_tensor_from_constant_attrs(attrs, out);
  ASSERT_TRUE(fill_status.is_ok()) << fill_status.message();
  EXPECT_TRUE(tensor_all_close(out, in, default_tolerance(DTypeCode::kInt64)));
}

// encode_tensor_to_attrs 的 2^53 界 fail-loud 校验(设计门建议项:防未来整数
// 折叠静默失精)。9007199254740994=2^53+2,超出界且是 int64 原生精确值
// (不同于 shape_infer 侧从 double 出发的测试,这里源头就是精确 int64,编码
// 到 double 时才会失精)。
TEST_F(ConstantOpEagerTest, Int64EncodeRejectsValueExceedingDoubleExactIntegerBound) {
  Tensor in = MakeTensor1D<std::int64_t>({9007199254740994LL});

  std::unordered_map<std::string, AttrValue> attrs;
  const Status encode_status = frame::ops::encode_tensor_to_attrs(in, attrs);
  ASSERT_FALSE(encode_status.is_ok());
  EXPECT_EQ(encode_status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(encode_status.message().find("exceeds the double-exact integer bound 2^53"),
            std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 4. 编译路径执行 + shape_inference pass 0 输入豁免回归。
// ---------------------------------------------------------------------------

TEST_F(ConstantOpEagerTest, CompiledExecutionMatchesAttrsData) {
  Graph graph("constant_only");
  frame::ir::Node* const_node =
      graph.create_node(std::string(kConstantOpName), {}, {MakeType(DType::of<float>(), {3})})
          .value();
  const_node->set_attr("value", AttrValue{std::vector<double>{1.5, -2.25, 3.0}});
  const_node->set_attr("shape", AttrValue{Shape({3})});
  const_node->set_attr("dtype", AttrValue{DType::of<float>()});
  ASSERT_TRUE(graph.mark_output(const_node->output(0)).is_ok());

  const Result<std::shared_ptr<Executable>> executable_result =
      frame::runtime::compile(graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable_result.is_ok()) << executable_result.status().message();

  std::vector<Tensor> inputs;  // constant 图恰 0 个图输入
  std::vector<Tensor> outputs{MakeTensor1D<float>({0.0f, 0.0f, 0.0f})};

  const Result<std::unique_ptr<frame::hal::Stream>> stream_result =
      backend_->create_stream(device_);
  ASSERT_TRUE(stream_result.is_ok());

  const Status run_status = executable_result.value()->run(inputs, outputs, *stream_result.value());
  ASSERT_TRUE(run_status.is_ok()) << run_status.message();

  Tensor expected = MakeTensor1D<float>({1.5f, -2.25f, 3.0f});
  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

// design-reviewer 必须修复项 4 的回归测试:含 constant 节点(0 输入)的图必须
// 能独立通过 shape_inference pass ——③dtype 复核这一步此前对 0 输入节点会
// 越界访问 ctx.input_types[0],现已加 !inputs().empty() 守卫并由 constant
// 实际触达该分支(见 src/compiler/passes/shape_inference.cpp:85-88)。
TEST(ConstantShapeInferencePassRegressionTest, GraphWithConstantNodePassesShapeInference) {
  Graph graph("constant_shape_inference_regression");
  frame::ir::Node* const_node =
      graph.create_node(std::string(kConstantOpName), {}, {MakeType(DType::of<float>(), {2})})
          .value();
  const_node->set_attr("value", AttrValue{std::vector<double>{1.0, 2.0}});
  const_node->set_attr("shape", AttrValue{Shape({2})});
  const_node->set_attr("dtype", AttrValue{DType::of<float>()});
  ASSERT_TRUE(graph.mark_output(const_node->output(0)).is_ok());

  const Result<std::unique_ptr<Pass>> pass = PassRegistry::instance().create("shape_inference");
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();
  const Status status = pass.value()->run(graph);
  EXPECT_TRUE(status.is_ok()) << status.message();
}

}  // namespace
