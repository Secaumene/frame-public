// shape_inference pass 单测(src/compiler/passes/shape_inference.cpp,ARCH-051):
// v0 为校验模式(m7-design-brief 决议点 1)——重算并比对既有输出类型,不写回。
//   1. golden 直通:既有类型与重算结果一致时图不变(复用
//      testdata/add_passthrough_{input,expected}.txt,该文本对 shape_inference
//      同样是恒等直通,无需另建一份等价 testdata,REUSE-001)。
//   2. 错误路径(不产出可比对的 dump 文本,故不经 golden 断言,改直接构图 +
//      跑 pass 断言错误):输出 shape 与重算结果不一致、输出 dtype 与输入 0
//      不一致、动态维(kDynamicDim)显式拒绝(ARCH-013/ARCH-044)、op 未注册。
// PassRegistry 是进程级 Meyer's singleton,本文件构造的未注册 op 名以
// "test_shape_inference_" 前缀跨全体测试文件保持进程级唯一(同既有纪律)。
#include <gtest/gtest.h>
#include <memory>
#include <string_view>

#include <frame/compiler/pass.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/ir/graph.h>

#include "../ir/ir_test_helpers.h"
#include "golden_test_helpers.h"

namespace {

using frame::DType;
using frame::ErrorCode;
using frame::kDynamicDim;
using frame::Result;
using frame::Status;
using frame::compiler::Pass;
using frame::compiler::PassRegistry;
using frame::compiler::testing::run_pass_matches_golden;
using frame::ir::Graph;
using frame::ir::TensorType;
using frame::ir::Value;
using frame::ir::testing::MakeFloat32Type;

constexpr std::string_view kInputPath = "tests/cpp/compiler/testdata/add_passthrough_input.txt";
constexpr std::string_view kExpectedPath =
    "tests/cpp/compiler/testdata/add_passthrough_expected.txt";

// 取一个可直接调用的 shape_inference pass 实例(经 PassRegistry::create,与
// golden_test_helpers.h::run_pass_matches_golden 内部使用同一条取得路径)。
Result<std::unique_ptr<Pass>> make_shape_inference_pass() {
  return PassRegistry::instance().create("shape_inference");
}

TEST(ShapeInferenceTest, ValidGraphIsGoldenPassthrough) {
  EXPECT_TRUE(run_pass_matches_golden("shape_inference", kInputPath, kExpectedPath));
}

TEST(ShapeInferenceTest, OutputShapeMismatchReturnsError) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  // 故意声明与重算结果([4])不一致的输出 shape([5]),校验模式下必须报错。
  ASSERT_TRUE(graph.create_node("relu", {input}, {MakeFloat32Type({5})}).is_ok());

  const Result<std::unique_ptr<Pass>> pass = make_shape_inference_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();
  const Status status = pass.value()->run(graph);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("relu"), std::string_view::npos);
  EXPECT_NE(status.message().find("shape mismatch"), std::string_view::npos);
}

TEST(ShapeInferenceTest, OutputDtypeMismatchReturnsError) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  TensorType mismatched_dtype_output = MakeFloat32Type({4});
  mismatched_dtype_output.dtype = DType::of<int32_t>();
  ASSERT_TRUE(graph.create_node("relu", {input}, {mismatched_dtype_output}).is_ok());

  const Result<std::unique_ptr<Pass>> pass = make_shape_inference_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();
  const Status status = pass.value()->run(graph);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("relu"), std::string_view::npos);
  EXPECT_NE(status.message().find("dtype"), std::string_view::npos);
}

TEST(ShapeInferenceTest, DynamicDimensionIsExplicitlyRejected) {
  // 仅一个 graph_input,其输出 shape 含 kDynamicDim:非 graph_input 节点的
  // 逐位比对循环天然不覆盖 graph_input(continue 跳过),必须靠"全 Value 无
  // unknown 维"这一显式收尾检查拦下(修订节 5-②)。
  //
  // 复核结论(test-writer 审计追加):本用例经直接构图 API + 直调 pass 断言
  // 错误,未走 run_pass_matches_golden——这是刻意选择而非遗漏。①事实核对:
  // parse_shape_dims(src/ir/serialization.cpp)对维度文本一律走
  // std::from_chars<int64_t>,并不专门拒绝 "-1",故 parse_text 本身并不会
  // 拒绝动态维文本(与本文件早期一版注释的说法不同,此处予以订正);真正拦下
  // 动态维的是本 pass 自身的收尾检查与 Graph::verify() 的 V5,均在 parse
  // 之后的下游环节生效。②即便 parse_text 技术上能读入动态维文本,ARCH-051 的
  // golden 契约本质是"运行成功后 dump_text 与期望文本逐字比对"这一正向路径;
  // 本用例的 pass->run() 从未成功过(在断言处直接返回错误),没有可比对的
  // "跑完后的图"可言,套用 run_pass_matches_golden 只能传一个永远不会被读取
  // 的哑 expected_path,徒增誤导性,与本文件其余三个错误路径用例
  // (OutputShapeMismatchReturnsError/OutputDtypeMismatchReturnsError/
  // UnregisteredOpReturnsNotFoundWithOpNameInMessage)的既定写法保持一致才是
  // 更清晰的选择。
  Graph graph;
  ASSERT_TRUE(graph.add_graph_input(MakeFloat32Type({4, kDynamicDim})).is_ok());

  const Result<std::unique_ptr<Pass>> pass = make_shape_inference_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();
  const Status status = pass.value()->run(graph);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("dynamic dimension"), std::string_view::npos);
}

TEST(ShapeInferenceTest, UnregisteredOpReturnsNotFoundWithOpNameInMessage) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  ASSERT_TRUE(
      graph.create_node("test_shape_inference_never_registered_op", {input}, {MakeFloat32Type({4})})
          .is_ok());

  const Result<std::unique_ptr<Pass>> pass = make_shape_inference_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();
  const Status status = pass.value()->run(graph);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kNotFound);
  EXPECT_NE(status.message().find("test_shape_inference_never_registered_op"),
            std::string_view::npos);
}

}  // namespace
