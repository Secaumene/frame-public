// Graph::verify / verify_structure 单测:OpQuery 回调缺失时 fail-closed、V3
// 未注册 op 被拒(fake 白名单回调)、V4 check_schema 错误透传、V6 device 不一致
// 被拒、graph_input 节点的 V3/V4 结构豁免、verify_structure 不依赖 OpQuery 即可
// 跑 V1/V2/V5/V6/V7。全程纯主机内存,不依赖任何已注册后端(Device 只是数据字段)。
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <vector>

#include <frame/ir/graph.h>

#include "ir_test_helpers.h"

namespace {

using frame::Device;
using frame::ErrorCode;
using frame::kCpuBackendName;
using frame::kCudaBackendName;
using frame::Status;
using frame::ir::Graph;
using frame::ir::kGraphInputOp;
using frame::ir::Node;
using frame::ir::OpQuery;
using frame::ir::Value;
using frame::ir::testing::MakeFloat32Type;

TEST(GraphVerifyTest, MissingOpRegisteredCallbackFailsClosedWithV3) {
  Graph graph;
  OpQuery query;
  query.check_schema = [](const Node&) { return Status::ok(); };  // 只留 check_schema

  const Status status = graph.verify(query);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("V3"), std::string_view::npos);
  EXPECT_NE(status.message().find("op_registered"), std::string_view::npos);
}

TEST(GraphVerifyTest, MissingCheckSchemaCallbackFailsClosedWithV4) {
  Graph graph;
  OpQuery query;
  query.op_registered = [](std::string_view) { return true; };  // 只留 op_registered

  const Status status = graph.verify(query);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("V4"), std::string_view::npos);
  EXPECT_NE(status.message().find("check_schema"), std::string_view::npos);
}

TEST(GraphVerifyTest, BothCallbacksMissingFailsClosedWithV3First) {
  Graph graph;
  const OpQuery query;  // 两个 std::function 均默认构造为空(fail-closed)
  const Status status = graph.verify(query);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  // graph.cpp::verify() 先查 op_registered 再查 check_schema,故两者皆空时
  // 报告的是 V3(先触发的那一个),而非 V4。
  EXPECT_NE(status.message().find("V3"), std::string_view::npos);
}

TEST(GraphVerifyTest, UnregisteredOpIsRejectedWithV3ViaAllowlistCallback) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  graph.create_node("mystery_op", {input}, {MakeFloat32Type({4})});

  OpQuery query;
  // fake 白名单:只承认 "relu" 已注册,mystery_op 不在白名单内应被拒。
  query.op_registered = [](std::string_view op_name) { return op_name == "relu"; };
  query.check_schema = [](const Node&) { return Status::ok(); };

  const Status status = graph.verify(query);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kNotFound);
  EXPECT_NE(status.message().find("V3"), std::string_view::npos);
  EXPECT_NE(status.message().find("mystery_op"), std::string_view::npos);
}

TEST(GraphVerifyTest, CheckSchemaErrorIsPassedThroughWithV4Prefix) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  graph.create_node("bad_schema_op", {input}, {MakeFloat32Type({4})});

  OpQuery query;
  query.op_registered = [](std::string_view) { return true; };
  query.check_schema = [](const Node& node) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "node '" + std::string(node.op()) + "' violates schema");
  };

  const Status status = graph.verify(query);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("V4"), std::string_view::npos);
  EXPECT_NE(status.message().find("violates schema"), std::string_view::npos);
}

TEST(GraphVerifyTest, InconsistentDeviceAcrossValuesIsRejectedWithV6) {
  Graph graph;
  Value* cpu_input =
      graph.add_graph_input(MakeFloat32Type({4}, Device{kCpuBackendName, 0})).value();
  // 第二个节点的输出显式落在不同 device(Device 只是数据字段,无需真实注册
  // 该后端即可构造出用于验证 V6 的不一致场景)。
  graph.create_node("copy", {cpu_input}, {MakeFloat32Type({4}, Device{kCudaBackendName, 0})});

  const Status status = graph.verify_structure();
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("V6"), std::string_view::npos);
}

TEST(GraphVerifyTest, GraphInputNodeIsExemptFromOpRegistrationCheck) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* relu = graph.create_node("relu", {input}, {MakeFloat32Type({4})}).value();
  ASSERT_TRUE(graph.mark_output(relu->output(0)).is_ok());

  std::vector<std::string> queried_ops;
  OpQuery query;
  query.op_registered = [&queried_ops](std::string_view op_name) {
    queried_ops.emplace_back(op_name);
    return true;
  };
  query.check_schema = [](const Node&) { return Status::ok(); };

  const Status status = graph.verify(query);
  EXPECT_TRUE(status.is_ok());
  // graph_input 节点(kGraphInputOp)必须从未被送进 op_registered 回调
  // (V3/V4 豁免,改做结构检查:0 输入、恰 1 输出、已登记于 inputs())。
  for (const std::string& op_name : queried_ops) {
    EXPECT_NE(op_name, kGraphInputOp);
  }
  EXPECT_EQ(queried_ops.size(), 1u);  // 仅 "relu" 这一个真实节点被查询过
}

TEST(GraphVerifyTest, VerifyStructureRunsWithoutAnyOpQueryCallback) {
  // verify_structure() 签名不接受 OpQuery,天然验证"不需要回调即可跑
  // V1/V2/V5/V6/V7":构造一个含多种属性类型、多节点的合法图,全程不构造/传入
  // 任何 OpQuery,应直接通过。
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* node = graph.create_node("relu", {input}, {MakeFloat32Type({4})}).value();
  node->set_attr("axis", frame::ir::AttrValue{int64_t{0}});  // 覆盖 V7 的一种属性类型
  node->set_attr("scale", frame::ir::AttrValue{1.5});        // 覆盖另一种
  ASSERT_TRUE(graph.mark_output(node->output(0)).is_ok());

  EXPECT_TRUE(graph.verify_structure().is_ok());
}

}  // namespace
