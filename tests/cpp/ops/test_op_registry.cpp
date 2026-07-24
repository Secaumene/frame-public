// OpRegistry 单测:FRAME_REGISTER_OP 宏在测试 TU 静态注册后 instance().find 可查;
// find 未注册名返回 nullptr;register_op 三类 fatal 违例中的①非法名/③保留名两类
// 死亡测试(②重名见 tests/cpp/ops/test_ops_stub.cpp:
// OpsStub.OpRegistryRejectsDuplicateName)。OpRegistry 是 Meyer's singleton,
// ops 层注册的 op 名跨全体 tests/cpp/ops/ 测试文件保持进程级唯一,本文件以
// "test_registry_" 前缀区分。
#include <gtest/gtest.h>

#include <frame/ops/op_registry.h>
#include <frame/ops/op_schema.h>

FRAME_REGISTER_OP("test_registry_macro_op")
    .input("x", "input tensor")
    .output("y", "output tensor")
    .trait(frame::ops::OpTrait::kElementwise);

namespace {

using frame::ops::OpRegistry;
using frame::ops::OpSchema;
using frame::ops::OpTrait;

TEST(OpRegistryTest, FrameRegisterOpMacroRegistersSchemaFindableAtRuntime) {
  const OpSchema* schema = OpRegistry::instance().find("test_registry_macro_op");
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->name(), "test_registry_macro_op");
  ASSERT_EQ(schema->inputs().size(), 1u);
  EXPECT_EQ(schema->inputs()[0].name, "x");
  ASSERT_EQ(schema->outputs().size(), 1u);
  EXPECT_EQ(schema->outputs()[0].name, "y");
  EXPECT_TRUE(schema->has_trait(OpTrait::kElementwise));
}

TEST(OpRegistryTest, FindReturnsNullptrForUnregisteredName) {
  EXPECT_EQ(OpRegistry::instance().find("test_registry_never_registered"), nullptr);
}

TEST(OpRegistryTest, RegisterOpAbortsOnInvalidName) {
  // ①非法名:不匹配 ^[a-z][a-z0-9_]*$(src/ops/op_registry.cpp
  // matches_op_name_charset),诊断串 "invalid op name" 与②重名/③保留名可区分。
  EXPECT_DEATH({ OpRegistry::instance().register_op("Bad-Name!"); }, "invalid op name");
}

TEST(OpRegistryTest, RegisterOpAbortsOnReservedGraphInputName) {
  // ③保留名之一:frame::ir::kGraphInputOp("graph_input")不得经 OpRegistry 注册
  // (只能经 ir::Graph::add_graph_input 创建)。
  EXPECT_DEATH({ OpRegistry::instance().register_op("graph_input"); }, "reserved op name");
}

TEST(OpRegistryTest, RegisterOpAbortsOnReservedGraphOutputName) {
  // ③保留名之二:frame::ir::kGraphOutputMarker("graph_output")是序列化层图输出
  // 标记的保留字,不得经 OpRegistry 注册。
  EXPECT_DEATH({ OpRegistry::instance().register_op("graph_output"); }, "reserved op name");
}

}  // namespace
