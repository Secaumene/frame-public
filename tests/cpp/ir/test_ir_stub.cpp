// ir 子系统"转正"测试:含 kDynamicDim(动态维哨兵)的图被 verify_structure()
// 拒绝(V5:无 unknown 维,ARCH-013/ARCH-044),错误消息含 "V5" 前缀
// (LANG-005,见 src/ir/graph.cpp 的 "V<N>: " 前缀约定)。其余 ir 层用例见
// tests/cpp/ir/test_graph_construction.cpp 等同目录文件。
#include <gtest/gtest.h>
#include <string_view>

#include <frame/core/shape.h>
#include <frame/ir/graph.h>

TEST(IrStub, GraphVerifyRejectsUnknownDims) {
  frame::ir::Graph graph;
  frame::ir::TensorType type;
  type.shape = frame::Shape({2, frame::kDynamicDim});  // 第二维为动态维哨兵

  const frame::Result<frame::ir::Value*> input = graph.add_graph_input(type);
  ASSERT_TRUE(input.is_ok());

  const frame::Status status = graph.verify_structure();
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), frame::ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("V5"), std::string_view::npos);
}
