// Graph::assign_layout 单测(M9,决议点 A,受控图变异 API③,
// include/frame/ir/graph.h 头注释)。覆盖:幂等重指派(kUnknown->具体、以及
// 同值重指派)+ 两类报错路径(指派 kUnknown / 跨图)。
//
// "具体->不同具体"报错路径(graph.cpp:assign_layout 的
// `current != Layout::kUnknown` 分支)在 v0 下不可达:Layout 封闭枚举
// (include/frame/ir/node.h)仅有 {kUnknown, kRowMajor} 两个具名取值,该分支
// 要求 current 与 layout 均为"非 kUnknown 且互不相同"的具体值,但 v0 只有
// 一个具体值(kRowMajor)可用,current==layout 时已被前一条幂等分支拦截,
// 无法在保持二者均为具名枚举值的前提下同时满足"均非 kUnknown"与"互不相同"。
// 用越界 static_cast<Layout> 人为构造第二个"具体值"亦不可行:该分支的错误
// 消息经 layout_debug_name()(graph.cpp 匿名命名空间)格式化,其 switch 对
// Layout 做穷举匹配、default 分支为 FRAME_CHECK(false)(ARCH-043 封闭枚举
// 纪律)——越界值会命中 default 触发 fatal,而非产出可断言的 Status 错误,
// 与本分支意在验证的"优雅报错"语义不符。故本文件不构造该场景,待 M11+
// 引入第二个具体 layout 后随该里程碑测试补齐(设计侧同一结论:决议点 A
// "转换须经显式转换节点,v0 单一 layout 下该场景不存在")。
#include <gtest/gtest.h>
#include <string_view>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>

#include "ir_test_helpers.h"

namespace {

using frame::cpu_device;
using frame::DType;
using frame::ErrorCode;
using frame::Shape;
using frame::Status;
using frame::ir::Graph;
using frame::ir::Layout;
using frame::ir::Node;
using frame::ir::TensorType;
using frame::ir::Value;
using frame::ir::testing::MakeFloat32Type;

TEST(GraphAssignLayoutTest, FirstAssignmentFromUnknownSucceeds) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  // MakeFloat32Type 恒置 kRowMajor(ir_test_helpers.h),本文件需要真正的
  // kUnknown 起点(TensorType::layout 默认值,node.h),故直接构造。
  TensorType unknown_type;
  unknown_type.dtype = DType::of<float>();
  unknown_type.shape = Shape({4});
  unknown_type.device = cpu_device();
  Node* node = graph.create_node("step", {input}, {unknown_type}).value();
  ASSERT_EQ(node->output(0)->type().layout, Layout::kUnknown);

  const Status status = graph.assign_layout(node->output(0), Layout::kRowMajor);
  EXPECT_TRUE(status.is_ok()) << status.message();
  EXPECT_EQ(node->output(0)->type().layout, Layout::kRowMajor);
}

TEST(GraphAssignLayoutTest, ReassigningSameLayoutIsIdempotent) {
  Graph graph;
  // MakeFloat32Type 恒置 kRowMajor,node 的输出天然已是 kRowMajor。
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* node = graph.create_node("step", {input}, {MakeFloat32Type({4})}).value();
  ASSERT_EQ(node->output(0)->type().layout, Layout::kRowMajor);

  const Status first = graph.assign_layout(node->output(0), Layout::kRowMajor);
  const Status second = graph.assign_layout(node->output(0), Layout::kRowMajor);
  EXPECT_TRUE(first.is_ok()) << first.message();
  EXPECT_TRUE(second.is_ok()) << second.message();
  EXPECT_EQ(node->output(0)->type().layout, Layout::kRowMajor);
}

TEST(GraphAssignLayoutTest, AssigningKUnknownIsRejected) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* node = graph.create_node("step", {input}, {MakeFloat32Type({4})}).value();

  const Status status = graph.assign_layout(node->output(0), Layout::kUnknown);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("must not be kUnknown"), std::string_view::npos);
  // 违例不改动既有 layout。
  EXPECT_EQ(node->output(0)->type().layout, Layout::kRowMajor);
}

TEST(GraphAssignLayoutTest, ValueFromDifferentGraphIsRejected) {
  Graph owner_graph;
  Value* input = owner_graph.add_graph_input(MakeFloat32Type({4})).value();
  Node* node = owner_graph.create_node("step", {input}, {MakeFloat32Type({4})}).value();

  Graph other_graph;
  const Status status = other_graph.assign_layout(node->output(0), Layout::kRowMajor);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("does not belong to this graph"), std::string_view::npos);
}

}  // namespace
