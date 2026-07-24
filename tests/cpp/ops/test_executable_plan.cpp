// ops::build_executable_plan 的 kUnimplemented 翻译分支直测(code-reviewer
// M11 建议):该分支此前仅由带卡环境的 cuda 回退用例覆盖,本文件让它在
// cpu-only 环境也有守护——构造含「有 schema 但无 cpu kernel」算子的图,
// 断言计划构建返回 ARCH-031 哨兵码且消息含算子名与后端名。

#include <gtest/gtest.h>
#include <string>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/executable_plan.h>

#include "../compiler/pass_test_common.h"

namespace {

using frame::Device;
using frame::DTypeCode;
using frame::ErrorCode;
using frame::Result;
using frame::Shape;
using frame::ir::Graph;
using frame::ir::TensorType;

TEST(ExecutablePlanTest, MissingCpuKernelYieldsUnimplementedWithOpAndBackendName) {
  frame::compiler::testing::ensure_pass_test_ops_registered();

  Graph graph("plan_missing_kernel");
  const TensorType type{frame::DType(DTypeCode::kFloat32), Shape({2}), frame::ir::Layout::kUnknown,
                        Device{"cpu", 0}};
  const Result<frame::ir::Value*> input = graph.add_graph_input(type);
  ASSERT_TRUE(input.is_ok());
  const Result<frame::ir::Node*> node = graph.create_node(
      std::string(frame::compiler::testing::kNoKernelOpName), {input.value()}, {type});
  ASSERT_TRUE(node.is_ok());
  ASSERT_TRUE(graph.mark_output(node.value(), 0).is_ok());

  const Result<frame::ops::ExecutablePlan> plan = frame::ops::build_executable_plan(graph, "cpu");
  ASSERT_FALSE(plan.is_ok());
  EXPECT_EQ(plan.status().code(), ErrorCode::kUnimplemented);
  const std::string message(plan.status().message());
  EXPECT_NE(message.find(frame::compiler::testing::kNoKernelOpName), std::string::npos);
  EXPECT_NE(message.find("cpu"), std::string::npos);
}

}  // namespace
