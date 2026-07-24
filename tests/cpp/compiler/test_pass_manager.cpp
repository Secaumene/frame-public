// PassManager 单测(include/frame/compiler/pass_manager.h、
// src/compiler/pass_manager.cpp):
//   1. 按 add_pass 顺序执行(测试用顺序追踪 pass,把自身名字按运行顺序追加进
//      共享 log,断言 log 与预期顺序逐一相同);
//   2. add(未注册名) 不立即报错,延迟到 run() 才报错(且保留首个错误,不被
//      后续 add 覆盖);
//   3. ARCH-022 判定:pass 产出非法图时管线立即报错——测试 pass 的 run_impl
//      用 Graph::create_node 造一个字符集合法但未在 OpRegistry 注册的 op,
//      断言 run() 返回错误、消息含 "pass '<name>': " 与 "V3",且两个前缀均
//      恰好出现一次(不双前缀);
//   4. set_dump_ir_after:命中时 ostream 内容与 ir::dump_text(graph) 逐字节
//      一致,未命中时无输出;
//   5. pass_names() 只读观测面:按装配序回读(与用例 1 的顺序追踪 pass 不
//      重复——那条断言的是"run() 的执行顺序",这条断言的是"未经 run() 之前
//      纯装配阶段的可读顺序"两件事);add(未注册名) 触发的延迟错误不影响此前
//      已成功装配的 pass 在 pass_names() 中可见(header 注释明确的契约)。
// 本文件全程使用 add_pass(直接传入 unique_ptr<Pass> 实例)而非
// PassRegistry::create,故测试 pass 无需注册进全局单例,不与其他
// tests/cpp/compiler/ 文件产生进程级命名冲突;pass 名仍以 "test_pass_manager_"
// 前缀,便于错误消息定位来源。
#include <gtest/gtest.h>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <frame/compiler/pass.h>
#include <frame/compiler/pass_manager.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ir/serialization.h>

#include "../ir/ir_test_helpers.h"

namespace {

using frame::compiler::Pass;
using frame::compiler::PassManager;
using frame::ir::dump_text;
using frame::ir::Graph;
using frame::ir::TensorType;
using frame::ir::testing::MakeFloat32Type;

// 顺序追踪 pass:run() 时把自身名字追加进调用方持有的共享 log,自身不触碰图
// (恒返回 Ok)。名字与 log 均经构造函数传入(不走 PassRegistry,不受
// PassType concept 的默认可构造性约束)。
class OrderTrackingPass final : public Pass {
 public:
  OrderTrackingPass(std::string name, std::vector<std::string>& log)
      : name_(std::move(name)), log_(log) {}

  std::string_view name() const override { return name_; }

  frame::Status run(Graph& /*graph*/) override {
    log_.push_back(name_);
    return frame::Status::ok();
  }

 private:
  std::string name_;
  std::vector<std::string>& log_;
};

// ARCH-022 判定专用 pass:run() 自身返回 Ok,但往图里插入一个字符集合法、
// 未在 OpRegistry 注册的 op 节点——run() 本身"成功"不代表产出的图合法,
// PassManager 必须在 run 之后的 verify() 拦下并立即报错(不可静默放行)。
class InjectUnregisteredOpPass final : public Pass {
 public:
  explicit InjectUnregisteredOpPass(std::string name) : name_(std::move(name)) {}

  std::string_view name() const override { return name_; }

  frame::Status run(Graph& graph) override {
    const frame::Result<frame::ir::Node*> node_result =
        graph.create_node("test_pass_manager_unregistered_op", {}, {TensorType{}});
    if (!node_result.is_ok()) return node_result.status();
    return frame::Status::ok();
  }

 private:
  std::string name_;
};

TEST(PassManagerTest, RunsAddedPassesInInsertionOrder) {
  std::vector<std::string> log;
  PassManager manager;
  manager.add_pass(std::make_unique<OrderTrackingPass>("test_pass_manager_order_a", log))
      .add_pass(std::make_unique<OrderTrackingPass>("test_pass_manager_order_b", log))
      .add_pass(std::make_unique<OrderTrackingPass>("test_pass_manager_order_c", log));

  Graph graph;
  const frame::Status status = manager.run(graph);
  ASSERT_TRUE(status.is_ok()) << status.message();

  const std::vector<std::string> expected = {
      "test_pass_manager_order_a", "test_pass_manager_order_b", "test_pass_manager_order_c"};
  EXPECT_EQ(log, expected);
}

TEST(PassManagerTest, AddUnregisteredNameDefersErrorToRun) {
  PassManager manager;
  manager.add("test_pass_manager_never_registered_first");
  // 第二次 add(未注册名):保留首个错误,不被本次覆盖(头文件 add() 注释)。
  manager.add("test_pass_manager_never_registered_second");

  Graph graph;
  const frame::Status status = manager.run(graph);
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("test_pass_manager_never_registered_first"),
            std::string_view::npos);
  EXPECT_EQ(status.message().find("test_pass_manager_never_registered_second"),
            std::string_view::npos);
}

TEST(PassManagerTest, PassNamesReflectsAssembledOrderAndSurvivesDeferredAddError) {
  std::vector<std::string> log;
  PassManager manager;
  manager.add_pass(std::make_unique<OrderTrackingPass>("test_pass_manager_pass_names_a", log))
      .add_pass(std::make_unique<OrderTrackingPass>("test_pass_manager_pass_names_b", log));

  // 装配序回读(未调用 run(),纯装配阶段即可读):与 add_pass 插入顺序一致。
  const std::vector<std::string_view> expected = {"test_pass_manager_pass_names_a",
                                                  "test_pass_manager_pass_names_b"};
  EXPECT_EQ(manager.pass_names(), expected);

  // add(未注册名) 触发 pending_error_(延迟到 run() 才报错,同
  // AddUnregisteredNameDefersErrorToRun 用例),但此前已成功装配的 pass 仍应
  // 出现在 pass_names() 中(header 注释明确的契约);解析失败的这次 add()
  // 调用本身不产生新条目。
  manager.add("test_pass_manager_pass_names_never_registered");
  EXPECT_EQ(manager.pass_names(), expected);
}

TEST(PassManagerTest, PassProducingIllegalGraphMakesRunFailWithPassAndV3PrefixOnce) {
  PassManager manager;
  manager.add_pass(std::make_unique<InjectUnregisteredOpPass>("test_pass_manager_injector"));

  Graph graph;
  const frame::Status status = manager.run(graph);
  ASSERT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), frame::ErrorCode::kNotFound);  // V3: op 未注册 → kNotFound

  const std::string_view message = status.message();
  EXPECT_NE(message.find("pass 'test_pass_manager_injector': "), std::string_view::npos);
  EXPECT_NE(message.find("V3"), std::string_view::npos);

  // 不双前缀:统计 "pass '" 与 "V3: " 各自恰好出现一次。
  auto count_occurrences = [](std::string_view haystack, std::string_view needle) {
    size_t count = 0;
    size_t pos = 0;
    while ((pos = haystack.find(needle, pos)) != std::string_view::npos) {
      ++count;
      pos += needle.size();
    }
    return count;
  };
  EXPECT_EQ(count_occurrences(message, "pass '"), 1u);
  EXPECT_EQ(count_occurrences(message, "V3: "), 1u);
}

TEST(PassManagerTest, SetDumpIrAfterCapturesGraphStateMatchingDumpText) {
  std::vector<std::string> log;
  PassManager manager;
  manager.add_pass(std::make_unique<OrderTrackingPass>("test_pass_manager_dump_target", log))
      .add_pass(std::make_unique<OrderTrackingPass>("test_pass_manager_dump_after", log));

  Graph graph;
  ASSERT_TRUE(graph.add_graph_input(MakeFloat32Type({2, 3})).is_ok());

  std::ostringstream dump;
  manager.set_dump_ir_after("test_pass_manager_dump_target", dump);

  const frame::Status status = manager.run(graph);
  ASSERT_TRUE(status.is_ok()) << status.message();

  EXPECT_EQ(dump.str(), dump_text(graph));
  EXPECT_NE(dump.str(), "");  // 非空图:确认 dump 确实命中过而非巧合的空字符串
}

TEST(PassManagerTest, SetDumpIrAfterProducesNoOutputWhenNameNeverMatches) {
  std::vector<std::string> log;
  PassManager manager;
  manager.add_pass(std::make_unique<OrderTrackingPass>("test_pass_manager_dump_absent_a", log))
      .add_pass(std::make_unique<OrderTrackingPass>("test_pass_manager_dump_absent_b", log));

  Graph graph;
  ASSERT_TRUE(graph.add_graph_input(MakeFloat32Type({2, 3})).is_ok());

  std::ostringstream dump;
  manager.set_dump_ir_after("test_pass_manager_dump_name_that_never_matches", dump);

  const frame::Status status = manager.run(graph);
  ASSERT_TRUE(status.is_ok()) << status.message();

  EXPECT_EQ(dump.str(), "");
}

}  // namespace
