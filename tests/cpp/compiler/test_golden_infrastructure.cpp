// golden 测试基架(tests/cpp/compiler/golden_test_helpers.h)自测:
// load_ir_text_file 正常解析 / 缺失文件报错;run_pass_matches_golden 直通
// 样例通过、期望文件故意不匹配时失败报告含双方文本("actual"/"expected" 两段
// 全文均出现在失败消息内)。testdata 见
// tests/cpp/compiler/testdata/add_passthrough_{input,expected,
// mismatched_expected}.txt(2 个 graph_input + 1 个 add 节点 + graph_output 的
// 最小规范形态,float32 静态 shape、单 device cpu:0)。
#include <gtest/gtest.h>
#include <string>
#include <string_view>

#include <frame/ir/graph.h>

#include "golden_test_helpers.h"

namespace {

using frame::compiler::testing::load_ir_text_file;
using frame::compiler::testing::run_pass_matches_golden;

constexpr std::string_view kInputPath = "tests/cpp/compiler/testdata/add_passthrough_input.txt";
constexpr std::string_view kExpectedPath =
    "tests/cpp/compiler/testdata/add_passthrough_expected.txt";
constexpr std::string_view kMismatchedExpectedPath =
    "tests/cpp/compiler/testdata/add_passthrough_mismatched_expected.txt";

TEST(GoldenTestHelpersTest, LoadIrTextFileParsesValidFile) {
  const frame::Result<frame::ir::Graph> result = load_ir_text_file(kInputPath);
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(result.value().inputs().size(), 2u);
  EXPECT_EQ(result.value().outputs().size(), 1u);
}

TEST(GoldenTestHelpersTest, LoadIrTextFileReturnsErrorForMissingFile) {
  const frame::Result<frame::ir::Graph> result =
      load_ir_text_file("tests/cpp/compiler/testdata/this_file_does_not_exist.txt");
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), frame::ErrorCode::kNotFound);
  EXPECT_NE(result.status().message().find("this_file_does_not_exist.txt"), std::string_view::npos);
}

// canonicalize 是 M6 直通桩(run_impl 恒 Status::ok(),见
// src/compiler/passes/canonicalize.cpp),故本用例断言恒等直通:输入=期望。
TEST(GoldenTestHelpersTest, RunPassMatchesGoldenPassesForIdentityPassthrough) {
  EXPECT_TRUE(run_pass_matches_golden("canonicalize", kInputPath, kExpectedPath));
}

TEST(GoldenTestHelpersTest, RunPassMatchesGoldenReportsBothTextsOnMismatch) {
  const ::testing::AssertionResult result =
      run_pass_matches_golden("canonicalize", kInputPath, kMismatchedExpectedPath);
  ASSERT_FALSE(result);
  const std::string message = result.message();
  // actual(真实 dump_text,来自 add 节点)与 expected(mismatched 文件,来自
  // mul 节点)全文均须出现在失败报告内,便于人工比对。
  EXPECT_NE(message.find("add(%0, %1)"), std::string::npos);
  EXPECT_NE(message.find("mul(%0, %1)"), std::string::npos);
}

}  // namespace
