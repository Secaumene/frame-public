// emit_cpp 单测(docs/architecture/frontend-dsl.md 第 5 节):产物落
// ::testing::TempDir() 子目录 -> 两文件存在 -> main.cpp/CMakeLists.txt 与
// testdata/ 下的 golden 文本逐字节比对 -> main.cpp 关键符号断言(
// build_backward_graph/build_sgd_update_graph 调用 + seed 常量)。
//
// golden 更新流程:确认 emitter 行为变更符合预期后,用测试失败时打印的实际
// 输出(下方 AssertFileMatchesGolden 在比对失败时把 actual/expected 双方全文
// 打印到测试日志)覆盖 testdata/tiny_mlp_main_expected.txt /
// testdata/tiny_mlp_cmakelists_expected.txt,并人工核对内容后再提交。

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <string_view>

#include <frame/core/status.h>
#include <frame/frontend/emitter.h>
#include <frame/frontend/model_spec.h>

#include "../compiler/golden_test_helpers.h"
#include "tiny_mlp_spec_helper.h"

namespace {

using frame::Result;
using frame::Status;
using frame::frontend::emit_cpp;
using frame::frontend::EmitOptions;
using frame::frontend::ModelSpec;
using frame::frontend::testing::make_tiny_mlp_spec;

// 读运行期生成的文件(绝对路径,非仓库根相对路径,故不复用
// golden_test_helpers.h::read_file_contents——该函数专为仓库内固定文本文件
// 设计,以 FRAME_REPO_ROOT_DIR 为基准)。
std::string ReadGeneratedFile(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    return std::string();
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

// std::filesystem 一律用 error_code 重载(CPP-020,无异常文化)。
bool PathExists(const std::filesystem::path& path) {
  std::error_code ec;
  const bool exists = std::filesystem::exists(path, ec);
  return !ec && exists;
}

::testing::AssertionResult AssertFileMatchesGolden(const std::filesystem::path& actual_path,
                                                   std::string_view expected_repo_relative_path) {
  if (!PathExists(actual_path)) {
    return ::testing::AssertionFailure()
           << "AssertFileMatchesGolden: generated file does not exist: " << actual_path;
  }
  const std::string actual_text = ReadGeneratedFile(actual_path);

  const Result<std::string> expected_result =
      frame::compiler::testing::read_file_contents(expected_repo_relative_path);
  if (!expected_result.is_ok()) {
    return ::testing::AssertionFailure()
           << "AssertFileMatchesGolden: failed to load expected '" << expected_repo_relative_path
           << "': " << expected_result.status().message();
  }
  const std::string& expected_text = expected_result.value();

  if (actual_text != expected_text) {
    return ::testing::AssertionFailure()
           << "AssertFileMatchesGolden: mismatch for " << actual_path << "\n"
           << "--- actual ---\n"
           << actual_text << "--- expected (from " << expected_repo_relative_path << ") ---\n"
           << expected_text;
  }
  return ::testing::AssertionSuccess();
}

class EmitCppTest : public ::testing::Test {
 protected:
  void SetUp() override {
    output_dir_ = std::filesystem::path(::testing::TempDir()) / "frame_test_emitter_tiny_mlp";
    std::error_code ec;
    std::filesystem::remove_all(output_dir_, ec);  // 幂等:清掉上一次运行残留。
  }

  std::filesystem::path output_dir_;
};

TEST_F(EmitCppTest, EmitsBothFilesMatchingGolden) {
  const ModelSpec spec = make_tiny_mlp_spec();
  EmitOptions options;
  options.output_dir = output_dir_.string();

  const Status status = emit_cpp(spec, options);
  ASSERT_TRUE(status.is_ok()) << status.message();

  const std::filesystem::path main_cpp_path = output_dir_ / "main.cpp";
  const std::filesystem::path cmakelists_path = output_dir_ / "CMakeLists.txt";
  ASSERT_TRUE(PathExists(main_cpp_path));
  ASSERT_TRUE(PathExists(cmakelists_path));

  EXPECT_TRUE(AssertFileMatchesGolden(main_cpp_path,
                                      "tests/cpp/frontend/testdata/tiny_mlp_main_expected.txt"));
  EXPECT_TRUE(AssertFileMatchesGolden(
      cmakelists_path, "tests/cpp/frontend/testdata/tiny_mlp_cmakelists_expected.txt"));
}

// main.cpp 关键符号断言(find_package 不在此列——那属 CMakeLists golden):
// build_backward_graph/build_sgd_update_graph 调用 + seed 常量。
TEST_F(EmitCppTest, MainCppContainsExpectedKeySymbols) {
  const ModelSpec spec = make_tiny_mlp_spec();
  EmitOptions options;
  options.output_dir = output_dir_.string();

  const Status status = emit_cpp(spec, options);
  ASSERT_TRUE(status.is_ok()) << status.message();

  const std::string main_cpp_text = ReadGeneratedFile(output_dir_ / "main.cpp");
  ASSERT_FALSE(main_cpp_text.empty());

  EXPECT_NE(main_cpp_text.find("build_backward_graph("), std::string::npos);
  EXPECT_NE(main_cpp_text.find("build_sgd_update_graph("), std::string::npos);
  EXPECT_NE(main_cpp_text.find("kSeed = 20260713U"), std::string::npos);
}

// emit_cpp 内部先调用 validate(spec),失败原样透传,不写任何文件。
TEST_F(EmitCppTest, PropagatesValidateFailureAndWritesNoFiles) {
  ModelSpec spec = make_tiny_mlp_spec();
  spec.loss.prediction = "nonexistent_layer";
  EmitOptions options;
  options.output_dir = output_dir_.string();

  const Status status = emit_cpp(spec, options);
  ASSERT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), frame::ErrorCode::kInvalidArgument);
  EXPECT_FALSE(PathExists(output_dir_ / "main.cpp"));
  EXPECT_FALSE(PathExists(output_dir_ / "CMakeLists.txt"));
}

}  // namespace
