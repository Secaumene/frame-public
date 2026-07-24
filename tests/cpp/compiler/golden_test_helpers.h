#pragma once
// compiler pass golden 测试基架(namespace frame::compiler::testing):从
// tests/cpp/compiler/testdata/ 下的文本文件读入图、跑指定 pass、与期望文本
// 逐字节比对。golden 文件必须是 ir::dump_text 产出的规范形态(先跑一次冒烟
// 测试拿到实际输出再存档,不要手写猜测格式;格式权威见
// include/frame/ir/serialization.h 头注释)。
//
// testdata 路径解析:ctest 实际工作目录随 gtest_discover_tests 默认值而定
// (通常是测试可执行文件所在的构建目录),不保证等于仓库根,故本文件全部
// 路径解析一律以编译期常量 FRAME_REPO_ROOT_DIR(由
// tests/CMakeLists.txt 经 target_compile_definitions 注入,值为
// ${PROJECT_SOURCE_DIR})为基准,不依赖运行期 cwd;调用方一律传仓库根相对
// 路径(如 "tests/cpp/compiler/testdata/xxx.txt")。

#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include <frame/compiler/pass.h>
#include <frame/core/status.h>
#include <frame/ir/graph.h>
#include <frame/ir/serialization.h>
#include <frame/ops/op_registry.h>

#ifndef FRAME_REPO_ROOT_DIR
#error "FRAME_REPO_ROOT_DIR must be defined by the build system (see tests/CMakeLists.txt)"
#endif

namespace frame::compiler::testing {

// 仓库根目录(编译期常量,来源见本文件头注释)。
inline constexpr std::string_view kRepoRootDir = FRAME_REPO_ROOT_DIR;

// 仓库根相对路径 → 绝对路径。
inline std::string resolve_repo_path(std::string_view repo_relative_path) {
  return std::string(kRepoRootDir) + "/" + std::string(repo_relative_path);
}

// 读文件全部内容为字符串;失败(文件不存在/不可读)返回英文错误(含解析后的
// 绝对路径,便于定位)。
inline frame::Result<std::string> read_file_contents(std::string_view repo_relative_path) {
  const std::string full_path = resolve_repo_path(repo_relative_path);
  std::ifstream in(full_path, std::ios::binary);
  if (!in.is_open()) {
    return frame::Status::make(frame::ErrorCode::kNotFound,
                               "golden_test_helpers: cannot open file '" + full_path + "'");
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

// 读文件 + ir::parse_text 反序列化为 Graph;仅接受 dump_text 产出的规范形态
// (见 include/frame/ir/serialization.h 头注释)。
inline frame::Result<frame::ir::Graph> load_ir_text_file(std::string_view repo_relative_path) {
  const frame::Result<std::string> contents = read_file_contents(repo_relative_path);
  if (!contents.is_ok()) return contents.status();
  return frame::ir::parse_text(contents.value());
}

// golden 断言:parse(input_path) → PassRegistry::create(pass_name) → pass->run →
// graph.verify(make_op_query()) → dump_text(graph) 与 expected_path 文件内容
// 逐字节比对。expected_path 按原始文本比对(不经 parse_text 再序列化 ——
// golden 文件本身就必须已是规范形态,不规范时本函数应能捕获出该不一致)。
// 任一环节失败均返回 AssertionFailure 并携带诊断信息;比对失败时同时输出
// actual/expected 双方全文,便于人工比对。
inline ::testing::AssertionResult run_pass_matches_golden(std::string_view pass_name,
                                                          std::string_view input_path,
                                                          std::string_view expected_path) {
  frame::Result<frame::ir::Graph> input_result = load_ir_text_file(input_path);
  if (!input_result.is_ok()) {
    return ::testing::AssertionFailure() << "run_pass_matches_golden: failed to load input '"
                                         << input_path << "': " << input_result.status().message();
  }
  // Graph 持有 vector<unique_ptr<Node>>,move-only,禁止拷贝(见
  // include/frame/ir/graph.h),故此处必须 std::move 而非直接拷贝构造。
  frame::ir::Graph graph = std::move(input_result.value());

  const frame::Result<std::unique_ptr<frame::compiler::Pass>> pass_result =
      frame::compiler::PassRegistry::instance().create(pass_name);
  if (!pass_result.is_ok()) {
    return ::testing::AssertionFailure()
           << "run_pass_matches_golden: PassRegistry::create('" << pass_name
           << "') failed: " << pass_result.status().message();
  }
  const std::unique_ptr<frame::compiler::Pass>& pass = pass_result.value();

  const frame::Status run_status = pass->run(graph);
  if (!run_status.is_ok()) {
    return ::testing::AssertionFailure() << "run_pass_matches_golden: pass '" << pass_name
                                         << "'.run() failed: " << run_status.message();
  }

  const frame::ir::OpQuery query = frame::ops::make_op_query();
  const frame::Status verify_status = graph.verify(query);
  if (!verify_status.is_ok()) {
    return ::testing::AssertionFailure()
           << "run_pass_matches_golden: post-run graph.verify() failed: "
           << verify_status.message();
  }

  const std::string actual_text = frame::ir::dump_text(graph);

  const frame::Result<std::string> expected_result = read_file_contents(expected_path);
  if (!expected_result.is_ok()) {
    return ::testing::AssertionFailure()
           << "run_pass_matches_golden: failed to load expected '" << expected_path
           << "': " << expected_result.status().message();
  }
  const std::string& expected_text = expected_result.value();

  if (actual_text != expected_text) {
    return ::testing::AssertionFailure()
           << "run_pass_matches_golden: dump_text mismatch after pass '" << pass_name << "'\n"
           << "--- actual (from " << input_path << ") ---\n"
           << actual_text << "--- expected (from " << expected_path << ") ---\n"
           << expected_text;
  }
  return ::testing::AssertionSuccess();
}

}  // namespace frame::compiler::testing
