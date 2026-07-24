// PassRegistry 单测(include/frame/compiler/pass.h、src/compiler/pass_registry.cpp):
// 九个 ARCH-053 标准 pass 名各自 create 成功且 name() 回读一致;未注册名
// create 返回 kNotFound 且消息含名;FRAME_REGISTER_PASS 宏静态注册生效;重名
// 注册 fatal 死亡测试(匹配 "duplicate pass")。本文件注册的测试专用 pass 名
// 以 "test_pass_registry_" 前缀,跨全体 tests/cpp/compiler/ 测试文件保持
// 进程级唯一(PassRegistry 是进程级 Meyer's singleton,同
// tests/cpp/ops/test_kernel_registry.cpp 头注释纪律)。
#include <array>
#include <gtest/gtest.h>
#include <memory>
#include <string_view>

#include <frame/compiler/pass.h>
#include <frame/ir/graph.h>

namespace {

// 死亡测试与宏注册测试共用的极简 pass 实现:run 恒返回 Ok,不触碰图。
class DummyPass final : public frame::compiler::PassBase<DummyPass> {
 public:
  static constexpr std::string_view kName = "test_pass_registry_dummy_kname_unused";

  frame::Status run_impl(frame::ir::Graph& /*graph*/) { return frame::Status::ok(); }
};

std::unique_ptr<frame::compiler::Pass> MakeDummyPass() { return std::make_unique<DummyPass>(); }

// FRAME_REGISTER_PASS 宏专用的独立 pass 类(kName 即注册键,进程级唯一)。
class MacroRegisteredPass final : public frame::compiler::PassBase<MacroRegisteredPass> {
 public:
  static constexpr std::string_view kName = "test_pass_registry_macro_target";

  frame::Status run_impl(frame::ir::Graph& /*graph*/) { return frame::Status::ok(); }
};

}  // namespace

FRAME_REGISTER_PASS(MacroRegisteredPass);

namespace {

using frame::ErrorCode;
using frame::compiler::Pass;
using frame::compiler::PassRegistry;

// ARCH-053 固定全序的九个标准 pass 名(唯一权威副本见
// include/frame/compiler/pipeline.h 头注释;本文件独立誊写一份用于逐名
// create 探测)。
constexpr std::array<std::string_view, 9> kStandardPassNames = {
    "canonicalize",          "shape_inference",
    "constant_folding",      "common_subexpression_elimination",
    "dead_node_elimination", "layout_assignment",
    "operator_fusion",       "memory_planning",
    "backend_lowering",
};

TEST(PassRegistryTest, NineStandardPassNamesAllCreateSuccessfullyWithMatchingName) {
  for (std::string_view name : kStandardPassNames) {
    SCOPED_TRACE(name);
    const frame::Result<std::unique_ptr<Pass>> result = PassRegistry::instance().create(name);
    ASSERT_TRUE(result.is_ok()) << result.status().message();
    ASSERT_NE(result.value(), nullptr);
    EXPECT_EQ(result.value()->name(), name);
  }
}

TEST(PassRegistryTest, CreateUnregisteredNameReturnsNotFoundWithNameInMessage) {
  const frame::Result<std::unique_ptr<Pass>> result =
      PassRegistry::instance().create("test_pass_registry_never_registered_xyz");
  EXPECT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kNotFound);
  EXPECT_NE(result.status().message().find("test_pass_registry_never_registered_xyz"),
            std::string_view::npos);
}

TEST(PassRegistryTest, FrameRegisterPassMacroRegistersFindableConstructiblePass) {
  const frame::Result<std::unique_ptr<Pass>> result =
      PassRegistry::instance().create("test_pass_registry_macro_target");
  ASSERT_TRUE(result.is_ok());
  ASSERT_NE(result.value(), nullptr);
  EXPECT_EQ(result.value()->name(), "test_pass_registry_macro_target");
}

TEST(PassRegistryDeathTest, DuplicateRegistrationIsFatal) {
  // EXPECT_DEATH 默认 "fast"(fork)风格:子进程是当前进程内存的完整拷贝,故
  // 先在"父进程"(fork 之前)成功注册一次,EXPECT_DEATH 语句内(fork 之后的
  // 子进程)重注册同名 pass 触发 fatal,诊断串 "duplicate pass" 与
  // src/compiler/pass_registry.cpp::register_pass 的错误消息文案一致。
  ASSERT_TRUE(PassRegistry::instance()
                  .register_pass("test_pass_registry_duplicate_target", &MakeDummyPass)
                  .is_ok());

  EXPECT_DEATH(
      {
        PassRegistry::instance().register_pass_or_die("test_pass_registry_duplicate_target",
                                                      &MakeDummyPass);
      },
      "duplicate pass");
}

}  // namespace
