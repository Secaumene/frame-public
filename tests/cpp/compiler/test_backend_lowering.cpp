// backend_lowering pass 单测(src/compiler/passes/backend_lowering.cpp,ARCH-051):
// v0 为支持性校验(m7-design-brief 决议点 2)——目标后端名取自图 device,经
// BackendRegistry::get 取后端存在性,再逐非 graph_input 节点查
// KernelRegistry::find(op, backend),缺失即报带算子名与后端名的英文错误
// (ARCH-031)。
//   1. golden 直通:cpu 后端("add" 已注册 cpu kernel)下图不变(复用
//      testdata/add_passthrough_{input,expected}.txt,REUSE-001)。
//   2. 空图/仅输入图跳过(决议点 2 修订):即便图 device 指向从未注册的后端名,
//      因无算子节点需要判定支持性,pass 仍直接放行。
//   3. 目标后端未注册 → BackendRegistry::get 失败,错误消息含后端名。
//   4. fake 后端已注册且逐算子均有 kernel → pass 放行。
//   5. fake 后端已注册但缺 kernel → 错误消息含算子名与后端名(ARCH-031)。
// BackendRegistry/KernelRegistry 均为进程级 Meyer's singleton,本文件注册的
// fake 后端名/kernel 键以 "test_backend_lowering_" 前缀跨全体测试文件保持
// 进程级唯一,且用例 4/5 两个 fake 后端名互不相同(不共用注册键,修订节 5-⑥
// 的纪律同样适用于此处)。
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <string_view>

#include <frame/compiler/pass.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/event.h>
#include <frame/hal/executable.h>
#include <frame/hal/stream.h>
#include <frame/ir/graph.h>
#include <frame/ops/kernel_registry.h>

#include "../ir/ir_test_helpers.h"
#include "golden_test_helpers.h"

namespace {

// 极简 Backend 测试替身:除 name() 外全部方法返回 kUnimplemented/占位值——
// backend_lowering 只调用 BackendRegistry::get(名字是否可取),不触碰
// compile/launch 等其余接口,同 tests/cpp/backends/test_backend_registry.cpp
// 的 FakeBackend 用途(Backend 属 HAL 白名单虚函数类型,测试 fixture 直接实现
// 该接口不违反铁律 #1①)。同一个类经不同注册键各注册一次,分饰"有 kernel"与
// "无 kernel"两个 fake 后端。
class FakeBackend : public frame::hal::Backend {
 public:
  explicit FakeBackend(std::string_view name) : name_(name) {}

  std::string_view name() const override { return name_; }
  frame::Result<int32_t> device_count() const override { return int32_t{0}; }
  frame::Result<std::unique_ptr<frame::hal::Stream>> create_stream(frame::Device) override {
    return frame::Status::make(frame::ErrorCode::kUnimplemented, "FakeBackend::create_stream");
  }
  frame::Result<std::unique_ptr<frame::hal::Event>> create_event(frame::Device) override {
    return frame::Status::make(frame::ErrorCode::kUnimplemented, "FakeBackend::create_event");
  }
  frame::hal::Allocator* allocator(frame::Device) override { return nullptr; }
  frame::Status copy(void*, frame::Device, const void*, frame::Device, size_t,
                     frame::hal::Stream*) override {
    return frame::Status::make(frame::ErrorCode::kUnimplemented, "FakeBackend::copy");
  }
  frame::Result<std::unique_ptr<frame::hal::Executable>> compile(
      const frame::ir::Graph&, const frame::hal::CompileOptions&) override {
    return FRAME_UNIMPLEMENTED();
  }
  frame::Status launch(const frame::hal::KernelInvocation&, frame::hal::Stream*) override {
    return FRAME_UNIMPLEMENTED();
  }

 private:
  std::string_view name_;
};

// 有 kernel 的 fake 后端名,与下方 "relu" kernel 注册配对;无 kernel 的 fake
// 后端名只注册后端本身,故意不为任何 op 注册 kernel。
constexpr std::string_view kFakeBackendWithKernel = "test_backend_lowering_fake_with_kernel";
constexpr std::string_view kFakeBackendNoKernel = "test_backend_lowering_fake_no_kernel";

// 静态注册两个 fake 后端(宏调用须在匿名命名空间之外,同
// tests/cpp/backends/test_backend_registry.cpp::MacroFakeBackend 用法)。
class FakeBackendWithKernel final : public FakeBackend {
 public:
  FakeBackendWithKernel() : FakeBackend(kFakeBackendWithKernel) {}
};
class FakeBackendNoKernel final : public FakeBackend {
 public:
  FakeBackendNoKernel() : FakeBackend(kFakeBackendNoKernel) {}
};

// "relu" 在 kFakeBackendWithKernel 下的占位 kernel:不做任何计算,仅用于让
// KernelRegistry::find(op, backend) 命中。
frame::Status FakeReluKernel(frame::ops::KernelContext&) { return frame::Status::ok(); }

}  // namespace

FRAME_REGISTER_BACKEND(kFakeBackendWithKernel, FakeBackendWithKernel);
FRAME_REGISTER_BACKEND(kFakeBackendNoKernel, FakeBackendNoKernel);
FRAME_REGISTER_KERNEL("relu", kFakeBackendWithKernel, FakeReluKernel);

namespace {

using frame::Device;
using frame::ErrorCode;
using frame::Result;
using frame::Status;
using frame::compiler::Pass;
using frame::compiler::PassRegistry;
using frame::compiler::testing::run_pass_matches_golden;
using frame::ir::Graph;
using frame::ir::Value;
using frame::ir::testing::MakeFloat32Type;

constexpr std::string_view kInputPath = "tests/cpp/compiler/testdata/add_passthrough_input.txt";
constexpr std::string_view kExpectedPath =
    "tests/cpp/compiler/testdata/add_passthrough_expected.txt";

Result<std::unique_ptr<Pass>> make_backend_lowering_pass() {
  return PassRegistry::instance().create("backend_lowering");
}

TEST(BackendLoweringTest, ValidCpuGraphIsGoldenPassthrough) {
  EXPECT_TRUE(run_pass_matches_golden("backend_lowering", kInputPath, kExpectedPath));
}

TEST(BackendLoweringTest, EmptyGraphSucceeds) {
  Graph graph;
  const Result<std::unique_ptr<Pass>> pass = make_backend_lowering_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();
  EXPECT_TRUE(pass.value()->run(graph).is_ok());
}

TEST(BackendLoweringTest, InputOnlyGraphSucceedsEvenWithUnregisteredDevice) {
  // 图 device 指向一个从未注册的后端名,但因无算子节点需要判定支持性,
  // 决议点 2 修订要求直接放行(否则 BackendRegistry::get 会失败)。
  Graph graph;
  ASSERT_TRUE(graph
                  .add_graph_input(MakeFloat32Type(
                      {4}, Device{"test_backend_lowering_never_registered_anywhere", 0}))
                  .is_ok());

  const Result<std::unique_ptr<Pass>> pass = make_backend_lowering_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();
  EXPECT_TRUE(pass.value()->run(graph).is_ok());
}

TEST(BackendLoweringTest, UnregisteredBackendReturnsNotFoundWithBackendNameInMessage) {
  Graph graph;
  const Device fake_device{"test_backend_lowering_never_registered_backend", 0};
  Value* input = graph.add_graph_input(MakeFloat32Type({4}, fake_device)).value();
  ASSERT_TRUE(graph.create_node("relu", {input}, {MakeFloat32Type({4}, fake_device)}).is_ok());

  const Result<std::unique_ptr<Pass>> pass = make_backend_lowering_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();
  const Status status = pass.value()->run(graph);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kNotFound);
  EXPECT_NE(status.message().find("test_backend_lowering_never_registered_backend"),
            std::string_view::npos);
}

TEST(BackendLoweringTest, FakeBackendWithAllKernelsRegisteredSucceeds) {
  Graph graph;
  const Device fake_device{kFakeBackendWithKernel, 0};
  Value* input = graph.add_graph_input(MakeFloat32Type({4}, fake_device)).value();
  ASSERT_TRUE(graph.create_node("relu", {input}, {MakeFloat32Type({4}, fake_device)}).is_ok());

  const Result<std::unique_ptr<Pass>> pass = make_backend_lowering_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();
  EXPECT_TRUE(pass.value()->run(graph).is_ok());
}

TEST(BackendLoweringTest, FakeBackendMissingKernelReturnsErrorWithOpAndBackendNameInMessage) {
  Graph graph;
  const Device fake_device{kFakeBackendNoKernel, 0};
  Value* input = graph.add_graph_input(MakeFloat32Type({4}, fake_device)).value();
  ASSERT_TRUE(graph.create_node("relu", {input}, {MakeFloat32Type({4}, fake_device)}).is_ok());

  const Result<std::unique_ptr<Pass>> pass = make_backend_lowering_pass();
  ASSERT_TRUE(pass.is_ok()) << pass.status().message();
  const Status status = pass.value()->run(graph);
  EXPECT_FALSE(status.is_ok());
  // 哨兵码翻译(design-reviewer REVISE 闭环修订 1,M10):逐节点 kernel 缺失
  // 统一翻译为 kUnimplemented,供 runtime::compile 据此触发回退链;
  // KernelRegistry::find 自身的 kNotFound 未变,本处断言的是翻译后的哨兵码。
  EXPECT_EQ(status.code(), ErrorCode::kUnimplemented);
  EXPECT_NE(status.message().find("relu"), std::string_view::npos);
  EXPECT_NE(status.message().find(std::string(kFakeBackendNoKernel)), std::string_view::npos);
}

}  // namespace
