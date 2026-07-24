// runtime::compile 编译缓存判据单测(m7-design-brief 决议点 4,修订节 5-⑥):
// 同图同签名二次 compile 不触发 Backend::compile;options.opt_level 或图结构
// 变化则 miss;同图不同 backend_name(缓存键的独立分量)则 miss;fake 后端缺
// kernel 时(ARCH-031,经 backend_lowering pass 在管线内触发 kUnimplemented)
// 触发 M10 回退链,经 cpu 参考实现(ARCH-041)成功回退(src/runtime/compile.cpp
// 接线,design-reviewer REVISE 闭环修订 1)。
// BackendRegistry/KernelRegistry 均为进程级 Meyer's singleton,本文件注册的
// fake 后端名/kernel 键以 "test_compilation_cache_" 前缀跨全体测试文件保持
// 进程级唯一;各用例互不共用注册键(修订节 5-⑥)。
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/event.h>
#include <frame/hal/executable.h>
#include <frame/hal/stream.h>
#include <frame/ir/graph.h>
#include <frame/ir/serialization.h>
#include <frame/ops/kernel_registry.h>
#include <frame/runtime/compile.h>
#include <frame/runtime/fallback_stats.h>

#include "../ir/ir_test_helpers.h"

namespace {

// 空签名 Executable 测试替身:本文件的用例只关心 Backend::compile 是否被
// 触发,不实际调用 run(),故其余方法只需满足接口。
class NoOpExecutable final : public frame::hal::Executable {
 public:
  frame::Status run(std::span<const frame::Tensor>, std::span<frame::Tensor>,
                    frame::hal::Stream&) override {
    return frame::Status::ok();
  }
  std::vector<frame::hal::IoSpec> input_signature() const override { return {}; }
  std::vector<frame::hal::IoSpec> output_signature() const override { return {}; }
};

// 计数 Backend 测试替身:compile() 每次真正被调用即计数(决议点 4 缓存判据
// 用例);其余方法未使用,返回 kUnimplemented。Backend 属 HAL 白名单虚函数
// 类型,测试直接实现该接口不违反铁律 #1①(同
// tests/cpp/backends/test_backend_registry.cpp::FakeBackend 用途)。
class CountingFakeBackend final : public frame::hal::Backend {
 public:
  explicit CountingFakeBackend(std::string_view name) : name_(name) {}

  std::string_view name() const override { return name_; }
  frame::Result<int32_t> device_count() const override { return int32_t{1}; }
  frame::Result<std::unique_ptr<frame::hal::Stream>> create_stream(frame::Device) override {
    return frame::Status::make(frame::ErrorCode::kUnimplemented,
                               "CountingFakeBackend::create_stream");
  }
  frame::Result<std::unique_ptr<frame::hal::Event>> create_event(frame::Device) override {
    return frame::Status::make(frame::ErrorCode::kUnimplemented,
                               "CountingFakeBackend::create_event");
  }
  frame::hal::Allocator* allocator(frame::Device) override { return nullptr; }
  frame::Status copy(void*, frame::Device, const void*, frame::Device, size_t,
                     frame::hal::Stream*) override {
    return frame::Status::make(frame::ErrorCode::kUnimplemented, "CountingFakeBackend::copy");
  }
  frame::Result<std::unique_ptr<frame::hal::Executable>> compile(
      const frame::ir::Graph&, const frame::hal::CompileOptions&) override {
    ++compile_call_count;
    return std::unique_ptr<frame::hal::Executable>(std::make_unique<NoOpExecutable>());
  }
  frame::Status launch(const frame::hal::KernelInvocation&, frame::hal::Stream*) override {
    return FRAME_UNIMPLEMENTED();
  }

  int compile_call_count = 0;

 private:
  std::string_view name_;
};

// "relu" 在计数 fake 后端下的占位 kernel:backend_lowering pass 需要它才能
// 放行,本身不做任何计算。
frame::Status NoOpReluKernel(frame::ops::KernelContext&) { return frame::Status::ok(); }

using frame::Device;
using frame::Result;
using frame::Status;
using frame::hal::BackendRegistry;
using frame::hal::CompileOptions;
using frame::hal::Executable;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::Value;
using frame::ir::testing::MakeFloat32Type;

TEST(CompilationCacheTest, SameGraphAndOptionsDoesNotRetriggerBackendCompile) {
  constexpr std::string_view kFakeName = "test_compilation_cache_counting_backend";
  auto backend_owned = std::make_unique<CountingFakeBackend>(kFakeName);
  CountingFakeBackend* backend_ptr = backend_owned.get();
  ASSERT_TRUE(
      BackendRegistry::instance().register_backend(kFakeName, std::move(backend_owned)).is_ok());
  ASSERT_TRUE(frame::ops::KernelRegistry::instance()
                  .register_kernel("relu", kFakeName, NoOpReluKernel)
                  .is_ok());

  const Device fake_device{kFakeName, 0};
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4}, fake_device)).value();
  Node* relu_node = graph.create_node("relu", {input}, {MakeFloat32Type({4}, fake_device)}).value();
  ASSERT_TRUE(graph.mark_output(relu_node, 0).is_ok());

  const CompileOptions options;  // opt_level 默认 1

  const Result<std::shared_ptr<Executable>> first =
      frame::runtime::compile(graph, kFakeName, options);
  ASSERT_TRUE(first.is_ok()) << first.status().message();
  EXPECT_EQ(backend_ptr->compile_call_count, 1);

  // 同图同签名二次 compile:命中缓存,不重触发 Backend::compile(判据核心)。
  const Result<std::shared_ptr<Executable>> second =
      frame::runtime::compile(graph, kFakeName, options);
  ASSERT_TRUE(second.is_ok()) << second.status().message();
  EXPECT_EQ(backend_ptr->compile_call_count, 1);
  EXPECT_EQ(second.value(), first.value());  // shared_ptr 与调用方共持同一 Executable

  // options.opt_level 变化 → 缓存键不同 → miss。
  CompileOptions different_options;
  different_options.opt_level = 0;
  const Result<std::shared_ptr<Executable>> third =
      frame::runtime::compile(graph, kFakeName, different_options);
  ASSERT_TRUE(third.is_ok()) << third.status().message();
  EXPECT_EQ(backend_ptr->compile_call_count, 2);

  // 图结构变化(不同 shape)→ dump_text 不同 → 缓存键不同 → miss。
  Graph graph2;
  Value* input2 = graph2.add_graph_input(MakeFloat32Type({8}, fake_device)).value();
  Node* relu_node2 =
      graph2.create_node("relu", {input2}, {MakeFloat32Type({8}, fake_device)}).value();
  ASSERT_TRUE(graph2.mark_output(relu_node2, 0).is_ok());

  const Result<std::shared_ptr<Executable>> fourth =
      frame::runtime::compile(graph2, kFakeName, options);
  ASSERT_TRUE(fourth.is_ok()) << fourth.status().message();
  EXPECT_EQ(backend_ptr->compile_call_count, 3);
}

TEST(CompilationCacheTest, MissingKernelForFakeBackendFallsBackToCpuReferenceKernel) {
  // 与上一用例使用不同的 fake 后端名,不共用注册键(修订节 5-⑥);故意不为
  // 任何 op 注册 kernel。M10(design-reviewer REVISE 闭环修订 1 +
  // src/runtime/compile.cpp 接线):backend_lowering 逐节点 kernel 缺失现翻译
  // 为 kUnimplemented,runtime::compile 据此触发回退链
  // (docs/architecture/execution-model.md 第5章);"relu" 在 cpu 后端有参考
  // 实现(ARCH-041 保证),该场景现落 ③ cpu-reference、compile 成功——本用例
  // 断言随之更新(原断言"compile 返回 ARCH-031 错误"已被 M10 回退链的既定行为
  // 取代)。compile_call_count 保持 0:回退产物绕过 Backend::compile,不触发
  // 该计数;FallbackStats 记一次(op='relu', backend=kFakeNoKernelName)。
  constexpr std::string_view kFakeNoKernelName = "test_compilation_cache_no_kernel_backend";
  auto backend_owned = std::make_unique<CountingFakeBackend>(kFakeNoKernelName);
  CountingFakeBackend* backend_ptr = backend_owned.get();
  ASSERT_TRUE(BackendRegistry::instance()
                  .register_backend(kFakeNoKernelName, std::move(backend_owned))
                  .is_ok());

  const Device fake_device{kFakeNoKernelName, 0};
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4}, fake_device)).value();
  Node* relu_node = graph.create_node("relu", {input}, {MakeFloat32Type({4}, fake_device)}).value();
  ASSERT_TRUE(graph.mark_output(relu_node, 0).is_ok());

  const Result<std::shared_ptr<Executable>> result =
      frame::runtime::compile(graph, kFakeNoKernelName, CompileOptions{});
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(backend_ptr->compile_call_count, 0);
  EXPECT_EQ(frame::runtime::FallbackStats::instance().count("relu", kFakeNoKernelName), 1);
}

TEST(CompilationCacheTest, FingerprintDistinguishesAllowTf32) {
  // ADR-0019:allow_tf32 是缓存键「编译选项哈希」分量,且默认必须为 false
  // (fp32 参考语义不悄然改变)。仅该字段不同的两组选项必须产生不同
  // fingerprint,否则 TF32 开/关两次编译串键(同键不同数值污染缓存)。
  CompileOptions strict_options;
  EXPECT_FALSE(strict_options.allow_tf32);

  CompileOptions tf32_options;
  tf32_options.allow_tf32 = true;
  EXPECT_NE(strict_options.fingerprint(), tf32_options.fingerprint());
}

TEST(CompilationCacheTest, DifferentBackendNameForSameGraphTextIsACacheMiss) {
  // 缓存键三分量为 backend_name + options.fingerprint() + dump_text(graph)
  // (src/runtime/compile.cpp::make_cache_key)。runtime::compile 入口本身会
  // 校验 backend_name 须与图 device 后端一致(compile.h 头注释),唯一的例外
  // 是空图(graph_device_backend 对无算子节点的图返回空 string_view,跳过该
  // 校验)——借此让同一个 Graph 对象以两个不同 backend_name 分别调用 compile
  // 成为合法操作,且两次调用的 dump_text(空图恒为 "")逐字节相同,从而把
  // "backend_name 是否为缓存键的独立分量"这一判据同"图文本是否变化"这一分量
  // 彻底隔离开(否则两个变量同时变化,无法确认到底是哪个分量导致 miss)。
  constexpr std::string_view kFakeNameA = "test_compilation_cache_backend_name_a";
  constexpr std::string_view kFakeNameB = "test_compilation_cache_backend_name_b";
  auto backend_a_owned = std::make_unique<CountingFakeBackend>(kFakeNameA);
  auto backend_b_owned = std::make_unique<CountingFakeBackend>(kFakeNameB);
  CountingFakeBackend* backend_a_ptr = backend_a_owned.get();
  CountingFakeBackend* backend_b_ptr = backend_b_owned.get();
  ASSERT_TRUE(
      BackendRegistry::instance().register_backend(kFakeNameA, std::move(backend_a_owned)).is_ok());
  ASSERT_TRUE(
      BackendRegistry::instance().register_backend(kFakeNameB, std::move(backend_b_owned)).is_ok());

  const Graph empty_graph;
  ASSERT_EQ(frame::ir::dump_text(empty_graph), "");  // 确认两次调用的图文本分量恒相同

  const CompileOptions options;

  const Result<std::shared_ptr<Executable>> first =
      frame::runtime::compile(empty_graph, kFakeNameA, options);
  ASSERT_TRUE(first.is_ok()) << first.status().message();
  EXPECT_EQ(backend_a_ptr->compile_call_count, 1);
  EXPECT_EQ(backend_b_ptr->compile_call_count, 0);

  const Result<std::shared_ptr<Executable>> second =
      frame::runtime::compile(empty_graph, kFakeNameB, options);
  ASSERT_TRUE(second.is_ok()) << second.status().message();
  EXPECT_EQ(backend_a_ptr->compile_call_count, 1);  // A 未被重复触发(证明未被误命中)
  EXPECT_EQ(backend_b_ptr->compile_call_count, 1);  // B 被独立触发(证明 backend_name 不同即 miss)
  EXPECT_NE(second.value(), first.value());         // 两次拿到彼此独立的 Executable
}

}  // namespace
