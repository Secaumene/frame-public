// BackendRegistry 单测:get 未注册名/available 排序契约/register_backend 重名
// (Status 路径)/register_backend_or_die 重名(fatal 死亡测试)/
// FRAME_REGISTER_BACKEND 宏静态注册生效。cpu 后端本身"可取且 name()=='cpu'"
// 的最小验证见 tests/cpp/backends/test_backends_stub.cpp:
// BackendsStub.CpuBackendRegistered。
//
// FakeBackend/MacroFakeBackend 是本文件自定义的 Backend 测试替身:Backend 属于
// HAL 白名单虚函数类型,白名单可扩张范围显式包含"各后端对上述接口的实现文件、
// 测试 fixture"(docs/architecture/backend-hal.md 第 1 章第 3 条),故测试直接
// 实现 Backend 接口不违反铁律 #1①。BackendRegistry 是进程级 Meyer's singleton,
// 本文件注册的后端名以 "test_backend_registry_" 前缀跨全体测试文件保持进程级
// 唯一(同 tests/cpp/ops/test_kernel_registry.cpp 头注释纪律)。
#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <string_view>
#include <vector>

#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/event.h>
#include <frame/hal/executable.h>
#include <frame/hal/stream.h>

namespace {

// 极简 Backend 测试替身:全部方法返回 kUnimplemented/占位值,仅用于验证
// BackendRegistry 的注册/查找/重名拒绝语义,不触碰任何具体硬件资源。
class FakeBackend final : public frame::hal::Backend {
 public:
  std::string_view name() const override { return "test_backend_registry_fake"; }
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
};

// FRAME_REGISTER_BACKEND 宏静态注册测试专用:name() 返回值与注册键
// kMacroBackendName 保持一致,便于测试断言二者相符。
constexpr std::string_view kMacroBackendName = "test_backend_registry_macro_target";

class MacroFakeBackend final : public frame::hal::Backend {
 public:
  std::string_view name() const override { return kMacroBackendName; }
  frame::Result<int32_t> device_count() const override { return int32_t{0}; }
  frame::Result<std::unique_ptr<frame::hal::Stream>> create_stream(frame::Device) override {
    return frame::Status::make(frame::ErrorCode::kUnimplemented, "MacroFakeBackend::create_stream");
  }
  frame::Result<std::unique_ptr<frame::hal::Event>> create_event(frame::Device) override {
    return frame::Status::make(frame::ErrorCode::kUnimplemented, "MacroFakeBackend::create_event");
  }
  frame::hal::Allocator* allocator(frame::Device) override { return nullptr; }
  frame::Status copy(void*, frame::Device, const void*, frame::Device, size_t,
                     frame::hal::Stream*) override {
    return frame::Status::make(frame::ErrorCode::kUnimplemented, "MacroFakeBackend::copy");
  }
  frame::Result<std::unique_ptr<frame::hal::Executable>> compile(
      const frame::ir::Graph&, const frame::hal::CompileOptions&) override {
    return FRAME_UNIMPLEMENTED();
  }
  frame::Status launch(const frame::hal::KernelInvocation&, frame::hal::Stream*) override {
    return FRAME_UNIMPLEMENTED();
  }
};

}  // namespace

// 宏调用须在(匿名)命名空间之外的作用域完成静态初始化;MacroFakeBackend 经
// 匿名命名空间的"注入外层作用域"规则在此处仍可见(与
// tests/cpp/ops/test_kernel_registry.cpp 的 FRAME_REGISTER_KERNEL 用法同型)。
FRAME_REGISTER_BACKEND(kMacroBackendName, MacroFakeBackend);

namespace {

using frame::ErrorCode;
using frame::hal::Backend;
using frame::hal::BackendRegistry;

TEST(BackendRegistryTest, GetUnregisteredNameReturnsNotFoundWithNameInMessage) {
  const frame::Result<Backend*> result =
      BackendRegistry::instance().get("test_backend_registry_never_registered");
  EXPECT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kNotFound);
  EXPECT_NE(result.status().message().find("test_backend_registry_never_registered"),
            std::string_view::npos);
}

TEST(BackendRegistryTest, AvailableContainsCpuAndIsLexicographicallySorted) {
  const std::vector<std::string_view> available = BackendRegistry::instance().available();
  EXPECT_TRUE(std::is_sorted(available.begin(), available.end()));
  EXPECT_NE(std::find(available.begin(), available.end(), frame::kCpuBackendName), available.end());
}

TEST(BackendRegistryTest, RegisterBackendRejectsDuplicateNameWithStatus) {
  BackendRegistry& registry = BackendRegistry::instance();
  const frame::Status first = registry.register_backend("test_backend_registry_status_dup",
                                                        std::make_unique<FakeBackend>());
  ASSERT_TRUE(first.is_ok());

  const frame::Status second = registry.register_backend("test_backend_registry_status_dup",
                                                         std::make_unique<FakeBackend>());
  EXPECT_FALSE(second.is_ok());
  EXPECT_EQ(second.code(), ErrorCode::kAlreadyExists);
  EXPECT_NE(second.message().find("test_backend_registry_status_dup"), std::string_view::npos);
}

// register_backend 拒绝空后端指针(kInvalidArgument),空实例不得入表。
TEST(BackendRegistryTest, RegisterBackendRejectsNullBackendPointer) {
  BackendRegistry& registry = BackendRegistry::instance();
  const frame::Status status =
      registry.register_backend("test_backend_registry_null_backend", nullptr);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_FALSE(registry.get("test_backend_registry_null_backend").is_ok());
}

TEST(BackendRegistryDeathTest, RegisterBackendOrDieAbortsOnDuplicateName) {
  // EXPECT_DEATH 默认 "fast"(fork)风格:先在父进程成功注册一次,子进程内
  // (EXPECT_DEATH 语句体)重注册同名后端触发 fatal;诊断串同时匹配
  // "FRAME_REGISTER_BACKEND fatal"(register_backend_or_die 的诊断前缀,见
  // src/runtime/backend_registry.cpp)与 "duplicate backend name"(底层
  // register_backend 的诊断内容)。
  BackendRegistry& registry = BackendRegistry::instance();
  ASSERT_TRUE(registry.register_backend_or_die("test_backend_registry_die_target",
                                               std::make_unique<FakeBackend>()));

  EXPECT_DEATH(
      {
        registry.register_backend_or_die("test_backend_registry_die_target",
                                         std::make_unique<FakeBackend>());
      },
      "FRAME_REGISTER_BACKEND fatal.*duplicate backend name");
}

TEST(BackendRegistryTest, FrameRegisterBackendMacroRegistersFindableBackend) {
  const frame::Result<Backend*> found = BackendRegistry::instance().get(kMacroBackendName);
  ASSERT_TRUE(found.is_ok());
  EXPECT_EQ(found.value()->name(), kMacroBackendName);
}

}  // namespace
