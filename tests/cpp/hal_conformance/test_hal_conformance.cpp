// HAL 一致性套件"首批"细粒度用例:以 BackendRegistry::available() 参数化,
// 逐项验证第 2 章接口方法的可观察契约,一行为一测试(便于精确定位失败原因)。
// 与 tests/cpp/hal_conformance/test_hal_conformance_stub.cpp 的端到端综合场景
// (HalConformanceStub.AllBackendsSatisfyHalContract)互补,存在有意的重叠覆盖。
// 覆盖范围(docs/architecture/backend-hal.md 第 2 章 + 第 5 章 checklist 第 6 项):
//   get 可取且 name 与注册键一致;device_count>=1;create_stream 成功;
//   create_event 成功且未 record 时 query()==true、synchronize()==Ok;
//   stream record(Event)→wait(Event)→synchronize 全 Ok;allocator(device0)
//   非空且 allocate/deallocate 往返(alignment 64);copy H2H 往返一致;
//   compile() 对不支持算子报错、码为 kUnimplemented 且消息含算子名(M10,
//   ARCH-031 加严判定方法——见下方 CompileRejectsUnsupportedOpWithUnimplem
//   entedErrorContainingOpName)。
// 当前 BackendRegistry::available() 仅 "cpu",本文件全程禁止对后端名做任何
// 特判,新后端接入后自动纳入。
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/event.h>
#include <frame/hal/executable.h>
#include <frame/hal/stream.h>
#include <frame/ir/graph.h>
#include <frame/ops/op_registry.h>

#include "../ir/ir_test_helpers.h"

namespace {

// M10 素材(ARCH-031 加严判定方法专用):恰 1 输入、全程不为任何后端注册
// kernel 的测试算子——保证套件"对任意注册后端参数化执行"这一性质长期成立
// (新后端接入后天然也缺这个算子的 kernel,无需逐后端特判)。schema 全局注册
// 一次,见文件尾 FRAME_REGISTER_OP。
constexpr std::string_view kUnsupportedOpName = "test_hal_conformance_unsupported_op";

frame::Result<std::vector<frame::Shape>> InferUnsupportedOpShape(
    const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'test_hal_conformance_unsupported_op' expects 1 input, got " +
                                   std::to_string(ctx.input_types.size()));
  }
  return std::vector<frame::Shape>{ctx.input_types[0].shape};
}

}  // namespace

FRAME_REGISTER_OP(kUnsupportedOpName)
    .input("x", "input tensor")
    .output("out",
            "M10 hal_conformance ARCH-031 test-only op (never given a kernel on any "
            "backend)")
    .shape_infer(&InferUnsupportedOpShape);

namespace {

// 见 test_hal_conformance_stub.cpp 同名函数头注释:BackendRegistry::available()
// 借用的 string_view 拷贝为 std::string,供参数化生成器长期持有。
std::vector<std::string> AvailableBackendNames() {
  std::vector<std::string> names;
  for (std::string_view name : frame::hal::BackendRegistry::instance().available()) {
    names.emplace_back(name);
  }
  return names;
}

class HalConformanceTest : public ::testing::TestWithParam<std::string> {
 protected:
  // GetParam() 恒取自 BackendRegistry::available(),故 get() 理应必然成功;
  // 若失败视为测试基础设施本身故障(而非被测行为),ADD_FAILURE 后返回
  // nullptr,调用方须 ASSERT_NE(backend, nullptr) 后再继续。
  frame::hal::Backend* GetBackend() {
    const frame::Result<frame::hal::Backend*> result =
        frame::hal::BackendRegistry::instance().get(GetParam());
    if (!result.is_ok()) {
      ADD_FAILURE() << "backend not found via BackendRegistry::available(): " << GetParam();
      return nullptr;
    }
    return result.value();
  }

  // 后端首个设备(v0 全部后端保证 device_count() >= 1,见下方
  // DeviceCountIsAtLeastOne)。backend 参数取自 GetBackend() 的返回值,借
  // Backend::name() 而非 GetParam() 构造,避免额外一次 string_view/string 比较。
  static frame::Device DeviceZero(frame::hal::Backend* backend) {
    return frame::Device{backend->name(), 0};
  }
};

TEST_P(HalConformanceTest, GetReturnsBackendWithMatchingName) {
  const frame::Result<frame::hal::Backend*> result =
      frame::hal::BackendRegistry::instance().get(GetParam());
  ASSERT_TRUE(result.is_ok());
  EXPECT_EQ(result.value()->name(), std::string_view(GetParam()));
}

TEST_P(HalConformanceTest, DeviceCountIsAtLeastOne) {
  frame::hal::Backend* backend = GetBackend();
  ASSERT_NE(backend, nullptr);

  const frame::Result<int32_t> count = backend->device_count();
  ASSERT_TRUE(count.is_ok());
  EXPECT_GE(count.value(), 1);
}

TEST_P(HalConformanceTest, CreateStreamSucceeds) {
  frame::hal::Backend* backend = GetBackend();
  ASSERT_NE(backend, nullptr);

  const frame::Result<std::unique_ptr<frame::hal::Stream>> stream =
      backend->create_stream(DeviceZero(backend));
  EXPECT_TRUE(stream.is_ok());
}

TEST_P(HalConformanceTest, CreateEventSucceedsAndUnrecordedEventIsCompleted) {
  frame::hal::Backend* backend = GetBackend();
  ASSERT_NE(backend, nullptr);

  const frame::Result<std::unique_ptr<frame::hal::Event>> event =
      backend->create_event(DeviceZero(backend));
  ASSERT_TRUE(event.is_ok());
  // 未 record 语义定案(backend-hal.md 2.3,与 CUDA cudaEventCreate 未 record
  // 即成功语义一致):query() 恒为 true,synchronize() 恒返回 Ok。
  EXPECT_TRUE(event.value()->query());
  EXPECT_TRUE(event.value()->synchronize().is_ok());
}

TEST_P(HalConformanceTest, StreamRecordThenWaitThenSynchronizeAllSucceed) {
  frame::hal::Backend* backend = GetBackend();
  ASSERT_NE(backend, nullptr);
  const frame::Device device = DeviceZero(backend);

  const frame::Result<std::unique_ptr<frame::hal::Stream>> stream = backend->create_stream(device);
  ASSERT_TRUE(stream.is_ok());
  const frame::Result<std::unique_ptr<frame::hal::Event>> event = backend->create_event(device);
  ASSERT_TRUE(event.is_ok());

  EXPECT_TRUE(stream.value()->record(*event.value()).is_ok());
  EXPECT_TRUE(stream.value()->wait(*event.value()).is_ok());
  EXPECT_TRUE(stream.value()->synchronize().is_ok());
}

// 分配指针不假设可 host 端直接解引用(完成判据见 docs/backends/cuda.md
// 第 4 章 Allocator 行待办标注):往返改经
// Backend::copy(H2D 写入、D2H 读回)中转校验,不再对 allocate() 返回的指针
// 做 host 端 memset/memcmp。cpu 路径 Backend::copy 的 H2H 语义等价 memmove,
// 往返结果不变;真实异步后端(如 cuda)在 memcmp 前需 stream->synchronize()
// (同 CopyHostToHostRoundTripPreservesBytes 用例的既有教训)。
TEST_P(HalConformanceTest, AllocatorRoundTripAtAlignment64) {
  frame::hal::Backend* backend = GetBackend();
  ASSERT_NE(backend, nullptr);
  const frame::Device device = DeviceZero(backend);
  frame::hal::Allocator* allocator = backend->allocator(device);
  ASSERT_NE(allocator, nullptr);

  constexpr size_t kAlignment = 64;
  constexpr size_t kBytes = 256;
  const frame::Result<void*> allocated = allocator->allocate(kBytes, kAlignment);
  ASSERT_TRUE(allocated.is_ok());
  void* ptr = allocated.value();
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % kAlignment, 0u);

  const frame::Result<std::unique_ptr<frame::hal::Stream>> stream = backend->create_stream(device);
  ASSERT_TRUE(stream.is_ok());

  // host 侧准备 pattern -> copy(H2D)写入分配指针 -> copy(D2H)读回 host 缓冲 ->
  // synchronize -> memcmp host 缓冲(全程不解引用 ptr)。
  std::vector<unsigned char> pattern(kBytes);
  for (size_t i = 0; i < kBytes; ++i) pattern[i] = static_cast<unsigned char>(i * 7 + 0x5A);
  std::vector<unsigned char> readback(kBytes, 0);
  ASSERT_TRUE(
      backend->copy(ptr, device, pattern.data(), frame::cpu_device(), kBytes, stream.value().get())
          .is_ok());
  ASSERT_TRUE(
      backend->copy(readback.data(), frame::cpu_device(), ptr, device, kBytes, stream.value().get())
          .is_ok());
  ASSERT_TRUE(stream.value()->synchronize().is_ok());
  EXPECT_EQ(std::memcmp(readback.data(), pattern.data(), kBytes), 0);

  allocator->deallocate(ptr);
}

TEST_P(HalConformanceTest, CopyHostToHostRoundTripPreservesBytes) {
  frame::hal::Backend* backend = GetBackend();
  ASSERT_NE(backend, nullptr);
  const frame::Device device = DeviceZero(backend);
  frame::hal::Allocator* allocator = backend->allocator(device);
  ASSERT_NE(allocator, nullptr);

  constexpr size_t kBytes = 128;
  const frame::Result<void*> src_result = allocator->allocate(kBytes, 64);
  const frame::Result<void*> dst_result = allocator->allocate(kBytes, 64);
  const frame::Result<void*> back_result = allocator->allocate(kBytes, 64);
  ASSERT_TRUE(src_result.is_ok());
  ASSERT_TRUE(dst_result.is_ok());
  ASSERT_TRUE(back_result.is_ok());
  void* src = src_result.value();
  void* dst = dst_result.value();
  void* back = back_result.value();

  std::vector<unsigned char> pattern(kBytes);
  for (size_t i = 0; i < kBytes; ++i) pattern[i] = static_cast<unsigned char>(i * 11 + 3);

  const frame::Result<std::unique_ptr<frame::hal::Stream>> stream = backend->create_stream(device);
  ASSERT_TRUE(stream.is_ok());

  // 分配指针不假设可 host 端直接解引用(同 AllocatorRoundTripAtAlignment64
  // 用例理由):pattern 经 backend->copy 写入 src(而非 std::memcpy),往返
  // src -> dst -> back(本用例主体,经两次 backend->copy)后再经 backend->copy
  // 读回 host 缓冲比对,全程不解引用 src/dst/back。
  ASSERT_TRUE(
      backend->copy(src, device, pattern.data(), frame::cpu_device(), kBytes, stream.value().get())
          .is_ok());
  ASSERT_TRUE(backend->copy(dst, device, src, device, kBytes, stream.value().get()).is_ok());
  ASSERT_TRUE(backend->copy(back, device, dst, device, kBytes, stream.value().get()).is_ok());
  std::vector<unsigned char> readback(kBytes, 0);
  ASSERT_TRUE(
      backend
          ->copy(readback.data(), frame::cpu_device(), back, device, kBytes, stream.value().get())
          .is_ok());
  // backend-hal.md 2.1:copy 是"异步拷贝"(方向由两端 Device 推导,stream 非空
  // 时排入该流异步执行,调用方负责同步)。此前本用例漏了这一步,对已注册的
  // 真实异步后端(如 cuda)两次 copy 后直接 host 端 memcmp 是竞态读取——同目录
  // test_hal_conformance_stub.cpp::AllBackendsSatisfyHalContract 的同款 H2H
  // 往返用例在 memcmp 前已有 stream->synchronize(),此处补齐保持一致(M11
  // 任务0审计结论:cuda 环境下本用例失败系测试自身缺同步,非 CudaBackend::copy
  // 实现缺陷)。
  ASSERT_TRUE(stream.value()->synchronize().is_ok());
  EXPECT_EQ(std::memcmp(readback.data(), pattern.data(), kBytes), 0);

  allocator->deallocate(src);
  allocator->deallocate(dst);
  allocator->deallocate(back);
}

// ARCH-031 加严判定方法(M10,design-reviewer REVISE 闭环修订1):不支持算子的
// compile() 报错码统一为 kUnimplemented(消息含算子名),供上层 runtime::compile
// 据此触发回退链(docs/architecture/execution-model.md 第5章)。本用例直接调用
// Backend::compile(不经 runtime::compile/标准管线),对 BackendRegistry::
// available() 参数化的任意已注册后端执行,套件保持"对任意注册后端参数化执行"
// 这一性质(新后端接入后自动纳入,无需改动本文件)。
TEST_P(HalConformanceTest, CompileRejectsUnsupportedOpWithUnimplementedErrorContainingOpName) {
  frame::hal::Backend* backend = GetBackend();
  ASSERT_NE(backend, nullptr);
  const frame::Device device = DeviceZero(backend);

  frame::ir::Graph graph;
  const frame::Result<frame::ir::Value*> input_result =
      graph.add_graph_input(frame::ir::testing::MakeFloat32Type({4}, device));
  ASSERT_TRUE(input_result.is_ok()) << input_result.status().message();
  const frame::Result<frame::ir::Node*> node_result =
      graph.create_node(std::string(kUnsupportedOpName), {input_result.value()},
                        {frame::ir::testing::MakeFloat32Type({4}, device)});
  ASSERT_TRUE(node_result.is_ok()) << node_result.status().message();
  ASSERT_TRUE(graph.mark_output(node_result.value(), 0).is_ok());

  const frame::Result<std::unique_ptr<frame::hal::Executable>> compiled =
      backend->compile(graph, frame::hal::CompileOptions{});
  ASSERT_FALSE(compiled.is_ok());
  EXPECT_EQ(compiled.status().code(), frame::ErrorCode::kUnimplemented);
  EXPECT_NE(compiled.status().message().find(std::string(kUnsupportedOpName)), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(RegisteredBackends, HalConformanceTest,
                         ::testing::ValuesIn(AvailableBackendNames()),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                           return info.param;
                         });

}  // namespace
