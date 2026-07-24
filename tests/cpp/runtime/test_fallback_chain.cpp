// runtime::compile eager 回退链①②③ 单测(M10,权威依据:
// docs/architecture/execution-model.md 第5章"v0 实现口径"、
// docs/architecture/backend-hal.md ARCH-031(加严后)、
// src/runtime/{compile.cpp,fallback_executable.h/.cpp,fallback_stats.h/.cpp}、
// scratchpad/m10-design-brief.md 决议点 A-E 与其"design-reviewer REVISE 闭环
// 裁决修订"1-4、scratchpad/m10-impl-spec.md 红旗清单)。
//
// 每条用例注释标注"触发面":
//   backend_lowering 面 = 经标准管线内 backend_lowering pass 的逐 kernel
//     支持性判定失败(哨兵码翻译落点之一,src/compiler/passes/
//     backend_lowering.cpp);
//   Backend::compile 面  = 经 Backend::compile 的整图支持性判定失败(哨兵码
//     翻译落点之二,如 src/backends/cpu/cpu_executable.cpp,或本文件
//     HostFakeBackend 对"整图模式"后端的模拟);
//   N/A(负例)         = 编译在到达任一哨兵码翻译落点之前就已失败(配置错误
//     /图非法),不涉及回退决策。
//
// 覆盖清单(交付清单编号对应报告):
//   1. UnsupportedOpWithNoDecompositionFallsBackToCpuReferenceKernel  —— ③跳 + 8a
//   2. SquareDecomposesToMulExecutedEagerlyOnTargetBackend            —— ②跳
//   3. SquareDecomposesThenFallsBackToCpuWhenTargetAlsoLacksMulKernel —— ②→③跳
//   4. SingleLevelDecompositionDoesNotRecurseAndHardFails...          —— 单层分解防递归硬失败
//   5. FusableChainFallbackConsumesUnfusedOriginalGraph...            —— 融合原图锁定
//   6. MisspelledBackendNameHardFailsWithoutSilentFallback            —— 负例(a)
//   7. IllegalGraphShapeMismatchPassesThroughWithoutFallback          —— 负例(b)
//   8. SentinelCodeTriggersAtBackendCompileWholeGraphFace             —— 双触发面之
//   Backend::compile 面
//   9. CacheHitDoesNotIncrementFallbackStatsAndResetZeroesCount       —— 缓存/统计语义
//  10. EagerFallbackEmitsAtLeastOneWarnLogLine                        —— WARN 日志
//
// OpRegistry/KernelRegistry/BackendRegistry 均为进程级 Meyer's singleton,本
// 文件注册的自定义算子名以 "test_fallback_chain_" 前缀、fake 后端名以
// "test_fallback_chain_backend_" 前缀,跨 frame_test_runtime 目标内全体源文件
// (test_runtime_compile.cpp/test_compilation_cache.cpp/本文件)保持进程级
// 唯一,且各用例互不共用注册键(同 test_compilation_cache.cpp 头注释纪律)。
//
// v0 内存边界(execution-model.md 第5章)下,全部runtime Tensor 输入/输出统一
// 经 cpu 后端 Allocator 构造、device=cpu——FallbackExecutable::run 的防御式
// device 校验（validate_host_capable_device）恒接受 cpu,故无需为每个 fake
// 目标后端单独分配内存,直接复用 cpu 端 buffer 驱动全部数值断言(BUILD-011,
// 容差经 tests/cpp/common/tolerance.h)。
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/event.h>
#include <frame/hal/executable.h>
#include <frame/hal/stream.h>
#include <frame/ir/graph.h>
#include <frame/ops/kernel_registry.h>
#include <frame/ops/op_registry.h>
#include <frame/runtime/compile.h>
#include <frame/runtime/fallback_stats.h>

#include "../common/tolerance.h"
#include "../ir/ir_test_helpers.h"

namespace {

using frame::DType;
using frame::DTypeCode;
using frame::ErrorCode;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::Tensor;
using frame::hal::Backend;
using frame::hal::BackendRegistry;
using frame::hal::CompileOptions;
using frame::hal::Executable;
using frame::hal::KernelInvocation;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::Value;
using frame::ir::testing::MakeFloat32Type;
using frame::testing::default_tolerance;
using frame::testing::tensor_all_close;

// ---------------------------------------------------------------------------
// 自定义测试算子(schema 全局注册一次,见文件尾 FRAME_REGISTER_OP/KERNEL)。
// ---------------------------------------------------------------------------

// 交付点1/9/10 素材:恰 1 输入、cpu 参考实现存在、目标后端故意不注册 kernel、
// 无 decomposition —— 用以驱动 ③ 跳(逐 kernel 面)。
constexpr std::string_view kOp1Name = "test_fallback_chain_op1";

// 交付点4 素材:outer 声明 decomposition,展开出 inner;inner 全程(任意后端与
// cpu)均不注册 kernel —— 单层分解防递归的"双缺硬失败"红旗分支专用。
constexpr std::string_view kDecomposeOuterOp = "test_fallback_chain_decompose_outer";
constexpr std::string_view kDecomposeInnerOp = "test_fallback_chain_decompose_inner";

// 三个自定义算子共用的 shape 推断:恰 1 输入,输出 shape 恒等于输入 shape
// (REUSE-002:三者结构相同,不各自复制一份同构校验)。
Result<std::vector<Shape>> InferIdentityUnaryShape(const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument, "op '" + std::string(ctx.op) +
                                                         "' expects 1 input, got " +
                                                         std::to_string(ctx.input_types.size()));
  }
  return std::vector<Shape>{ctx.input_types[0].shape};
}

// op1 的 cpu 参考实现:output = input + 10(非平凡、可区分于恒等/未初始化值,
// 便于数值比对失败时一眼看出差异)。
Status Op1CpuKernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1 || ctx.outputs.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'test_fallback_chain_op1' cpu kernel expects 1 input and 1 output");
  }
  const float* input = static_cast<const float*>(ctx.inputs[0].raw_data());
  float* output = ctx.outputs[0].data<float>();
  const int64_t numel = ctx.outputs[0].numel();
  for (int64_t i = 0; i < numel; ++i) output[i] = input[i] + 10.0F;
  return Status::ok();
}

// outer 的 decomposition:纯转发给 inner(square_decompose 同款结构精简版,
// src/ops/schemas/elementwise.cpp)——微图仅 1 个 graph_input、1 个 inner
// 节点、mark_output 该节点唯一输出,按位对应契约天然满足。
Result<Graph> DecomposeOuterToInner(const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'test_fallback_chain_decompose_outer' decomposition expects 1 input, "
                        "got " +
                            std::to_string(ctx.input_types.size()));
  }
  Graph graph("test_fallback_chain_decompose_outer_micro");
  const Result<Value*> input_result = graph.add_graph_input(ctx.input_types[0]);
  if (!input_result.is_ok()) return input_result.status();
  Value* x = input_result.value();

  const Result<Node*> inner_result =
      graph.create_node(std::string(kDecomposeInnerOp), {x}, {ctx.input_types[0]});
  if (!inner_result.is_ok()) return inner_result.status();
  Node* inner_node = inner_result.value();

  const Status mark_status = graph.mark_output(inner_node, 0);
  if (!mark_status.is_ok()) return mark_status;
  return graph;
}

// ---------------------------------------------------------------------------
// 功能性 fake 后端(m10-design-brief 决议点 E 先例扩展,REUSE-002:全部用例
// 共用同一个类,只是构造实例时传不同的注册名/是否注册 kernel):
//   allocator() = host(委托 cpu 参考后端的分配器,v0 内存边界"现注册面全为
//     host 内存后端"——不新写第二套 malloc/free 实现);
//   launch()    = 委托 KernelRegistry::find(op, name()) 后直调(与
//     CpuBackend::launch 同型,REUSE-002),使①(eager launch)方案可真执行;
//   compile()   = 恒对整图返回 kUnimplemented(消息含图内全部算子名),模拟
//     "整图模式"后端(如 intel_npu,backend_lowering.cpp 头注释)对 ARCH-031
//     的第二落点(Backend::compile 面);compile_call_count 供断言"该落点是否
//     被触达"(区分两个触发面的关键信号)。
// ---------------------------------------------------------------------------
class HostFakeBackend final : public Backend {
 public:
  explicit HostFakeBackend(std::string_view name) : name_(name) {}

  std::string_view name() const override { return name_; }
  Result<int32_t> device_count() const override { return int32_t{1}; }
  Result<std::unique_ptr<frame::hal::Stream>> create_stream(frame::Device) override {
    return Status::make(ErrorCode::kUnimplemented, "HostFakeBackend::create_stream");
  }
  Result<std::unique_ptr<frame::hal::Event>> create_event(frame::Device) override {
    return Status::make(ErrorCode::kUnimplemented, "HostFakeBackend::create_event");
  }
  frame::hal::Allocator* allocator(frame::Device) override {
    const Result<Backend*> cpu = BackendRegistry::instance().get(frame::kCpuBackendName);
    if (!cpu.is_ok()) return nullptr;
    return cpu.value()->allocator(frame::cpu_device());
  }
  Status copy(void*, frame::Device, const void*, frame::Device, size_t,
              frame::hal::Stream*) override {
    return Status::make(ErrorCode::kUnimplemented, "HostFakeBackend::copy");
  }
  Result<std::unique_ptr<Executable>> compile(const Graph& graph, const CompileOptions&) override {
    ++compile_call_count;
    std::string op_names;
    for (const Node* node : graph.topological_order()) {
      if (node->op() == frame::ir::kGraphInputOp) continue;
      if (!op_names.empty()) op_names += ", ";
      op_names += node->op();
    }
    return Status::make(ErrorCode::kUnimplemented,
                        "HostFakeBackend::compile: unsupported in whole-graph mode (test double "
                        "simulating an ARCH-031 whole-graph-mode backend), op(s): [" +
                            op_names + "]");
  }
  Status launch(const KernelInvocation& invocation, frame::hal::Stream* stream) override {
    const Result<frame::ops::KernelFn> kernel =
        frame::ops::KernelRegistry::instance().find(invocation.op, name_);
    if (!kernel.is_ok()) return kernel.status();
    frame::ops::KernelContext context{invocation.inputs, invocation.outputs, invocation.attrs,
                                      invocation.device, stream};
    return kernel.value()(context);
  }

  int compile_call_count = 0;

 private:
  std::string_view name_;
};

// ---------------------------------------------------------------------------
// 共用图构造辅助(REUSE-002:全部一元算子用例共用同一形状 {4})。
// ---------------------------------------------------------------------------
Graph BuildUnaryOpGraph(std::string_view op, frame::Device device) {
  Graph graph;
  Value* input = graph.add_graph_input(MakeFloat32Type({4}, device)).value();
  Node* node = graph.create_node(std::string(op), {input}, {MakeFloat32Type({4}, device)}).value();
  graph.mark_output(node, 0);
  return graph;
}

Graph BuildAddReluGraph(frame::Device device) {
  Graph graph;
  Value* a = graph.add_graph_input(MakeFloat32Type({4}, device)).value();
  Value* b = graph.add_graph_input(MakeFloat32Type({4}, device)).value();
  Node* add_node = graph.create_node("add", {a, b}, {MakeFloat32Type({4}, device)}).value();
  Node* relu_node =
      graph.create_node("relu", {add_node->output(0)}, {MakeFloat32Type({4}, device)}).value();
  graph.mark_output(relu_node, 0);
  return graph;
}

// ---------------------------------------------------------------------------
// 共用 fixture:cpu 后端真实 Allocator/Stream(v0 内存边界,见文件头注释)。
// ---------------------------------------------------------------------------
class FallbackChainTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const Result<Backend*> result = BackendRegistry::instance().get(frame::kCpuBackendName);
    ASSERT_TRUE(result.is_ok());
    cpu_backend_ = result.value();
    cpu_device_ = frame::cpu_device();
    allocator_ = cpu_backend_->allocator(cpu_device_);
    ASSERT_NE(allocator_, nullptr);
    Result<std::unique_ptr<frame::hal::Stream>> stream_result =
        cpu_backend_->create_stream(cpu_device_);
    ASSERT_TRUE(stream_result.is_ok());
    stream_ = std::move(stream_result.value());
  }

  // 非平凡、可区分的填充值(同 test_runtime_compile.cpp::MakeFilledTensor
  // 惯例,独立实现不复用其文件私有辅助函数——同 test_operator_fusion.cpp 的
  // 既有先例注释,REUSE-002 意义上属"过于琐碎不值得抽取共享头"的合理例外)。
  Tensor MakeFilledTensor(const Shape& shape, float start) {
    Tensor tensor = Tensor::empty(shape, DType::of<float>(), cpu_device_, *allocator_).value();
    float* data = tensor.data<float>();
    for (int64_t i = 0; i < tensor.numel(); ++i) data[i] = start + static_cast<float>(i) * 0.5F;
    return tensor;
  }

  Tensor RunSingleOutput(Executable& executable, const std::vector<Tensor>& inputs,
                         const Shape& output_shape) {
    std::vector<Tensor> outputs{
        Tensor::empty(output_shape, DType::of<float>(), cpu_device_, *allocator_).value()};
    const Status status = executable.run(inputs, outputs, *stream_);
    EXPECT_TRUE(status.is_ok()) << status.message();
    return outputs[0];
  }

  Backend* cpu_backend_ = nullptr;
  frame::Device cpu_device_{};
  frame::hal::Allocator* allocator_ = nullptr;
  std::unique_ptr<frame::hal::Stream> stream_;
};

// =============================================================================
// 交付点1:③跳(触发面 = backend_lowering,逐 kernel 面)。
// 亦覆盖交付点8a(双触发面之一):compile_call_count==0 证明 Backend::compile
// 全程未被调用——触发点确系管线内 backend_lowering pass,而非 Backend::compile。
// =============================================================================
TEST_F(FallbackChainTest, UnsupportedOpWithNoDecompositionFallsBackToCpuReferenceKernel) {
  constexpr std::string_view kBackendName = "test_fallback_chain_backend_op1_no_kernel";
  auto backend_owned = std::make_unique<HostFakeBackend>(kBackendName);
  HostFakeBackend* backend_ptr = backend_owned.get();
  ASSERT_TRUE(
      BackendRegistry::instance().register_backend(kBackendName, std::move(backend_owned)).is_ok());
  // 不为 kOp1Name 在该后端注册任何 kernel。

  const frame::Device fake_device{kBackendName, 0};
  Graph fake_graph = BuildUnaryOpGraph(kOp1Name, fake_device);
  Graph cpu_graph = BuildUnaryOpGraph(kOp1Name, cpu_device_);

  const Result<std::shared_ptr<Executable>> fake_result =
      frame::runtime::compile(fake_graph, kBackendName, CompileOptions{});
  ASSERT_TRUE(fake_result.is_ok()) << fake_result.status().message();
  const Result<std::shared_ptr<Executable>> cpu_result =
      frame::runtime::compile(cpu_graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(cpu_result.is_ok()) << cpu_result.status().message();

  Tensor x = MakeFilledTensor(Shape({4}), 1.0F);
  Tensor fake_output = RunSingleOutput(*fake_result.value(), {x}, Shape({4}));
  Tensor cpu_output = RunSingleOutput(*cpu_result.value(), {x}, Shape({4}));

  EXPECT_TRUE(tensor_all_close(fake_output, cpu_output, default_tolerance(DTypeCode::kFloat32)));
  EXPECT_EQ(backend_ptr->compile_call_count, 0);  // 交付点8a:证明触发面是 backend_lowering
  EXPECT_EQ(frame::runtime::FallbackStats::instance().count(kOp1Name, kBackendName), 1);
}

// =============================================================================
// 交付点2:②跳(触发面 = backend_lowering,逐 kernel 面)。square 在该后端无
// kernel,但该后端有 mul kernel(与 cpu 版字面同一函数指针)—— decomposition
// 展开后 mul 经①在该后端 eager 执行。
// =============================================================================
TEST_F(FallbackChainTest, SquareDecomposesToMulExecutedEagerlyOnTargetBackend) {
  constexpr std::string_view kBackendName = "test_fallback_chain_backend_square_with_mul";
  auto backend_owned = std::make_unique<HostFakeBackend>(kBackendName);
  ASSERT_TRUE(
      BackendRegistry::instance().register_backend(kBackendName, std::move(backend_owned)).is_ok());

  const Result<frame::ops::KernelFn> cpu_mul_kernel =
      frame::ops::KernelRegistry::instance().find("mul", frame::kCpuBackendName);
  ASSERT_TRUE(cpu_mul_kernel.is_ok());
  ASSERT_TRUE(frame::ops::KernelRegistry::instance()
                  .register_kernel("mul", kBackendName, cpu_mul_kernel.value())
                  .is_ok());
  // 不为该后端注册 "square" kernel。

  const frame::Device fake_device{kBackendName, 0};
  Graph fake_graph = BuildUnaryOpGraph("square", fake_device);
  Graph cpu_graph = BuildUnaryOpGraph("square", cpu_device_);

  const Result<std::shared_ptr<Executable>> fake_result =
      frame::runtime::compile(fake_graph, kBackendName, CompileOptions{});
  ASSERT_TRUE(fake_result.is_ok()) << fake_result.status().message();
  const Result<std::shared_ptr<Executable>> cpu_result =
      frame::runtime::compile(cpu_graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(cpu_result.is_ok()) << cpu_result.status().message();

  Tensor x = MakeFilledTensor(Shape({4}), 2.0F);
  Tensor fake_output = RunSingleOutput(*fake_result.value(), {x}, Shape({4}));
  Tensor cpu_output = RunSingleOutput(*cpu_result.value(), {x}, Shape({4}));

  EXPECT_TRUE(tensor_all_close(fake_output, cpu_output, default_tolerance(DTypeCode::kFloat32)));
  // 肇因算子粒度:square 本身记一次(② 展开);内部 mul 经①直接命中,不计数。
  EXPECT_EQ(frame::runtime::FallbackStats::instance().count("square", kBackendName), 1);
  EXPECT_EQ(frame::runtime::FallbackStats::instance().count("mul", kBackendName), 0);
}

// =============================================================================
// 交付点3:②→③复合跳(触发面 = backend_lowering,逐 kernel 面)。该后端既无
// square 也无 mul kernel:②展开后微图节点(mul)再降级至③(cpu 参考实现)。
// =============================================================================
TEST_F(FallbackChainTest, SquareDecomposesThenFallsBackToCpuWhenTargetAlsoLacksMulKernel) {
  constexpr std::string_view kBackendName = "test_fallback_chain_backend_square_no_mul";
  auto backend_owned = std::make_unique<HostFakeBackend>(kBackendName);
  ASSERT_TRUE(
      BackendRegistry::instance().register_backend(kBackendName, std::move(backend_owned)).is_ok());
  // 不为该后端注册 "square" 或 "mul" 任何 kernel。

  const frame::Device fake_device{kBackendName, 0};
  Graph fake_graph = BuildUnaryOpGraph("square", fake_device);
  Graph cpu_graph = BuildUnaryOpGraph("square", cpu_device_);

  const Result<std::shared_ptr<Executable>> fake_result =
      frame::runtime::compile(fake_graph, kBackendName, CompileOptions{});
  ASSERT_TRUE(fake_result.is_ok()) << fake_result.status().message();
  const Result<std::shared_ptr<Executable>> cpu_result =
      frame::runtime::compile(cpu_graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(cpu_result.is_ok()) << cpu_result.status().message();

  Tensor x = MakeFilledTensor(Shape({4}), -1.0F);
  Tensor fake_output = RunSingleOutput(*fake_result.value(), {x}, Shape({4}));
  Tensor cpu_output = RunSingleOutput(*cpu_result.value(), {x}, Shape({4}));

  EXPECT_TRUE(tensor_all_close(fake_output, cpu_output, default_tolerance(DTypeCode::kFloat32)));
  // 计数粒度以肇因算子为单位:query("square", ...) 仍为1,不因内部 mul 再降级
  // 而重复计数该键;内部 mul 节点在其自身键下独立记一次(不同键)。
  EXPECT_EQ(frame::runtime::FallbackStats::instance().count("square", kBackendName), 1);
  EXPECT_EQ(frame::runtime::FallbackStats::instance().count("mul", kBackendName), 1);
}

// =============================================================================
// 交付点4:单层分解防递归(触发面 = backend_lowering,逐 kernel 面)。outer 的
// decomposition 展开出 inner;inner 在目标后端与 cpu 均无 kernel——m10-impl-
// spec.md 红旗清单第4条:双缺即整体硬失败,消息须含内外算子名。
// =============================================================================
TEST_F(FallbackChainTest,
       SingleLevelDecompositionDoesNotRecurseAndHardFailsWhenMicroGraphNodeLacksAnyKernel) {
  constexpr std::string_view kBackendName = "test_fallback_chain_backend_decompose_recursion";
  auto backend_owned = std::make_unique<HostFakeBackend>(kBackendName);
  ASSERT_TRUE(
      BackendRegistry::instance().register_backend(kBackendName, std::move(backend_owned)).is_ok());
  // 不为 outer/inner 任一算子在该后端注册 kernel;inner 算子在 cpu 后端也无
  // kernel(违反 ARCH-041,但这正是本用例要触达的防御分支——真实算子经
  // register_op+register_kernel 两步注册流程不会出现这种"只注册 schema 不注册
  // cpu kernel"的半成品状态)。

  const frame::Device fake_device{kBackendName, 0};
  Graph graph = BuildUnaryOpGraph(kDecomposeOuterOp, fake_device);

  const Result<std::shared_ptr<Executable>> result =
      frame::runtime::compile(graph, kBackendName, CompileOptions{});
  ASSERT_FALSE(result.is_ok());
  EXPECT_NE(result.status().message().find(std::string(kDecomposeOuterOp)), std::string::npos);
  EXPECT_NE(result.status().message().find(std::string(kDecomposeInnerOp)), std::string::npos);
}

// =============================================================================
// 交付点5(裁决修订4d):≥2 个 kFusable 算子的图(add→relu)在不支持后端回退
// 正确(触发面 = backend_lowering,逐 kernel 面——落在融合产物
// fused_elementwise_internal 上)。锁死"消费未融合原图":断言 FallbackStats
// 计数落在 add/relu 各自键下(而非 fused_elementwise_internal),证明
// FallbackExecutable::build 消费的是 runtime::compile 持有的原始 graph 参数,
// 而非管线内已被 operator_fusion 改写的 working_copy——若误用后者,该图会被
// 融合为单一 fused_elementwise_internal 节点,统计键会变为该融合算子名而非
// add/relu,与本断言矛盾从而暴露错误(design-reviewer REVISE 闭环修订2)。
// =============================================================================
TEST_F(FallbackChainTest, FusableChainFallbackConsumesUnfusedOriginalGraphNotFusedWorkingCopy) {
  constexpr std::string_view kBackendName = "test_fallback_chain_backend_fusion_lock";
  auto backend_owned = std::make_unique<HostFakeBackend>(kBackendName);
  ASSERT_TRUE(
      BackendRegistry::instance().register_backend(kBackendName, std::move(backend_owned)).is_ok());
  // 不为该后端注册 "add"/"relu"/"fused_elementwise_internal" 任何 kernel。

  const frame::Device fake_device{kBackendName, 0};
  Graph fake_graph = BuildAddReluGraph(fake_device);
  Graph cpu_graph = BuildAddReluGraph(cpu_device_);

  const Result<std::shared_ptr<Executable>> fake_result =
      frame::runtime::compile(fake_graph, kBackendName, CompileOptions{});
  ASSERT_TRUE(fake_result.is_ok()) << fake_result.status().message();
  const Result<std::shared_ptr<Executable>> cpu_result =
      frame::runtime::compile(cpu_graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(cpu_result.is_ok()) << cpu_result.status().message();

  // a+b 部分和为负,确保 relu 实际发生截断(而非恒等透传)。
  Tensor a = MakeFilledTensor(Shape({4}), 1.0F);   // [1, 1.5, 2, 2.5]
  Tensor b = MakeFilledTensor(Shape({4}), -3.0F);  // [-3, -2.5, -2, -1.5]
  Tensor fake_output = RunSingleOutput(*fake_result.value(), {a, b}, Shape({4}));
  Tensor cpu_output = RunSingleOutput(*cpu_result.value(), {a, b}, Shape({4}));

  EXPECT_TRUE(tensor_all_close(fake_output, cpu_output, default_tolerance(DTypeCode::kFloat32)));
  EXPECT_EQ(frame::runtime::FallbackStats::instance().count("add", kBackendName), 1);
  EXPECT_EQ(frame::runtime::FallbackStats::instance().count("relu", kBackendName), 1);
  EXPECT_EQ(
      frame::runtime::FallbackStats::instance().count("fused_elementwise_internal", kBackendName),
      0);
}

// =============================================================================
// 交付点6(裁决修订4a,负例):拼错后端名 —— N/A 触发面,BackendRegistry::get
// 本身先失败(kNotFound,未被翻译为 kUnimplemented——后端名不存在是配置错误,
// 不属"不支持"),不触发回退,绝不静默降级。
// =============================================================================
TEST_F(FallbackChainTest, MisspelledBackendNameHardFailsWithoutSilentFallback) {
  constexpr std::string_view kNeverRegisteredBackend = "test_fallback_chain_typo_backend_zzz";
  const frame::Device device{kNeverRegisteredBackend, 0};
  Graph graph = BuildUnaryOpGraph("relu", device);

  const Result<std::shared_ptr<Executable>> result =
      frame::runtime::compile(graph, kNeverRegisteredBackend, CompileOptions{});
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kNotFound);
  EXPECT_NE(result.status().message().find(std::string(kNeverRegisteredBackend)),
            std::string::npos);
  EXPECT_EQ(frame::runtime::FallbackStats::instance().count("relu", kNeverRegisteredBackend), 0);
}

// =============================================================================
// 交付点7(负例):非法图(shape_inference 报 kInvalidArgument)—— N/A 触发面,
// 错误发生在 shape_inference pass(标准管线第2段,先于 backend_lowering 第9
// 段),原样透传不回退——验证 resolve_unimplemented_with_fallback 的错误码
// 门槛仅 kUnimplemented 触发,其余错误码(含 kInvalidArgument)一律透传。
// =============================================================================
TEST_F(FallbackChainTest, IllegalGraphShapeMismatchPassesThroughWithoutFallback) {
  Graph graph;
  Value* a = graph.add_graph_input(MakeFloat32Type({4})).value();
  Value* b = graph.add_graph_input(MakeFloat32Type({4})).value();
  // 故意声明与推断结果不符的输出 shape({8} 而非正确的 {4}),触发
  // shape_inference pass 报错(src/compiler/passes/shape_inference.cpp)。
  Node* add_node = graph.create_node("add", {a, b}, {MakeFloat32Type({8})}).value();
  ASSERT_TRUE(graph.mark_output(add_node, 0).is_ok());

  const Result<std::shared_ptr<Executable>> result =
      frame::runtime::compile(graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_EQ(frame::runtime::FallbackStats::instance().count("add", frame::kCpuBackendName), 0);
}

// =============================================================================
// 交付点8(裁决修订4b 第二例,双触发面之 Backend::compile 面):该后端对
// "relu" 注册了与 cpu 字面同一个 kernel 指针(backend_lowering 逐 kernel 判定
// 因而放行),但 compile() 本身模拟"整图模式"后端对整图无条件拒绝(消息含
// 算子名)——先独立验证该 Status 本身即符合哨兵码约定(不经 runtime::compile
// 包装,消息未被回退吞掉前直接可见),再验证 runtime::compile 消费该错误码后
// 仍能经①(eager launch,直接命中,不计入 FallbackStats)成功回退。
// =============================================================================
TEST_F(FallbackChainTest, SentinelCodeTriggersAtBackendCompileWholeGraphFace) {
  constexpr std::string_view kBackendName = "test_fallback_chain_backend_compile_face";
  auto backend_owned = std::make_unique<HostFakeBackend>(kBackendName);
  HostFakeBackend* backend_ptr = backend_owned.get();
  ASSERT_TRUE(
      BackendRegistry::instance().register_backend(kBackendName, std::move(backend_owned)).is_ok());

  const Result<frame::ops::KernelFn> cpu_relu_kernel =
      frame::ops::KernelRegistry::instance().find("relu", frame::kCpuBackendName);
  ASSERT_TRUE(cpu_relu_kernel.is_ok());
  ASSERT_TRUE(frame::ops::KernelRegistry::instance()
                  .register_kernel("relu", kBackendName, cpu_relu_kernel.value())
                  .is_ok());

  const frame::Device fake_device{kBackendName, 0};
  Graph graph = BuildUnaryOpGraph("relu", fake_device);

  // 触发面隔离断言:直接调用 Backend::compile,不经 runtime::compile 包装。
  const Result<std::unique_ptr<Executable>> direct_compile_result =
      backend_ptr->compile(graph, CompileOptions{});
  ASSERT_FALSE(direct_compile_result.is_ok());
  EXPECT_EQ(direct_compile_result.status().code(), ErrorCode::kUnimplemented);
  EXPECT_NE(direct_compile_result.status().message().find("relu"), std::string::npos);
  EXPECT_EQ(backend_ptr->compile_call_count, 1);

  // 整图级 WARN 回归锁(code-reviewer 裁决级建议):全①命中时肇因算子级
  // 记录为零,回退决策点的整图级 WARN 是唯一可观测面,断言其存在与要素。
  testing::internal::CaptureStderr();
  const Result<std::shared_ptr<Executable>> result =
      frame::runtime::compile(graph, kBackendName, CompileOptions{});
  const std::string captured_stderr = testing::internal::GetCapturedStderr();
  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_EQ(backend_ptr->compile_call_count, 2);  // runtime::compile 内部又调用一次
  EXPECT_EQ(frame::runtime::FallbackStats::instance().count("relu", kBackendName),
            0);  // ①直接命中不计数
  EXPECT_NE(captured_stderr.find("[frame][WARN]"), std::string::npos);
  EXPECT_NE(captured_stderr.find(kBackendName), std::string::npos);

  Graph cpu_graph = BuildUnaryOpGraph("relu", cpu_device_);
  const Result<std::shared_ptr<Executable>> cpu_result =
      frame::runtime::compile(cpu_graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(cpu_result.is_ok()) << cpu_result.status().message();

  Tensor x = MakeFilledTensor(Shape({4}), -1.0F);
  Tensor fake_output = RunSingleOutput(*result.value(), {x}, Shape({4}));
  Tensor cpu_output = RunSingleOutput(*cpu_result.value(), {x}, Shape({4}));
  EXPECT_TRUE(tensor_all_close(fake_output, cpu_output, default_tolerance(DTypeCode::kFloat32)));
}

// =============================================================================
// 交付点9:缓存与统计语义(触发面 = backend_lowering,逐 kernel 面;仅关注
// 缓存/统计,复用 op1/③跳素材)。决议点 A/C:统计语义 = 回退决策次数,缓存
// 命中不重复触发;reset() 供测试隔离,清零后重新查询为0。
//
// 全局单例说明:FallbackStats::reset() 清空的是进程级全局表,但本文件每条
// 用例均使用互不重叠的 (op, backend) 键组合、且断言仅在触发动作之后立即读取
// 自身键的计数,不依赖其他用例遗留的累计状态,故此处调用全局 reset() 不会
// 影响本文件内其余用例的正确性(无论 gtest 实际执行顺序如何)。
// =============================================================================
TEST_F(FallbackChainTest, CacheHitDoesNotIncrementFallbackStatsAndResetZeroesCount) {
  constexpr std::string_view kBackendName = "test_fallback_chain_backend_cache_semantics";
  auto backend_owned = std::make_unique<HostFakeBackend>(kBackendName);
  ASSERT_TRUE(
      BackendRegistry::instance().register_backend(kBackendName, std::move(backend_owned)).is_ok());

  const frame::Device fake_device{kBackendName, 0};
  Graph graph = BuildUnaryOpGraph(kOp1Name, fake_device);

  const CompileOptions options;
  const Result<std::shared_ptr<Executable>> first =
      frame::runtime::compile(graph, kBackendName, options);
  ASSERT_TRUE(first.is_ok()) << first.status().message();
  EXPECT_EQ(frame::runtime::FallbackStats::instance().count(kOp1Name, kBackendName), 1);

  const Result<std::shared_ptr<Executable>> second =
      frame::runtime::compile(graph, kBackendName, options);
  ASSERT_TRUE(second.is_ok()) << second.status().message();
  EXPECT_EQ(second.value(), first.value());  // 缓存命中:同一 Executable
  EXPECT_EQ(frame::runtime::FallbackStats::instance().count(kOp1Name, kBackendName),
            1);  // 不重复计数

  frame::runtime::FallbackStats::instance().reset();
  EXPECT_EQ(frame::runtime::FallbackStats::instance().count(kOp1Name, kBackendName), 0);
}

// =============================================================================
// 交付点10:WARN 日志(触发面 = backend_lowering,逐 kernel 面;复用 op1/③跳
// 素材)。取舍说明:warn_log 实现是单份 fprintf(stderr, "[frame][WARN] ...")
// (src/runtime/warn_log.cpp),经 testing::internal::CaptureStderr()/
// GetCapturedStderr() 直接捕获——这是 GoogleTest 内建能力,无需额外基础设施
// 成本,且比"以 FallbackStats 计数代证"更贴近 execution-model.md 第5章"日志"
// 要求原文本身(消息字段:算子名/后端名/回退目标/原因),故取直接捕获而非
// 代理断言。
// =============================================================================
TEST_F(FallbackChainTest, EagerFallbackEmitsAtLeastOneWarnLogLine) {
  constexpr std::string_view kBackendName = "test_fallback_chain_backend_warn_log";
  auto backend_owned = std::make_unique<HostFakeBackend>(kBackendName);
  ASSERT_TRUE(
      BackendRegistry::instance().register_backend(kBackendName, std::move(backend_owned)).is_ok());

  const frame::Device fake_device{kBackendName, 0};
  Graph graph = BuildUnaryOpGraph(kOp1Name, fake_device);

  testing::internal::CaptureStderr();
  const Result<std::shared_ptr<Executable>> result =
      frame::runtime::compile(graph, kBackendName, CompileOptions{});
  const std::string captured = testing::internal::GetCapturedStderr();

  ASSERT_TRUE(result.is_ok()) << result.status().message();
  EXPECT_NE(captured.find("[frame][WARN]"), std::string::npos);
  EXPECT_NE(captured.find(std::string(kOp1Name)), std::string::npos);
  EXPECT_NE(captured.find(std::string(kBackendName)), std::string::npos);
  EXPECT_NE(captured.find("cpu-reference"), std::string::npos);
}

}  // namespace

FRAME_REGISTER_OP(kOp1Name)
    .input("x", "input tensor")
    .output("out", "op1(x) = x + 10 (M10 fallback chain test-only op)")
    .shape_infer(&InferIdentityUnaryShape);

FRAME_REGISTER_OP(kDecomposeInnerOp)
    .input("x", "inner op input")
    .output("out",
            "inner op output (deliberately never given a kernel on any backend, including cpu, to "
            "exercise the single-level-decomposition double-miss defensive branch)")
    .shape_infer(&InferIdentityUnaryShape);

FRAME_REGISTER_OP(kDecomposeOuterOp)
    .input("x", "outer op input")
    .output("out", "outer op output")
    .shape_infer(&InferIdentityUnaryShape)
    .decomposition(&DecomposeOuterToInner);

FRAME_REGISTER_KERNEL(kOp1Name, frame::kCpuBackendName, Op1CpuKernel);
