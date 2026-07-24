// runtime::compile 端到端单测(m7-design-brief 决议点 4/6):
//   1. 端到端:matmul([2,3]x[3,4]) -> add(+bias[2,4]) -> relu 经
//      runtime::compile("cpu") 编译执行,结果与 eager 三连 launch(ARCH-011
//      第3类准入)数值一致(容差工具断言,fp32 默认档);同一图形态另补 fp16
//      一条(default_tolerance(fp16)),覆盖非 fp32 dtype 路径。
//   2. run() 签名校验:输入个数/shape/dtype、输出 shape 不符均拒绝。
//   3. --dump-ir-after 可观测性(compile.h 头注释"窄接口"一节所述能力):经
//      直接组合 PassManager 在端到端图上冒烟一条(M6 已测机制本身,这里只
//      验证与本里程碑图形态的组合)。
//   4. 边角:恒等/仅输入图(单图输出)已覆盖;补一条多图输出(matmul 中间结果
//      与 relu 最终结果各自 mark_output),验证 CpuExecutable 的
//      output_slots_/output_signature_ 在图输出个数 >1 时逐个正确处理。
//   5. sum(带 axes 属性)入编译路径,且刻意在原始 Graph 析构后才调用
//      Executable::run——验证 CpuExecutable::Step::attrs 是编译期自持的值
//      拷贝(见 src/backends/cpu/cpu_executable.h 头注释),运行期不回指已
//      销毁的 Graph/Node。
// 主路径全程编译执行(ARCH-010);eager 三连 launch 仅作数值参照基线,属
// ARCH-011 第3类单算子单测准入。
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>
#include <string_view>
#include <vector>

#include <frame/compiler/pass_manager.h>
#include <frame/compiler/pipeline.h>
#include <frame/core/dtype.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/executable.h>
#include <frame/hal/stream.h>
#include <frame/ir/attribute.h>
#include <frame/ir/graph.h>
#include <frame/ir/serialization.h>
#include <frame/runtime/compile.h>

#include "../common/tolerance.h"
#include "../ir/ir_test_helpers.h"
#include "../ops/elementwise_op_test_helpers.h"

namespace {

using frame::DType;
using frame::DTypeCode;
using frame::ErrorCode;
using frame::float16_t;
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
using frame::ops::testing::MakeType;
using frame::testing::default_tolerance;
using frame::testing::tensor_all_close;

// 构建 matmul([2,3]x[3,4]) -> add(+bias[2,4]) -> relu 的编译路径图:3 个
// graph_input(x, w, bias),mark_output 唯一输出。dtype 参数化(默认 fp32)以
// 复用同一构图逻辑覆盖 fp16 端到端用例,避免维护第二份雷同图构造代码
// (REUSE-002)。
Graph BuildMatmulAddReluGraph(DType dtype = DType::of<float>()) {
  Graph graph("matmul_add_relu");
  Value* x = graph.add_graph_input(MakeType(dtype, {2, 3})).value();
  Value* w = graph.add_graph_input(MakeType(dtype, {3, 4})).value();
  Value* bias = graph.add_graph_input(MakeType(dtype, {2, 4})).value();
  Node* matmul_node = graph.create_node("matmul", {x, w}, {MakeType(dtype, {2, 4})}).value();
  Node* add_node =
      graph.create_node("add", {matmul_node->output(0), bias}, {MakeType(dtype, {2, 4})}).value();
  Node* relu_node =
      graph.create_node("relu", {add_node->output(0)}, {MakeType(dtype, {2, 4})}).value();
  graph.mark_output(relu_node, 0);
  return graph;
}

// 非平凡输入值:逐元素填充可区分的浮点序列(而非全零/全一)。
Tensor MakeFilledTensor(const Shape& shape, float start, frame::Device device,
                        frame::hal::Allocator& allocator) {
  Tensor tensor = Tensor::empty(shape, DType::of<float>(), device, allocator).value();
  float* data = tensor.data<float>();
  for (int64_t i = 0; i < tensor.numel(); ++i) {
    data[i] = start + static_cast<float>(i) * 0.5F;
  }
  return tensor;
}

// fp16 版本:值先以 float 计算再经既有位级转换降精度(REUSE-002:复用
// dtype.h 的 float_to_float16,不新写第二套转换逻辑)。
Tensor MakeFilledFloat16Tensor(const Shape& shape, float start, frame::Device device,
                               frame::hal::Allocator& allocator) {
  Tensor tensor = Tensor::empty(shape, DType::of<float16_t>(), device, allocator).value();
  float16_t* data = tensor.data<float16_t>();
  for (int64_t i = 0; i < tensor.numel(); ++i) {
    data[i] = frame::float_to_float16(start + static_cast<float>(i) * 0.5F);
  }
  return tensor;
}

class RuntimeCompileTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const Result<Backend*> result = BackendRegistry::instance().get(frame::kCpuBackendName);
    ASSERT_TRUE(result.is_ok());
    backend_ = result.value();
    device_ = frame::cpu_device();
    allocator_ = backend_->allocator(device_);
    ASSERT_NE(allocator_, nullptr);
  }

  // 单算子 eager launch 便捷封装(ARCH-011 第3类准入:单算子单元测试)。
  // inputs/outputs 相邻同类型形参(clang-tidy bugprone-easily-swappable-
  // parameters)——语义上二者确不可合并/重排(kernel 调用天然区分输入输出),
  // 且全部调用点均以具名局部变量传参、误置换会在下一步断言立即暴露(shape
  // 不符),抑制而非强行重排参数顺序或拆分成两个更别扭的重载。
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  Status EagerLaunch(std::string_view op, std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                     frame::hal::Stream& stream) {
    KernelInvocation invocation;
    invocation.op = op;
    invocation.inputs = inputs;
    invocation.outputs = outputs;
    invocation.device = device_;
    return backend_->launch(invocation, &stream);
  }

  Backend* backend_ = nullptr;
  frame::Device device_{};
  frame::hal::Allocator* allocator_ = nullptr;
};

TEST_F(RuntimeCompileTest, EndToEndMatmulAddReluMatchesEagerThreeLaunches) {
  const Graph graph = BuildMatmulAddReluGraph();

  const Result<std::shared_ptr<Executable>> executable_result =
      frame::runtime::compile(graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable_result.is_ok()) << executable_result.status().message();
  const std::shared_ptr<Executable>& executable = executable_result.value();

  Tensor x = MakeFilledTensor(Shape({2, 3}), 1.0F, device_, *allocator_);
  Tensor w = MakeFilledTensor(Shape({3, 4}), -0.5F, device_, *allocator_);
  Tensor bias = MakeFilledTensor(Shape({2, 4}), 0.25F, device_, *allocator_);

  std::vector<Tensor> inputs{x, w, bias};
  std::vector<Tensor> outputs{
      Tensor::empty(Shape({2, 4}), DType::of<float>(), device_, *allocator_).value()};

  const Result<std::unique_ptr<frame::hal::Stream>> stream_result =
      backend_->create_stream(device_);
  ASSERT_TRUE(stream_result.is_ok());

  const Status run_status = executable->run(inputs, outputs, *stream_result.value());
  ASSERT_TRUE(run_status.is_ok()) << run_status.message();

  // eager 三连 launch 参照:matmul -> add -> relu,与编译执行共用同一批 cpu
  // kernel 实现(KernelRegistry::find(op, "cpu")),数值理应一致。
  std::vector<Tensor> matmul_inputs{x, w};
  std::vector<Tensor> matmul_outputs{
      Tensor::empty(Shape({2, 4}), DType::of<float>(), device_, *allocator_).value()};
  ASSERT_TRUE(EagerLaunch("matmul", matmul_inputs, matmul_outputs, *stream_result.value()).is_ok());

  std::vector<Tensor> add_inputs{matmul_outputs[0], bias};
  std::vector<Tensor> add_outputs{
      Tensor::empty(Shape({2, 4}), DType::of<float>(), device_, *allocator_).value()};
  ASSERT_TRUE(EagerLaunch("add", add_inputs, add_outputs, *stream_result.value()).is_ok());

  std::vector<Tensor> relu_inputs{add_outputs[0]};
  std::vector<Tensor> relu_outputs{
      Tensor::empty(Shape({2, 4}), DType::of<float>(), device_, *allocator_).value()};
  ASSERT_TRUE(EagerLaunch("relu", relu_inputs, relu_outputs, *stream_result.value()).is_ok());

  EXPECT_TRUE(
      tensor_all_close(outputs[0], relu_outputs[0], default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(RuntimeCompileTest, EndToEndMatmulAddReluMatchesEagerThreeLaunchesFloat16) {
  // 同一图形态(matmul->add->relu)的 fp16 版本:编译执行与 eager 三连 launch
  // 调用的是完全相同的一批 cpu kernel 实现,数值应一致,容差用 fp16 档
  // (BUILD-011)。
  const Graph graph = BuildMatmulAddReluGraph(DType::of<float16_t>());

  const Result<std::shared_ptr<Executable>> executable_result =
      frame::runtime::compile(graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable_result.is_ok()) << executable_result.status().message();
  const std::shared_ptr<Executable>& executable = executable_result.value();

  Tensor x = MakeFilledFloat16Tensor(Shape({2, 3}), 1.0F, device_, *allocator_);
  Tensor w = MakeFilledFloat16Tensor(Shape({3, 4}), -0.5F, device_, *allocator_);
  Tensor bias = MakeFilledFloat16Tensor(Shape({2, 4}), 0.25F, device_, *allocator_);

  std::vector<Tensor> inputs{x, w, bias};
  std::vector<Tensor> outputs{
      Tensor::empty(Shape({2, 4}), DType::of<float16_t>(), device_, *allocator_).value()};

  const Result<std::unique_ptr<frame::hal::Stream>> stream_result =
      backend_->create_stream(device_);
  ASSERT_TRUE(stream_result.is_ok());

  const Status run_status = executable->run(inputs, outputs, *stream_result.value());
  ASSERT_TRUE(run_status.is_ok()) << run_status.message();

  std::vector<Tensor> matmul_inputs{x, w};
  std::vector<Tensor> matmul_outputs{
      Tensor::empty(Shape({2, 4}), DType::of<float16_t>(), device_, *allocator_).value()};
  ASSERT_TRUE(EagerLaunch("matmul", matmul_inputs, matmul_outputs, *stream_result.value()).is_ok());

  std::vector<Tensor> add_inputs{matmul_outputs[0], bias};
  std::vector<Tensor> add_outputs{
      Tensor::empty(Shape({2, 4}), DType::of<float16_t>(), device_, *allocator_).value()};
  ASSERT_TRUE(EagerLaunch("add", add_inputs, add_outputs, *stream_result.value()).is_ok());

  std::vector<Tensor> relu_inputs{add_outputs[0]};
  std::vector<Tensor> relu_outputs{
      Tensor::empty(Shape({2, 4}), DType::of<float16_t>(), device_, *allocator_).value()};
  ASSERT_TRUE(EagerLaunch("relu", relu_inputs, relu_outputs, *stream_result.value()).is_ok());

  EXPECT_TRUE(
      tensor_all_close(outputs[0], relu_outputs[0], default_tolerance(DTypeCode::kFloat16)));
}

TEST_F(RuntimeCompileTest, RunRejectsWrongInputCount) {
  const Graph graph = BuildMatmulAddReluGraph();
  const Result<std::shared_ptr<Executable>> executable_result =
      frame::runtime::compile(graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable_result.is_ok()) << executable_result.status().message();

  Tensor x = MakeFilledTensor(Shape({2, 3}), 1.0F, device_, *allocator_);
  Tensor w = MakeFilledTensor(Shape({3, 4}), -0.5F, device_, *allocator_);
  std::vector<Tensor> inputs{x, w};  // 缺 bias:输入个数不足
  std::vector<Tensor> outputs{
      Tensor::empty(Shape({2, 4}), DType::of<float>(), device_, *allocator_).value()};

  const Result<std::unique_ptr<frame::hal::Stream>> stream_result =
      backend_->create_stream(device_);
  ASSERT_TRUE(stream_result.is_ok());

  const Status status = executable_result.value()->run(inputs, outputs, *stream_result.value());
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
}

TEST_F(RuntimeCompileTest, RunRejectsWrongInputShape) {
  const Graph graph = BuildMatmulAddReluGraph();
  const Result<std::shared_ptr<Executable>> executable_result =
      frame::runtime::compile(graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable_result.is_ok()) << executable_result.status().message();

  // x 应为 [2,3],此处故意传 [2,4]。
  Tensor x = MakeFilledTensor(Shape({2, 4}), 1.0F, device_, *allocator_);
  Tensor w = MakeFilledTensor(Shape({3, 4}), -0.5F, device_, *allocator_);
  Tensor bias = MakeFilledTensor(Shape({2, 4}), 0.25F, device_, *allocator_);
  std::vector<Tensor> inputs{x, w, bias};
  std::vector<Tensor> outputs{
      Tensor::empty(Shape({2, 4}), DType::of<float>(), device_, *allocator_).value()};

  const Result<std::unique_ptr<frame::hal::Stream>> stream_result =
      backend_->create_stream(device_);
  ASSERT_TRUE(stream_result.is_ok());

  const Status status = executable_result.value()->run(inputs, outputs, *stream_result.value());
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
}

TEST_F(RuntimeCompileTest, RunRejectsWrongOutputDtype) {
  const Graph graph = BuildMatmulAddReluGraph();
  const Result<std::shared_ptr<Executable>> executable_result =
      frame::runtime::compile(graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable_result.is_ok()) << executable_result.status().message();

  Tensor x = MakeFilledTensor(Shape({2, 3}), 1.0F, device_, *allocator_);
  Tensor w = MakeFilledTensor(Shape({3, 4}), -0.5F, device_, *allocator_);
  Tensor bias = MakeFilledTensor(Shape({2, 4}), 0.25F, device_, *allocator_);
  std::vector<Tensor> inputs{x, w, bias};
  // 输出应为 float32,此处故意传 int32。
  std::vector<Tensor> outputs{
      Tensor::empty(Shape({2, 4}), DType::of<std::int32_t>(), device_, *allocator_).value()};

  const Result<std::unique_ptr<frame::hal::Stream>> stream_result =
      backend_->create_stream(device_);
  ASSERT_TRUE(stream_result.is_ok());

  const Status status = executable_result.value()->run(inputs, outputs, *stream_result.value());
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
}

// 恒等/仅输入图(m7-design-brief 决议点 3 修订节 5-⑥"裁定仅输入图/恒等图为
// 支持场景"):零算子节点,图输出直接 mark_output 某个图输入,CpuExecutable
// 的 slot 映射机制天然覆盖此情形(输出 slot 即输入 slot),run() 应把输入
// 原样拷贝进输出。
TEST_F(RuntimeCompileTest, IdentityGraphRunCopiesInputToOutput) {
  Graph graph("identity");
  Value* x = graph.add_graph_input(MakeFloat32Type({4})).value();
  ASSERT_TRUE(graph.mark_output(x).is_ok());

  const Result<std::shared_ptr<Executable>> executable_result =
      frame::runtime::compile(graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable_result.is_ok()) << executable_result.status().message();

  Tensor x_tensor = MakeFilledTensor(Shape({4}), 3.0F, device_, *allocator_);
  std::vector<Tensor> inputs{x_tensor};
  std::vector<Tensor> outputs{
      Tensor::empty(Shape({4}), DType::of<float>(), device_, *allocator_).value()};

  const Result<std::unique_ptr<frame::hal::Stream>> stream_result =
      backend_->create_stream(device_);
  ASSERT_TRUE(stream_result.is_ok());

  const Status run_status = executable_result.value()->run(inputs, outputs, *stream_result.value());
  ASSERT_TRUE(run_status.is_ok()) << run_status.message();

  EXPECT_TRUE(tensor_all_close(outputs[0], x_tensor, default_tolerance(DTypeCode::kFloat32)));
}

// 回归(批1-Task4b):run_with_allocated_outputs 产出张量的 Device::backend
// 视图必须别名注册表持有的稳定后端名(Backend::name(),注册表为进程级单例,
// 名字存储与进程同寿),而不是调用方传入的 backend_name 缓冲——否则调用方
// 缓冲(如 Python 绑定包装对象的成员字符串)销毁后视图悬垂(use-after-free,
// 半精度审计附带发现)。以指针同一性断言判定别名对象,不读悬垂内存(无 UB)。
TEST_F(RuntimeCompileTest, AllocatedOutputDeviceBackendAliasesRegistryOwnedName) {
  Graph graph("identity_for_device_lifetime");
  Value* x = graph.add_graph_input(MakeFloat32Type({4})).value();
  ASSERT_TRUE(graph.mark_output(x).is_ok());

  const Result<std::shared_ptr<Executable>> executable_result =
      frame::runtime::compile(graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable_result.is_ok()) << executable_result.status().message();

  // 刻意用短命堆上字符串作为 backend_name 实参(模拟绑定层包装对象成员)。
  const std::string transient_name(frame::kCpuBackendName);
  Tensor x_tensor = MakeFilledTensor(Shape({4}), 3.0F, device_, *allocator_);
  const std::vector<Tensor> inputs{x_tensor};
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable_result.value(), transient_name, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  ASSERT_EQ(outputs.value().size(), 1U);

  const std::string_view output_backend = outputs.value()[0].device().backend;
  EXPECT_NE(static_cast<const void*>(output_backend.data()),
            static_cast<const void*>(transient_name.data()));
  EXPECT_EQ(static_cast<const void*>(output_backend.data()),
            static_cast<const void*>(backend_->name().data()));
}

// 多图输出边角:matmul 的中间结果与 relu 的最终结果各自 mark_output,验证
// CpuExecutable 的 output_slots_/output_signature_ 在图输出个数 >1 时逐个
// 正确处理(此前用例均只覆盖单一图输出场景)。
TEST_F(RuntimeCompileTest, MultipleGraphOutputsAreEachCopiedCorrectly) {
  Graph graph("matmul_relu_multi_output");
  Value* x = graph.add_graph_input(MakeFloat32Type({2, 3})).value();
  Value* w = graph.add_graph_input(MakeFloat32Type({3, 4})).value();
  Node* matmul_node = graph.create_node("matmul", {x, w}, {MakeFloat32Type({2, 4})}).value();
  Node* relu_node =
      graph.create_node("relu", {matmul_node->output(0)}, {MakeFloat32Type({2, 4})}).value();
  ASSERT_TRUE(graph.mark_output(matmul_node->output(0)).is_ok());
  ASSERT_TRUE(graph.mark_output(relu_node->output(0)).is_ok());

  const Result<std::shared_ptr<Executable>> executable_result =
      frame::runtime::compile(graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable_result.is_ok()) << executable_result.status().message();

  Tensor x_tensor = MakeFilledTensor(Shape({2, 3}), 1.0F, device_, *allocator_);
  Tensor w_tensor = MakeFilledTensor(Shape({3, 4}), -0.5F, device_, *allocator_);
  std::vector<Tensor> inputs{x_tensor, w_tensor};
  std::vector<Tensor> outputs{
      Tensor::empty(Shape({2, 4}), DType::of<float>(), device_, *allocator_).value(),
      Tensor::empty(Shape({2, 4}), DType::of<float>(), device_, *allocator_).value()};

  const Result<std::unique_ptr<frame::hal::Stream>> stream_result =
      backend_->create_stream(device_);
  ASSERT_TRUE(stream_result.is_ok());

  const Status run_status = executable_result.value()->run(inputs, outputs, *stream_result.value());
  ASSERT_TRUE(run_status.is_ok()) << run_status.message();

  // eager 参照:matmul 单独一步的输出即第一个图输出的期望值;relu(matmul 输出)
  // 即第二个图输出的期望值。
  std::vector<Tensor> matmul_inputs{x_tensor, w_tensor};
  std::vector<Tensor> matmul_outputs{
      Tensor::empty(Shape({2, 4}), DType::of<float>(), device_, *allocator_).value()};
  ASSERT_TRUE(EagerLaunch("matmul", matmul_inputs, matmul_outputs, *stream_result.value()).is_ok());

  std::vector<Tensor> relu_inputs{matmul_outputs[0]};
  std::vector<Tensor> relu_outputs{
      Tensor::empty(Shape({2, 4}), DType::of<float>(), device_, *allocator_).value()};
  ASSERT_TRUE(EagerLaunch("relu", relu_inputs, relu_outputs, *stream_result.value()).is_ok());

  EXPECT_TRUE(
      tensor_all_close(outputs[0], matmul_outputs[0], default_tolerance(DTypeCode::kFloat32)));
  EXPECT_TRUE(
      tensor_all_close(outputs[1], relu_outputs[0], default_tolerance(DTypeCode::kFloat32)));
}

// sum(带必填 axes 属性)入编译路径,且刻意在原始 Graph 析构后才调用
// Executable::run——验证 CpuExecutable::Step::attrs 是编译期自持的值拷贝
// (m7-design-brief 决议点 3),运行期不回指已销毁的 Graph/Node。
TEST_F(RuntimeCompileTest, SumWithAttrsExecutesCorrectlyAfterOriginalGraphIsDestroyed) {
  std::shared_ptr<Executable> executable;
  {
    Graph graph("sum_with_attrs");
    Value* x = graph.add_graph_input(MakeFloat32Type({2, 3})).value();
    Node* sum_node = graph.create_node("sum", {x}, {MakeFloat32Type({3})}).value();
    sum_node->set_attr("axes", frame::ir::AttrValue{std::vector<int64_t>{0}});
    ASSERT_TRUE(graph.mark_output(sum_node->output(0)).is_ok());

    const Result<std::shared_ptr<Executable>> executable_result =
        frame::runtime::compile(graph, frame::kCpuBackendName, CompileOptions{});
    ASSERT_TRUE(executable_result.is_ok()) << executable_result.status().message();
    executable = executable_result.value();
    // graph(含 sum_node 及其 "axes" 属性)在此作用域结束时析构。
  }

  // x = [[1.0,1.5,2.0],[2.5,3.0,3.5]](MakeFilledTensor(start=1.0, step=0.5)),
  // axis=0 归约按列求和:[1.0+2.5, 1.5+3.0, 2.0+3.5] = [3.5, 4.5, 5.5]
  // (独立手算,未复用被测 sum kernel 的实现)。
  Tensor x_tensor = MakeFilledTensor(Shape({2, 3}), 1.0F, device_, *allocator_);
  std::vector<Tensor> inputs{x_tensor};
  std::vector<Tensor> outputs{
      Tensor::empty(Shape({3}), DType::of<float>(), device_, *allocator_).value()};

  const Result<std::unique_ptr<frame::hal::Stream>> stream_result =
      backend_->create_stream(device_);
  ASSERT_TRUE(stream_result.is_ok());

  const Status run_status = executable->run(inputs, outputs, *stream_result.value());
  ASSERT_TRUE(run_status.is_ok()) << run_status.message();

  Tensor expected = Tensor::empty(Shape({3}), DType::of<float>(), device_, *allocator_).value();
  float* expected_data = expected.data<float>();
  expected_data[0] = 3.5F;
  expected_data[1] = 4.5F;
  expected_data[2] = 5.5F;

  EXPECT_TRUE(tensor_all_close(outputs[0], expected, default_tolerance(DTypeCode::kFloat32)));
}

TEST(RuntimeCompileDumpTest, DumpIrAfterObservableOnEndToEndGraphViaDirectPassManagerComposition) {
  // runtime::compile 窄接口不带 dump 参数(compile.h 头注释"窄接口取舍"
  // 一节);--dump-ir-after 的可观测性经直接组合 PassManager 验证(机制本身
  // M6 已测,这里只在本里程碑的端到端图形态上补一条冒烟)。dump 目标固定选
  // 标准管线的末位 pass "backend_lowering"(M9 起 layout_assignment/
  // operator_fusion 均真实改图——本图的 add->relu 满足链条件会被融合,见
  // src/compiler/passes/operator_fusion.cpp——故若 dump 目标选在中途 pass,
  // 其后仍有 pass 会继续改图,dump 快照与最终 dump_text(graph) 不再相等;
  // backend_lowering 是纯校验 pass 不改图,取其"运行后、verify 前"快照即
  // 等价于管线运行结束后的最终图文本)。
  Graph graph = BuildMatmulAddReluGraph();

  Result<frame::compiler::PassManager> pipeline_result =
      frame::compiler::standard_pipeline(frame::kCpuBackendName);
  ASSERT_TRUE(pipeline_result.is_ok()) << pipeline_result.status().message();
  frame::compiler::PassManager& pipeline = pipeline_result.value();

  std::ostringstream dump;
  pipeline.set_dump_ir_after("backend_lowering", dump);

  const Status status = pipeline.run(graph);
  ASSERT_TRUE(status.is_ok()) << status.message();

  EXPECT_EQ(dump.str(), frame::ir::dump_text(graph));
  EXPECT_NE(dump.str(), "");
}

}  // namespace
