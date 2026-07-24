// ONNX 权重交换单测(ADR-0013,docs/decisions/0013-onnx-weight-exchange-minimal-codec.md)。
//   1. 字节级往返:三 dtype(float32/float16/bfloat16)张量 save→load,
//      name/dims/dtype/位模式逐项相等(循 M12 位级往返先例,EXPECT_EQ bits)。
//   2. 错误路径:文件不存在/截断文件/伪造未知字段(wire 0/1/2/5)被跳过仍读出
//      目标张量/groups(wire 3)拒绝/子集外 data_type(INT64=7)拒绝含枚举值/
//      save 非 cpu 张量拒绝/save 白名单外 dtype 拒绝。
//   3. train-save-import-infer 链路(ADR 判定②):复用
//      tests/cpp/compiler/mlp_forward_graph_helper.h 的 MLP 训练前向图(与
//      test_training_loop.cpp 训练闭环用例同构,REUSE-002),训练若干步后导出
//      {w1,b1,w2}、重建同构推理专用前向图并加载权重、编译执行,与「训练末态
//      参数直接推理」逐元素一致(BUILD-011)。
//
// interop 本身仅依赖 core(include/frame/interop/onnx_weights.h 头注释);
// 本测试目标链接 frame::frame 聚合库,故在 hal 依赖面内,经 BackendRegistry
// 取得真实 cpu 分配器后按维护者裁决(方案 b)注入 load_onnx_weights。

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <frame/compiler/autograd.h>
#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/executable.h>
#include <frame/interop/onnx_weights.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/graph_builder.h>
#include <frame/ops/op_registry.h>
#include <frame/runtime/compile.h>

#include "../common/tolerance.h"
#include "../compiler/mlp_forward_graph_helper.h"
#include "../ops/elementwise_op_test_helpers.h"

namespace {

using frame::bfloat16_t;
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
using frame::interop::load_onnx_weights;
using frame::interop::NamedTensor;
using frame::interop::save_onnx_weights;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::Value;
using frame::ops::create_node_with_inferred_types;
using frame::ops::testing::MakeType;
using frame::testing::default_tolerance;
using frame::testing::tensor_all_close;

// 推理专用 MLP 前向图(4 输入:x,w1,b1,w2;无 target/mse_loss,输出为
// matmul2 预测值)。与 mlp_forward_graph_helper.h::BuildMlpForwardGraph
// (训练用,5 输入含 target/loss)共享的重合部分(matmul1/add1/relu1/matmul2
// 四条节点构造)不足 20 行,且用途不同(推理 vs 训练),不构成需提取共享的
// ≥20 行同构重复,故保留在本文件内、不进一步提取。
Graph BuildMlpInferenceGraph() {
  using frame::compiler::testing::kMlpBatchSize;
  using frame::compiler::testing::kMlpHiddenDim;
  using frame::compiler::testing::kMlpInputDim;
  using frame::compiler::testing::kMlpOutputDim;

  Graph graph("mlp_inference");
  Value* x =
      graph.add_graph_input(MakeType(DType::of<float>(), {kMlpBatchSize, kMlpInputDim})).value();
  Value* w1 =
      graph.add_graph_input(MakeType(DType::of<float>(), {kMlpInputDim, kMlpHiddenDim})).value();
  Value* b1 =
      graph.add_graph_input(MakeType(DType::of<float>(), {kMlpBatchSize, kMlpHiddenDim})).value();
  Value* w2 =
      graph.add_graph_input(MakeType(DType::of<float>(), {kMlpHiddenDim, kMlpOutputDim})).value();

  Node* matmul1 = create_node_with_inferred_types(graph, "matmul", {x, w1}).value();
  Node* add1 = create_node_with_inferred_types(graph, "add", {matmul1->output(0), b1}).value();
  Node* relu1 = create_node_with_inferred_types(graph, "relu", {add1->output(0)}).value();
  Node* matmul2 = create_node_with_inferred_types(graph, "matmul", {relu1->output(0), w2}).value();
  const Status mark_status = graph.mark_output(matmul2->output(0));
  EXPECT_TRUE(mark_status.is_ok());
  return graph;
}

// 手写最小 varint/tag 编码(测试侧独立实现,不复用被测编解码器
// src/interop/onnx_weights.cpp 的内部函数——那些是文件内匿名命名空间符号,
// 本就无法跨翻译单元访问;这里独立重写是为了以"黑盒伪造字节流"的方式验证
// 解码器的健壮性,而非验证编码器自身)。仅供下方跳过/拒绝类错误路径用例
// 拼接测试夹具。
void AppendTestVarint(std::string& out, uint64_t value) {
  while (value >= 0x80) {
    out.push_back(static_cast<char>((value & 0x7f) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<char>(value));
}

void AppendTestTag(std::string& out, int field_number, int wire_type) {
  AppendTestVarint(out,
                   (static_cast<uint64_t>(field_number) << 3) | static_cast<uint64_t>(wire_type));
}

void AppendTestLengthDelimited(std::string& out, int field_number, std::string_view payload) {
  AppendTestTag(out, field_number, /*wire_type=*/2);
  AppendTestVarint(out, payload.size());
  out.append(payload);
}

class OnnxWeightsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const Result<Backend*> result = BackendRegistry::instance().get(frame::kCpuBackendName);
    ASSERT_TRUE(result.is_ok());
    backend_ = result.value();
    device_ = frame::cpu_device();
    allocator_ = backend_->allocator(device_);
    ASSERT_NE(allocator_, nullptr);
  }

  template <typename T>
  Tensor MakeTensorFromValues(const std::vector<T>& values, const Shape& shape) {
    Tensor tensor = Tensor::empty(shape, DType::of<T>(), device_, *allocator_).value();
    T* data = tensor.data<T>();
    for (size_t i = 0; i < values.size(); ++i) data[i] = values[i];
    return tensor;
  }

  // 仓库临时目录下的唯一文件路径(按当前 test suite/name/suffix 拼接,避免
  // 同进程内多个用例互相覆盖;不做清理,与本仓其余测试的既有风格一致——无
  // 先例对临时文件做显式清理)。
  static std::string TempFilePath(std::string_view suffix) {
    const ::testing::TestInfo* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    const std::filesystem::path dir = std::filesystem::temp_directory_path();
    const std::filesystem::path file =
        dir / ("frame_onnx_weights_test_" + std::string(test_info->test_suite_name()) + "_" +
               std::string(test_info->name()) + "_" + std::string(suffix) + ".onnx");
    return file.string();
  }

  static void WriteFile(const std::string& path, std::string_view contents) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  }

  static std::string ReadFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
  }

  Backend* backend_ = nullptr;
  frame::Device device_{};
  frame::hal::Allocator* allocator_ = nullptr;
};

// ---------------------------------------------------------------------------
// 1. 字节级往返。
// ---------------------------------------------------------------------------

TEST_F(OnnxWeightsTest, ByteLevelRoundTripAcrossThreeDtypes) {
  const std::vector<float> fp32_values{1.5F, -2.25F, 0.0F, 3.125F, -100.0F, 0.001F};
  const Tensor fp32_tensor = MakeTensorFromValues<float>(fp32_values, Shape({2, 3}));

  const std::vector<float16_t> fp16_values{
      float16_t{0x3c00u},  // 1.0
      float16_t{0xbc00u},  // -1.0
      float16_t{0x7c00u},  // +inf
      float16_t{0xfc00u},  // -inf
      float16_t{0x0000u},  // +0
      float16_t{0x8000u},  // -0
  };
  const Tensor fp16_tensor = MakeTensorFromValues<float16_t>(fp16_values, Shape({3, 2}));

  const std::vector<bfloat16_t> bf16_values{
      bfloat16_t{0x3f80u},  // 1.0
      bfloat16_t{0xbf80u},  // -1.0
      bfloat16_t{0x7f80u},  // +inf
      bfloat16_t{0xff80u},  // -inf
  };
  const Tensor bf16_tensor = MakeTensorFromValues<bfloat16_t>(bf16_values, Shape({2, 2}));

  std::vector<NamedTensor> weights;
  weights.push_back(NamedTensor{"w_fp32", fp32_tensor});
  weights.push_back(NamedTensor{"w_fp16", fp16_tensor});
  weights.push_back(NamedTensor{"w_bf16", bf16_tensor});

  const std::string path = TempFilePath("round_trip");
  const Status save_status = save_onnx_weights(path, weights);
  ASSERT_TRUE(save_status.is_ok()) << save_status.message();

  Result<std::vector<NamedTensor>> loaded = load_onnx_weights(path, *allocator_);
  ASSERT_TRUE(loaded.is_ok()) << loaded.status().message();
  ASSERT_EQ(loaded.value().size(), 3u);

  NamedTensor& loaded_fp32 = loaded.value()[0];
  EXPECT_EQ(loaded_fp32.name, "w_fp32");
  EXPECT_EQ(loaded_fp32.tensor.shape().dims(), (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(loaded_fp32.tensor.dtype().code(), DTypeCode::kFloat32);
  for (int64_t i = 0; i < loaded_fp32.tensor.numel(); ++i) {
    EXPECT_EQ(loaded_fp32.tensor.data<float>()[i], fp32_values[static_cast<size_t>(i)])
        << "index " << i;
  }

  NamedTensor& loaded_fp16 = loaded.value()[1];
  EXPECT_EQ(loaded_fp16.name, "w_fp16");
  EXPECT_EQ(loaded_fp16.tensor.shape().dims(), (std::vector<int64_t>{3, 2}));
  EXPECT_EQ(loaded_fp16.tensor.dtype().code(), DTypeCode::kFloat16);
  for (int64_t i = 0; i < loaded_fp16.tensor.numel(); ++i) {
    EXPECT_EQ(loaded_fp16.tensor.data<float16_t>()[i].bits,
              fp16_values[static_cast<size_t>(i)].bits)
        << "index " << i;
  }

  NamedTensor& loaded_bf16 = loaded.value()[2];
  EXPECT_EQ(loaded_bf16.name, "w_bf16");
  EXPECT_EQ(loaded_bf16.tensor.shape().dims(), (std::vector<int64_t>{2, 2}));
  EXPECT_EQ(loaded_bf16.tensor.dtype().code(), DTypeCode::kBFloat16);
  for (int64_t i = 0; i < loaded_bf16.tensor.numel(); ++i) {
    EXPECT_EQ(loaded_bf16.tensor.data<bfloat16_t>()[i].bits,
              bf16_values[static_cast<size_t>(i)].bits)
        << "index " << i;
  }
}

// ---------------------------------------------------------------------------
// 2. 错误路径。
// ---------------------------------------------------------------------------

TEST_F(OnnxWeightsTest, LoadReturnsNotFoundForMissingFile) {
  const std::string path = TempFilePath("definitely_absent");
  const Result<std::vector<NamedTensor>> loaded = load_onnx_weights(path, *allocator_);
  ASSERT_FALSE(loaded.is_ok());
  EXPECT_EQ(loaded.status().code(), ErrorCode::kNotFound);
}

TEST_F(OnnxWeightsTest, LoadRejectsTruncatedFile) {
  std::vector<NamedTensor> weights;
  weights.push_back(
      NamedTensor{"w", MakeTensorFromValues<float>({1.0F, 2.0F, 3.0F, 4.0F}, Shape({2, 2}))});
  const std::string path = TempFilePath("truncated");
  ASSERT_TRUE(save_onnx_weights(path, weights).is_ok());

  const std::string contents = ReadFile(path);
  ASSERT_GT(contents.size(), 30u);
  WriteFile(path, std::string_view(contents).substr(0, 10));

  const Result<std::vector<NamedTensor>> loaded = load_onnx_weights(path, *allocator_);
  ASSERT_FALSE(loaded.is_ok());
  EXPECT_EQ(loaded.status().code(), ErrorCode::kInvalidArgument);
}

TEST_F(OnnxWeightsTest, LoadSkipsUnknownFieldsOfEachSupportedWireType) {
  std::vector<NamedTensor> weights;
  weights.push_back(NamedTensor{"w", MakeTensorFromValues<float>({1.0F, 2.0F}, Shape({2}))});
  const std::string path = TempFilePath("skip_unknown");
  ASSERT_TRUE(save_onnx_weights(path, weights).is_ok());

  // 在合法文件末尾追加 4 个伪造的顶层未知字段(field number 99/98/97/96),
  // 依次覆盖 wire 0(varint)/1(64 位定长)/5(32 位定长)/2(length-delimited)。
  std::string contents = ReadFile(path);
  AppendTestTag(contents, 99, /*wire_type=*/0);
  AppendTestVarint(contents, 123456);
  AppendTestTag(contents, 98, /*wire_type=*/1);
  contents.append(8, '\xAB');
  AppendTestTag(contents, 97, /*wire_type=*/5);
  contents.append(4, '\xCD');
  AppendTestLengthDelimited(contents, 96, "xyz");
  WriteFile(path, contents);

  const Result<std::vector<NamedTensor>> loaded = load_onnx_weights(path, *allocator_);
  ASSERT_TRUE(loaded.is_ok()) << loaded.status().message();
  ASSERT_EQ(loaded.value().size(), 1u);
  EXPECT_EQ(loaded.value()[0].name, "w");
}

TEST_F(OnnxWeightsTest, LoadRejectsDeprecatedGroupWireType) {
  std::vector<NamedTensor> weights;
  weights.push_back(NamedTensor{"w", MakeTensorFromValues<float>({1.0F}, Shape({1}))});
  const std::string path = TempFilePath("group_wire_type");
  ASSERT_TRUE(save_onnx_weights(path, weights).is_ok());

  std::string contents = ReadFile(path);
  AppendTestTag(contents, 50, /*wire_type=*/3);  // 已废弃 group(wire 3)
  WriteFile(path, contents);

  const Result<std::vector<NamedTensor>> loaded = load_onnx_weights(path, *allocator_);
  ASSERT_FALSE(loaded.is_ok());
  EXPECT_EQ(loaded.status().code(), ErrorCode::kUnimplemented);
}

TEST_F(OnnxWeightsTest, LoadRejectsDataTypeOutsideWhitelist) {
  // 手写最小 ModelProto:仅含 graph.initializer[0].data_type = 7(INT64,
  // 子集外),不依赖 save_onnx_weights(纯解码器黑盒测试)。
  std::string tensor_message;
  AppendTestTag(tensor_message, /*field_number=*/2, /*wire_type=*/0);  // TensorProto.data_type
  AppendTestVarint(tensor_message, 7);                                 // INT64 = 7

  std::string graph_message;
  AppendTestLengthDelimited(graph_message, /*field_number=*/5,
                            tensor_message);  // GraphProto.initializer

  std::string model_message;
  AppendTestLengthDelimited(model_message, /*field_number=*/7, graph_message);  // ModelProto.graph

  const std::string path = TempFilePath("bad_data_type");
  WriteFile(path, model_message);

  const Result<std::vector<NamedTensor>> loaded = load_onnx_weights(path, *allocator_);
  ASSERT_FALSE(loaded.is_ok());
  EXPECT_EQ(loaded.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(loaded.status().message().find('7'), std::string_view::npos);
}

TEST_F(OnnxWeightsTest, SaveRejectsNonCpuTensor) {
  // Storage/device 一致性由调用方保证、Storage 本身不校验
  // (include/frame/core/storage.h 头部注释②),故可用真实 cpu allocator 分配
  // 但显式声明非 cpu 的 device,构造出"设备元数据为非 cpu"的张量以驱动本用例
  // (不需要真实注册 cuda 后端)。
  const frame::Device fake_non_cpu_device{"cuda", 0};
  const Result<Tensor> tensor_result =
      Tensor::empty(Shape({2}), DType::of<float>(), fake_non_cpu_device, *allocator_);
  ASSERT_TRUE(tensor_result.is_ok());

  std::vector<NamedTensor> weights;
  weights.push_back(NamedTensor{"w", tensor_result.value()});
  const std::string path = TempFilePath("non_cpu");
  const Status status = save_onnx_weights(path, weights);
  ASSERT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("cpu"), std::string_view::npos);
}

TEST_F(OnnxWeightsTest, SaveRejectsDtypeOutsideWhitelist) {
  const Result<Tensor> tensor_result =
      Tensor::empty(Shape({2}), DType::of<int32_t>(), device_, *allocator_);
  ASSERT_TRUE(tensor_result.is_ok());

  std::vector<NamedTensor> weights;
  weights.push_back(NamedTensor{"w", tensor_result.value()});
  const std::string path = TempFilePath("bad_dtype");
  const Status status = save_onnx_weights(path, weights);
  ASSERT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("int32"), std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 3. train-save-import-infer 链路(ADR-0013 判定②)。
// ---------------------------------------------------------------------------

TEST_F(OnnxWeightsTest, TrainSaveImportInferMatchesDirectInference) {
  using frame::compiler::build_backward_graph;
  using frame::compiler::build_sgd_update_graph;
  using frame::compiler::testing::BuildMlpForwardGraph;
  using frame::compiler::testing::kMlpBatchSize;
  using frame::compiler::testing::kMlpHiddenDim;
  using frame::compiler::testing::kMlpInputDim;
  using frame::compiler::testing::kMlpOutputDim;
  using frame::ir::TensorType;

  // 固定种子(测试可复现性,非 bug)。
  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 rng(20260713U);
  std::uniform_real_distribution<float> data_dist(-1.0F, 1.0F);
  std::uniform_real_distribution<float> small_weight_dist(-0.1F, 0.1F);
  std::uniform_real_distribution<float> bias_dist(0.5F, 1.0F);

  std::vector<float> x_values(static_cast<size_t>(kMlpBatchSize * kMlpInputDim));
  for (float& v : x_values) v = data_dist(rng);
  std::vector<float> target_values(static_cast<size_t>(kMlpBatchSize * kMlpOutputDim));
  for (float& v : target_values) v = data_dist(rng);
  std::vector<float> w1_values(static_cast<size_t>(kMlpInputDim * kMlpHiddenDim));
  for (float& v : w1_values) v = small_weight_dist(rng);
  std::vector<float> b1_values(static_cast<size_t>(kMlpBatchSize * kMlpHiddenDim));
  for (float& v : b1_values) v = bias_dist(rng);
  std::vector<float> w2_values(static_cast<size_t>(kMlpHiddenDim * kMlpOutputDim));
  for (float& v : w2_values) v = small_weight_dist(rng);

  Tensor x = MakeTensorFromValues<float>(x_values, Shape({kMlpBatchSize, kMlpInputDim}));
  Tensor target = MakeTensorFromValues<float>(target_values, Shape({kMlpBatchSize, kMlpOutputDim}));
  Tensor w1 = MakeTensorFromValues<float>(w1_values, Shape({kMlpInputDim, kMlpHiddenDim}));
  Tensor b1 = MakeTensorFromValues<float>(b1_values, Shape({kMlpBatchSize, kMlpHiddenDim}));
  Tensor w2 = MakeTensorFromValues<float>(w2_values, Shape({kMlpHiddenDim, kMlpOutputDim}));

  // 训练图(build_backward_graph,wrt=w1,b1,w2)与更新图各编译一次(循环外,
  // 与 test_training_loop.cpp 训练闭环用例同构手法)。
  const Graph forward = BuildMlpForwardGraph();
  const std::vector<int32_t> wrt{1, 2, 3};
  const Result<Graph> training = build_backward_graph(forward, 0, wrt);
  ASSERT_TRUE(training.is_ok()) << training.status().message();
  const frame::ir::OpQuery train_query = frame::ops::make_op_query();
  ASSERT_TRUE(training.value().verify(train_query).is_ok());
  const Result<std::shared_ptr<Executable>> train_executable =
      frame::runtime::compile(training.value(), frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(train_executable.is_ok()) << train_executable.status().message();

  const std::vector<TensorType> param_types{
      MakeType(DType::of<float>(), {kMlpInputDim, kMlpHiddenDim}),
      MakeType(DType::of<float>(), {kMlpBatchSize, kMlpHiddenDim}),
      MakeType(DType::of<float>(), {kMlpHiddenDim, kMlpOutputDim})};
  constexpr double kLearningRate = 0.05;
  const Result<Graph> update = build_sgd_update_graph(param_types, kLearningRate);
  ASSERT_TRUE(update.is_ok()) << update.status().message();
  const frame::ir::OpQuery update_query = frame::ops::make_op_query();
  ASSERT_TRUE(update.value().verify(update_query).is_ok());
  const Result<std::shared_ptr<Executable>> update_executable =
      frame::runtime::compile(update.value(), frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(update_executable.is_ok()) << update_executable.status().message();

  // 训练若干步(不要求收敛,只需权重发生有意义演化——本用例验证的是 save/load
  // 往返保真度,不是收敛质量,BUILD-011 判据不适用于此处的步数取舍)。
  constexpr int kNumSteps = 20;
  for (int step = 0; step < kNumSteps; ++step) {
    std::vector<Tensor> train_inputs{x, w1, b1, w2, target};
    const Result<std::vector<Tensor>> train_outputs = frame::runtime::run_with_allocated_outputs(
        *train_executable.value(), frame::kCpuBackendName, train_inputs);
    ASSERT_TRUE(train_outputs.is_ok())
        << "step " << step << ": " << train_outputs.status().message();
    ASSERT_EQ(train_outputs.value().size(), 4u);

    std::vector<Tensor> update_inputs{
        w1, b1, w2, train_outputs.value()[1], train_outputs.value()[2], train_outputs.value()[3]};
    const Result<std::vector<Tensor>> update_outputs = frame::runtime::run_with_allocated_outputs(
        *update_executable.value(), frame::kCpuBackendName, update_inputs);
    ASSERT_TRUE(update_outputs.is_ok())
        << "step " << step << ": " << update_outputs.status().message();
    ASSERT_EQ(update_outputs.value().size(), 3u);

    w1 = update_outputs.value()[0];
    b1 = update_outputs.value()[1];
    w2 = update_outputs.value()[2];
  }

  // 训练末态 w1/b1/w2 就绪。构造推理专用前向图(4 输入,无 target/loss)并
  // 编译一次,分别喂入「训练末态参数直接推理」与「save→load 往返后的参数」两
  // 组输入,断言输出逐元素一致。
  const Graph inference_graph = BuildMlpInferenceGraph();
  const Result<std::shared_ptr<Executable>> inference_executable =
      frame::runtime::compile(inference_graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(inference_executable.is_ok()) << inference_executable.status().message();

  std::vector<Tensor> direct_inputs{x, w1, b1, w2};
  const Result<std::vector<Tensor>> direct_outputs = frame::runtime::run_with_allocated_outputs(
      *inference_executable.value(), frame::kCpuBackendName, direct_inputs);
  ASSERT_TRUE(direct_outputs.is_ok()) << direct_outputs.status().message();
  ASSERT_EQ(direct_outputs.value().size(), 1u);

  std::vector<NamedTensor> weights;
  weights.push_back(NamedTensor{"w1", w1});
  weights.push_back(NamedTensor{"b1", b1});
  weights.push_back(NamedTensor{"w2", w2});
  const std::string path = TempFilePath("train_save_import_infer");
  const Status save_status = save_onnx_weights(path, weights);
  ASSERT_TRUE(save_status.is_ok()) << save_status.message();

  Result<std::vector<NamedTensor>> loaded = load_onnx_weights(path, *allocator_);
  ASSERT_TRUE(loaded.is_ok()) << loaded.status().message();
  ASSERT_EQ(loaded.value().size(), 3u);
  EXPECT_EQ(loaded.value()[0].name, "w1");
  EXPECT_EQ(loaded.value()[1].name, "b1");
  EXPECT_EQ(loaded.value()[2].name, "w2");

  std::vector<Tensor> loaded_inputs{x, loaded.value()[0].tensor, loaded.value()[1].tensor,
                                    loaded.value()[2].tensor};
  const Result<std::vector<Tensor>> loaded_outputs = frame::runtime::run_with_allocated_outputs(
      *inference_executable.value(), frame::kCpuBackendName, loaded_inputs);
  ASSERT_TRUE(loaded_outputs.is_ok()) << loaded_outputs.status().message();
  ASSERT_EQ(loaded_outputs.value().size(), 1u);

  EXPECT_TRUE(tensor_all_close(direct_outputs.value()[0], loaded_outputs.value()[0],
                               default_tolerance(DTypeCode::kFloat32)))
      << "train-save-import-infer round trip diverged from direct inference";
}

}  // namespace
