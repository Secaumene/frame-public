// M23 批5 T6:FNO 前向数值对照(验收硬门 5,docs/plan/
// 2026-07-21-batch5-m23-fft.md §1.5/§3)+ Fourier Mamba 频域支线
// (FourierFilter1d)前向数值对照。四组用例:
//   1. 数值回归锚点("golden"):固定种子确定性初始化 x/参数 -> 经
//      runtime::compile("cpu") 真实执行一次记录输出值,后续按 BUILD-011
//      fp32 容差(tests/cpp/common/tolerance.h)逐元素比对——浮点值无法做
//      字符串逐字节比较,取代之以 tensor_all_close;结构性的"逐字节"由随附
//      的 IR dump_text golden 承担(同 M21/M22 网络 golden 先例,复用
//      convergence_test_helpers.h::AssertGraphMatchesGolden)。
//   2. IR 前向图 dump_text golden(构图确定性,逐字节比对)。
//   3. Fno1dBlock 额外验证 CPU/CUDA 前向一致:整图 device 参数化为 cuda(x
//      与全部参数 graph input 均声明 device=cuda,内部 constant 节点经既有
//      "device 取自 x"路径自动跟随),经 runtime::compile("cuda") 编译执行,
//      容差同 tolerance.h fp32 档;SKIP 口径同
//      tests/cpp/backends/test_cuda_backend.cpp::CudaBackendTest 的既定
//      GTEST_SKIP 手法(BUILD-010/M24:cuda 未注册或本机无设备时 SKIP 并输出
//      英文原因,本机 RTX 5070 Ti 环境下须真实执行、不得 SKIP)。
//   4. FourierFilter1d 额外验证"逐样本参数语义"(设计门建议 3,§1.5):两个
//      batch 行的输入 x 相同、w_re/w_im 不同,输出必不同(证明并非跨样本共享
//      滤波器,勿与常规逐通道共享滤波器混淆)。
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/executable.h>
#include <frame/hal/stream.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/nn/layers.h>
#include <frame/nn/module.h>
#include <frame/runtime/compile.h>

#include "../common/tolerance.h"
#include "convergence_test_helpers.h"

namespace {

using frame::cpu_device;
using frame::Device;
using frame::DType;
using frame::DTypeCode;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::Tensor;
using frame::hal::Allocator;
using frame::hal::Backend;
using frame::hal::BackendRegistry;
using frame::hal::CompileOptions;
using frame::hal::Executable;
using frame::hal::Stream;
using frame::ir::Graph;
using frame::ir::TensorType;
using frame::ir::Value;
using frame::nn::add_parameter_inputs;
using frame::nn::Fno1dBlock;
using frame::nn::FourierFilter1d;
using frame::nn::FourierMamba;
using frame::nn::Mamba;
using frame::nn::Module;
using frame::nn::ParamSpec;
using frame::nn::testing::AssertGraphMatchesGolden;
using frame::nn::testing::MakeTensorFromFloats;
using frame::nn::testing::MakeUniformParamTensors;
using frame::testing::default_tolerance;
using frame::testing::tensor_all_close;

// ---------------------------------------------------------------------------
// 公共形状常量(Fno1dBlock 复用 T4/T5 校准形状:batch=2,in=3,out=4,n=8,
// modes=2;FourierFilter1d 复用 T5 校准形状:batch=2,channels=3,n=8)。
// ---------------------------------------------------------------------------

constexpr int64_t kFnoBatch = 2;
constexpr int64_t kFnoIn = 3;
constexpr int64_t kFnoOut = 4;
constexpr int64_t kFnoN = 8;
constexpr int64_t kFnoModes = 2;

constexpr int64_t kFilterBatch = 2;
constexpr int64_t kFilterChannels = 3;
constexpr int64_t kFilterN = 8;

constexpr int64_t kSsmBatch = 1;
constexpr int64_t kSsmChannels = 2;
constexpr int64_t kSsmSteps = 4;
constexpr int64_t kSsmKernel = 2;

// 任意 device 的 TensorType 构造(REUSE-002:同
// tests/cpp/backends/test_cuda_backend.cpp::MakeDeviceType 先例,独立持一份
// ——本文件与该文件分属不同编译单元,该 helper 是其内部私有静态方法,无法跨
// 文件直接复用)。
TensorType MakeDeviceTensorType(DType dtype, std::vector<int64_t> dims, Device device) {
  TensorType type;
  type.dtype = dtype;
  type.shape = Shape(std::move(dims));
  type.device = device;
  return type;
}

// 按 param_specs 的 shape/dtype 重新声明 device 后逐个 add_graph_input(同
// frame::nn::add_parameter_inputs 骨架,仅 device 字段可覆写——
// src/nn/module.cpp::add_parameter_inputs 对 device 无参数化开口,故本文件
// 独立持一份用于跨 device 的图构造,REUSE-002)。
Result<std::vector<Value*>> AddParameterInputsWithDevice(Graph& graph,
                                                         const std::vector<ParamSpec>& params,
                                                         Device device) {
  std::vector<Value*> values;
  values.reserve(params.size());
  for (const ParamSpec& param : params) {
    TensorType type = param.type;
    type.device = device;
    const Result<Value*> value = graph.add_graph_input(type);
    if (!value.is_ok()) return value.status();
    values.push_back(value.value());
  }
  return values;
}

std::string FloatTensorValues(const Tensor& tensor) {
  const float* data = static_cast<const float*>(tensor.raw_data());
  std::ostringstream stream;
  for (int64_t i = 0; i < tensor.numel(); ++i) {
    if (i != 0) stream << ", ";
    stream << data[i] << 'F';
  }
  return stream.str();
}

// ---------------------------------------------------------------------------
// Fno1dBlock / FourierFilter1d 前向图构造(device 参数化:cpu 数值 golden 与
// cuda 一致性用例共用同一构造函数,仅 device 实参不同)。
// ---------------------------------------------------------------------------

struct FftNetworkForwardBundle {
  Graph graph;
  std::vector<ParamSpec> param_specs;
};

Result<FftNetworkForwardBundle> BuildFnoForwardGraph(Device device) {
  Graph graph("fno1d_forward_numeric");
  const Result<Value*> x_result = graph.add_graph_input(
      MakeDeviceTensorType(DType::of<float>(), {kFnoBatch, kFnoIn, kFnoN}, device));
  if (!x_result.is_ok()) return x_result.status();
  Value* x = x_result.value();

  const Module fno =
      Fno1dBlock("fno", kFnoBatch, kFnoIn, kFnoOut, kFnoN, kFnoModes, DType::of<float>());
  const std::vector<ParamSpec> param_specs = fno.parameters();

  const Result<std::vector<Value*>> param_inputs =
      AddParameterInputsWithDevice(graph, param_specs, device);
  if (!param_inputs.is_ok()) return param_inputs.status();

  const Result<std::vector<Value*>> outputs =
      fno.build(graph, std::vector<Value*>{x}, param_inputs.value());
  if (!outputs.is_ok()) return outputs.status();
  const Status mark_status = graph.mark_output(outputs.value()[0]);
  if (!mark_status.is_ok()) return mark_status;

  FftNetworkForwardBundle bundle;
  bundle.graph = std::move(graph);
  bundle.param_specs = param_specs;
  return bundle;
}

Result<FftNetworkForwardBundle> BuildFourierFilterForwardGraph(Device device) {
  Graph graph("fourier_filter1d_forward_numeric");
  const Result<Value*> x_result = graph.add_graph_input(
      MakeDeviceTensorType(DType::of<float>(), {kFilterBatch, kFilterChannels, kFilterN}, device));
  if (!x_result.is_ok()) return x_result.status();
  Value* x = x_result.value();

  const Module filter =
      FourierFilter1d("ff", kFilterBatch, kFilterChannels, kFilterN, DType::of<float>());
  const std::vector<ParamSpec> param_specs = filter.parameters();

  const Result<std::vector<Value*>> param_inputs =
      AddParameterInputsWithDevice(graph, param_specs, device);
  if (!param_inputs.is_ok()) return param_inputs.status();

  const Result<std::vector<Value*>> outputs =
      filter.build(graph, std::vector<Value*>{x}, param_inputs.value());
  if (!outputs.is_ok()) return outputs.status();
  const Status mark_status = graph.mark_output(outputs.value()[0]);
  if (!mark_status.is_ok()) return mark_status;

  FftNetworkForwardBundle bundle;
  bundle.graph = std::move(graph);
  bundle.param_specs = param_specs;
  return bundle;
}

Result<FftNetworkForwardBundle> BuildSsmForwardGraph(Device device, bool with_fourier) {
  Graph graph(with_fourier ? "fourier_mamba_forward_numeric" : "mamba_forward_numeric");
  const Result<Value*> x_result = graph.add_graph_input(
      MakeDeviceTensorType(DType::of<float>(), {kSsmBatch, kSsmChannels, kSsmSteps}, device));
  if (!x_result.is_ok()) return x_result.status();

  const Module model =
      with_fourier
          ? FourierMamba("fm", kSsmBatch, kSsmChannels, kSsmSteps, kSsmKernel, DType::of<float>())
          : Mamba("m", kSsmBatch, kSsmChannels, kSsmSteps, kSsmKernel, DType::of<float>());
  const std::vector<ParamSpec> param_specs = model.parameters();
  const Result<std::vector<Value*>> param_inputs =
      AddParameterInputsWithDevice(graph, param_specs, device);
  if (!param_inputs.is_ok()) return param_inputs.status();
  const Result<std::vector<Value*>> outputs =
      model.build(graph, std::vector<Value*>{x_result.value()}, param_inputs.value());
  if (!outputs.is_ok()) return outputs.status();
  const Status mark_status = graph.mark_output(outputs.value()[0]);
  if (!mark_status.is_ok()) return mark_status;

  FftNetworkForwardBundle bundle;
  bundle.graph = std::move(graph);
  bundle.param_specs = param_specs;
  return bundle;
}

// ---------------------------------------------------------------------------
// 1a/2a. Fno1dBlock 数值 golden + IR dump_text golden(cpu 单后端)。
// ---------------------------------------------------------------------------

class FftBatchForwardNumericTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const Result<Backend*> backend_result = BackendRegistry::instance().get(frame::kCpuBackendName);
    ASSERT_TRUE(backend_result.is_ok());
    backend_ = backend_result.value();
    device_ = cpu_device();
    allocator_ = backend_->allocator(device_);
    ASSERT_NE(allocator_, nullptr);
  }

  Backend* backend_ = nullptr;
  frame::Device device_{};
  Allocator* allocator_ = nullptr;
};

TEST_F(FftBatchForwardNumericTest, Fno1dBlockForwardMatchesRecordedGolden) {
  Result<FftNetworkForwardBundle> bundle_result = BuildFnoForwardGraph(device_);
  ASSERT_TRUE(bundle_result.is_ok()) << bundle_result.status().message();
  FftNetworkForwardBundle bundle = std::move(bundle_result.value());

  const Result<std::shared_ptr<Executable>> executable =
      frame::runtime::compile(bundle.graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  // 固定种子(mt19937)确定性初始化:先 x 后各参数(与 bundle.param_specs 同
  // 序),同 test_lstm_smoke.cpp/test_wavelet_convergence.cpp 既有抽取次序
  // 纪律。
  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 rng(20260721U);
  std::uniform_real_distribution<float> x_dist(-1.0F, 1.0F);
  std::vector<float> x_values(static_cast<size_t>(kFnoBatch * kFnoIn * kFnoN));
  for (float& v : x_values) v = x_dist(rng);
  const Tensor x_tensor =
      MakeTensorFromFloats(x_values, Shape({kFnoBatch, kFnoIn, kFnoN}), device_, *allocator_);

  const std::vector<Tensor> params =
      MakeUniformParamTensors(bundle.param_specs, rng, -0.2F, 0.2F, device_, *allocator_);

  std::vector<Tensor> inputs{x_tensor};
  for (const Tensor& p : params) inputs.push_back(p);
  const Result<std::vector<Tensor>> run_result = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(run_result.is_ok()) << run_result.status().message();
  ASSERT_EQ(run_result.value().size(), 1u);

  // golden 值 = 本用例首次实测记录(固定种子确定性初始化,可逐位复现,
  // REUSE-002:同 wavelet/cnn/lstm/transformer 网络 golden 先例;浮点数值
  // 无法做字符串逐字节比较,故用 BUILD-011 fp32 容差 tensor_all_close 代替
  // ——结构性"逐字节"由下方 Fno1dBlockForwardGraphMatchesGolden 承担)。
  const std::vector<float> golden_values{
      -0.142254621F, 0.0878191292F,  -0.413874269F, -0.315697998F,   -0.14185971F,
      0.0810541064F, -0.0625197068F, 0.149302706F,  -0.00307452655F, -0.25855726F,
      -0.106026828F, -0.112305768F,  -0.137402371F, -0.160761416F,   -0.122772135F,
      0.0327333249F, 0.155535668F,   0.25585255F,   0.0125275562F,   -0.0223297514F,
      0.076360859F,  0.258753777F,   0.436229527F,  0.108027503F,    -0.201586947F,
      0.198284626F,  -0.0589263178F, -0.145102903F, 0.0444124304F,   0.116750307F,
      -0.121794172F, -0.0518580116F, -0.346798509F, -0.352757961F,   -0.278434038F,
      -0.116704792F, -0.0084060682F, -0.152185887F, -0.101110376F,   0.0575585067F,
      -0.126145199F, -0.0496915467F, -0.136585295F, 0.0344511531F,   -0.18171379F,
      0.0668817833F, 0.0962022543F,  -0.16888006F,  0.125951812F,    -0.122130416F,
      0.128545374F,  -0.0867895111F, 0.405165672F,  0.162741721F,    -0.0166221522F,
      0.23664321F,   -0.115138523F,  0.0196794868F, -0.207598194F,   -0.112164319F,
      0.139643028F,  -0.270038396F,  -0.091733031F, -0.0603340901F};
  const Tensor golden =
      MakeTensorFromFloats(golden_values, Shape({kFnoBatch, kFnoOut, kFnoN}), device_, *allocator_);
  EXPECT_TRUE(
      tensor_all_close(run_result.value()[0], golden, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(FftBatchForwardNumericTest, Fno1dBlockForwardGraphMatchesGolden) {
  Result<FftNetworkForwardBundle> bundle_result = BuildFnoForwardGraph(device_);
  ASSERT_TRUE(bundle_result.is_ok()) << bundle_result.status().message();
  FftNetworkForwardBundle bundle = std::move(bundle_result.value());

  EXPECT_TRUE(
      AssertGraphMatchesGolden(bundle.graph, "tests/cpp/nn/testdata/fno1d_forward_expected.txt"));
}

// ---------------------------------------------------------------------------
// 1b/2b. FourierFilter1d 数值 golden + IR dump_text golden(cpu 单后端)。
// ---------------------------------------------------------------------------

TEST_F(FftBatchForwardNumericTest, FourierFilter1dForwardMatchesRecordedGolden) {
  Result<FftNetworkForwardBundle> bundle_result = BuildFourierFilterForwardGraph(device_);
  ASSERT_TRUE(bundle_result.is_ok()) << bundle_result.status().message();
  FftNetworkForwardBundle bundle = std::move(bundle_result.value());

  const Result<std::shared_ptr<Executable>> executable =
      frame::runtime::compile(bundle.graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 rng(20260722U);
  std::uniform_real_distribution<float> x_dist(-1.0F, 1.0F);
  std::vector<float> x_values(static_cast<size_t>(kFilterBatch * kFilterChannels * kFilterN));
  for (float& v : x_values) v = x_dist(rng);
  const Tensor x_tensor = MakeTensorFromFloats(
      x_values, Shape({kFilterBatch, kFilterChannels, kFilterN}), device_, *allocator_);

  const std::vector<Tensor> params =
      MakeUniformParamTensors(bundle.param_specs, rng, -0.3F, 0.3F, device_, *allocator_);

  std::vector<Tensor> inputs{x_tensor};
  for (const Tensor& p : params) inputs.push_back(p);
  const Result<std::vector<Tensor>> run_result = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(run_result.is_ok()) << run_result.status().message();
  ASSERT_EQ(run_result.value().size(), 1u);

  // golden 值 = 本用例首次实测记录(同上 Fno1dBlock 用例头注释判据)。
  const std::vector<float> golden_values{
      0.0996775627F,   -0.153041467F,  -0.120685458F,  0.0649767146F,   0.102422252F,
      -0.181810305F,   0.00427447259F, 0.172333956F,   0.045943737F,    -0.0693236366F,
      0.000256940722F, -0.0313603096F, -0.25935927F,   -0.0708117411F,  -0.216009989F,
      -0.0333811082F,  -0.0195416398F, -0.209431827F,  -0.128422976F,   0.030044619F,
      0.0373080932F,   0.0202538148F,  -0.0876683295F, -0.0548662134F,  -0.0273980796F,
      -0.00900672376F, -0.0782283396F, -0.0134409741F, 0.280059844F,    0.242970809F,
      0.00272133201F,  -0.0849491656F, -0.15874587F,   -0.00847481191F, -0.109840281F,
      0.225723296F,    -0.170684472F,  0.108642906F,   -0.210940659F,   0.280749947F,
      0.0711828247F,   0.0483017117F,  -0.0799261332F, -0.152345344F,   -0.118453361F,
      -0.136317685F,   0.0910016596F,  -0.0388787799F};
  const Tensor golden = MakeTensorFromFloats(
      golden_values, Shape({kFilterBatch, kFilterChannels, kFilterN}), device_, *allocator_);
  EXPECT_TRUE(
      tensor_all_close(run_result.value()[0], golden, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(FftBatchForwardNumericTest, FourierFilter1dForwardGraphMatchesGolden) {
  Result<FftNetworkForwardBundle> bundle_result = BuildFourierFilterForwardGraph(device_);
  ASSERT_TRUE(bundle_result.is_ok()) << bundle_result.status().message();
  FftNetworkForwardBundle bundle = std::move(bundle_result.value());

  EXPECT_TRUE(AssertGraphMatchesGolden(
      bundle.graph, "tests/cpp/nn/testdata/fourier_filter1d_forward_expected.txt"));
}

// ---------------------------------------------------------------------------
// M25 Mamba/FourierMamba 固定种子前向数值与 IR golden。
// ---------------------------------------------------------------------------

TEST_F(FftBatchForwardNumericTest, MambaForwardMatchesRecordedGolden) {
  Result<FftNetworkForwardBundle> bundle_result = BuildSsmForwardGraph(device_, false);
  ASSERT_TRUE(bundle_result.is_ok()) << bundle_result.status().message();
  FftNetworkForwardBundle bundle = std::move(bundle_result.value());
  const Result<std::shared_ptr<Executable>> executable =
      frame::runtime::compile(bundle.graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  // 固定 seed 与取样区间共同定义数值回归素材。
  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 rng(20260725U);
  std::uniform_real_distribution<float> x_dist(-1.0F, 1.0F);
  std::vector<float> x_values(static_cast<size_t>(kSsmBatch * kSsmChannels * kSsmSteps));
  for (float& value : x_values) value = x_dist(rng);
  const Tensor x = MakeTensorFromFloats(x_values, Shape({kSsmBatch, kSsmChannels, kSsmSteps}),
                                        device_, *allocator_);
  const std::vector<Tensor> params =
      MakeUniformParamTensors(bundle.param_specs, rng, -0.2F, 0.2F, device_, *allocator_);
  std::vector<Tensor> inputs{x};
  inputs.insert(inputs.end(), params.begin(), params.end());
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  const std::vector<float> golden_values{-0.0180842F, -0.0164103F, -0.134907F, -0.0954698F,
                                         0.0307592F,  0.123954F,   -0.159442F, 0.0699166F};
  const Tensor golden = MakeTensorFromFloats(
      golden_values, Shape({kSsmBatch, kSsmChannels, kSsmSteps}), device_, *allocator_);
  EXPECT_TRUE(tensor_all_close(outputs.value()[0], golden, default_tolerance(DTypeCode::kFloat32)))
      << FloatTensorValues(outputs.value()[0]);
}

TEST_F(FftBatchForwardNumericTest, FourierMambaForwardMatchesRecordedGolden) {
  Result<FftNetworkForwardBundle> bundle_result = BuildSsmForwardGraph(device_, true);
  ASSERT_TRUE(bundle_result.is_ok()) << bundle_result.status().message();
  FftNetworkForwardBundle bundle = std::move(bundle_result.value());
  const Result<std::shared_ptr<Executable>> executable =
      frame::runtime::compile(bundle.graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 rng(20260726U);
  std::uniform_real_distribution<float> x_dist(-1.0F, 1.0F);
  std::vector<float> x_values(static_cast<size_t>(kSsmBatch * kSsmChannels * kSsmSteps));
  for (float& value : x_values) value = x_dist(rng);
  const Tensor x = MakeTensorFromFloats(x_values, Shape({kSsmBatch, kSsmChannels, kSsmSteps}),
                                        device_, *allocator_);
  const std::vector<Tensor> params =
      MakeUniformParamTensors(bundle.param_specs, rng, -0.2F, 0.2F, device_, *allocator_);
  std::vector<Tensor> inputs{x};
  inputs.insert(inputs.end(), params.begin(), params.end());
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();

  const std::vector<float> golden_values{0.0184308F,  0.0579612F, 0.194091F,  0.213085F,
                                         -0.0754015F, -0.181585F, -0.129196F, -0.175023F};
  const Tensor golden = MakeTensorFromFloats(
      golden_values, Shape({kSsmBatch, kSsmChannels, kSsmSteps}), device_, *allocator_);
  EXPECT_TRUE(tensor_all_close(outputs.value()[0], golden, default_tolerance(DTypeCode::kFloat32)))
      << FloatTensorValues(outputs.value()[0]);
}

TEST_F(FftBatchForwardNumericTest, MambaForwardGraphMatchesGolden) {
  Result<FftNetworkForwardBundle> bundle_result = BuildSsmForwardGraph(device_, false);
  ASSERT_TRUE(bundle_result.is_ok()) << bundle_result.status().message();
  EXPECT_TRUE(AssertGraphMatchesGolden(bundle_result.value().graph,
                                       "tests/cpp/nn/testdata/mamba_forward_expected.txt"));
}

TEST_F(FftBatchForwardNumericTest, FourierMambaForwardGraphMatchesGolden) {
  Result<FftNetworkForwardBundle> bundle_result = BuildSsmForwardGraph(device_, true);
  ASSERT_TRUE(bundle_result.is_ok()) << bundle_result.status().message();
  EXPECT_TRUE(AssertGraphMatchesGolden(bundle_result.value().graph,
                                       "tests/cpp/nn/testdata/fourier_mamba_forward_expected.txt"));
}

// ---------------------------------------------------------------------------
// 4. FourierFilter1d"逐样本参数语义"(设计门建议 3,§1.5):两个 batch 行的
// 输入 x 相同、w_re/w_im 不同,输出必不同(证明并非跨样本共享滤波器)。
// ---------------------------------------------------------------------------

TEST_F(FftBatchForwardNumericTest, FourierFilter1dPerSampleWeightsProduceDifferentOutputs) {
  Result<FftNetworkForwardBundle> bundle_result = BuildFourierFilterForwardGraph(device_);
  ASSERT_TRUE(bundle_result.is_ok()) << bundle_result.status().message();
  FftNetworkForwardBundle bundle = std::move(bundle_result.value());
  ASSERT_EQ(bundle.param_specs.size(), 2u);  // w_re, w_im,各 [B,C,K,1]

  const Result<std::shared_ptr<Executable>> executable =
      frame::runtime::compile(bundle.graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  // 两个 batch 行的 x 恰相同(逐 channel/位置字面重复),若 FourierFilter1d
  // 是"跨样本共享滤波器"则两行输出会相同;而 w_re/w_im 逐样本取不同常量值,
  // 正确实现下两行输出必不同。
  std::vector<float> x_row{1.0F, -0.5F, 0.25F, 0.75F, -1.0F, 0.5F, -0.25F, 0.125F};
  ASSERT_EQ(static_cast<int64_t>(x_row.size()), kFilterN);
  std::vector<float> x_values;
  x_values.reserve(static_cast<size_t>(kFilterBatch * kFilterChannels * kFilterN));
  for (int64_t b = 0; b < kFilterBatch; ++b) {
    for (int64_t c = 0; c < kFilterChannels; ++c) {
      for (float v : x_row) x_values.push_back(v);
    }
  }
  const Tensor x_tensor = MakeTensorFromFloats(
      x_values, Shape({kFilterBatch, kFilterChannels, kFilterN}), device_, *allocator_);

  constexpr int64_t kK = kFilterN / 2 + 1;  // 5
  const int64_t per_batch_numel = kFilterChannels * kK;
  std::vector<float> w_re_values(static_cast<size_t>(kFilterBatch * per_batch_numel));
  std::vector<float> w_im_values(static_cast<size_t>(kFilterBatch * per_batch_numel));
  for (int64_t b = 0; b < kFilterBatch; ++b) {
    // batch 0 全体 w_re=1,w_im=0(恒等滤波器的一种);batch 1 全体 w_re=0,
    // w_im=1(90 度相移滤波器)——两者数学上必产生不同输出,除非输入该模态
    // 幅值恰为 0(非本例情形,x_row 含多个非零频点)。
    const float re_fill = (b == 0) ? 1.0F : 0.0F;
    const float im_fill = (b == 0) ? 0.0F : 1.0F;
    for (int64_t i = 0; i < per_batch_numel; ++i) {
      w_re_values[static_cast<size_t>(b * per_batch_numel + i)] = re_fill;
      w_im_values[static_cast<size_t>(b * per_batch_numel + i)] = im_fill;
    }
  }
  const Shape w_shape({kFilterBatch, kFilterChannels, kK, 1});
  const Tensor w_re_tensor = MakeTensorFromFloats(w_re_values, w_shape, device_, *allocator_);
  const Tensor w_im_tensor = MakeTensorFromFloats(w_im_values, w_shape, device_, *allocator_);

  std::vector<Tensor> inputs{x_tensor, w_re_tensor, w_im_tensor};
  const Result<std::vector<Tensor>> run_result = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(run_result.is_ok()) << run_result.status().message();
  ASSERT_EQ(run_result.value().size(), 1u);

  // 两行是否"不同"复用 BUILD-011 同一容差工具做否定判据(EXPECT_FALSE +
  // tensor_all_close),不手写自造阈值:把两个 batch 行的输出各自拆成独立
  // Tensor 后,断言二者在 BUILD-011 fp32 容差下不相等。
  const float* actual = static_cast<const float*>(run_result.value()[0].raw_data());
  const int64_t per_sample_numel = kFilterChannels * kFilterN;
  std::vector<float> row0_values(actual, actual + per_sample_numel);
  std::vector<float> row1_values(actual + per_sample_numel, actual + 2 * per_sample_numel);
  const Shape row_shape({kFilterChannels, kFilterN});
  const Tensor row0_tensor = MakeTensorFromFloats(row0_values, row_shape, device_, *allocator_);
  const Tensor row1_tensor = MakeTensorFromFloats(row1_values, row_shape, device_, *allocator_);
  EXPECT_FALSE(tensor_all_close(row0_tensor, row1_tensor, default_tolerance(DTypeCode::kFloat32)))
      << "FourierFilter1d 的两个 batch 行(相同 x、不同 w_re/w_im)产出了相同输出,疑似把逐"
         "样本参数当成了跨样本共享滤波器使用";
}

// ---------------------------------------------------------------------------
// 3. Fno1dBlock CPU/CUDA 前向一致(BUILD-010/M24 SKIP 口径同
// tests/cpp/backends/test_cuda_backend.cpp::CudaBackendTest)。
// ---------------------------------------------------------------------------

class FftNetworkCudaConsistencyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const Result<Backend*> cuda_result = BackendRegistry::instance().get(frame::kCudaBackendName);
    if (!cuda_result.is_ok()) {
      GTEST_SKIP() << "CUDA backend 'cuda' is not registered in this build (configured without "
                      "FRAME_ENABLE_CUDA or CUDA Toolkit was not detected at configure time)";
    }
    cuda_backend_ = cuda_result.value();

    const Result<int32_t> count = cuda_backend_->device_count();
    ASSERT_TRUE(count.is_ok()) << count.status().message();
    if (count.value() < 1) {
      GTEST_SKIP() << "no CUDA device available on this machine (CudaBackend::device_count() "
                      "returned 0)";
    }

    const Result<Backend*> cpu_result = BackendRegistry::instance().get(frame::kCpuBackendName);
    ASSERT_TRUE(cpu_result.is_ok()) << cpu_result.status().message();
    cpu_backend_ = cpu_result.value();

    cuda_device_ = Device{frame::kCudaBackendName, 0};
    cpu_device_ = cpu_device();

    cuda_allocator_ = cuda_backend_->allocator(cuda_device_);
    ASSERT_NE(cuda_allocator_, nullptr);
    cpu_allocator_ = cpu_backend_->allocator(cpu_device_);
    ASSERT_NE(cpu_allocator_, nullptr);

    Result<std::unique_ptr<Stream>> stream_result = cuda_backend_->create_stream(cuda_device_);
    ASSERT_TRUE(stream_result.is_ok()) << stream_result.status().message();
    stream_ = std::move(stream_result.value());
  }

  Tensor CopyToDevice(const Tensor& host) {
    Result<Tensor> device_result =
        Tensor::empty(host.shape(), host.dtype(), cuda_device_, *cuda_allocator_);
    EXPECT_TRUE(device_result.is_ok()) << device_result.status().message();
    Tensor device_tensor = device_result.value();
    const size_t bytes = static_cast<size_t>(host.numel()) * host.dtype().itemsize();
    if (bytes > 0) {
      const Status status = cuda_backend_->copy(device_tensor.raw_data(), cuda_device_,
                                                host.raw_data(), cpu_device_, bytes, stream_.get());
      EXPECT_TRUE(status.is_ok()) << status.message();
    }
    return device_tensor;
  }

  Tensor CopyToHost(const Tensor& device_tensor) {
    Result<Tensor> host_result =
        Tensor::empty(device_tensor.shape(), device_tensor.dtype(), cpu_device_, *cpu_allocator_);
    EXPECT_TRUE(host_result.is_ok()) << host_result.status().message();
    Tensor host_tensor = host_result.value();
    const size_t bytes =
        static_cast<size_t>(device_tensor.numel()) * device_tensor.dtype().itemsize();
    if (bytes > 0) {
      const Status status =
          cuda_backend_->copy(host_tensor.raw_data(), cpu_device_, device_tensor.raw_data(),
                              cuda_device_, bytes, stream_.get());
      EXPECT_TRUE(status.is_ok()) << status.message();
    }
    return host_tensor;
  }

  Backend* cuda_backend_ = nullptr;
  Backend* cpu_backend_ = nullptr;
  Device cuda_device_{};
  Device cpu_device_{};
  Allocator* cuda_allocator_ = nullptr;
  Allocator* cpu_allocator_ = nullptr;
  std::unique_ptr<Stream> stream_;
};

TEST_F(FftNetworkCudaConsistencyTest, Fno1dBlockCudaForwardMatchesCpu) {
  Result<FftNetworkForwardBundle> cpu_bundle_result = BuildFnoForwardGraph(cpu_device_);
  ASSERT_TRUE(cpu_bundle_result.is_ok()) << cpu_bundle_result.status().message();
  FftNetworkForwardBundle cpu_bundle = std::move(cpu_bundle_result.value());
  Result<FftNetworkForwardBundle> cuda_bundle_result = BuildFnoForwardGraph(cuda_device_);
  ASSERT_TRUE(cuda_bundle_result.is_ok()) << cuda_bundle_result.status().message();
  FftNetworkForwardBundle cuda_bundle = std::move(cuda_bundle_result.value());

  const Result<std::shared_ptr<Executable>> cpu_executable =
      frame::runtime::compile(cpu_bundle.graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(cpu_executable.is_ok()) << cpu_executable.status().message();
  const Result<std::shared_ptr<Executable>> cuda_executable =
      frame::runtime::compile(cuda_bundle.graph, frame::kCudaBackendName, CompileOptions{});
  ASSERT_TRUE(cuda_executable.is_ok()) << cuda_executable.status().message();

  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 rng(20260723U);
  std::uniform_real_distribution<float> x_dist(-1.0F, 1.0F);
  std::vector<float> x_values(static_cast<size_t>(kFnoBatch * kFnoIn * kFnoN));
  for (float& v : x_values) v = x_dist(rng);
  const Tensor x_host = MakeTensorFromFloats(x_values, Shape({kFnoBatch, kFnoIn, kFnoN}),
                                             cpu_device_, *cpu_allocator_);
  const std::vector<Tensor> params_host = MakeUniformParamTensors(
      cpu_bundle.param_specs, rng, -0.2F, 0.2F, cpu_device_, *cpu_allocator_);

  std::vector<Tensor> cpu_inputs{x_host};
  for (const Tensor& p : params_host) cpu_inputs.push_back(p);
  const Result<std::vector<Tensor>> cpu_outputs = frame::runtime::run_with_allocated_outputs(
      *cpu_executable.value(), frame::kCpuBackendName, cpu_inputs);
  ASSERT_TRUE(cpu_outputs.is_ok()) << cpu_outputs.status().message();
  ASSERT_EQ(cpu_outputs.value().size(), 1u);

  std::vector<Tensor> cuda_inputs{CopyToDevice(x_host)};
  for (const Tensor& p : params_host) cuda_inputs.push_back(CopyToDevice(p));
  const Result<std::vector<Tensor>> cuda_outputs = frame::runtime::run_with_allocated_outputs(
      *cuda_executable.value(), frame::kCudaBackendName, cuda_inputs);
  ASSERT_TRUE(cuda_outputs.is_ok()) << cuda_outputs.status().message();
  ASSERT_EQ(cuda_outputs.value().size(), 1u);
  const Tensor cuda_out_host = CopyToHost(cuda_outputs.value()[0]);
  ASSERT_TRUE(stream_->synchronize().is_ok());

  EXPECT_TRUE(tensor_all_close(cuda_out_host, cpu_outputs.value()[0],
                               default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(FftNetworkCudaConsistencyTest, FourierFilter1dCudaForwardMatchesCpu) {
  Result<FftNetworkForwardBundle> cpu_bundle_result = BuildFourierFilterForwardGraph(cpu_device_);
  ASSERT_TRUE(cpu_bundle_result.is_ok()) << cpu_bundle_result.status().message();
  FftNetworkForwardBundle cpu_bundle = std::move(cpu_bundle_result.value());
  Result<FftNetworkForwardBundle> cuda_bundle_result = BuildFourierFilterForwardGraph(cuda_device_);
  ASSERT_TRUE(cuda_bundle_result.is_ok()) << cuda_bundle_result.status().message();
  FftNetworkForwardBundle cuda_bundle = std::move(cuda_bundle_result.value());

  const Result<std::shared_ptr<Executable>> cpu_executable =
      frame::runtime::compile(cpu_bundle.graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(cpu_executable.is_ok()) << cpu_executable.status().message();
  const Result<std::shared_ptr<Executable>> cuda_executable =
      frame::runtime::compile(cuda_bundle.graph, frame::kCudaBackendName, CompileOptions{});
  ASSERT_TRUE(cuda_executable.is_ok()) << cuda_executable.status().message();

  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 rng(20260724U);
  std::uniform_real_distribution<float> x_dist(-1.0F, 1.0F);
  std::vector<float> x_values(static_cast<size_t>(kFilterBatch * kFilterChannels * kFilterN));
  for (float& v : x_values) v = x_dist(rng);
  const Tensor x_host = MakeTensorFromFloats(
      x_values, Shape({kFilterBatch, kFilterChannels, kFilterN}), cpu_device_, *cpu_allocator_);
  const std::vector<Tensor> params_host = MakeUniformParamTensors(
      cpu_bundle.param_specs, rng, -0.3F, 0.3F, cpu_device_, *cpu_allocator_);

  std::vector<Tensor> cpu_inputs{x_host};
  for (const Tensor& p : params_host) cpu_inputs.push_back(p);
  const Result<std::vector<Tensor>> cpu_outputs = frame::runtime::run_with_allocated_outputs(
      *cpu_executable.value(), frame::kCpuBackendName, cpu_inputs);
  ASSERT_TRUE(cpu_outputs.is_ok()) << cpu_outputs.status().message();
  ASSERT_EQ(cpu_outputs.value().size(), 1u);

  std::vector<Tensor> cuda_inputs{CopyToDevice(x_host)};
  for (const Tensor& p : params_host) cuda_inputs.push_back(CopyToDevice(p));
  const Result<std::vector<Tensor>> cuda_outputs = frame::runtime::run_with_allocated_outputs(
      *cuda_executable.value(), frame::kCudaBackendName, cuda_inputs);
  ASSERT_TRUE(cuda_outputs.is_ok()) << cuda_outputs.status().message();
  ASSERT_EQ(cuda_outputs.value().size(), 1u);
  const Tensor cuda_out_host = CopyToHost(cuda_outputs.value()[0]);
  ASSERT_TRUE(stream_->synchronize().is_ok());

  EXPECT_TRUE(tensor_all_close(cuda_out_host, cpu_outputs.value()[0],
                               default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(FftNetworkCudaConsistencyTest, MambaCudaForwardMatchesCpu) {
  Result<FftNetworkForwardBundle> cpu_bundle_result = BuildSsmForwardGraph(cpu_device_, false);
  ASSERT_TRUE(cpu_bundle_result.is_ok()) << cpu_bundle_result.status().message();
  Result<FftNetworkForwardBundle> cuda_bundle_result = BuildSsmForwardGraph(cuda_device_, false);
  ASSERT_TRUE(cuda_bundle_result.is_ok()) << cuda_bundle_result.status().message();
  FftNetworkForwardBundle cpu_bundle = std::move(cpu_bundle_result.value());
  FftNetworkForwardBundle cuda_bundle = std::move(cuda_bundle_result.value());

  const Result<std::shared_ptr<Executable>> cpu_executable =
      frame::runtime::compile(cpu_bundle.graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(cpu_executable.is_ok()) << cpu_executable.status().message();
  const Result<std::shared_ptr<Executable>> cuda_executable =
      frame::runtime::compile(cuda_bundle.graph, frame::kCudaBackendName, CompileOptions{});
  ASSERT_TRUE(cuda_executable.is_ok()) << cuda_executable.status().message();

  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 rng(20260727U);
  std::uniform_real_distribution<float> x_dist(-1.0F, 1.0F);
  std::vector<float> x_values(static_cast<size_t>(kSsmBatch * kSsmChannels * kSsmSteps));
  for (float& value : x_values) value = x_dist(rng);
  const Tensor x = MakeTensorFromFloats(x_values, Shape({kSsmBatch, kSsmChannels, kSsmSteps}),
                                        cpu_device_, *cpu_allocator_);
  const std::vector<Tensor> params = MakeUniformParamTensors(cpu_bundle.param_specs, rng, -0.2F,
                                                             0.2F, cpu_device_, *cpu_allocator_);
  std::vector<Tensor> cpu_inputs{x};
  cpu_inputs.insert(cpu_inputs.end(), params.begin(), params.end());
  const Result<std::vector<Tensor>> cpu_outputs = frame::runtime::run_with_allocated_outputs(
      *cpu_executable.value(), frame::kCpuBackendName, cpu_inputs);
  ASSERT_TRUE(cpu_outputs.is_ok()) << cpu_outputs.status().message();

  std::vector<Tensor> cuda_inputs{CopyToDevice(x)};
  for (const Tensor& param : params) cuda_inputs.push_back(CopyToDevice(param));
  const Result<std::vector<Tensor>> cuda_outputs = frame::runtime::run_with_allocated_outputs(
      *cuda_executable.value(), frame::kCudaBackendName, cuda_inputs);
  ASSERT_TRUE(cuda_outputs.is_ok()) << cuda_outputs.status().message();
  const Tensor cuda_host = CopyToHost(cuda_outputs.value()[0]);
  ASSERT_TRUE(stream_->synchronize().is_ok());
  EXPECT_TRUE(
      tensor_all_close(cuda_host, cpu_outputs.value()[0], default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(FftNetworkCudaConsistencyTest, FourierMambaCudaForwardMatchesCpu) {
  Result<FftNetworkForwardBundle> cpu_bundle_result = BuildSsmForwardGraph(cpu_device_, true);
  ASSERT_TRUE(cpu_bundle_result.is_ok()) << cpu_bundle_result.status().message();
  Result<FftNetworkForwardBundle> cuda_bundle_result = BuildSsmForwardGraph(cuda_device_, true);
  ASSERT_TRUE(cuda_bundle_result.is_ok()) << cuda_bundle_result.status().message();
  FftNetworkForwardBundle cpu_bundle = std::move(cpu_bundle_result.value());
  FftNetworkForwardBundle cuda_bundle = std::move(cuda_bundle_result.value());

  const Result<std::shared_ptr<Executable>> cpu_executable =
      frame::runtime::compile(cpu_bundle.graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(cpu_executable.is_ok()) << cpu_executable.status().message();
  const Result<std::shared_ptr<Executable>> cuda_executable =
      frame::runtime::compile(cuda_bundle.graph, frame::kCudaBackendName, CompileOptions{});
  ASSERT_TRUE(cuda_executable.is_ok()) << cuda_executable.status().message();

  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 rng(20260728U);
  std::uniform_real_distribution<float> x_dist(-1.0F, 1.0F);
  std::vector<float> x_values(static_cast<size_t>(kSsmBatch * kSsmChannels * kSsmSteps));
  for (float& value : x_values) value = x_dist(rng);
  const Tensor x = MakeTensorFromFloats(x_values, Shape({kSsmBatch, kSsmChannels, kSsmSteps}),
                                        cpu_device_, *cpu_allocator_);
  const std::vector<Tensor> params = MakeUniformParamTensors(cpu_bundle.param_specs, rng, -0.2F,
                                                             0.2F, cpu_device_, *cpu_allocator_);
  std::vector<Tensor> cpu_inputs{x};
  cpu_inputs.insert(cpu_inputs.end(), params.begin(), params.end());
  const Result<std::vector<Tensor>> cpu_outputs = frame::runtime::run_with_allocated_outputs(
      *cpu_executable.value(), frame::kCpuBackendName, cpu_inputs);
  ASSERT_TRUE(cpu_outputs.is_ok()) << cpu_outputs.status().message();

  std::vector<Tensor> cuda_inputs{CopyToDevice(x)};
  for (const Tensor& param : params) cuda_inputs.push_back(CopyToDevice(param));
  const Result<std::vector<Tensor>> cuda_outputs = frame::runtime::run_with_allocated_outputs(
      *cuda_executable.value(), frame::kCudaBackendName, cuda_inputs);
  ASSERT_TRUE(cuda_outputs.is_ok()) << cuda_outputs.status().message();
  const Tensor cuda_host = CopyToHost(cuda_outputs.value()[0]);
  ASSERT_TRUE(stream_->synchronize().is_ok());
  EXPECT_TRUE(
      tensor_all_close(cuda_host, cpu_outputs.value()[0], default_tolerance(DTypeCode::kFloat32)));
}

}  // namespace
