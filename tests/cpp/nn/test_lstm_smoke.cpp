// nn::LSTM 端到端数值冒烟(M22 批4 T5,docs/plan/2026-07-19-batch4-m22-seq.md
// §1.7):小图经 runtime::compile("cpu") 前向执行一次,与手算参考值对照
// (门序 i,f,g,o 固定,h0=c0=0,z=x_t·W_ih+h·W_hh+bias)。手法同
// tests/cpp/nn/test_aff_smoke.cpp fixture(取真实 cpu 后端 Allocator 经
// BackendRegistry)与 tests/cpp/nn/test_training_smoke.cpp 的
// MakeTensorFromFloats 手法。
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/executable.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/nn/layers.h>
#include <frame/nn/module.h>
#include <frame/runtime/compile.h>

namespace {

using frame::cpu_device;
using frame::DType;
using frame::Result;
using frame::Shape;
using frame::Tensor;
using frame::hal::Backend;
using frame::hal::BackendRegistry;
using frame::hal::CompileOptions;
using frame::hal::Executable;
using frame::ir::Graph;
using frame::ir::TensorType;
using frame::ir::Value;
using frame::nn::add_parameter_inputs;
using frame::nn::LSTM;
using frame::nn::Module;
using frame::nn::ParamSpec;

TensorType MakeCpuTensorType(std::vector<int64_t> dims) {
  TensorType type;
  type.dtype = DType::of<float>();
  type.shape = Shape(std::move(dims));
  type.device = cpu_device();
  return type;
}

double Sigmoid(double v) { return 1.0 / (1.0 + std::exp(-v)); }

class LSTMSmokeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const Result<Backend*> backend_result = BackendRegistry::instance().get(frame::kCpuBackendName);
    ASSERT_TRUE(backend_result.is_ok());
    backend_ = backend_result.value();
    device_ = cpu_device();
    allocator_ = backend_->allocator(device_);
    ASSERT_NE(allocator_, nullptr);
  }

  Tensor MakeTensorFromFloats(const std::vector<float>& values, const Shape& shape) {
    Tensor tensor = Tensor::empty(shape, DType::of<float>(), device_, *allocator_).value();
    float* data = tensor.data<float>();
    for (size_t i = 0; i < values.size(); ++i) data[i] = values[i];
    return tensor;
  }

  Backend* backend_ = nullptr;
  frame::Device device_{};
  frame::hal::Allocator* allocator_ = nullptr;
};

TEST_F(LSTMSmokeTest, ForwardMatchesHandComputedReference) {
  constexpr int64_t kBatch = 2;
  constexpr int64_t kNumSteps = 2;
  constexpr int64_t kInputDim = 2;
  constexpr int64_t kHiddenDim = 2;

  Graph graph("lstm_smoke");
  Value* x = graph.add_graph_input(MakeCpuTensorType({kBatch, kNumSteps, kInputDim})).value();

  const Module lstm = LSTM("lstm", kBatch, kNumSteps, kInputDim, kHiddenDim, DType::of<float>());
  const std::vector<ParamSpec> param_specs = lstm.parameters();
  const Result<std::vector<Value*>> param_inputs = add_parameter_inputs(graph, param_specs);
  ASSERT_TRUE(param_inputs.is_ok()) << param_inputs.status().message();
  const Result<std::vector<Value*>> outputs =
      lstm.build(graph, std::vector<Value*>{x}, param_inputs.value());
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  ASSERT_TRUE(graph.mark_output(outputs.value()[0]).is_ok());

  const Result<std::shared_ptr<Executable>> executable =
      frame::runtime::compile(graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  // 固定种子生成小规模数值配方(可复现,同 test_training_smoke.cpp 同款抑制)。
  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 rng(20260719U);
  std::uniform_real_distribution<float> dist(-0.5F, 0.5F);
  std::vector<float> x_values(static_cast<size_t>(kBatch * kNumSteps * kInputDim));
  for (float& v : x_values) v = dist(rng);
  std::vector<float> w_ih_values(static_cast<size_t>(kInputDim * 4 * kHiddenDim));
  for (float& v : w_ih_values) v = dist(rng);
  std::vector<float> w_hh_values(static_cast<size_t>(kHiddenDim * 4 * kHiddenDim));
  for (float& v : w_hh_values) v = dist(rng);
  std::vector<float> bias_values(static_cast<size_t>(kBatch * 4 * kHiddenDim));
  for (float& v : bias_values) v = dist(rng);

  std::vector<Tensor> inputs{MakeTensorFromFloats(x_values, Shape({kBatch, kNumSteps, kInputDim})),
                             MakeTensorFromFloats(w_ih_values, Shape({kInputDim, 4 * kHiddenDim})),
                             MakeTensorFromFloats(w_hh_values, Shape({kHiddenDim, 4 * kHiddenDim})),
                             MakeTensorFromFloats(bias_values, Shape({kBatch, 4 * kHiddenDim}))};
  const Result<std::vector<Tensor>> run_result = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(run_result.is_ok()) << run_result.status().message();
  ASSERT_EQ(run_result.value().size(), 1u);
  const float* actual = static_cast<const float*>(run_result.value()[0].raw_data());

  // 手算参考(double 精度,门序 i,f,g,o 固定,h0=c0=0;同
  // src/nn/layers.cpp::LSTM 的构图公式逐步复算)。
  std::vector<double> h(static_cast<size_t>(kBatch * kHiddenDim), 0.0);
  std::vector<double> c(static_cast<size_t>(kBatch * kHiddenDim), 0.0);
  for (int64_t t = 0; t < kNumSteps; ++t) {
    std::vector<double> z(static_cast<size_t>(kBatch * 4 * kHiddenDim), 0.0);
    for (int64_t b = 0; b < kBatch; ++b) {
      for (int64_t j = 0; j < 4 * kHiddenDim; ++j) {
        double acc = static_cast<double>(bias_values[static_cast<size_t>(b * 4 * kHiddenDim + j)]);
        for (int64_t k = 0; k < kInputDim; ++k) {
          const double x_t = static_cast<double>(
              x_values[static_cast<size_t>((b * kNumSteps + t) * kInputDim + k)]);
          acc +=
              x_t * static_cast<double>(w_ih_values[static_cast<size_t>(k * 4 * kHiddenDim + j)]);
        }
        for (int64_t k = 0; k < kHiddenDim; ++k) {
          acc += h[static_cast<size_t>(b * kHiddenDim + k)] *
                 static_cast<double>(w_hh_values[static_cast<size_t>(k * 4 * kHiddenDim + j)]);
        }
        z[static_cast<size_t>(b * 4 * kHiddenDim + j)] = acc;
      }
    }
    for (int64_t b = 0; b < kBatch; ++b) {
      for (int64_t j = 0; j < kHiddenDim; ++j) {
        const double zi = z[static_cast<size_t>(b * 4 * kHiddenDim + j)];
        const double zf = z[static_cast<size_t>(b * 4 * kHiddenDim + kHiddenDim + j)];
        const double zg = z[static_cast<size_t>(b * 4 * kHiddenDim + 2 * kHiddenDim + j)];
        const double zo = z[static_cast<size_t>(b * 4 * kHiddenDim + 3 * kHiddenDim + j)];
        const double gate_i = Sigmoid(zi);
        const double gate_f = Sigmoid(zf);
        const double gate_g = std::tanh(zg);
        const double gate_o = Sigmoid(zo);
        const double c_prev = c[static_cast<size_t>(b * kHiddenDim + j)];
        const double c_new = gate_f * c_prev + gate_i * gate_g;
        c[static_cast<size_t>(b * kHiddenDim + j)] = c_new;
        h[static_cast<size_t>(b * kHiddenDim + j)] = gate_o * std::tanh(c_new);
      }
    }
  }

  for (size_t i = 0; i < h.size(); ++i) {
    EXPECT_NEAR(static_cast<double>(actual[i]), h[i], 1e-3)
        << "index " << i << " actual=" << actual[i] << " expected=" << h[i];
  }
}

}  // namespace
