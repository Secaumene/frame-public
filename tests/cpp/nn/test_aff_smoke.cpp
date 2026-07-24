// nn::AFF 端到端数值冒烟(M21 批3 T6,docs/plan/2026-07-18-batch3-m21-conv.md
// 第1.4节):小图(N=1,C=1,H=W=1)经 runtime::compile("cpu") 前向执行一次,
// 与手算参考值对照(out = M*X + (1-M)*Y,M=sigmoid(c2(relu(c1(X+Y))))),
// BUILD-011 float32 容差。手法同
// tests/cpp/nn/test_training_smoke.cpp::NnTrainingSmokeTest fixture(取真实
// cpu 后端 Allocator 经 BackendRegistry),本文件为单次前向(非训练循环)故不
// 复用其训练线组装,仅编译执行前向图。
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
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
using frame::nn::AFF;
using frame::nn::Module;
using frame::nn::ParamSpec;

TensorType MakeCpuTensorType(std::vector<int64_t> dims) {
  TensorType type;
  type.dtype = DType::of<float>();
  type.shape = Shape(std::move(dims));
  type.device = cpu_device();
  return type;
}

class AFFSmokeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const Result<Backend*> backend_result = BackendRegistry::instance().get(frame::kCpuBackendName);
    ASSERT_TRUE(backend_result.is_ok());
    backend_ = backend_result.value();
    device_ = cpu_device();
    allocator_ = backend_->allocator(device_);
    ASSERT_NE(allocator_, nullptr);
  }

  // shape 由调用方显式指定:X/Y/weight 是 [1,1,1,1](conv2d 输入/1x1 权重的秩
  // 4 形态),bias 是 [1](裁决点①:conv2d bias 为 [Cout] 秩1,非全形)。
  Tensor MakeSingleValueTensor(float value, const Shape& shape) {
    Tensor tensor = Tensor::empty(shape, DType::of<float>(), device_, *allocator_).value();
    *tensor.data<float>() = value;
    return tensor;
  }

  Backend* backend_ = nullptr;
  frame::Device device_{};
  frame::hal::Allocator* allocator_ = nullptr;
};

TEST_F(AFFSmokeTest, SinglePixelForwardMatchesHandComputedReference) {
  // channels=1、N=1、H=W=1(1x1 卷积在单像素上退化为标量仿射变换,
  // 手算参考值可逐位核对)。
  Graph graph("aff_smoke");
  Value* x = graph.add_graph_input(MakeCpuTensorType({1, 1, 1, 1})).value();
  Value* y = graph.add_graph_input(MakeCpuTensorType({1, 1, 1, 1})).value();

  const Module aff = AFF("aff", /*channels=*/1, DType::of<float>());
  const std::vector<ParamSpec> param_specs = aff.parameters();
  ASSERT_EQ(param_specs.size(), 4u);  // c1.weight, c1.bias, c2.weight, c2.bias

  const Result<std::vector<Value*>> param_inputs = add_parameter_inputs(graph, param_specs);
  ASSERT_TRUE(param_inputs.is_ok()) << param_inputs.status().message();

  const Result<std::vector<Value*>> outputs =
      aff.build(graph, std::vector<Value*>{x, y}, param_inputs.value());
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  ASSERT_TRUE(graph.mark_output(outputs.value()[0]).is_ok());

  const Result<std::shared_ptr<Executable>> executable =
      frame::runtime::compile(graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  // 固定数值配方(手算参考,见文件头注释推导):X=2, Y=3, w1=0.5, b1=1.0,
  // w2=0.2, b2=-0.1。
  constexpr float kX = 2.0F;
  constexpr float kY = 3.0F;
  constexpr float kW1 = 0.5F;
  constexpr float kB1 = 1.0F;
  constexpr float kW2 = 0.2F;
  constexpr float kB2 = -0.1F;

  const Shape kRank4Shape({1, 1, 1, 1});
  const Shape kBiasShape({1});
  std::vector<Tensor> inputs{
      MakeSingleValueTensor(kX, kRank4Shape),  MakeSingleValueTensor(kY, kRank4Shape),
      MakeSingleValueTensor(kW1, kRank4Shape), MakeSingleValueTensor(kB1, kBiasShape),
      MakeSingleValueTensor(kW2, kRank4Shape), MakeSingleValueTensor(kB2, kBiasShape)};
  const Result<std::vector<Tensor>> run_result = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(run_result.is_ok()) << run_result.status().message();
  ASSERT_EQ(run_result.value().size(), 1u);

  const float actual = *static_cast<const float*>(run_result.value()[0].raw_data());

  // 手算参考(double 精度):sum=X+Y; t1=w1*sum+b1; r=relu(t1);
  // t2=w2*r+b2; m=sigmoid(t2); out=m*X+(1-m)*Y = Y - m*(Y-X)。
  const double sum = static_cast<double>(kX) + static_cast<double>(kY);
  const double t1 = static_cast<double>(kW1) * sum + static_cast<double>(kB1);
  const double r = t1 > 0.0 ? t1 : 0.0;
  const double t2 = static_cast<double>(kW2) * r + static_cast<double>(kB2);
  const double m = 1.0 / (1.0 + std::exp(-t2));
  const double expected = m * static_cast<double>(kX) + (1.0 - m) * static_cast<double>(kY);

  EXPECT_NEAR(static_cast<double>(actual), expected, 1e-5)
      << "actual=" << actual << " expected=" << expected;
}

}  // namespace
