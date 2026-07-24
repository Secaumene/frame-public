// nn::MultiheadAttention 端到端数值冒烟(M22 批4 T5,docs/plan/
// 2026-07-19-batch4-m22-seq.md §1.7):小图经 runtime::compile("cpu") 前向
// 执行一次,与手算参考值对照(per-(b,h) 静态展开:行 slice+列 slice+
// matmul/transpose/softmax+concat)。手法同 tests/cpp/nn/test_aff_smoke.cpp
// fixture(取真实 cpu 后端 Allocator 经 BackendRegistry)。
#include <algorithm>
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
using frame::nn::Module;
using frame::nn::MultiheadAttention;
using frame::nn::ParamSpec;

TensorType MakeCpuTensorType(std::vector<int64_t> dims) {
  TensorType type;
  type.dtype = DType::of<float>();
  type.shape = Shape(std::move(dims));
  type.device = cpu_device();
  return type;
}

// --- 手算参考(double 精度)用的最小矩阵工具:行主序 vector<double> + 显式
// 行列数(REUSE-002:本文件内部专用,不入公开 API,不复用仓内既有 kernel——
// 那些针对 Tensor/dtype dispatch,与本处的纯 double 手算参考属不同抽象层)。
// m/k/n 三个相邻 int64_t 形参语义上不可合并(标准矩阵乘维度契约);全部调用点
// 以具名局部变量按数学式直接传入,误置换会使矩阵乘 numel 不匹配、越界访问
// 立即崩溃或断言失败,不会静默产出错误结果。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::vector<double> MatMul(const std::vector<double>& a, int64_t m, int64_t k,
                           const std::vector<double>& b, int64_t n) {
  std::vector<double> out(static_cast<size_t>(m * n), 0.0);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int64_t p = 0; p < k; ++p) {
        acc += a[static_cast<size_t>(i * k + p)] * b[static_cast<size_t>(p * n + j)];
      }
      out[static_cast<size_t>(i * n + j)] = acc;
    }
  }
  return out;
}

// row_start/row_count/col_start/col_count 四个相邻形参语义上不可合并(块切片
// 契约);全部调用点以具名局部变量传入,误置换会使切片越界或形状不符,在本
// 文件配套的手算参考 vs 图执行结果对照断言中以数值不符立即失败。
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
std::vector<double> ExtractBlock(const std::vector<double>& mat, int64_t total_cols,
                                 int64_t row_start, int64_t row_count, int64_t col_start,
                                 int64_t col_count) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  std::vector<double> out(static_cast<size_t>(row_count * col_count));
  for (int64_t i = 0; i < row_count; ++i) {
    for (int64_t j = 0; j < col_count; ++j) {
      out[static_cast<size_t>(i * col_count + j)] =
          mat[static_cast<size_t>((row_start + i) * total_cols + col_start + j)];
    }
  }
  return out;
}

std::vector<double> Transpose(const std::vector<double>& mat, int64_t rows, int64_t cols) {
  std::vector<double> out(static_cast<size_t>(rows * cols));
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      out[static_cast<size_t>(j * rows + i)] = mat[static_cast<size_t>(i * cols + j)];
    }
  }
  return out;
}

// rows/cols 两个相邻形参:本文件唯一调用点传入的是注意力分数方阵
// (rows==cols),误置换在该场景不可观测、亦无害;保留具名形参以贴合逐行
// softmax 语义可读性,经 NOLINT 抑制该启发式误报。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void SoftmaxRowsInPlace(std::vector<double>& mat, int64_t rows, int64_t cols) {
  for (int64_t i = 0; i < rows; ++i) {
    double max_val = mat[static_cast<size_t>(i * cols)];
    for (int64_t j = 1; j < cols; ++j) {
      max_val = std::max(max_val, mat[static_cast<size_t>(i * cols + j)]);
    }
    double sum = 0.0;
    for (int64_t j = 0; j < cols; ++j) {
      const double e = std::exp(mat[static_cast<size_t>(i * cols + j)] - max_val);
      mat[static_cast<size_t>(i * cols + j)] = e;
      sum += e;
    }
    for (int64_t j = 0; j < cols; ++j) {
      mat[static_cast<size_t>(i * cols + j)] /= sum;
    }
  }
}

class MultiheadAttentionSmokeTest : public ::testing::Test {
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

TEST_F(MultiheadAttentionSmokeTest, ForwardMatchesHandComputedReference) {
  constexpr int64_t kBatch = 2;
  constexpr int64_t kSeqLen = 2;
  constexpr int64_t kEmbedDim = 4;
  constexpr int64_t kNumHeads = 2;
  constexpr int64_t kDh = kEmbedDim / kNumHeads;
  constexpr int64_t kRows = kBatch * kSeqLen;

  Graph graph("mha_smoke");
  Value* x = graph.add_graph_input(MakeCpuTensorType({kRows, kEmbedDim})).value();

  const Module mha = MultiheadAttention("mha", kBatch, kSeqLen, kEmbedDim, kNumHeads,
                                        /*with_bias=*/true, DType::of<float>());
  const std::vector<ParamSpec> param_specs = mha.parameters();
  ASSERT_EQ(param_specs.size(), 8u);
  const Result<std::vector<Value*>> param_inputs = add_parameter_inputs(graph, param_specs);
  ASSERT_TRUE(param_inputs.is_ok()) << param_inputs.status().message();
  const Result<std::vector<Value*>> outputs =
      mha.build(graph, std::vector<Value*>{x}, param_inputs.value());
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  ASSERT_TRUE(graph.mark_output(outputs.value()[0]).is_ok());

  const Result<std::shared_ptr<Executable>> executable =
      frame::runtime::compile(graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 rng(20260719U);
  std::uniform_real_distribution<float> dist(-0.5F, 0.5F);
  auto make_values = [&](size_t count) {
    std::vector<float> values(count);
    for (float& v : values) v = dist(rng);
    return values;
  };
  const std::vector<float> x_values = make_values(static_cast<size_t>(kRows * kEmbedDim));
  const std::vector<float> wq_values = make_values(static_cast<size_t>(kEmbedDim * kEmbedDim));
  const std::vector<float> bq_values = make_values(static_cast<size_t>(kRows * kEmbedDim));
  const std::vector<float> wk_values = make_values(static_cast<size_t>(kEmbedDim * kEmbedDim));
  const std::vector<float> bk_values = make_values(static_cast<size_t>(kRows * kEmbedDim));
  const std::vector<float> wv_values = make_values(static_cast<size_t>(kEmbedDim * kEmbedDim));
  const std::vector<float> bv_values = make_values(static_cast<size_t>(kRows * kEmbedDim));
  const std::vector<float> wo_values = make_values(static_cast<size_t>(kEmbedDim * kEmbedDim));
  const std::vector<float> bo_values = make_values(static_cast<size_t>(kRows * kEmbedDim));

  std::vector<Tensor> inputs{MakeTensorFromFloats(x_values, Shape({kRows, kEmbedDim})),
                             MakeTensorFromFloats(wq_values, Shape({kEmbedDim, kEmbedDim})),
                             MakeTensorFromFloats(bq_values, Shape({kRows, kEmbedDim})),
                             MakeTensorFromFloats(wk_values, Shape({kEmbedDim, kEmbedDim})),
                             MakeTensorFromFloats(bk_values, Shape({kRows, kEmbedDim})),
                             MakeTensorFromFloats(wv_values, Shape({kEmbedDim, kEmbedDim})),
                             MakeTensorFromFloats(bv_values, Shape({kRows, kEmbedDim})),
                             MakeTensorFromFloats(wo_values, Shape({kEmbedDim, kEmbedDim})),
                             MakeTensorFromFloats(bo_values, Shape({kRows, kEmbedDim}))};
  const Result<std::vector<Tensor>> run_result = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(run_result.is_ok()) << run_result.status().message();
  ASSERT_EQ(run_result.value().size(), 1u);
  const float* actual = static_cast<const float*>(run_result.value()[0].raw_data());

  // 手算参考(double 精度,同 src/nn/layers.cpp::MultiheadAttention 构图公式
  // 逐步复算):q/k/v=x@w+b;per-(b,h) scores=q_bh@k_bh^T*scale,softmax,
  // o_bh=softmax@v_bh;头沿列拼接、批沿行拼接;末投影 o=attn@wo+bo。
  auto to_double = [](const std::vector<float>& values) {
    return std::vector<double>(values.begin(), values.end());
  };
  const std::vector<double> x_d = to_double(x_values);
  auto linear = [&](const std::vector<double>& w, const std::vector<double>& b) {
    std::vector<double> out = MatMul(x_d, kRows, kEmbedDim, w, kEmbedDim);
    for (size_t i = 0; i < out.size(); ++i) out[i] += b[i];
    return out;
  };
  const std::vector<double> q = linear(to_double(wq_values), to_double(bq_values));
  const std::vector<double> k = linear(to_double(wk_values), to_double(bk_values));
  const std::vector<double> v = linear(to_double(wv_values), to_double(bv_values));

  const double scale = 1.0 / std::sqrt(static_cast<double>(kDh));
  std::vector<double> attn_out(static_cast<size_t>(kRows * kEmbedDim), 0.0);
  for (int64_t b = 0; b < kBatch; ++b) {
    for (int64_t h = 0; h < kNumHeads; ++h) {
      const std::vector<double> q_bh =
          ExtractBlock(q, kEmbedDim, b * kSeqLen, kSeqLen, h * kDh, kDh);
      const std::vector<double> k_bh =
          ExtractBlock(k, kEmbedDim, b * kSeqLen, kSeqLen, h * kDh, kDh);
      const std::vector<double> v_bh =
          ExtractBlock(v, kEmbedDim, b * kSeqLen, kSeqLen, h * kDh, kDh);
      const std::vector<double> k_bh_t = Transpose(k_bh, kSeqLen, kDh);
      std::vector<double> scores = MatMul(q_bh, kSeqLen, kDh, k_bh_t, kSeqLen);
      for (double& s : scores) s *= scale;
      SoftmaxRowsInPlace(scores, kSeqLen, kSeqLen);
      const std::vector<double> o_bh = MatMul(scores, kSeqLen, kSeqLen, v_bh, kDh);
      for (int64_t i = 0; i < kSeqLen; ++i) {
        for (int64_t j = 0; j < kDh; ++j) {
          attn_out[static_cast<size_t>((b * kSeqLen + i) * kEmbedDim + h * kDh + j)] =
              o_bh[static_cast<size_t>(i * kDh + j)];
        }
      }
    }
  }
  std::vector<double> expected =
      MatMul(attn_out, kRows, kEmbedDim, to_double(wo_values), kEmbedDim);
  const std::vector<double> bo_d = to_double(bo_values);
  for (size_t i = 0; i < expected.size(); ++i) expected[i] += bo_d[i];

  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(static_cast<double>(actual[i]), expected[i], 1e-3)
        << "index " << i << " actual=" << actual[i] << " expected=" << expected[i];
  }
}

}  // namespace
