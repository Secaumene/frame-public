// compiler::build_sgd_update_graph 单测 + 端到端 MLP 训练收敛测试(M18,
// docs/architecture/autograd.md 第5/6章,ARCH-065)。
//   1. build_sgd_update_graph 单测:单参数一步更新数值 ≡ 手算 p - lr·g;
//      错误路径(空 param_types/lr 非有限/dtype 白名单外)。
//   2. 端到端 MLP 训练收敛用例:host 驱动训练循环(autograd.md 第6章形态)——
//      前向+反向图 build_backward_graph 一次、runtime::compile("cpu") 一次;
//      build_sgd_update_graph 一次、runtime::compile("cpu") 一次;循环 N 步
//      只调用 run(),不重复 compile(两个 Executable 各自的 shared_ptr 在循环
//      外构造一次、循环内原样复用,结构性保证"零重编译",铁律 #1④)。
//      断言:①首末 loss 显著下降且末值低于绝对阈值;②FallbackStats 对图内
//      涉及的全部算子零计数(佐证全程走编译期 kernel 执行路径,未触发 M10
//      eager 回退链,ARCH-060/062 同 test_autograd.cpp 判定手法);③loss
//      序列全程有限(无 NaN/inf)。
//   3. fp16/bf16 SGD 单步数值用例(M19 Task 4,BUILD-011「fp16/bf16 经 fp32
//      解析参照验证」专款):同结构 build_sgd_update_graph 分别以 dtype=
//      float32(参照,结果下量化到目标 dtype)与 dtype=float16/bfloat16
//      (native)各 compile 一次、run 一次,按 default_tolerance(对应 dtype)
//      比较——不手算 p - lr·g 直接比较,因 native 图内部以量化常量
//      (-learning_rate)与逐步舍入的 mul/add 计算,与整体下量化的手算结果
//      存在细微量化路径差异。
//
// v0 约束下的最小可收敛 MLP 结构取舍(v0 无广播,见
// docs/architecture/operator-system.md 逐元素算子约束):标准 MLP 的 bias
// 通常是形如 [hidden_dim] 的向量,经广播加到 [batch, hidden_dim] 的
// matmul 输出上;v0 mul/add 均要求两操作数 shape 完全相同,无法表达这种
// 广播。本文件按 add 的同 shape 约束,把 bias 直接建模为与 matmul 输出同形的
// [batch, hidden_dim] 张量(每个样本持有一份独立的加性偏置,而非全 batch
// 共享同一份 [hidden_dim] 偏置向量)——这是 v0 无广播下的语义妥协,不是标准
// MLP bias 语义,但足以验证 M18 训练闭环(编译执行+反向图+更新图+收敛)本身,
// 与 tests/cpp/compiler/test_autograd.cpp::BuildCombinedLossGraph 的
// bias=[2,4] 同款处理手法一致。是否为 v0 引入广播属另案裁决(ADR-010 触发
// 清单),不在本里程碑范围。
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>
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
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/constant_utils.h>
#include <frame/ops/fused_elementwise_utils.h>
#include <frame/ops/graph_builder.h>
#include <frame/ops/op_registry.h>
#include <frame/runtime/compile.h>
#include <frame/runtime/fallback_stats.h>

#include "../common/tolerance.h"
#include "../ops/elementwise_op_test_helpers.h"
#include "mlp_forward_graph_helper.h"

namespace {

using frame::bfloat16_t;
using frame::DType;
using frame::DTypeCode;
using frame::ErrorCode;
using frame::float16_t;
using frame::float_to_bfloat16;
using frame::float_to_float16;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::Tensor;
using frame::compiler::build_backward_graph;
using frame::compiler::build_sgd_update_graph;
using frame::compiler::testing::BuildMlpForwardGraph;
using frame::compiler::testing::kMlpBatchSize;
using frame::compiler::testing::kMlpHiddenDim;
using frame::compiler::testing::kMlpInputDim;
using frame::compiler::testing::kMlpOutputDim;
using frame::hal::Backend;
using frame::hal::BackendRegistry;
using frame::hal::CompileOptions;
using frame::hal::Executable;
using frame::ir::Graph;
using frame::ir::TensorType;
using frame::ops::AttrMap;
using frame::ops::testing::MakeType;
using frame::runtime::FallbackStats;
using frame::testing::default_tolerance;
using frame::testing::tensor_all_close;

// 通用 fixture:取 cpu 后端真实 Allocator(经 BackendRegistry)+ 按 float
// 向量构造 Tensor 的 helper。与 test_autograd.cpp::AutogradTest 同思路。
class TrainingLoopTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const Result<Backend*> result = BackendRegistry::instance().get(frame::kCpuBackendName);
    ASSERT_TRUE(result.is_ok());
    backend_ = result.value();
    device_ = frame::cpu_device();
    allocator_ = backend_->allocator(device_);
    ASSERT_NE(allocator_, nullptr);
  }

  Tensor MakeTensorFromFloats(const std::vector<float>& values, const Shape& shape) {
    Tensor tensor = Tensor::empty(shape, DType::of<float>(), device_, *allocator_).value();
    float* data = tensor.data<float>();
    for (size_t i = 0; i < values.size(); ++i) data[i] = values[i];
    return tensor;
  }

  Tensor MakeFilledTensor(const Shape& shape, float value) {
    Tensor tensor = Tensor::empty(shape, DType::of<float>(), device_, *allocator_).value();
    float* data = tensor.data<float>();
    for (int64_t i = 0; i < tensor.numel(); ++i) data[i] = value;
    return tensor;
  }

  // fp16/bf16 SGD 单步数值用例专用(BUILD-011「fp16/bf16 经 fp32 解析参照
  // 验证」专款,与 test_autograd.cpp::AutogradTest 的
  // MakeFp16TensorFromFloats/ConvertFloatTensorToFp16 同一思路,但用模板
  // 合并 fp16/bf16 两份同构代码为一份,避免重复——铁律 #1②编译期机制优先):
  // 按 float 值列表构造给定标量类型 T(T ∈ {float, float16_t, bfloat16_t})的
  // Tensor,fp16/bf16 经各自位级转换函数量化。
  template <typename T>
  Tensor MakeTensorFromFloatsAs(const std::vector<float>& values, const Shape& shape) {
    Tensor tensor = Tensor::empty(shape, DType::of<T>(), device_, *allocator_).value();
    T* data = tensor.data<T>();
    for (size_t i = 0; i < values.size(); ++i) {
      if constexpr (std::is_same_v<T, float>) {
        data[i] = values[i];
      } else if constexpr (std::is_same_v<T, float16_t>) {
        data[i] = float_to_float16(values[i]);
      } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        data[i] = float_to_bfloat16(values[i]);
      }
    }
    return tensor;
  }

  // 把一份 fp32 Tensor(fp32 解析参照计算结果)逐元素下量化为给定标量类型 T,
  // 供与 native dtype 计算结果比较(ARCH-066 同款「fp32 解析参照」套路)。
  template <typename T>
  Tensor ConvertFloatTensorTo(const Tensor& source) {
    Tensor result = Tensor::empty(source.shape(), DType::of<T>(), device_, *allocator_).value();
    const float* src_data = static_cast<const float*>(source.raw_data());
    T* dst_data = result.data<T>();
    for (int64_t i = 0; i < source.numel(); ++i) {
      if constexpr (std::is_same_v<T, float>) {
        dst_data[i] = src_data[i];
      } else if constexpr (std::is_same_v<T, float16_t>) {
        dst_data[i] = float_to_float16(src_data[i]);
      } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        dst_data[i] = float_to_bfloat16(src_data[i]);
      }
    }
    return result;
  }

  // fp16/bf16 SGD 单步数值核心断言(BUILD-011「fp16/bf16 不与数值微分直接比,
  // 经 fp32 解析参照验证」专款——本用例不是梯度而是 SGD 更新值,但同一方法论
  // 适用:build_sgd_update_graph 内部对目标 dtype 以量化常量(-learning_rate)
  // 与逐步舍入计算 mul/add,与"手算 p - lr·g 后再整体下量化"存在细微量化路径
  // 差异,不能假定精确相等,故构造同结构的 fp32 参照图算出结果、逐元素下
  // 量化到 T 作为参照,与 native dtype 图结果按 default_tolerance(T 对应
  // dtype)比较——两侧均是"解析计算"结果(非数值微分),不适用数值微分专款
  // 的放宽一档,套用各自 dtype 的常规容差表)。取值均为 fp16/bf16 皆可精确
  // 表示的二进制小数(整数、二分之一、四分之一),避免额外舍入误差混入判定,
  // 与 test_autograd.cpp::ReluGradientFp16MatchesFp32AnalyticReference 同
  // 取值风格。
  template <typename T>
  void RunSgdSingleStepMatchesFp32AnalyticReference() {
    const std::vector<float> param_values{1.0F, 2.0F, -3.0F, 0.5F, -0.5F, 4.0F};
    const std::vector<float> grad_values{0.25F, -0.5F, 0.25F, 1.0F, -1.0F, 2.0F};
    constexpr double kLearningRate = 0.1;
    const Shape shape({2, 3});
    const DTypeCode target_code = DType::of<T>().code();

    // ①fp32 解析参照:同结构 build_sgd_update_graph(dtype=float32)→
    // compile("cpu") → run,结果逐元素下量化到 T。
    const TensorType fp32_param_type = MakeType(DType::of<float>(), {2, 3});
    const std::vector<TensorType> fp32_param_types{fp32_param_type};
    const Result<Graph> fp32_update_graph = build_sgd_update_graph(fp32_param_types, kLearningRate);
    ASSERT_TRUE(fp32_update_graph.is_ok()) << fp32_update_graph.status().message();
    const Result<std::shared_ptr<Executable>> fp32_executable = frame::runtime::compile(
        fp32_update_graph.value(), frame::kCpuBackendName, CompileOptions{});
    ASSERT_TRUE(fp32_executable.is_ok()) << fp32_executable.status().message();
    Tensor fp32_param = MakeTensorFromFloatsAs<float>(param_values, shape);
    Tensor fp32_grad = MakeTensorFromFloatsAs<float>(grad_values, shape);
    std::vector<Tensor> fp32_inputs{fp32_param, fp32_grad};
    const Result<std::vector<Tensor>> fp32_outputs = frame::runtime::run_with_allocated_outputs(
        *fp32_executable.value(), frame::kCpuBackendName, fp32_inputs);
    ASSERT_TRUE(fp32_outputs.is_ok()) << fp32_outputs.status().message();
    ASSERT_EQ(fp32_outputs.value().size(), 1u);
    const Tensor expected = ConvertFloatTensorTo<T>(fp32_outputs.value()[0]);

    // ②native dtype 图:dtype=T,同一组(量化后的)参数/梯度值。
    const TensorType native_param_type = MakeType(DType::of<T>(), {2, 3});
    const std::vector<TensorType> native_param_types{native_param_type};
    const Result<Graph> native_update_graph =
        build_sgd_update_graph(native_param_types, kLearningRate);
    ASSERT_TRUE(native_update_graph.is_ok()) << native_update_graph.status().message();
    const Result<std::shared_ptr<Executable>> native_executable = frame::runtime::compile(
        native_update_graph.value(), frame::kCpuBackendName, CompileOptions{});
    ASSERT_TRUE(native_executable.is_ok()) << native_executable.status().message();
    Tensor native_param = MakeTensorFromFloatsAs<T>(param_values, shape);
    Tensor native_grad = MakeTensorFromFloatsAs<T>(grad_values, shape);
    std::vector<Tensor> native_inputs{native_param, native_grad};
    const Result<std::vector<Tensor>> native_outputs = frame::runtime::run_with_allocated_outputs(
        *native_executable.value(), frame::kCpuBackendName, native_inputs);
    ASSERT_TRUE(native_outputs.is_ok()) << native_outputs.status().message();
    ASSERT_EQ(native_outputs.value().size(), 1u);

    EXPECT_TRUE(
        tensor_all_close(native_outputs.value()[0], expected, default_tolerance(target_code)));
  }

  Backend* backend_ = nullptr;
  frame::Device device_{};
  frame::hal::Allocator* allocator_ = nullptr;
};

// ---------------------------------------------------------------------------
// 1. build_sgd_update_graph 单测。
// ---------------------------------------------------------------------------

TEST_F(TrainingLoopTest, SingleParamOneStepMatchesManualComputation) {
  // 单参数 shape [2,3];手算 new_param = param - lr * grad(逐元素)。
  const TensorType param_type = MakeType(DType::of<float>(), {2, 3});
  const std::vector<TensorType> param_types{param_type};
  constexpr double kLearningRate = 0.1;

  const Result<Graph> update_graph = build_sgd_update_graph(param_types, kLearningRate);
  ASSERT_TRUE(update_graph.is_ok()) << update_graph.status().message();

  const Result<std::shared_ptr<Executable>> executable =
      frame::runtime::compile(update_graph.value(), frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  const std::vector<float> param_values{1.0F, 2.0F, -3.0F, 0.5F, -0.5F, 4.0F};
  const std::vector<float> grad_values{0.1F, -0.2F, 0.3F, 1.0F, -1.0F, 2.0F};
  Tensor param = MakeTensorFromFloats(param_values, Shape({2, 3}));
  Tensor grad = MakeTensorFromFloats(grad_values, Shape({2, 3}));
  std::vector<Tensor> inputs{param, grad};

  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCpuBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  ASSERT_EQ(outputs.value().size(), 1u);

  std::vector<float> expected(param_values.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = param_values[i] - static_cast<float>(kLearningRate) * grad_values[i];
  }
  const Tensor expected_tensor = MakeTensorFromFloats(expected, Shape({2, 3}));
  EXPECT_TRUE(tensor_all_close(outputs.value()[0], expected_tensor,
                               default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(TrainingLoopTest, SgdSingleStepFp16MatchesFp32AnalyticReference) {
  RunSgdSingleStepMatchesFp32AnalyticReference<float16_t>();
}

TEST_F(TrainingLoopTest, SgdSingleStepBf16MatchesFp32AnalyticReference) {
  RunSgdSingleStepMatchesFp32AnalyticReference<bfloat16_t>();
}

TEST(BuildSgdUpdateGraphErrorPathTest, RejectsEmptyParamTypes) {
  const std::vector<TensorType> empty_param_types;
  const Result<Graph> result = build_sgd_update_graph(empty_param_types, 0.1);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("param_types must not be empty"),
            std::string_view::npos);
}

TEST(BuildSgdUpdateGraphErrorPathTest, RejectsNonFiniteLearningRate) {
  const TensorType param_type = MakeType(DType::of<float>(), {2, 2});
  const std::vector<TensorType> param_types{param_type};

  const Result<Graph> nan_result =
      build_sgd_update_graph(param_types, std::numeric_limits<double>::quiet_NaN());
  ASSERT_FALSE(nan_result.is_ok());
  EXPECT_EQ(nan_result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(nan_result.status().message().find("learning_rate must be finite"),
            std::string_view::npos);

  const Result<Graph> inf_result =
      build_sgd_update_graph(param_types, std::numeric_limits<double>::infinity());
  ASSERT_FALSE(inf_result.is_ok());
  EXPECT_EQ(inf_result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(inf_result.status().message().find("learning_rate must be finite"),
            std::string_view::npos);
}

// int32/int64 已于 M22(批4 T3,决议点A)扩入 ops::is_constant_dtype_supported
// 白名单(本函数据此白名单校验 param dtype),不再是拒绝用例;int8 仍在
// 白名单外。
TEST(BuildSgdUpdateGraphErrorPathTest, RejectsDtypeOutsideWhitelist) {
  const TensorType param_type = MakeType(DType::of<int8_t>(), {2, 2});
  const std::vector<TensorType> param_types{param_type};
  const Result<Graph> result = build_sgd_update_graph(param_types, 0.1);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("param_types[0] has unsupported dtype"),
            std::string_view::npos);
  EXPECT_NE(result.status().message().find("v0 supports float32/float16/bfloat16 only"),
            std::string_view::npos);
}

// ---------------------------------------------------------------------------
// 2. 端到端 MLP 训练收敛用例。
// ---------------------------------------------------------------------------

// 网络维度(小规模,加速测试执行):x[8,4] -> matmul(W1[4,8]) -> add(b1[8,8])
// -> relu -> matmul(W2[8,1]) -> mse_loss(., target[8,1])。b1 取 [8,8]
// (与 matmul(x,W1) 输出同形)是本文件头注释所述 v0 无广播取舍的具体实例。
// 维度常量与图构造函数 BuildMlpForwardGraph() 已提取至共享头
// mlp_forward_graph_helper.h(namespace frame::compiler::testing),供
// test_onnx_weights.cpp(ADR-0013 train-save-import-infer 链路用例)复用同一份
// 图结构(REUSE-002,见上方 using 声明)。

TEST_F(TrainingLoopTest, MlpTrainingLoopConvergesWithSingleCompilePerGraph) {
  // 固定种子(std::mt19937,seed=20260713)——按下方注释顺序依次抽取:①x
  // ②W_true(仅用于生成 target,不进图)③W1 初值④b1 初值⑤W2 初值。target =
  // x·W_true(host 计算,精确线性可拟合目标,不加噪声——kNumSteps/kLearningRate
  // 已足以驱动收敛,噪声只会增加断言波动风险,BUILD-011「解析梯度 ≡ 数值微分
  // 校验」专款不适用本用例,此处不套用该口径,仅为一般收敛判据)。b1 初值取
  // 正偏移(uniform[0.5,1.0))而非居中于 0:matmul(x,W1) 项在本数据/权重量级下
  // 远小于该偏移,使各隐藏单元在训练初期大概率落在 relu 的恒等分支(x>0),
  // 避免"死 relu"(x<=0 处梯度恒 0,该隐藏单元永不再更新)导致收敛不稳定——
  // 训练过程中 W1/b1 仍可自由演化到负值区间,该初始化只影响起点,不改变
  // v0 no-broadcast MLP 结构本身。
  // 常量种子是刻意选择(测试可复现性,非 bug):
  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 rng(20260713U);
  std::uniform_real_distribution<float> data_dist(-1.0F, 1.0F);
  std::uniform_real_distribution<float> small_weight_dist(-0.1F, 0.1F);
  std::uniform_real_distribution<float> bias_dist(0.5F, 1.0F);

  std::vector<float> x_values(static_cast<size_t>(kMlpBatchSize * kMlpInputDim));
  for (float& v : x_values) v = data_dist(rng);

  std::vector<float> w_true_values(static_cast<size_t>(kMlpInputDim * kMlpOutputDim));
  for (float& v : w_true_values) v = data_dist(rng);

  std::vector<float> w1_values(static_cast<size_t>(kMlpInputDim * kMlpHiddenDim));
  for (float& v : w1_values) v = small_weight_dist(rng);

  std::vector<float> b1_values(static_cast<size_t>(kMlpBatchSize * kMlpHiddenDim));
  for (float& v : b1_values) v = bias_dist(rng);

  std::vector<float> w2_values(static_cast<size_t>(kMlpHiddenDim * kMlpOutputDim));
  for (float& v : w2_values) v = small_weight_dist(rng);

  // target = x @ W_true(host 三重循环,不经 IR)。
  std::vector<float> target_values(static_cast<size_t>(kMlpBatchSize * kMlpOutputDim), 0.0F);
  for (int64_t i = 0; i < kMlpBatchSize; ++i) {
    for (int64_t j = 0; j < kMlpOutputDim; ++j) {
      float acc = 0.0F;
      for (int64_t k = 0; k < kMlpInputDim; ++k) {
        acc += x_values[static_cast<size_t>(i * kMlpInputDim + k)] *
               w_true_values[static_cast<size_t>(k * kMlpOutputDim + j)];
      }
      target_values[static_cast<size_t>(i * kMlpOutputDim + j)] = acc;
    }
  }

  Tensor x = MakeTensorFromFloats(x_values, Shape({kMlpBatchSize, kMlpInputDim}));
  Tensor target = MakeTensorFromFloats(target_values, Shape({kMlpBatchSize, kMlpOutputDim}));
  Tensor w1 = MakeTensorFromFloats(w1_values, Shape({kMlpInputDim, kMlpHiddenDim}));
  Tensor b1 = MakeTensorFromFloats(b1_values, Shape({kMlpBatchSize, kMlpHiddenDim}));
  Tensor w2 = MakeTensorFromFloats(w2_values, Shape({kMlpHiddenDim, kMlpOutputDim}));

  // ①前向图 + build_backward_graph(wrt=w1,b1,w2)→ verify → compile("cpu")
  // 一次(循环外,shared_ptr 复用,结构性保证零重编译)。
  const Graph forward = BuildMlpForwardGraph();
  const std::vector<int32_t> wrt{1, 2, 3};
  const Result<Graph> training = build_backward_graph(forward, 0, wrt);
  ASSERT_TRUE(training.is_ok()) << training.status().message();
  const frame::ir::OpQuery train_query = frame::ops::make_op_query();
  ASSERT_TRUE(training.value().verify(train_query).is_ok());
  const Result<std::shared_ptr<Executable>> train_executable =
      frame::runtime::compile(training.value(), frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(train_executable.is_ok()) << train_executable.status().message();

  // ②build_sgd_update_graph(param_types=[w1,b1,w2])→ verify → compile("cpu")
  // 一次(循环外)。
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

  // FallbackStats 清零(全局单例,test_fallback_chain.cpp 同款隔离手法),循环
  // 结束后校验图内涉及算子零计数——佐证全程走编译期 kernel 执行路径,未触发
  // M10 eager 回退链。
  FallbackStats::instance().reset();

  constexpr int kNumSteps = 300;
  std::vector<double> loss_history;
  loss_history.reserve(static_cast<size_t>(kNumSteps));

  for (int step = 0; step < kNumSteps; ++step) {
    std::vector<Tensor> train_inputs{x, w1, b1, w2, target};
    const Result<std::vector<Tensor>> train_outputs = frame::runtime::run_with_allocated_outputs(
        *train_executable.value(), frame::kCpuBackendName, train_inputs);
    ASSERT_TRUE(train_outputs.is_ok())
        << "step " << step << ": " << train_outputs.status().message();
    ASSERT_EQ(train_outputs.value().size(), 4u);

    const Tensor& loss_tensor = train_outputs.value()[0];
    const float loss_value = *static_cast<const float*>(loss_tensor.raw_data());
    ASSERT_TRUE(std::isfinite(loss_value)) << "loss is not finite at step " << step;
    loss_history.push_back(static_cast<double>(loss_value));

    std::vector<Tensor> update_inputs{
        w1, b1, w2, train_outputs.value()[1], train_outputs.value()[2], train_outputs.value()[3]};
    const Result<std::vector<Tensor>> update_outputs = frame::runtime::run_with_allocated_outputs(
        *update_executable.value(), frame::kCpuBackendName, update_inputs);
    ASSERT_TRUE(update_outputs.is_ok())
        << "step " << step << ": " << update_outputs.status().message();
    ASSERT_EQ(update_outputs.value().size(), 3u);

    // 参数指针轮换(autograd.md 第6章③):Tensor 是共享 Storage 的值语义句柄
    // (include/frame/core/tensor.h 头注释),重新赋值即完成"轮换到新一步参数"
    // 而不做数据拷贝。
    w1 = update_outputs.value()[0];
    b1 = update_outputs.value()[1];
    w2 = update_outputs.value()[2];
  }

  ASSERT_EQ(loss_history.size(), static_cast<size_t>(kNumSteps));
  const double initial_loss = loss_history.front();
  const double final_loss = loss_history.back();

  // 收敛断言(阈值经本机实测校准:kSeed=20260713、kNumSteps=300、
  // kLearningRate=0.05 下实测 initial_loss≈0.2123、final_loss≈5.6e-05,
  // 二者均留出充分安全边际,不是刚好卡阈值):
  // ①末值 < 首值 * 0.1(显著下降,实测比值约 2.7e-4,远优于 0.1 判据);
  // ②末值 < 绝对阈值 0.05(实测约 5.6e-05,余量约 3 个数量级)。
  EXPECT_LT(final_loss, initial_loss * 0.1)
      << "initial_loss=" << initial_loss << " final_loss=" << final_loss;
  EXPECT_LT(final_loss, 0.05) << "final_loss=" << final_loss;

  // FallbackStats 零计数(图内出现的全部算子,含融合后可能出现的
  // fused_elementwise_internal)——不静默接受任何 eager 回退。
  const std::vector<std::string_view> ops_to_check{
      "matmul",
      "add",
      "relu",
      "mse_loss",
      "mul",
      frame::ops::kConstantOpName,
      "relu_grad_internal",
      "matmul_grad_lhs_internal",
      "matmul_grad_rhs_internal",
      "mse_loss_grad_internal",
      frame::ops::kFusedElementwiseOpName,
  };
  for (const std::string_view op : ops_to_check) {
    EXPECT_EQ(FallbackStats::instance().count(op, frame::kCpuBackendName), 0)
        << "op '" << op << "' unexpectedly triggered eager fallback";
  }
}

}  // namespace
