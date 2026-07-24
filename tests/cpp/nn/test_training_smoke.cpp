// nn 构图 + 调用方自行组合 compiler::build_backward_graph +
// build_sgd_update_graph + runtime::compile("cpu") 的端到端训练冒烟
// (docs/architecture/nn-design.md §8 判定⑤佐证:证明 nn 产出的图可被既有
// 训练线正常消费,即便 nn 本身不依赖 compiler/runtime,ARCH-070)。本文件
// include compiler/runtime 头是"调用方"角色的合法用法——nn 源码
// (include/frame/nn/、src/nn/)本身不依赖它们,才是 ARCH-070 判定的对象,
// 本文件不属于 nn 源码。
//
// 网络结构与维度/种子/学习率/步数直接复用
// tests/cpp/compiler/test_training_loop.cpp::
// MlpTrainingLoopConvergesWithSingleCompilePerGraph 已实测校准的收敛配置
// (x[8,4] -> Linear(w0[4,8],b0[8,8],with_bias=true) -> relu ->
// Linear(w2[8,1],with_bias=false) -> mse_loss(.,target[8,1]),seed=20260713,
// lr=0.05,300 步;target=x·W_true 的精确线性可拟合目标),仅将前向图的构造
// 方式从"手工 create_node_with_inferred_types 调用序列"换成"nn::Sequential +
// nn::MseLoss 构图",其余数值配方原样沿用,避免重新试凑一组新种子/初始化
// 范围引入不必要的收敛不确定性;b0=[batch,hidden] 是 v0 无广播下的 bias 建模
// 取舍,论证见 test_training_loop.cpp 文件头注释,本文件不重复。
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include <frame/compiler/autograd.h>
#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/data/dataloader.h>
#include <frame/data/dataset.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/executable.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/nn/layers.h>
#include <frame/nn/module.h>
#include <frame/ops/op_registry.h>
#include <frame/runtime/compile.h>

namespace {

using frame::cpu_device;
using frame::DType;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::Tensor;
using frame::compiler::build_backward_graph;
using frame::compiler::build_sgd_update_graph;
using frame::hal::Backend;
using frame::hal::BackendRegistry;
using frame::hal::CompileOptions;
using frame::hal::Executable;
using frame::ir::Graph;
using frame::ir::OpQuery;
using frame::ir::TensorType;
using frame::ir::Value;
using frame::nn::add_parameter_inputs;
using frame::nn::Linear;
using frame::nn::Module;
using frame::nn::MseLoss;
using frame::nn::ParamSpec;
using frame::nn::Relu;
using frame::nn::Sequential;
using frame::ops::make_op_query;

constexpr int64_t kBatch = 8;
constexpr int64_t kInDim = 4;
constexpr int64_t kHiddenDim = 8;
constexpr int64_t kOutDim = 1;

TensorType MakeCpuTensorType(std::vector<int64_t> dims) {
  TensorType type;
  type.dtype = DType::of<float>();
  type.shape = Shape(std::move(dims));
  type.device = cpu_device();
  return type;
}

// fixture:取 cpu 后端真实 Allocator(经 BackendRegistry),同
// tests/cpp/compiler/test_training_loop.cpp::TrainingLoopTest 同思路。
class NnTrainingSmokeTest : public ::testing::Test {
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

TEST_F(NnTrainingSmokeTest, NnComposedGraphConvergesUnderCallerOwnedBackwardAndSgdGraphs) {
  // --- ①nn 构图:Sequential(Linear("0",with_bias),Relu("1"),
  // Linear("2",no_bias)) + MseLoss("loss")。图输入序契约(调用方自定,
  // ARCH-074 职责边界裁定):[x, mlp.0.weight, mlp.0.bias, mlp.2.weight,
  // target]。---
  Graph forward("nn_training_smoke");
  Value* x = forward.add_graph_input(MakeCpuTensorType({kBatch, kInDim})).value();

  const Module model = Sequential(
      "mlp",
      {Linear("0", kBatch, kInDim, kHiddenDim, /*with_bias=*/true, DType::of<float>()), Relu("1"),
       Linear("2", kBatch, kHiddenDim, kOutDim, /*with_bias=*/false, DType::of<float>())});
  const std::vector<ParamSpec> param_specs = model.parameters();
  ASSERT_EQ(param_specs.size(), 3u);  // mlp.0.weight, mlp.0.bias, mlp.2.weight

  const Result<std::vector<Value*>> param_inputs = add_parameter_inputs(forward, param_specs);
  ASSERT_TRUE(param_inputs.is_ok()) << param_inputs.status().message();

  const Result<std::vector<Value*>> forward_outputs =
      model.build(forward, std::vector<Value*>{x}, param_inputs.value());
  ASSERT_TRUE(forward_outputs.is_ok()) << forward_outputs.status().message();
  ASSERT_EQ(forward_outputs.value().size(), 1u);
  Value* prediction = forward_outputs.value()[0];

  Value* target = forward.add_graph_input(MakeCpuTensorType({kBatch, kOutDim})).value();

  const Module loss_module = MseLoss("loss");
  const Result<std::vector<Value*>> loss_outputs =
      loss_module.build(forward, std::vector<Value*>{prediction, target}, std::vector<Value*>{});
  ASSERT_TRUE(loss_outputs.is_ok()) << loss_outputs.status().message();
  ASSERT_EQ(loss_outputs.value().size(), 1u);
  ASSERT_TRUE(forward.mark_output(loss_outputs.value()[0]).is_ok());

  const OpQuery query = make_op_query();
  const Status forward_verify_status = forward.verify(query);
  ASSERT_TRUE(forward_verify_status.is_ok()) << forward_verify_status.message();
  ASSERT_EQ(forward.inputs().size(), 5u);  // [x, w0, b0, w2, target]

  // --- ②调用方(本测试)组合 build_backward_graph + build_sgd_update_graph
  // (nn 源码不依赖 compiler/runtime;本文件是调用方,不是 nn 实现的一部分,
  // ARCH-070 判定⑤)。---
  const std::vector<int32_t> wrt{1, 2, 3};  // mlp.0.weight/bias、mlp.2.weight 的图输入下标
  const Result<Graph> training = build_backward_graph(forward, /*loss_output_index=*/0, wrt);
  ASSERT_TRUE(training.is_ok()) << training.status().message();
  const Status training_verify_status = training.value().verify(query);
  ASSERT_TRUE(training_verify_status.is_ok()) << training_verify_status.message();
  const Result<std::shared_ptr<Executable>> train_executable =
      frame::runtime::compile(training.value(), frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(train_executable.is_ok()) << train_executable.status().message();

  const std::vector<TensorType> param_types{param_specs[0].type, param_specs[1].type,
                                            param_specs[2].type};
  constexpr double kLearningRate = 0.05;
  const Result<Graph> update = build_sgd_update_graph(param_types, kLearningRate);
  ASSERT_TRUE(update.is_ok()) << update.status().message();
  const Status update_verify_status = update.value().verify(query);
  ASSERT_TRUE(update_verify_status.is_ok()) << update_verify_status.message();
  const Result<std::shared_ptr<Executable>> update_executable =
      frame::runtime::compile(update.value(), frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(update_executable.is_ok()) << update_executable.status().message();

  // --- ③固定种子生成数据/初值(数值配方与 test_training_loop.cpp 同款,
  // 见本文件头注释)。---
  // 常量种子是刻意选择(测试可复现性,非 bug),与 test_training_loop.cpp 同款
  // 抑制方式:
  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 rng(20260713U);
  std::uniform_real_distribution<float> data_dist(-1.0F, 1.0F);
  std::uniform_real_distribution<float> small_weight_dist(-0.1F, 0.1F);
  std::uniform_real_distribution<float> bias_dist(0.5F, 1.0F);

  std::vector<float> x_values(static_cast<size_t>(kBatch * kInDim));
  for (float& v : x_values) v = data_dist(rng);

  std::vector<float> w_true_values(static_cast<size_t>(kInDim * kOutDim));
  for (float& v : w_true_values) v = data_dist(rng);

  std::vector<float> w0_values(static_cast<size_t>(kInDim * kHiddenDim));
  for (float& v : w0_values) v = small_weight_dist(rng);

  std::vector<float> b0_values(static_cast<size_t>(kBatch * kHiddenDim));
  for (float& v : b0_values) v = bias_dist(rng);

  std::vector<float> w2_values(static_cast<size_t>(kHiddenDim * kOutDim));
  for (float& v : w2_values) v = small_weight_dist(rng);

  // target = x @ W_true(host 三重循环,不经 IR)。
  std::vector<float> target_values(static_cast<size_t>(kBatch * kOutDim), 0.0F);
  for (int64_t i = 0; i < kBatch; ++i) {
    for (int64_t j = 0; j < kOutDim; ++j) {
      float acc = 0.0F;
      for (int64_t k = 0; k < kInDim; ++k) {
        acc += x_values[static_cast<size_t>(i * kInDim + k)] *
               w_true_values[static_cast<size_t>(k * kOutDim + j)];
      }
      target_values[static_cast<size_t>(i * kOutDim + j)] = acc;
    }
  }

  Tensor x_tensor = MakeTensorFromFloats(x_values, Shape({kBatch, kInDim}));
  Tensor target_tensor = MakeTensorFromFloats(target_values, Shape({kBatch, kOutDim}));
  Tensor w0 = MakeTensorFromFloats(w0_values, Shape({kInDim, kHiddenDim}));
  Tensor b0 = MakeTensorFromFloats(b0_values, Shape({kBatch, kHiddenDim}));
  Tensor w2 = MakeTensorFromFloats(w2_values, Shape({kHiddenDim, kOutDim}));

  constexpr int kNumSteps = 300;
  std::vector<double> loss_history;
  loss_history.reserve(static_cast<size_t>(kNumSteps));

  for (int step = 0; step < kNumSteps; ++step) {
    std::vector<Tensor> train_inputs{x_tensor, w0, b0, w2, target_tensor};
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
        w0, b0, w2, train_outputs.value()[1], train_outputs.value()[2], train_outputs.value()[3]};
    const Result<std::vector<Tensor>> update_outputs = frame::runtime::run_with_allocated_outputs(
        *update_executable.value(), frame::kCpuBackendName, update_inputs);
    ASSERT_TRUE(update_outputs.is_ok())
        << "step " << step << ": " << update_outputs.status().message();
    ASSERT_EQ(update_outputs.value().size(), 3u);

    // 参数指针轮换(Tensor 是共享 Storage 的值语义句柄,重新赋值即完成"轮换到
    // 新一步参数"而不做数据拷贝,同 test_training_loop.cpp 同款手法)。
    w0 = update_outputs.value()[0];
    b0 = update_outputs.value()[1];
    w2 = update_outputs.value()[2];
  }

  ASSERT_EQ(loss_history.size(), static_cast<size_t>(kNumSteps));
  const double initial_loss = loss_history.front();
  const double final_loss = loss_history.back();

  // 收敛断言口径与 test_training_loop.cpp 同款(同一组数值配方在该文件已实测
  // 校准,本文件复用同一配方,预期收敛趋势一致):①末值显著低于首值;②末值
  // 低于绝对阈值。阈值留有安全边际,不追求与该文件实测值位对位复现——nn 构图
  // 路径与手工构图路径产生的 IR 节点集合结构等价但非同一份图对象,浮点求值
  // 顺序理论上可能有极细微差异,不强行断言逐位相等。
  EXPECT_LT(final_loss, initial_loss * 0.1)
      << "initial_loss=" << initial_loss << " final_loss=" << final_loss;
  EXPECT_LT(final_loss, 0.1) << "final_loss=" << final_loss;
}

// M20 收口示范(批2 Task 6):训练数据改经 frame::data::DataLoader 喂批——
// 16 样本数据集,batch_size=kBatch=8、shuffle+固定种子、drop_last,每 epoch
// 2 个小批;图形状与上一用例完全同款(batch 维=8),仅数据供给方式不同。
// 本文件是调用方角色:nn 构图 + data 喂批 + compiler/runtime 训练线组合。
TEST_F(NnTrainingSmokeTest, DataLoaderFedMiniBatchTrainingConverges) {
  Graph forward("nn_dataloader_training");
  Value* x = forward.add_graph_input(MakeCpuTensorType({kBatch, kInDim})).value();
  const Module model = Sequential(
      "mlp",
      {Linear("0", kBatch, kInDim, kHiddenDim, /*with_bias=*/true, DType::of<float>()), Relu("1"),
       Linear("2", kBatch, kHiddenDim, kOutDim, /*with_bias=*/false, DType::of<float>())});
  const std::vector<ParamSpec> param_specs = model.parameters();
  const Result<std::vector<Value*>> param_inputs = add_parameter_inputs(forward, param_specs);
  ASSERT_TRUE(param_inputs.is_ok()) << param_inputs.status().message();
  const Result<std::vector<Value*>> forward_outputs =
      model.build(forward, std::vector<Value*>{x}, param_inputs.value());
  ASSERT_TRUE(forward_outputs.is_ok()) << forward_outputs.status().message();
  Value* target = forward.add_graph_input(MakeCpuTensorType({kBatch, kOutDim})).value();
  const Result<std::vector<Value*>> loss_outputs = MseLoss("loss").build(
      forward, std::vector<Value*>{forward_outputs.value()[0], target}, std::vector<Value*>{});
  ASSERT_TRUE(loss_outputs.is_ok()) << loss_outputs.status().message();
  ASSERT_TRUE(forward.mark_output(loss_outputs.value()[0]).is_ok());

  const std::vector<int32_t> wrt{1, 2, 3};
  const Result<Graph> training = build_backward_graph(forward, /*loss_output_index=*/0, wrt);
  ASSERT_TRUE(training.is_ok()) << training.status().message();
  const Result<std::shared_ptr<Executable>> train_executable =
      frame::runtime::compile(training.value(), frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(train_executable.is_ok()) << train_executable.status().message();
  const std::vector<TensorType> param_types{param_specs[0].type, param_specs[1].type,
                                            param_specs[2].type};
  const Result<Graph> update = build_sgd_update_graph(param_types, /*learning_rate=*/0.05);
  ASSERT_TRUE(update.is_ok()) << update.status().message();
  const Result<std::shared_ptr<Executable>> update_executable =
      frame::runtime::compile(update.value(), frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(update_executable.is_ok()) << update_executable.status().message();

  // 数据集:16 样本,target = x @ W_true(与上一用例同款教师配方,种子同款)。
  constexpr int64_t kNumSamples = 16;
  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  std::mt19937 rng(20260713U);
  std::uniform_real_distribution<float> data_dist(-1.0F, 1.0F);
  std::uniform_real_distribution<float> small_weight_dist(-0.1F, 0.1F);
  std::uniform_real_distribution<float> bias_dist(0.5F, 1.0F);
  std::vector<float> features(static_cast<size_t>(kNumSamples * kInDim));
  for (float& v : features) v = data_dist(rng);
  std::vector<float> w_true(static_cast<size_t>(kInDim * kOutDim));
  for (float& v : w_true) v = data_dist(rng);
  std::vector<float> targets(static_cast<size_t>(kNumSamples * kOutDim), 0.0F);
  for (int64_t i = 0; i < kNumSamples; ++i) {
    for (int64_t j = 0; j < kOutDim; ++j) {
      float acc = 0.0F;
      for (int64_t k = 0; k < kInDim; ++k) {
        acc += features[static_cast<size_t>(i * kInDim + k)] *
               w_true[static_cast<size_t>(k * kOutDim + j)];
      }
      targets[static_cast<size_t>(i * kOutDim + j)] = acc;
    }
  }
  std::vector<float> w0_values(static_cast<size_t>(kInDim * kHiddenDim));
  for (float& v : w0_values) v = small_weight_dist(rng);
  std::vector<float> b0_values(static_cast<size_t>(kBatch * kHiddenDim));
  for (float& v : b0_values) v = bias_dist(rng);
  std::vector<float> w2_values(static_cast<size_t>(kHiddenDim * kOutDim));
  for (float& v : w2_values) v = small_weight_dist(rng);
  Tensor w0 = MakeTensorFromFloats(w0_values, Shape({kInDim, kHiddenDim}));
  Tensor b0 = MakeTensorFromFloats(b0_values, Shape({kBatch, kHiddenDim}));
  Tensor w2 = MakeTensorFromFloats(w2_values, Shape({kHiddenDim, kOutDim}));

  const Result<frame::data::TensorDataset> dataset = frame::data::TensorDataset::create(
      {MakeTensorFromFloats(features, Shape({kNumSamples, kInDim})),
       MakeTensorFromFloats(targets, Shape({kNumSamples, kOutDim}))});
  ASSERT_TRUE(dataset.is_ok()) << dataset.status().message();
  frame::data::DataLoaderOptions options;
  options.batch_size = kBatch;
  options.shuffle = true;
  options.seed = 20260713U;
  options.drop_last = true;
  Result<frame::data::DataLoader> loader =
      frame::data::DataLoader::create(dataset.value(), options);
  ASSERT_TRUE(loader.is_ok()) << loader.status().message();

  constexpr int kNumEpochs = 150;  // 每 epoch 2 批,总步数与上一用例同量级
  double initial_loss = -1.0;
  double final_loss = -1.0;
  for (int epoch = 0; epoch < kNumEpochs; ++epoch) {
    while (true) {
      Result<std::optional<std::vector<Tensor>>> batch = loader.value().next(*allocator_);
      ASSERT_TRUE(batch.is_ok()) << batch.status().message();
      // 同一 optional 对象上判空后经 * 解引用(bugprone-unchecked-optional-access
      // 的数据流追踪不跨两次 .value() 调用,绑定引用后判空是其可识别形态)。
      std::optional<std::vector<Tensor>>& maybe_columns = batch.value();
      if (!maybe_columns.has_value()) break;  // epoch 尽头哨兵,进入下一 epoch
      std::vector<Tensor>& columns = *maybe_columns;
      std::vector<Tensor> train_inputs{columns[0], w0, b0, w2, columns[1]};
      const Result<std::vector<Tensor>> train_outputs = frame::runtime::run_with_allocated_outputs(
          *train_executable.value(), frame::kCpuBackendName, train_inputs);
      ASSERT_TRUE(train_outputs.is_ok()) << train_outputs.status().message();
      const float loss_value = *static_cast<const float*>(train_outputs.value()[0].raw_data());
      ASSERT_TRUE(std::isfinite(loss_value));
      if (initial_loss < 0.0) initial_loss = static_cast<double>(loss_value);
      final_loss = static_cast<double>(loss_value);
      std::vector<Tensor> update_inputs{
          w0, b0, w2, train_outputs.value()[1], train_outputs.value()[2], train_outputs.value()[3]};
      const Result<std::vector<Tensor>> update_outputs = frame::runtime::run_with_allocated_outputs(
          *update_executable.value(), frame::kCpuBackendName, update_inputs);
      ASSERT_TRUE(update_outputs.is_ok()) << update_outputs.status().message();
      w0 = update_outputs.value()[0];
      b0 = update_outputs.value()[1];
      w2 = update_outputs.value()[2];
    }
  }

  // 小批 SGD 在精确线性可拟合目标上应显著收敛;阈值同上一用例口径留裕度。
  EXPECT_LT(final_loss, initial_loss * 0.1)
      << "initial_loss=" << initial_loss << " final_loss=" << final_loss;
  EXPECT_LT(final_loss, 0.1) << "final_loss=" << final_loss;
}

}  // namespace
