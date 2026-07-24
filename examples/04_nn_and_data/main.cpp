// =============================================================================
// 示例 04:nn 与数据。
//
// 学习目标:迭代固定形状的小批数据,并用 nn 模块构造、验证静态图。
// 前置章节:help/04-nn-and-data/README.md。
// 预期 PASS:打印 DataLoader epoch 与神经网络图验证成功的 PASS 行。
// 运行边界:DataLoader 只操作 CPU Tensor;nn::Module::build 只构图、不执行数值。
// =============================================================================

#include <cstdint>
#include <iostream>
#include <optional>
#include <utility>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/data/dataloader.h>
#include <frame/data/dataset.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/nn/layers.h>
#include <frame/nn/module.h>
#include <frame/ops/op_registry.h>

namespace {

// 构造本示例统一使用的 float32 CPU 静态类型。
frame::ir::TensorType make_example_04_tensor_type(std::vector<int64_t> dims) {
  frame::ir::TensorType type;
  type.dtype = frame::DType::of<float>();
  type.shape = frame::Shape(std::move(dims));
  type.device = frame::cpu_device();
  return type;
}

}  // namespace

int main() {
  // ---- 1. 从注册表取得 CPU 后端及其分配器。----
  const frame::Result<frame::hal::Backend*> backend_result =
      frame::hal::BackendRegistry::instance().get(frame::kCpuBackendName);
  if (!backend_result.is_ok()) {
    std::cerr << "failed to get cpu backend: " << backend_result.status().message() << "\n";
    return 1;
  }
  frame::hal::Backend* backend = backend_result.value();
  frame::hal::Allocator* allocator = backend->allocator(frame::cpu_device());
  if (allocator == nullptr) {
    std::cerr << "cpu backend returned a null allocator\n";
    return 1;
  }

  // ---- 2. 创建五条样本。----
  // features 的 axis0 是样本维,每条样本有两个特征;targets 每条样本一个值。
  const frame::Result<frame::Tensor> features_result = frame::Tensor::empty(
      frame::Shape({5, 2}), frame::DType::of<float>(), frame::cpu_device(), *allocator);
  if (!features_result.is_ok()) {
    std::cerr << "failed to allocate features: " << features_result.status().message() << "\n";
    return 1;
  }
  frame::Tensor features = features_result.value();
  const std::vector<float> feature_values{0.0F, 0.5F, 1.0F, 1.5F, 2.0F,
                                          2.5F, 3.0F, 3.5F, 4.0F, 4.5F};
  for (size_t i = 0; i < feature_values.size(); ++i) {
    features.data<float>()[i] = feature_values[i];
  }

  const frame::Result<frame::Tensor> targets_result = frame::Tensor::empty(
      frame::Shape({5, 1}), frame::DType::of<float>(), frame::cpu_device(), *allocator);
  if (!targets_result.is_ok()) {
    std::cerr << "failed to allocate targets: " << targets_result.status().message() << "\n";
    return 1;
  }
  frame::Tensor targets = targets_result.value();
  const std::vector<float> target_values{0.0F, 2.0F, 4.0F, 6.0F, 8.0F};
  for (size_t i = 0; i < target_values.size(); ++i) {
    targets.data<float>()[i] = target_values[i];
  }

  // ---- 3. 把两列张量组合为数据集。----
  // TensorDataset::create 会验证列非空、均为 CPU 张量且 axis0 长度一致。
  const frame::Result<frame::data::TensorDataset> dataset_result =
      frame::data::TensorDataset::create({features, targets});
  if (!dataset_result.is_ok()) {
    std::cerr << "failed to create dataset: " << dataset_result.status().message() << "\n";
    return 1;
  }
  if (dataset_result.value().size() != 5) {
    std::cerr << "dataset reported an unexpected sample count\n";
    return 1;
  }

  // ---- 4. 创建确定性 DataLoader。----
  // batch_size=2、shuffle=false 保证样本顺序不变;drop_last=true 会丢弃
  // 最后一条不足一个完整批次的样本,因此本 epoch 应恰有两个批次。
  frame::data::DataLoaderOptions loader_options;
  loader_options.batch_size = 2;
  loader_options.shuffle = false;
  loader_options.seed = 0;
  loader_options.drop_last = true;
  frame::Result<frame::data::DataLoader> loader_result =
      frame::data::DataLoader::create(dataset_result.value(), loader_options);
  if (!loader_result.is_ok()) {
    std::cerr << "failed to create dataloader: " << loader_result.status().message() << "\n";
    return 1;
  }
  frame::data::DataLoader loader = std::move(loader_result.value());
  if (loader.batches_per_epoch() != 2) {
    std::cerr << "dataloader reported an unexpected batch count\n";
    return 1;
  }

  // ---- 5. 消费完整 epoch 并验证批次。----
  int64_t observed_batches = 0;
  float feature_checksum = 0.0F;
  float target_checksum = 0.0F;
  while (true) {
    frame::Result<std::optional<std::vector<frame::Tensor>>> batch_result = loader.next(*allocator);
    if (!batch_result.is_ok()) {
      std::cerr << "failed to load a batch: " << batch_result.status().message() << "\n";
      return 1;
    }

    // 空 optional 是当前 epoch 的结束哨兵;DataLoader 同时推进到下一 epoch。
    const std::optional<std::vector<frame::Tensor>>& maybe_columns = batch_result.value();
    if (!maybe_columns.has_value()) break;
    const std::vector<frame::Tensor>& columns = *maybe_columns;
    if (columns.size() != 2 || !(columns[0].shape() == frame::Shape({2, 2})) ||
        !(columns[1].shape() == frame::Shape({2, 1}))) {
      std::cerr << "dataloader produced an unexpected batch structure\n";
      return 1;
    }

    // shuffle=false 时第 b 批从原数据第 b*batch_size 行开始;校验第一个
    // 特征值可同时确认批次顺序与行切片边界。
    const float expected_first_feature =
        feature_values[static_cast<size_t>(observed_batches * loader_options.batch_size * 2)];
    const float* batch_features = static_cast<const float*>(columns[0].raw_data());
    const float* batch_targets = static_cast<const float*>(columns[1].raw_data());
    if (batch_features[0] != expected_first_feature) {
      std::cerr << "dataloader produced an unexpected sample order\n";
      return 1;
    }
    for (int64_t i = 0; i < columns[0].numel(); ++i) feature_checksum += batch_features[i];
    for (int64_t i = 0; i < columns[1].numel(); ++i) target_checksum += batch_targets[i];
    ++observed_batches;
  }
  // 输入仅含整数和 0.5 的倍数,累计结果均可由 float32 精确表示。
  if (observed_batches != 2 || feature_checksum != 14.0F || target_checksum != 12.0F) {
    std::cerr << "dataloader epoch checksum validation failed\n";
    return 1;
  }

  // ---- 6. 构造一个静态批次的 MLP 前向图。----
  frame::ir::Graph graph("nn_and_data_example");
  const frame::Result<frame::ir::Value*> x_result =
      graph.add_graph_input(make_example_04_tensor_type({2, 2}));
  if (!x_result.is_ok()) {
    std::cerr << "failed to add x graph input: " << x_result.status().message() << "\n";
    return 1;
  }

  // Linear v0 尚未使用一维广播 bias。with_bias=true 时,bias 形状与该层
  // 完整输出相同:[batch,out_dim],而不是常见的 [out_dim]。因此本模型参数
  // 的稳定先序为:
  //   mlp.0.weight [2,3], mlp.0.bias [2,3],
  //   mlp.2.weight [3,1], mlp.2.bias [2,1]。
  const frame::nn::Module model = frame::nn::Sequential(
      "mlp", {frame::nn::Linear("0", /*batch=*/2, /*in_dim=*/2, /*out_dim=*/3,
                                /*with_bias=*/true, frame::DType::of<float>()),
              frame::nn::Relu("1"),
              frame::nn::Linear("2", /*batch=*/2, /*in_dim=*/3, /*out_dim=*/1,
                                /*with_bias=*/true, frame::DType::of<float>())});
  const std::vector<frame::nn::ParamSpec> parameter_specs = model.parameters();
  if (parameter_specs.size() != 4 || parameter_specs[0].name != "mlp.0.weight" ||
      parameter_specs[1].name != "mlp.0.bias" || parameter_specs[2].name != "mlp.2.weight" ||
      parameter_specs[3].name != "mlp.2.bias") {
    std::cerr << "model parameters are not in the documented preorder\n";
    return 1;
  }

  // add_parameter_inputs 按上面的 parameters() 顺序紧跟 x 添加图输入。
  // 因而此时输入顺序为 [x,w0,b0,w2,b2]。
  const frame::Result<std::vector<frame::ir::Value*>> parameter_inputs =
      frame::nn::add_parameter_inputs(graph, parameter_specs);
  if (!parameter_inputs.is_ok()) {
    std::cerr << "failed to add parameter inputs: " << parameter_inputs.status().message() << "\n";
    return 1;
  }
  const frame::Result<std::vector<frame::ir::Value*>> prediction_result = model.build(
      graph, std::vector<frame::ir::Value*>{x_result.value()}, parameter_inputs.value());
  if (!prediction_result.is_ok() || prediction_result.value().size() != 1) {
    std::cerr << "failed to build the sequential model";
    if (!prediction_result.is_ok()) std::cerr << ": " << prediction_result.status().message();
    std::cerr << "\n";
    return 1;
  }

  // target 由调用方在参数之后添加,最终图输入顺序为 [x,w0,b0,w2,b2,target]。
  const frame::Result<frame::ir::Value*> target_input_result =
      graph.add_graph_input(make_example_04_tensor_type({2, 1}));
  if (!target_input_result.is_ok()) {
    std::cerr << "failed to add target graph input: " << target_input_result.status().message()
              << "\n";
    return 1;
  }

  // MseLoss 的两个输入顺序固定为 [prediction,target],输出是标量 loss。
  const frame::Result<std::vector<frame::ir::Value*>> loss_result =
      frame::nn::MseLoss("loss").build(
          graph,
          std::vector<frame::ir::Value*>{prediction_result.value()[0], target_input_result.value()},
          std::vector<frame::ir::Value*>{});
  if (!loss_result.is_ok() || loss_result.value().size() != 1) {
    std::cerr << "failed to build mse loss";
    if (!loss_result.is_ok()) std::cerr << ": " << loss_result.status().message();
    std::cerr << "\n";
    return 1;
  }
  const frame::Status mark_status = graph.mark_output(loss_result.value()[0]);
  if (!mark_status.is_ok()) {
    std::cerr << "failed to mark loss output: " << mark_status.message() << "\n";
    return 1;
  }

  // make_op_query() 把算子注册表契约注入 Graph::verify;验证通过即说明
  // Sequential 与 MseLoss 产出的静态图在结构、schema 和 shape 上自洽。
  const frame::Status verify_status = graph.verify(frame::ops::make_op_query());
  if (!verify_status.is_ok()) {
    std::cerr << "graph verification failed: " << verify_status.message() << "\n";
    return 1;
  }

  std::cout << "PASS: dataloader epoch and neural network graph verified\n";
  return 0;
}
