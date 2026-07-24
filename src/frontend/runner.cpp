// run_training:进程内训练执行,结构严格镜像
// tests/cpp/compiler/test_training_loop.cpp:200-323 的端到端训练循环——
// lower → build_backward_graph → build_sgd_update_graph → 两图各 verify →
// runtime::compile 各一次(循环外,shared_ptr 复用,结构性保证零重编译)→
// 循环 training.steps 步(run_with_allocated_outputs + Tensor 值语义参数
// 轮换)→ lower_to_inference_graph 编译执行一次填充 final_predictions。
// 三段各自独立的重复/嵌套逻辑(verify+compile、逐层参数张量生成、训练步循环)
// 下沉至私有 helper(CompileVerifiedGraph/BuildParamTensors/RunTrainingSteps),
// 降低 run_training 本体的认知复杂度;行为与拆分前逐字节一致。

#include <algorithm>
#include <memory>
#include <random>
#include <vector>

#include <frame/compiler/autograd.h>
#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/tensor.h>
#include <frame/frontend/lowering.h>
#include <frame/frontend/model_spec.h>
#include <frame/frontend/runner.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/executable.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/op_registry.h>
#include <frame/runtime/compile.h>

namespace frame::frontend {

using frame::hal::Allocator;
using frame::hal::Backend;
using frame::hal::BackendRegistry;
using frame::hal::CompileOptions;
using frame::hal::Executable;
using frame::ir::Graph;
using frame::ir::OpQuery;

namespace {

// 逐张量按 frontend-dsl.md 第 2 节顺序(数据输入 → target → 逐层参数,weight
// 先于 bias)生成的宿主侧数值;kind == kInline 时直接取内联值(不消耗 rng
// 状态),kind == kUniformSeeded 时从 rng 依序抽取——emitter 生成的代码必须
// 复用同一顺序约定,保证同 seed 同轨迹。
struct HostTensorValues {
  std::vector<float> input_values;
  std::vector<float> target_values;
  std::vector<std::vector<float>> layer_weight_values;
  std::vector<std::vector<float>> layer_bias_values;  // 无 bias 的层对应空 vector
};

// lo/hi 是均匀分布的下上界(数学含义互不可替换,不存在"传反了也编译通过但
// 语义错乱"之外的额外风险;两个 float 形参相邻同型,规避
// bugprone-easily-swappable-parameters,手法同 src/ir/serialization.cpp
// find_unquoted 与 src/interop/onnx_weights.cpp append_varint_field 既有
// 先例)。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::vector<float> DrawUniform(std::mt19937& rng, float lo, float hi, int64_t count) {
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> values(static_cast<size_t>(count));
  for (float& v : values) {
    v = dist(rng);
  }
  return values;
}

std::vector<float> ResolveTensorData(std::mt19937& rng, const TensorDataSpec& data, int64_t numel) {
  if (data.kind == InitKind::kInline) {
    return data.values;
  }
  return DrawUniform(rng, data.lo, data.hi, numel);
}

// spec 须已通过 validate()(由调用方保证);本函数不重复校验。
HostTensorValues GenerateHostTensorValues(const ModelSpec& spec) {
  HostTensorValues values;
  std::mt19937 rng(spec.training.seed);

  const InputSpec& input = spec.inputs[0];  // v0 恰一个数据输入
  values.input_values =
      ResolveTensorData(rng, spec.data.at(input.name), Shape(input.shape).numel());
  values.target_values =
      ResolveTensorData(rng, spec.data.at("target"), Shape(spec.loss.target_shape).numel());

  values.layer_weight_values.reserve(spec.layers.size());
  values.layer_bias_values.reserve(spec.layers.size());
  for (const LinearLayerSpec& layer : spec.layers) {
    values.layer_weight_values.push_back(DrawUniform(rng, spec.param_init.weight_lo,
                                                     spec.param_init.weight_hi,
                                                     Shape(layer.weight_shape).numel()));
    if (layer.bias_shape.has_value()) {
      values.layer_bias_values.push_back(DrawUniform(
          rng, spec.param_init.bias_lo, spec.param_init.bias_hi, Shape(*layer.bias_shape).numel()));
    } else {
      values.layer_bias_values.emplace_back();
    }
  }
  return values;
}

// Tensor::empty + data<float>() + std::copy(examples/02_graph_compile/
// main.cpp:81-122 同款手法)。
Result<Tensor> MakeFloat32Tensor(const std::vector<float>& values, const std::vector<int64_t>& dims,
                                 Device device, Allocator& allocator) {
  const Result<Tensor> tensor_result =
      Tensor::empty(Shape(dims), DType::of<float>(), device, allocator);
  if (!tensor_result.is_ok()) {
    return tensor_result.status();
  }
  Tensor tensor = tensor_result.value();
  std::copy(values.begin(), values.end(), tensor.data<float>());
  return tensor;
}

// verify(query) 后 runtime::compile 一次;training/update/inference 三张图
// 编译前均须走这一步,合并为一个 helper 避免三份同构 verify+compile 重复
// (REUSE-002)。
Result<std::shared_ptr<Executable>> CompileVerifiedGraph(const Graph& graph, const OpQuery& query,
                                                         const std::string& backend) {
  const Status verify_status = graph.verify(query);
  if (!verify_status.is_ok()) {
    return verify_status;
  }
  return runtime::compile(graph, backend, CompileOptions{});
}

// 逐层生成 weight[+bias] 参数张量(host_values 已按 frontend-dsl.md 第 2 节
// 顺序备好数值);expected_count 须等于 lowered.param_types.size(),不等视为
// 内部错误(与 lower_to_graph 的参数元信息契约不符)。
Result<std::vector<Tensor>> BuildParamTensors(const ModelSpec& spec,
                                              const HostTensorValues& host_values,
                                              size_t expected_count, Device device,
                                              Allocator& allocator) {
  std::vector<Tensor> params;
  params.reserve(expected_count);
  for (size_t i = 0; i < spec.layers.size(); ++i) {
    const LinearLayerSpec& layer = spec.layers[i];
    const Result<Tensor> weight_result = MakeFloat32Tensor(host_values.layer_weight_values[i],
                                                           layer.weight_shape, device, allocator);
    if (!weight_result.is_ok()) {
      return weight_result.status();
    }
    params.push_back(weight_result.value());

    if (layer.bias_shape.has_value()) {
      const Result<Tensor> bias_result =
          MakeFloat32Tensor(host_values.layer_bias_values[i], *layer.bias_shape, device, allocator);
      if (!bias_result.is_ok()) {
        return bias_result.status();
      }
      params.push_back(bias_result.value());
    }
  }
  if (params.size() != expected_count) {
    return Status::make(ErrorCode::kInternal,
                        "internal error: generated parameter tensor count does not match "
                        "lowered.param_types");
  }
  return params;
}

// 训练步循环结果:逐步 loss(下标 = step)+ 循环结束时的参数张量(供调用方
// 续接推理阶段)。
struct TrainingStepsResult {
  std::vector<double> loss_history;
  std::vector<Tensor> final_params;
};

// 训练循环:训练图输入序 [x, 逐层参数..., target],输出序
// [loss, grad_0..grad_{n-1}];更新图输入序 [param_0..param_{n-1},
// grad_0..grad_{n-1}],输出序 [new_param_0..new_param_{n-1}]。params 按值
// 接收初始参数、循环内逐步轮换(Tensor 是共享 Storage 的值语义句柄,重新赋值
// 即完成"轮换到新一步参数"而不做数据拷贝,autograd.md 第 5 章③),循环结束时
// 随结果一并返回。
Result<TrainingStepsResult> RunTrainingSteps(Executable& train_executable,
                                             Executable& update_executable,
                                             const std::string& backend, int32_t num_steps,
                                             const Tensor& x, const Tensor& target,
                                             std::vector<Tensor> params) {
  TrainingStepsResult result;
  result.loss_history.reserve(static_cast<size_t>(num_steps));

  for (int32_t step = 0; step < num_steps; ++step) {
    std::vector<Tensor> train_inputs;
    train_inputs.reserve(params.size() + 2);
    train_inputs.push_back(x);
    for (const Tensor& param : params) {
      train_inputs.push_back(param);
    }
    train_inputs.push_back(target);

    const Result<std::vector<Tensor>> train_outputs =
        runtime::run_with_allocated_outputs(train_executable, backend, train_inputs);
    if (!train_outputs.is_ok()) {
      return train_outputs.status();
    }
    if (train_outputs.value().size() != params.size() + 1) {
      return Status::make(ErrorCode::kInternal,
                          "training graph produced an unexpected output count");
    }

    const Tensor& loss_tensor = train_outputs.value()[0];
    const float loss_value = *static_cast<const float*>(loss_tensor.raw_data());
    result.loss_history.push_back(static_cast<double>(loss_value));

    std::vector<Tensor> update_inputs;
    update_inputs.reserve(params.size() * 2);
    for (const Tensor& param : params) {
      update_inputs.push_back(param);
    }
    for (size_t i = 0; i < params.size(); ++i) {
      update_inputs.push_back(train_outputs.value()[1 + i]);
    }

    const Result<std::vector<Tensor>> update_outputs =
        runtime::run_with_allocated_outputs(update_executable, backend, update_inputs);
    if (!update_outputs.is_ok()) {
      return update_outputs.status();
    }
    if (update_outputs.value().size() != params.size()) {
      return Status::make(ErrorCode::kInternal, "update graph produced an unexpected output count");
    }
    params = update_outputs.value();
  }

  result.final_params = std::move(params);
  return result;
}

}  // namespace

Result<RunReport> run_training(const ModelSpec& spec, const RunOptions& options) {
  const Result<LoweredModel> lowered_result = lower_to_graph(spec);
  if (!lowered_result.is_ok()) {
    return lowered_result.status();
  }
  const LoweredModel& lowered = lowered_result.value();

  const OpQuery query = ops::make_op_query();

  // ①前向图 + build_backward_graph(wrt=全体参数下标)→ verify → compile 一次
  //   (循环外)。
  const Result<Graph> training_graph = compiler::build_backward_graph(
      lowered.forward, /*loss_output_index=*/0, lowered.wrt_input_indices);
  if (!training_graph.is_ok()) {
    return training_graph.status();
  }
  const Result<std::shared_ptr<Executable>> train_executable =
      CompileVerifiedGraph(training_graph.value(), query, options.backend);
  if (!train_executable.is_ok()) {
    return train_executable.status();
  }

  // ②build_sgd_update_graph(param_types)→ verify → compile 一次(循环外)。
  const Result<Graph> update_graph =
      compiler::build_sgd_update_graph(lowered.param_types, spec.optimizer.learning_rate);
  if (!update_graph.is_ok()) {
    return update_graph.status();
  }
  const Result<std::shared_ptr<Executable>> update_executable =
      CompileVerifiedGraph(update_graph.value(), query, options.backend);
  if (!update_executable.is_ok()) {
    return update_executable.status();
  }

  // ---- 取后端 allocator,建 x/target/逐层参数张量 ----
  const Result<Backend*> backend_result = BackendRegistry::instance().get(options.backend);
  if (!backend_result.is_ok()) {
    return backend_result.status();
  }
  Backend* backend = backend_result.value();
  const Device device{options.backend, 0};
  Allocator* allocator = backend->allocator(device);
  if (allocator == nullptr) {
    return Status::make(ErrorCode::kInternal,
                        "backend '" + options.backend + "' returned a null allocator");
  }

  const HostTensorValues host_values = GenerateHostTensorValues(spec);

  const Result<Tensor> x_result =
      MakeFloat32Tensor(host_values.input_values, spec.inputs[0].shape, device, *allocator);
  if (!x_result.is_ok()) {
    return x_result.status();
  }
  const Tensor& x = x_result.value();

  const Result<Tensor> target_result =
      MakeFloat32Tensor(host_values.target_values, spec.loss.target_shape, device, *allocator);
  if (!target_result.is_ok()) {
    return target_result.status();
  }
  const Tensor& target = target_result.value();

  // params 顺序 = lowered.param_types 顺序 = 逐层 weight 先于 bias。
  Result<std::vector<Tensor>> initial_params_result =
      BuildParamTensors(spec, host_values, lowered.param_types.size(), device, *allocator);
  if (!initial_params_result.is_ok()) {
    return initial_params_result.status();
  }

  // ---- 训练循环(training.steps 步) ----
  Result<TrainingStepsResult> steps_result =
      RunTrainingSteps(*train_executable.value(), *update_executable.value(), options.backend,
                       spec.training.steps, x, target, std::move(initial_params_result.value()));
  if (!steps_result.is_ok()) {
    return steps_result.status();
  }
  TrainingStepsResult steps = std::move(steps_result.value());

  RunReport report;
  report.loss_history = std::move(steps.loss_history);
  report.final_loss = report.loss_history.empty() ? 0.0 : report.loss_history.back();
  const std::vector<Tensor>& params = steps.final_params;

  // ---- 训练完毕:lower_to_inference_graph 编译执行一次,填充 final_predictions ----
  const Result<Graph> inference_graph = lower_to_inference_graph(spec);
  if (!inference_graph.is_ok()) {
    return inference_graph.status();
  }
  const Result<std::shared_ptr<Executable>> inference_executable =
      CompileVerifiedGraph(inference_graph.value(), query, options.backend);
  if (!inference_executable.is_ok()) {
    return inference_executable.status();
  }

  std::vector<Tensor> inference_inputs;
  inference_inputs.reserve(params.size() + 1);
  inference_inputs.push_back(x);
  for (const Tensor& param : params) {
    inference_inputs.push_back(param);
  }

  const Result<std::vector<Tensor>> inference_outputs = runtime::run_with_allocated_outputs(
      *inference_executable.value(), options.backend, inference_inputs);
  if (!inference_outputs.is_ok()) {
    return inference_outputs.status();
  }
  if (inference_outputs.value().size() != 1) {
    return Status::make(ErrorCode::kInternal, "inference graph must produce exactly one output");
  }

  const Tensor& prediction_tensor = inference_outputs.value()[0];
  const float* prediction_data = static_cast<const float*>(prediction_tensor.raw_data());
  report.final_predictions.assign(prediction_data, prediction_data + prediction_tensor.numel());

  return report;
}

}  // namespace frame::frontend
