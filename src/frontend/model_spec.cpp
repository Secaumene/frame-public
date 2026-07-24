// validate():逐条执行 docs/architecture/frontend-dsl.md 第 3 节 FE-002~005
// 与第 1 节的结构性约束(FE-001 的 schema_version 校验属 JSON 层职责,不在
// 本库范围)。校验逻辑按小节拆分为若干私有 helper(validate_hyperparameters/
// validate_layer/check_tensor_data/validate_data),各 helper 认知复杂度均
// 远低于阈值;validate() 本身仅编排调用顺序,行为与拆分前逐字节一致。

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <frame/core/shape.h>
#include <frame/frontend/model_spec.h>

namespace frame::frontend {
namespace {

// 逐维正值校验,field_name 用于拼接错误消息(含违例字段名,LANG-005)。
Status CheckPositiveDims(const std::vector<int64_t>& dims, const std::string& field_name) {
  for (const int64_t d : dims) {
    if (d <= 0) {
      return Status::make(ErrorCode::kInvalidArgument, field_name + " must have all dims > 0");
    }
  }
  return Status::ok();
}

// 第 1 节结构性数值约束:batch/数据输入数量/学习率/训练步数/日志间隔。
Status validate_hyperparameters(const ModelSpec& spec) {
  if (spec.batch <= 0) {
    return Status::make(ErrorCode::kInvalidArgument, "model.batch must be positive");
  }
  if (spec.inputs.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "ModelSpec must declare exactly one data input (v0 constraint), got " +
                            std::to_string(spec.inputs.size()));
  }
  if (!(spec.optimizer.learning_rate > 0.0)) {
    return Status::make(ErrorCode::kInvalidArgument, "optimizer.learning_rate must be positive");
  }
  if (spec.training.steps <= 0) {
    return Status::make(ErrorCode::kInvalidArgument, "training.steps must be positive");
  }
  if (spec.training.log_every < 0) {
    return Status::make(ErrorCode::kInvalidArgument, "training.log_every must be non-negative");
  }
  return Status::ok();
}

// 校验并处理单个 layer:名字唯一非空(FE-002)、输入引用命中已声明的输入或
// 前序层、形状链一致(FE-003,含 weight_shape 秩 2 与 bias_shape 全形状)、
// activation 枚举白名单(FE-004)。成功时把该层输出形状写入 shapes_by_name、
// 层名写入 layer_names,供 validate() 主循环与收尾阶段(loss 引用/形状校验)
// 复用。
Status validate_layer(const LinearLayerSpec& layer, int64_t batch,
                      std::unordered_set<std::string>& defined_names,
                      std::unordered_set<std::string>& layer_names,
                      std::unordered_map<std::string, std::vector<int64_t>>& shapes_by_name) {
  if (layer.name.empty()) {
    return Status::make(ErrorCode::kInvalidArgument, "layers[].name must not be empty");
  }
  if (!defined_names.insert(layer.name).second) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "duplicate name across inputs/layers: '" + layer.name + "'");
  }
  if (!shapes_by_name.contains(layer.input)) {
    return Status::make(ErrorCode::kInvalidArgument, "layers['" + layer.name + "'].input '" +
                                                         layer.input +
                                                         "' does not reference a declared "
                                                         "input or a preceding layer");
  }

  if (layer.weight_shape.size() != 2) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "layers['" + layer.name + "'].weight_shape must be rank 2 [in, out]");
  }
  Status weight_dims_status =
      CheckPositiveDims(layer.weight_shape, "layers['" + layer.name + "'].weight_shape");
  if (!weight_dims_status.is_ok()) {
    return weight_dims_status;
  }

  const std::vector<int64_t>& layer_input_shape = shapes_by_name.at(layer.input);
  if (layer_input_shape.back() != layer.weight_shape[0]) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "layers['" + layer.name + "'] input's last dim does not match weight_shape[0]");
  }

  const std::vector<int64_t> output_shape{batch, layer.weight_shape[1]};
  if (layer.bias_shape.has_value() && *layer.bias_shape != output_shape) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "layers['" + layer.name + "'].bias_shape must equal [batch, weight_shape[1]]");
  }

  // FE-004:枚举白名单(loss.kind/optimizer.kind/model.dtype/layers[].kind
  // 在 v0 ModelSpec 中已由结构体设计固定为唯一合法取值,不再需要运行时校验;
  // 仅 activation 有两种合法取值需要防御 static_cast 等构造出的越界位模式)。
  if (layer.activation != Activation::kNone && layer.activation != Activation::kRelu) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "layers['" + layer.name + "'].activation has an unrecognized value");
  }

  layer_names.insert(layer.name);
  shapes_by_name.emplace(layer.name, output_shape);
  return Status::ok();
}

// FE-005 单个张量条目(数据输入或 target)完整性:必须存在于 spec.data;
// kind == kInline 时元素数须等于 shape numel;kind == kUniformSeeded 时
// lo < hi。
Status check_tensor_data(const ModelSpec& spec, const std::string& key,
                         const std::vector<int64_t>& shape) {
  const auto it = spec.data.find(key);
  if (it == spec.data.end()) {
    return Status::make(ErrorCode::kInvalidArgument, "data['" + key + "'] is required");
  }
  const TensorDataSpec& tensor_data = it->second;
  if (tensor_data.kind == InitKind::kInline) {
    const int64_t expected_numel = Shape(shape).numel();
    if (static_cast<int64_t>(tensor_data.values.size()) != expected_numel) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "data['" + key + "'].values element count (" + std::to_string(tensor_data.values.size()) +
              ") does not match shape numel (" + std::to_string(expected_numel) + ")");
    }
  } else if (!(tensor_data.lo < tensor_data.hi)) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "data['" + key + "'] uniform range must satisfy lo < hi");
  }
  return Status::ok();
}

// FE-005:data 段完整性——数据输入与 target 张量(经 check_tensor_data)、
// data.params 两档取值范围(weight_range/bias_range 均须 lo < hi)。
Status validate_data(const ModelSpec& spec, const InputSpec& input) {
  Status input_data_status = check_tensor_data(spec, input.name, input.shape);
  if (!input_data_status.is_ok()) {
    return input_data_status;
  }
  Status target_data_status = check_tensor_data(spec, "target", spec.loss.target_shape);
  if (!target_data_status.is_ok()) {
    return target_data_status;
  }

  if (!(spec.param_init.weight_lo < spec.param_init.weight_hi)) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "data.params.weight_range must satisfy lo < hi");
  }
  if (!(spec.param_init.bias_lo < spec.param_init.bias_hi)) {
    return Status::make(ErrorCode::kInvalidArgument, "data.params.bias_range must satisfy lo < hi");
  }
  return Status::ok();
}

}  // namespace

Status validate(const ModelSpec& spec) {
  Status hyperparameter_status = validate_hyperparameters(spec);
  if (!hyperparameter_status.is_ok()) {
    return hyperparameter_status;
  }

  // ---- FE-002(名字部分):唯一且非空;shapes_by_name 同步建立供下方
  //      FE-003 形状链一致校验复用。 ----
  const InputSpec& input = spec.inputs[0];
  if (input.name.empty()) {
    return Status::make(ErrorCode::kInvalidArgument, "inputs[0].name must not be empty");
  }
  std::unordered_set<std::string> defined_names{input.name};

  if (input.shape.empty()) {
    return Status::make(ErrorCode::kInvalidArgument, "inputs[0].shape must not be empty");
  }
  Status input_dims_status = CheckPositiveDims(input.shape, "inputs[0].shape");
  if (!input_dims_status.is_ok()) {
    return input_dims_status;
  }
  if (input.shape[0] != spec.batch) {
    return Status::make(ErrorCode::kInvalidArgument, "inputs[0].shape[0] must equal model.batch");
  }

  std::unordered_set<std::string> layer_names;
  std::unordered_map<std::string, std::vector<int64_t>> shapes_by_name;
  shapes_by_name.emplace(input.name, input.shape);

  // ---- FE-003:形状链一致(与名字闭包同一次遍历完成,逐层校验下沉至
  //      validate_layer)。 ----
  for (const LinearLayerSpec& layer : spec.layers) {
    Status layer_status =
        validate_layer(layer, spec.batch, defined_names, layer_names, shapes_by_name);
    if (!layer_status.is_ok()) {
      return layer_status;
    }
  }

  // ---- FE-002(续):loss.prediction 必须是某 layer 名 ----
  if (!layer_names.contains(spec.loss.prediction)) {
    return Status::make(ErrorCode::kInvalidArgument, "loss.prediction '" + spec.loss.prediction +
                                                         "' does not reference a declared layer");
  }
  // ---- FE-003(续):loss.target_shape == 末层输出形状 ----
  if (spec.loss.target_shape != shapes_by_name.at(spec.loss.prediction)) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "loss.target_shape must equal the output shape of layer '" + spec.loss.prediction + "'");
  }

  return validate_data(spec, input);
}

}  // namespace frame::frontend
