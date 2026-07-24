// json_loader:JSON -> ModelSpec 结构化载入的实现。全程使用 nlohmann/json 的
// 免异常读取路径(先 is_object()/is_array()/is_number()/is_string() 等类型
// 判定,再取值),不触发 nlohmann 的类型不符异常,与全仓无异常文化一致
// (CPP-020,ADR-0018)。

#include "json_loader.h"

#include <fstream>
#include <ios>
#include <nlohmann/json.hpp>
#include <utility>
#include <vector>

namespace frame_dslc {
namespace {

using Json = nlohmann::json;
using frame::ErrorCode;
using frame::Result;
using frame::Status;
using frame::frontend::Activation;
using frame::frontend::InitKind;
using frame::frontend::InputSpec;
using frame::frontend::LinearLayerSpec;
using frame::frontend::LossSpec;
using frame::frontend::ModelSpec;
using frame::frontend::OptimizerSpec;
using frame::frontend::ParamInitSpec;
using frame::frontend::TensorDataSpec;
using frame::frontend::TrainingSpec;

// obj 是否为 JSON 对象且含 key(非对象类型上调用 contains() 不会抛异常,直接
// 返回 false,故此处无需先行判定 obj.is_object())。
bool HasField(const Json& obj, const std::string& key) {
  return obj.is_object() && obj.contains(key);
}

// 取必填字符串字段;field_path 用于错误消息(如 "model.name",LANG-005)。
// key/field_path 是两个语义不同的字符串(JSON 字段名 vs. 拼接好的错误消息
// 前缀),调用点均为具名字面量传参,误置换风险低——NOLINT 规避
// bugprone-easily-swappable-parameters(手法同 src/frontend/runner.cpp
// DrawUniform 既有先例)。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Result<std::string> GetString(const Json& obj, const std::string& key,
                              const std::string& field_path) {
  if (!HasField(obj, key)) {
    return Status::make(ErrorCode::kInvalidArgument, field_path + " is required");
  }
  const Json& value = obj.at(key);
  if (!value.is_string()) {
    return Status::make(ErrorCode::kInvalidArgument, field_path + " must be a string");
  }
  return value.get<std::string>();
}

// 取必填整数字段(is_number_integer(),涵盖有符号/无符号两种底层表示)。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Result<int64_t> GetInt64(const Json& obj, const std::string& key, const std::string& field_path) {
  if (!HasField(obj, key)) {
    return Status::make(ErrorCode::kInvalidArgument, field_path + " is required");
  }
  const Json& value = obj.at(key);
  if (!value.is_number_integer()) {
    return Status::make(ErrorCode::kInvalidArgument, field_path + " must be an integer");
  }
  return value.get<int64_t>();
}

// 取必填数值字段(is_number(),整数/浮点均可,用于 learning_rate 一类超参)。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Result<double> GetDouble(const Json& obj, const std::string& key, const std::string& field_path) {
  if (!HasField(obj, key)) {
    return Status::make(ErrorCode::kInvalidArgument, field_path + " is required");
  }
  const Json& value = obj.at(key);
  if (!value.is_number()) {
    return Status::make(ErrorCode::kInvalidArgument, field_path + " must be a number");
  }
  return value.get<double>();
}

// 取必填对象字段,返回其值的拷贝(供调用方继续在其上取子字段)。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Result<Json> GetObject(const Json& obj, const std::string& key, const std::string& field_path) {
  if (!HasField(obj, key)) {
    return Status::make(ErrorCode::kInvalidArgument, field_path + " is required");
  }
  const Json& value = obj.at(key);
  if (!value.is_object()) {
    return Status::make(ErrorCode::kInvalidArgument, field_path + " must be an object");
  }
  return value;
}

// 取必填数组字段,返回其值的拷贝。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Result<Json> GetArray(const Json& obj, const std::string& key, const std::string& field_path) {
  if (!HasField(obj, key)) {
    return Status::make(ErrorCode::kInvalidArgument, field_path + " is required");
  }
  const Json& value = obj.at(key);
  if (!value.is_array()) {
    return Status::make(ErrorCode::kInvalidArgument, field_path + " must be an array");
  }
  return value;
}

// 取必填整数数组字段(如 shape/weight_shape/bias_shape/target_shape)。
Result<std::vector<int64_t>> GetInt64Array(const Json& obj, const std::string& key,
                                           const std::string& field_path) {
  const Result<Json> array_result = GetArray(obj, key, field_path);
  if (!array_result.is_ok()) {
    return array_result.status();
  }
  const Json& array = array_result.value();
  std::vector<int64_t> dims;
  dims.reserve(array.size());
  for (size_t i = 0; i < array.size(); ++i) {
    const Json& item = array.at(i);
    if (!item.is_number_integer()) {
      return Status::make(ErrorCode::kInvalidArgument,
                          field_path + "[" + std::to_string(i) + "] must be an integer");
    }
    dims.push_back(item.get<int64_t>());
  }
  return dims;
}

// 取必填浮点数组字段(data.<name>.values 内联取值)。
Result<std::vector<float>> GetFloatArray(const Json& obj, const std::string& key,
                                         const std::string& field_path) {
  const Result<Json> array_result = GetArray(obj, key, field_path);
  if (!array_result.is_ok()) {
    return array_result.status();
  }
  const Json& array = array_result.value();
  std::vector<float> values;
  values.reserve(array.size());
  for (size_t i = 0; i < array.size(); ++i) {
    const Json& item = array.at(i);
    if (!item.is_number()) {
      return Status::make(ErrorCode::kInvalidArgument,
                          field_path + "[" + std::to_string(i) + "] must be a number");
    }
    values.push_back(item.get<float>());
  }
  return values;
}

// 取恰好两个数值的数组字段(data.<name>.range / data.params.{weight,bias}_range)。
Result<std::pair<float, float>> GetNumberPair(const Json& obj, const std::string& key,
                                              const std::string& field_path) {
  const Result<Json> array_result = GetArray(obj, key, field_path);
  if (!array_result.is_ok()) {
    return array_result.status();
  }
  const Json& array = array_result.value();
  if (array.size() != 2) {
    return Status::make(ErrorCode::kInvalidArgument, field_path + " must have exactly 2 elements");
  }
  if (!array.at(0).is_number() || !array.at(1).is_number()) {
    return Status::make(ErrorCode::kInvalidArgument, field_path + " elements must be numbers");
  }
  return std::make_pair(array.at(0).get<float>(), array.at(1).get<float>());
}

// 取必填字符串字段并校验其取值恰等于 expected(用于 v0 已固定为唯一取值、故
// 不在 ModelSpec 结构体中落地存储的枚举字段:layers[].kind/loss.kind/
// optimizer.kind/model.dtype/data.params.kind;frontend-dsl.md 第 3 节
// FE-004「不认识 = 报错,禁止静默忽略」)。key/expected 两个相邻同型形参语义
// 不同(JSON 字段名 vs. 期望取值)——NOLINT 规避
// bugprone-easily-swappable-parameters(理由同 GetString)。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Status RequireExactString(const Json& obj, const std::string& key, const std::string& expected,
                          const std::string& field_path) {
  const Result<std::string> value_result = GetString(obj, key, field_path);
  if (!value_result.is_ok()) {
    return value_result.status();
  }
  if (value_result.value() != expected) {
    return Status::make(ErrorCode::kInvalidArgument, field_path + " must be '" + expected +
                                                         "', got '" + value_result.value() + "'");
  }
  return Status::ok();
}

// activation 字段:可选,缺省 kNone;给出时须为 "relu"/"none"(FE-004)。
Result<Activation> ParseActivation(const Json& item, const std::string& field_path) {
  if (!HasField(item, "activation")) {
    return Activation::kNone;
  }
  const Json& value = item.at("activation");
  if (!value.is_string()) {
    return Status::make(ErrorCode::kInvalidArgument, field_path + " must be a string");
  }
  const std::string text = value.get<std::string>();
  if (text == "none") {
    return Activation::kNone;
  }
  if (text == "relu") {
    return Activation::kRelu;
  }
  return Status::make(ErrorCode::kInvalidArgument,
                      field_path + " has an unrecognized value: '" + text + "'");
}

Result<InputSpec> ParseInputSpec(const Json& item, size_t index) {
  const std::string field_path = "inputs[" + std::to_string(index) + "]";
  InputSpec spec;

  const Result<std::string> name_result = GetString(item, "name", field_path + ".name");
  if (!name_result.is_ok()) {
    return name_result.status();
  }
  spec.name = name_result.value();

  const Result<std::vector<int64_t>> shape_result =
      GetInt64Array(item, "shape", field_path + ".shape");
  if (!shape_result.is_ok()) {
    return shape_result.status();
  }
  spec.shape = shape_result.value();

  return spec;
}

Result<LinearLayerSpec> ParseLayerSpec(const Json& item, size_t index) {
  const std::string field_path = "layers[" + std::to_string(index) + "]";
  LinearLayerSpec layer;

  const Result<std::string> name_result = GetString(item, "name", field_path + ".name");
  if (!name_result.is_ok()) {
    return name_result.status();
  }
  layer.name = name_result.value();

  const Status kind_status = RequireExactString(item, "kind", "linear", field_path + ".kind");
  if (!kind_status.is_ok()) {
    return kind_status;
  }

  const Result<std::string> input_result = GetString(item, "input", field_path + ".input");
  if (!input_result.is_ok()) {
    return input_result.status();
  }
  layer.input = input_result.value();

  const Result<std::vector<int64_t>> weight_shape_result =
      GetInt64Array(item, "weight_shape", field_path + ".weight_shape");
  if (!weight_shape_result.is_ok()) {
    return weight_shape_result.status();
  }
  layer.weight_shape = weight_shape_result.value();

  if (HasField(item, "bias_shape")) {
    const Result<std::vector<int64_t>> bias_shape_result =
        GetInt64Array(item, "bias_shape", field_path + ".bias_shape");
    if (!bias_shape_result.is_ok()) {
      return bias_shape_result.status();
    }
    layer.bias_shape = bias_shape_result.value();
  }

  const Result<Activation> activation_result = ParseActivation(item, field_path + ".activation");
  if (!activation_result.is_ok()) {
    return activation_result.status();
  }
  layer.activation = activation_result.value();

  return layer;
}

// data 段单个张量条目(数据输入或 target),field_path 形如 "data['x']"。
Result<TensorDataSpec> ParseTensorDataSpec(const Json& item, const std::string& key) {
  const std::string field_path = "data['" + key + "']";

  const Result<std::string> kind_result = GetString(item, "kind", field_path + ".kind");
  if (!kind_result.is_ok()) {
    return kind_result.status();
  }
  const std::string& kind = kind_result.value();

  TensorDataSpec data;
  if (kind == "inline") {
    data.kind = InitKind::kInline;
    const Result<std::vector<float>> values_result =
        GetFloatArray(item, "values", field_path + ".values");
    if (!values_result.is_ok()) {
      return values_result.status();
    }
    data.values = values_result.value();
  } else if (kind == "uniform_seeded") {
    data.kind = InitKind::kUniformSeeded;
    const Result<std::pair<float, float>> range_result =
        GetNumberPair(item, "range", field_path + ".range");
    if (!range_result.is_ok()) {
      return range_result.status();
    }
    data.lo = range_result.value().first;
    data.hi = range_result.value().second;
  } else {
    return Status::make(ErrorCode::kInvalidArgument,
                        field_path + ".kind has an unrecognized value: '" + kind + "'");
  }
  return data;
}

// data.params:全部参数的统一初始化范围(v0 恒为 uniform_seeded,不支持内联)。
Result<ParamInitSpec> ParseParamInitSpec(const Json& item) {
  const Status kind_status = RequireExactString(item, "kind", "uniform_seeded", "data.params.kind");
  if (!kind_status.is_ok()) {
    return kind_status;
  }

  ParamInitSpec param_init;
  const Result<std::pair<float, float>> weight_range_result =
      GetNumberPair(item, "weight_range", "data.params.weight_range");
  if (!weight_range_result.is_ok()) {
    return weight_range_result.status();
  }
  param_init.weight_lo = weight_range_result.value().first;
  param_init.weight_hi = weight_range_result.value().second;

  const Result<std::pair<float, float>> bias_range_result =
      GetNumberPair(item, "bias_range", "data.params.bias_range");
  if (!bias_range_result.is_ok()) {
    return bias_range_result.status();
  }
  param_init.bias_lo = bias_range_result.value().first;
  param_init.bias_hi = bias_range_result.value().second;

  return param_init;
}

}  // namespace

Result<ModelSpec> load_model_spec_from_json_file(const std::string& path) {
  std::ifstream file(path, std::ios::in);
  if (!file.is_open()) {
    return Status::make(ErrorCode::kNotFound, "failed to open JSON spec file '" + path + "'");
  }

  // 免异常解析路径(ADR-0018):parse 失败不抛异常,经 is_discarded() 判定。
  const Json root = Json::parse(file, /*cb=*/nullptr, /*allow_exceptions=*/false);
  if (root.is_discarded()) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "failed to parse '" + path + "': malformed JSON syntax");
  }
  if (!root.is_object()) {
    return Status::make(ErrorCode::kInvalidArgument, "root must be a JSON object");
  }

  // ---- schema_version(FE-001,收到的版本号入错误消息)----
  const Result<int64_t> schema_version_result = GetInt64(root, "schema_version", "schema_version");
  if (!schema_version_result.is_ok()) {
    return schema_version_result.status();
  }
  if (schema_version_result.value() != 0) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "schema_version must be 0, got " + std::to_string(schema_version_result.value()));
  }

  ModelSpec spec;

  // ---- model 段 ----
  const Result<Json> model_result = GetObject(root, "model", "model");
  if (!model_result.is_ok()) {
    return model_result.status();
  }
  const Json& model_obj = model_result.value();

  const Result<std::string> name_result = GetString(model_obj, "name", "model.name");
  if (!name_result.is_ok()) {
    return name_result.status();
  }
  spec.name = name_result.value();

  const Status dtype_status = RequireExactString(model_obj, "dtype", "float32", "model.dtype");
  if (!dtype_status.is_ok()) {
    return dtype_status;
  }

  const Result<int64_t> batch_result = GetInt64(model_obj, "batch", "model.batch");
  if (!batch_result.is_ok()) {
    return batch_result.status();
  }
  spec.batch = batch_result.value();

  // ---- inputs 段 ----
  const Result<Json> inputs_result = GetArray(root, "inputs", "inputs");
  if (!inputs_result.is_ok()) {
    return inputs_result.status();
  }
  const Json& inputs_arr = inputs_result.value();
  spec.inputs.reserve(inputs_arr.size());
  for (size_t i = 0; i < inputs_arr.size(); ++i) {
    const Json& item = inputs_arr.at(i);
    if (!item.is_object()) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "inputs[" + std::to_string(i) + "] must be an object");
    }
    const Result<InputSpec> input_result = ParseInputSpec(item, i);
    if (!input_result.is_ok()) {
      return input_result.status();
    }
    spec.inputs.push_back(input_result.value());
  }

  // ---- layers 段 ----
  const Result<Json> layers_result = GetArray(root, "layers", "layers");
  if (!layers_result.is_ok()) {
    return layers_result.status();
  }
  const Json& layers_arr = layers_result.value();
  spec.layers.reserve(layers_arr.size());
  for (size_t i = 0; i < layers_arr.size(); ++i) {
    const Json& item = layers_arr.at(i);
    if (!item.is_object()) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "layers[" + std::to_string(i) + "] must be an object");
    }
    const Result<LinearLayerSpec> layer_result = ParseLayerSpec(item, i);
    if (!layer_result.is_ok()) {
      return layer_result.status();
    }
    spec.layers.push_back(layer_result.value());
  }

  // ---- loss 段 ----
  const Result<Json> loss_result = GetObject(root, "loss", "loss");
  if (!loss_result.is_ok()) {
    return loss_result.status();
  }
  const Json& loss_obj = loss_result.value();

  const Status loss_kind_status = RequireExactString(loss_obj, "kind", "mse", "loss.kind");
  if (!loss_kind_status.is_ok()) {
    return loss_kind_status;
  }

  const Result<std::string> prediction_result =
      GetString(loss_obj, "prediction", "loss.prediction");
  if (!prediction_result.is_ok()) {
    return prediction_result.status();
  }
  spec.loss.prediction = prediction_result.value();

  const Result<std::vector<int64_t>> target_shape_result =
      GetInt64Array(loss_obj, "target_shape", "loss.target_shape");
  if (!target_shape_result.is_ok()) {
    return target_shape_result.status();
  }
  spec.loss.target_shape = target_shape_result.value();

  // ---- optimizer 段 ----
  const Result<Json> optimizer_result = GetObject(root, "optimizer", "optimizer");
  if (!optimizer_result.is_ok()) {
    return optimizer_result.status();
  }
  const Json& optimizer_obj = optimizer_result.value();

  const Status optimizer_kind_status =
      RequireExactString(optimizer_obj, "kind", "sgd", "optimizer.kind");
  if (!optimizer_kind_status.is_ok()) {
    return optimizer_kind_status;
  }

  const Result<double> learning_rate_result =
      GetDouble(optimizer_obj, "learning_rate", "optimizer.learning_rate");
  if (!learning_rate_result.is_ok()) {
    return learning_rate_result.status();
  }
  spec.optimizer.learning_rate = learning_rate_result.value();

  // ---- training 段 ----
  const Result<Json> training_result = GetObject(root, "training", "training");
  if (!training_result.is_ok()) {
    return training_result.status();
  }
  const Json& training_obj = training_result.value();

  const Result<int64_t> steps_result = GetInt64(training_obj, "steps", "training.steps");
  if (!steps_result.is_ok()) {
    return steps_result.status();
  }
  spec.training.steps = static_cast<int32_t>(steps_result.value());

  const Result<int64_t> seed_result = GetInt64(training_obj, "seed", "training.seed");
  if (!seed_result.is_ok()) {
    return seed_result.status();
  }
  spec.training.seed = static_cast<uint32_t>(seed_result.value());

  const Result<int64_t> log_every_result =
      GetInt64(training_obj, "log_every", "training.log_every");
  if (!log_every_result.is_ok()) {
    return log_every_result.status();
  }
  spec.training.log_every = static_cast<int32_t>(log_every_result.value());

  // ---- data(逐条目:"params" 走 ParamInitSpec,其余走 TensorDataSpec)----
  const Result<Json> data_result = GetObject(root, "data", "data");
  if (!data_result.is_ok()) {
    return data_result.status();
  }
  const Json& data_obj = data_result.value();
  for (const auto& [key, value] : data_obj.items()) {
    if (!value.is_object()) {
      return Status::make(ErrorCode::kInvalidArgument, "data['" + key + "'] must be an object");
    }
    if (key == "params") {
      const Result<ParamInitSpec> param_init_result = ParseParamInitSpec(value);
      if (!param_init_result.is_ok()) {
        return param_init_result.status();
      }
      spec.param_init = param_init_result.value();
      continue;
    }
    const Result<TensorDataSpec> tensor_data_result = ParseTensorDataSpec(value, key);
    if (!tensor_data_result.is_ok()) {
      return tensor_data_result.status();
    }
    spec.data.emplace(key, tensor_data_result.value());
  }

  return spec;
}

}  // namespace frame_dslc
