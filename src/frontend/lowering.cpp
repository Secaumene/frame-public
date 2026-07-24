// lower_to_graph / lower_to_inference_graph:ModelSpec → ir::Graph。两个入口
// 共用 BuildForwardBase 这一段构图逻辑(REUSE-002,避免训练图/推理图两份
// ≥20 行同构重复)。网络结构段(逐层 matmul[+bias][+relu])改经 frame::nn
// 模块构图(ADR-0020 判定③、docs/architecture/nn-design.md §4 蓝图 + ARCH-
// 074/075):逐层独立 nn::Linear[+nn::Relu](不包 Sequential——层间数据流仍按
// ModelSpec 声明的 layer.input 名字解析,value_by_name 映射,与旧实现
// AppendLayer 同一套名字解析流,保留 frontend-dsl.md FE-002 允许的任意合法
// DAG 拓扑:loss.prediction 可指向非末层、层输入可跳连/复用数据输入)→
// nn::add_parameter_inputs 批量物化参数图输入 → 逐层 module.build;损失子图
// 经 nn::MseLoss 构图(ARCH-074 职责边界裁定)。LoweredModel 三元组
// (param_names/param_types/wrt_input_indices)与图输入总序契约逐位不变
// (ARCH-074 判定方法①);批量参数图输入前置于逐层 build 之前,forward 图与
// 旧实现(逐层交错 weight-input/matmul/bias-input/add)相比 dump_text 发射序
// 整体重排,但节点集与数据流边集拓扑等价(判定方法②)——chain 形态(现有
// tests/tools 样例)的逐层 build 顺序与重排前一致,故其 golden 未再变化;
// 非 chain 形态(跳连/prediction 非末层)见 test_lowering.cpp 新增用例(不落
// golden 文件,断言节点数/边即可)。

#include <cstddef>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/macros.h>
#include <frame/core/shape.h>
#include <frame/frontend/lowering.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/nn/layers.h>
#include <frame/nn/module.h>

namespace frame::frontend {

// 本翻译单元内共用的短别名(REUSE-002:helper 与两个公开入口共用同一组,
// using 声明位于 frame::frontend 命名空间作用域,不泄漏到本文件之外)。
using frame::ir::Graph;
using frame::ir::TensorType;
using frame::ir::Value;

namespace {

// 构造 float32/cpu 的 TensorType(v0 唯一支持组合;examples/02_graph_compile/
// main.cpp 同款手法)。仅用于填充 LoweredModel 的参数/数据/target 元信息,
// 与实际构图节点解耦(网络结构节点改经 nn 模块创建,见下)。
TensorType MakeCpuFloat32TensorType(const std::vector<int64_t>& dims) {
  TensorType type;
  type.dtype = DType::of<float>();
  type.shape = Shape(dims);
  type.device = cpu_device();
  return type;
}

// BuildForwardBase 的产物:共享构图段落 + 两个上层函数各自所需的收尾信息。
struct BaseBuildResult {
  Graph graph;
  Value* prediction = nullptr;  // loss.prediction 引用层的输出
  std::vector<std::string> param_names;
  std::vector<TensorType> param_types;
  std::vector<int32_t> wrt_input_indices;
};

// 建「数据输入 -> 逐层 matmul[+bias][+relu]」这一段前向图,不追加
// target/mse_loss,也不 mark_output(由调用方按各自需求收尾)。调用方须先行
// validate(spec)——本函数假定 spec 已合法,不重复校验。
Result<BaseBuildResult> BuildForwardBase(const ModelSpec& spec) {
  Graph graph(spec.name);
  std::unordered_map<std::string, Value*> value_by_name;

  // v0 恰一个数据输入(validate 已保证)。
  const InputSpec& input = spec.inputs[0];
  const Result<Value*> input_value = graph.add_graph_input(MakeCpuFloat32TensorType(input.shape));
  if (!input_value.is_ok()) {
    return input_value.status();
  }
  value_by_name.emplace(input.name, input_value.value());

  // ---- 逐层独立 nn::Linear 模块(不包 Sequential)+ 参数元信息
  // (param_names/param_types/wrt_input_indices):按 ModelSpec 层名生成,与
  // 旧实现逐位相同,独立于 nn 内部路径命名(nn parameters() 的路径前缀命名
  // 如 "layer0.weight" 仅供内部构图消费,不对外暴露;LoweredModel 契约以
  // 本段计算结果为准)。 ----
  BaseBuildResult result;
  std::vector<nn::Module> layer_modules;
  layer_modules.reserve(spec.layers.size());
  int32_t next_input_index = static_cast<int32_t>(spec.inputs.size());
  for (const LinearLayerSpec& layer : spec.layers) {
    // validate(spec) 已保证 layer.weight_shape 秩为 2,故此处可直接按
    // [in, out] 取值。
    layer_modules.push_back(nn::Linear(layer.name, spec.batch, layer.weight_shape[0],
                                       layer.weight_shape[1], layer.bias_shape.has_value(),
                                       DType::of<float>()));

    result.param_names.push_back(layer.name + "_weight");
    result.param_types.push_back(MakeCpuFloat32TensorType(layer.weight_shape));
    result.wrt_input_indices.push_back(next_input_index++);
    if (layer.bias_shape.has_value()) {
      result.param_names.push_back(layer.name + "_bias");
      result.param_types.push_back(MakeCpuFloat32TensorType(*layer.bias_shape));
      result.wrt_input_indices.push_back(next_input_index++);
    }
  }

  // ---- nn 构图(ARCH-074):逐层 Linear 模块的 parameters() 依层序拼接后
  // 整体 add_parameter_inputs 一次性物化(批量前置);拼接序 = 逐层
  // [weight,(bias)?](Linear 工厂声明序),与上面独立算出的 param_names 顺序
  // 一致,故每层在拼接结果中的切片可用同一份逐层计数复原。 ----
  std::vector<nn::ParamSpec> nn_param_specs;
  std::vector<size_t> layer_param_counts;
  layer_param_counts.reserve(layer_modules.size());
  for (const nn::Module& layer_module : layer_modules) {
    const std::vector<nn::ParamSpec> layer_specs = layer_module.parameters();
    layer_param_counts.push_back(layer_specs.size());
    nn_param_specs.insert(nn_param_specs.end(), layer_specs.begin(), layer_specs.end());
  }
  const Result<std::vector<Value*>> nn_param_values =
      nn::add_parameter_inputs(graph, nn_param_specs);
  if (!nn_param_values.is_ok()) {
    return nn_param_values.status();
  }

  // ---- 逐层 build:layer.input 按名字从 value_by_name 解析(与旧实现
  // AppendLayer 同一套名字解析流,支持非紧邻前层/复用数据输入的跳连
  // 拓扑,FE-002 允许的任意合法 DAG);该层参数 = nn_param_values 中对应
  // 偏移区间的切片;relu 经独立 nn::Relu 模块串接。 ----
  const std::span<Value* const> all_param_values(nn_param_values.value());
  size_t param_offset = 0;
  for (size_t i = 0; i < spec.layers.size(); ++i) {
    const LinearLayerSpec& layer = spec.layers[i];
    Value* layer_input = value_by_name.at(layer.input);  // validate 已保证存在

    const size_t param_count = layer_param_counts[i];
    const std::span<Value* const> layer_params =
        all_param_values.subspan(param_offset, param_count);
    param_offset += param_count;

    const Result<std::vector<Value*>> linear_outputs =
        layer_modules[i].build(graph, std::vector<Value*>{layer_input}, layer_params);
    if (!linear_outputs.is_ok()) {
      return linear_outputs.status();
    }
    // nn::Linear 恰一个输出;违反即内部不变量被破坏,非用户输入错误。
    FRAME_CHECK(linear_outputs.value().size() == 1);
    Value* current = linear_outputs.value()[0];

    if (layer.activation == Activation::kRelu) {
      const nn::Module relu_module = nn::Relu(layer.name + "_relu");
      const Result<std::vector<Value*>> relu_outputs =
          relu_module.build(graph, std::vector<Value*>{current}, std::vector<Value*>{});
      if (!relu_outputs.is_ok()) {
        return relu_outputs.status();
      }
      // nn::Relu 恰一个输出;违反即内部不变量被破坏,非用户输入错误。
      FRAME_CHECK(relu_outputs.value().size() == 1);
      current = relu_outputs.value()[0];
    }

    value_by_name[layer.name] = current;
    if (layer.name == spec.loss.prediction) {
      result.prediction = current;
    }
  }

  // validate() 已保证 loss.prediction 命中某层名,故此处必已赋值。
  FRAME_CHECK(result.prediction != nullptr);

  result.graph = std::move(graph);
  return result;
}

}  // namespace

Result<LoweredModel> lower_to_graph(const ModelSpec& spec) {
  const Status validate_status = validate(spec);
  if (!validate_status.is_ok()) {
    return validate_status;
  }

  Result<BaseBuildResult> base_result = BuildForwardBase(spec);
  if (!base_result.is_ok()) {
    return base_result.status();
  }
  BaseBuildResult base = std::move(base_result.value());

  const TensorType target_type = MakeCpuFloat32TensorType(spec.loss.target_shape);
  const Result<Value*> target_value = base.graph.add_graph_input(target_type);
  if (!target_value.is_ok()) {
    return target_value.status();
  }

  // 损失子图的构图原语归 nn(ARCH-074 职责边界裁定);是否装配损失、target
  // 从哪来、mark_output 均由调用方(本函数)决定。
  const nn::Module loss_module = nn::MseLoss("mse_loss");
  const Result<std::vector<Value*>> loss_outputs =
      loss_module.build(base.graph, std::vector<Value*>{base.prediction, target_value.value()},
                        std::vector<Value*>{});
  if (!loss_outputs.is_ok()) {
    return loss_outputs.status();
  }
  FRAME_CHECK(loss_outputs.value().size() == 1);

  const Status mark_status = base.graph.mark_output(loss_outputs.value()[0]);
  if (!mark_status.is_ok()) {
    return mark_status;
  }

  LoweredModel model;
  model.forward = std::move(base.graph);
  model.param_names = std::move(base.param_names);
  model.param_types = std::move(base.param_types);
  model.wrt_input_indices = std::move(base.wrt_input_indices);
  return model;
}

Result<Graph> lower_to_inference_graph(const ModelSpec& spec) {
  const Status validate_status = validate(spec);
  if (!validate_status.is_ok()) {
    return validate_status;
  }

  Result<BaseBuildResult> base_result = BuildForwardBase(spec);
  if (!base_result.is_ok()) {
    return base_result.status();
  }
  BaseBuildResult base = std::move(base_result.value());

  const Status mark_status = base.graph.mark_output(base.prediction);
  if (!mark_status.is_ok()) {
    return mark_status;
  }

  return std::move(base.graph);
}

}  // namespace frame::frontend
