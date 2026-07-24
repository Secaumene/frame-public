// emit_cpp:把 ModelSpec 生成为自包含 C++ 训练/推理源码工程(main.cpp +
// CMakeLists.txt),docs/architecture/frontend-dsl.md 第 5 节。生成的 main.cpp
// 风格逐段镜像 examples/02_graph_compile/main.cpp(CheckOk + std::cerr 错误
// 检查、中文注释、英文程序输出),随机生成顺序(数据输入 → target → 逐层
// 参数,weight 先于 bias)与 run_training/GenerateHostTensorValues 完全一致
// (同 seed 同轨迹)。产出文本不含时间戳/随机内容,给定同一 spec 恒生成逐字节
// 相同的内容(确定性)。

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#include <frame/core/shape.h>
#include <frame/frontend/emitter.h>

namespace frame::frontend {
namespace {

// int64 形状字面量,格式 "{d0, d1, ...}",供生成代码内
// MakeFloat32TensorType({...}) 调用使用。
std::string FormatInt64Braced(const std::vector<int64_t>& dims) {
  std::ostringstream out;
  out << "{";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << dims[i];
  }
  out << "}";
  return out.str();
}

// float 字面量,足够精度(max_digits10=9)保证 round-trip,追加项目风格的
// "F" 后缀(如 "1.0F")。
std::string FormatFloatLiteral(float value) {
  std::ostringstream out;
  out << std::setprecision(9) << value;
  std::string text = out.str();
  if (text.find('.') == std::string::npos && text.find('e') == std::string::npos &&
      text.find('E') == std::string::npos && text.find("inf") == std::string::npos &&
      text.find("nan") == std::string::npos) {
    text += ".0";
  }
  text += "F";
  return text;
}

// double 字面量(足够精度,max_digits10=17),用于学习率等超参烘焙。
std::string FormatDoubleLiteral(double value) {
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

std::string FormatFloatVectorLiteral(const std::vector<float>& values) {
  std::ostringstream out;
  out << "{";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << FormatFloatLiteral(values[i]);
  }
  out << "}";
  return out.str();
}

// 为第 i 层生成 matmul[+bias][+relu] 构图代码,追加到 out;两处调用点
// (BuildTrainingGraph/BuildInferenceGraph 生成文本)共用本函数(REUSE-002,
// 避免两份手写重复)。input_var 是该层输入 Value* 对应的生成代码变量名;
// 返回该层最终输出 Value* 对应的生成代码变量名。
std::string AppendLayerBlock(std::ostringstream& out, int index, const LinearLayerSpec& layer,
                             const std::string& input_var) {
  const std::string idx = std::to_string(index);
  const std::string weight_var = "layer" + idx + "_weight";
  const std::string matmul_var = "layer" + idx + "_matmul";

  out << "  // Layer " << index << " (\"" << layer.name << "\"): matmul";
  if (layer.bias_shape.has_value()) {
    out << " + bias";
  }
  if (layer.activation == Activation::kRelu) {
    out << " + relu";
  }
  out << "\n";
  out << "  const frame::Result<Value*> " << weight_var << "_result = graph.add_graph_input(\n"
      << "      MakeFloat32TensorType(" << FormatInt64Braced(layer.weight_shape) << "));\n"
      << "  if (!" << weight_var << "_result.is_ok()) return " << weight_var
      << "_result.status();\n"
      << "  Value* " << weight_var << " = " << weight_var << "_result.value();\n"
      << "  const frame::Result<Node*> " << matmul_var << "_result =\n"
      << "      create_node_with_inferred_types(graph, \"matmul\", {" << input_var << ", "
      << weight_var << "});\n"
      << "  if (!" << matmul_var << "_result.is_ok()) return " << matmul_var
      << "_result.status();\n"
      << "  Value* " << matmul_var << "_out = " << matmul_var << "_result.value()->output(0);\n";
  std::string current_var = matmul_var + "_out";

  if (layer.bias_shape.has_value()) {
    const std::string bias_var = "layer" + idx + "_bias";
    const std::string add_var = "layer" + idx + "_add";
    out << "  const frame::Result<Value*> " << bias_var << "_result = graph.add_graph_input(\n"
        << "      MakeFloat32TensorType(" << FormatInt64Braced(*layer.bias_shape) << "));\n"
        << "  if (!" << bias_var << "_result.is_ok()) return " << bias_var << "_result.status();\n"
        << "  Value* " << bias_var << " = " << bias_var << "_result.value();\n"
        << "  const frame::Result<Node*> " << add_var << "_result =\n"
        << "      create_node_with_inferred_types(graph, \"add\", {" << current_var << ", "
        << bias_var << "});\n"
        << "  if (!" << add_var << "_result.is_ok()) return " << add_var << "_result.status();\n"
        << "  Value* " << add_var << "_out = " << add_var << "_result.value()->output(0);\n";
    current_var = add_var + "_out";
  }

  if (layer.activation == Activation::kRelu) {
    const std::string relu_var = "layer" + idx + "_relu";
    out << "  const frame::Result<Node*> " << relu_var << "_result =\n"
        << "      create_node_with_inferred_types(graph, \"relu\", {" << current_var << "});\n"
        << "  if (!" << relu_var << "_result.is_ok()) return " << relu_var << "_result.status();\n"
        << "  Value* " << relu_var << "_out = " << relu_var << "_result.value()->output(0);\n";
    current_var = relu_var + "_out";
  }

  const std::string output_var = "layer" + idx + "_output";
  out << "  Value* " << output_var << " = " << current_var << ";\n\n";
  return output_var;
}

// 生成 BuildTrainingGraph()/BuildInferenceGraph() 共用的图头(建 Graph + 建
// data_input)与逐层构图段落;name_to_var 记录 spec 内名字 →
// 生成代码变量名的映射(供图尾按 loss.prediction 查找)。
void AppendGraphPrologue(std::ostringstream& out, const ModelSpec& spec,
                         const std::string& graph_name,
                         std::unordered_map<std::string, std::string>& name_to_var) {
  const InputSpec& input = spec.inputs[0];
  out << "  Graph graph(\"" << graph_name << "\");\n"
      << "  const frame::Result<Value*> data_input_result =\n"
      << "      graph.add_graph_input(MakeFloat32TensorType(" << FormatInt64Braced(input.shape)
      << "));\n"
      << "  if (!data_input_result.is_ok()) return data_input_result.status();\n"
      << "  Value* data_input = data_input_result.value();\n\n";

  name_to_var[input.name] = "data_input";
  for (size_t i = 0; i < spec.layers.size(); ++i) {
    const LinearLayerSpec& layer = spec.layers[i];
    const std::string input_var = name_to_var.at(layer.input);
    name_to_var[layer.name] = AppendLayerBlock(out, static_cast<int>(i), layer, input_var);
  }
}

// BuildTrainingGraph() 全函数文本:数据输入 -> 逐层 -> target -> mse_loss ->
// mark_output(loss)。
std::string BuildTrainingGraphText(const ModelSpec& spec) {
  std::ostringstream out;
  out << "// 建训练前向图,图输入序 [data_input, 逐层 weight,bias..., target],\n"
      << "// 单输出 = loss(与 frame::frontend::lower_to_graph 同一构图口径)。\n"
      << "frame::Result<frame::ir::Graph> BuildTrainingGraph() {\n"
      << "  using frame::ir::Graph;\n"
      << "  using frame::ir::Node;\n"
      << "  using frame::ir::Value;\n"
      << "  using frame::ops::create_node_with_inferred_types;\n\n";

  std::unordered_map<std::string, std::string> name_to_var;
  AppendGraphPrologue(out, spec, spec.name, name_to_var);

  out << "  const frame::Result<Value*> target_result = graph.add_graph_input(\n"
      << "      MakeFloat32TensorType(" << FormatInt64Braced(spec.loss.target_shape) << "));\n"
      << "  if (!target_result.is_ok()) return target_result.status();\n"
      << "  Value* target = target_result.value();\n\n"
      << "  const frame::Result<Node*> loss_result = create_node_with_inferred_types(\n"
      << "      graph, \"mse_loss\", {" << name_to_var.at(spec.loss.prediction) << ", target});\n"
      << "  if (!loss_result.is_ok()) return loss_result.status();\n"
      << "  if (!CheckOk(graph.mark_output(loss_result.value()->output(0)), "
         "\"mark_output(loss)\")) {\n"
      << "    return frame::Status::make(frame::ErrorCode::kInternal, \"mark_output(loss) "
         "failed\");\n"
      << "  }\n"
      << "  return graph;\n"
      << "}\n\n";
  return out.str();
}

// BuildInferenceGraph() 全函数文本:数据输入 -> 逐层 -> mark_output(预测值),
// 不含 target/mse_loss。
std::string BuildInferenceGraphText(const ModelSpec& spec) {
  std::ostringstream out;
  out << "// 建推理图,图输入序 [data_input, 逐层 weight,bias...](无 target),\n"
      << "// 单输出 = loss.prediction 引用层的输出(与\n"
      << "// frame::frontend::lower_to_inference_graph 同一构图口径)。\n"
      << "frame::Result<frame::ir::Graph> BuildInferenceGraph() {\n"
      << "  using frame::ir::Graph;\n"
      << "  using frame::ir::Node;\n"
      << "  using frame::ir::Value;\n"
      << "  using frame::ops::create_node_with_inferred_types;\n\n";

  std::unordered_map<std::string, std::string> name_to_var;
  AppendGraphPrologue(out, spec, spec.name + "_inference", name_to_var);

  out << "  if (!CheckOk(graph.mark_output(" << name_to_var.at(spec.loss.prediction)
      << "), \"mark_output(prediction)\")) {\n"
      << "    return frame::Status::make(frame::ErrorCode::kInternal, "
         "\"mark_output(prediction) failed\");\n"
      << "  }\n"
      << "  return graph;\n"
      << "}\n\n";
  return out.str();
}

// main() 内数据/参数生成段落:std::mt19937(kSeed),顺序 = 数据输入 -> target
// -> 逐层参数(weight 先于 bias),与 run_training/GenerateHostTensorValues
// 完全一致(frontend-dsl.md 第 2 节)。
std::string BuildDataGenerationText(const ModelSpec& spec) {
  const InputSpec& input = spec.inputs[0];
  const TensorDataSpec& input_data = spec.data.at(input.name);
  const TensorDataSpec& target_data = spec.data.at("target");

  std::ostringstream out;
  out << "  // ---- 4. 生成宿主侧数据/参数(std::mt19937(kSeed),顺序 = 数据输入\n"
      << "  //    -> target -> 逐层参数[weight 先 bias],与 frame_dslc --run 完全\n"
      << "  //    一致的抽取顺序,同 seed 同轨迹)----\n"
      << "  std::mt19937 rng(kSeed);\n"
      << "  auto draw_uniform = [&rng](float lo, float hi, int64_t count) {\n"
      << "    std::uniform_real_distribution<float> dist(lo, hi);\n"
      << "    std::vector<float> values(static_cast<size_t>(count));\n"
      << "    for (float& v : values) v = dist(rng);\n"
      << "    return values;\n"
      << "  };\n\n";

  if (input_data.kind == InitKind::kInline) {
    out << "  const std::vector<float> data_input_values = "
        << FormatFloatVectorLiteral(input_data.values) << ";\n";
  } else {
    out << "  const std::vector<float> data_input_values = draw_uniform("
        << FormatFloatLiteral(input_data.lo) << ", " << FormatFloatLiteral(input_data.hi) << ", "
        << Shape(input.shape).numel() << ");\n";
  }
  if (target_data.kind == InitKind::kInline) {
    out << "  const std::vector<float> target_values = "
        << FormatFloatVectorLiteral(target_data.values) << ";\n";
  } else {
    out << "  const std::vector<float> target_values = draw_uniform("
        << FormatFloatLiteral(target_data.lo) << ", " << FormatFloatLiteral(target_data.hi) << ", "
        << Shape(spec.loss.target_shape).numel() << ");\n";
  }
  out << "\n";

  for (size_t i = 0; i < spec.layers.size(); ++i) {
    const LinearLayerSpec& layer = spec.layers[i];
    out << "  const std::vector<float> layer" << i << "_weight_values =\n"
        << "      draw_uniform(kParamWeightLo, kParamWeightHi, "
        << Shape(layer.weight_shape).numel() << ");\n";
    if (layer.bias_shape.has_value()) {
      out << "  const std::vector<float> layer" << i << "_bias_values =\n"
          << "      draw_uniform(kParamBiasLo, kParamBiasHi, " << Shape(*layer.bias_shape).numel()
          << ");\n";
    }
  }
  out << "\n";
  return out.str();
}

// main() 内张量构造段落(examples/02_graph_compile/main.cpp:81-122 同款
// Tensor::empty + data<float>() + std::copy 手法),同时构造 params 顺序
// vector(weight 先 bias,与 param_types 顺序一致)。
std::string BuildTensorConstructionText(const ModelSpec& spec) {
  const InputSpec& input = spec.inputs[0];
  std::ostringstream out;
  out << "  // ---- 5. 建 Tensor ----\n"
      << "  const frame::Result<frame::Tensor> data_input_tensor_result = frame::Tensor::empty(\n"
      << "      frame::Shape(" << FormatInt64Braced(input.shape)
      << "), frame::DType::of<float>(), device, *allocator);\n"
      << "  if (!CheckOk(data_input_tensor_result.status(), \"Tensor::empty(data_input)\")) return "
         "1;\n"
      << "  frame::Tensor data_input_tensor = data_input_tensor_result.value();\n"
      << "  std::copy(data_input_values.begin(), data_input_values.end(), "
         "data_input_tensor.data<float>());\n\n"
      << "  const frame::Result<frame::Tensor> target_tensor_result = frame::Tensor::empty(\n"
      << "      frame::Shape(" << FormatInt64Braced(spec.loss.target_shape)
      << "), frame::DType::of<float>(), device, *allocator);\n"
      << "  if (!CheckOk(target_tensor_result.status(), \"Tensor::empty(target)\")) return 1;\n"
      << "  frame::Tensor target_tensor = target_tensor_result.value();\n"
      << "  std::copy(target_values.begin(), target_values.end(), "
         "target_tensor.data<float>());\n\n";

  out << "  std::vector<frame::Tensor> params;\n";
  for (size_t i = 0; i < spec.layers.size(); ++i) {
    const LinearLayerSpec& layer = spec.layers[i];
    const std::string idx = std::to_string(i);
    out << "  const frame::Result<frame::Tensor> layer" << idx << "_weight_tensor_result =\n"
        << "      frame::Tensor::empty(frame::Shape(" << FormatInt64Braced(layer.weight_shape)
        << "), frame::DType::of<float>(), device, *allocator);\n"
        << "  if (!CheckOk(layer" << idx << "_weight_tensor_result.status(), \"Tensor::empty(layer"
        << idx << "_weight)\")) return 1;\n"
        << "  frame::Tensor layer" << idx << "_weight_tensor = layer" << idx
        << "_weight_tensor_result.value();\n"
        << "  std::copy(layer" << idx << "_weight_values.begin(), layer" << idx
        << "_weight_values.end(), layer" << idx << "_weight_tensor.data<float>());\n"
        << "  params.push_back(layer" << idx << "_weight_tensor);\n";
    if (layer.bias_shape.has_value()) {
      out << "  const frame::Result<frame::Tensor> layer" << idx << "_bias_tensor_result =\n"
          << "      frame::Tensor::empty(frame::Shape(" << FormatInt64Braced(*layer.bias_shape)
          << "), frame::DType::of<float>(), device, *allocator);\n"
          << "  if (!CheckOk(layer" << idx << "_bias_tensor_result.status(), \"Tensor::empty(layer"
          << idx << "_bias)\")) return 1;\n"
          << "  frame::Tensor layer" << idx << "_bias_tensor = layer" << idx
          << "_bias_tensor_result.value();\n"
          << "  std::copy(layer" << idx << "_bias_values.begin(), layer" << idx
          << "_bias_values.end(), layer" << idx << "_bias_tensor.data<float>());\n"
          << "  params.push_back(layer" << idx << "_bias_tensor);\n";
    }
  }
  out << "\n";
  return out.str();
}

// 生成 main.cpp 全文。
std::string GenerateMainCppText(const ModelSpec& spec) {
  size_t param_count = 0;
  for (const LinearLayerSpec& layer : spec.layers) {
    param_count += layer.bias_shape.has_value() ? 2 : 1;
  }

  std::ostringstream wrt_list;
  for (size_t i = 0; i < param_count; ++i) {
    if (i > 0) {
      wrt_list << ", ";
    }
    wrt_list << (i + 1);
  }

  std::ostringstream out;
  out << "// ============================================================================\n"
      << "// " << spec.name << ":由 frame_dslc --emit 从 JSON DSL 生成的自包含训练/推理程序。\n"
      << "// 请勿手工编辑;如需修改模型结构,请修改源 DSL 文件后重新运行 --emit。\n"
      << "//\n"
      << "// 语言纪律(铁律 #4):标识符/程序输出/错误消息英文;注释中文。\n"
      << "// ============================================================================\n\n"
      << "#include <algorithm>\n"
      << "#include <cstdint>\n"
      << "#include <iostream>\n"
      << "#include <memory>\n"
      << "#include <random>\n"
      << "#include <string_view>\n"
      << "#include <vector>\n\n"
      << "#include <frame/compiler/autograd.h>\n"
      << "#include <frame/frame.h>\n"
      << "#include <frame/ops/graph_builder.h>\n\n"
      << "namespace {\n\n"
      << "// 打印失败 Status 到 stderr 并返回 false;调用方据此立即返回。\n"
      << "bool CheckOk(const frame::Status& status, std::string_view what) {\n"
      << "  if (status.is_ok()) return true;\n"
      << "  std::cerr << what << \" failed: \" << status.message() << \"\\n\";\n"
      << "  return false;\n"
      << "}\n\n"
      << "// 构造 float32/cpu 的 TensorType。\n"
      << "frame::ir::TensorType MakeFloat32TensorType(std::vector<int64_t> dims) {\n"
      << "  frame::ir::TensorType type;\n"
      << "  type.dtype = frame::DType::of<float>();\n"
      << "  type.shape = frame::Shape(std::move(dims));\n"
      << "  type.device = frame::cpu_device();\n"
      << "  return type;\n"
      << "}\n\n";

  out << "constexpr double kLearningRate = " << FormatDoubleLiteral(spec.optimizer.learning_rate)
      << ";\n"
      << "constexpr int32_t kNumSteps = " << spec.training.steps << ";\n"
      << "constexpr int32_t kLogEvery = " << spec.training.log_every << ";\n"
      << "constexpr uint32_t kSeed = " << spec.training.seed << "U;\n"
      << "constexpr float kParamWeightLo = " << FormatFloatLiteral(spec.param_init.weight_lo)
      << ";\n"
      << "constexpr float kParamWeightHi = " << FormatFloatLiteral(spec.param_init.weight_hi)
      << ";\n"
      << "constexpr float kParamBiasLo = " << FormatFloatLiteral(spec.param_init.bias_lo) << ";\n"
      << "constexpr float kParamBiasHi = " << FormatFloatLiteral(spec.param_init.bias_hi)
      << ";\n\n";

  out << BuildTrainingGraphText(spec);
  out << BuildInferenceGraphText(spec);
  out << "}  // namespace\n\n";

  out << "int main() {\n"
      << "  // ---- 1. 建训练前向图 + 反向图 + 编译 ----\n"
      << "  const frame::Result<frame::ir::Graph> training_graph_result = BuildTrainingGraph();\n"
      << "  if (!CheckOk(training_graph_result.status(), \"BuildTrainingGraph\")) return 1;\n\n"
      << "  const std::vector<int32_t> wrt{" << wrt_list.str() << "};\n"
      << "  const frame::Result<frame::ir::Graph> backward_graph_result =\n"
      << "      frame::compiler::build_backward_graph(training_graph_result.value(), 0, wrt);\n"
      << "  if (!CheckOk(backward_graph_result.status(), \"build_backward_graph\")) return 1;\n\n"
      << "  const frame::ir::OpQuery op_query = frame::ops::make_op_query();\n"
      << "  if (!CheckOk(backward_graph_result.value().verify(op_query), "
         "\"verify(backward_graph)\"))"
         " return 1;\n\n"
      << "  const frame::Result<std::shared_ptr<frame::hal::Executable>> train_executable_result "
         "=\n"
      << "      frame::runtime::compile(backward_graph_result.value(), frame::kCpuBackendName,\n"
      << "                             frame::hal::CompileOptions{});\n"
      << "  if (!CheckOk(train_executable_result.status(), \"runtime::compile(train)\")) return "
         "1;\n\n";

  out << "  // ---- 2. 建 SGD 更新图 + 编译 ----\n"
      << "  std::vector<frame::ir::TensorType> param_types;\n";
  for (size_t i = 0; i < spec.layers.size(); ++i) {
    const LinearLayerSpec& layer = spec.layers[i];
    out << "  param_types.push_back(MakeFloat32TensorType(" << FormatInt64Braced(layer.weight_shape)
        << "));\n";
    if (layer.bias_shape.has_value()) {
      out << "  param_types.push_back(MakeFloat32TensorType("
          << FormatInt64Braced(*layer.bias_shape) << "));\n";
    }
  }
  out << "\n"
      << "  const frame::Result<frame::ir::Graph> update_graph_result =\n"
      << "      frame::compiler::build_sgd_update_graph(param_types, kLearningRate);\n"
      << "  if (!CheckOk(update_graph_result.status(), \"build_sgd_update_graph\")) return 1;\n"
      << "  if (!CheckOk(update_graph_result.value().verify(op_query), \"verify(update_graph)\")) "
         "return 1;\n\n"
      << "  const frame::Result<std::shared_ptr<frame::hal::Executable>> update_executable_result "
         "=\n"
      << "      frame::runtime::compile(update_graph_result.value(), frame::kCpuBackendName,\n"
      << "                             frame::hal::CompileOptions{});\n"
      << "  if (!CheckOk(update_executable_result.status(), \"runtime::compile(update)\")) return "
         "1;\n\n";

  out << "  // ---- 3. 取 cpu 后端 allocator ----\n"
      << "  const frame::Result<frame::hal::Backend*> backend_result =\n"
      << "      frame::hal::BackendRegistry::instance().get(frame::kCpuBackendName);\n"
      << "  if (!CheckOk(backend_result.status(), \"BackendRegistry::get\")) return 1;\n"
      << "  frame::hal::Backend* backend = backend_result.value();\n"
      << "  const frame::Device device = frame::cpu_device();\n"
      << "  frame::hal::Allocator* allocator = backend->allocator(device);\n\n";

  out << BuildDataGenerationText(spec);
  out << BuildTensorConstructionText(spec);

  out << "  // ---- 6. 训练循环 ----\n"
      << "  double final_loss = 0.0;\n"
      << "  for (int32_t step = 0; step < kNumSteps; ++step) {\n"
      << "    std::vector<frame::Tensor> train_inputs;\n"
      << "    train_inputs.push_back(data_input_tensor);\n"
      << "    for (const frame::Tensor& p : params) train_inputs.push_back(p);\n"
      << "    train_inputs.push_back(target_tensor);\n\n"
      << "    const frame::Result<std::vector<frame::Tensor>> train_outputs_result =\n"
      << "        frame::runtime::run_with_allocated_outputs(*train_executable_result.value(),\n"
      << "                                                   frame::kCpuBackendName, "
         "train_inputs);\n"
      << "    if (!CheckOk(train_outputs_result.status(), \"run_with_allocated_outputs(train)\")) "
         "return 1;\n"
      << "    const std::vector<frame::Tensor>& train_outputs = train_outputs_result.value();\n\n"
      << "    const float loss_value = *static_cast<const float*>(train_outputs[0].raw_data());\n"
      << "    final_loss = static_cast<double>(loss_value);\n"
      << "    if (kLogEvery > 0 && (step % kLogEvery == 0 || step == kNumSteps - 1)) {\n"
      << "      std::cout << \"step \" << step << \": loss = \" << loss_value << \"\\n\";\n"
      << "    }\n\n"
      << "    std::vector<frame::Tensor> update_inputs;\n"
      << "    for (const frame::Tensor& p : params) update_inputs.push_back(p);\n"
      << "    for (size_t i = 1; i < train_outputs.size(); ++i) "
         "update_inputs.push_back(train_outputs[i]);\n\n"
      << "    const frame::Result<std::vector<frame::Tensor>> update_outputs_result =\n"
      << "        frame::runtime::run_with_allocated_outputs(*update_executable_result.value(),\n"
      << "                                                   frame::kCpuBackendName, "
         "update_inputs);\n"
      << "    if (!CheckOk(update_outputs_result.status(), "
         "\"run_with_allocated_outputs(update)\")) "
         "return 1;\n"
      << "    params = update_outputs_result.value();\n"
      << "  }\n"
      << "  std::cout << \"final loss = \" << final_loss << \"\\n\";\n\n";

  out << "  // ---- 7. 推理:BuildInferenceGraph + compile + 执行一次 ----\n"
      << "  const frame::Result<frame::ir::Graph> inference_graph_result = BuildInferenceGraph();\n"
      << "  if (!CheckOk(inference_graph_result.status(), \"BuildInferenceGraph\")) return 1;\n"
      << "  if (!CheckOk(inference_graph_result.value().verify(op_query), "
         "\"verify(inference_graph)\"))"
         " return 1;\n\n"
      << "  const frame::Result<std::shared_ptr<frame::hal::Executable>> "
         "inference_executable_result =\n"
      << "      frame::runtime::compile(inference_graph_result.value(), frame::kCpuBackendName,\n"
      << "                             frame::hal::CompileOptions{});\n"
      << "  if (!CheckOk(inference_executable_result.status(), \"runtime::compile(inference)\")) "
         "return 1;\n\n"
      << "  std::vector<frame::Tensor> inference_inputs;\n"
      << "  inference_inputs.push_back(data_input_tensor);\n"
      << "  for (const frame::Tensor& p : params) inference_inputs.push_back(p);\n\n"
      << "  const frame::Result<std::vector<frame::Tensor>> inference_outputs_result =\n"
      << "      frame::runtime::run_with_allocated_outputs(*inference_executable_result.value(),\n"
      << "                                                 frame::kCpuBackendName, "
         "inference_inputs);\n"
      << "  if (!CheckOk(inference_outputs_result.status(), "
         "\"run_with_allocated_outputs(inference)\"))"
         " return 1;\n\n"
      << "  const frame::Tensor& prediction_tensor = inference_outputs_result.value()[0];\n"
      << "  const float* prediction_data = static_cast<const "
         "float*>(prediction_tensor.raw_data());\n"
      << "  std::cout << \"predictions =\";\n"
      << "  for (int64_t i = 0; i < prediction_tensor.numel(); ++i) {\n"
      << "    std::cout << \" \" << prediction_data[i];\n"
      << "  }\n"
      << "  std::cout << \"\\n\";\n\n"
      << "  return 0;\n"
      << "}\n";

  return out.str();
}

// 生成 CMakeLists.txt 全文(frontend-dsl.md 第 5 节:cmake_minimum_required
// (3.25)、find_package(frame REQUIRED)、链 frame::frame)。
std::string GenerateCMakeListsText(const ModelSpec& spec) {
  std::ostringstream out;
  out << "# 由 frame_dslc --emit 生成,请勿手工编辑。\n"
      << "cmake_minimum_required(VERSION 3.25)\n\n"
      << "project(" << spec.name
      << ")\n\n"
      // frame 导出包未将 C++20 要求通过 target_compile_features 传递给消费方
      // (仅顶层 CMakeLists.txt 以变量形式设定,不随 install(EXPORT) 导出),
      // 消费方须显式声明,否则 <frame/core/dtype.h> 等依赖 C++20 concept 的
      // 头文件在消费方默认标准下编译失败。
      << "set(CMAKE_CXX_STANDARD 20)\n"
      << "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n\n"
      << "find_package(frame REQUIRED)\n\n"
      << "add_executable(" << spec.name << "_train main.cpp)\n"
      << "target_link_libraries(" << spec.name << "_train PRIVATE frame::frame)\n";
  return out.str();
}

}  // namespace

Status emit_cpp(const ModelSpec& spec, const EmitOptions& options) {
  Status validate_status = validate(spec);
  if (!validate_status.is_ok()) {
    return validate_status;
  }

  std::error_code ec;
  std::filesystem::create_directories(options.output_dir, ec);
  if (ec) {
    return Status::make(ErrorCode::kInternal, "failed to create output directory '" +
                                                  options.output_dir + "': " + ec.message());
  }

  const std::filesystem::path output_dir(options.output_dir);

  const std::string main_cpp_text = GenerateMainCppText(spec);
  std::ofstream main_cpp_file(output_dir / "main.cpp", std::ios::out | std::ios::trunc);
  if (!main_cpp_file.is_open()) {
    return Status::make(ErrorCode::kInternal, "failed to open main.cpp for writing");
  }
  main_cpp_file << main_cpp_text;
  main_cpp_file.close();
  if (main_cpp_file.fail()) {
    return Status::make(ErrorCode::kInternal, "failed to write main.cpp");
  }

  const std::string cmakelists_text = GenerateCMakeListsText(spec);
  std::ofstream cmakelists_file(output_dir / "CMakeLists.txt", std::ios::out | std::ios::trunc);
  if (!cmakelists_file.is_open()) {
    return Status::make(ErrorCode::kInternal, "failed to open CMakeLists.txt for writing");
  }
  cmakelists_file << cmakelists_text;
  cmakelists_file.close();
  if (cmakelists_file.fail()) {
    return Status::make(ErrorCode::kInternal, "failed to write CMakeLists.txt");
  }

  return Status::ok();
}

}  // namespace frame::frontend
