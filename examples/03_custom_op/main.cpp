// =============================================================================
// 示例 03:自定义算子扩展。
//
// 学习目标:通过 schema 与 CPU kernel 注册扩展算子,再走标准编译路径执行。
// 前置章节:help/09-add-operator/README.md。
// 预期 PASS:打印 scaled_relu 的确定结果与注册算子执行成功的 PASS 行。
// 运行边界:本例只注册 CPU kernel,不演示其他后端实现。
// =============================================================================

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include <frame/frame.h>
#include <frame/ops/graph_builder.h>

// -----------------------------------------------------------------------------
// 步骤 1:声明算子契约(schema)。
//
// 约定:算子名须匹配 ^[a-z][a-z0-9_]*$ 且全局唯一(ARCH-040),重名/非法名在启动期
// 报错(英文消息)。builder 方法链式返回 *this。此处以一个逐元素 "scaled_relu"
// (输出 = max(0, x) * scale)为例。
// -----------------------------------------------------------------------------

namespace {

// scaled_relu 的 shape 推断:逐元素算子,恰 1 输入,输出 shape 恒等于输入 shape。
frame::Result<std::vector<frame::Shape>> infer_scaled_relu_shape(
    const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'scaled_relu' expects 1 input, got " + std::to_string(ctx.input_types.size()));
  }
  return std::vector<frame::Shape>{ctx.input_types[0].shape};
}

}  // namespace

// FRAME_REGISTER_OP 返回 OpRegistry::register_op(name) 的 OpSchema&,支持链式声明:
FRAME_REGISTER_OP("scaled_relu")
    .input("x", "input tensor")
    .attr("scale", frame::ir::AttrType::kDouble, /*required=*/true)
    .output("y", "elementwise scaled relu of x: max(0, x) * scale")
    .trait(frame::ops::OpTrait::kElementwise)
    .trait(frame::ops::OpTrait::kFusable)
    .shape_infer(&infer_scaled_relu_shape);

// -----------------------------------------------------------------------------
// 步骤 2:为目标后端注册内核实现。
//
// 内核签名 = Status (*)(KernelContext&)(必须可转普通函数指针:无捕获 lambda /
// 自由函数,由 KernelImpl concept 编译期强制)。dtype 差异在内核内部经
// dispatch_dtype 一次性转回编译期类型,此后全程模板(铁律 #1②)。
// -----------------------------------------------------------------------------

namespace {

// 三档浮点(fp32/fp16/bf16,与内置 elementwise kernel 同一支持范围)各自经
// if constexpr 特化(编译期展开,CPP-012/ARCH-042,内层循环无运行时 dtype
// 分支)。fp16/bf16 借用既有位级转换(见 include/frame/core/dtype.h)升 float
// 计算后转回。dispatch_dtype 对 DTypeCode 全体成员做编译期穷举,故本函数也会
// 为其余 dtype(如 int32_t/bool)实例化;这些分支本体留空 ——
// scaled_relu_cpu_kernel 在调用 dispatch_dtype 前已拒绝这些 dtype,运行时不可达。
// numel/scale 相邻且均可隐式互转(clang-tidy bugprone-easily-swappable-
// parameters)——语义上二者不可合并(元素个数 vs 缩放系数),本函数是唯一
// 调用点(见下方 dispatch_dtype 内),实参均为具名局部变量传入,误置换会在
// 数值上立即暴露,抑制而非强行重排/拆分。
template <typename T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void apply_scaled_relu(const T* in, T* out, int64_t numel, double scale) {
  if constexpr (std::is_same_v<T, float>) {
    const float scale_f = static_cast<float>(scale);
    for (int64_t i = 0; i < numel; ++i) {
      out[i] = std::max(in[i], 0.0F) * scale_f;
    }
  } else if constexpr (std::is_same_v<T, frame::float16_t>) {
    const float scale_f = static_cast<float>(scale);
    for (int64_t i = 0; i < numel; ++i) {
      const float value = std::max(frame::float16_to_float(in[i]), 0.0F) * scale_f;
      out[i] = frame::float_to_float16(value);
    }
  } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
    const float scale_f = static_cast<float>(scale);
    for (int64_t i = 0; i < numel; ++i) {
      const float value = std::max(frame::bfloat16_to_float(in[i]), 0.0F) * scale_f;
      out[i] = frame::float_to_bfloat16(value);
    }
  }
}

// CPU 参考实现(REUSE-011:参考实现,数值校验用,禁作性能路径)。防御性校验:
// 输入/输出个数、x/y 的 shape/dtype 一致、dtype 限 v0 浮点三档、必填属性
// 'scale' 存在且类型为 double;任一违例返回英文错误(ARCH-031 口径:不静默
// 降级)。
frame::Status scaled_relu_cpu_kernel(frame::ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'scaled_relu' cpu kernel expects 1 input, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'scaled_relu' cpu kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }

  const frame::Tensor& x = ctx.inputs[0];
  frame::Tensor& y = ctx.outputs[0];

  const frame::DType x_elem_type = x.dtype();
  const frame::DType y_elem_type = y.dtype();
  if (!(x_elem_type == y_elem_type)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'scaled_relu' cpu kernel requires x/y of the same dtype, got '" +
                                   std::string(x_elem_type.name()) + "', '" +
                                   std::string(y_elem_type.name()) + "'");
  }
  if (!(x.shape() == y.shape())) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'scaled_relu' cpu kernel requires x/y of the same shape, got " +
                                   x.shape().to_string() + ", " + y.shape().to_string());
  }

  const frame::DTypeCode code = x_elem_type.code();
  const bool supported = (code == frame::DTypeCode::kFloat32) ||
                         (code == frame::DTypeCode::kFloat16) ||
                         (code == frame::DTypeCode::kBFloat16);
  if (!supported) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'scaled_relu' cpu kernel does not support dtype '" +
                                   std::string(x_elem_type.name()) +
                                   "' (v0 supports float32/float16/bfloat16 only)");
  }

  // scale(必填)从 ctx.attrs 取(借用契约:attrs 指针仅在调用期间有效,见
  // include/frame/ops/kernel_registry.h)。
  if (ctx.attrs == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'scaled_relu' cpu kernel is missing required attribute "
                               "'scale' (double): no attrs provided");
  }
  const auto scale_it = ctx.attrs->find("scale");
  if (scale_it == ctx.attrs->end()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'scaled_relu' cpu kernel is missing required attribute "
                               "'scale' (double)");
  }
  const double* scale_ptr = std::get_if<double>(&scale_it->second);
  if (scale_ptr == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'scaled_relu' cpu kernel attribute 'scale' has the wrong type, expected double");
  }

  const int64_t numel = x.numel();
  return frame::dispatch_dtype(code, [&]<typename T>() -> frame::Status {
    const T* x_data = static_cast<const T*>(x.raw_data());
    T* y_data = y.data<T>();
    apply_scaled_relu<T>(x_data, y_data, numel, *scale_ptr);
    return frame::Status::ok();
  });
}

}  // namespace

// (op 名, backend 名)为注册键;不含 dtype。此处注册到 CPU 参考后端:
FRAME_REGISTER_KERNEL("scaled_relu", frame::kCpuBackendName, scaled_relu_cpu_kernel);

// 若要支持更多后端,只需在各后端目录再注册一次(键不同),核心零改动(铁律 #3):
//   FRAME_REGISTER_KERNEL("scaled_relu", frame::kCudaBackendName, scaled_relu_cuda_kernel);

// -----------------------------------------------------------------------------
// 步骤 3:建图(经 ops::create_node_with_inferred_types 查 schema/调 shape_infer,
// 省去手工拼接输出 TensorType)→ runtime::compile 编译 → 执行 → 打印/校验
// (铁律 #1① 编译优先,主路径不出现任何逐算子 eager 调用)。
// -----------------------------------------------------------------------------
int main() {
  frame::ir::Graph graph("scaled_relu_example");

  frame::ir::TensorType x_type;
  x_type.dtype = frame::DType::of<float>();
  x_type.shape = frame::Shape({2, 3});
  x_type.device = frame::cpu_device();

  const frame::Result<frame::ir::Value*> x_result = graph.add_graph_input(x_type);
  if (!x_result.is_ok()) {
    std::cerr << "add_graph_input(x) failed: " << x_result.status().message() << "\n";
    return 1;
  }
  frame::ir::Value* x = x_result.value();

  const frame::ops::AttrMap attrs{{"scale", frame::ir::AttrValue{2.0}}};
  const frame::Result<frame::ir::Node*> node_result =
      frame::ops::create_node_with_inferred_types(graph, "scaled_relu", {x}, attrs);
  if (!node_result.is_ok()) {
    std::cerr << "create_node_with_inferred_types(scaled_relu) failed: "
              << node_result.status().message() << "\n";
    return 1;
  }
  frame::ir::Node* node = node_result.value();

  const frame::Status mark_status = graph.mark_output(node, 0);
  if (!mark_status.is_ok()) {
    std::cerr << "mark_output failed: " << mark_status.message() << "\n";
    return 1;
  }

  const frame::Result<std::shared_ptr<frame::hal::Executable>> executable_result =
      frame::runtime::compile(graph, frame::kCpuBackendName, frame::hal::CompileOptions{});
  if (!executable_result.is_ok()) {
    std::cerr << "runtime::compile failed: " << executable_result.status().message() << "\n";
    return 1;
  }
  const std::shared_ptr<frame::hal::Executable>& executable = executable_result.value();

  const frame::Result<frame::hal::Backend*> backend_result =
      frame::hal::BackendRegistry::instance().get(frame::kCpuBackendName);
  if (!backend_result.is_ok()) {
    std::cerr << "BackendRegistry::get failed: " << backend_result.status().message() << "\n";
    return 1;
  }
  frame::hal::Backend* backend = backend_result.value();
  const frame::Device device = frame::cpu_device();
  frame::hal::Allocator* allocator = backend->allocator(device);

  const frame::Result<frame::Tensor> x_tensor_result =
      frame::Tensor::empty(frame::Shape({2, 3}), frame::DType::of<float>(), device, *allocator);
  if (!x_tensor_result.is_ok()) {
    std::cerr << "Tensor::empty(x) failed: " << x_tensor_result.status().message() << "\n";
    return 1;
  }
  frame::Tensor x_tensor = x_tensor_result.value();
  const std::vector<float> x_values{-1.0F, 2.0F, -3.0F, 4.0F, -5.0F, 6.0F};
  std::copy(x_values.begin(), x_values.end(), x_tensor.data<float>());

  const std::vector<frame::Tensor> inputs{x_tensor};
  const frame::Result<std::vector<frame::Tensor>> outputs_result =
      frame::runtime::run_with_allocated_outputs(*executable, frame::kCpuBackendName, inputs);
  if (!outputs_result.is_ok()) {
    std::cerr << "run_with_allocated_outputs failed: " << outputs_result.status().message() << "\n";
    return 1;
  }
  const frame::Tensor& result = outputs_result.value()[0];

  std::cout << "scaled_relu(x, scale=2.0) =";
  const float* result_data = static_cast<const float*>(result.raw_data());
  for (int64_t i = 0; i < result.numel(); ++i) {
    std::cout << " " << result_data[i];
  }
  std::cout << "\n";

  // 手算期望值(未复用被测 kernel 的实现,与
  // tests/cpp/runtime/test_runtime_compile.cpp 的 SumWithAttrs 用例同款纪律):
  // max(0, x) * 2.0。
  const std::vector<float> expected{0.0F, 4.0F, 0.0F, 8.0F, 0.0F, 12.0F};
  for (int64_t i = 0; i < result.numel(); ++i) {
    // 输入与 scale 均为整数值,乘法和 ReLU 结果可精确表示。
    if (result_data[i] != expected[static_cast<size_t>(i)]) {
      std::cerr << "result does not match the hand-computed expected value\n";
      return 1;
    }
  }

  std::cout << "PASS: custom op scaled_relu compiled and executed correctly\n";
  return 0;
}
