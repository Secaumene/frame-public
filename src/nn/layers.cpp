// 首批模块工厂实现(Linear/Relu/Sequential/MseLoss,ARCH-071/072 §2);以及
// M21 批3 T6 新增的卷积批次工厂(Conv2d/Conv1d/MaxPool2d/AvgPool2d/Sigmoid/
// Flatten/AFF/Dwt2d/Dwt1d,docs/plan/2026-07-18-batch3-m21-conv.md 第1.4节)、
// M22 批4 T5 新增的序列批次工厂(LayerNorm/LSTM/MultiheadAttention/
// TransformerEncoderBlock,docs/plan/2026-07-19-batch4-m22-seq.md §1.7)、
// M23 批5 T5 新增的频域批次工厂(SpectralConv1d/FourierFilter1d/Fno1dBlock,
// docs/plan/2026-07-21-batch5-m23-fft.md §1.5),以及 M25 批6 T4 新增的
// 状态空间模型工厂(Mamba/FourierMamba,docs/plan/
// 2026-07-23-batch6-m25-ssm.md §1.4)、M27 脉冲工厂(LIFCell/
// SnnClassifier)与 M28 图网络工厂(GraphConv/HypergraphConv)。见
// include/frame/nn/layers.h 头注释及对应批次计划。

#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/shape.h>
#include <frame/ir/graph.h>
#include <frame/nn/layers.h>
#include <frame/ops/constant_utils.h>
#include <frame/ops/graph_builder.h>

namespace frame::nn {

namespace {

using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::TensorType;
using frame::ir::Value;
using frame::ops::AttrMap;
using frame::ops::create_node_with_inferred_types;
using frame::ops::kConstantOpName;
using frame::ops::kMaxDoubleExactInteger;

// 构造 cpu 设备上的 TensorType(v0 全 cpu,同
// src/frontend/lowering.cpp::MakeCpuFloat32TensorType 手法;dtype 由调用方
// 显式传入,不固定 float32)。
TensorType MakeCpuTensorType(DType dtype, std::vector<int64_t> dims) {
  TensorType type;
  type.dtype = dtype;
  type.shape = Shape(std::move(dims));
  type.device = cpu_device();
  return type;
}

// 构造给定行优先值的 constant 节点。本文件全部常量和固定拓扑索引共享这一
// AttrMap 物化入口,禁止各工厂复制第二份 constant 属性拼装逻辑。
Result<Node*> make_constant_values(Graph& graph, const Shape& shape, DType dtype, Device device,
                                   std::vector<double> values) {
  const AttrMap attrs{
      {"value", std::move(values)},
      {"shape", shape},
      {"dtype", dtype},
  };
  return create_node_with_inferred_types(graph, kConstantOpName, device, attrs);
}

// 构造 shape 全形、值全为 fill_value 的 constant 节点,复用上方唯一物化入口。
Result<Node*> make_constant_splat(Graph& graph, const Shape& shape, DType dtype, Device device,
                                  double fill_value) {
  const int64_t numel = shape.numel();
  return make_constant_values(graph, shape, dtype, device,
                              std::vector<double>(static_cast<size_t>(numel), fill_value));
}

// 固定拓扑索引统一编码为 int64 constant,复用唯一 constant 属性物化入口。
Status validate_index_constant_precision(std::string_view role, std::span<const int64_t> indices) {
  for (size_t position = 0; position < indices.size(); ++position) {
    const int64_t index = indices[position];
    if (index > kMaxDoubleExactInteger || index < -kMaxDoubleExactInteger) {
      return Status::make(ErrorCode::kInvalidArgument,
                          std::string(role) + " index at position " + std::to_string(position) +
                              " exceeds the double-exact integer "
                              "bound 2^53=" +
                              std::to_string(kMaxDoubleExactInteger) +
                              ", cannot be encoded as a constant without precision loss");
    }
  }
  return Status::ok();
}

Result<Node*> make_index_constant(Graph& graph, const std::vector<int64_t>& indices,
                                  Device device) {
  if (!std::in_range<int64_t>(indices.size())) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "nn index constant length exceeds int64 range");
  }
  const Status precision_status = validate_index_constant_precision("nn index constant", indices);
  if (!precision_status.is_ok()) return precision_status;
  std::vector<double> values;
  values.reserve(indices.size());
  for (int64_t index : indices) values.push_back(static_cast<double>(index));
  return make_constant_values(graph, Shape({static_cast<int64_t>(indices.size())}),
                              DType(DTypeCode::kInt64), device, std::move(values));
}

bool is_supported_float_dtype(DType dtype) {
  const DTypeCode code = dtype.code();
  return code == DTypeCode::kFloat32 || code == DTypeCode::kFloat16 || code == DTypeCode::kBFloat16;
}

// 静态工厂在返回 Module 前不能返回 Status；所有会进入 shape/容量计算的
// 非负因子先经此处判溢出，失败时由工厂用安全占位形状构造 children，并在
// BuildFn 的任何图改动前 fail-loud。仓内未找到可复用的 int64 连乘 helper。
std::optional<int64_t> checked_nonnegative_product(std::initializer_list<int64_t> factors) {
  int64_t product = 1;
  for (const int64_t factor : factors) {
    if (factor < 0) return std::nullopt;
    if (factor != 0 && product > std::numeric_limits<int64_t>::max() / factor) {
      return std::nullopt;
    }
    product *= factor;
  }
  return product;
}

// Mamba 与 FourierMamba 共享工厂期 shape 算术合同。数组参数避免四个相邻
// int64 形参可误置换；返回 false 时调用方必须用占位 children 并在 build
// 首个图改动前返回 InvalidArgument。
bool mamba_shape_arithmetic_valid(const std::array<int64_t, 4>& config) {
  const int64_t batch = config[0];
  const int64_t channels = config[1];
  const int64_t steps = config[2];
  const int64_t kernel_size = config[3];
  if (batch <= 0 || channels <= 0 || steps <= 0 || kernel_size <= 0) return false;

  const int64_t padding = kernel_size - 1;
  if (padding > std::numeric_limits<int64_t>::max() / 2 ||
      steps > std::numeric_limits<int64_t>::max() - 2 * padding) {
    return false;
  }
  const int64_t conv_steps = steps + padding;
  return checked_nonnegative_product({batch, steps}).has_value() &&
         checked_nonnegative_product({batch, channels, steps}).has_value() &&
         checked_nonnegative_product({channels, kernel_size}).has_value() &&
         checked_nonnegative_product({channels, channels}).has_value() &&
         checked_nonnegative_product({batch, channels, conv_steps}).has_value();
}

// max_pool2d/avg_pool2d 工厂共用构建逻辑(REUSE-002):二者唯一差异是算子名
// 字符串,kernel/stride/padding 三个 kInt64Array 属性透传规则完全相同。
// 相邻同族形参(name/op_name 字符串对与三个 2 元数组)为本工厂固定契约序,
// 仅两个内部调用点且均以具名字面量传入,误置换会立即产生非法 op 名报错
// (同 Conv2d 签名论证)。
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
Module MakePool2dModule(std::string name, const std::string& op_name,
                        std::array<int64_t, 2> kernel_hw, std::array<int64_t, 2> stride_hw,
                        std::array<int64_t, 2> padding_hw) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  const std::vector<int64_t> kernel(kernel_hw.begin(), kernel_hw.end());
  const std::vector<int64_t> stride(stride_hw.begin(), stride_hw.end());
  const std::vector<int64_t> padding(padding_hw.begin(), padding_hw.end());

  BuildFn build_fn = [op_name, kernel, stride, padding](
                         Graph& graph, std::span<Value* const> inputs,
                         std::span<Value* const> /*params*/) -> Result<std::vector<Value*>> {
    if (inputs.size() != 1) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::" + op_name + " build() expects 1 input, got " + std::to_string(inputs.size()));
    }
    const AttrMap attrs{
        {"kernel", kernel},
        {"stride", stride},
        {"padding", padding},
    };
    const Result<Node*> pool_node =
        create_node_with_inferred_types(graph, op_name, {inputs[0]}, attrs);
    if (!pool_node.is_ok()) {
      return pool_node.status();
    }
    return std::vector<Value*>{pool_node.value()->output(0)};
  };

  return Module{std::move(name), {}, {}, std::move(build_fn)};
}

// 通用 children params 先序分段切片(M22 批4 T5,ARCH-071 切片不变式的通用
// 形式):按各 child parameters().size() 顺序切 params 段,返回逐子模块 span
// 清单。供本文件 MultiheadAttention/TransformerEncoderBlock 共用(REUSE-002:
// 二者子模块数分别为 4/5,不能照抄 AFF 的两段特判写法;Sequential/AFF 既有
// 实现分属各自闭包体、且均非本批改动范围,不在此合并,仅本批新增的两处调用点
// 共用本函数,避免本文件内部产出第二份复制)。
Result<std::vector<std::span<Value* const>>> SliceChildParams(const std::vector<Module>& children,
                                                              std::span<Value* const> params,
                                                              const std::string& caller_name) {
  std::vector<std::span<Value* const>> slices;
  slices.reserve(children.size());
  size_t offset = 0;
  for (const Module& child : children) {
    const size_t count = child.parameters().size();
    if (offset + count > params.size()) {
      // Module::build() 已校验 params.size() == parameters().size()(=各子
      // parameters().size() 之和),故此分支理论上不可达;仍保留防御性检查,
      // 不静默越界(同 Sequential/AFF 先例,ARCH-031 口径)。
      return Status::make(
          ErrorCode::kInternal,
          "nn::" + caller_name + " build() params slicing overruns for child '" + child.name + "'");
    }
    slices.push_back(params.subspan(offset, count));
    offset += count;
  }
  return slices;
}

}  // namespace

// batch/in_dim/out_dim 三个相邻同型形参语义上不可合并(nn-design.md §2 定稿
// 签名,ARCH-073 工厂期静态形状特化);全部调用点以具名实参传入,误置换会在
// build 期 shape 推断立即报错,抑制而非重排签名(同 EagerLaunch 先例)。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Module Linear(std::string name, int64_t batch, int64_t in_dim, int64_t out_dim, bool with_bias,
              DType dtype) {
  std::vector<ParamSpec> params;

  ParamSpec weight;
  weight.name = "weight";
  weight.type = MakeCpuTensorType(dtype, {in_dim, out_dim});
  // TODO(FRAME-DESIGN): Linear 工厂签名未透传参数初始化范围(lo/hi),故
  //   weight/bias 的 InitSpec 恒为默认 kUniformSeeded{0,0} 占位;M20 内数值
  //   物化仍走既有 frontend GenerateHostTensorValues 路径(ARCH-073),本字段
  //   何时/如何与 ModelSpec.param_init 对接留待 Task 4 或后续批次裁定。参考:
  //   docs/architecture/nn-design.md ARCH-073。完成判据:Task 4(或后续批次)
  //   为 Linear 等工厂补上初始化范围形参并据此填充 ParamSpec.init,或明确
  //   裁定该字段维持默认值、由调用方另行覆盖。
  params.push_back(weight);

  if (with_bias) {
    ParamSpec bias;
    bias.name = "bias";
    // bias 的 TensorType = [batch, out_dim]:与输出(matmul 结果)同形,照抄
    // 现 lowering(见 include/frame/nn/layers.h 头注释与
    // src/frontend/lowering.cpp::AppendLayer)。
    bias.type = MakeCpuTensorType(dtype, {batch, out_dim});
    params.push_back(bias);
  }

  BuildFn build_fn = [with_bias](Graph& graph, std::span<Value* const> inputs,
                                 std::span<Value* const> params) -> Result<std::vector<Value*>> {
    if (inputs.size() != 1) {
      return Status::make(ErrorCode::kInvalidArgument, "nn::Linear build() expects 1 input, got " +
                                                           std::to_string(inputs.size()));
    }

    const Result<Node*> matmul_node =
        create_node_with_inferred_types(graph, "matmul", {inputs[0], params[0]});
    if (!matmul_node.is_ok()) {
      return matmul_node.status();
    }
    Value* current = matmul_node.value()->output(0);

    if (with_bias) {
      const Result<Node*> add_node =
          create_node_with_inferred_types(graph, "add", {current, params[1]});
      if (!add_node.is_ok()) {
        return add_node.status();
      }
      current = add_node.value()->output(0);
    }

    return std::vector<Value*>{current};
  };

  return Module{std::move(name), std::move(params), {}, std::move(build_fn)};
}

Module Relu(std::string name) {
  BuildFn build_fn = [](Graph& graph, std::span<Value* const> inputs,
                        std::span<Value* const> /*params*/) -> Result<std::vector<Value*>> {
    if (inputs.size() != 1) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::Relu build() expects 1 input, got " + std::to_string(inputs.size()));
    }
    const Result<Node*> relu_node = create_node_with_inferred_types(graph, "relu", {inputs[0]});
    if (!relu_node.is_ok()) {
      return relu_node.status();
    }
    return std::vector<Value*>{relu_node.value()->output(0)};
  };
  return Module{std::move(name), {}, {}, std::move(build_fn)};
}

Module Sequential(std::string name, std::vector<Module> children) {
  // build_fn 闭包按值捕获子树副本:BuildFn 签名(ARCH-071)不携带 Module 自身
  // 上下文,组合模块须自持子树以便按 parameters().size() 分段切 params。
  const std::vector<Module> children_for_build = children;
  // inputs/params 相邻同型 span 形参是 BuildFn 契约固定形态(ARCH-071),
  // 语义上不可合并/重排;误置换会在切片尺寸校验与 shape 推断立即暴露。
  // NOLINTBEGIN(bugprone-easily-swappable-parameters)
  BuildFn build_fn = [children_for_build](
                         Graph& graph, std::span<Value* const> inputs,
                         std::span<Value* const> params) -> Result<std::vector<Value*>> {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    std::vector<Value*> current(inputs.begin(), inputs.end());
    size_t param_offset = 0;
    for (const Module& child : children_for_build) {
      const size_t child_param_count = child.parameters().size();
      if (param_offset + child_param_count > params.size()) {
        // Module::build() 已校验 params.size() == parameters().size()(自身
        // 无直接参数,故等于全体子模块 parameters().size() 之和),故此分支
        // 理论上不可达;仍保留防御性检查,不静默越界(ARCH-031 口径)。
        return Status::make(
            ErrorCode::kInternal,
            "nn::Sequential build() params slicing overruns for child '" + child.name + "'");
      }
      const std::span<Value* const> child_params = params.subspan(param_offset, child_param_count);
      const Result<std::vector<Value*>> child_result = child.build(graph, current, child_params);
      if (!child_result.is_ok()) {
        return child_result.status();
      }
      current = child_result.value();
      param_offset += child_param_count;
    }
    return current;
  };
  return Module{std::move(name), {}, std::move(children), std::move(build_fn)};
}

Module MseLoss(std::string name) {
  BuildFn build_fn = [](Graph& graph, std::span<Value* const> inputs,
                        std::span<Value* const> /*params*/) -> Result<std::vector<Value*>> {
    if (inputs.size() != 2) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::MseLoss build() expects 2 inputs (pred, target), got " +
                              std::to_string(inputs.size()));
    }
    const Result<Node*> loss_node =
        create_node_with_inferred_types(graph, "mse_loss", {inputs[0], inputs[1]});
    if (!loss_node.is_ok()) {
      return loss_node.status();
    }
    return std::vector<Value*>{loss_node.value()->output(0)};
  };
  return Module{std::move(name), {}, {}, std::move(build_fn)};
}

// ---------------------------------------------------------------------------
// M21 批3 T6:卷积批次工厂(docs/plan/2026-07-18-batch3-m21-conv.md 第1.4节)。
// ---------------------------------------------------------------------------

// in_channels/out_channels/kernel_hw/stride_hw/padding_hw/groups 六个相邻
// 形参语义上不可合并(计划 1.4 节定稿签名,对齐 Linear 的既有裁决:ARCH-073
// 工厂期静态形状特化);全部调用点以具名实参传入,误置换会在 build 期 shape
// 推断立即报错(同 Linear 头注释先例)。
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
Module Conv2d(std::string name, int64_t in_channels, int64_t out_channels,
              std::array<int64_t, 2> kernel_hw, std::array<int64_t, 2> stride_hw,
              std::array<int64_t, 2> padding_hw, int64_t groups, bool with_bias, DType dtype) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  std::vector<ParamSpec> params;

  // groups<=0 是非法输入(conv2d 算子 shape 推断会在 build() 期拒绝并报错,
  // 见 src/ops/schemas/conv.cpp::compute_conv2d_geometry);此处仅避免整数除法
  // 除零的未定义行为,不重复该算子已有的校验逻辑(错误仍在 build() 期经
  // Result 正常传播,ARCH-031 口径)。
  const int64_t cin_per_group = groups > 0 ? in_channels / groups : 0;

  ParamSpec weight;
  weight.name = "weight";
  weight.type = MakeCpuTensorType(dtype, {out_channels, cin_per_group, kernel_hw[0], kernel_hw[1]});
  // TODO(FRAME-DESIGN): 同 Linear 工厂的既有缺口(src/nn/layers.cpp::Linear
  //   头注释)——Conv2d 签名未透传参数初始化范围,weight/bias 的 InitSpec 恒为
  //   默认占位。参考:docs/architecture/nn-design.md ARCH-073。完成判据:后续
  //   批次为 Conv2d 等工厂补上初始化范围形参并据此填充 ParamSpec.init,或明确
  //   裁定该字段维持默认值、由调用方另行覆盖。
  params.push_back(weight);

  if (with_bias) {
    ParamSpec bias;
    bias.name = "bias";
    // bias 的 TensorType = [out_channels](裁决点①:conv2d 算子内可选第三
    // 输入,通道维在算子语义内广播,与 Linear 的 [batch,out_dim] 全形 bias
    // 口径不同——CNN 语义要求 bias 跨 batch/空间共享,计划 1.4 节)。
    bias.type = MakeCpuTensorType(dtype, {out_channels});
    params.push_back(bias);
  }

  const std::vector<int64_t> stride(stride_hw.begin(), stride_hw.end());
  const std::vector<int64_t> padding(padding_hw.begin(), padding_hw.end());

  BuildFn build_fn = [with_bias, stride, padding, groups](
                         Graph& graph, std::span<Value* const> inputs,
                         std::span<Value* const> params) -> Result<std::vector<Value*>> {
    if (inputs.size() != 1) {
      return Status::make(ErrorCode::kInvalidArgument, "nn::Conv2d build() expects 1 input, got " +
                                                           std::to_string(inputs.size()));
    }

    std::vector<Value*> conv_inputs{inputs[0], params[0]};
    if (with_bias) conv_inputs.push_back(params[1]);
    const AttrMap attrs{
        {"stride", stride},
        {"padding", padding},
        {"groups", groups},
    };
    const Result<Node*> conv_node =
        create_node_with_inferred_types(graph, "conv2d", conv_inputs, attrs);
    if (!conv_node.is_ok()) {
      return conv_node.status();
    }
    return std::vector<Value*>{conv_node.value()->output(0)};
  };

  return Module{std::move(name), std::move(params), {}, std::move(build_fn)};
}

// in_channels/out_channels/kernel/stride/padding/groups 六个相邻形参同
// Conv2d 头注释论证,语义上不可合并。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Module Conv1d(std::string name, int64_t in_channels, int64_t out_channels, int64_t kernel,
              int64_t stride, int64_t padding, int64_t groups, bool with_bias, DType dtype) {
  std::vector<ParamSpec> params;

  // 除零防御同 Conv2d 头注释(groups<=0 由 conv1d 算子 shape 推断在 build()
  // 期拒绝)。
  const int64_t cin_per_group = groups > 0 ? in_channels / groups : 0;

  ParamSpec weight;
  weight.name = "weight";
  weight.type = MakeCpuTensorType(dtype, {out_channels, cin_per_group, kernel});
  // TODO(FRAME-DESIGN): 同 Linear 工厂的既有缺口(src/nn/layers.cpp::Linear
  //   头注释)——Conv1d 签名未透传参数初始化范围,weight/bias 的 InitSpec 恒为
  //   默认占位。参考:docs/architecture/nn-design.md ARCH-073。完成判据:后续
  //   批次为 Conv1d 等工厂补上初始化范围形参并据此填充 ParamSpec.init,或明确
  //   裁定该字段维持默认值、由调用方另行覆盖。
  params.push_back(weight);

  if (with_bias) {
    ParamSpec bias;
    bias.name = "bias";
    bias.type = MakeCpuTensorType(dtype, {out_channels});
    params.push_back(bias);
  }

  BuildFn build_fn = [with_bias, stride, padding, groups](
                         Graph& graph, std::span<Value* const> inputs,
                         std::span<Value* const> params) -> Result<std::vector<Value*>> {
    if (inputs.size() != 1) {
      return Status::make(ErrorCode::kInvalidArgument, "nn::Conv1d build() expects 1 input, got " +
                                                           std::to_string(inputs.size()));
    }

    std::vector<Value*> conv_inputs{inputs[0], params[0]};
    if (with_bias) conv_inputs.push_back(params[1]);
    const AttrMap attrs{
        {"stride", stride},
        {"padding", padding},
        {"groups", groups},
    };
    const Result<Node*> conv_node =
        create_node_with_inferred_types(graph, "conv1d", conv_inputs, attrs);
    if (!conv_node.is_ok()) {
      return conv_node.status();
    }
    return std::vector<Value*>{conv_node.value()->output(0)};
  };

  return Module{std::move(name), std::move(params), {}, std::move(build_fn)};
}

Module MaxPool2d(std::string name, std::array<int64_t, 2> kernel_hw,
                 std::array<int64_t, 2> stride_hw, std::array<int64_t, 2> padding_hw) {
  return MakePool2dModule(std::move(name), "max_pool2d", kernel_hw, stride_hw, padding_hw);
}

Module AvgPool2d(std::string name, std::array<int64_t, 2> kernel_hw,
                 std::array<int64_t, 2> stride_hw, std::array<int64_t, 2> padding_hw) {
  return MakePool2dModule(std::move(name), "avg_pool2d", kernel_hw, stride_hw, padding_hw);
}

Module Sigmoid(std::string name) {
  BuildFn build_fn = [](Graph& graph, std::span<Value* const> inputs,
                        std::span<Value* const> /*params*/) -> Result<std::vector<Value*>> {
    if (inputs.size() != 1) {
      return Status::make(ErrorCode::kInvalidArgument, "nn::Sigmoid build() expects 1 input, got " +
                                                           std::to_string(inputs.size()));
    }
    const Result<Node*> sigmoid_node =
        create_node_with_inferred_types(graph, "sigmoid", {inputs[0]});
    if (!sigmoid_node.is_ok()) {
      return sigmoid_node.status();
    }
    return std::vector<Value*>{sigmoid_node.value()->output(0)};
  };
  return Module{std::move(name), {}, {}, std::move(build_fn)};
}

Module Flatten(std::string name) {
  BuildFn build_fn = [](Graph& graph, std::span<Value* const> inputs,
                        std::span<Value* const> /*params*/) -> Result<std::vector<Value*>> {
    if (inputs.size() != 1) {
      return Status::make(ErrorCode::kInvalidArgument, "nn::Flatten build() expects 1 input, got " +
                                                           std::to_string(inputs.size()));
    }
    const Shape& x_shape = inputs[0]->type().shape;
    if (x_shape.rank() < 1) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::Flatten build() requires input rank >= 1, got " + x_shape.to_string());
    }
    const int64_t n = x_shape.dim(0);
    int64_t prod = 1;
    for (int64_t i = 1; i < x_shape.rank(); ++i) {
      prod *= x_shape.dim(i);
    }
    const Shape target_shape({n, prod});
    const AttrMap attrs{{"target_shape", target_shape}};
    const Result<Node*> reshape_node =
        create_node_with_inferred_types(graph, "reshape", {inputs[0]}, attrs);
    if (!reshape_node.is_ok()) {
      return reshape_node.status();
    }
    return std::vector<Value*>{reshape_node.value()->output(0)};
  };
  return Module{std::move(name), {}, {}, std::move(build_fn)};
}

Module AFF(std::string name, int64_t channels, DType dtype) {
  constexpr std::array<int64_t, 2> kPointwiseKernel{1, 1};
  constexpr std::array<int64_t, 2> kUnitStride{1, 1};
  constexpr std::array<int64_t, 2> kNoPadding{0, 0};
  // 两个 1x1 Conv2d 子模块(注意力分支:c1 -> relu -> c2 -> sigmoid,命名与
  // 计划 1.4 节公式 M=sigmoid(c2(relu(c1(X+Y)))) 逐一对应)。with_bias=true 是
  // 计划未明确处的工程取舍(常规卷积基线配置,无 BN 层可吸收 bias)。
  std::vector<Module> children{
      Conv2d("c1", channels, channels, kPointwiseKernel, kUnitStride, kNoPadding, /*groups=*/1,
             /*with_bias=*/true, dtype),
      Conv2d("c2", channels, channels, kPointwiseKernel, kUnitStride, kNoPadding, /*groups=*/1,
             /*with_bias=*/true, dtype)};
  // build_fn 闭包按值捕获子树副本,同 Sequential 头注释先例(BuildFn 签名不
  // 携带 Module 自身上下文)。
  const std::vector<Module> children_for_build = children;

  BuildFn build_fn = [dtype, children_for_build](
                         Graph& graph, std::span<Value* const> inputs,
                         std::span<Value* const> params) -> Result<std::vector<Value*>> {
    if (inputs.size() != 2) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::AFF build() expects 2 inputs (X, Y), got " + std::to_string(inputs.size()));
    }
    Value* x = inputs[0];
    Value* y = inputs[1];

    const Module& c1 = children_for_build[0];
    const Module& c2 = children_for_build[1];
    const size_t c1_param_count = c1.parameters().size();
    if (c1_param_count > params.size()) {
      // Module::build() 已校验 params.size() == parameters().size(),此分支
      // 理论上不可达;仍保留防御性检查,不静默越界(同 Sequential 先例)。
      return Status::make(ErrorCode::kInternal,
                          "nn::AFF build() params slicing overruns for child 'c1'");
    }
    const std::span<Value* const> c1_params = params.subspan(0, c1_param_count);
    const std::span<Value* const> c2_params =
        params.subspan(c1_param_count, params.size() - c1_param_count);

    const Result<Node*> sum_node = create_node_with_inferred_types(graph, "add", {x, y});
    if (!sum_node.is_ok()) return sum_node.status();

    const Result<std::vector<Value*>> t1 =
        c1.build(graph, std::vector<Value*>{sum_node.value()->output(0)}, c1_params);
    if (!t1.is_ok()) return t1.status();

    const Result<Node*> relu_node = create_node_with_inferred_types(graph, "relu", {t1.value()[0]});
    if (!relu_node.is_ok()) return relu_node.status();

    const Result<std::vector<Value*>> t2 =
        c2.build(graph, std::vector<Value*>{relu_node.value()->output(0)}, c2_params);
    if (!t2.is_ok()) return t2.status();

    const Result<Node*> sigmoid_node =
        create_node_with_inferred_types(graph, "sigmoid", {t2.value()[0]});
    if (!sigmoid_node.is_ok()) return sigmoid_node.status();
    Value* m = sigmoid_node.value()->output(0);

    // one_minus_M = add(ones, mul(neg_one, M))(无 sub 算子,经两个 constant
    // splat 系数 1/-1 与 mul/add 组合,计划 1.4 节)。splat 的 shape/device 取
    // 自 X 的静态类型(AFF 设计上 M 与 X 同形:c1/c2 均 channels->channels
    // 1x1/stride1/pad0 卷积,空间维不变),dtype 取工厂自身 dtype 形参。
    const Shape& splat_shape = x->type().shape;
    const Device splat_device = x->type().device;

    const Result<Node*> ones_node =
        make_constant_splat(graph, splat_shape, dtype, splat_device, 1.0);
    if (!ones_node.is_ok()) return ones_node.status();
    const Result<Node*> neg_one_node =
        make_constant_splat(graph, splat_shape, dtype, splat_device, -1.0);
    if (!neg_one_node.is_ok()) return neg_one_node.status();

    const Result<Node*> neg_m_node =
        create_node_with_inferred_types(graph, "mul", {neg_one_node.value()->output(0), m});
    if (!neg_m_node.is_ok()) return neg_m_node.status();
    const Result<Node*> one_minus_m_node = create_node_with_inferred_types(
        graph, "add", {ones_node.value()->output(0), neg_m_node.value()->output(0)});
    if (!one_minus_m_node.is_ok()) return one_minus_m_node.status();

    const Result<Node*> mx_node = create_node_with_inferred_types(graph, "mul", {m, x});
    if (!mx_node.is_ok()) return mx_node.status();
    const Result<Node*> my_node =
        create_node_with_inferred_types(graph, "mul", {one_minus_m_node.value()->output(0), y});
    if (!my_node.is_ok()) return my_node.status();

    const Result<Node*> out_node = create_node_with_inferred_types(
        graph, "add", {mx_node.value()->output(0), my_node.value()->output(0)});
    if (!out_node.is_ok()) return out_node.status();

    return std::vector<Value*>{out_node.value()->output(0)};
  };

  return Module{std::move(name), {}, std::move(children), std::move(build_fn)};
}

Module Dwt2d(std::string name, int64_t channels, WaveletKind wavelet_kind) {
  BuildFn build_fn = [channels, wavelet_kind](
                         Graph& graph, std::span<Value* const> inputs,
                         std::span<Value* const> /*params*/) -> Result<std::vector<Value*>> {
    if (inputs.size() != 1) {
      return Status::make(ErrorCode::kInvalidArgument, "nn::Dwt2d build() expects 1 input, got " +
                                                           std::to_string(inputs.size()));
    }
    if (wavelet_kind != WaveletKind::kHaar) {
      // v1 Dwt2d 仅支持 kHaar;kDb4 需要可分离两趟卷积,留后续批次扩展
      // (计划 1.4 节:「2d db4(可分离两趟)留后续需要时扩 kind」)。
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::Dwt2d build() only supports wavelet_kind kHaar in v1 (kDb4 is reserved for a "
          "future 2d separable extension)");
    }

    // 二维 Haar 正交归一系数(2x2 核,行优先展平 [k(0,0),k(0,1),k(1,0),k(1,1)],
    // 与 cross-correlation 卷积语义一致:y[oh,ow]=sum_{kh,kw} w[kh,kw]*
    // x[oh*s+kh, ow*s+kw],见 src/backends/cpu/kernels/conv.cpp),子带次序
    // LL/LH/HL/HH(计划 1.4 节字面量原文)。
    constexpr std::array<double, 4> kLL{0.5, 0.5, 0.5, 0.5};
    constexpr std::array<double, 4> kLH{0.5, 0.5, -0.5, -0.5};
    constexpr std::array<double, 4> kHL{0.5, -0.5, 0.5, -0.5};
    constexpr std::array<double, 4> kHH{0.5, -0.5, -0.5, 0.5};

    // 输出通道交织:groups=channels、Cout=4*channels 下,分组卷积第 g 组的
    // Cout/groups=4 个输出通道恰对应输入通道 g(g=0..channels-1);故按"通道
    // 外层、子带内层(LL,LH,HL,HH)"顺序展平 value,天然得到交织布局(计划
    // 1.4 节"四子带交织通道维免 concat"),无需额外 concat 节点。
    std::vector<double> filter_values;
    filter_values.reserve(static_cast<size_t>(channels) * 16);
    for (int64_t c = 0; c < channels; ++c) {
      for (double v : kLL) filter_values.push_back(v);
      for (double v : kLH) filter_values.push_back(v);
      for (double v : kHL) filter_values.push_back(v);
      for (double v : kHH) filter_values.push_back(v);
    }

    const Shape filter_shape({4 * channels, 1, 2, 2});
    const DType filter_dtype = inputs[0]->type().dtype;
    const Device filter_device = inputs[0]->type().device;
    const AttrMap const_attrs{
        {"value", filter_values},
        {"shape", filter_shape},
        {"dtype", filter_dtype},
    };
    const Result<Node*> filter_node =
        create_node_with_inferred_types(graph, kConstantOpName, filter_device, const_attrs);
    if (!filter_node.is_ok()) return filter_node.status();

    const AttrMap conv_attrs{
        {"stride", std::vector<int64_t>{2, 2}},
        {"padding", std::vector<int64_t>{0, 0}},
        {"groups", channels},
    };
    const Result<Node*> conv_node = create_node_with_inferred_types(
        graph, "conv2d", {inputs[0], filter_node.value()->output(0)}, conv_attrs);
    if (!conv_node.is_ok()) return conv_node.status();

    return std::vector<Value*>{conv_node.value()->output(0)};
  };

  return Module{std::move(name), {}, {}, std::move(build_fn)};
}

Module Dwt1d(std::string name, int64_t channels, WaveletKind wavelet_kind) {
  BuildFn build_fn = [channels, wavelet_kind](
                         Graph& graph, std::span<Value* const> inputs,
                         std::span<Value* const> /*params*/) -> Result<std::vector<Value*>> {
    if (inputs.size() != 1) {
      return Status::make(ErrorCode::kInvalidArgument, "nn::Dwt1d build() expects 1 input, got " +
                                                           std::to_string(inputs.size()));
    }

    std::vector<double> low;
    std::vector<double> high;
    if (wavelet_kind == WaveletKind::kHaar) {
      // 1D Haar 正交归一系数(K=2):低通 [1/sqrt(2), 1/sqrt(2)]、高通
      // [1/sqrt(2), -1/sqrt(2)]。
      const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
      low = {inv_sqrt2, inv_sqrt2};
      high = {inv_sqrt2, -inv_sqrt2};
    } else if (wavelet_kind == WaveletKind::kDb4) {
      // 标准 Daubechies-4(db2,4 抽头)尺度函数(低通)系数闭式公式:
      // h0=(1+sqrt(3))/(4*sqrt(2))、h1=(3+sqrt(3))/(4*sqrt(2))、
      // h2=(3-sqrt(3))/(4*sqrt(2))、h3=(1-sqrt(3))/(4*sqrt(2))(2 阶消失矩正交
      // 小波族闭式解);对应小波(高通)系数经正交镜像滤波器关系
      // g[k]=(-1)^k * h[3-k] 导出(标准 QMF 构造,保证低通/高通正交)。
      const double sqrt3 = std::sqrt(3.0);
      const double denom = 4.0 * std::sqrt(2.0);
      const double h0 = (1.0 + sqrt3) / denom;
      const double h1 = (3.0 + sqrt3) / denom;
      const double h2 = (3.0 - sqrt3) / denom;
      const double h3 = (1.0 - sqrt3) / denom;
      low = {h0, h1, h2, h3};
      high = {h3, -h2, h1, -h0};
    } else {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::Dwt1d build() got an unrecognized wavelet_kind value");
    }

    const int64_t k = static_cast<int64_t>(low.size());

    // 输出通道交织:groups=channels、Cout=2*channels 下,同 Dwt2d 头注释推理,
    // 按"通道外层、子带内层(低通,高通)"顺序展平 value 天然得到交织布局。
    std::vector<double> filter_values;
    filter_values.reserve(static_cast<size_t>(channels) * static_cast<size_t>(2 * k));
    for (int64_t c = 0; c < channels; ++c) {
      for (double v : low) filter_values.push_back(v);
      for (double v : high) filter_values.push_back(v);
    }

    const Shape filter_shape({2 * channels, 1, k});
    const DType filter_dtype = inputs[0]->type().dtype;
    const Device filter_device = inputs[0]->type().device;
    const AttrMap const_attrs{
        {"value", filter_values},
        {"shape", filter_shape},
        {"dtype", filter_dtype},
    };
    const Result<Node*> filter_node =
        create_node_with_inferred_types(graph, kConstantOpName, filter_device, const_attrs);
    if (!filter_node.is_ok()) return filter_node.status();

    constexpr int64_t kStride = 2;
    constexpr int64_t kPadding = 0;
    const AttrMap conv_attrs{
        {"stride", kStride},
        {"padding", kPadding},
        {"groups", channels},
    };
    const Result<Node*> conv_node = create_node_with_inferred_types(
        graph, "conv1d", {inputs[0], filter_node.value()->output(0)}, conv_attrs);
    if (!conv_node.is_ok()) return conv_node.status();

    return std::vector<Value*>{conv_node.value()->output(0)};
  };

  return Module{std::move(name), {}, {}, std::move(build_fn)};
}

// ---------------------------------------------------------------------------
// M22 批4 T5:序列批次工厂(docs/plan/2026-07-19-batch4-m22-seq.md §1.7,
// design-reviewer 两轮 APPROVE)。
// ---------------------------------------------------------------------------

// dim/eps 两个相邻形参语义上不可合并(§1.7 定稿签名):误置换会在 build() 期
// gamma/beta 形状不符(dim 误传入 eps 槽位)或 eps<=0 校验(eps 误传入 dim
// 槽位,int64_t->double 隐式转换后大概率非法)立即报错,同 Linear/Conv2d 头
// 注释论证。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Module LayerNorm(std::string name, int64_t dim, double eps, DType dtype) {
  std::vector<ParamSpec> params;

  ParamSpec gamma;
  gamma.name = "gamma";
  gamma.type = MakeCpuTensorType(dtype, {dim});
  params.push_back(gamma);

  ParamSpec beta;
  beta.name = "beta";
  beta.type = MakeCpuTensorType(dtype, {dim});
  params.push_back(beta);

  BuildFn build_fn = [eps](Graph& graph, std::span<Value* const> inputs,
                           std::span<Value* const> params) -> Result<std::vector<Value*>> {
    if (inputs.size() != 1) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::LayerNorm build() expects 1 input, got " + std::to_string(inputs.size()));
    }
    // params 尺寸(==2)已由 Module::build() 校验(ARCH-071),此处直接按位取用
    // gamma/beta;eps<=0 由 layer_norm 算子自身 shape_infer 在 build() 期拒绝
    // (不在本工厂重复校验)。
    const AttrMap attrs{{"eps", eps}};
    const Result<Node*> ln_node = create_node_with_inferred_types(
        graph, "layer_norm", {inputs[0], params[0], params[1]}, attrs);
    if (!ln_node.is_ok()) return ln_node.status();
    return std::vector<Value*>{ln_node.value()->output(0)};
  };

  return Module{std::move(name), std::move(params), {}, std::move(build_fn)};
}

// batch/num_steps/input_dim/hidden_dim 四个相邻 int64_t 形参语义上不可合并
// (§1.7 定稿签名,同 Linear/Conv2d 头注释论证:build() 期 slice/reshape/matmul
// 的 shape 推断会在误置换时立即报错)。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Module LSTM(std::string name, int64_t batch, int64_t num_steps, int64_t input_dim,
            int64_t hidden_dim, DType dtype) {
  std::vector<ParamSpec> params;

  ParamSpec w_ih;
  w_ih.name = "W_ih";
  w_ih.type = MakeCpuTensorType(dtype, {input_dim, 4 * hidden_dim});
  params.push_back(w_ih);

  ParamSpec w_hh;
  w_hh.name = "W_hh";
  w_hh.type = MakeCpuTensorType(dtype, {hidden_dim, 4 * hidden_dim});
  params.push_back(w_hh);

  ParamSpec bias;
  bias.name = "bias";
  // bias 的 TensorType = [batch, 4*hidden_dim]:Linear bias 同形先例(§1.7)。
  bias.type = MakeCpuTensorType(dtype, {batch, 4 * hidden_dim});
  params.push_back(bias);

  // inputs/params 相邻同型 span 形参是 BuildFn 契约固定形态(ARCH-071),同
  // Sequential 头注释先例。
  // NOLINTBEGIN(bugprone-easily-swappable-parameters)
  BuildFn build_fn = [batch, num_steps, input_dim, hidden_dim, dtype](
                         Graph& graph, std::span<Value* const> inputs,
                         std::span<Value* const> params) -> Result<std::vector<Value*>> {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    if (inputs.size() != 1) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::LSTM build() expects 1 input, got " + std::to_string(inputs.size()));
    }
    Value* x = inputs[0];
    Value* w_ih_v = params[0];
    Value* w_hh_v = params[1];
    Value* bias_v = params[2];
    const Device device = x->type().device;

    // 初始隐状态与细胞状态置零:h0/c0 = constant(0) splat [B,H](§1.7)。
    const Shape h_shape({batch, hidden_dim});
    const Result<Node*> h0_node = make_constant_splat(graph, h_shape, dtype, device, 0.0);
    if (!h0_node.is_ok()) return h0_node.status();
    const Result<Node*> c0_node = make_constant_splat(graph, h_shape, dtype, device, 0.0);
    if (!c0_node.is_ok()) return c0_node.status();
    Value* h = h0_node.value()->output(0);
    Value* c = c0_node.value()->output(0);

    for (int64_t t = 0; t < num_steps; ++t) {
      // 取第 t 步输入切片并展平:x_t = reshape(slice(x, axis=1, t, t+1), [B,E])。
      const AttrMap x_slice_attrs{{"axis", int64_t{1}}, {"start", t}, {"stop", t + 1}};
      const Result<Node*> x_slice_node =
          create_node_with_inferred_types(graph, "slice", {x}, x_slice_attrs);
      if (!x_slice_node.is_ok()) return x_slice_node.status();
      const AttrMap x_t_reshape_attrs{{"target_shape", Shape({batch, input_dim})}};
      const Result<Node*> x_t_node = create_node_with_inferred_types(
          graph, "reshape", {x_slice_node.value()->output(0)}, x_t_reshape_attrs);
      if (!x_t_node.is_ok()) return x_t_node.status();
      Value* x_t = x_t_node.value()->output(0);

      // 门控预激活线性变换:z = x_t·W_ih + h·W_hh + bias。
      const Result<Node*> xw_node = create_node_with_inferred_types(graph, "matmul", {x_t, w_ih_v});
      if (!xw_node.is_ok()) return xw_node.status();
      const Result<Node*> hw_node = create_node_with_inferred_types(graph, "matmul", {h, w_hh_v});
      if (!hw_node.is_ok()) return hw_node.status();
      const Result<Node*> sum_node = create_node_with_inferred_types(
          graph, "add", {xw_node.value()->output(0), hw_node.value()->output(0)});
      if (!sum_node.is_ok()) return sum_node.status();
      const Result<Node*> z_node =
          create_node_with_inferred_types(graph, "add", {sum_node.value()->output(0), bias_v});
      if (!z_node.is_ok()) return z_node.status();
      Value* z = z_node.value()->output(0);

      // 门序 i,f,g,o 固定(§1.7 原文),各门沿 axis=1 连续切列
      // [0,H)/[H,2H)/[2H,3H)/[3H,4H)。
      const AttrMap i_slice_attrs{
          {"axis", int64_t{1}}, {"start", int64_t{0}}, {"stop", hidden_dim}};
      const Result<Node*> i_pre_node =
          create_node_with_inferred_types(graph, "slice", {z}, i_slice_attrs);
      if (!i_pre_node.is_ok()) return i_pre_node.status();
      const AttrMap f_slice_attrs{
          {"axis", int64_t{1}}, {"start", hidden_dim}, {"stop", 2 * hidden_dim}};
      const Result<Node*> f_pre_node =
          create_node_with_inferred_types(graph, "slice", {z}, f_slice_attrs);
      if (!f_pre_node.is_ok()) return f_pre_node.status();
      const AttrMap g_slice_attrs{
          {"axis", int64_t{1}}, {"start", 2 * hidden_dim}, {"stop", 3 * hidden_dim}};
      const Result<Node*> g_pre_node =
          create_node_with_inferred_types(graph, "slice", {z}, g_slice_attrs);
      if (!g_pre_node.is_ok()) return g_pre_node.status();
      const AttrMap o_slice_attrs{
          {"axis", int64_t{1}}, {"start", 3 * hidden_dim}, {"stop", 4 * hidden_dim}};
      const Result<Node*> o_pre_node =
          create_node_with_inferred_types(graph, "slice", {z}, o_slice_attrs);
      if (!o_pre_node.is_ok()) return o_pre_node.status();

      const Result<Node*> i_node =
          create_node_with_inferred_types(graph, "sigmoid", {i_pre_node.value()->output(0)});
      if (!i_node.is_ok()) return i_node.status();
      const Result<Node*> f_node =
          create_node_with_inferred_types(graph, "sigmoid", {f_pre_node.value()->output(0)});
      if (!f_node.is_ok()) return f_node.status();
      const Result<Node*> g_node =
          create_node_with_inferred_types(graph, "tanh", {g_pre_node.value()->output(0)});
      if (!g_node.is_ok()) return g_node.status();
      const Result<Node*> o_node =
          create_node_with_inferred_types(graph, "sigmoid", {o_pre_node.value()->output(0)});
      if (!o_node.is_ok()) return o_node.status();

      // 更新细胞状态与隐状态:c = f⊙c + i⊙g;h = o⊙tanh(c)。
      const Result<Node*> fc_node =
          create_node_with_inferred_types(graph, "mul", {f_node.value()->output(0), c});
      if (!fc_node.is_ok()) return fc_node.status();
      const Result<Node*> ig_node = create_node_with_inferred_types(
          graph, "mul", {i_node.value()->output(0), g_node.value()->output(0)});
      if (!ig_node.is_ok()) return ig_node.status();
      const Result<Node*> c_new_node = create_node_with_inferred_types(
          graph, "add", {fc_node.value()->output(0), ig_node.value()->output(0)});
      if (!c_new_node.is_ok()) return c_new_node.status();
      c = c_new_node.value()->output(0);

      const Result<Node*> tanh_c_node = create_node_with_inferred_types(graph, "tanh", {c});
      if (!tanh_c_node.is_ok()) return tanh_c_node.status();
      const Result<Node*> h_new_node = create_node_with_inferred_types(
          graph, "mul", {o_node.value()->output(0), tanh_c_node.value()->output(0)});
      if (!h_new_node.is_ok()) return h_new_node.status();
      h = h_new_node.value()->output(0);
    }

    return std::vector<Value*>{h};
  };

  return Module{std::move(name), std::move(params), {}, std::move(build_fn)};
}

// batch/seq_len/embed_dim/num_heads 四个相邻 int64_t 形参语义上不可合并
// (§1.7 定稿签名,同 LSTM/Conv2d 头注释论证:build() 期 embed_dim%num_heads
// 校验与 slice/matmul 的 shape 推断会在误置换时立即报错)。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Module MultiheadAttention(std::string name, int64_t batch, int64_t seq_len, int64_t embed_dim,
                          int64_t num_heads, bool with_bias, DType dtype) {
  // children = 4 个 Linear(先序 q/k/v/o,§1.7),batch 形参=batch·seq_len、
  // in_dim=out_dim=embed_dim、with_bias 透传。
  std::vector<Module> children{
      Linear("q", batch * seq_len, embed_dim, embed_dim, with_bias, dtype),
      Linear("k", batch * seq_len, embed_dim, embed_dim, with_bias, dtype),
      Linear("v", batch * seq_len, embed_dim, embed_dim, with_bias, dtype),
      Linear("o", batch * seq_len, embed_dim, embed_dim, with_bias, dtype),
  };
  // build_fn 闭包按值捕获子树副本,同 Sequential/AFF 头注释先例(BuildFn 签名
  // 不携带 Module 自身上下文)。
  const std::vector<Module> children_for_build = children;

  // inputs/params 相邻同型 span 形参是 BuildFn 契约固定形态(ARCH-071),同
  // Sequential 头注释先例。
  // NOLINTBEGIN(bugprone-easily-swappable-parameters)
  BuildFn build_fn = [batch, seq_len, embed_dim, num_heads, dtype, children_for_build](
                         Graph& graph, std::span<Value* const> inputs,
                         std::span<Value* const> params) -> Result<std::vector<Value*>> {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    if (inputs.size() != 1) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::MultiheadAttention build() expects 1 input, got " + std::to_string(inputs.size()));
    }
    if (embed_dim % num_heads != 0) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::MultiheadAttention build() requires embed_dim to be divisible by num_heads, got "
          "embed_dim=" +
              std::to_string(embed_dim) + " num_heads=" + std::to_string(num_heads));
    }
    const int64_t dh = embed_dim / num_heads;
    Value* x = inputs[0];

    const Result<std::vector<std::span<Value* const>>> child_params =
        SliceChildParams(children_for_build, params, "MultiheadAttention");
    if (!child_params.is_ok()) return child_params.status();
    const std::span<Value* const> q_params = child_params.value()[0];
    const std::span<Value* const> k_params = child_params.value()[1];
    const std::span<Value* const> v_params = child_params.value()[2];
    const std::span<Value* const> o_params = child_params.value()[3];

    const Result<std::vector<Value*>> q_out =
        children_for_build[0].build(graph, std::vector<Value*>{x}, q_params);
    if (!q_out.is_ok()) return q_out.status();
    const Result<std::vector<Value*>> k_out =
        children_for_build[1].build(graph, std::vector<Value*>{x}, k_params);
    if (!k_out.is_ok()) return k_out.status();
    const Result<std::vector<Value*>> v_out =
        children_for_build[2].build(graph, std::vector<Value*>{x}, v_params);
    if (!v_out.is_ok()) return v_out.status();
    Value* q = q_out.value()[0];
    Value* k = k_out.value()[0];
    Value* v = v_out.value()[0];

    const Device device = x->type().device;
    const double scale = 1.0 / std::sqrt(static_cast<double>(dh));

    // per-(b,h) 静态展开(§1.7):行 slice[bS,(b+1)S) → 列 slice[hDh,(h+1)Dh)
    // 得 Q/K/V_bh[S,Dh];头沿 axis=1 concat → 批沿 axis=0 concat。
    std::vector<Value*> batch_outputs;
    batch_outputs.reserve(static_cast<size_t>(batch));
    for (int64_t b = 0; b < batch; ++b) {
      const int64_t row_start = b * seq_len;
      const int64_t row_stop = row_start + seq_len;
      const AttrMap row_slice_attrs{{"axis", int64_t{0}}, {"start", row_start}, {"stop", row_stop}};
      const Result<Node*> q_row =
          create_node_with_inferred_types(graph, "slice", {q}, row_slice_attrs);
      if (!q_row.is_ok()) return q_row.status();
      const Result<Node*> k_row =
          create_node_with_inferred_types(graph, "slice", {k}, row_slice_attrs);
      if (!k_row.is_ok()) return k_row.status();
      const Result<Node*> v_row =
          create_node_with_inferred_types(graph, "slice", {v}, row_slice_attrs);
      if (!v_row.is_ok()) return v_row.status();

      std::vector<Value*> head_outputs;
      head_outputs.reserve(static_cast<size_t>(num_heads));
      for (int64_t h = 0; h < num_heads; ++h) {
        const int64_t col_start = h * dh;
        const int64_t col_stop = col_start + dh;
        const AttrMap col_slice_attrs{
            {"axis", int64_t{1}}, {"start", col_start}, {"stop", col_stop}};
        const Result<Node*> q_bh = create_node_with_inferred_types(
            graph, "slice", {q_row.value()->output(0)}, col_slice_attrs);
        if (!q_bh.is_ok()) return q_bh.status();
        const Result<Node*> k_bh = create_node_with_inferred_types(
            graph, "slice", {k_row.value()->output(0)}, col_slice_attrs);
        if (!k_bh.is_ok()) return k_bh.status();
        const Result<Node*> v_bh = create_node_with_inferred_types(
            graph, "slice", {v_row.value()->output(0)}, col_slice_attrs);
        if (!v_bh.is_ok()) return v_bh.status();

        // 缩放点积注意力:scores = matmul(Q_bh, transpose(K_bh, [1,0]));
        // scaled = scores·constant(1/√Dh) splat;A = softmax(scaled);
        // O_bh = matmul(A, V_bh)。
        const AttrMap transpose_attrs{{"perm", std::vector<int64_t>{1, 0}}};
        const Result<Node*> k_bh_t = create_node_with_inferred_types(
            graph, "transpose", {k_bh.value()->output(0)}, transpose_attrs);
        if (!k_bh_t.is_ok()) return k_bh_t.status();

        const Result<Node*> scores_node = create_node_with_inferred_types(
            graph, "matmul", {q_bh.value()->output(0), k_bh_t.value()->output(0)});
        if (!scores_node.is_ok()) return scores_node.status();

        const Shape scores_shape = scores_node.value()->output(0)->type().shape;
        const Result<Node*> scale_node =
            make_constant_splat(graph, scores_shape, dtype, device, scale);
        if (!scale_node.is_ok()) return scale_node.status();
        const Result<Node*> scaled_node = create_node_with_inferred_types(
            graph, "mul", {scores_node.value()->output(0), scale_node.value()->output(0)});
        if (!scaled_node.is_ok()) return scaled_node.status();

        const Result<Node*> attn_node =
            create_node_with_inferred_types(graph, "softmax", {scaled_node.value()->output(0)});
        if (!attn_node.is_ok()) return attn_node.status();

        const Result<Node*> o_bh_node = create_node_with_inferred_types(
            graph, "matmul", {attn_node.value()->output(0), v_bh.value()->output(0)});
        if (!o_bh_node.is_ok()) return o_bh_node.status();

        head_outputs.push_back(o_bh_node.value()->output(0));
      }

      const AttrMap concat_heads_attrs{{"axis", int64_t{1}}};
      const Result<Node*> o_b_node =
          create_node_with_inferred_types(graph, "concat", head_outputs, concat_heads_attrs);
      if (!o_b_node.is_ok()) return o_b_node.status();
      batch_outputs.push_back(o_b_node.value()->output(0));
    }

    const AttrMap concat_batches_attrs{{"axis", int64_t{0}}};
    const Result<Node*> attn_out_node =
        create_node_with_inferred_types(graph, "concat", batch_outputs, concat_batches_attrs);
    if (!attn_out_node.is_ok()) return attn_out_node.status();

    // o 投影(§1.7 末步)。
    return children_for_build[3].build(graph, std::vector<Value*>{attn_out_node.value()->output(0)},
                                       o_params);
  };

  return Module{std::move(name), {}, std::move(children), std::move(build_fn)};
}

// batch/seq_len/embed_dim/num_heads/ffn_dim 五个相邻 int64_t 形参语义上不可
// 合并(§1.7 定稿签名,同 MultiheadAttention/Conv2d 头注释论证)。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Module TransformerEncoderBlock(std::string name, int64_t batch, int64_t seq_len, int64_t embed_dim,
                               int64_t num_heads, int64_t ffn_dim, bool with_bias, DType dtype) {
  // LayerNorm 的 eps 固定为 1e-5(§1.7 原文本工厂签名未透传该形参,采用
  // transformer 文献惯用默认值,layers.h 头注释)。
  constexpr double kLayerNormEps = 1e-5;
  const int64_t rows = batch * seq_len;

  // children=[mha, ln1, ffn1, ffn2, ln2](先序=构建执行序,§1.7:
  // y1=ln1(x+mha(x));y=ln2(y1+ffn2(relu(ffn1(y1))))。relu 无参数,内联构图、
  // 非独立 child(同 AFF 先例)。
  std::vector<Module> children{
      MultiheadAttention("mha", batch, seq_len, embed_dim, num_heads, with_bias, dtype),
      LayerNorm("ln1", embed_dim, kLayerNormEps, dtype),
      Linear("ffn1", rows, embed_dim, ffn_dim, with_bias, dtype),
      Linear("ffn2", rows, ffn_dim, embed_dim, with_bias, dtype),
      LayerNorm("ln2", embed_dim, kLayerNormEps, dtype),
  };
  const std::vector<Module> children_for_build = children;

  // inputs/params 相邻同型 span 形参是 BuildFn 契约固定形态(ARCH-071),同
  // Sequential 头注释先例。
  // NOLINTBEGIN(bugprone-easily-swappable-parameters)
  BuildFn build_fn = [children_for_build](
                         Graph& graph, std::span<Value* const> inputs,
                         std::span<Value* const> params) -> Result<std::vector<Value*>> {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    if (inputs.size() != 1) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::TransformerEncoderBlock build() expects 1 input, got " +
                              std::to_string(inputs.size()));
    }
    Value* x = inputs[0];

    const Result<std::vector<std::span<Value* const>>> child_params =
        SliceChildParams(children_for_build, params, "TransformerEncoderBlock");
    if (!child_params.is_ok()) return child_params.status();
    const std::span<Value* const> mha_params = child_params.value()[0];
    const std::span<Value* const> ln1_params = child_params.value()[1];
    const std::span<Value* const> ffn1_params = child_params.value()[2];
    const std::span<Value* const> ffn2_params = child_params.value()[3];
    const std::span<Value* const> ln2_params = child_params.value()[4];

    // 第一子层(自注意力 + 残差 + 归一化):y1 = ln1(x + mha(x))。
    const Result<std::vector<Value*>> mha_out =
        children_for_build[0].build(graph, std::vector<Value*>{x}, mha_params);
    if (!mha_out.is_ok()) return mha_out.status();
    const Result<Node*> residual1_node =
        create_node_with_inferred_types(graph, "add", {x, mha_out.value()[0]});
    if (!residual1_node.is_ok()) return residual1_node.status();
    const Result<std::vector<Value*>> y1_out = children_for_build[1].build(
        graph, std::vector<Value*>{residual1_node.value()->output(0)}, ln1_params);
    if (!y1_out.is_ok()) return y1_out.status();
    Value* y1 = y1_out.value()[0];

    // 第二子层(前馈网络 + 残差 + 归一化):y = ln2(y1 + ffn2(relu(ffn1(y1))))。
    const Result<std::vector<Value*>> ffn1_out =
        children_for_build[2].build(graph, std::vector<Value*>{y1}, ffn1_params);
    if (!ffn1_out.is_ok()) return ffn1_out.status();
    const Result<Node*> relu_node =
        create_node_with_inferred_types(graph, "relu", {ffn1_out.value()[0]});
    if (!relu_node.is_ok()) return relu_node.status();
    const Result<std::vector<Value*>> ffn2_out = children_for_build[3].build(
        graph, std::vector<Value*>{relu_node.value()->output(0)}, ffn2_params);
    if (!ffn2_out.is_ok()) return ffn2_out.status();

    const Result<Node*> residual2_node =
        create_node_with_inferred_types(graph, "add", {y1, ffn2_out.value()[0]});
    if (!residual2_node.is_ok()) return residual2_node.status();
    return children_for_build[4].build(
        graph, std::vector<Value*>{residual2_node.value()->output(0)}, ln2_params);
  };

  return Module{std::move(name), {}, std::move(children), std::move(build_fn)};
}

// ---------------------------------------------------------------------------
// M23 批5 T5:频域批次工厂(docs/plan/2026-07-21-batch5-m23-fft.md §1.5,
// design-reviewer APPROVE)。
// ---------------------------------------------------------------------------

// batch/in_channels/out_channels/n/modes 五个相邻 int64_t 形参语义上不可合并
// (§1.5 定稿签名,同 LSTM/Conv2d 头注释论证:build() 期 modes<=k 校验与
// slice/matmul/reshape 的 shape 推断会在误置换时立即报错)。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Module SpectralConv1d(std::string name, int64_t batch, int64_t in_channels, int64_t out_channels,
                      int64_t n, int64_t modes, DType dtype) {
  const int64_t k = n / 2 + 1;
  std::vector<ParamSpec> params;

  ParamSpec w_re;
  w_re.name = "W_re";
  w_re.type = MakeCpuTensorType(dtype, {in_channels, modes * out_channels});
  params.push_back(w_re);

  ParamSpec w_im;
  w_im.name = "W_im";
  w_im.type = MakeCpuTensorType(dtype, {in_channels, modes * out_channels});
  params.push_back(w_im);

  // inputs/params 相邻同型 span 形参是 BuildFn 契约固定形态(ARCH-071),同
  // Sequential 头注释先例。
  // NOLINTBEGIN(bugprone-easily-swappable-parameters)
  BuildFn build_fn = [batch, in_channels, out_channels, n, modes, k, dtype](
                         Graph& graph, std::span<Value* const> inputs,
                         std::span<Value* const> params) -> Result<std::vector<Value*>> {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    if (inputs.size() != 1) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::SpectralConv1d build() expects 1 input, got " + std::to_string(inputs.size()));
    }
    if (modes > k) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::SpectralConv1d build() requires modes <= n/2+1, got modes=" + std::to_string(modes) +
              " n=" + std::to_string(n) + " (k=" + std::to_string(k) + ")");
    }
    Value* x = inputs[0];
    Value* w_re_v = params[0];
    Value* w_im_v = params[1];
    const Device device = x->type().device;

    // rfft -> [B, Cin, K, 2](决议点A,§1.1)。
    const Result<Node*> rfft_node = create_node_with_inferred_types(graph, "rfft", {x});
    if (!rfft_node.is_ok()) return rfft_node.status();

    // slice 前 modes 模态:[B, Cin, modes, 2]。
    const AttrMap modes_slice_attrs{{"axis", int64_t{2}}, {"start", int64_t{0}}, {"stop", modes}};
    const Result<Node*> modes_sliced_node = create_node_with_inferred_types(
        graph, "slice", {rfft_node.value()->output(0)}, modes_slice_attrs);
    if (!modes_sliced_node.is_ok()) return modes_sliced_node.status();
    Value* z_modes = modes_sliced_node.value()->output(0);

    // 逐模态静态展开(§1.5,M22 MHA per-(b,h) 展开同款)。
    std::vector<Value*> mode_outputs;
    mode_outputs.reserve(static_cast<size_t>(modes));
    for (int64_t j = 0; j < modes; ++j) {
      // 该模态的 re/im:先按模态轴 slice 出 [B,Cin,1,2],再按末轴 slice 拆
      // re/im 各 [B,Cin,1,1],reshape 展平为 matmul 所需的 rank-2 [B,Cin]。
      const AttrMap mode_slice_attrs{{"axis", int64_t{2}}, {"start", j}, {"stop", j + 1}};
      const Result<Node*> mode_slice_node =
          create_node_with_inferred_types(graph, "slice", {z_modes}, mode_slice_attrs);
      if (!mode_slice_node.is_ok()) return mode_slice_node.status();
      Value* mode_slice = mode_slice_node.value()->output(0);

      const AttrMap re_slice_attrs{
          {"axis", int64_t{3}}, {"start", int64_t{0}}, {"stop", int64_t{1}}};
      const Result<Node*> re_4d_node =
          create_node_with_inferred_types(graph, "slice", {mode_slice}, re_slice_attrs);
      if (!re_4d_node.is_ok()) return re_4d_node.status();
      const AttrMap im_slice_attrs{
          {"axis", int64_t{3}}, {"start", int64_t{1}}, {"stop", int64_t{2}}};
      const Result<Node*> im_4d_node =
          create_node_with_inferred_types(graph, "slice", {mode_slice}, im_slice_attrs);
      if (!im_4d_node.is_ok()) return im_4d_node.status();

      const AttrMap in_2d_reshape_attrs{{"target_shape", Shape({batch, in_channels})}};
      const Result<Node*> re_2d_node = create_node_with_inferred_types(
          graph, "reshape", {re_4d_node.value()->output(0)}, in_2d_reshape_attrs);
      if (!re_2d_node.is_ok()) return re_2d_node.status();
      const Result<Node*> im_2d_node = create_node_with_inferred_types(
          graph, "reshape", {im_4d_node.value()->output(0)}, in_2d_reshape_attrs);
      if (!im_2d_node.is_ok()) return im_2d_node.status();
      Value* x_re = re_2d_node.value()->output(0);  // [B, Cin]
      Value* x_im = im_2d_node.value()->output(0);  // [B, Cin]

      // 该模态的权重列切片:[Cin, Cout](第 j 模态占列
      // [j*Cout, (j+1)*Cout))。
      const AttrMap w_col_slice_attrs{
          {"axis", int64_t{1}}, {"start", j * out_channels}, {"stop", (j + 1) * out_channels}};
      const Result<Node*> w_re_j_node =
          create_node_with_inferred_types(graph, "slice", {w_re_v}, w_col_slice_attrs);
      if (!w_re_j_node.is_ok()) return w_re_j_node.status();
      const Result<Node*> w_im_j_node =
          create_node_with_inferred_types(graph, "slice", {w_im_v}, w_col_slice_attrs);
      if (!w_im_j_node.is_ok()) return w_im_j_node.status();
      Value* w_re_j = w_re_j_node.value()->output(0);
      Value* w_im_j = w_im_j_node.value()->output(0);

      // 4 次 matmul + 复乘组合(无 sub:constant(-1)+mul+add,§1.5):
      // re_out = re@W_re - im@W_im;im_out = re@W_im + im@W_re。
      const Result<Node*> re_wre_node =
          create_node_with_inferred_types(graph, "matmul", {x_re, w_re_j});
      if (!re_wre_node.is_ok()) return re_wre_node.status();
      const Result<Node*> im_wim_node =
          create_node_with_inferred_types(graph, "matmul", {x_im, w_im_j});
      if (!im_wim_node.is_ok()) return im_wim_node.status();
      const Result<Node*> re_wim_node =
          create_node_with_inferred_types(graph, "matmul", {x_re, w_im_j});
      if (!re_wim_node.is_ok()) return re_wim_node.status();
      const Result<Node*> im_wre_node =
          create_node_with_inferred_types(graph, "matmul", {x_im, w_re_j});
      if (!im_wre_node.is_ok()) return im_wre_node.status();

      const Shape out_2d_shape = re_wre_node.value()->output(0)->type().shape;  // [B, Cout]
      const Result<Node*> neg_one_node =
          make_constant_splat(graph, out_2d_shape, dtype, device, -1.0);
      if (!neg_one_node.is_ok()) return neg_one_node.status();
      const Result<Node*> neg_im_wim_node = create_node_with_inferred_types(
          graph, "mul", {neg_one_node.value()->output(0), im_wim_node.value()->output(0)});
      if (!neg_im_wim_node.is_ok()) return neg_im_wim_node.status();
      const Result<Node*> re_out_node = create_node_with_inferred_types(
          graph, "add", {re_wre_node.value()->output(0), neg_im_wim_node.value()->output(0)});
      if (!re_out_node.is_ok()) return re_out_node.status();
      const Result<Node*> im_out_node = create_node_with_inferred_types(
          graph, "add", {re_wim_node.value()->output(0), im_wre_node.value()->output(0)});
      if (!im_out_node.is_ok()) return im_out_node.status();

      // reshape/concat 回拼 [B, Cout, 1, 2]。
      const AttrMap out_4d_reshape_attrs{{"target_shape", Shape({batch, out_channels, 1, 1})}};
      const Result<Node*> re_out_4d_node = create_node_with_inferred_types(
          graph, "reshape", {re_out_node.value()->output(0)}, out_4d_reshape_attrs);
      if (!re_out_4d_node.is_ok()) return re_out_4d_node.status();
      const Result<Node*> im_out_4d_node = create_node_with_inferred_types(
          graph, "reshape", {im_out_node.value()->output(0)}, out_4d_reshape_attrs);
      if (!im_out_4d_node.is_ok()) return im_out_4d_node.status();

      const AttrMap concat_ri_attrs{{"axis", int64_t{3}}};
      const Result<Node*> mode_out_node = create_node_with_inferred_types(
          graph, "concat", {re_out_4d_node.value()->output(0), im_out_4d_node.value()->output(0)},
          concat_ri_attrs);
      if (!mode_out_node.is_ok()) return mode_out_node.status();

      mode_outputs.push_back(mode_out_node.value()->output(0));
    }

    // 各模态沿模态轴(axis=2)拼回 [B, Cout, modes, 2]。
    const AttrMap concat_modes_attrs{{"axis", int64_t{2}}};
    const Result<Node*> modes_out_node =
        create_node_with_inferred_types(graph, "concat", mode_outputs, concat_modes_attrs);
    if (!modes_out_node.is_ok()) return modes_out_node.status();
    Value* spectrum_truncated = modes_out_node.value()->output(0);

    // 零补到 K(constant(0) splat + concat,§1.5):modes==k 时无需补零
    // (slice_gradient 零宽度跳过同款先例,concat 单输入=恒等拷贝亦无需触发)。
    Value* spectrum_full = spectrum_truncated;
    if (modes < k) {
      const Shape zero_pad_shape({batch, out_channels, k - modes, 2});
      const Result<Node*> zero_pad_node =
          make_constant_splat(graph, zero_pad_shape, dtype, device, 0.0);
      if (!zero_pad_node.is_ok()) return zero_pad_node.status();
      const AttrMap concat_pad_attrs{{"axis", int64_t{2}}};
      const Result<Node*> padded_node = create_node_with_inferred_types(
          graph, "concat", {spectrum_truncated, zero_pad_node.value()->output(0)},
          concat_pad_attrs);
      if (!padded_node.is_ok()) return padded_node.status();
      spectrum_full = padded_node.value()->output(0);
    }

    // irfft(n) 还原回时域,输出 [B, Cout, N]。
    const AttrMap irfft_attrs{{"n", n}};
    const Result<Node*> irfft_node =
        create_node_with_inferred_types(graph, "irfft", {spectrum_full}, irfft_attrs);
    if (!irfft_node.is_ok()) return irfft_node.status();

    return std::vector<Value*>{irfft_node.value()->output(0)};
  };

  return Module{std::move(name), std::move(params), {}, std::move(build_fn)};
}

// batch/channels/n 三个相邻 int64_t 形参语义上不可合并(§1.5 定稿签名,同
// SpectralConv1d/Conv2d 头注释论证)。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Module FourierFilter1d(std::string name, int64_t batch, int64_t channels, int64_t n, DType dtype) {
  const int64_t k = n / 2 + 1;
  std::vector<ParamSpec> params;

  ParamSpec w_re;
  w_re.name = "w_re";
  // 注意逐样本参数语义(设计门建议 3,§1.5):w_re/w_im 形状含 batch 维,是
  // "每样本独立滤波器"而非共享滤波器——调用方组装 batch 维输入/参数时须与此
  // 语义自洽(勿当作跨样本共享的滤波器使用,layers.h 头注释)。
  w_re.type = MakeCpuTensorType(dtype, {batch, channels, k, 1});
  params.push_back(w_re);

  ParamSpec w_im;
  w_im.name = "w_im";
  w_im.type = MakeCpuTensorType(dtype, {batch, channels, k, 1});
  params.push_back(w_im);

  // NOLINTBEGIN(bugprone-easily-swappable-parameters)
  BuildFn build_fn = [dtype, n](Graph& graph, std::span<Value* const> inputs,
                                std::span<Value* const> params) -> Result<std::vector<Value*>> {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    if (inputs.size() != 1) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::FourierFilter1d build() expects 1 input, got " + std::to_string(inputs.size()));
    }
    Value* x = inputs[0];
    Value* w_re_v = params[0];
    Value* w_im_v = params[1];
    const Device device = x->type().device;

    // rfft 变换到频域,输出 [B, C, K, 2]。
    const Result<Node*> rfft_node = create_node_with_inferred_types(graph, "rfft", {x});
    if (!rfft_node.is_ok()) return rfft_node.status();
    Value* z = rfft_node.value()->output(0);

    // x_re/x_im 经末轴(axis=3)slice 取得,shape 均 [B, C, K, 1](与
    // w_re/w_im 同形)。
    const AttrMap re_slice_attrs{{"axis", int64_t{3}}, {"start", int64_t{0}}, {"stop", int64_t{1}}};
    const Result<Node*> x_re_node =
        create_node_with_inferred_types(graph, "slice", {z}, re_slice_attrs);
    if (!x_re_node.is_ok()) return x_re_node.status();
    const AttrMap im_slice_attrs{{"axis", int64_t{3}}, {"start", int64_t{1}}, {"stop", int64_t{2}}};
    const Result<Node*> x_im_node =
        create_node_with_inferred_types(graph, "slice", {z}, im_slice_attrs);
    if (!x_im_node.is_ok()) return x_im_node.status();
    Value* x_re = x_re_node.value()->output(0);
    Value* x_im = x_im_node.value()->output(0);

    // y_re = w_re⊙x_re - w_im⊙x_im(无 sub:constant(-1)+mul+add,同
    // SpectralConv1d 先例)。
    const Result<Node*> wre_xre_node =
        create_node_with_inferred_types(graph, "mul", {w_re_v, x_re});
    if (!wre_xre_node.is_ok()) return wre_xre_node.status();
    const Result<Node*> wim_xim_node =
        create_node_with_inferred_types(graph, "mul", {w_im_v, x_im});
    if (!wim_xim_node.is_ok()) return wim_xim_node.status();
    const Shape re_shape = x_re->type().shape;
    const Result<Node*> neg_one_node = make_constant_splat(graph, re_shape, dtype, device, -1.0);
    if (!neg_one_node.is_ok()) return neg_one_node.status();
    const Result<Node*> neg_wim_xim_node = create_node_with_inferred_types(
        graph, "mul", {neg_one_node.value()->output(0), wim_xim_node.value()->output(0)});
    if (!neg_wim_xim_node.is_ok()) return neg_wim_xim_node.status();
    const Result<Node*> y_re_node = create_node_with_inferred_types(
        graph, "add", {wre_xre_node.value()->output(0), neg_wim_xim_node.value()->output(0)});
    if (!y_re_node.is_ok()) return y_re_node.status();

    // y_im = w_re⊙x_im + w_im⊙x_re(对偶,直接 add,无需 constant(-1))。
    const Result<Node*> wre_xim_node =
        create_node_with_inferred_types(graph, "mul", {w_re_v, x_im});
    if (!wre_xim_node.is_ok()) return wre_xim_node.status();
    const Result<Node*> wim_xre_node =
        create_node_with_inferred_types(graph, "mul", {w_im_v, x_re});
    if (!wim_xre_node.is_ok()) return wim_xre_node.status();
    const Result<Node*> y_im_node = create_node_with_inferred_types(
        graph, "add", {wre_xim_node.value()->output(0), wim_xre_node.value()->output(0)});
    if (!y_im_node.is_ok()) return y_im_node.status();

    // concat 回 [B, C, K, 2] 后 irfft(n) -> [B, C, N]。
    const AttrMap concat_ri_attrs{{"axis", int64_t{3}}};
    const Result<Node*> z_out_node = create_node_with_inferred_types(
        graph, "concat", {y_re_node.value()->output(0), y_im_node.value()->output(0)},
        concat_ri_attrs);
    if (!z_out_node.is_ok()) return z_out_node.status();

    const AttrMap irfft_attrs{{"n", n}};
    const Result<Node*> irfft_node = create_node_with_inferred_types(
        graph, "irfft", {z_out_node.value()->output(0)}, irfft_attrs);
    if (!irfft_node.is_ok()) return irfft_node.status();

    return std::vector<Value*>{irfft_node.value()->output(0)};
  };

  return Module{std::move(name), std::move(params), {}, std::move(build_fn)};
}

// batch/in_channels/out_channels/n/modes 五个相邻 int64_t 形参同 SpectralConv1d
// 头注释论证,语义上不可合并。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Module Fno1dBlock(std::string name, int64_t batch, int64_t in_channels, int64_t out_channels,
                  int64_t n, int64_t modes, DType dtype) {
  // children=[SpectralConv1d, Conv1d(逐点旁路)](§1.5)。Conv1d 的
  // with_bias=true 是计划未明确处的工程取舍(同 AFF 头注释先例:常规卷积基线
  // 配置,无 BN 层可吸收 bias)。
  std::vector<Module> children{
      SpectralConv1d("spectral", batch, in_channels, out_channels, n, modes, dtype),
      Conv1d("bypass", in_channels, out_channels, /*kernel=*/1, /*stride=*/1, /*padding=*/0,
             /*groups=*/1, /*with_bias=*/true, dtype),
  };
  const std::vector<Module> children_for_build = children;

  // NOLINTBEGIN(bugprone-easily-swappable-parameters)
  BuildFn build_fn = [children_for_build](
                         Graph& graph, std::span<Value* const> inputs,
                         std::span<Value* const> params) -> Result<std::vector<Value*>> {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    if (inputs.size() != 1) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::Fno1dBlock build() expects 1 input, got " + std::to_string(inputs.size()));
    }
    Value* x = inputs[0];

    const Result<std::vector<std::span<Value* const>>> child_params =
        SliceChildParams(children_for_build, params, "Fno1dBlock");
    if (!child_params.is_ok()) return child_params.status();
    const std::span<Value* const> spectral_params = child_params.value()[0];
    const std::span<Value* const> bypass_params = child_params.value()[1];

    const Result<std::vector<Value*>> spectral_out =
        children_for_build[0].build(graph, std::vector<Value*>{x}, spectral_params);
    if (!spectral_out.is_ok()) return spectral_out.status();
    const Result<std::vector<Value*>> bypass_out =
        children_for_build[1].build(graph, std::vector<Value*>{x}, bypass_params);
    if (!bypass_out.is_ok()) return bypass_out.status();

    // y = tanh(add(spectral(x), conv1x1(x)))(§1.5,FNO 惯例块)。
    const Result<Node*> sum_node = create_node_with_inferred_types(
        graph, "add", {spectral_out.value()[0], bypass_out.value()[0]});
    if (!sum_node.is_ok()) return sum_node.status();
    const Result<Node*> y_node =
        create_node_with_inferred_types(graph, "tanh", {sum_node.value()->output(0)});
    if (!y_node.is_ok()) return y_node.status();

    return std::vector<Value*>{y_node.value()->output(0)};
  };

  return Module{std::move(name), {}, std::move(children), std::move(build_fn)};
}

// ---------------------------------------------------------------------------
// M25 批6 T4:状态空间模型工厂(docs/plan/
// 2026-07-23-batch6-m25-ssm.md §1.4,design-reviewer T3 补审 APPROVE)。
// ---------------------------------------------------------------------------

// batch/channels/steps/kernel_size 四个相邻 int64_t 形参是已批准签名,
// 语义上不可合并;误置换会在 conv/slice/reshape/Linear 的 shape 推断中
// 立即暴露,同 Conv1d/SpectralConv1d 先例。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Module Mamba(std::string name, int64_t batch, int64_t channels, int64_t steps, int64_t kernel_size,
             DType dtype) {
  const std::optional<int64_t> rows_result = checked_nonnegative_product({batch, steps});
  const bool arithmetic_valid = mamba_shape_arithmetic_valid({batch, channels, steps, kernel_size});

  // 非法配置仍须返回可调用的 Module，由 build() 原子地返回 InvalidArgument；
  // 因此 children 用最小合法占位形状，避免工厂阶段先产生溢出或非法 Shape。
  const int64_t child_rows = arithmetic_valid ? rows_result.value_or(1) : 1;
  const int64_t child_channels = arithmetic_valid ? channels : 1;
  const int64_t child_kernel = arithmetic_valid ? kernel_size : 1;
  const int64_t child_padding = child_kernel - 1;

  // children 名称、顺序与配置均是 T3 补审的精确合同。六路中间
  // Linear 与 out Linear 均为 C->C 且带 bias;conv 为带 bias 的
  // depthwise causal Conv1d。
  std::vector<Module> children{
      Conv1d("conv", child_channels, child_channels, child_kernel, /*stride=*/1,
             /*padding=*/child_padding, /*groups=*/child_channels, /*with_bias=*/true, dtype),
      Linear("input", child_rows, child_channels, child_channels, /*with_bias=*/true, dtype),
      Linear("a", child_rows, child_channels, child_channels, /*with_bias=*/true, dtype),
      Linear("b", child_rows, child_channels, child_channels, /*with_bias=*/true, dtype),
      Linear("c", child_rows, child_channels, child_channels, /*with_bias=*/true, dtype),
      Linear("d", child_rows, child_channels, child_channels, /*with_bias=*/true, dtype),
      Linear("gate", child_rows, child_channels, child_channels, /*with_bias=*/true, dtype),
      Linear("out", child_rows, child_channels, child_channels, /*with_bias=*/true, dtype),
  };
  const std::vector<Module> children_for_build = children;

  // NOLINTBEGIN(bugprone-easily-swappable-parameters)
  BuildFn build_fn = [batch, channels, steps, kernel_size, dtype, arithmetic_valid, child_rows,
                      children_for_build](
                         Graph& graph, std::span<Value* const> inputs,
                         std::span<Value* const> params) -> Result<std::vector<Value*>> {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    if (inputs.size() != 1) {
      return Status::make(ErrorCode::kInvalidArgument, "nn::Mamba build() expects 1 input, got " +
                                                           std::to_string(inputs.size()));
    }
    if (batch <= 0 || channels <= 0 || steps <= 0 || kernel_size <= 0) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::Mamba build() requires batch, channels, steps, and kernel_size to be positive, "
          "got batch=" +
              std::to_string(batch) + " channels=" + std::to_string(channels) +
              " steps=" + std::to_string(steps) + " kernel_size=" + std::to_string(kernel_size));
    }
    if (!arithmetic_valid) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::Mamba build() configuration exceeds the supported int64 shape range");
    }
    if (!is_supported_float_dtype(dtype)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::Mamba build() does not support dtype '" + std::string(dtype.name()) +
                              "' (supports float32/float16/bfloat16 only)");
    }
    Value* x = inputs[0];
    const Shape expected_shape({batch, channels, steps});
    if (!(x->type().shape == expected_shape)) {
      return Status::make(ErrorCode::kInvalidArgument, "nn::Mamba build() requires input shape " +
                                                           expected_shape.to_string() + ", got " +
                                                           x->type().shape.to_string());
    }
    if (!(x->type().dtype == dtype)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::Mamba build() requires input dtype '" + std::string(dtype.name()) +
                              "', got '" + std::string(x->type().dtype.name()) + "'");
    }

    const Result<std::vector<std::span<Value* const>>> child_params =
        SliceChildParams(children_for_build, params, "Mamba");
    if (!child_params.is_ok()) return child_params.status();

    // depthwise causal conv 先得 [B,C,L+K-1],再沿时间轴裁前 L 步。
    const Result<std::vector<Value*>> conv_out =
        children_for_build[0].build(graph, std::vector<Value*>{x}, child_params.value()[0]);
    if (!conv_out.is_ok()) return conv_out.status();
    const AttrMap causal_slice_attrs{{"axis", int64_t{2}}, {"start", int64_t{0}}, {"stop", steps}};
    const Result<Node*> causal_node =
        create_node_with_inferred_types(graph, "slice", {conv_out.value()[0]}, causal_slice_attrs);
    if (!causal_node.is_ok()) return causal_node.status();

    // [B,C,L] -> [B,L,C] -> [B*L,C],作为六路 Linear 的共享输入。
    const AttrMap to_blc_attrs{{"perm", std::vector<int64_t>{0, 2, 1}}};
    const Result<Node*> conv_blc_node = create_node_with_inferred_types(
        graph, "transpose", {causal_node.value()->output(0)}, to_blc_attrs);
    if (!conv_blc_node.is_ok()) return conv_blc_node.status();
    const AttrMap flatten_attrs{{"target_shape", Shape({child_rows, channels})}};
    const Result<Node*> flat_node = create_node_with_inferred_types(
        graph, "reshape", {conv_blc_node.value()->output(0)}, flatten_attrs);
    if (!flat_node.is_ok()) return flat_node.status();
    Value* flat = flat_node.value()->output(0);

    // 六路投影共用「Linear -> 激活 -> [B,L,C] -> [B,C,L]」图构建步骤;
    // child_index 与 activation_op 由下方获批映射的六个具名调用点固定。
    const auto build_projection = [&](size_t child_index,
                                      const std::string& activation_op) -> Result<Value*> {
      const Result<std::vector<Value*>> projected = children_for_build[child_index].build(
          graph, std::vector<Value*>{flat}, child_params.value()[child_index]);
      if (!projected.is_ok()) return projected.status();
      const Result<Node*> activated_node =
          create_node_with_inferred_types(graph, activation_op, {projected.value()[0]});
      if (!activated_node.is_ok()) return activated_node.status();

      const AttrMap restore_blc_attrs{{"target_shape", Shape({batch, steps, channels})}};
      const Result<Node*> restored_blc_node = create_node_with_inferred_types(
          graph, "reshape", {activated_node.value()->output(0)}, restore_blc_attrs);
      if (!restored_blc_node.is_ok()) return restored_blc_node.status();
      const AttrMap restore_bcl_attrs{{"perm", std::vector<int64_t>{0, 2, 1}}};
      const Result<Node*> restored_bcl_node = create_node_with_inferred_types(
          graph, "transpose", {restored_blc_node.value()->output(0)}, restore_bcl_attrs);
      if (!restored_bcl_node.is_ok()) return restored_bcl_node.status();
      return restored_bcl_node.value()->output(0);
    };

    const Result<Value*> input_projection = build_projection(1, "tanh");
    if (!input_projection.is_ok()) return input_projection.status();
    const Result<Value*> a_projection = build_projection(2, "sigmoid");
    if (!a_projection.is_ok()) return a_projection.status();
    const Result<Value*> b_projection = build_projection(3, "tanh");
    if (!b_projection.is_ok()) return b_projection.status();
    const Result<Value*> c_projection = build_projection(4, "tanh");
    if (!c_projection.is_ok()) return c_projection.status();
    const Result<Value*> d_projection = build_projection(5, "tanh");
    if (!d_projection.is_ok()) return d_projection.status();
    const Result<Value*> gate_projection = build_projection(6, "sigmoid");
    if (!gate_projection.is_ok()) return gate_projection.status();

    const Result<Node*> scan_node = create_node_with_inferred_types(
        graph, "selective_scan",
        {input_projection.value(), a_projection.value(), b_projection.value(), c_projection.value(),
         d_projection.value()});
    if (!scan_node.is_ok()) return scan_node.status();
    const Result<Node*> merged_node = create_node_with_inferred_types(
        graph, "mul", {scan_node.value()->output(0), gate_projection.value()});
    if (!merged_node.is_ok()) return merged_node.status();

    // [B,C,L] -> [B,L,C] -> [B*L,C],转发给 out Linear 后恢复原形。
    const Result<Node*> merged_blc_node = create_node_with_inferred_types(
        graph, "transpose", {merged_node.value()->output(0)}, to_blc_attrs);
    if (!merged_blc_node.is_ok()) return merged_blc_node.status();
    const Result<Node*> merged_flat_node = create_node_with_inferred_types(
        graph, "reshape", {merged_blc_node.value()->output(0)}, flatten_attrs);
    if (!merged_flat_node.is_ok()) return merged_flat_node.status();
    const Result<std::vector<Value*>> out_projection = children_for_build[7].build(
        graph, std::vector<Value*>{merged_flat_node.value()->output(0)}, child_params.value()[7]);
    if (!out_projection.is_ok()) return out_projection.status();

    const AttrMap out_blc_attrs{{"target_shape", Shape({batch, steps, channels})}};
    const Result<Node*> out_blc_node = create_node_with_inferred_types(
        graph, "reshape", {out_projection.value()[0]}, out_blc_attrs);
    if (!out_blc_node.is_ok()) return out_blc_node.status();
    const AttrMap out_bcl_attrs{{"perm", std::vector<int64_t>{0, 2, 1}}};
    const Result<Node*> out_bcl_node = create_node_with_inferred_types(
        graph, "transpose", {out_blc_node.value()->output(0)}, out_bcl_attrs);
    if (!out_bcl_node.is_ok()) return out_bcl_node.status();

    return std::vector<Value*>{out_bcl_node.value()->output(0)};
  };

  return Module{std::move(name), {}, std::move(children), std::move(build_fn)};
}

// batch/channels/steps/kernel_size 四个相邻 int64_t 形参同 Mamba 签名论证。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Module FourierMamba(std::string name, int64_t batch, int64_t channels, int64_t steps,
                    int64_t kernel_size, DType dtype) {
  const int64_t frequency_bins = steps > 0 ? steps / 2 + 1 : 1;
  const bool configuration_valid =
      mamba_shape_arithmetic_valid({batch, channels, steps, kernel_size}) &&
      checked_nonnegative_product({batch, channels, frequency_bins, int64_t{2}}).has_value();
  const int64_t fourier_batch = configuration_valid ? batch : 1;
  const int64_t fourier_channels = configuration_valid ? channels : 1;
  const int64_t fourier_steps = configuration_valid ? steps : 1;
  // children=[mamba,fourier] 是 T3 补审的精确合同;频域分支直接复用
  // M23 FourierFilter1d,不复制 FFT 或滤波逻辑。
  std::vector<Module> children{
      Mamba("mamba", batch, channels, steps, kernel_size, dtype),
      FourierFilter1d("fourier", fourier_batch, fourier_channels, fourier_steps, dtype),
  };
  const std::vector<Module> children_for_build = children;

  // NOLINTBEGIN(bugprone-easily-swappable-parameters)
  BuildFn build_fn = [batch, channels, steps, kernel_size, dtype, configuration_valid,
                      children_for_build](
                         Graph& graph, std::span<Value* const> inputs,
                         std::span<Value* const> params) -> Result<std::vector<Value*>> {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    if (inputs.size() != 1) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::FourierMamba build() expects 1 input, got " + std::to_string(inputs.size()));
    }
    if (batch <= 0 || channels <= 0 || steps <= 0 || kernel_size <= 0) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::FourierMamba build() requires batch, channels, steps, and kernel_size to be "
          "positive");
    }
    if (!configuration_valid) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::FourierMamba build() configuration exceeds the supported int64 shape range");
    }
    if (!is_supported_float_dtype(dtype)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::FourierMamba build() does not support dtype '" +
                              std::string(dtype.name()) +
                              "' (supports float32/float16/bfloat16 only)");
    }
    Value* x = inputs[0];
    const Shape expected_shape({batch, channels, steps});
    if (!(x->type().shape == expected_shape)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::FourierMamba build() requires input shape " +
                              expected_shape.to_string() + ", got " + x->type().shape.to_string());
    }
    if (!(x->type().dtype == dtype)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::FourierMamba build() requires input dtype '" +
                              std::string(dtype.name()) + "', got '" +
                              std::string(x->type().dtype.name()) + "'");
    }

    const Result<std::vector<std::span<Value* const>>> child_params =
        SliceChildParams(children_for_build, params, "FourierMamba");
    if (!child_params.is_ok()) return child_params.status();

    const Result<std::vector<Value*>> mamba_out =
        children_for_build[0].build(graph, std::vector<Value*>{x}, child_params.value()[0]);
    if (!mamba_out.is_ok()) return mamba_out.status();
    const Result<std::vector<Value*>> fourier_out =
        children_for_build[1].build(graph, std::vector<Value*>{x}, child_params.value()[1]);
    if (!fourier_out.is_ok()) return fourier_out.status();

    const Result<Node*> sum_node = create_node_with_inferred_types(
        graph, "add", {mamba_out.value()[0], fourier_out.value()[0]});
    if (!sum_node.is_ok()) return sum_node.status();
    const Result<Node*> y_node =
        create_node_with_inferred_types(graph, "tanh", {sum_node.value()->output(0)});
    if (!y_node.is_ok()) return y_node.status();

    return std::vector<Value*>{y_node.value()->output(0)};
  };

  return Module{std::move(name), {}, std::move(children), std::move(build_fn)};
}

// batch/num_steps/features 与三个浮点超参数均为已批准的 LIFCell 固定签名。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Module LIFCell(std::string name, int64_t batch, int64_t num_steps, int64_t features, double decay,
               double threshold, double alpha, DType dtype) {
  BuildFn build_fn = [batch, num_steps, features, decay, threshold, alpha, dtype](
                         Graph& graph, std::span<Value* const> inputs,
                         std::span<Value* const> /*params*/) -> Result<std::vector<Value*>> {
    if (inputs.size() != 1) {
      return Status::make(ErrorCode::kInvalidArgument, "nn::LIFCell build() expects 1 input, got " +
                                                           std::to_string(inputs.size()));
    }
    if (batch <= 0 || num_steps <= 0 || features <= 0) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::LIFCell build() requires batch, num_steps, and features to be positive, got "
          "batch=" +
              std::to_string(batch) + " num_steps=" + std::to_string(num_steps) +
              " features=" + std::to_string(features));
    }
    if (!checked_nonnegative_product({batch, num_steps, features}).has_value() ||
        !checked_nonnegative_product({batch, features}).has_value()) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::LIFCell build() configuration exceeds the supported int64 shape range");
    }
    if (!std::isfinite(decay) || decay < 0.0 || !(decay < 1.0)) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::LIFCell build() requires finite decay in [0, 1), got " + std::to_string(decay));
    }
    if (!std::isfinite(threshold) || !(threshold > 0.0)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::LIFCell build() requires finite positive threshold, got " +
                              std::to_string(threshold));
    }
    if (!std::isfinite(alpha) || !(alpha > 0.0)) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::LIFCell build() requires finite positive alpha, got " + std::to_string(alpha));
    }
    const DTypeCode code = dtype.code();
    const bool supported =
        code == DTypeCode::kFloat32 || code == DTypeCode::kFloat16 || code == DTypeCode::kBFloat16;
    if (!supported) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::LIFCell build() does not support dtype '" +
                              std::string(dtype.name()) +
                              "' (supports float32/float16/bfloat16 only)");
    }

    Value* x = inputs[0];
    const Shape expected_shape({batch, num_steps, features});
    if (!(x->type().shape == expected_shape)) {
      return Status::make(ErrorCode::kInvalidArgument, "nn::LIFCell build() requires input shape " +
                                                           expected_shape.to_string() + ", got " +
                                                           x->type().shape.to_string());
    }
    if (!(x->type().dtype == dtype)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::LIFCell build() requires input dtype '" + std::string(dtype.name()) +
                              "', got '" + std::string(x->type().dtype.name()) + "'");
    }

    const Device device = x->type().device;
    const Shape state_shape({batch, features});
    const Result<Node*> v0_node = make_constant_splat(graph, state_shape, dtype, device, 0.0);
    if (!v0_node.is_ok()) return v0_node.status();
    const Result<Node*> decay_node = make_constant_splat(graph, state_shape, dtype, device, decay);
    if (!decay_node.is_ok()) return decay_node.status();
    const Result<Node*> neg_threshold_node =
        make_constant_splat(graph, state_shape, dtype, device, -threshold);
    if (!neg_threshold_node.is_ok()) return neg_threshold_node.status();
    const Result<Node*> one_node = make_constant_splat(graph, state_shape, dtype, device, 1.0);
    if (!one_node.is_ok()) return one_node.status();
    const Result<Node*> neg_one_node = make_constant_splat(graph, state_shape, dtype, device, -1.0);
    if (!neg_one_node.is_ok()) return neg_one_node.status();

    Value* v_prev = v0_node.value()->output(0);
    std::vector<Value*> spikes;
    spikes.reserve(static_cast<size_t>(num_steps));
    for (int64_t t = 0; t < num_steps; ++t) {
      const AttrMap slice_attrs{{"axis", int64_t{1}}, {"start", t}, {"stop", t + 1}};
      const Result<Node*> x_slice_node =
          create_node_with_inferred_types(graph, "slice", {x}, slice_attrs);
      if (!x_slice_node.is_ok()) return x_slice_node.status();
      const AttrMap x_t_attrs{{"target_shape", state_shape}};
      const Result<Node*> x_t_node = create_node_with_inferred_types(
          graph, "reshape", {x_slice_node.value()->output(0)}, x_t_attrs);
      if (!x_t_node.is_ok()) return x_t_node.status();

      const Result<Node*> decayed_node =
          create_node_with_inferred_types(graph, "mul", {decay_node.value()->output(0), v_prev});
      if (!decayed_node.is_ok()) return decayed_node.status();
      const Result<Node*> v_pre_node = create_node_with_inferred_types(
          graph, "add", {decayed_node.value()->output(0), x_t_node.value()->output(0)});
      if (!v_pre_node.is_ok()) return v_pre_node.status();
      Value* v_pre = v_pre_node.value()->output(0);

      const Result<Node*> shifted_node = create_node_with_inferred_types(
          graph, "add", {v_pre, neg_threshold_node.value()->output(0)});
      if (!shifted_node.is_ok()) return shifted_node.status();
      const AttrMap heaviside_attrs{{"alpha", alpha}};
      const Result<Node*> spike_node = create_node_with_inferred_types(
          graph, "heaviside_surrogate", {shifted_node.value()->output(0)}, heaviside_attrs);
      if (!spike_node.is_ok()) return spike_node.status();
      Value* spike = spike_node.value()->output(0);

      const Result<Node*> neg_spike_node =
          create_node_with_inferred_types(graph, "mul", {neg_one_node.value()->output(0), spike});
      if (!neg_spike_node.is_ok()) return neg_spike_node.status();
      const Result<Node*> one_minus_spike_node = create_node_with_inferred_types(
          graph, "add", {one_node.value()->output(0), neg_spike_node.value()->output(0)});
      if (!one_minus_spike_node.is_ok()) return one_minus_spike_node.status();
      const Result<Node*> v_next_node = create_node_with_inferred_types(
          graph, "mul", {v_pre, one_minus_spike_node.value()->output(0)});
      if (!v_next_node.is_ok()) return v_next_node.status();
      v_prev = v_next_node.value()->output(0);

      const AttrMap spike_step_attrs{{"target_shape", Shape({batch, 1, features})}};
      const Result<Node*> spike_step_node =
          create_node_with_inferred_types(graph, "reshape", {spike}, spike_step_attrs);
      if (!spike_step_node.is_ok()) return spike_step_node.status();
      spikes.push_back(spike_step_node.value()->output(0));
    }

    const AttrMap concat_attrs{{"axis", int64_t{1}}};
    const Result<Node*> output_node =
        create_node_with_inferred_types(graph, "concat", spikes, concat_attrs);
    if (!output_node.is_ok()) return output_node.status();
    return std::vector<Value*>{output_node.value()->output(0)};
  };

  return Module{std::move(name), {}, {}, std::move(build_fn)};
}

// SnnClassifier 的相邻整数与浮点形参均为已批准固定接口。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Module SnnClassifier(std::string name, int64_t batch, int64_t num_steps, int64_t input_dim,
                     int64_t hidden_dim, int64_t num_classes, double decay, double threshold,
                     double alpha, bool with_bias, DType dtype) {
  const std::optional<int64_t> rows_result = checked_nonnegative_product({batch, num_steps});
  const bool arithmetic_valid =
      rows_result.has_value() &&
      checked_nonnegative_product({batch, num_steps, input_dim}).has_value() &&
      checked_nonnegative_product({batch, num_steps, hidden_dim}).has_value() &&
      checked_nonnegative_product({batch, num_steps, num_classes}).has_value() &&
      checked_nonnegative_product({input_dim, hidden_dim}).has_value() &&
      checked_nonnegative_product({hidden_dim, num_classes}).has_value();
  const int64_t child_rows = arithmetic_valid ? rows_result.value() : 1;
  const int64_t child_input_dim = arithmetic_valid ? input_dim : 1;
  const int64_t child_hidden_dim = arithmetic_valid ? hidden_dim : 1;
  const int64_t child_num_classes = arithmetic_valid ? num_classes : 1;
  std::vector<Module> children{
      Linear("input", child_rows, child_input_dim, child_hidden_dim, with_bias, dtype),
      LIFCell("lif", arithmetic_valid ? batch : 1, arithmetic_valid ? num_steps : 1,
              child_hidden_dim, decay, threshold, alpha, dtype),
      Linear("output", child_rows, child_hidden_dim, child_num_classes, with_bias, dtype),
  };
  const std::vector<Module> children_for_build = children;

  // inputs/params 相邻同型 span 形参是 BuildFn 契约固定形态(ARCH-071)。
  // NOLINTBEGIN(bugprone-easily-swappable-parameters)
  BuildFn build_fn = [batch, num_steps, input_dim, hidden_dim, num_classes, decay, threshold, alpha,
                      dtype, arithmetic_valid, child_rows, children_for_build](
                         Graph& graph, std::span<Value* const> inputs,
                         std::span<Value* const> params) -> Result<std::vector<Value*>> {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    if (inputs.size() != 1) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::SnnClassifier build() expects 1 input, got " + std::to_string(inputs.size()));
    }
    if (batch <= 0 || num_steps <= 0 || input_dim <= 0 || hidden_dim <= 0 || num_classes <= 0) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::SnnClassifier build() requires all dimensions to be positive, got batch=" +
              std::to_string(batch) + " num_steps=" + std::to_string(num_steps) + " input_dim=" +
              std::to_string(input_dim) + " hidden_dim=" + std::to_string(hidden_dim) +
              " num_classes=" + std::to_string(num_classes));
    }
    if (!arithmetic_valid) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::SnnClassifier build() configuration exceeds the supported int64 shape range");
    }
    if (!std::isfinite(decay) || decay < 0.0 || !(decay < 1.0)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::SnnClassifier build() requires finite decay in [0, 1), got " +
                              std::to_string(decay));
    }
    if (!std::isfinite(threshold) || !(threshold > 0.0)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::SnnClassifier build() requires finite positive threshold, got " +
                              std::to_string(threshold));
    }
    if (!std::isfinite(alpha) || !(alpha > 0.0)) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::SnnClassifier build() requires finite positive alpha, got " + std::to_string(alpha));
    }
    const DTypeCode code = dtype.code();
    const bool supported =
        code == DTypeCode::kFloat32 || code == DTypeCode::kFloat16 || code == DTypeCode::kBFloat16;
    if (!supported) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::SnnClassifier build() does not support dtype '" +
                              std::string(dtype.name()) +
                              "' (supports float32/float16/bfloat16 only)");
    }

    Value* x = inputs[0];
    const Shape expected_shape({batch, num_steps, input_dim});
    if (!(x->type().shape == expected_shape)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::SnnClassifier build() requires input shape " +
                              expected_shape.to_string() + ", got " + x->type().shape.to_string());
    }
    if (!(x->type().dtype == dtype)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::SnnClassifier build() requires input dtype '" +
                              std::string(dtype.name()) + "', got '" +
                              std::string(x->type().dtype.name()) + "'");
    }

    const Result<std::vector<std::span<Value* const>>> child_params =
        SliceChildParams(children_for_build, params, "SnnClassifier");
    if (!child_params.is_ok()) return child_params.status();

    const AttrMap input_flat_attrs{{"target_shape", Shape({child_rows, input_dim})}};
    const Result<Node*> input_flat_node =
        create_node_with_inferred_types(graph, "reshape", {x}, input_flat_attrs);
    if (!input_flat_node.is_ok()) return input_flat_node.status();
    const Result<std::vector<Value*>> input_out = children_for_build[0].build(
        graph, std::vector<Value*>{input_flat_node.value()->output(0)}, child_params.value()[0]);
    if (!input_out.is_ok()) return input_out.status();

    const AttrMap hidden_steps_attrs{{"target_shape", Shape({batch, num_steps, hidden_dim})}};
    const Result<Node*> hidden_steps_node = create_node_with_inferred_types(
        graph, "reshape", {input_out.value()[0]}, hidden_steps_attrs);
    if (!hidden_steps_node.is_ok()) return hidden_steps_node.status();
    const Result<std::vector<Value*>> spikes_out = children_for_build[1].build(
        graph, std::vector<Value*>{hidden_steps_node.value()->output(0)}, child_params.value()[1]);
    if (!spikes_out.is_ok()) return spikes_out.status();

    const AttrMap spike_flat_attrs{{"target_shape", Shape({child_rows, hidden_dim})}};
    const Result<Node*> spike_flat_node = create_node_with_inferred_types(
        graph, "reshape", {spikes_out.value()[0]}, spike_flat_attrs);
    if (!spike_flat_node.is_ok()) return spike_flat_node.status();
    const Result<std::vector<Value*>> output_out = children_for_build[2].build(
        graph, std::vector<Value*>{spike_flat_node.value()->output(0)}, child_params.value()[2]);
    if (!output_out.is_ok()) return output_out.status();

    const AttrMap logits_steps_attrs{{"target_shape", Shape({batch, num_steps, num_classes})}};
    const Result<Node*> logits_steps_node = create_node_with_inferred_types(
        graph, "reshape", {output_out.value()[0]}, logits_steps_attrs);
    if (!logits_steps_node.is_ok()) return logits_steps_node.status();
    const AttrMap sum_attrs{
        {"axes", std::vector<int64_t>{1}},
        {"keepdims", false},
    };
    const Result<Node*> logits_node = create_node_with_inferred_types(
        graph, "sum", {logits_steps_node.value()->output(0)}, sum_attrs);
    if (!logits_node.is_ok()) return logits_node.status();
    return std::vector<Value*>{logits_node.value()->output(0)};
  };

  return Module{std::move(name), {}, std::move(children), std::move(build_fn)};
}

// num_nodes/in_features/out_features 与两条拓扑向量是已批准固定接口。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Module GraphConv(std::string name, int64_t num_nodes, int64_t in_features, int64_t out_features,
                 std::vector<int64_t> source_indices, std::vector<int64_t> target_indices,
                 DType dtype) {
  const bool edge_count_valid = std::in_range<int64_t>(source_indices.size());
  const int64_t edge_count =
      edge_count_valid ? static_cast<int64_t>(source_indices.size()) : int64_t{0};
  const std::optional<int64_t> edge_output_numel_result =
      checked_nonnegative_product({edge_count, out_features});
  const bool arithmetic_valid =
      edge_count_valid && checked_nonnegative_product({num_nodes, in_features}).has_value() &&
      checked_nonnegative_product({num_nodes, out_features}).has_value() &&
      checked_nonnegative_product({edge_count, in_features}).has_value() &&
      checked_nonnegative_product({in_features, out_features}).has_value() &&
      edge_output_numel_result.has_value();
  const int64_t child_edge_count = arithmetic_valid ? edge_count : 1;
  const int64_t child_in_features = arithmetic_valid ? in_features : 1;
  const int64_t child_out_features = arithmetic_valid ? out_features : 1;
  std::vector<Module> children{
      Linear("linear", child_edge_count, child_in_features, child_out_features,
             /*with_bias=*/false, dtype),
  };
  const std::vector<Module> children_for_build = children;

  // 拓扑向量按值进入工厂后移入闭包,使 Module 独立持有静态图结构。
  // inputs/params 相邻同型 span 形参是 BuildFn 契约固定形态(ARCH-071)。
  // NOLINTBEGIN(bugprone-easily-swappable-parameters)
  BuildFn build_fn =
      [num_nodes, in_features, out_features, edge_count, edge_count_valid, arithmetic_valid,
       edge_output_numel = edge_output_numel_result.value_or(int64_t{0}),
       source_indices = std::move(source_indices), target_indices = std::move(target_indices),
       dtype, children_for_build](Graph& graph, std::span<Value* const> inputs,
                                  std::span<Value* const> params) -> Result<std::vector<Value*>> {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    if (inputs.size() != 1) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::GraphConv build() expects 1 input, got " + std::to_string(inputs.size()));
    }
    if (num_nodes <= 0 || in_features <= 0 || out_features <= 0) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::GraphConv build() requires num_nodes, in_features, and out_features to be "
          "positive, got num_nodes=" +
              std::to_string(num_nodes) + " in_features=" + std::to_string(in_features) +
              " out_features=" + std::to_string(out_features));
    }
    if (!edge_count_valid) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::GraphConv build() edge count exceeds int64 range");
    }
    if (!arithmetic_valid) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::GraphConv build() configuration exceeds the supported int64 shape range");
    }
    if (source_indices.empty()) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::GraphConv build() requires at least one edge");
    }
    if (source_indices.size() != target_indices.size()) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::GraphConv build() requires source_indices and target_indices of equal length, got " +
              std::to_string(source_indices.size()) + " and " +
              std::to_string(target_indices.size()));
    }
    const Status source_precision =
        validate_index_constant_precision("nn::GraphConv build() source", source_indices);
    if (!source_precision.is_ok()) return source_precision;
    const Status target_precision =
        validate_index_constant_precision("nn::GraphConv build() target", target_indices);
    if (!target_precision.is_ok()) return target_precision;
    for (size_t edge = 0; edge < source_indices.size(); ++edge) {
      if (source_indices[edge] < 0 || source_indices[edge] >= num_nodes) {
        return Status::make(ErrorCode::kInvalidArgument,
                            "nn::GraphConv build() source index out of range at edge " +
                                std::to_string(edge) + ": " + std::to_string(source_indices[edge]));
      }
      if (target_indices[edge] < 0 || target_indices[edge] >= num_nodes) {
        return Status::make(ErrorCode::kInvalidArgument,
                            "nn::GraphConv build() target index out of range at edge " +
                                std::to_string(edge) + ": " + std::to_string(target_indices[edge]));
      }
    }
    if (!is_supported_float_dtype(dtype)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::GraphConv build() does not support dtype '" +
                              std::string(dtype.name()) +
                              "' (supports float32/float16/bfloat16 only)");
    }

    Value* x = inputs[0];
    const Shape expected_shape({num_nodes, in_features});
    if (!(x->type().shape == expected_shape)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::GraphConv build() requires input shape " +
                              expected_shape.to_string() + ", got " + x->type().shape.to_string());
    }
    if (!(x->type().dtype == dtype)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::GraphConv build() requires input dtype '" +
                              std::string(dtype.name()) + "', got '" +
                              std::string(x->type().dtype.name()) + "'");
    }

    std::vector<int64_t> out_degree(static_cast<size_t>(num_nodes), 0);
    std::vector<int64_t> in_degree(static_cast<size_t>(num_nodes), 0);
    for (size_t edge = 0; edge < source_indices.size(); ++edge) {
      ++out_degree[static_cast<size_t>(source_indices[edge])];
      ++in_degree[static_cast<size_t>(target_indices[edge])];
    }
    std::vector<double> weight_values;
    weight_values.reserve(static_cast<size_t>(edge_output_numel));
    for (size_t edge = 0; edge < source_indices.size(); ++edge) {
      const double degree_product =
          static_cast<double>(out_degree[static_cast<size_t>(source_indices[edge])]) *
          static_cast<double>(in_degree[static_cast<size_t>(target_indices[edge])]);
      const double weight = 1.0 / std::sqrt(degree_product);
      for (int64_t feature = 0; feature < out_features; ++feature) {
        weight_values.push_back(weight);
      }
    }

    const Result<std::vector<std::span<Value* const>>> child_params =
        SliceChildParams(children_for_build, params, "GraphConv");
    if (!child_params.is_ok()) return child_params.status();
    const Device device = x->type().device;
    const Result<Node*> source_node = make_index_constant(graph, source_indices, device);
    if (!source_node.is_ok()) return source_node.status();
    const Result<Node*> target_node = make_index_constant(graph, target_indices, device);
    if (!target_node.is_ok()) return target_node.status();

    const Result<Node*> neighbors_node =
        create_node_with_inferred_types(graph, "gather", {x, source_node.value()->output(0)});
    if (!neighbors_node.is_ok()) return neighbors_node.status();
    const Result<std::vector<Value*>> messages = children_for_build[0].build(
        graph, std::vector<Value*>{neighbors_node.value()->output(0)}, child_params.value()[0]);
    if (!messages.is_ok()) return messages.status();
    const Shape weights_shape({edge_count, out_features});
    const Result<Node*> weights_node =
        make_constant_values(graph, weights_shape, dtype, device, std::move(weight_values));
    if (!weights_node.is_ok()) return weights_node.status();
    const Result<Node*> weighted_node = create_node_with_inferred_types(
        graph, "mul", {messages.value()[0], weights_node.value()->output(0)});
    if (!weighted_node.is_ok()) return weighted_node.status();
    const AttrMap scatter_attrs{{"output_shape", Shape({num_nodes, out_features})}};
    const Result<Node*> output_node = create_node_with_inferred_types(
        graph, "scatter_add", {weighted_node.value()->output(0), target_node.value()->output(0)},
        scatter_attrs);
    if (!output_node.is_ok()) return output_node.status();
    return std::vector<Value*>{output_node.value()->output(0)};
  };

  return Module{std::move(name), {}, std::move(children), std::move(build_fn)};
}

// num_nodes/num_hyperedges/in_features/out_features 与关联向量是批准接口。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Module HypergraphConv(std::string name, int64_t num_nodes, int64_t num_hyperedges,
                      int64_t in_features, int64_t out_features, std::vector<int64_t> node_indices,
                      std::vector<int64_t> hyperedge_indices, DType dtype) {
  const bool incidence_count_valid = std::in_range<int64_t>(node_indices.size());
  const int64_t incidence_count =
      incidence_count_valid ? static_cast<int64_t>(node_indices.size()) : int64_t{0};
  const std::optional<int64_t> node_input_numel_result =
      checked_nonnegative_product({num_nodes, in_features});
  const std::optional<int64_t> incidence_input_numel_result =
      checked_nonnegative_product({incidence_count, in_features});
  const bool arithmetic_valid =
      incidence_count_valid && node_input_numel_result.has_value() &&
      checked_nonnegative_product({num_nodes, out_features}).has_value() &&
      checked_nonnegative_product({num_hyperedges, in_features}).has_value() &&
      checked_nonnegative_product({in_features, out_features}).has_value() &&
      incidence_input_numel_result.has_value();
  const int64_t child_num_nodes = arithmetic_valid ? num_nodes : 1;
  const int64_t child_in_features = arithmetic_valid ? in_features : 1;
  const int64_t child_out_features = arithmetic_valid ? out_features : 1;
  std::vector<Module> children{
      Linear("linear", child_num_nodes, child_in_features, child_out_features,
             /*with_bias=*/false, dtype),
  };
  const std::vector<Module> children_for_build = children;

  // 关联向量按值进入工厂后移入闭包,使 Module 独立持有静态超图结构。
  // inputs/params 相邻同型 span 形参是 BuildFn 契约固定形态(ARCH-071)。
  // NOLINTBEGIN(bugprone-easily-swappable-parameters)
  BuildFn build_fn =
      [num_nodes, num_hyperedges, in_features, out_features, incidence_count, incidence_count_valid,
       arithmetic_valid, node_input_numel = node_input_numel_result.value_or(int64_t{0}),
       incidence_input_numel = incidence_input_numel_result.value_or(int64_t{0}),
       node_indices = std::move(node_indices), hyperedge_indices = std::move(hyperedge_indices),
       dtype, children_for_build](Graph& graph, std::span<Value* const> inputs,
                                  std::span<Value* const> params) -> Result<std::vector<Value*>> {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    if (inputs.size() != 1) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::HypergraphConv build() expects 1 input, got " + std::to_string(inputs.size()));
    }
    if (num_nodes <= 0 || num_hyperedges <= 0 || in_features <= 0 || out_features <= 0) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::HypergraphConv build() requires all dimensions to be positive, got num_nodes=" +
              std::to_string(num_nodes) + " num_hyperedges=" + std::to_string(num_hyperedges) +
              " in_features=" + std::to_string(in_features) +
              " out_features=" + std::to_string(out_features));
    }
    if (!incidence_count_valid) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::HypergraphConv build() incidence count exceeds int64 range");
    }
    if (!arithmetic_valid) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::HypergraphConv build() configuration exceeds the supported int64 shape range");
    }
    if (node_indices.size() != hyperedge_indices.size()) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "nn::HypergraphConv build() requires node_indices and hyperedge_indices of equal "
          "length, got " +
              std::to_string(node_indices.size()) + " and " +
              std::to_string(hyperedge_indices.size()));
    }
    const Status node_precision =
        validate_index_constant_precision("nn::HypergraphConv build() node", node_indices);
    if (!node_precision.is_ok()) return node_precision;
    const Status hyperedge_precision = validate_index_constant_precision(
        "nn::HypergraphConv build() hyperedge", hyperedge_indices);
    if (!hyperedge_precision.is_ok()) return hyperedge_precision;
    std::set<std::pair<int64_t, int64_t>> incidences;
    for (size_t incidence = 0; incidence < node_indices.size(); ++incidence) {
      const int64_t node = node_indices[incidence];
      const int64_t hyperedge = hyperedge_indices[incidence];
      if (node < 0 || node >= num_nodes) {
        return Status::make(ErrorCode::kInvalidArgument,
                            "nn::HypergraphConv build() node index out of range at incidence " +
                                std::to_string(incidence) + ": " + std::to_string(node));
      }
      if (hyperedge < 0 || hyperedge >= num_hyperedges) {
        return Status::make(
            ErrorCode::kInvalidArgument,
            "nn::HypergraphConv build() hyperedge index out of range at incidence " +
                std::to_string(incidence) + ": " + std::to_string(hyperedge));
      }
      if (!incidences.emplace(node, hyperedge).second) {
        return Status::make(ErrorCode::kInvalidArgument,
                            "nn::HypergraphConv build() rejects duplicate incidence pair (" +
                                std::to_string(node) + ", " + std::to_string(hyperedge) + ")");
      }
    }
    if (!is_supported_float_dtype(dtype)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::HypergraphConv build() does not support dtype '" +
                              std::string(dtype.name()) +
                              "' (supports float32/float16/bfloat16 only)");
    }

    Value* x = inputs[0];
    const Shape expected_shape({num_nodes, in_features});
    if (!(x->type().shape == expected_shape)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::HypergraphConv build() requires input shape " +
                              expected_shape.to_string() + ", got " + x->type().shape.to_string());
    }
    if (!(x->type().dtype == dtype)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "nn::HypergraphConv build() requires input dtype '" +
                              std::string(dtype.name()) + "', got '" +
                              std::string(x->type().dtype.name()) + "'");
    }

    std::vector<int64_t> node_degree(static_cast<size_t>(num_nodes), 0);
    std::vector<int64_t> edge_degree(static_cast<size_t>(num_hyperedges), 0);
    for (size_t incidence = 0; incidence < node_indices.size(); ++incidence) {
      ++node_degree[static_cast<size_t>(node_indices[incidence])];
      ++edge_degree[static_cast<size_t>(hyperedge_indices[incidence])];
    }
    std::vector<double> node_scale_values;
    node_scale_values.reserve(static_cast<size_t>(node_input_numel));
    for (int64_t node = 0; node < num_nodes; ++node) {
      const int64_t degree = node_degree[static_cast<size_t>(node)];
      const double scale = degree == 0 ? 0.0 : 1.0 / std::sqrt(static_cast<double>(degree));
      for (int64_t feature = 0; feature < in_features; ++feature) {
        node_scale_values.push_back(scale);
      }
    }
    std::vector<double> edge_scale_values;
    edge_scale_values.reserve(static_cast<size_t>(incidence_input_numel));
    for (size_t incidence = 0; incidence < node_indices.size(); ++incidence) {
      const double scale =
          1.0 / static_cast<double>(edge_degree[static_cast<size_t>(hyperedge_indices[incidence])]);
      for (int64_t feature = 0; feature < in_features; ++feature) {
        edge_scale_values.push_back(scale);
      }
    }

    const Result<std::vector<std::span<Value* const>>> child_params =
        SliceChildParams(children_for_build, params, "HypergraphConv");
    if (!child_params.is_ok()) return child_params.status();
    const Device device = x->type().device;
    const Result<Node*> node_indices_node = make_index_constant(graph, node_indices, device);
    if (!node_indices_node.is_ok()) return node_indices_node.status();
    const Result<Node*> hyperedge_indices_node =
        make_index_constant(graph, hyperedge_indices, device);
    if (!hyperedge_indices_node.is_ok()) return hyperedge_indices_node.status();
    const Result<Node*> node_scale_node = make_constant_values(
        graph, Shape({num_nodes, in_features}), dtype, device, std::move(node_scale_values));
    if (!node_scale_node.is_ok()) return node_scale_node.status();

    const Result<Node*> normalized_node =
        create_node_with_inferred_types(graph, "mul", {x, node_scale_node.value()->output(0)});
    if (!normalized_node.is_ok()) return normalized_node.status();
    const Result<Node*> incidence_values_node = create_node_with_inferred_types(
        graph, "gather",
        {normalized_node.value()->output(0), node_indices_node.value()->output(0)});
    if (!incidence_values_node.is_ok()) return incidence_values_node.status();
    const Result<Node*> edge_scale_node = make_constant_values(
        graph, Shape({incidence_count, in_features}), dtype, device, std::move(edge_scale_values));
    if (!edge_scale_node.is_ok()) return edge_scale_node.status();
    const Result<Node*> scaled_incidence_node = create_node_with_inferred_types(
        graph, "mul",
        {incidence_values_node.value()->output(0), edge_scale_node.value()->output(0)});
    if (!scaled_incidence_node.is_ok()) return scaled_incidence_node.status();
    const AttrMap to_edges_attrs{{"output_shape", Shape({num_hyperedges, in_features})}};
    const Result<Node*> edge_values_node = create_node_with_inferred_types(
        graph, "scatter_add",
        {scaled_incidence_node.value()->output(0), hyperedge_indices_node.value()->output(0)},
        to_edges_attrs);
    if (!edge_values_node.is_ok()) return edge_values_node.status();
    const Result<Node*> back_values_node = create_node_with_inferred_types(
        graph, "gather",
        {edge_values_node.value()->output(0), hyperedge_indices_node.value()->output(0)});
    if (!back_values_node.is_ok()) return back_values_node.status();
    const AttrMap to_nodes_attrs{{"output_shape", Shape({num_nodes, in_features})}};
    const Result<Node*> nodes_node = create_node_with_inferred_types(
        graph, "scatter_add",
        {back_values_node.value()->output(0), node_indices_node.value()->output(0)},
        to_nodes_attrs);
    if (!nodes_node.is_ok()) return nodes_node.status();
    const Result<Node*> renormalized_node = create_node_with_inferred_types(
        graph, "mul", {nodes_node.value()->output(0), node_scale_node.value()->output(0)});
    if (!renormalized_node.is_ok()) return renormalized_node.status();
    const Result<std::vector<Value*>> output = children_for_build[0].build(
        graph, std::vector<Value*>{renormalized_node.value()->output(0)}, child_params.value()[0]);
    if (!output.is_ok()) return output.status();
    return output.value();
  };

  return Module{std::move(name), {}, std::move(children), std::move(build_fn)};
}

}  // namespace frame::nn
