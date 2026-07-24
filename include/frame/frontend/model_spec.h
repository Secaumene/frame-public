#pragma once
// ModelSpec:前端 DSL(JSON 模型描述,schema_version 0)对应的纯 C++ 模型描述
// 结构体家族;唯一权威契约见 docs/architecture/frontend-dsl.md(ADR-0017)。
// 本文件不解析 JSON——JSON 层的解析与 schema_version 校验(FE-001)属工具层
// (tools/frame_dslc)职责,不下沉进本库(ADR-0018)。frontend-dsl.md 第 3 节
// FE-002~005 按职责分层执行:FE-002/003/005 全部、以及 FE-004 中 Activation
// 位模式部分在本文件的 validate() 内执行;FE-001(schema_version)与 FE-004
// 中 kind/loss.kind/optimizer.kind/model.dtype 等字符串枚举部分由 JSON 载入层
// (tools/frame_dslc/json_loader)在解析阶段拦截,不下沉进本函数。分层依据见
// docs/architecture/frontend-dsl.md 第 3 节。

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <frame/core/macros.h>
#include <frame/core/status.h>

namespace frame::frontend {

// linear 层可选的激活函数(v0 仅两种取值,frontend-dsl.md 第 1 节)。底层类型
// 显式收窄为 uint8_t(仓内 AttrType/OpTrait/ErrorCode 等封闭枚举同款惯例,
// 规避 performance-enum-size)。
enum class Activation : uint8_t {
  kNone,
  kRelu,
};

// data 段单个张量条目的初始化方式(frontend-dsl.md 第 2 节)。底层类型显式
// 收窄为 uint8_t(理由同 Activation)。
enum class InitKind : uint8_t {
  kInline,
  kUniformSeeded,
};

// 单个数据输入的名字与静态形状(v0 恰一个数据输入)。
struct InputSpec {
  std::string name;
  std::vector<int64_t> shape;
};

// linear 层描述:lower 为 matmul(+ 可选 add bias)(+ 可选 relu)。bias_shape
// 缺省表示无 bias;给出时须为全形状 [batch, out](v0 无广播)。
struct LinearLayerSpec {
  std::string name;
  std::string input;
  std::vector<int64_t> weight_shape;
  std::optional<std::vector<int64_t>> bias_shape;
  Activation activation = Activation::kNone;
};

// mse 损失描述:prediction 引用某 layer 名,target_shape 须与该层输出形状一致。
struct LossSpec {
  std::string prediction;
  std::vector<int64_t> target_shape;
};

// sgd 优化器描述(v0 唯一支持的优化器,固定学习率)。
struct OptimizerSpec {
  double learning_rate = 0.0;
};

// 训练循环控制参数。
struct TrainingSpec {
  int32_t steps = 0;
  uint32_t seed = 0;
  int32_t log_every = 0;
};

// 单个张量(数据输入或 target)的初始化描述:kind == kInline 时按 values 内联
// 取值(元素数须等于对应形状 numel);kind == kUniformSeeded 时从 [lo, hi) 均匀
// 采样(v0 唯一随机源 std::mt19937,见 frontend-dsl.md 第 2 节)。
struct TensorDataSpec {
  InitKind kind = InitKind::kUniformSeeded;
  std::vector<float> values;
  float lo = 0.0F;
  float hi = 0.0F;
};

// 全部参数(逐层 weight/bias)的统一初始化范围(v0 不支持参数 inline,恒为
// 均匀采样;frontend-dsl.md 第 2 节 data.params)。
struct ParamInitSpec {
  float weight_lo = 0.0F;
  float weight_hi = 0.0F;
  float bias_lo = 0.0F;
  float bias_hi = 0.0F;
};

// 完整模型描述:与 JSON 顶层结构一一对应(schema_version/model.dtype/
// layers[].kind/loss.kind/optimizer.kind 在 v0 恒为唯一取值,已由本结构体
// 设计固定,不再以字符串字段出现)。
struct ModelSpec {
  std::string name;
  int64_t batch = 0;
  std::vector<InputSpec> inputs;
  std::vector<LinearLayerSpec> layers;
  LossSpec loss;
  OptimizerSpec optimizer;
  TrainingSpec training;
  std::unordered_map<std::string, TensorDataSpec> data;
  ParamInitSpec param_init;
};

// 校验 spec 是否满足 frontend-dsl.md 第 3 节 FE-002~005(名字引用闭包、形状链
// 一致、枚举白名单、data 完整性)以及第 1 节的结构性约束(batch/steps/
// learning_rate 等正值约束、v0 恰一个数据输入)。违例返回 kInvalidArgument,
// 消息英文并含违例字段名(LANG-005)。lower_to_graph/lower_to_inference_graph
// 内部均先调用本函数。
FRAME_API Status validate(const ModelSpec& spec);

}  // namespace frame::frontend
