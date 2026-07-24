#pragma once
// frame::nn 的 Module 数据模型与组合机制(docs/architecture/nn-design.md
// ARCH-071/072/073/074)。Module 是**值语义构图组合子**(非虚基类):持
// name/自身直接参数声明/子模块树(值语义)/类型擦除构建器闭包;build() 一律
// 经 ops::create_node_with_inferred_types 构图(REUSE-002,与 frontend
// lowering 同一份 helper),**零 eager、不触数值**——不得创建 Tensor、不得调
// kernel、不得读写数值内存。首批模块工厂(Linear/Relu/Sequential/MseLoss)见
// 同目录 layers.h。

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <frame/core/macros.h>
#include <frame/core/status.h>
#include <frame/ir/node.h>  // TensorType/Value/Node

namespace frame::ir {
class Graph;  // 前向声明:仅按引用使用,同 include/frame/ops/graph_builder.h 先例
}  // namespace frame::ir

namespace frame::nn {

// 参数初始化方式(ARCH-073):对齐 frame::frontend::InitKind/TensorDataSpec
// 现有两种初始化取值(nn 依赖集不含 frontend,故独立声明,语义对齐、数据不
// 共享,非重复实现——二者结构相同仅因面向同一概念,无可提取的公共逻辑)。
// 底层类型显式收窄为 uint8_t(仓内 ErrorCode/DTypeCode 等封闭枚举同款惯例)。
enum class InitKind : uint8_t {
  kUniformSeeded,
  kInline,
};

// 单个参数的初始化声明(仅声明,不物化数值——数值物化归调用方,M20 内 runner
// 仍走既有 GenerateHostTensorValues 路径不改,ARCH-073)。kind ==
// kUniformSeeded 时约定从 [lo, hi) 均匀采样;kind == kInline 时约定按 values
// 内联取值(元素数须等于对应 ParamSpec.type 的 numel);两种约定的校验/物化
// 逻辑均属调用方职责,本类型自身不做任何校验。
struct InitSpec {
  InitKind kind = InitKind::kUniformSeeded;
  float lo = 0.0F;
  float hi = 0.0F;
  std::vector<float> values;
};

// 单个参数声明(ARCH-073):name 是**局部名**(不带路径前缀,前缀由
// Module::parameters() 遍历时逐层拼接)、张量类型、初始化声明。
struct ParamSpec {
  std::string name;
  ir::TensorType type;
  InitSpec init;
};

// Module::build 的类型擦除构建器签名(ARCH-071):经 graph/inputs/params 构图,
// 返回输出 Value* 列表;异构容器(Sequential)与 Python 动态组合一律经本闭包
// 类型实现(同 include/frame/ops/kernel_registry.h::KernelFn、
// include/frame/ops/op_schema.h::GradientFn 先例)。禁止在闭包体内创建
// Tensor/调 kernel/读写数值内存(零 eager,ARCH-072)。
using BuildFn = std::function<Result<std::vector<ir::Value*>>(
    ir::Graph& graph, std::span<ir::Value* const> inputs, std::span<ir::Value* const> params)>;

// Module:值语义构图组合子(ARCH-071)。持 name(ASCII 标识)、自身直接参数
// 声明 params、子模块树 children(值语义,先序遍历)、类型擦除构建器
// build_fn。组合禁止白名单外 virtual/dynamic_cast/typeid(CPP-010/011)。
struct Module {
  std::string name;
  std::vector<ParamSpec> params;
  std::vector<Module> children;
  BuildFn build_fn;

  // 先序遍历扁平参数清单(ARCH-073):自身直接参数按声明序,再逐子模块递归
  // (先序);名字带路径前缀,分隔符固定为 "."(如 "mlp.layer0.weight"),同一
  // Module 树任意两次调用结果逐位相同(parameters() 是纯函数,确定性)。
  std::vector<ParamSpec> parameters() const;

  // 校验 params 尺寸 == parameters().size() 后转发给 build_fn(ARCH-071)。
  // 复合模块(如 Sequential)的**params 切片不变式**——按
  // [自身直接参数…, 子0 子树参数…, 子1 子树参数…] 先序分段切片,每段长度
  // = 对应子 parameters().size()——由各自的 build_fn 闭包内部实施,本函数
  // 只做整体尺寸校验。尺寸不符返回 kInvalidArgument(英文消息);校验通过后
  // 原样透传 build_fn 的 Result。
  Result<std::vector<ir::Value*>> build(ir::Graph& graph, std::span<ir::Value* const> inputs,
                                        std::span<ir::Value* const> params) const;
};

// 参数图输入的物化 helper(ARCH-074):按 params 清单序逐个
// graph.add_graph_input(param.type),返回 Value* 序与清单逐位对应;任一失败
// 原样透传其 Status(不吞错、不继续构图)。是否装配损失、target 从哪来、图
// 输入总序、mark_output 等装配决策一律归调用方,本函数只负责参数这一段。
FRAME_API Result<std::vector<ir::Value*>> add_parameter_inputs(ir::Graph& graph,
                                                               std::span<const ParamSpec> params);

}  // namespace frame::nn
