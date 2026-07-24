// Module::parameters()/build() 与 add_parameter_inputs 的实现(ARCH-071/073/
// 074,docs/architecture/nn-design.md)。见 include/frame/nn/module.h 头注释。

#include <string>
#include <utility>
#include <vector>

#include <frame/ir/graph.h>
#include <frame/nn/module.h>

namespace frame::nn {

namespace {

// 先序遍历收集参数清单:prefix 是"到 module 上一层"的已拼接路径(不含
// module 自己的 name),首层调用传空串;分隔符固定为 "."(ARCH-073)。
void CollectParameters(const Module& module, const std::string& prefix,
                       std::vector<ParamSpec>& out) {
  const std::string self_prefix = prefix.empty() ? module.name : prefix + "." + module.name;
  for (const ParamSpec& param : module.params) {
    ParamSpec qualified = param;
    qualified.name = self_prefix + "." + param.name;
    out.push_back(std::move(qualified));
  }
  for (const Module& child : module.children) {
    CollectParameters(child, self_prefix, out);
  }
}

}  // namespace

std::vector<ParamSpec> Module::parameters() const {
  std::vector<ParamSpec> out;
  CollectParameters(*this, "", out);
  return out;
}

Result<std::vector<ir::Value*>> Module::build(ir::Graph& graph, std::span<ir::Value* const> inputs,
                                              std::span<ir::Value* const> params) const {
  const std::vector<ParamSpec> flat_params = parameters();
  if (params.size() != flat_params.size()) {
    return Status::make(ErrorCode::kInvalidArgument, "module '" + name + "' build() params size (" +
                                                         std::to_string(params.size()) +
                                                         ") does not match parameters().size() (" +
                                                         std::to_string(flat_params.size()) + ")");
  }
  return build_fn(graph, inputs, params);
}

Result<std::vector<ir::Value*>> add_parameter_inputs(ir::Graph& graph,
                                                     std::span<const ParamSpec> params) {
  std::vector<ir::Value*> values;
  values.reserve(params.size());
  for (const ParamSpec& param : params) {
    const Result<ir::Value*> value = graph.add_graph_input(param.type);
    if (!value.is_ok()) {
      return value.status();
    }
    values.push_back(value.value());
  }
  return values;
}

}  // namespace frame::nn
