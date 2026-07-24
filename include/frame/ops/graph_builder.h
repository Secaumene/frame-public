#pragma once
// 图构建便捷 helper:create_node_with_inferred_types —— 经 OpRegistry 查
// schema、调用其 shape_infer 得到输出 shape,结合 dtype/device 契约构造完整
// TensorType 列表后调用 ir::Graph::create_node,省去调用方手工拼接
// output_types 的重复劳动(M12 决议点 C)。C++/Python 两侧共用同一份实现——
// Python 绑定层的七算子薄函数均经本 helper 构图(见 python/src/bind_ops.cpp),
// 不在绑定层另写第二份 schema 查询/shape 推断逻辑(REUSE-002)。

#include <string_view>
#include <unordered_map>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/macros.h>
#include <frame/core/status.h>
#include <frame/ir/attribute.h>
#include <frame/ir/node.h>

namespace frame::ir {
class Graph;  // 前向声明:仅按引用使用,见 include/frame/ir/graph.h
}  // namespace frame::ir

namespace frame::ops {

// 属性字典别名,与 ir::Node::attrs()/NodeContext::attrs 同一底层类型。
using AttrMap = std::unordered_map<std::string, ir::AttrValue>;

// 有输入版本(裁决修订 5):inputs 须非空(0 输入请改用下方 Device 重载)。
// 流程:①OpRegistry::find(op) 查 schema(不存在返回 kNotFound,消息含 op
// 名);②构造 NodeContext(input_types 取自 inputs 逐个 Value::type())调
// schema 的 shape_infer(未设置返回 kInvalidArgument;推断失败原样透传其
// Status,消息追加 op 名上下文);③按输入契约逐输出构造 TensorType——dtype/
// device 取 inputs[0] 的类型,layout 固定 Layout::kUnknown(具体布局由
// layout_assignment pass 事后指派,不在构图期确定);④调用
// graph.create_node 建节点;⑤逐条 attrs 写入 node->set_attr。
FRAME_API Result<ir::Node*> create_node_with_inferred_types(ir::Graph& graph, std::string_view op,
                                                            std::vector<ir::Value*> inputs,
                                                            const AttrMap& attrs = {});

// 0 输入版本(裁决修订 5,constant 等 0 输入算子专用):无输入可供推导
// dtype/device,故 device 由调用方经本形参显式提供;dtype 取自 attrs 的
// "dtype" 属性(不存在或类型不符返回 kInvalidArgument)。其余流程(查
// schema/调 shape_infer/create_node/写属性)与有输入版本一致。
FRAME_API Result<ir::Node*> create_node_with_inferred_types(ir::Graph& graph, std::string_view op,
                                                            Device device,
                                                            const AttrMap& attrs = {});

}  // namespace frame::ops
