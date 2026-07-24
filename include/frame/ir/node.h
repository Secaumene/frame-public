#pragma once
// Node / Value / TensorType:图 IR 的算子实例与 SSA 值。均为值语义,无虚函数。
// 见 docs/architecture/ir-design.md 第2章。

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/ir/attribute.h>

namespace frame::ir {

class Node;  // 前向声明:Value 的 producer

// 数据布局标签(v0 最小集,layout_assignment pass 负责细化)。
enum class Layout : uint8_t {
  kUnknown = 0,
  kRowMajor,
};

// 张量类型:显式携带 dtype/shape/layout/device,编译期完成推断。
struct TensorType {
  DType dtype{DTypeCode::kFloat32};
  Shape shape;
  Layout layout = Layout::kUnknown;
  Device device{};
};

// Value:SSA 值 —— 恰有一个 producer(node + 输出序号,不变量 V1)。
class Value {
 public:
  Value() = default;

  const TensorType& type() const { return type_; }
  Node* producer() const { return producer_; }
  int32_t output_index() const { return output_index_; }

 private:
  friend class Node;
  friend class Graph;
  TensorType type_{};
  Node* producer_ = nullptr;
  int32_t output_index_ = 0;
};

// Node:算子实例 —— op 名(指向已注册 OpSchema)+ 输入 Value* + 输出 Value + 属性字典。
class Node {
 public:
  Node() = default;
  explicit Node(std::string op) : op_(std::move(op)) {}

  std::string_view op() const { return op_; }
  const std::vector<Value*>& inputs() const { return inputs_; }
  const std::vector<Value>& outputs() const { return outputs_; }

  // 按下标取输出 Value 的可变指针:仅暴露单个元素指针,不暴露可变容器本身——
  // 收紧目标是防止 push_back/resize 等操作使已发出的 Value* 失效,破坏 SSA
  // 指针稳定性(决议点 5-②,取代此前的非 const outputs() 整体访问)。越界
  // (index < 0 或 >= outputs_.size())返回 nullptr。外部合法可变用途(如
  // decomposition 内标记新建节点唯一输出,见 Graph::mark_output(Node*,
  // int32_t) 重载)经此访问器取得指针即可。
  Value* output(int32_t index) {
    if (index < 0 || static_cast<size_t>(index) >= outputs_.size()) return nullptr;
    return &outputs_[static_cast<size_t>(index)];
  }

  // 构图期属性写入(公开):按名覆盖或新增一个属性。
  void set_attr(std::string name, AttrValue value);

  // 按名取回原始属性变体;不存在返回 nullptr。
  const AttrValue* find_attr(std::string_view name) const;

  // 按名取回强类型属性;不存在或类型不符返回 nullptr。
  template <typename T>
  const T* attr(std::string_view name) const {
    const AttrValue* value = find_attr(name);
    if (value == nullptr) return nullptr;
    return std::get_if<T>(value);
  }

  // 只读枚举全部属性(序列化/调试遍历用)。枚举顺序不确定(底层是
  // unordered_map,顺序随实现/运行而变);任何需要确定性输出的场景
  // (序列化文本、按字典序排列的错误消息列表等)必须由调用方自行按名排序——
  // 字典序是序列化层的契约,不是 Node 本身的契约。
  const std::unordered_map<std::string, AttrValue>& attrs() const { return attrs_; }

 private:
  friend class Graph;
  std::string op_;
  std::vector<Value*> inputs_;
  std::vector<Value> outputs_;
  std::unordered_map<std::string, AttrValue> attrs_;
};

}  // namespace frame::ir
