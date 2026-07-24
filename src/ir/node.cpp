// Node / Value 图 IR 的实现单元。

#include <utility>

#include <frame/ir/node.h>

namespace frame::ir {

void Node::set_attr(std::string name, AttrValue value) {
  attrs_[std::move(name)] = std::move(value);
}

const AttrValue* Node::find_attr(std::string_view name) const {
  const auto it = attrs_.find(std::string(name));
  if (it == attrs_.end()) return nullptr;
  return &it->second;
}

}  // namespace frame::ir
