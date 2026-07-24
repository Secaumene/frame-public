// OpRegistry 算子 schema 注册表的实现单元。
// FRAME_REGISTER_OP 宏(见 include/frame/ops/op_registry.h)展开为内部链接
// (static)的静态初始化器,返回本文件 register_op 的 OpSchema& 供链式声明。

#include <algorithm>
#include <string>
#include <unordered_map>

#include <frame/ops/op_registry.h>

#include "registration_diagnostics.h"  // fatal_registration_error(REUSE-002,M9 起与 OpSchema 共享单份)

namespace frame::ops {

OpRegistry& OpRegistry::instance() noexcept {
  static OpRegistry registry;
  return registry;
}

OpSchema& OpRegistry::register_op(std::string_view name) noexcept {
  // ①正则校验:共用 ir 层公开工具函数(ir::matches_op_name_charset,见
  // include/frame/ir/graph.h),避免与 ir 构图侧同签名同函数体的第二份复制
  // (REUSE-002);ops→ir 是合法依赖方向(ARCH-001)。
  if (!ir::matches_op_name_charset(name)) {
    fatal_registration_error(name, "invalid op name, must match ^[a-z][a-z0-9_]*$ (ARCH-040)");
  }
  const std::string key(name);
  // ②全局唯一校验。
  if (schemas_.find(key) != schemas_.end()) {
    fatal_registration_error(name, "duplicate op name, already registered (ARCH-040)");
  }
  // ③非 ir 层保留名校验(kGraphInputOp/kGraphOutputMarker 均不得经此注册)。
  if (name == ir::kGraphInputOp || name == ir::kGraphOutputMarker) {
    fatal_registration_error(
        name, "reserved op name, frame::ir::kGraphInputOp/kGraphOutputMarker are not registrable");
  }

  OpSchema& schema = schemas_.emplace(key, OpSchema{}).first->second;
  schema.name_ = key;
  return schema;
}

const OpSchema* OpRegistry::find(std::string_view name) const {
  const auto it = schemas_.find(std::string(name));
  if (it == schemas_.end()) return nullptr;
  return &it->second;
}

ir::OpQuery make_op_query() {
  ir::OpQuery query;

  query.op_registered = [](std::string_view op_name) {
    return OpRegistry::instance().find(op_name) != nullptr;
  };

  query.check_schema = [](const ir::Node& node) -> Status {
    const OpSchema* schema = OpRegistry::instance().find(node.op());
    if (schema == nullptr) {
      return Status::make(ErrorCode::kNotFound,
                          "op '" + std::string(node.op()) + "' is not registered");
    }
    // 输入个数校验(M9 前置设计:OpSchema 变长输入支持):有变长输入组(尾随,
    // note A/B)时改判"至少 min_input_count() 个",否则维持既有的定长恒等
    // 比较——两分支错误消息均含 op 名/实际数/(下限或恰好)数,ARCH-031 口径。
    if (schema->has_variadic_inputs()) {
      if (static_cast<int32_t>(node.inputs().size()) < schema->min_input_count()) {
        return Status::make(ErrorCode::kInvalidArgument,
                            "op '" + std::string(node.op()) + "' expects at least " +
                                std::to_string(schema->min_input_count()) + " input(s), got " +
                                std::to_string(node.inputs().size()));
      }
    } else if (node.inputs().size() != schema->inputs().size()) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "op '" + std::string(node.op()) + "' expects " +
                              std::to_string(schema->inputs().size()) + " input(s), got " +
                              std::to_string(node.inputs().size()));
    }
    if (node.outputs().size() != schema->outputs().size()) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "op '" + std::string(node.op()) + "' expects " +
                              std::to_string(schema->outputs().size()) + " output(s), got " +
                              std::to_string(node.outputs().size()));
    }

    // 必需属性存在 + 已存在属性类型匹配(ir::attr_type_of)。
    for (const OpAttrSpec& attr_spec : schema->attrs()) {
      const ir::AttrValue* value = node.find_attr(attr_spec.name);
      if (value == nullptr) {
        if (attr_spec.required) {
          return Status::make(ErrorCode::kInvalidArgument, "op '" + std::string(node.op()) +
                                                               "' is missing required attribute '" +
                                                               attr_spec.name + "'");
        }
        continue;
      }
      const ir::AttrType actual_type = ir::attr_type_of(*value);
      if (actual_type != attr_spec.type) {
        return Status::make(ErrorCode::kInvalidArgument,
                            "op '" + std::string(node.op()) + "' attribute '" + attr_spec.name +
                                "' has type " + std::string(ir::attr_type_name(actual_type)) +
                                ", expected " + std::string(ir::attr_type_name(attr_spec.type)));
      }
    }

    // 未知属性拒绝:枚举 node.attrs(),任何不在 schema 声明集合内的属性名均报错。
    for (const auto& [attr_name, attr_value] : node.attrs()) {
      (void)attr_value;
      const bool declared =
          std::any_of(schema->attrs().begin(), schema->attrs().end(),
                      [&attr_name](const OpAttrSpec& spec) { return spec.name == attr_name; });
      if (!declared) {
        return Status::make(
            ErrorCode::kInvalidArgument,
            "op '" + std::string(node.op()) + "' has undeclared attribute '" + attr_name + "'");
      }
    }

    return Status::ok();
  };

  return query;
}

}  // namespace frame::ops
