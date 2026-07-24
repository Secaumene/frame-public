// 节点属性 AttrType / AttrValue 的实现单元。

#include <frame/core/macros.h>
#include <frame/ir/attribute.h>

namespace frame::ir {

AttrType attr_type_of(const AttrValue& value) {
  // AttrValue 的 variant 备选项顺序与 AttrType 枚举顺序逐一对应(见头文件
  // 注释),variant::index() 直接强转即为对应的 AttrType。
  return static_cast<AttrType>(value.index());
}

std::string_view attr_type_name(AttrType type) {
  switch (type) {
    case AttrType::kInt64:
      return "int64";
    case AttrType::kDouble:
      return "double";
    case AttrType::kString:
      return "string";
    case AttrType::kBool:
      return "bool";
    case AttrType::kInt64Array:
      return "int64_array";
    case AttrType::kDoubleArray:
      return "double_array";
    case AttrType::kDType:
      return "dtype";
    case AttrType::kShape:
      return "shape";
  }
  // 不可达:AttrType 是封闭枚举(ARCH-020),switch 已穷举全部合法取值;
  // 命中此处即违反调用方不变量(非法枚举值)。
  FRAME_CHECK(false);
  return {};
}

}  // namespace frame::ir
