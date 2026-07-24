#pragma once
// 节点属性:AttrType 封闭枚举 + AttrValue 变体类型。均为值语义,无虚函数。
// 【ARCH-020】属性类型限于封闭集合,不得超出;扩展需先修订
// docs/architecture/ir-design.md 并通过 design-reviewer。

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <frame/core/dtype.h>
#include <frame/core/shape.h>

namespace frame::ir {

// 属性类型封闭枚举(与 AttrValue 变体各备选项一一对应,ARCH-020)。
enum class AttrType : uint8_t {
  kInt64,
  kDouble,
  kString,
  kBool,
  kInt64Array,
  kDoubleArray,
  kDType,
  kShape,
};

// AttrValue:属性值变体。备选项集合 = {int64, double, string, bool,
// int64 数组, double 数组, dtype, shape}(ARCH-020,与 AttrType 逐一对应)。
using AttrValue = std::variant<int64_t,               // kInt64
                               double,                // kDouble
                               std::string,           // kString
                               bool,                  // kBool
                               std::vector<int64_t>,  // kInt64Array
                               std::vector<double>,   // kDoubleArray
                               DType,                 // kDType
                               Shape>;                // kShape

// 变体备选项数与 AttrType 封闭枚举末位一致的编译期校验(ARCH-020):任一方
// 新增/重排备选项而另一方未同步时,此断言立即编译失败。引用末位枚举子
// kShape(而非另加 kCount 哨兵——加哨兵会使 AttrType 的穷举 switch 出现
// 未处理分支,触碰 ARCH-020 封闭枚举纪律)。
static_assert(std::variant_size_v<AttrValue> == static_cast<std::size_t>(AttrType::kShape) + 1,
              "AttrValue variant size must match AttrType's closed enum span (ARCH-020)");

// variant 备选项下标 → AttrType 的一一映射(ARCH-020,与 AttrValue 声明顺序
// 严格一致;二者任一方新增/重排备选项都必须同步修改另一方)。
AttrType attr_type_of(const AttrValue& value);

// AttrType 的英文名(序列化文本/错误消息用,LANG-005)。
std::string_view attr_type_name(AttrType type);

}  // namespace frame::ir
