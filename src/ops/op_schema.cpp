// OpSchema builder 方法的实现单元:input/output/attr/trait/shape_infer/
// decomposition 等非内联成员的定义落点在此(头文件只留声明,避免每个包含者
// 重复生成代码)。noexcept 规约理由见头文件对应声明处的注释。

#include <string>

#include <frame/ops/op_schema.h>

#include "registration_diagnostics.h"  // fatal_registration_error(REUSE-002,与 op_registry.cpp 共享单份)

namespace frame::ops {

OpSchema& OpSchema::input(std::string_view name, std::string_view doc) noexcept {
  // note A:变长组必须尾随——已声明变长组后再调用 input() 违反该硬约束。
  if (has_variadic_input_) {
    fatal_registration_error(
        name_,
        "input() called after variadic_input(), the variadic input group must be trailing "
        "(M9 note A)");
  }
  inputs_.push_back(OpParam{std::string(name), std::string(doc)});
  return *this;
}

OpSchema& OpSchema::variadic_input(std::string_view name, std::string_view doc,
                                   int32_t min_count) noexcept {
  // note A:至多一个变长组。
  if (has_variadic_input_) {
    fatal_registration_error(
        name_,
        "variadic_input() called more than once, at most one variadic input group is "
        "allowed per schema (M9 note A)");
  }
  // note A:min_count 须 >= 0。
  if (min_count < 0) {
    const std::string reason = "variadic_input() min_count must be >= 0, got " +
                               std::to_string(min_count) + " (M9 note A)";
    fatal_registration_error(name_, reason);
  }
  variadic_input_param_ = OpParam{std::string(name), std::string(doc)};
  variadic_min_count_ = min_count;
  has_variadic_input_ = true;
  return *this;
}

OpSchema& OpSchema::output(std::string_view name, std::string_view doc) noexcept {
  outputs_.push_back(OpParam{std::string(name), std::string(doc)});
  return *this;
}

OpSchema& OpSchema::attr(std::string_view name, ir::AttrType type, bool required) noexcept {
  attrs_.push_back(OpAttrSpec{std::string(name), type, required});
  return *this;
}

OpSchema& OpSchema::trait(OpTrait trait) noexcept {
  traits_ |= static_cast<uint8_t>(1U << static_cast<uint8_t>(trait));
  return *this;
}

OpSchema& OpSchema::shape_infer(ShapeInferFn fn) noexcept {
  shape_infer_ = fn;
  return *this;
}

OpSchema& OpSchema::decomposition(DecomposeFn fn) noexcept {
  decompose_ = fn;
  return *this;
}

OpSchema& OpSchema::gradient(GradientFn fn) noexcept {
  // ARCH-062:带 kHasSideEffect trait 的算子不可微,builder 期即时 fail-fast
  // (与 M9 variadic_input 两条硬约束同一纪律)。该校验依赖调用方已先经
  // trait(OpTrait::kHasSideEffect) 声明该 trait(见头文件声明处顺序敏感说明)。
  if (has_trait(OpTrait::kHasSideEffect)) {
    fatal_registration_error(
        name_,
        "gradient() rejected: op has OpTrait::kHasSideEffect and is not differentiable "
        "(ARCH-062)");
  }
  gradient_ = fn;
  return *this;
}

}  // namespace frame::ops
