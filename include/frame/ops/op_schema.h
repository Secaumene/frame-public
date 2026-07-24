#pragma once
// OpSchema:算子契约(与后端无关,全局唯一)。builder 风格链式声明,注册期一次性
// 构建,运行期只读。见 docs/architecture/operator-system.md 第2/3章。

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include <frame/core/macros.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/ir/attribute.h>
#include <frame/ir/node.h>

namespace frame::ir {
// 前向声明:DecomposeFn 的返回值类型。函数指针类型的声明不要求 Graph 完整定义
// (返回类型完整性只在真正构造/读取 Result<ir::Graph> 的调用点才需要,该调用点
// 须自行 include frame/ir/graph.h);此后 Graph 不得引入破坏移动语义的成员
// (DecomposeFn 依赖 Graph 可移动,ir::Node 堆地址稳定,见 include/frame/ir/graph.h)。
class Graph;
}  // namespace frame::ir

namespace frame::ops {

// OpTrait:封闭枚举(ARCH-043)。仅允许以下四项;新增 trait 必须先修订
// docs/architecture/operator-system.md 与本文件并通过 design-reviewer。
enum class OpTrait : uint8_t {
  kElementwise,    // 逐元素:输出 shape 等于输入 shape,无跨元素依赖(fusion 候选依据)
  kFusable,        // 允许被 operator_fusion pass 合并
  kHasSideEffect,  // 有副作用:禁止被 CSE/DCE 消除或重排
  kCommutative,    // 输入可交换(CSE 归一化依据)
};

// shape 推断与分解函数的只读节点视图。
// 借用契约:attrs 指针仅在调用期间有效(借用 ir::Node 的属性表),可空 = 无属性。
struct NodeContext {
  std::string_view op;
  std::vector<ir::TensorType> input_types;  // 按位输入类型(值持有)
  const std::unordered_map<std::string, ir::AttrValue>* attrs = nullptr;

  // 按名取回强类型属性;不存在、attrs 为空或类型不符均返回 nullptr。
  template <typename T>
  const T* attr(std::string_view name) const;
};

template <typename T>
const T* NodeContext::attr(std::string_view name) const {
  if (attrs == nullptr) return nullptr;
  const auto it = attrs->find(std::string(name));
  if (it == attrs->end()) return nullptr;
  return std::get_if<T>(&it->second);
}

// shape 推断函数:函数指针(零开销)。返回各输出的 Shape;无法静态确定时返回错误。
using ShapeInferFn = Result<std::vector<Shape>> (*)(const NodeContext&);

// 分解函数:产出语义等价微图——graph_inputs 按位对应本算子输入、图输出按位对应
// 本算子输出。纯函数、不修改既有图(ARCH-021;运行时回退 M10 执行微图,编译期
// 展开由 pass 负责)。
using DecomposeFn = Result<ir::Graph> (*)(const NodeContext&);

// 梯度函数(M17,反向模式自动微分,与 DecomposeFn 同形态先例):产出梯度微图
// ——graph_inputs 按位 = [x_0..x_{n-1}, y_0..y_{m-1}, gy_0..gy_{m-1}](本算子的
// n 个前向输入、m 个前向输出、m 个输出梯度;不需要的位也须占位声明),图输出
// 按位 = [gx_0..gx_{n-1}](对每个前向输入的梯度,v0 要求全部输入位均产出梯度)。
// y/gy 位的类型在函数体内经本算子自身 shape_infer(ctx) 重算获得(gy 类型恒等
// 于对应 y);纯函数、不修改既有图(ARCH-021),产出的微图是独立合法可执行图
// (与 DecomposeFn 同纪律,具体类型、可过 verify)。完整契约见
// docs/architecture/autograd.md 第3章(ARCH-063)。
using GradientFn = Result<ir::Graph> (*)(const NodeContext&);

// 输入/输出声明:名字 + 文档串(builder input()/output() 的存储项)。
struct OpParam {
  std::string name;
  std::string doc;
};

// 属性声明:名字 + 类型 + 是否必需(builder attr() 的存储项)。
struct OpAttrSpec {
  std::string name;
  ir::AttrType type = ir::AttrType::kInt64;
  bool required = false;
};

// OpSchema:算子定义。builder 方法均返回 *this 以支持链式调用。
// 注:动态 shape v0 一律拒绝注册 —— shape 推断遇不可静态确定维度必须返回错误
// (ARCH-044,呼应 ARCH-013),禁止符号维度机制,动态 shape 支持是 ADR 议题。
class FRAME_API OpSchema {
 public:
  // builder 七方法均标 noexcept:项目禁用异常(CPP-020),这些方法只在
  // FRAME_REGISTER_OP 宏展开的链式调用中于静态初始化期被调用;标准容器扩容失败
  // (bad_alloc)经 noexcept 转为 std::terminate 即 fail-fast,与启动期注册失败的
  // 既定契约(register_op 校验失败同样 fatal)一致,如实反映"不可恢复"语义。
  // 声明一个输入(名字 + 文档串)。builder 期新增 fail-fast(M9 前置设计:
  // OpSchema 变长输入支持,note A):若本 schema 已声明变长输入组(见
  // variadic_input),视为"变长组之后又追加定长输入"违反"变长组须尾随"的硬
  // 约束,经 fatal_registration_error(REUSE-002,src/ops/
  // registration_diagnostics.h)fatal——口径对齐 register_op 启动期 fail-fast。
  OpSchema& input(std::string_view name, std::string_view doc) noexcept;
  // 声明尾随变长输入组(M9 前置设计,独立设计门,已获 design-reviewer
  // APPROVE):至多一组、须位于全部定长 input() 之后、min_count 须 >= 0,
  // 三条硬约束均在本方法内即时 fail-fast(经 fatal_registration_error,与
  // register_op 同一份实现,REUSE-002)。查询面见下方 has_variadic_inputs()/
  // min_input_count()。输出侧不做变长(v0 无需求)。
  OpSchema& variadic_input(std::string_view name, std::string_view doc, int32_t min_count) noexcept;
  // 声明一个输出(名字 + 文档串)。
  OpSchema& output(std::string_view name, std::string_view doc) noexcept;
  // 声明一个属性(名字 + 类型 + 是否必需)。
  OpSchema& attr(std::string_view name, ir::AttrType type, bool required) noexcept;
  // 标注一个 trait(封闭枚举)。
  OpSchema& trait(OpTrait trait) noexcept;
  // 设置 shape 推断函数。
  OpSchema& shape_infer(ShapeInferFn fn) noexcept;
  // 设置可选 decomposition(把本算子分解为更细算子的组合,是回退链第②跳的依据,
  // 见 docs/architecture/execution-model.md 第 5 章)。
  OpSchema& decomposition(DecomposeFn fn) noexcept;
  // 设置梯度函数(M17,ARCH-063)。带 kHasSideEffect trait 的算子不可微,调用本
  // 方法时经 registration_diagnostics fail-fast(ARCH-062)——该校验只在本方法
  // 调用的即时时刻生效,故调用方须保证先经 trait(OpTrait::kHasSideEffect) 声明
  // 该 trait、再调用 gradient(),方能被本方法捕获(与 M9 variadic_input 两条
  // 硬约束"builder 期即时校验"同一纪律,顺序敏感)。
  OpSchema& gradient(GradientFn fn) noexcept;

  // 算子名(register_op 注入,只读)。
  std::string_view name() const { return name_; }
  // 只读输入声明列表(定长部分;不含变长组,见下方 has_variadic_inputs()/
  // min_input_count())。
  const std::vector<OpParam>& inputs() const { return inputs_; }
  // 是否声明了尾随变长输入组。
  bool has_variadic_inputs() const { return has_variadic_input_; }
  // 最小输入个数 = 定长输入数 + 变长组 min_count(未声明变长组时等于
  // inputs().size(),但该场景下调用方应改用 inputs().size() 做恒等比较;本
  // 方法主要供 has_variadic_inputs() 为真时使用)。
  int32_t min_input_count() const {
    return static_cast<int32_t>(inputs_.size()) + variadic_min_count_;
  }
  // 只读输出声明列表。
  const std::vector<OpParam>& outputs() const { return outputs_; }
  // 只读属性声明列表。
  const std::vector<OpAttrSpec>& attrs() const { return attrs_; }
  // 是否标注了某 trait。
  bool has_trait(OpTrait trait) const {
    return (traits_ & static_cast<uint8_t>(1U << static_cast<uint8_t>(trait))) != 0;
  }
  // 只读 shape 推断函数(未设置为 nullptr)。
  ShapeInferFn shape_infer() const { return shape_infer_; }
  // 只读 decomposition 函数(未设置为 nullptr)。
  DecomposeFn decomposition() const { return decompose_; }
  // 只读梯度函数(未设置为 nullptr,M17)。
  GradientFn gradient() const { return gradient_; }

 private:
  // register_op 是 name_ 注入的唯一合法路径(OpSchema 自身不对外暴露 setter)。
  friend class OpRegistry;

  std::string name_;
  std::vector<OpParam> inputs_;
  bool has_variadic_input_ = false;  // 是否已声明尾随变长输入组(M9)
  OpParam variadic_input_param_;     // 变长组的名字+文档(与 inputs_ 各元素同构,不并入 inputs_)
  int32_t variadic_min_count_ = 0;   // 变长组最小个数(min_input_count() = inputs_.size() + 本值)
  std::vector<OpParam> outputs_;
  std::vector<OpAttrSpec> attrs_;
  uint8_t traits_ = 0;  // 4 位 bitmask,每 bit 对应一个 OpTrait(ARCH-043 封闭枚举)
  ShapeInferFn shape_infer_ = nullptr;
  DecomposeFn decompose_ = nullptr;
  GradientFn gradient_ = nullptr;  // M17
};

}  // namespace frame::ops
