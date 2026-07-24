// 内置归约算子 schema 注册桩(sum/mean/max 等)。
// 归约含 axes/keepdims 属性,shape 推断据此消维;有跨元素依赖,非逐元素
// (无 trait 标注)。sum 已在本文件下方注册;其余算子见下方待办标注。

// TODO(FRAME-IMPL): mean/max 待落地,属未来批次,不在 M5 首批范围(M5 首批
//   归约算子仅 sum,见 PLAN.md「M5 内置算子 v0 批次」行)。参考:
//   docs/architecture/operator-system.md 第2/3章;include/frame/ops/op_schema.h。
//   完成判据:OpRegistry::find("mean")/("max") 落地时均可取到 schema(含
//   shape_infer),tests/cpp/ops/ 用例通过。

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/ir/attribute.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/graph_builder.h>
#include <frame/ops/op_registry.h>

namespace {

// axes 校验,供 infer_sum_shape 与 infer_sum_grad_internal_shape 共用
// (REUSE-002:M17 起两处调用点,超过 1 个调用点即按仓内既有先例——如
// src/ops/schemas/elementwise.cpp::infer_binary_elementwise_shape——收敛为
// 共用函数,不再各自内联复制)。axes 语义:空数组 = 全维归约;非空须满足
// ∀a∈[0, rank) 且无重复,三类违例(负值/越界/重复)各自返回英文错误(消息
// 含违例值、op_name 供调用方拼错误消息前缀)。命中的轴写入 reduced[axis]=
// true。
//
// 拒绝负索引的独立论证(不借用 ARCH-044 编号——ARCH-044 约束的是"shape 推断
// 遇不可静态确定维度",这里是另一类问题:属性值本身不合法):v0 显式优于
// 隐式。若允许负索引隐式归一化(如 -1 表示最后一维),等价于在 shape 推断内
// 新增一套与 rank 相关的隐式换算规则,读者必须先心算 rank 才能确定 axes
// 实际指向哪一维,且换算规则本身(是否支持 -1、-2 ...)又是一处需要额外
// 文档化的隐藏状态。拒绝负值把该决策显式下推给调用方(构图侧必须自行按
// rank 换算为非负索引后再写入属性),shape 推断保持"看到什么就是什么"的
// 单一语义,不引入维度相关的运行期换算逻辑。
frame::Status validate_reduction_axes(std::string_view op_name, int64_t rank,
                                      const std::vector<int64_t>& axes,
                                      std::vector<bool>& reduced) {
  reduced.assign(static_cast<size_t>(rank), false);
  if (axes.empty()) {
    // 空数组 = 全维归约(design-reviewer 决议,m5-design-brief 决议点 3)。
    reduced.assign(static_cast<size_t>(rank), true);
    return frame::Status::ok();
  }
  for (int64_t axis : axes) {
    if (axis < 0) {
      return frame::Status::make(
          frame::ErrorCode::kInvalidArgument,
          "op '" + std::string(op_name) + "' axes entry " + std::to_string(axis) +
              " is negative; v0 requires 0 <= axis < rank and does not normalize negative "
              "indices");
    }
    if (axis >= rank) {
      return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                                 "op '" + std::string(op_name) + "' axes entry " +
                                     std::to_string(axis) + " is out of range for rank " +
                                     std::to_string(rank) + " (must satisfy 0 <= axis < rank)");
    }
    if (reduced[static_cast<size_t>(axis)]) {
      return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                                 "op '" + std::string(op_name) + "' axes entry " +
                                     std::to_string(axis) + " is duplicated");
    }
    reduced[static_cast<size_t>(axis)] = true;
  }
  return frame::Status::ok();
}

// sum 的 shape 推断:1 输入 + 必填 axes(kInt64Array)+ 可选 keepdims(kBool,
// 缺省 false,design-reviewer 决议 m5-design-brief 决议点 3)。
frame::Result<std::vector<frame::Shape>> infer_sum_shape(const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sum' expects 1 input, got " + std::to_string(ctx.input_types.size()));
  }
  const frame::ir::TensorType& x = ctx.input_types[0];
  const int64_t rank = x.shape.rank();

  const std::vector<int64_t>* axes = ctx.attr<std::vector<int64_t>>("axes");
  if (axes == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'sum' is missing required attribute 'axes' (int64 array)");
  }
  bool keepdims = false;
  const bool* keepdims_ptr = ctx.attr<bool>("keepdims");
  if (keepdims_ptr != nullptr) {
    keepdims = *keepdims_ptr;
  }

  std::vector<bool> reduced;
  const frame::Status axes_status = validate_reduction_axes("sum", rank, *axes, reduced);
  if (!axes_status.is_ok()) return axes_status;

  std::vector<int64_t> out_dims;
  out_dims.reserve(x.shape.dims().size());
  for (size_t d = 0; d < x.shape.dims().size(); ++d) {
    if (reduced[d]) {
      if (keepdims) out_dims.push_back(1);
    } else {
      out_dims.push_back(x.shape.dims()[d]);
    }
  }
  return std::vector<frame::Shape>{frame::Shape(out_dims)};
}

// 由 sum_grad_internal 的 gy shape 恢复原 sum 的 keepdims 取值。attrs 只
// 保存 input_shape/axes(M17 既有序列化契约),M26 不新增属性,而是把两种
// 合法输出 shape 都精确构造后匹配;不匹配即 fail-loud。
frame::Result<bool> infer_sum_grad_keepdims(const frame::ir::TensorType& gy,
                                            const frame::Shape& input_shape,
                                            const std::vector<int64_t>& axes) {
  std::vector<bool> reduced;
  const frame::Status axes_status =
      validate_reduction_axes("sum_grad_internal", input_shape.rank(), axes, reduced);
  if (!axes_status.is_ok()) return axes_status;

  std::vector<int64_t> squeezed_dims;
  std::vector<int64_t> kept_dims;
  squeezed_dims.reserve(input_shape.dims().size());
  kept_dims.reserve(input_shape.dims().size());
  for (size_t d = 0; d < input_shape.dims().size(); ++d) {
    if (reduced[d]) {
      kept_dims.push_back(1);
    } else {
      squeezed_dims.push_back(input_shape.dims()[d]);
      kept_dims.push_back(input_shape.dims()[d]);
    }
  }

  const frame::Shape squeezed_shape(std::move(squeezed_dims));
  const frame::Shape kept_shape(std::move(kept_dims));
  const bool matches_squeezed = gy.shape == squeezed_shape;
  const bool matches_kept = gy.shape == kept_shape;
  if (!matches_squeezed && !matches_kept) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sum_grad_internal' requires gy shape to match sum(input_shape, axes) with "
        "keepdims=false or true, got " +
            gy.shape.to_string() + ", expected " + squeezed_shape.to_string() + " or " +
            kept_shape.to_string());
  }
  // rank-0 输入等退化情形可能两者同形;选择 false 不改变 shape 或数学语义。
  return matches_kept && !matches_squeezed;
}

// sum_grad_internal(gy) 的 shape 推断(M17,sum 的梯度 internal 算子):恰
// 1 输入(gy);属性 input_shape(kShape,sum 原节点输入 x 的静态 shape)+
// axes(kInt64Array,sum 原节点的 axes 属性原样转发)。输出 shape = input_shape
// (即 x 的原 shape;gy 沿归约轴复制展开回该 shape 是 kernel 层语义,不影响
// shape 推断本身——kernel 侧另需按 gy 的实际 rank 判定 keepdims 是否生效,
// 见 cpu kernel 头注释)。axes 仍按 validate_reduction_axes 校验(与 sum 的
// 校验规则一致,REUSE-002 共用)。
frame::Result<std::vector<frame::Shape>> infer_sum_grad_internal_shape(
    const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sum_grad_internal' expects 1 input, got " + std::to_string(ctx.input_types.size()));
  }
  const frame::Shape* input_shape = ctx.attr<frame::Shape>("input_shape");
  if (input_shape == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sum_grad_internal' is missing required attribute 'input_shape'");
  }
  if (input_shape->has_dynamic_dim()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'sum_grad_internal' attribute 'input_shape' " +
                                   input_shape->to_string() +
                                   " has a dynamic dimension, static shape required (ARCH-013)");
  }
  const std::vector<int64_t>* axes = ctx.attr<std::vector<int64_t>>("axes");
  if (axes == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sum_grad_internal' is missing required attribute 'axes' (int64 array)");
  }

  const frame::Result<bool> keepdims =
      infer_sum_grad_keepdims(ctx.input_types[0], *input_shape, *axes);
  if (!keepdims.is_ok()) return keepdims.status();

  return std::vector<frame::Shape>{*input_shape};
}

// sum(x,axes) 的梯度(M17,§4 清单):gx=sum_grad_internal(gy)(沿归约轴复制
// 展开回输入 shape;input_shape 经 kShape attr、axes 经 kInt64Array attr 携带
// ——gy 已丢失被归约维,仅凭输入 shape 不足以消除同尺寸维歧义,axes 是原
// sum 节点自身的属性,ctx.attrs 天然携带,原样转发)。
frame::Result<frame::ir::Graph> sum_gradient(const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sum' gradient expects 1 input, got " + std::to_string(ctx.input_types.size()));
  }
  const std::vector<int64_t>* axes = ctx.attr<std::vector<int64_t>>("axes");
  if (axes == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'sum' gradient is missing required attribute 'axes' (int64 "
                               "array) on the original sum node");
  }

  frame::ir::Graph graph("sum_gradient");

  const frame::Result<frame::ir::Value*> x_in = graph.add_graph_input(ctx.input_types[0]);
  if (!x_in.is_ok()) return x_in.status();

  const frame::Result<std::vector<frame::Shape>> y_shapes = infer_sum_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const frame::Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const frame::Result<frame::ir::Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();

  const frame::ops::AttrMap sum_grad_attrs{
      {"input_shape", ctx.input_types[0].shape},
      {"axes", *axes},
  };
  const frame::Result<frame::ir::Node*> gx_node = frame::ops::create_node_with_inferred_types(
      graph, "sum_grad_internal", {gy_in.value()}, sum_grad_attrs);
  if (!gx_node.is_ok()) return gx_node.status();

  const frame::Status mark_gx = graph.mark_output(gx_node.value(), 0);
  if (!mark_gx.is_ok()) return mark_gx;

  return graph;
}

// sum_grad_internal(gy) 的梯度(M26,ARCH-068):对展开结果 gx 的上游梯度
// ggx 沿原归约轴求和,恢复为 gy 的 shape。keepdims 必须从 gy/input_shape/
// axes 的合法 shape 精确恢复,不能依赖 sum 的默认值。
frame::Result<frame::ir::Graph> sum_grad_internal_gradient(const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'sum_grad_internal' gradient expects 1 input, got " +
                                   std::to_string(ctx.input_types.size()));
  }
  const frame::Shape* input_shape = ctx.attr<frame::Shape>("input_shape");
  const std::vector<int64_t>* axes = ctx.attr<std::vector<int64_t>>("axes");
  if (input_shape == nullptr || axes == nullptr) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'sum_grad_internal' gradient requires 'input_shape' and 'axes' attributes");
  }
  const frame::Result<bool> keepdims =
      infer_sum_grad_keepdims(ctx.input_types[0], *input_shape, *axes);
  if (!keepdims.is_ok()) return keepdims.status();

  frame::ir::Graph graph("sum_grad_internal_gradient");
  const frame::Result<frame::ir::Value*> gy_in = graph.add_graph_input(ctx.input_types[0]);
  if (!gy_in.is_ok()) return gy_in.status();
  frame::ir::TensorType gx_type = ctx.input_types[0];
  gx_type.shape = *input_shape;
  const frame::Result<frame::ir::Value*> gx_in = graph.add_graph_input(gx_type);
  if (!gx_in.is_ok()) return gx_in.status();
  const frame::Result<frame::ir::Value*> ggx_in = graph.add_graph_input(gx_type);
  if (!ggx_in.is_ok()) return ggx_in.status();

  const frame::ops::AttrMap sum_attrs{{"axes", *axes}, {"keepdims", keepdims.value()}};
  const frame::Result<frame::ir::Node*> ggy_node =
      frame::ops::create_node_with_inferred_types(graph, "sum", {ggx_in.value()}, sum_attrs);
  if (!ggy_node.is_ok()) return ggy_node.status();
  if (!(ggy_node.value()->output(0)->type().shape == ctx.input_types[0].shape)) {
    return frame::Status::make(
        frame::ErrorCode::kInternal,
        "op 'sum_grad_internal' gradient restored an output shape different from gy");
  }
  const frame::Status mark_status = graph.mark_output(ggy_node.value(), 0);
  if (!mark_status.is_ok()) return mark_status;
  return graph;
}

}  // namespace

FRAME_REGISTER_OP("sum")
    .input("x", "input tensor")
    .attr("axes", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .attr("keepdims", frame::ir::AttrType::kBool, /*required=*/false)
    .output("out", "reduced sum of x along axes")
    .shape_infer(&infer_sum_shape)
    .gradient(&sum_gradient);

// sum_grad_internal(M17,sum 的梯度 internal 算子):不面向用户(_internal
// 后缀,PY-021 天然豁免),仅供 sum_gradient 内联使用;cpu kernel 见
// src/backends/cpu/kernels/reduction.cpp。
FRAME_REGISTER_OP("sum_grad_internal")
    .input("gy", "upstream gradient of sum's output")
    .attr("input_shape", frame::ir::AttrType::kShape, /*required=*/true)
    .attr("axes", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .output("gx", "gradient w.r.t. x: gy broadcast back to input_shape along axes")
    .shape_infer(&infer_sum_grad_internal_shape)
    .gradient(&sum_grad_internal_gradient);
