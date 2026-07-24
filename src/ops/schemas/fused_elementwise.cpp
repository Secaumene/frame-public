// 内置 fused_elementwise_internal 算子 schema 注册桩(M9,决议点 B 覆盖版):
// operator_fusion pass 的产物表示(src/compiler/passes/operator_fusion.cpp)
// ——不新增 IR 对象类型,以普通注册算子表示一条线性化的逐元素子算子链。
// 变长输入(min_count=1,前置设计:OpSchema 变长输入支持)、单输出、trait 仅
// kElementwise(不标 kFusable——v0 不做融合节点再融合,避免嵌套编码)。
// 非用户构图算子(PY-021 判定见 include/frame/ops/fused_elementwise_utils.h
// 头注释)。cpu kernel 见 src/backends/cpu/kernels/fused_elementwise.cpp。

#include <cstdint>
#include <string>
#include <vector>

#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/ir/node.h>
#include <frame/ops/fused_elementwise_utils.h>
#include <frame/ops/op_registry.h>

namespace {

// shape 推断:输出 = 第 0 输入类型(全链均为逐元素算子,shape 全程恒等,
// 决议点 B);另调 decode_and_validate_fused_chain 复核 attrs 自洽性
// (期望输入数 = 实际输入数)。
frame::Result<std::vector<frame::Shape>> infer_fused_elementwise_shape(
    const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.empty()) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'fused_elementwise_internal' expects at least 1 input, got 0");
  }
  if (ctx.attrs == nullptr) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'fused_elementwise_internal' requires non-null attrs");
  }

  frame::ops::FusedElementwiseChain chain;
  const frame::Status decode_status = frame::ops::decode_and_validate_fused_chain(
      *ctx.attrs, static_cast<int64_t>(ctx.input_types.size()), chain);
  if (!decode_status.is_ok()) return decode_status;

  return std::vector<frame::Shape>{ctx.input_types[0].shape};
}

}  // namespace

FRAME_REGISTER_OP(frame::ops::kFusedElementwiseOpName)
    .variadic_input("inputs", "external inputs of the fused elementwise chain", /*min_count=*/1)
    .output("out", "output of the fused elementwise chain")
    .attr("ops", frame::ir::AttrType::kString, /*required=*/true)
    .attr("arities", frame::ir::AttrType::kInt64Array, /*required=*/true)
    .trait(frame::ops::OpTrait::kElementwise)
    .shape_infer(&infer_fused_elementwise_shape);
