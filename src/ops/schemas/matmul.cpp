// 内置矩阵乘算子 schema 注册桩(matmul)。
// shape 推断依据两输入的秩与收缩维;动态维一律拒绝注册(ARCH-044)。
// v0 仅支持 rank-2(2D-only,design-reviewer 决议,m5-design-brief 决议点 2);
// batched/广播是未来议题,需先修订 docs/architecture/operator-system.md 后才可
// 实现,本文件不做任何隐式维度扩展或动态维处理(ARCH-044 口径不变)。

#include <cstdint>
#include <string>
#include <vector>

#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/graph_builder.h>
#include <frame/ops/op_registry.h>

namespace {

// matmul 的 shape 推断:恰 2 输入、均 rank-2、同 dtype、收缩维一致
// ([m,k]×[k2,n] 要求 k==k2),输出 [m,n]。任一违例返回英文错误(消息含
// 实际值,ARCH-031 口径:不静默降级)。非逐元素、不可交换(AB≠BA),不标
// 任何 trait。
frame::Result<std::vector<frame::Shape>> infer_matmul_shape(const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'matmul' expects 2 inputs, got " + std::to_string(ctx.input_types.size()));
  }
  const frame::ir::TensorType& lhs = ctx.input_types[0];
  const frame::ir::TensorType& rhs = ctx.input_types[1];

  if (lhs.shape.rank() != 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'matmul' requires lhs to be rank-2, got rank " + std::to_string(lhs.shape.rank()));
  }
  if (rhs.shape.rank() != 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'matmul' requires rhs to be rank-2, got rank " + std::to_string(rhs.shape.rank()));
  }
  if (!(lhs.dtype == rhs.dtype)) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'matmul' requires lhs and rhs of the same dtype, got '" +
                                   std::string(lhs.dtype.name()) + "' and '" +
                                   std::string(rhs.dtype.name()) + "'");
  }

  const int64_t m = lhs.shape.dim(0);
  const int64_t k = lhs.shape.dim(1);
  const int64_t k2 = rhs.shape.dim(0);
  const int64_t n = rhs.shape.dim(1);
  if (k != k2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'matmul' contraction dimension mismatch: lhs " + lhs.shape.to_string() + " has k=" +
            std::to_string(k) + ", rhs " + rhs.shape.to_string() + " has k=" + std::to_string(k2) +
            " (lhs's last dimension must equal rhs's first "
            "dimension)");
  }

  return std::vector<frame::Shape>{frame::Shape({m, n})};
}

// matmul_grad_lhs_internal(gy, b) 的 shape 推断(M17,matmul 的梯度 internal
// 算子之一):ga = gy·bᵀ。gy [m,n]、b [k,n](b 是原 matmul 的 rhs 操作数,原样
// 传入未转置——kernel 内直接按转置索引读取,不物化 transpose,见 cpu kernel
// 头注释);输出 [m,k]。与 infer_matmul_grad_rhs_internal_shape 结构相似但
// 收缩维/输出维在各自输入中的位置互不相同(前者收缩维是双方的 dim1,后者是
// 双方的 dim0),强行合并需要额外的"取维位置"参数化,增加的间接层不足以抵销
// 两个各一处调用点的短函数的合并收益,故各自独立实现(与 infer_matmul_shape
// 本身不与任何函数合并同一先例)。
frame::Result<std::vector<frame::Shape>> infer_matmul_grad_lhs_internal_shape(
    const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'matmul_grad_lhs_internal' expects 2 inputs, got " +
                                   std::to_string(ctx.input_types.size()));
  }
  const frame::ir::TensorType& gy = ctx.input_types[0];
  const frame::ir::TensorType& b = ctx.input_types[1];

  if (gy.shape.rank() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'matmul_grad_lhs_internal' requires gy to be rank-2, got rank " +
                                   std::to_string(gy.shape.rank()));
  }
  if (b.shape.rank() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'matmul_grad_lhs_internal' requires b to be rank-2, got rank " +
                                   std::to_string(b.shape.rank()));
  }
  if (!(gy.dtype == b.dtype)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'matmul_grad_lhs_internal' requires gy and b of the same dtype, got '" +
            std::string(gy.dtype.name()) + "' and '" + std::string(b.dtype.name()) + "'");
  }

  const int64_t m = gy.shape.dim(0);
  const int64_t n = gy.shape.dim(1);
  const int64_t k = b.shape.dim(0);
  const int64_t n2 = b.shape.dim(1);
  if (n != n2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'matmul_grad_lhs_internal' contraction dimension mismatch: gy " +
                                   gy.shape.to_string() + " has n=" + std::to_string(n) + ", b " +
                                   b.shape.to_string() + " has n=" + std::to_string(n2) +
                                   " (gy's second dimension must equal b's second dimension)");
  }

  return std::vector<frame::Shape>{frame::Shape({m, k})};
}

// matmul_grad_rhs_internal(a, gy) 的 shape 推断(M17,matmul 的梯度 internal
// 算子之二):gb = aᵀ·gy。a [m,k]、gy [m,n](收缩维是双方的 dim0);输出
// [k,n]。与上一函数的非合并理由同上。
frame::Result<std::vector<frame::Shape>> infer_matmul_grad_rhs_internal_shape(
    const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'matmul_grad_rhs_internal' expects 2 inputs, got " +
                                   std::to_string(ctx.input_types.size()));
  }
  const frame::ir::TensorType& a = ctx.input_types[0];
  const frame::ir::TensorType& gy = ctx.input_types[1];

  if (a.shape.rank() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'matmul_grad_rhs_internal' requires a to be rank-2, got rank " +
                                   std::to_string(a.shape.rank()));
  }
  if (gy.shape.rank() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'matmul_grad_rhs_internal' requires gy to be rank-2, got rank " +
                                   std::to_string(gy.shape.rank()));
  }
  if (!(a.dtype == gy.dtype)) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'matmul_grad_rhs_internal' requires a and gy of the same dtype, got '" +
            std::string(a.dtype.name()) + "' and '" + std::string(gy.dtype.name()) + "'");
  }

  const int64_t m = a.shape.dim(0);
  const int64_t k = a.shape.dim(1);
  const int64_t m2 = gy.shape.dim(0);
  const int64_t n = gy.shape.dim(1);
  if (m != m2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'matmul_grad_rhs_internal' contraction dimension mismatch: a " +
                                   a.shape.to_string() + " has m=" + std::to_string(m) + ", gy " +
                                   gy.shape.to_string() + " has m=" + std::to_string(m2) +
                                   " (a's first dimension must equal gy's first dimension)");
  }

  return std::vector<frame::Shape>{frame::Shape({k, n})};
}

// matmul(a,b) 的梯度(M17,§4 清单):ga=matmul_grad_lhs_internal(gy,b)=gy·bᵀ、
// gb=matmul_grad_rhs_internal(a,gy)=aᵀ·gy。
frame::Result<frame::ir::Graph> matmul_gradient(const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 2) {
    return frame::Status::make(
        frame::ErrorCode::kInvalidArgument,
        "op 'matmul' gradient expects 2 inputs, got " + std::to_string(ctx.input_types.size()));
  }
  frame::ir::Graph graph("matmul_gradient");

  const frame::Result<frame::ir::Value*> a_in = graph.add_graph_input(ctx.input_types[0]);
  if (!a_in.is_ok()) return a_in.status();
  const frame::Result<frame::ir::Value*> b_in = graph.add_graph_input(ctx.input_types[1]);
  if (!b_in.is_ok()) return b_in.status();

  const frame::Result<std::vector<frame::Shape>> y_shapes = infer_matmul_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const frame::Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const frame::Result<frame::ir::Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();

  const frame::Result<frame::ir::Node*> ga_node = frame::ops::create_node_with_inferred_types(
      graph, "matmul_grad_lhs_internal", {gy_in.value(), b_in.value()});
  if (!ga_node.is_ok()) return ga_node.status();
  const frame::Result<frame::ir::Node*> gb_node = frame::ops::create_node_with_inferred_types(
      graph, "matmul_grad_rhs_internal", {a_in.value(), gy_in.value()});
  if (!gb_node.is_ok()) return gb_node.status();

  const frame::Status mark_ga = graph.mark_output(ga_node.value(), 0);
  if (!mark_ga.is_ok()) return mark_ga;
  const frame::Status mark_gb = graph.mark_output(gb_node.value(), 0);
  if (!mark_gb.is_ok()) return mark_gb;

  return graph;
}

// matmul_grad_lhs_internal(gy,b)=gy·b^T 的梯度(M26,ARCH-068):
// g_gy=gga·b;g_b=gga^T·gy。第二项直接复用 matmul_grad_rhs_internal,
// 不物化 transpose。
frame::Result<frame::ir::Graph> matmul_grad_lhs_internal_gradient(
    const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'matmul_grad_lhs_internal' gradient expects 2 inputs, got " +
                                   std::to_string(ctx.input_types.size()));
  }
  const frame::Result<std::vector<frame::Shape>> output_shapes =
      infer_matmul_grad_lhs_internal_shape(ctx);
  if (!output_shapes.is_ok()) return output_shapes.status();

  frame::ir::Graph graph("matmul_grad_lhs_internal_gradient");
  const frame::Result<frame::ir::Value*> gy_in = graph.add_graph_input(ctx.input_types[0]);
  if (!gy_in.is_ok()) return gy_in.status();
  const frame::Result<frame::ir::Value*> b_in = graph.add_graph_input(ctx.input_types[1]);
  if (!b_in.is_ok()) return b_in.status();
  frame::ir::TensorType ga_type = ctx.input_types[0];
  ga_type.shape = output_shapes.value()[0];
  const frame::Result<frame::ir::Value*> ga_in = graph.add_graph_input(ga_type);
  if (!ga_in.is_ok()) return ga_in.status();
  const frame::Result<frame::ir::Value*> gga_in = graph.add_graph_input(ga_type);
  if (!gga_in.is_ok()) return gga_in.status();

  const frame::Result<frame::ir::Node*> g_gy_node =
      frame::ops::create_node_with_inferred_types(graph, "matmul", {gga_in.value(), b_in.value()});
  if (!g_gy_node.is_ok()) return g_gy_node.status();
  const frame::Result<frame::ir::Node*> g_b_node = frame::ops::create_node_with_inferred_types(
      graph, "matmul_grad_rhs_internal", {gga_in.value(), gy_in.value()});
  if (!g_b_node.is_ok()) return g_b_node.status();

  const frame::Status mark_g_gy = graph.mark_output(g_gy_node.value(), 0);
  if (!mark_g_gy.is_ok()) return mark_g_gy;
  const frame::Status mark_g_b = graph.mark_output(g_b_node.value(), 0);
  if (!mark_g_b.is_ok()) return mark_g_b;
  return graph;
}

// matmul_grad_rhs_internal(a,gy)=a^T·gy 的梯度(M26,ARCH-068):
// g_a=gy·ggb^T;g_gy=a·ggb。第一项复用 matmul_grad_lhs_internal。
frame::Result<frame::ir::Graph> matmul_grad_rhs_internal_gradient(
    const frame::ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 2) {
    return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                               "op 'matmul_grad_rhs_internal' gradient expects 2 inputs, got " +
                                   std::to_string(ctx.input_types.size()));
  }
  const frame::Result<std::vector<frame::Shape>> output_shapes =
      infer_matmul_grad_rhs_internal_shape(ctx);
  if (!output_shapes.is_ok()) return output_shapes.status();

  frame::ir::Graph graph("matmul_grad_rhs_internal_gradient");
  const frame::Result<frame::ir::Value*> a_in = graph.add_graph_input(ctx.input_types[0]);
  if (!a_in.is_ok()) return a_in.status();
  const frame::Result<frame::ir::Value*> gy_in = graph.add_graph_input(ctx.input_types[1]);
  if (!gy_in.is_ok()) return gy_in.status();
  frame::ir::TensorType gb_type = ctx.input_types[0];
  gb_type.shape = output_shapes.value()[0];
  const frame::Result<frame::ir::Value*> gb_in = graph.add_graph_input(gb_type);
  if (!gb_in.is_ok()) return gb_in.status();
  const frame::Result<frame::ir::Value*> ggb_in = graph.add_graph_input(gb_type);
  if (!ggb_in.is_ok()) return ggb_in.status();

  const frame::Result<frame::ir::Node*> g_a_node = frame::ops::create_node_with_inferred_types(
      graph, "matmul_grad_lhs_internal", {gy_in.value(), ggb_in.value()});
  if (!g_a_node.is_ok()) return g_a_node.status();
  const frame::Result<frame::ir::Node*> g_gy_node =
      frame::ops::create_node_with_inferred_types(graph, "matmul", {a_in.value(), ggb_in.value()});
  if (!g_gy_node.is_ok()) return g_gy_node.status();

  const frame::Status mark_g_a = graph.mark_output(g_a_node.value(), 0);
  if (!mark_g_a.is_ok()) return mark_g_a;
  const frame::Status mark_g_gy = graph.mark_output(g_gy_node.value(), 0);
  if (!mark_g_gy.is_ok()) return mark_g_gy;
  return graph;
}

}  // namespace

FRAME_REGISTER_OP("matmul")
    .input("lhs", "left operand, rank-2 [m, k]")
    .input("rhs", "right operand, rank-2 [k, n]")
    .output("out", "matrix product of lhs and rhs, rank-2 [m, n]")
    .shape_infer(&infer_matmul_shape)
    .gradient(&matmul_gradient);

// matmul_grad_{lhs,rhs}_internal(M17,matmul 的梯度 internal 算子):不面向
// 用户(_internal 后缀,PY-021 天然豁免),仅供 matmul_gradient 内联使用;cpu
// kernel 见 src/backends/cpu/kernels/matmul.cpp。
FRAME_REGISTER_OP("matmul_grad_lhs_internal")
    .input("gy", "upstream gradient of matmul's output, rank-2 [m, n]")
    .input("b", "matmul's original rhs operand, rank-2 [k, n]")
    .output("ga", "gradient w.r.t. lhs: gy * b^T, rank-2 [m, k]")
    .shape_infer(&infer_matmul_grad_lhs_internal_shape)
    .gradient(&matmul_grad_lhs_internal_gradient);

FRAME_REGISTER_OP("matmul_grad_rhs_internal")
    .input("a", "matmul's original lhs operand, rank-2 [m, k]")
    .input("gy", "upstream gradient of matmul's output, rank-2 [m, n]")
    .output("gb", "gradient w.r.t. rhs: a^T * gy, rank-2 [k, n]")
    .shape_infer(&infer_matmul_grad_rhs_internal_shape)
    .gradient(&matmul_grad_rhs_internal_gradient);
