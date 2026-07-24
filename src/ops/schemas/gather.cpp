// 内置 gather/scatter-add 算子 schema 注册桩(M22 gather,M28 scatter_add):
// gather、scatter_add(公开)+ gather_grad_internal(内部,梯度专用)。
// 设计依据:docs/plan/2026-07-19-batch4-m22-seq.md §1.5 与
// docs/plan/2026-07-23-batch9-m28-gnn.md §1.1。gather 签名:
// gather(x[N,F], indices[K]) -> out[K,F],axis 固定 0
// (embedding = gather(权重[V,E], ids[K]) 即特例);x 限 rank-2 浮点三档、
// indices 限 rank-1 且 dtype∈{int32,int64}(唯一整数张量消费者,§1.1 决议点A);
// 输出 dtype = x dtype(输入 0 约定,shape_inference.cpp:84 的"输出
// dtype==输入0 dtype"复核依赖此序,x 必须在输入 0、indices 必须在输入 1)。
// 索引值域属运行时,静态图无法静态校验张量值,cpu kernel 逐元素校验
// 0<=idx<N(见 src/backends/cpu/kernels/gather.cpp)。梯度:wrt x =
// gather_grad_internal(gy,indices)+input_shape 属性;wrt indices =
// constant(0) 整数 splat(M21 pool 零梯度 splat 先例的整数扩展,ARCH-063 全部
// 输入位均须产出梯度)。

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/graph_builder.h>
#include <frame/ops/op_registry.h>

#include "schema_math.h"

namespace {

using frame::DType;
using frame::DTypeCode;
using frame::ErrorCode;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::ops::NodeContext;

// 构造 shape 全形常量节点:同目录共享工具(铁律 5,见 schema_math.h 头注释)。
using frame::ops::schemas::make_constant_splat;
using frame::ops::schemas::static_shape_numel_fits_int64;

// scatter-add 族 schema 的文件内参数契约。
struct ScatterAddSchemaContract {
  std::string_view op_name;
  std::string_view value_role;
  std::string_view shape_attr_name;
  std::string_view output_dim_name;
  bool verify_all_dimensions;
};

// x/gy(value)侧 dtype 校验:v0 浮点三档(gather 族的 value 操作数与其余
// elementwise 系算子同一白名单,§1.5)。op_name/role 仅用于拼错误消息。
Status validate_value_dtype(std::string_view op_name, std::string_view role, DType dtype) {
  const DTypeCode code = dtype.code();
  const bool supported =
      code == DTypeCode::kFloat32 || code == DTypeCode::kFloat16 || code == DTypeCode::kBFloat16;
  if (!supported) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) + "' does not support " + std::string(role) +
                            " dtype '" + std::string(dtype.name()) +
                            "' (v0 supports float32/float16/bfloat16 only)");
  }
  return Status::ok();
}

// indices 侧 dtype 校验:int32/int64 二档(唯一整数张量消费者,§1.1 决议点A)。
Status validate_indices_dtype(std::string_view op_name, DType dtype) {
  const DTypeCode code = dtype.code();
  const bool supported = code == DTypeCode::kInt32 || code == DTypeCode::kInt64;
  if (!supported) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) +
                            "' requires indices dtype to be int32 or int64, got '" +
                            std::string(dtype.name()) + "'");
  }
  return Status::ok();
}

// gather(x[N,F], indices[K]) 的 shape 推断:x 限 rank-2 浮点三档、indices 限
// rank-1 且 dtype∈{int32,int64}。输出 [K,F],dtype=x dtype(输入 0 约定)。
Result<std::vector<Shape>> infer_gather_shape(const NodeContext& ctx) {
  if (ctx.input_types.size() != 2) {
    return Status::make(ErrorCode::kInvalidArgument, "op 'gather' expects 2 inputs, got " +
                                                         std::to_string(ctx.input_types.size()));
  }
  const frame::ir::TensorType& x = ctx.input_types[0];
  const frame::ir::TensorType& indices = ctx.input_types[1];

  if (x.shape.rank() != 2) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "op 'gather' requires x to be rank-2 [N, F], got rank " + std::to_string(x.shape.rank()));
  }
  if (indices.shape.rank() != 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'gather' requires indices to be rank-1 [K], got rank " +
                            std::to_string(indices.shape.rank()));
  }
  const Status value_status = validate_value_dtype("gather", "x", x.dtype);
  if (!value_status.is_ok()) return value_status;
  const Status indices_status = validate_indices_dtype("gather", indices.dtype);
  if (!indices_status.is_ok()) return indices_status;

  const int64_t k = indices.shape.dim(0);
  const int64_t f = x.shape.dim(1);
  return std::vector<Shape>{Shape({k, f})};
}

// scatter-add 族的共享 shape 推断核心:公共 scatter_add 与旧
// gather_grad_internal 仅以算子名、值输入角色、shape 属性名和输出首维符号
// 区分诊断;校验与输出 shape 语义保持单份。
Result<std::vector<Shape>> infer_scatter_add_shape(const NodeContext& ctx,
                                                   const ScatterAddSchemaContract& contract) {
  const std::string_view op_name = contract.op_name;
  const std::string_view value_role = contract.value_role;
  const std::string_view shape_attr_name = contract.shape_attr_name;
  const std::string_view output_dim_name = contract.output_dim_name;
  if (ctx.input_types.size() != 2) {
    return Status::make(ErrorCode::kInvalidArgument, "op '" + std::string(op_name) +
                                                         "' expects 2 inputs, got " +
                                                         std::to_string(ctx.input_types.size()));
  }
  const frame::ir::TensorType& values = ctx.input_types[0];
  const frame::ir::TensorType& indices = ctx.input_types[1];

  if (values.shape.rank() != 2) {
    return Status::make(ErrorCode::kInvalidArgument, "op '" + std::string(op_name) + "' requires " +
                                                         std::string(value_role) +
                                                         " to be rank-2 [K, F], got rank " +
                                                         std::to_string(values.shape.rank()));
  }
  if (indices.shape.rank() != 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) +
                            "' requires indices to be rank-1 [K], got rank " +
                            std::to_string(indices.shape.rank()));
  }
  const Status value_status = validate_value_dtype(op_name, value_role, values.dtype);
  if (!value_status.is_ok()) return value_status;
  const Status indices_status = validate_indices_dtype(op_name, indices.dtype);
  if (!indices_status.is_ok()) return indices_status;

  if (indices.shape.dim(0) != values.shape.dim(0)) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) + "' requires indices size to equal " +
                            std::string(value_role) +
                            "'s first dimension K=" + std::to_string(values.shape.dim(0)) +
                            ", got " + std::to_string(indices.shape.dim(0)));
  }

  const Shape* output_shape = ctx.attr<Shape>(shape_attr_name);
  if (output_shape == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument, "op '" + std::string(op_name) +
                                                         "' is missing required attribute '" +
                                                         std::string(shape_attr_name) + "'");
  }
  if (contract.verify_all_dimensions) {
    const Status shape_status = output_shape->verify();
    if (!shape_status.is_ok()) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "op '" + std::string(op_name) + "' attribute '" +
                              std::string(shape_attr_name) +
                              "' is invalid: " + std::string(shape_status.message()));
    }
  }
  if (output_shape->has_dynamic_dim()) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) + "' attribute '" +
                            std::string(shape_attr_name) + "' " + output_shape->to_string() +
                            " has a dynamic dimension, static shape required (ARCH-013)");
  }
  if (output_shape->rank() != 2) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) + "' attribute '" +
                            std::string(shape_attr_name) + "' " + output_shape->to_string() +
                            " must be rank-2 [" + std::string(output_dim_name) + ", F]");
  }
  const int64_t f = values.shape.dim(1);
  if (output_shape->dim(1) != f) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(op_name) + "' attribute '" +
                            std::string(shape_attr_name) + "' " + output_shape->to_string() +
                            " second dimension must equal " + std::string(value_role) +
                            "'s F=" + std::to_string(f));
  }

  return std::vector<Shape>{*output_shape};
}

// 旧 internal schema 的薄 wrapper:属性名和全部诊断保持 M22 原样。
Result<std::vector<Shape>> infer_gather_grad_internal_shape(const NodeContext& ctx) {
  constexpr ScatterAddSchemaContract kContract{.op_name = "gather_grad_internal",
                                               .value_role = "gy",
                                               .shape_attr_name = "input_shape",
                                               .output_dim_name = "N",
                                               .verify_all_dimensions = false};
  return infer_scatter_add_shape(ctx, kContract);
}

// 公共 scatter_add(updates[K,F],indices[K];output_shape=[V,F]) 的薄 wrapper。
Result<std::vector<Shape>> infer_public_scatter_add_shape(const NodeContext& ctx) {
  if (ctx.input_types.size() == 2) {
    for (size_t i = 0; i < ctx.input_types.size(); ++i) {
      const Status shape_status = ctx.input_types[i].shape.verify();
      if (!shape_status.is_ok()) {
        return Status::make(ErrorCode::kInvalidArgument,
                            "op 'scatter_add' input " + std::to_string(i) + " shape " +
                                ctx.input_types[i].shape.to_string() +
                                " is invalid: " + std::string(shape_status.message()));
      }
      if (!static_shape_numel_fits_int64(ctx.input_types[i].shape)) {
        return Status::make(ErrorCode::kInvalidArgument, "op 'scatter_add' input " +
                                                             std::to_string(i) + " shape " +
                                                             ctx.input_types[i].shape.to_string() +
                                                             " element count overflows int64");
      }
    }
  }
  constexpr ScatterAddSchemaContract kContract{.op_name = "scatter_add",
                                               .value_role = "updates",
                                               .shape_attr_name = "output_shape",
                                               .output_dim_name = "V",
                                               .verify_all_dimensions = true};
  Result<std::vector<Shape>> shapes = infer_scatter_add_shape(ctx, kContract);
  if (!shapes.is_ok()) return shapes.status();
  if (!static_shape_numel_fits_int64(shapes.value()[0])) {
    return Status::make(ErrorCode::kInvalidArgument, "op 'scatter_add' attribute 'output_shape' " +
                                                         shapes.value()[0].to_string() +
                                                         " element count overflows int64");
  }
  return shapes;
}

// gather(x,indices) 的梯度(§1.5):wrt x = gather_grad_internal(gy,indices) +
// input_shape 属性;wrt indices = constant(0) 整数 splat(同 indices
// shape/dtype,M21 pool 零梯度 splat 先例的整数扩展)。
Result<frame::ir::Graph> gather_gradient(const NodeContext& ctx) {
  if (ctx.input_types.size() != 2) {
    return Status::make(ErrorCode::kInvalidArgument, "op 'gather' gradient expects 2 inputs, got " +
                                                         std::to_string(ctx.input_types.size()));
  }
  frame::ir::Graph graph("gather_gradient");

  const Result<frame::ir::Value*> x_in = graph.add_graph_input(ctx.input_types[0]);
  if (!x_in.is_ok()) return x_in.status();
  const Result<frame::ir::Value*> indices_in = graph.add_graph_input(ctx.input_types[1]);
  if (!indices_in.is_ok()) return indices_in.status();

  const Result<std::vector<Shape>> y_shapes = infer_gather_shape(ctx);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const Result<frame::ir::Value*> gy_in = graph.add_graph_input(y_type);
  if (!gy_in.is_ok()) return gy_in.status();

  const frame::ops::AttrMap gx_attrs{{"input_shape", ctx.input_types[0].shape}};
  const Result<frame::ir::Node*> gx_node = frame::ops::create_node_with_inferred_types(
      graph, "gather_grad_internal", {gy_in.value(), indices_in.value()}, gx_attrs);
  if (!gx_node.is_ok()) return gx_node.status();
  const Status mark_gx = graph.mark_output(gx_node.value(), 0);
  if (!mark_gx.is_ok()) return mark_gx;

  const frame::ir::TensorType& indices_type = ctx.input_types[1];
  const Result<frame::ir::Node*> wrt_indices_node =
      make_constant_splat(graph, indices_type.shape, indices_type.dtype, indices_type.device, 0.0);
  if (!wrt_indices_node.is_ok()) return wrt_indices_node.status();
  const Status mark_wrt_indices = graph.mark_output(wrt_indices_node.value(), 0);
  if (!mark_wrt_indices.is_ok()) return mark_wrt_indices;

  return graph;
}

// scatter-add 族共享梯度核心:wrt values = gather(g,indices);wrt indices =
// constant(0) 整数 splat。公共 op 与旧 internal 仅参数化名称和 shape 属性。
Result<frame::ir::Graph> scatter_add_gradient_impl(const NodeContext& ctx,
                                                   const ScatterAddSchemaContract& contract,
                                                   std::string_view graph_name) {
  const std::string_view op_name = contract.op_name;
  if (ctx.input_types.size() != 2) {
    return Status::make(ErrorCode::kInvalidArgument, "op '" + std::string(op_name) +
                                                         "' gradient expects 2 inputs, got " +
                                                         std::to_string(ctx.input_types.size()));
  }
  frame::ir::Graph graph{std::string(graph_name)};

  const Result<frame::ir::Value*> data_in = graph.add_graph_input(ctx.input_types[0]);
  if (!data_in.is_ok()) return data_in.status();
  const Result<frame::ir::Value*> indices_in = graph.add_graph_input(ctx.input_types[1]);
  if (!indices_in.is_ok()) return indices_in.status();

  const Result<std::vector<Shape>> y_shapes = infer_scatter_add_shape(ctx, contract);
  if (!y_shapes.is_ok()) return y_shapes.status();
  frame::ir::TensorType y_type = ctx.input_types[0];
  y_type.shape = y_shapes.value()[0];
  const Result<frame::ir::Value*> y_in = graph.add_graph_input(y_type);
  if (!y_in.is_ok()) return y_in.status();
  const Result<frame::ir::Value*> g_in = graph.add_graph_input(y_type);
  if (!g_in.is_ok()) return g_in.status();

  const Result<frame::ir::Node*> wrt_data_node = frame::ops::create_node_with_inferred_types(
      graph, "gather", {g_in.value(), indices_in.value()});
  if (!wrt_data_node.is_ok()) return wrt_data_node.status();
  const Status mark_wrt_data = graph.mark_output(wrt_data_node.value(), 0);
  if (!mark_wrt_data.is_ok()) return mark_wrt_data;

  const frame::ir::TensorType& indices_type = ctx.input_types[1];
  const Result<frame::ir::Node*> wrt_indices_node =
      make_constant_splat(graph, indices_type.shape, indices_type.dtype, indices_type.device, 0.0);
  if (!wrt_indices_node.is_ok()) return wrt_indices_node.status();
  const Status mark_wrt_indices = graph.mark_output(wrt_indices_node.value(), 0);
  if (!mark_wrt_indices.is_ok()) return mark_wrt_indices;

  return graph;
}

// gather_grad_internal 自身梯度的兼容 wrapper:旧图名与诊断保持不变。
Result<frame::ir::Graph> gather_grad_internal_gradient(const NodeContext& ctx) {
  constexpr ScatterAddSchemaContract kContract{.op_name = "gather_grad_internal",
                                               .value_role = "gy",
                                               .shape_attr_name = "input_shape",
                                               .output_dim_name = "N",
                                               .verify_all_dimensions = false};
  return scatter_add_gradient_impl(ctx, kContract, "gather_grad_internal_gradient");
}

// 公共 scatter_add 的 GradientFn 薄 wrapper。
Result<frame::ir::Graph> public_scatter_add_gradient(const NodeContext& ctx) {
  constexpr ScatterAddSchemaContract kContract{.op_name = "scatter_add",
                                               .value_role = "updates",
                                               .shape_attr_name = "output_shape",
                                               .output_dim_name = "V",
                                               .verify_all_dimensions = true};
  return scatter_add_gradient_impl(ctx, kContract, "scatter_add_gradient");
}

}  // namespace

FRAME_REGISTER_OP("gather")
    .input("x", "value tensor to gather rows from, rank-2 [N, F]")
    .input("indices", "row indices, rank-1 [K], dtype int32 or int64")
    .output("out", "gathered rows: out[k, :] = x[indices[k], :], rank-2 [K, F]")
    .shape_infer(&infer_gather_shape)
    .gradient(&gather_gradient);

// 公共 scatter_add:按 indices 将 updates 行累加到静态 output_shape 对应行。
FRAME_REGISTER_OP("scatter_add")
    .input("updates", "rows to scatter-add, rank-2 [K, F]")
    .input("indices", "output row indices, rank-1 [K], dtype int32 or int64")
    .attr("output_shape", frame::ir::AttrType::kShape, /*required=*/true)
    .output("out", "scatter-added rows with duplicate indices accumulated, rank-2 [V, F]")
    .shape_infer(&infer_public_scatter_add_shape)
    .gradient(&public_scatter_add_gradient);

// gather_grad_internal(M22,gather 的梯度 internal 算子):不面向用户
// (_internal 后缀,PY-021 天然豁免),仅供 gather_gradient 内联使用;cpu
// kernel 见 src/backends/cpu/kernels/gather.cpp。
FRAME_REGISTER_OP("gather_grad_internal")
    .input("gy", "upstream gradient of gather's output, rank-2 [K, F]")
    .input("indices", "gather's original row indices, rank-1 [K]")
    .attr("input_shape", frame::ir::AttrType::kShape, /*required=*/true)
    .output("gx",
            "gradient w.r.t. x: gy scatter-added back to indices' positions (duplicate indices "
            "accumulate), rank-2 [N, F]")
    .shape_infer(&infer_gather_grad_internal_shape)
    .gradient(&gather_grad_internal_gradient);
