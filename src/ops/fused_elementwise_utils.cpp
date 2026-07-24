// fused_elementwise_internal 算子共用工具的实现单元(见
// include/frame/ops/fused_elementwise_utils.h)。

#include <cstddef>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <frame/core/macros.h>
#include <frame/core/tensor.h>
#include <frame/ops/fused_elementwise_utils.h>
#include <frame/ops/op_registry.h>

namespace frame::ops {

void encode_fused_chain(const std::vector<std::string>& ops, const std::vector<int64_t>& arities,
                        std::unordered_map<std::string, ir::AttrValue>& attrs) {
  FRAME_CHECK(!ops.empty());
  FRAME_CHECK(ops.size() == arities.size());

  std::string joined_ops;
  for (size_t i = 0; i < ops.size(); ++i) {
    if (i != 0) joined_ops += ';';
    joined_ops += ops[i];
  }

  attrs["ops"] = ir::AttrValue{std::move(joined_ops)};
  attrs["arities"] = ir::AttrValue{arities};
}

Status decode_and_validate_fused_chain(const std::unordered_map<std::string, ir::AttrValue>& attrs,
                                       int64_t expected_input_count, FusedElementwiseChain& out) {
  const auto ops_it = attrs.find("ops");
  if (ops_it == attrs.end()) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(kFusedElementwiseOpName) +
                            "' is missing required attribute 'ops' (string)");
  }
  const std::string* joined_ops = std::get_if<std::string>(&ops_it->second);
  if (joined_ops == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(kFusedElementwiseOpName) +
                            "' attribute 'ops' has wrong type, expected string");
  }

  const auto arities_it = attrs.find("arities");
  if (arities_it == attrs.end()) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(kFusedElementwiseOpName) +
                            "' is missing required attribute 'arities' (int64 array)");
  }
  const std::vector<int64_t>* arities = std::get_if<std::vector<int64_t>>(&arities_it->second);
  if (arities == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(kFusedElementwiseOpName) +
                            "' attribute 'arities' has wrong type, expected int64 array");
  }

  // 按 ';' 切分 "ops"(分隔符编码,AttrType 无字符串数组,与 encode_fused_chain
  // 对称)。
  std::vector<std::string> sub_ops;
  {
    size_t start = 0;
    while (true) {
      const size_t sep = joined_ops->find(';', start);
      if (sep == std::string::npos) {
        sub_ops.push_back(joined_ops->substr(start));
        break;
      }
      sub_ops.push_back(joined_ops->substr(start, sep - start));
      start = sep + 1;
    }
  }

  if (sub_ops.size() != arities->size()) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(kFusedElementwiseOpName) + "' attribute 'ops' has " +
                            std::to_string(sub_ops.size()) +
                            " segment(s), attribute 'arities' has " +
                            std::to_string(arities->size()) + " element(s), expected equal counts");
  }
  if (sub_ops.empty()) {
    return Status::make(ErrorCode::kInvalidArgument, "op '" + std::string(kFusedElementwiseOpName) +
                                                         "' attribute 'ops' must not be empty");
  }

  // note C:sum(arities) - (段数 - 1) == expected_input_count(接线约定,见
  // 头文件声明处注释)。
  int64_t arity_sum = 0;
  for (const int64_t arity : *arities) arity_sum += arity;
  const int64_t computed_input_count = arity_sum - static_cast<int64_t>(sub_ops.size() - 1);
  if (computed_input_count != expected_input_count) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op '" + std::string(kFusedElementwiseOpName) +
                            "' arities are inconsistent with input count: sum(arities)=" +
                            std::to_string(arity_sum) +
                            ", segment count=" + std::to_string(sub_ops.size()) +
                            ", computed input count=" + std::to_string(computed_input_count) +
                            ", expected " + std::to_string(expected_input_count));
  }

  // 每个子算子已注册且带 kElementwise + kFusable trait。
  for (const std::string& sub_op : sub_ops) {
    const OpSchema* schema = OpRegistry::instance().find(sub_op);
    if (schema == nullptr) {
      return Status::make(ErrorCode::kNotFound, "op '" + std::string(kFusedElementwiseOpName) +
                                                    "' sub-op '" + sub_op + "' is not registered");
    }
    if (!schema->has_trait(OpTrait::kElementwise) || !schema->has_trait(OpTrait::kFusable)) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "op '" + std::string(kFusedElementwiseOpName) + "' sub-op '" + sub_op +
                              "' must have both kElementwise and kFusable traits");
    }
  }

  out.sub_ops = std::move(sub_ops);
  out.arities = *arities;
  return Status::ok();
}

Status execute_fused_chain(const KernelContext& ctx, hal::Allocator& allocator) {
  if (ctx.outputs.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument, "op '" + std::string(kFusedElementwiseOpName) +
                                                         "' expects 1 output, got " +
                                                         std::to_string(ctx.outputs.size()));
  }
  if (ctx.attrs == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument, "op '" + std::string(kFusedElementwiseOpName) +
                                                         "' requires non-null attrs");
  }

  // 解码(防御性再校验,note C):schema 侧已在构图/shape_inference 阶段
  // 校验过一次,此处独立再校验一次(kernel 不信任 schema 侧校验为唯一防线)。
  FusedElementwiseChain chain;
  // 非 const:允许 return 时自动移动(performance-no-automatic-move)。
  Status decode_status =
      decode_and_validate_fused_chain(*ctx.attrs, static_cast<int64_t>(ctx.inputs.size()), chain);
  if (!decode_status.is_ok()) return decode_status;

  // 接线(决议点 B 固定约定,schema 校验保证自洽):第 i 段(i>0)的第 0 输入
  // = 前一段输出;其余输入按序消费 ctx.inputs 的下一个外部输入。
  Tensor previous_output;  // 上一段输出(段 0 时未使用)
  size_t next_external_input = 0;

  for (size_t seg = 0; seg < chain.sub_ops.size(); ++seg) {
    const int64_t arity = chain.arities[seg];
    std::vector<Tensor> seg_inputs;
    seg_inputs.reserve(static_cast<size_t>(arity));
    if (seg != 0) {
      seg_inputs.push_back(previous_output);
    }
    while (seg_inputs.size() < static_cast<size_t>(arity)) {
      if (next_external_input >= ctx.inputs.size()) {
        // 理论不可达:decode_and_validate_fused_chain 已校验
        // sum(arities)-(段数-1)==ctx.inputs.size(),此处 fail-fast 而非静默
        // 越界读取。
        return Status::make(ErrorCode::kInternal,
                            "op '" + std::string(kFusedElementwiseOpName) + "' sub-op '" +
                                chain.sub_ops[seg] +
                                "' ran out of external inputs (violates arities self-consistency)");
      }
      seg_inputs.push_back(ctx.inputs[next_external_input]);
      ++next_external_input;
    }

    const Result<KernelFn> sub_kernel =
        KernelRegistry::instance().find(chain.sub_ops[seg], ctx.device.backend);
    if (!sub_kernel.is_ok()) {
      return Status::make(sub_kernel.status().code(),
                          "op '" + std::string(kFusedElementwiseOpName) +
                              "': " + std::string(sub_kernel.status().message()));
    }

    // 末段直接写入调用方预分配的 ctx.outputs[0];中间段分配临时张量(全链同
    // shape/dtype——kElementwise 保证,与 ctx.outputs[0] 一致)。
    const bool is_last_segment = (seg + 1 == chain.sub_ops.size());
    Tensor seg_output;
    if (is_last_segment) {
      seg_output = ctx.outputs[0];
    } else {
      const Result<Tensor> allocated =
          Tensor::empty(ctx.outputs[0].shape(), ctx.outputs[0].dtype(), ctx.device, allocator);
      if (!allocated.is_ok()) {
        return Status::make(allocated.status().code(),
                            "op '" + std::string(kFusedElementwiseOpName) +
                                "': " + std::string(allocated.status().message()));
      }
      seg_output = allocated.value();
    }

    std::vector<Tensor> seg_outputs{seg_output};
    // 无属性 sub-op(v0:被融合 sub-op 自身不携带 attrs,attrs 传 nullptr)。
    KernelContext seg_ctx{seg_inputs, seg_outputs, nullptr, ctx.device, ctx.stream};
    const Status seg_status = sub_kernel.value()(seg_ctx);
    if (!seg_status.is_ok()) {
      return Status::make(seg_status.code(), "op '" + std::string(kFusedElementwiseOpName) +
                                                 "': sub-op '" + chain.sub_ops[seg] +
                                                 "' failed: " + std::string(seg_status.message()));
    }

    previous_output = seg_output;
  }

  return Status::ok();
}

}  // namespace frame::ops
