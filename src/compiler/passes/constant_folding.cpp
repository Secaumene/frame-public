// 内置 pass:常量折叠(constant_folding)——把仅依赖常量输入的子图预先求值。
// pass 名 = 全词文件名(见 include/frame/compiler/pipeline.h)。
//
// 机制(M8,design-reviewer 已批,决议点 D):单遍拓扑序遍历即可完成级联折叠
// ——上游节点若在本遍内被折成常量,graph.replace_all_uses 会就地把下游节点
// 的输入 Value* 改写为新常量的输出,故下游节点在【同一遍】遍历到时,其输入
// producer 已经是新常量节点,天然满足"全部输入为常量"的可折叠判定,无需
// fixpoint 循环。
// 快照下标论证:遍历对象是运行前拍下的 topological_order() 快照
// (std::vector<Node*> 拷贝);循环体内每次至多 erase_node 当前正在处理的
// 那一个节点本身,快照中其余尚未处理/已处理过的节点指针均不受影响(erase_node
// 只删除被显式传入的那个节点),故快照在整个遍历期间保持逐项有效,无需在
// erase 后额外失效性处理。
// 编译期求值语义(§3.3 契约,决议点 D):恒用 cpu 参考 kernel 求值,与图的
// 目标后端无关(ARCH-041:cpu = 数值基准);folding 内部构造的 KernelContext
// 属编译期内部求值,不代表任何真实运行时执行,故 device 固定为 cpu、
// stream 固定为 nullptr(cpu kernel 不解引用 stream,M4 既有契约)。

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <frame/compiler/pass.h>
#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/backend.h>
#include <frame/ir/graph.h>
#include <frame/ops/constant_utils.h>
#include <frame/ops/kernel_registry.h>
#include <frame/ops/op_registry.h>

namespace frame::compiler {
namespace {

// v0 常量可编码 dtype 白名单:与 include/frame/ops/constant_utils.h 精度论证
// 一致(float32/float16/bfloat16)。折叠产物最终要经 encode_tensor_to_attrs
// 编码回常量属性,输出 dtype 不在白名单内的节点直接跳过不折(留给未来 dtype
// 扩展议题)。
bool is_foldable_dtype(DTypeCode code) {
  return code == DTypeCode::kFloat32 || code == DTypeCode::kFloat16 || code == DTypeCode::kBFloat16;
}

// 判定 node 本遍是否可折叠:op != constant;schema 无 kHasSideEffect;单输出
// (v0 全部算子单输出,多输出留扩展点不折);全部输入的 producer 均为
// constant;KernelRegistry 命中 cpu kernel(未命中跳过不报错,ARCH-041 参考
// kernel 缺失不是编译错误——该算子本就还没有 cpu 实现,由 backend_lowering
// 在真正需要该算子时报错);输出 dtype 属白名单。graph_input 节点因未注册
// schema(find 返回 nullptr)天然被排除,不需要显式判空 0 输入的情形。
bool is_foldable_node(const ir::Node* node) {
  if (node->op() == ops::kConstantOpName) return false;
  const ops::OpSchema* schema = ops::OpRegistry::instance().find(node->op());
  if (schema == nullptr) return false;  // 含 graph_input:未注册,天然排除
  if (schema->has_trait(ops::OpTrait::kHasSideEffect)) return false;
  if (node->outputs().size() != 1) return false;
  for (const ir::Value* input : node->inputs()) {
    const ir::Node* producer = input->producer();
    if (producer == nullptr || producer->op() != ops::kConstantOpName) return false;
  }
  return is_foldable_dtype(node->outputs()[0].type().dtype.code());
}

// CRTP 接入:只需提供 static constexpr kName 与 run_impl(ir::Graph&)。
class ConstantFoldingPass final : public PassBase<ConstantFoldingPass> {
 public:
  static constexpr std::string_view kName = "constant_folding";

  Status run_impl(ir::Graph& graph) {
    const Result<hal::Backend*> backend = hal::BackendRegistry::instance().get(kCpuBackendName);
    if (!backend.is_ok()) {
      return Status::make(backend.status().code(),
                          "constant_folding: " + std::string(backend.status().message()));
    }
    hal::Allocator* allocator = backend.value()->allocator(cpu_device());
    if (allocator == nullptr) {
      return Status::make(ErrorCode::kInternal, "constant_folding: cpu allocator unavailable");
    }

    const std::vector<ir::Node*> snapshot = graph.topological_order();
    for (ir::Node* node : snapshot) {
      if (!is_foldable_node(node)) continue;

      const Result<ops::KernelFn> kernel =
          ops::KernelRegistry::instance().find(node->op(), kCpuBackendName);
      if (!kernel.is_ok()) continue;  // 未命中不报错,跳过(见上方判定注释)

      FRAME_RETURN_IF_ERROR(fold_node(graph, *node, kernel.value(), *allocator));
    }
    return Status::ok();
  }

 private:
  // 折叠单个节点:物化输入 -> 预分配输出 -> 直调 kernel -> 编码回常量属性 ->
  // 建新 constant 节点 -> replace_all_uses -> erase_node 旧节点(自删是
  // §3.3 后置条件要求——仅清 uses 不删,旧节点仍满足可折叠判定,违反
  // "图中无可折叠节点"的后置条件)。
  static Status fold_node(ir::Graph& graph, ir::Node& node, ops::KernelFn kernel,
                          hal::Allocator& allocator) {
    std::vector<Tensor> input_tensors;
    input_tensors.reserve(node.inputs().size());
    for (const ir::Value* input : node.inputs()) {
      const ir::TensorType& type = input->type();
      Result<Tensor> tensor = Tensor::empty(type.shape, type.dtype, cpu_device(), allocator);
      if (!tensor.is_ok()) {
        return Status::make(tensor.status().code(),
                            "constant_folding: " + std::string(tensor.status().message()));
      }
      // 输入的 producer 已由 is_foldable_node 确认为 constant 节点,直接取其
      // attrs 物化(REUSE-002:与 cpu constant kernel 共用同一份 helper)。
      FRAME_RETURN_IF_ERROR(
          ops::fill_tensor_from_constant_attrs(input->producer()->attrs(), tensor.value()));
      input_tensors.push_back(std::move(tensor.value()));
    }

    const ir::TensorType& out_type = node.outputs()[0].type();
    Result<Tensor> out_tensor =
        Tensor::empty(out_type.shape, out_type.dtype, cpu_device(), allocator);
    if (!out_tensor.is_ok()) {
      return Status::make(out_tensor.status().code(),
                          "constant_folding: " + std::string(out_tensor.status().message()));
    }
    std::vector<Tensor> output_tensors{out_tensor.value()};

    ops::KernelContext ctx{input_tensors, output_tensors, &node.attrs(), cpu_device(), nullptr};
    const Status kernel_status = kernel(ctx);
    if (!kernel_status.is_ok()) {
      // 编译期可判定的数值错误本就该在编译期暴露,不静默跳过(决议点 D)。
      return Status::make(kernel_status.code(),
                          "constant_folding: node '" + std::string(node.op()) +
                              "' kernel failed: " + std::string(kernel_status.message()));
    }

    std::unordered_map<std::string, ir::AttrValue> new_attrs;
    FRAME_RETURN_IF_ERROR(ops::encode_tensor_to_attrs(output_tensors[0], new_attrs));

    Result<ir::Node*> new_node_result =
        graph.create_node(std::string(ops::kConstantOpName), {}, {out_type});
    if (!new_node_result.is_ok()) return new_node_result.status();
    ir::Node* new_node = new_node_result.value();
    for (auto& [name, value] : new_attrs) {
      new_node->set_attr(name, std::move(value));
    }

    // replace_all_uses 触发 B-1 拓扑序重定位(新 constant 节点被 create_node
    // 追加在拓扑序尾部,需要重定位到旧节点紧前,详见 ir/graph.h 头注释)。
    FRAME_RETURN_IF_ERROR(graph.replace_all_uses(node.output(0), new_node->output(0)));
    FRAME_RETURN_IF_ERROR(graph.erase_node(&node));
    return Status::ok();
  }
};

}  // namespace

FRAME_REGISTER_PASS(ConstantFoldingPass);

}  // namespace frame::compiler
