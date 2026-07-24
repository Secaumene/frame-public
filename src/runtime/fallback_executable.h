#pragma once
// FallbackExecutable:eager 回退链①②③ 落地产物
// (docs/architecture/execution-model.md 第5章)。runtime 内部实现细节——公共面
// 只经 runtime::compile 返回 shared_ptr<hal::Executable>,不进 include/frame/
// (m10-design-brief 决议点 A)。
//
// 构造入口 build() 消费 runtime::compile 持有的**原始未融合 graph**
// (design-reviewer REVISE 闭环修订 2:标准管线的 operator_fusion 先于
// backend_lowering,working_copy 到达 backend_lowering 时可能已含
// fused_elementwise_internal 等融合节点;target 后端既无其 kernel 亦无
// decomposition,基于融合后的图回退反而会失败)。逐非 graph_input 节点解析
// 执行方案(brief 决议点 B):
//   ① KernelRegistry::find(op, target_backend) 命中 → 目标后端 eager kernel,
//     经 Backend::launch;
//   ② 未命中且 OpSchema 有 decomposition → DecomposeFn 展开微图,微图各节点
//     再按①→③解析(单层分解:微图内节点不再走②,防递归失控;微图节点
//     target 与 cpu 双缺 → 整体 kUnimplemented,消息含内外算子名);
//   ③ KernelRegistry::find(op, "cpu")(ARCH-041 保证存在,防御式查失败仍报错)。
// 每次降级(某节点的 ① 查找失败,需要 ② 或 ③ 才能解决)打一条 warn_log +
// FallbackStats::record 一次;计数发生在 build 期(编译决策一次性),run() 不
// 重复计数(execution-model.md 第5章"v0 实现口径")。
//
// v0 内存边界:现仅 cpu 后端(+ 测试 fake 后端,其 allocator 均为 host 内存)
// 已注册,①/③ 混跑不做跨设备搬运;run() 对非 cpu/目标后端的张量 device 防御式
// 硬失败(kUnimplemented,消息指向 M11)。中间张量朴素逐步分配(回退是逃生舱
// 非性能路径,REUSE-011 精神,不做 arena)。

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/executable.h>
#include <frame/hal/stream.h>
#include <frame/ir/attribute.h>
#include <frame/ops/kernel_registry.h>

namespace frame::ir {
class Graph;        // 前向声明:见 include/frame/ir/graph.h
struct TensorType;  // 前向声明:见 include/frame/ir/node.h
}  // namespace frame::ir

namespace frame::runtime {

// 见文件头注释。虚函数依据 include/frame/hal/backend.h 头部 R1∧R2∧R3 判定
// (Executable 属 HAL 白名单六类之一)。
class FallbackExecutable final : public hal::Executable {
 public:
  // 构造入口:见文件头注释。target_backend 由调用方(runtime::compile)保证
  // 已在 BackendRegistry 注册(本类不重复校验该前提,理由见 .cpp 头注释)。
  static Result<std::unique_ptr<FallbackExecutable>> build(const ir::Graph& graph,
                                                           std::string_view target_backend);

  // 一次执行整图:两侧签名校验经 hal::validate_io_signature(M10 抽取,
  // design-reviewer REVISE 闭环修订 3);逐步驱动 slot 表——① 方案经
  // Backend::launch(白名单调用点,ARCH-011 第2类:编译路径不支持时的自动
  // 回退);③ 方案直调 KernelFn(device 固定为 cpu 形态,stream=nullptr,见
  // 决议点 B)。中间张量朴素逐步分配,run() 结束前把图输出对应 slot 逐字节
  // 拷贝进调用方 outputs。
  Status run(std::span<const Tensor> inputs, std::span<Tensor> outputs,
             hal::Stream& stream) override;

  std::vector<hal::IoSpec> input_signature() const override { return input_signature_; }
  std::vector<hal::IoSpec> output_signature() const override { return output_signature_; }

  // 默认构造仅供 build() 内部经 std::make_unique 使用(CPP-060:所有权一律走
  // std::unique_ptr,禁止裸 new,故构造函数须公开);字段随后由 build() 逐一
  // 填入,构造完成前不对外可见。
  FallbackExecutable() = default;

 private:
  // 单步执行方案:kEagerLaunch 经 Backend::launch(target_backend_ + 本步
  // device_,cpu_kernel 恒 nullptr);kCpuReference 直调 cpu_kernel(不经
  // Backend::launch,device 固定为 cpu 形态)。
  enum class PlanKind : uint8_t {
    kEagerLaunch,
    kCpuReference,
  };

  // 单个执行步骤:编译期自持全部所需信息,不回指 Graph/微图(decomposition
  // 展开后的微图节点同样各自产出一个 Step,内联进 steps_,无需在 run() 期区分
  // "属于哪个原始节点")。
  struct Step {
    std::string op;
    PlanKind kind = PlanKind::kCpuReference;
    ops::KernelFn cpu_kernel = nullptr;  // 仅 kind==kCpuReference 时非空
    std::unordered_map<std::string, ir::AttrValue> attrs;
    std::vector<int32_t> input_slots;
    std::vector<int32_t> output_slots;
    std::vector<hal::IoSpec> output_specs;
  };

  // 编译期逐节点解析执行方案(brief 决议点 B ①②③),递归处理 decomposition
  // 展开的微图节点(allow_decomposition=false 时禁止再次分解,防递归失控)。
  // input_types/input_slots 按位对应 op 的输入;成功时把该 op 的输出 slot(按
  // 输出位次)写入 out_output_slots(size 恒等于 output_types.size())。
  static Status resolve_node(FallbackExecutable& executable, std::string_view op,
                             const std::unordered_map<std::string, ir::AttrValue>& attrs,
                             const std::vector<ir::TensorType>& input_types,
                             const std::vector<int32_t>& input_slots,
                             const std::vector<ir::TensorType>& output_types,
                             std::string_view target_backend, bool allow_decomposition,
                             int32_t& next_slot, std::vector<int32_t>& out_output_slots);

  // 追加一个 Step(① 或 ③ 方案共用):按 output_types 分配新 slot、登记
  // output_specs,并把新分配的 slot 写入 out_output_slots。
  void append_step(std::string op, PlanKind kind, ops::KernelFn cpu_kernel,
                   std::unordered_map<std::string, ir::AttrValue> attrs,
                   const std::vector<int32_t>& input_slots,
                   const std::vector<ir::TensorType>& output_types, int32_t& next_slot,
                   std::vector<int32_t>& out_output_slots);

  std::vector<hal::IoSpec> input_signature_;
  std::vector<hal::IoSpec> output_signature_;
  std::vector<int32_t> output_slots_;  // 按图 outputs() 序,记录各图输出对应的 slot 索引
  std::vector<Step> steps_;            // 内联展开后的执行步骤(拓扑序 + 微图展开序)
  int32_t slot_count_ = 0;
  std::string target_backend_;  // 原目标后端名(owned copy:产物可能被编译缓存长期持有)
  Device device_{};             // 原图 device(① 方案的 KernelInvocation.device)
};

}  // namespace frame::runtime
