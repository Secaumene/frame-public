#pragma once
// fused_elementwise_internal 算子共用工具(M9,决议点 B 覆盖版):
// operator_fusion pass 的编码(构造融合节点 attrs)与 cpu kernel 的解码
// (还原子算子链并组合调用)共用单份实现(REUSE-002,与 M8 constant_utils
// 同模式)。见 docs/architecture/compiler-passes.md §3.7、
// docs/architecture/operator-system.md。

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <frame/core/macros.h>
#include <frame/core/status.h>
#include <frame/ir/attribute.h>
#include <frame/ops/kernel_registry.h>

namespace frame::hal {
class Allocator;  // 前向声明:ops 公共头不 include hal 头(同 kernel_registry.h
                  // 对 hal::Stream 的先例,M1 依赖倒置纪律)
}  // namespace frame::hal

namespace frame::ops {

// fused_elementwise_internal 算子注册名(pass 产物,非用户构图算子;PY-021
// 判定"面向用户"经 "_internal" 后缀天然豁免,零新规则,决议点 B)。
inline constexpr std::string_view kFusedElementwiseOpName = "fused_elementwise_internal";

// 解码并自洽校验后的链结构:sub_ops 按融合顺序(第 0 段最先执行)排列的子
// 算子名;arities 按位对应各子算子的输入个数(与 sub_ops 等长)。
struct FusedElementwiseChain {
  std::vector<std::string> sub_ops;
  std::vector<int64_t> arities;
};

// 把 ops 名序列(融合顺序)与各自的输入个数(arities)编码进 attrs(覆盖式
// 写入 "ops"/"arities" 两键,已存在的同名键会被替换,其余键不受影响,同
// constant_utils.h::encode_tensor_to_attrs 的覆盖写惯例):ops 以 ';' 连接
// 存为 kString(AttrType 无字符串数组,分隔符编码);arities 存为
// kInt64Array。调用方保证 ops 非空且 ops.size() == arities.size()
// (FRAME_CHECK,构造期不变量,不经 Status——违例属调用方逻辑错误而非用户
// 输入错误)。
FRAME_API void encode_fused_chain(const std::vector<std::string>& ops,
                                  const std::vector<int64_t>& arities,
                                  std::unordered_map<std::string, ir::AttrValue>& attrs);

// encode_fused_chain 的逆向 + 自洽校验(note C):
//   ①"ops"/"arities" 两键存在且类型正确;
//   ②按 ';' 切分 "ops" 得到的段数与 "arities" 元素数一致;
//   ③sum(arities) - (段数 - 1) == expected_input_count(接线约定下,第 i>0
//     段的第 0 输入复用前段输出,故融合节点的外部输入总数比 sum(arities) 少
//     "段数-1";expected_input_count 由调用方传入——schema 侧传实际输入数,
//     kernel 侧传 ctx.inputs.size());
//   ④每个子算子已在 OpRegistry 注册且同时带 kElementwise + kFusable trait。
// 任一违例返回英文错误(消息含 op 名/实际值)。
FRAME_API Status
decode_and_validate_fused_chain(const std::unordered_map<std::string, ir::AttrValue>& attrs,
                                int64_t expected_input_count, FusedElementwiseChain& out);

// fused_elementwise_internal 的后端无关执行体(M11,design-reviewer REVISE
// 闭环裁决修订⑥上提):此前各后端各自实现的"组合调用"现体(cpu 首发,见
// src/backends/cpu/kernels/fused_elementwise.cpp 历史版本)迁入本函数——解析
// attrs 还原子算子链后,逐子算子经 KernelRegistry::find(sub_op,
// ctx.device.backend) 取该后端既有 kernel 直接调用,数值与未融合严格同源。
// allocator 由调用方注入(而非本函数内查 BackendRegistry 取当前设备的
// allocator)——ops 保持零 hal 依赖(仅前向声明 hal::Allocator),各后端 kernel
// wrapper 自行解析并传入自身 allocator(决议点 C 建议①)。已知代价:中间段
// (非首、非末段)产生的临时张量不做 arena/复用规划,融合执行是机制验证而非
// 内存优化(与 cpu 版历史注释同一立场)。ctx 借用契约同 KernelContext(仅在
// 调用期间有效);ctx 声明为 const 引用,但 ctx.outputs 是 std::span<Tensor>
// 视图,元素可写(span 的 const 只约束能否重新绑定该视图本身)。
FRAME_API Status execute_fused_chain(const KernelContext& ctx, hal::Allocator& allocator);

}  // namespace frame::ops
