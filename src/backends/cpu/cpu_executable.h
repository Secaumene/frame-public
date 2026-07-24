#pragma once
// CpuExecutable:CPU 参考后端的整图编译产物(后端专有,铁律 #3 合规,不进
// include/)。逐 kernel 拼装(m7-design-brief 决议点 3):编译期消费共享组件
// ops::build_executable_plan(M11,include/frame/ops/executable_plan.h)产出的
// 执行计划(slot 表 + KernelFn 解析 + arena 偏移),run() 按计划逐步调用
// KernelFn,恒等/仅输入图(图输出直接来自图输入的 slot)天然由同一套 slot
// 映射机制覆盖,无需特判。
// M9 memory_planning 落地(决议点 D 覆盖版):中间 Value 经单块 arena
// Storage 按偏移复用,不再逐步各自分配;已知代价——fused_elementwise_internal
// (决议点 B,src/backends/cpu/kernels/fused_elementwise.cpp)组合调用内部
// 产生的中间临时张量不在本类的 slot/arena 规划范围内,融合在 cpu 上是机制
// 验证而非内存优化。

#include <memory>
#include <span>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/executable.h>
#include <frame/hal/stream.h>
#include <frame/ir/graph.h>
#include <frame/ops/executable_plan.h>

namespace frame::backends::cpu {

// CpuExecutable:见文件头注释。虚函数依据同 include/frame/hal/backend.h 头部
// R1∧R2∧R3 判定(Executable 属 HAL 白名单六类之一)。
class CpuExecutable final : public hal::Executable {
 public:
  // 编译入口:由 CpuBackend::compile 调用(该处已校验图非空且 device 为
  // cpu,并把校验通过的 device 直接传入——本函数不重新从图推导 device,避免
  // 与调用方的推导逻辑出现第二份不一致实现)。逐非 graph_input 节点解析
  // KernelFn(缺失 → ARCH-031 英文错误,含算子名与后端名);构建 slot 表 =
  // 图输入(按 inputs() 序)+ 各节点输出(按拓扑序);编译期自持 op 名
  // (string 拷贝)、attrs(值拷贝)、KernelFn、输入 slot 索引、输出 slot 索引
  // 与输出签名——生命周期独立于 Graph。allocator 借用契约:CpuBackend 进程级
  // 单例 allocator,生命周期贯穿进程,本类仅存其观察指针(同
  // include/frame/core/storage.h 契约①)。
  static Result<std::unique_ptr<CpuExecutable>> compile(const ir::Graph& graph,
                                                        hal::Allocator& allocator, Device device);

  // 一次执行整图:校验 inputs/outputs 的 span 尺寸与编译期签名逐位一致
  // (dtype+shape,违例返回英文错误);中间(非图输出)Value 的张量经一次性
  // 分配的单块 arena Storage 按 compile() 期算好的偏移切片构造
  // (Tensor::from_storage_slice,M9 memory_planning 落地,决议点 D 覆盖版,
  // 替换此前的朴素逐次分配);图输出对应的张量仍独立分配(Tensor::empty),
  // 生命周期覆盖整个 run() 调用,交还调用方前逐字节拷出。逐步构造
  // ops::KernelContext 调 KernelFn;图输出最终从对应 slot 拷贝进调用方
  // outputs。stream 由调用方保证非悬挂,cpu 后端同步执行、不消费其内容。
  Status run(std::span<const Tensor> inputs, std::span<Tensor> outputs,
             hal::Stream& stream) override;

  // 编译期确定的输入/输出签名(dtype/shape 列表),供调用方在 run() 之前预
  // 分配 outputs、供编译缓存键使用。
  std::vector<hal::IoSpec> input_signature() const override { return input_signature_; }
  std::vector<hal::IoSpec> output_signature() const override { return output_signature_; }

  // 默认构造仅供 compile() 内部经 std::make_unique 使用(CPP-060:所有权一律
  // 走 std::unique_ptr,禁止裸 new,故构造函数须公开);字段随后由 compile()
  // 逐一填入,构造完成前不对外可见。
  CpuExecutable() = default;

 private:
  // 共享执行计划(M11,include/frame/ops/executable_plan.h):slot 表、KernelFn、
  // arena 偏移均由 ops::build_executable_plan 一次性算好,本类只负责结合
  // device/allocator 执行与 hal::IoSpec 转换(消费方职责,见头注释)。
  ops::ExecutablePlan plan_;
  std::vector<hal::IoSpec> input_signature_;   // 由 plan_.input_types 转换而来,供签名接口使用
  std::vector<hal::IoSpec> output_signature_;  // 由 plan_.output_types 转换而来,供签名接口使用
  Device device_{};
  hal::Allocator* allocator_ = nullptr;  // 借用指针,契约见类头注释
};

}  // namespace frame::backends::cpu
