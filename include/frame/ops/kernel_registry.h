#pragma once
// KernelRegistry:内核注册表(每后端各自注册)+ 自定义算子第 2 步入口。
// 注册键 = (op 名字符串, backend 名字符串),**不含 dtype** —— dtype 差异在 kernel
// 内部经 dispatch_dtype 编译期展开(见 include/frame/core/dtype.h),避免注册表爆炸。
// 见 docs/architecture/operator-system.md 第3/4章。

#include <concepts>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <frame/core/device.h>
#include <frame/core/macros.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/ir/attribute.h>

namespace frame::hal {
class Stream;           // 前向声明:ops 公共头不 include hal 头(M1 依赖倒置先例)
struct CompileOptions;  // 前向声明:与 Stream 同款「ops 借用 hal 指针」先例(ADR-0019)
}  // namespace frame::hal

namespace frame::ops {

// eager kernel 执行上下文:KernelInvocation 的借用视图 + 执行流。
// 借用契约与 NodeContext 同口径:全部指针/span 仅在 kernel 调用期间有效。
struct KernelContext {
  std::span<const Tensor> inputs;
  std::span<Tensor> outputs;  // 调用方预分配
  const std::unordered_map<std::string, ir::AttrValue>* attrs = nullptr;
  Device device;
  hal::Stream* stream = nullptr;
  // 编译选项的借用视图(ADR-0019):可空 = 默认选项(nullptr 视同
  // hal::CompileOptions{}),kernel 侧必须判空后读,不得假设非空。backing
  // store 为对应 Executable 的成员,借用于该 Executable 存活期(长于单次
  // run() 调用),但 kernel 本身仍不得越调用期保留此指针(与本结构体其余
  // 指针/span 同一借用纪律)。首例消费者:allow_tf32(matmul cuda kernel,
  // 见 src/backends/cuda/kernels/matmul.cpp)。
  const hal::CompileOptions* compile_options = nullptr;
};

// 内核函数:函数指针(零间接开销、可静态存表)。
using KernelFn = Status (*)(KernelContext&);

// concept:强制注册体可转为普通函数指针(= 无捕获 lambda / 自由函数)。
template <typename F>
concept KernelImpl = std::convertible_to<F, KernelFn>;

// KernelRegistry:(op, backend) -> KernelFn 的全局注册表。
class FRAME_API KernelRegistry {
 public:
  // instance()/register_kernel/register_kernel_or_die 均标 noexcept:项目禁用
  // 异常(CPP-020),三者均位于 FRAME_REGISTER_KERNEL 静态初始化链路上;标准容器
  // 扩容失败(bad_alloc)经 noexcept 转为 std::terminate 即 fail-fast,如实反映
  // 启动期注册"不可恢复"的既定契约。
  static KernelRegistry& instance() noexcept;

  // 以 (op, backend) 为键登记内核;重复注册返回错误(英文消息,含键)。
  Status register_kernel(std::string_view op, std::string_view backend, KernelFn fn) noexcept;

  // 启动期注册宏专用的 fail-fast 包装(CPP-020):调用 register_kernel,失败时向
  // stderr 输出可区分的英文诊断后 std::abort(),成功返回 true。FRAME_REGISTER_KERNEL
  // 宏直接调用本方法(而非在宏内联 lambda),使 fprintf/abort 落在本 .cpp 的普通
  // 函数体内、不再处于任何静态初始化式之中,避免 clang-tidy
  // bugprone-throwing-static-initialization 逐条扫描 IIFE lambda 体内调用
  // (glibc fprintf 未标 noexcept)而对调用处误报。
  bool register_kernel_or_die(std::string_view op, std::string_view backend, KernelFn fn) noexcept;

  // 按 (op, backend) 查找内核;不存在返回错误。
  Result<KernelFn> find(std::string_view op, std::string_view backend) const;

 private:
  // (op, backend) 键的哈希:std::pair 无内建 std::hash 特化,用
  // boost::hash_combine 同款位运算组合两个 std::hash<std::string> 结果。
  struct KeyHash {
    size_t operator()(const std::pair<std::string, std::string>& key) const {
      const size_t first_hash = std::hash<std::string>()(key.first);
      const size_t second_hash = std::hash<std::string>()(key.second);
      return first_hash ^ (second_hash + 0x9e3779b9U + (first_hash << 6) + (first_hash >> 2));
    }
  };

  // Meyer's singleton(instance())内的内核表;FRAME_REGISTER_KERNEL 宏生成的
  // 初始化器只经 instance() 访问该单例,不直接触碰 kernels_,无跨 TU 静态初始化
  // 顺序问题(同 OpRegistry,见 op_registry.h)。
  std::unordered_map<std::pair<std::string, std::string>, KernelFn, KeyHash> kernels_;
};

}  // namespace frame::ops

// ---------------------------------------------------------------------------
// FRAME_REGISTER_KERNEL(op_name, backend_name, fn):自定义算子第 2 步 —— 为某后端
// 注册内核(fn 须满足 KernelImpl concept)。dtype 分派留在 fn 内部经 dispatch_dtype
// 完成。展开为内部链接(static)的静态初始化器:对 noexcept 函数
// register_kernel_or_die 的单次直接调用(与 FRAME_REGISTER_OP 同型,见
// op_registry.h)——不再内联 lambda,fprintf/abort 的诊断逻辑收敛在
// src/ops/kernel_registry.cpp 的普通函数体内。记号拼接用
// include/frame/core/macros.h 的 FRAME_CONCAT(全部注册宏共用同一份两层拼接宏,
// REUSE-002)保证 __COUNTER__ 先展开为具体数值。
// ---------------------------------------------------------------------------
#define FRAME_REGISTER_KERNEL(op_name, backend_name, fn)                                         \
  [[maybe_unused]] static const bool FRAME_CONCAT(frame_kernel_reg_, __COUNTER__) =              \
      ::frame::ops::KernelRegistry::instance().register_kernel_or_die((op_name), (backend_name), \
                                                                      (fn))
