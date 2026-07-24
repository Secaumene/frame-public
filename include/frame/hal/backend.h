#pragma once
// ============================================================================
// Backend HAL —— 硬件后端统一抽象(铁律 #3 后端矩阵 / 铁律 #1① 编译优先)。
//
// 【虚函数判定规则(全项目引用;机械校验见 scripts/check_iron_rules.sh)】
// 允许出现 `virtual` 当且仅当 R1 ∧ R2 ∧ R3 同时成立:
//   R1 粒度:单次调用完成"资源/任务级"工作(一次分配、一次 kernel 启动、
//            一次整图编译/执行、一次 pass 运行),耗时 ≥ 微秒级,虚调用开销占比可忽略;
//   R2 可替换性:实现必须能在运行时替换(不同硬件、外部插件、用户扩展);
//   R3 白名单:该类型属于以下 6 类之一 ——
//            Backend / Stream / Event / Allocator / Executable(HAL 五类)、
//            Pass(compiler 扩展点)。
// 白名单可扩张范围仅限:各后端对上述接口的实现文件、测试 fixture。
// 其余一切场景(逐元素计算、shape/dtype 分派、内联热路径、Tensor 等值类型)
// 一律禁止虚函数,必须用 template / concept / CRTP / constexpr / 编译期特化表(铁律 #1②)。
// 机械校验:scripts/check_iron_rules.sh 对 include/ 与 src/ 的 `virtual` token
//   出现位置做白名单比对;白名单外新增 virtual 需 ADR(CPP-010)。
// ============================================================================

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/macros.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/ir/attribute.h>

namespace frame::ir {
class Graph;  // 前向声明:compile() 的输入(编译路径唯一入口)
}

namespace frame::hal {

class Stream;      // 见 include/frame/hal/stream.h
class Event;       // 见 include/frame/hal/event.h
class Allocator;   // 见 include/frame/hal/allocator.h
class Executable;  // 见 include/frame/hal/executable.h

// 编译选项:由 Backend::compile 接收,控制后端内部 lowering/codegen 优化档位;
// 后端可将其按值拷贝入 Executable,并于 run 期经 KernelContext 借用视图下传
// kernel(首例 ADR-0019 allow_tf32,精度论证见该 ADR,不在此复述)。
// 标准 pass 管线由 compiler 层固定(ARCH-053),不受本结构控制。
// 纪律:任何新增字段必须同步进入 fingerprint(),否则编译缓存键失真
// (execution-model.md 4.1)——code-reviewer 按此判定。
struct CompileOptions {
  int32_t opt_level = 1;  // 0 = 关闭后端内部优化(调试);1 = 默认档位
  // 图级 fp32 降精度数学模式许可(ADR-0019;允许而非强制):开启时后端可对
  // fp32 计算采用 TF32 级(尾数 >=10 位、fp32 指数域、fp32 累加)数学模式,
  // 无等价模式的后端维持严格 fp32;fp16/bf16 路径与本开关无关。默认关闭:
  // fp32 参考语义不悄然改变。matmul 为首个消费者,conv(M21)沿用本开关。
  bool allow_tf32 = false;
  // 确定性 key=value 串,充当缓存键的「编译选项哈希」分量。
  // 样例:"opt_level=1;allow_tf32=0"。
  std::string fingerprint() const {
    return "opt_level=" + std::to_string(opt_level) + ";allow_tf32=" + (allow_tf32 ? "1" : "0");
  }
};

// eager 单算子启动描述(execution-model.md 2.2)。借用契约与 ops::NodeContext 相同:
// attrs 指针仅在 launch 调用期间有效,可空 = 无属性;spans 借用调用方存储。
// launch 不做形状推断与输出分配:outputs 由调用方按 OpSchema::shape_infer 结果预分配。
struct KernelInvocation {
  std::string_view op;
  std::span<const Tensor> inputs;
  std::span<Tensor> outputs;
  const std::unordered_map<std::string, ir::AttrValue>* attrs = nullptr;
  Device device;  // 显式携带(空输入算子亦有明确目标),不从张量推导
};

// Backend:后端统一抽象。虚函数依据 R1∧R2∧R3(见文件头判定规则)。
class FRAME_API Backend {
 public:
  virtual ~Backend() = default;

  // 注册键字符串(如 "cpu"/"cuda")。
  virtual std::string_view name() const = 0;

  // 可用设备数。
  virtual Result<int32_t> device_count() const = 0;

  // 创建执行流(所有权交调用方)。
  virtual Result<std::unique_ptr<Stream>> create_stream(Device device) = 0;

  // 创建事件(设备域生命周期;可在流 A record、流 B wait)。
  // 未 record 的 Event:query() == true、synchronize() == Ok(「视为已完成」,
  // 与 CUDA 语义一致;见 backend-hal.md 2.3,一致性套件据此断言)。
  virtual Result<std::unique_ptr<Event>> create_event(Device device) = 0;

  // 指定设备的分配器(所有权归后端,返回观察指针)。
  virtual Allocator* allocator(Device device) = 0;

  // 设备间/主机拷贝;方向(H2D/D2H/D2D)由两端 Device 推导,主机内存以 cpu 后端 Device 表示。
  virtual Status copy(void* dst, Device dst_device, const void* src, Device src_device,
                      size_t bytes, Stream* stream) = 0;

  // ★ 铁律 #1① 首要职责:把整图编译为 Executable(编译路径的唯一入口)。
  virtual Result<std::unique_ptr<Executable>> compile(const ir::Graph& graph,
                                                      const CompileOptions& options) = 0;

  // eager 兜底:单 kernel 启动;仅在整图编译不可用/调试/单算子测试时使用,
  // 准入见 docs/architecture/execution-model.md(ARCH-011)。
  virtual Status launch(const KernelInvocation& invocation, Stream* stream) = 0;
};

// 全局后端注册表:以注册键**字符串**为键(不设 DeviceType 枚举)。
// 不强制注册键封闭清单(该清单约束 in-tree 后端命名,见 backend-hal.md 第 3 章;
// 外部插件可注册其他键,呼应 R2 运行时可替换性)。
class FRAME_API BackendRegistry {
 public:
  // instance()/register_backend/register_backend_or_die 均标 noexcept:项目禁用
  // 异常(CPP-020),三者均位于 FRAME_REGISTER_BACKEND 静态初始化链路上;标准容器
  // 扩容失败(bad_alloc)经 noexcept 转为 std::terminate 即 fail-fast,如实反映
  // 启动期注册"不可恢复"的既定契约(与 KernelRegistry 同款理由,见
  // include/frame/ops/kernel_registry.h)。
  static BackendRegistry& instance() noexcept;

  // 以注册键字符串登记后端(所有权归注册表);空名或重名返回错误(英文消息,含名字)。
  Status register_backend(std::string_view name, std::unique_ptr<Backend> backend) noexcept;

  // 非模板核心:重名/空名时输出可区分英文诊断(fprintf stderr)后 std::abort;定义在 .cpp。
  bool register_backend_or_die(std::string_view name, std::unique_ptr<Backend> backend) noexcept;

  // 模板薄壳:宏的唯一直接调用点(noexcept 使 checker 不扫描体内 make_unique)。
  template <typename BackendClass>
  bool register_backend_or_die(std::string_view name) noexcept {
    return register_backend_or_die(name, std::make_unique<BackendClass>());
  }

  // 按注册键查找后端;不存在返回 kNotFound(含名字)。
  Result<Backend*> get(std::string_view name);

  // 已注册后端的注册键列表,按字典序排序(确定性输出)。
  std::vector<std::string_view> available() const;

 private:
  std::unordered_map<std::string, std::unique_ptr<Backend>> backends_;
};

}  // namespace frame::hal

// ---------------------------------------------------------------------------
// FRAME_REGISTER_BACKEND(name_string, BackendClass):后端实现文件末尾写一行完成静态注册。
// 注意:聚合库须以 $<LINK_LIBRARY:WHOLE_ARCHIVE,...> 链接后端静态库,
//   否则静态初始化器符号会被链接器裁剪导致注册丢失(见 cmake/frame_backend.cmake)。
// 展开为一条内部链接(static)的静态初始化器:对 noexcept 的模板薄壳
// register_backend_or_die<BackendClass>(name_string) 的单次直接调用(与
// FRAME_REGISTER_KERNEL 同型,见 include/frame/ops/kernel_registry.h)——诊断
// 逻辑(fprintf/abort)收敛在 src/runtime/backend_registry.cpp 的非模板普通函数
// 体内。记号拼接用 include/frame/core/macros.h 的 FRAME_CONCAT(全部注册宏共用
// 同一份两层拼接宏,REUSE-002)保证 __COUNTER__ 先展开为具体数值。
// ---------------------------------------------------------------------------
#define FRAME_REGISTER_BACKEND(name_string, BackendClass)                            \
  [[maybe_unused]] static const bool FRAME_CONCAT(frame_backend_reg_, __COUNTER__) = \
      ::frame::hal::BackendRegistry::instance().register_backend_or_die<BackendClass>(name_string)
