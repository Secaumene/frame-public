#pragma once
// runtime::compile:编译路径的唯一编排入口(m7-design-brief 决议点 4)。
//
// 流程:①backend_name 与 graph 的 device 后端一致性校验(不一致 →
// kInvalidArgument;图无算子节点、无从判定 device 时跳过本校验,交由后续
// 管线/Backend::compile 自身的空图错误路径处理)。②以
// backend_name + 编译选项指纹 + `ir::dump_text(graph)` 拼接为缓存键(实现见
// src/runtime/compile.cpp;不引入哈希函数,键即字符串本身)查编译缓存,命中
// 则直接返回缓存的 `shared_ptr`(不重跑管线、不重调 `Backend::compile`)。
// ③未命中:跑标准 pass 管线(`include/frame/compiler/pipeline.h`,每 pass 后
// verify 由 `PassManager` 保证,ARCH-022)→ 经 `BackendRegistry::get` 取后端、
// 调 `Backend::compile` 产出 `Executable`,存入缓存后返回。
//
// 【窄接口取舍】本入口不带 dump 参数、不做入图前的初始 `verify()`:
// - `--dump-ir-after` 观测经 `PassManager::set_dump_ir_after` 在更低层的组合
//   完成(见 `architecture/compiler-passes.md` 第2章),不在本入口重复暴露。
// - 非法图不在入口处提前拦截,错误挂在第一个 pass(`canonicalize`)名下返回,
//   与 `PassManager::run()` 既定口径一致(见
//   `include/frame/compiler/pass_manager.h` 头注释、M6 既有实现),避免每次
//   `compile()` 多付出一次全量 verify 的开销。
//
// 【缓存键计算与 dump_text 的关系】键的图结构分量取自对*管线运行前*的输入
// `graph` 调用 `ir::dump_text`(而非管线跑完之后的图)——这是刻意取舍,使
// "同图同签名二次 compile 不触发管线与 Backend::compile" 的判据可在管线开始
// 之前的最早时机短路。`dump_text` 本身不返回 `Result`(签名恒返回
// `std::string`,不失败);其内部前置条件(所引用的每个 `Value*` 均由本图
// 某节点产出)由 `ir::Graph` 的全部公开构图 API(`create_node` 等)在构造期
// 强制维持,故对任何经公开 API 构建的图调用 `dump_text` 均安全,不会触发其
// 内部 `FRAME_CHECK`。`dump_text` 对 `Device::backend` 取值不设限(见
// `include/frame/ir/serialization.h` 头注释第4条),故缓存键计算对自定义/
// 插件后端名同样成立。
//
// 【管线运行所需的可变工作副本】`ir::Graph` 持有 `vector<unique_ptr<Node>>`,
// move-only、不可拷贝,而标准 pass 管线需要 `ir::Graph&`(非 const)。本入口
// 调用 `ir::clone_graph`(`include/frame/ir/graph.h`,M17 提升为 ir 层公开
// 工具,REUSE-002 单份实现,`compiler::build_backward_graph` 共用同一份)取得
// 该副本:经全部既有公开构图 API(`create_node`/`add_graph_input`/
// `mark_output`/`set_attr`)按拓扑序重建一份等价的独立图——**不经**
// `ir::parse_text(ir::dump_text(graph))` 往返(`parse_text` 侧的后端名仅认
// `include/frame/core/device.h` 内置五个注册键常量,见
// `include/frame/ir/serialization.h` 头注释第4条;若借道文本往返取得工作
// 副本,会使本入口对自定义/插件后端名的图报错,与决议点 2/4 要求的 fake
// 后端测试场景冲突),因而本入口对任意 `Device::backend` 取值均可编译,不
// 附加 M2 序列化层的后端名限制。

#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <frame/core/macros.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/backend.h>
#include <frame/hal/executable.h>

namespace frame::ir {
class Graph;  // 前向声明:见 include/frame/ir/graph.h
}  // namespace frame::ir

namespace frame::runtime {

// 编译入口。backend_name 须与 graph 的 device 后端一致;返回值与编译缓存共持
// 同一 Executable(shared_ptr)。
FRAME_API Result<std::shared_ptr<hal::Executable>> compile(const ir::Graph& graph,
                                                           std::string_view backend_name,
                                                           const hal::CompileOptions& options);

// run_with_allocated_outputs:按 Executable::output_signature() 预分配输出
// 张量并执行整图(M12 决议点 A,design-reviewer REVISE 闭环修订 1)——内存
// 管理(取后端/allocator/建流/分配输出/执行/流同步)全部下沉本函数,是
// Python 绑定 `Executable.run` 的唯一 C++ 侧调用点(铁律 #2:业务逻辑不下沉
// 到 Python,绑定层只做参数转换 + 调用 + 错误转换)。
// 流程:①BackendRegistry::get(backend_name) 取后端;②按下方"输出设备约定"
// 取得输出设备,取其 allocator;③按 executable.output_signature() 逐条
// Tensor::empty 分配输出;④backend->create_stream 建流;⑤
// executable.run(inputs, outputs, stream);⑥stream->synchronize();⑦返回
// outputs。
// 输出设备约定:hal::IoSpec 无 device 字段,本函数也不持有 ir::Graph、无法
// 读取图的 device;输出统一分配在注册表后端的 0 号设备(v0 单卡默认约定,
// 与仓内既有端到端测试的单设备场景一致)。生命周期契约:输出张量的
// Device::backend 视图别名注册表持有的稳定后端名(Backend::name()),
// backend_name 形参只用于查注册表——调用方缓冲可在返回后立即销毁。
FRAME_API Result<std::vector<Tensor>> run_with_allocated_outputs(hal::Executable& executable,
                                                                 std::string_view backend_name,
                                                                 std::span<const Tensor> inputs);

}  // namespace frame::runtime
