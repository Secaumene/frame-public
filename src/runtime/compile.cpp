// runtime::compile 编排入口的实现单元(声明与流程说明见
// include/frame/runtime/compile.h 头注释)。编译产物缓存
// (CompilationCache,决议点 4)是本翻译单元内部实现细节,不对外暴露。

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <frame/compiler/pipeline.h>
#include <frame/hal/allocator.h>
#include <frame/hal/stream.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ir/serialization.h>
#include <frame/runtime/compile.h>

#include "fallback_executable.h"
#include "warn_log.h"

namespace frame::runtime {

namespace {

// 编译产物缓存:进程级 Meyer's 单例,std::mutex 保护(粒度粗即可 v0,决议点
// 4)——单把互斥量覆盖查找与插入,多线程编译入口安全。
// 已知边界(design-reviewer 建议②,已登记):无淘汰策略,容量随不同图数线性
// 增长;M10 交付物①"编译缓存"因而标注「已提前至 M7」,淘汰策略留待 M10
// 视实际需要补充。
class CompilationCache {
 public:
  static CompilationCache& instance() {
    static CompilationCache cache;
    return cache;
  }

  std::shared_ptr<hal::Executable> find(const std::string& key) {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto it = entries_.find(key);
    if (it == entries_.end()) return nullptr;
    return it->second;
  }

  void insert(std::string key, std::shared_ptr<hal::Executable> executable) {
    const std::lock_guard<std::mutex> lock(mutex_);
    entries_.emplace(std::move(key), std::move(executable));
  }

 private:
  CompilationCache() = default;

  std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<hal::Executable>> entries_;
};

// 缓存键拼接:'\x1f'(ASCII Unit Separator)分隔三段,规避分量内部可能出现的
// 常见分隔符(dump_text 大量使用逗号/冒号/空格,fingerprint 用 "key=value"
// 形式)。四元组对应关系(execution-model.md 4.1):backend_name 对应"后端
// 名"分量;options.fingerprint() 对应"编译选项哈希"分量;graph_text(即
// ir::dump_text(graph))同时承载"图结构"与"输入 dtype/shape 签名"两个分量
// ——两者均已完整体现在 dump_text 的文本内容中(节点/属性/张量类型逐一列出,
// 图输入的类型即输入签名),故不需要为它们各自拼接单独的字符串段。
std::string make_cache_key(std::string_view backend_name, const hal::CompileOptions& options,
                           const std::string& graph_text) {
  std::string key;
  key.reserve(backend_name.size() + graph_text.size() + 32);
  key += backend_name;
  key += '\x1f';
  key += options.fingerprint();
  key += '\x1f';
  key += graph_text;
  return key;
}

// 取图 device 的后端名:V6 保证全图所有 Value 的 device 一致
// (docs/architecture/ir-design.md 第4章),任取拓扑序中第一个带输出的节点
// (含 graph_input,其恰有 1 输出)即可代表整图,与
// src/compiler/passes/backend_lowering.cpp、
// src/backends/cpu/cpu_backend.cpp::CpuBackend::compile 的取法一致。图无
// 算子节点(空图/仅输入图之外——此处特指连 graph_input 都没有的空图,因为
// 仅输入图仍有 graph_input 节点、有输出)时返回空 string_view,调用方按"无从
// 判定"处理。
std::string_view graph_device_backend(const ir::Graph& graph) {
  for (const ir::Node* node : graph.topological_order()) {
    if (!node->outputs().empty()) {
      return node->outputs()[0].type().device.backend;
    }
  }
  return {};
}

// 哨兵码回退(design-reviewer REVISE 闭环修订 1/6):管线失败与 Backend::compile
// 失败两处共用同一份判定+回退逻辑(REUSE-002)。仅当 failure.code()==
// ErrorCode::kUnimplemented 时尝试 FallbackExecutable::build——消费本函数的
// graph 形参(修订 2:未融合原图,而非管线运行用的 working_copy,working_copy
// 到达 backend_lowering 时可能已含 fusion 产出的内部节点,基于其回退反而会
// 失败)。build 成功则入编译缓存(键同正常产物——回退对上层透明,execution-
// model.md 4.1"两种模式对上层完全透明"同精神)并返回;build 失败则透传 build()
// 自身的 Status(比原 failure 更具体);failure 非哨兵码则原样透传 failure,
// 不尝试回退(图非法/内部错等不属于"不支持",不触发回退链)。
Result<std::shared_ptr<hal::Executable>> resolve_unimplemented_with_fallback(
    const Status& failure, const ir::Graph& graph, std::string_view backend_name,
    const std::string& cache_key, CompilationCache& cache) {
  if (failure.code() != ErrorCode::kUnimplemented) return failure;

  // 整图级回退决策 WARN(code-reviewer 裁决级建议采纳):保证任何一次回退
  // 决策至少产生一条日志——整图模式后端(Backend::compile 整图拒绝)且逐
  // 节点全部命中 ① 时,肇因算子级 WARN/统计为零,若无本条则回退不可观测。
  // 统计维持肇因算子粒度不变(整图级无算子键,不入 FallbackStats)。
  warn_log("runtime::compile: graph compilation unsupported on backend '" +
           std::string(backend_name) +
           "', falling back to eager execution (reason: " + std::string(failure.message()) + ")");

  Result<std::unique_ptr<FallbackExecutable>> built =
      FallbackExecutable::build(graph, backend_name);
  if (!built.is_ok()) return built.status();

  std::shared_ptr<hal::Executable> shared_executable = std::move(built.value());
  cache.insert(cache_key, shared_executable);
  return shared_executable;
}

}  // namespace

Result<std::shared_ptr<hal::Executable>> compile(const ir::Graph& graph,
                                                 std::string_view backend_name,
                                                 const hal::CompileOptions& options) {
  const std::string_view graph_backend = graph_device_backend(graph);
  if (!graph_backend.empty() && graph_backend != backend_name) {
    return Status::make(ErrorCode::kInvalidArgument, "runtime::compile: backend_name '" +
                                                         std::string(backend_name) +
                                                         "' does not match graph device backend '" +
                                                         std::string(graph_backend) + "'");
  }

  // 键在管线前算,取管线运行前的图文本(见 compile.h 头注释"缓存键计算与
  // dump_text 的关系"一节)。
  const std::string graph_text = ir::dump_text(graph);
  const std::string cache_key = make_cache_key(backend_name, options, graph_text);

  CompilationCache& cache = CompilationCache::instance();
  if (std::shared_ptr<hal::Executable> cached = cache.find(cache_key); cached != nullptr) {
    return cached;
  }

  // 管线运行所需的可变工作副本(见 compile.h 头注释"管线运行所需的可变工作
  // 副本"一节)。clone_graph 已提升为 ir 层公开工具(REUSE-002 单份实现,
  // docs/architecture/autograd.md 第2章),本入口与 compiler::build_backward_graph
  // 共用同一份实现。
  Result<ir::Graph> working_copy = ir::clone_graph(graph);
  if (!working_copy.is_ok()) return working_copy.status();

  Result<compiler::PassManager> pipeline = compiler::standard_pipeline(backend_name);
  if (!pipeline.is_ok()) return pipeline.status();

  const Status pipeline_status = pipeline.value().run(working_copy.value());
  if (!pipeline_status.is_ok()) {
    return resolve_unimplemented_with_fallback(pipeline_status, graph, backend_name, cache_key,
                                               cache);
  }

  const Result<hal::Backend*> backend = hal::BackendRegistry::instance().get(backend_name);
  if (!backend.is_ok()) return backend.status();

  Result<std::unique_ptr<hal::Executable>> executable =
      backend.value()->compile(working_copy.value(), options);
  if (!executable.is_ok()) {
    return resolve_unimplemented_with_fallback(executable.status(), graph, backend_name, cache_key,
                                               cache);
  }

  std::shared_ptr<hal::Executable> shared_executable = std::move(executable.value());
  cache.insert(cache_key, shared_executable);
  return shared_executable;
}

Result<std::vector<Tensor>> run_with_allocated_outputs(hal::Executable& executable,
                                                       std::string_view backend_name,
                                                       std::span<const Tensor> inputs) {
  const Result<hal::Backend*> backend = hal::BackendRegistry::instance().get(backend_name);
  if (!backend.is_ok()) return backend.status();

  // 输出设备约定见 include/frame/runtime/compile.h 头注释:IoSpec/Executable
  // 均无 device 字段,统一分配在注册表后端的 0 号设备。Device::backend 视图
  // 必须别名注册表持有的稳定名(Backend::name(),注册表为进程级单例,名字
  // 存储与进程同寿),不得别名调用方的 backend_name 缓冲——调用方缓冲(如
  // Python 绑定包装对象的成员字符串)先于输出张量销毁时视图悬垂(批1-Task4b
  // 回归测试 AllocatedOutputDeviceBackendAliasesRegistryOwnedName)。
  const Device output_device{backend.value()->name(), 0};

  hal::Allocator* allocator = backend.value()->allocator(output_device);
  if (allocator == nullptr) {
    return Status::make(ErrorCode::kInternal, "run_with_allocated_outputs: backend '" +
                                                  std::string(backend_name) +
                                                  "' returned a null allocator for device index 0");
  }

  std::vector<Tensor> outputs;
  const std::vector<hal::IoSpec> output_signature = executable.output_signature();
  outputs.reserve(output_signature.size());
  for (const hal::IoSpec& spec : output_signature) {
    Result<Tensor> allocated = Tensor::empty(spec.shape, spec.dtype, output_device, *allocator);
    if (!allocated.is_ok()) return allocated.status();
    outputs.push_back(std::move(allocated.value()));
  }

  const Result<std::unique_ptr<hal::Stream>> stream = backend.value()->create_stream(output_device);
  if (!stream.is_ok()) return stream.status();

  const Status run_status = executable.run(inputs, outputs, *stream.value());
  if (!run_status.is_ok()) return run_status;

  const Status sync_status = stream.value()->synchronize();
  if (!sync_status.is_ok()) return sync_status;

  return outputs;
}

}  // namespace frame::runtime
