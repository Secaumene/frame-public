#pragma once
// PassManager:pass 流水线。按加入顺序执行;每个 pass 运行后自动调用
// graph.verify()(ARCH-022),pass 自身不得跳过。

#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <frame/compiler/pass.h>
#include <frame/core/macros.h>
#include <frame/core/status.h>

namespace frame::ir {
class Graph;  // 前向声明
}

namespace frame::compiler {

// PassManager:有序 pass 容器 + 执行器。
class FRAME_API PassManager {
 public:
  PassManager() = default;

  // 直接加入一个 pass 实例(所有权转移)。返回自身以支持链式调用。
  PassManager& add_pass(std::unique_ptr<Pass> pass);

  // 按名从 PassRegistry::create 解析并加入。返回自身以支持链式调用;本方法
  // 签名不携带错误通道,解析失败时把错误暂存(保留首个错误,不被后续
  // add/add_pass 覆盖),run() 开始前原样返回该错误、不执行任何 pass。
  PassManager& add(std::string_view pass_name);

  // 依次运行全部 pass:每个 pass 运行后,若命中 set_dump_ir_after 设置的名字
  // 则先把 ir::dump_text(graph) 写入其 ostream,再调用 graph.verify()
  // (ARCH-022);run 或 verify 任一失败,立即以 "pass '<name>': " 为前缀包装该
  // Status 并返回(verify 失败消息已自带 "V<N>: " 前缀,此处只再加一层 pass
  // 前缀,不重复加)。run() 本身不做入图前的初始 verify——构图 API
  // (Graph::create_node 等)已保证结构合法,这是调用方职责;若调用方传入的
  // 图本身非法,错误会挂在第一个 pass 名下返回(而非归因于图本身),这是刻意
  // 取舍:避免每次 run() 多付出一次全量 verify 的开销。
  Status run(ir::Graph& graph) const;

  // 调试开关:命中 pass_name 的 pass 运行成功(run 返回 OK)后,把 verify 前
  // 的中间图以 ir::dump_text 写入 os。取舍:dump 时机选在 verify 之前,是因为
  // verify 失败时恰恰最需要看到当时的图内容用于诊断,若放在 verify 之后,
  // verify 失败路径就永远看不到 dump。pass_name 以 std::string 拷贝存储、os
  // 以指针存储(生命周期由调用方保证覆盖 run() 调用期间);未设置(默认)=
  // 调试开关关闭。
  void set_dump_ir_after(std::string_view pass_name, std::ostream& os);

  // 只读观测面:按装配序返回已加入各 pass 的 name()(供顺序断言与调试日志;
  // 不暴露 Pass 指针,调用方无法经此获得可变访问权)。add(name) 的延迟错误
  // 语义下,已成功加入 passes_ 的 pass 仍会出现在返回结果中——pending_error_
  // 只影响 run() 是否执行,不影响此前已装配好的 pass 列表本身;解析失败的那
  // 一次 add() 调用本身不产生条目。
  std::vector<std::string_view> pass_names() const;

 private:
  std::vector<std::unique_ptr<Pass>> passes_;
  Status pending_error_;                 // add(name) 解析失败时暂存的首个错误
  std::string dump_pass_name_;           // set_dump_ir_after 命中的 pass 名;空 = 关闭
  std::ostream* dump_stream_ = nullptr;  // 非拥有指针,未设置为 nullptr
};

}  // namespace frame::compiler
