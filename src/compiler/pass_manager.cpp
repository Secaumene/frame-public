// PassManager pass 流水线的实现单元。

#include <ostream>
#include <string>
#include <utility>

#include <frame/compiler/pass_manager.h>
#include <frame/ir/graph.h>
#include <frame/ir/serialization.h>
#include <frame/ops/op_registry.h>

namespace frame::compiler {

PassManager& PassManager::add_pass(std::unique_ptr<Pass> pass) {
  passes_.push_back(std::move(pass));
  return *this;
}

PassManager& PassManager::add(std::string_view pass_name) {
  // 已有更早的错误时保持首个错误不被覆盖(见头文件 add() 注释)。
  if (!pending_error_.is_ok()) return *this;
  Result<std::unique_ptr<Pass>> result = PassRegistry::instance().create(pass_name);
  if (!result.is_ok()) {
    pending_error_ = result.status();
    return *this;
  }
  passes_.push_back(std::move(result.value()));
  return *this;
}

Status PassManager::run(ir::Graph& graph) const {
  if (!pending_error_.is_ok()) return pending_error_;

  const ir::OpQuery op_query = ops::make_op_query();
  for (const std::unique_ptr<Pass>& pass : passes_) {
    const Status run_status = pass->run(graph);
    if (!run_status.is_ok()) {
      return Status::make(run_status.code(), "pass '" + std::string(pass->name()) +
                                                 "': " + std::string(run_status.message()));
    }

    // dump 时机:run 成功后、verify 前(取舍见头文件 set_dump_ir_after 注释)。
    if (dump_stream_ != nullptr && pass->name() == dump_pass_name_) {
      *dump_stream_ << ir::dump_text(graph);
    }

    const Status verify_status = graph.verify(op_query);
    if (!verify_status.is_ok()) {
      // verify_status 消息已带 "V<N>: " 前缀,这里只再加一层 pass 前缀,不
      // 重复加(单前缀纪律)。
      return Status::make(verify_status.code(), "pass '" + std::string(pass->name()) +
                                                    "': " + std::string(verify_status.message()));
    }
  }
  return Status::ok();
}

void PassManager::set_dump_ir_after(std::string_view pass_name, std::ostream& os) {
  dump_pass_name_ = std::string(pass_name);
  dump_stream_ = &os;
}

std::vector<std::string_view> PassManager::pass_names() const {
  std::vector<std::string_view> names;
  names.reserve(passes_.size());
  for (const std::unique_ptr<Pass>& pass : passes_) {
    names.push_back(pass->name());
  }
  return names;
}

}  // namespace frame::compiler
