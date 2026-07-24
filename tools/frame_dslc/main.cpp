// frame_dslc:JSON DSL 命令行工具(ADR-0017)。三种模式:
//   --check <spec.json>                载入 + 校验,成功打印 "OK <model 名>"
//   --run   <spec.json> [--backend b]  进程内训练(frontend::run_training)
//   --emit  <spec.json> --out <dir>    生成自包含 C++ 训练/推理工程
// 手写 argv 循环,不引 CLI 库(参数规模小,三种互斥模式)。语言纪律(铁律
// #4):标识符/程序输出/错误消息英文;注释中文。

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <frame/frame.h>

#include "json_loader.h"

namespace {

void PrintUsage(std::ostream& out) {
  out << "Usage:\n"
      << "  frame_dslc --check <spec.json>\n"
      << "  frame_dslc --run <spec.json> [--backend <name>]\n"
      << "  frame_dslc --emit <spec.json> --out <dir>\n";
}

// 打印失败 Status 到 stderr,统一格式(examples/02_graph_compile/main.cpp 同款
// CheckOk 手法,REUSE-002:此处不新造一套错误打印约定)。
void PrintError(const std::string& what, const frame::Status& status) {
  std::cerr << what << " failed: " << status.message() << "\n";
}

int RunCheck(const std::string& spec_path) {
  const frame::Result<frame::frontend::ModelSpec> spec_result =
      frame_dslc::load_model_spec_from_json_file(spec_path);
  if (!spec_result.is_ok()) {
    PrintError("load_model_spec_from_json_file", spec_result.status());
    return 1;
  }
  const frame::frontend::ModelSpec& spec = spec_result.value();

  const frame::Status validate_status = frame::frontend::validate(spec);
  if (!validate_status.is_ok()) {
    PrintError("validate", validate_status);
    return 1;
  }

  std::cout << "OK " << spec.name << "\n";
  return 0;
}

// spec_path/backend 语义不同(文件路径 vs. 后端注册键),调用点均为具名变量
// /字面量传参,误置换风险低——NOLINT 规避 bugprone-easily-swappable-parameters
// (手法同 tools/frame_dslc/json_loader.cpp GetString 既有先例)。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
int RunTrain(const std::string& spec_path, const std::string& backend) {
  const frame::Result<frame::frontend::ModelSpec> spec_result =
      frame_dslc::load_model_spec_from_json_file(spec_path);
  if (!spec_result.is_ok()) {
    PrintError("load_model_spec_from_json_file", spec_result.status());
    return 1;
  }
  const frame::frontend::ModelSpec& spec = spec_result.value();

  frame::frontend::RunOptions options;
  options.backend = backend;
  const frame::Result<frame::frontend::RunReport> report_result =
      frame::frontend::run_training(spec, options);
  if (!report_result.is_ok()) {
    PrintError("run_training", report_result.status());
    return 1;
  }
  const frame::frontend::RunReport& report = report_result.value();

  // 逐步 loss 打印:与 emitter 生成代码的打印条件一致(kLogEvery > 0 且
  // step 为其整数倍,或为末步),REUSE-002(同一日志节奏约定不重复发明)。
  const int32_t log_every = spec.training.log_every;
  const auto num_steps = static_cast<int32_t>(report.loss_history.size());
  for (int32_t step = 0; step < num_steps; ++step) {
    if (log_every > 0 && (step % log_every == 0 || step == num_steps - 1)) {
      std::cout << "step " << step << " loss " << report.loss_history[static_cast<size_t>(step)]
                << "\n";
    }
  }
  std::cout << "final_loss " << report.final_loss << "\n";

  std::cout << "predictions =";
  for (const float value : report.final_predictions) {
    std::cout << " " << value;
  }
  std::cout << "\n";
  return 0;
}

// spec_path/out_dir 语义不同(输入文件路径 vs. 输出目录路径),理由同
// RunTrain。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
int RunEmit(const std::string& spec_path, const std::string& out_dir) {
  const frame::Result<frame::frontend::ModelSpec> spec_result =
      frame_dslc::load_model_spec_from_json_file(spec_path);
  if (!spec_result.is_ok()) {
    PrintError("load_model_spec_from_json_file", spec_result.status());
    return 1;
  }
  const frame::frontend::ModelSpec& spec = spec_result.value();

  frame::frontend::EmitOptions options;
  options.output_dir = out_dir;
  const frame::Status emit_status = frame::frontend::emit_cpp(spec, options);
  if (!emit_status.is_ok()) {
    PrintError("emit_cpp", emit_status);
    return 1;
  }

  std::cout << "emitted to " << out_dir << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args(argv + 1, argv + argc);
  if (args.empty()) {
    PrintUsage(std::cerr);
    return 1;
  }

  const std::string& mode = args[0];

  if (mode == "--check") {
    if (args.size() != 2) {
      PrintUsage(std::cerr);
      return 1;
    }
    return RunCheck(args[1]);
  }

  if (mode == "--run") {
    if (args.size() != 2 && args.size() != 4) {
      PrintUsage(std::cerr);
      return 1;
    }
    std::string backend = "cpu";
    if (args.size() == 4) {
      if (args[2] != "--backend") {
        PrintUsage(std::cerr);
        return 1;
      }
      backend = args[3];
    }
    return RunTrain(args[1], backend);
  }

  if (mode == "--emit") {
    if (args.size() != 4 || args[2] != "--out") {
      PrintUsage(std::cerr);
      return 1;
    }
    return RunEmit(args[1], args[3]);
  }

  PrintUsage(std::cerr);
  return 1;
}
