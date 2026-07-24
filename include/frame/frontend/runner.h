#pragma once
// run_training:进程内训练执行入口。实现严格镜像
// tests/cpp/compiler/test_training_loop.cpp 的端到端训练循环结构:lower →
// build_backward_graph → build_sgd_update_graph → 两图各 verify → 各
// runtime::compile 一次(循环外)→ 训练循环(参数张量轮换)→
// lower_to_inference_graph 编译执行一次填充最终预测。

#include <string>
#include <vector>

#include <frame/core/macros.h>
#include <frame/core/status.h>
#include <frame/frontend/model_spec.h>

namespace frame::frontend {

// 运行选项:目标后端注册键。lower_to_graph/lower_to_inference_graph 产出的
// 图张量固定挂在 cpu device 上(v0 范围),故本字段实际须为 "cpu";取其他值
// 会在 runtime::compile 的 device 一致性校验处失败。
struct RunOptions {
  std::string backend = "cpu";
};

// 训练报告:逐步 loss 历史(下标 = step)、末步 loss、训练完毕后一次推理图
// 执行得到的预测值(按行优先展开的扁平 float 数组)。
struct RunReport {
  std::vector<double> loss_history;
  double final_loss = 0.0;
  std::vector<float> final_predictions;
};

// 进程内训练执行。数据/参数张量按 frontend-dsl.md 第 2 节顺序(数据输入 →
// target → 逐层参数,weight 先于 bias)以 std::mt19937(training.seed) 生成
// 或取 inline 值(emitter 生成的代码必须复用同一顺序约定,保证同 seed 同
// 轨迹)。库内不向 stdout/stderr 打印,结果全部经 RunReport 返回。
FRAME_API Result<RunReport> run_training(const ModelSpec& spec, const RunOptions& options);

}  // namespace frame::frontend
