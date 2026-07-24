#pragma once
// emit_cpp:把 ModelSpec 生成为自包含 C++ 训练/推理源码工程(docs/architecture/
// frontend-dsl.md 第 5 节),风格逐段镜像 examples/02_graph_compile/main.cpp
// (CheckOk + std::cerr 错误检查、中文注释、英文程序输出)。生成的 main.cpp
// 不依赖 frame::frontend(仅链 frame::frame),随机生成顺序与 run_training
// 完全一致(同 seed 同轨迹)。

#include <string>

#include <frame/core/macros.h>
#include <frame/core/status.h>
#include <frame/frontend/model_spec.h>

namespace frame::frontend {

// 产出目录:<output_dir>/main.cpp + <output_dir>/CMakeLists.txt。
struct EmitOptions {
  std::string output_dir;
};

// 生成自包含 C++ 训练/推理工程。output_dir 不存在时经
// std::filesystem::create_directories(错误码重载)创建;产物为确定性文本
// (给定同一 spec 恒生成逐字节相同的内容,不含时间戳)。内部先调用
// validate(spec),失败原样透传。
FRAME_API Status emit_cpp(const ModelSpec& spec, const EmitOptions& options);

}  // namespace frame::frontend
