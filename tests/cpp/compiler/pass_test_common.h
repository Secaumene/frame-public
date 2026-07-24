#pragma once
// M8 四 pass 测试共用设施:测试专用算子 test_side_effect_op /
// test_no_kernel_op 的幂等注册(沿用 M7 test_backend_lowering.cpp::FakeBackend
// 的"测试注册"先例:测试用算子直接注册进进程级 OpRegistry/KernelRegistry
// 单例)。canonicalize/constant_folding/common_subexpression_elimination/
// dead_node_elimination 四个测试文件均需引用这两个算子(kHasSideEffect 除外
// 条款、无 cpu kernel 跳过条款),故抽到共用头(REUSE-002:两个算子只注册
// 一次,不在四个 .cpp 里各自复制一份注册代码)。
//
// 幂等注册的实现取舍:FRAME_REGISTER_OP/FRAME_REGISTER_KERNEL 展开为文件作用域
// 的 static 初始化器,若直接把宏调用写在本头文件里,四个 .cpp 各自 #include
// 本头会让每个翻译单元各生成一份同名 static 初始化器,同一进程内对
// OpRegistry::register_op 重复调用同一算子名 —— 而 register_op 对重名注册
// 是 fail-fast(fatal,见 include/frame/ops/op_registry.h),故不能照搬
// FRAME_REGISTER_OP 直接写文件作用域。改用"函数局部 static"模式:
// ensure_pass_test_ops_registered() 声明为非 static 的 inline 函数——C++
// 标准保证同一 inline 函数在整个程序中只有一份定义,其函数局部 static 变量
// 因此在全部翻译单元间共享同一个实例,只会被初始化一次(即便该 inline
// 函数在四个不同 .cpp 里各自被调用)。各测试文件在用到这两个算子前先调用
// 本函数(如在 TEST 内或 fixture SetUp() 内),不依赖调用顺序、多次调用安全。
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include <frame/core/dtype.h>
#include <frame/core/macros.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/ir/node.h>
#include <frame/ops/kernel_registry.h>
#include <frame/ops/op_registry.h>

namespace frame::compiler::testing {

// 带 kHasSideEffect trait 的恒等算子(1 输入 1 输出,输出 shape/dtype 恒等于
// 输入):constant_folding 的"有副作用节点不折"、
// common_subexpression_elimination/dead_node_elimination 的"有副作用节点
// 除外/保留"专项测试共用。算子名过字符集(^[a-z][a-z0-9_]*$)。
inline constexpr std::string_view kSideEffectOpName = "test_side_effect_op";

// 有 schema 但故意不注册 cpu kernel 的算子(同为 1 输入 1 输出恒等 shape):
// constant_folding "无 cpu kernel 的算子跳过不折(不报错)" 专项测试专用。
inline constexpr std::string_view kNoKernelOpName = "test_no_kernel_op";

namespace detail {

inline Result<std::vector<Shape>> infer_pass_test_identity_shape(const ops::NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument, "op '" + std::string(ctx.op) +
                                                         "' expects 1 input, got " +
                                                         std::to_string(ctx.input_types.size()));
  }
  return std::vector<Shape>{ctx.input_types[0].shape};
}

// 恒等 cpu kernel:整段拷贝输入内存到输出(dtype 未知,按字节拷贝即可——两侧
// TensorType 在 shape_infer 约束下必然同 dtype 同 shape,故字节拷贝等价于
// 逐元素拷贝,无需 dispatch_dtype)。
inline Status identity_cpu_kernel(ops::KernelContext& ctx) {
  if (ctx.inputs.size() != 1) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "pass_test identity kernel expects 1 input, got " + std::to_string(ctx.inputs.size()));
  }
  if (ctx.outputs.size() != 1) {
    return Status::make(
        ErrorCode::kInvalidArgument,
        "pass_test identity kernel expects 1 output, got " + std::to_string(ctx.outputs.size()));
  }
  const Tensor& in = ctx.inputs[0];
  Tensor& out = ctx.outputs[0];
  if (!(in.dtype() == out.dtype()) || !(in.shape() == out.shape())) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "pass_test identity kernel requires in/out of the same dtype/shape");
  }
  const size_t byte_size = static_cast<size_t>(in.numel()) * in.dtype().itemsize();
  std::memcpy(out.raw_data(), in.raw_data(), byte_size);
  return Status::ok();
}

}  // namespace detail

// 幂等注册入口(见本文件头注释):安全地多次调用。
inline void ensure_pass_test_ops_registered() {
  static const bool registered = [] {
    ops::OpRegistry::instance()
        .register_op(kSideEffectOpName)
        .input("in", "identity input")
        .output("out", "identity output, bit-for-bit copy of input")
        .trait(ops::OpTrait::kHasSideEffect)
        .shape_infer(&detail::infer_pass_test_identity_shape);
    const Status side_effect_kernel_status = ops::KernelRegistry::instance().register_kernel(
        kSideEffectOpName, kCpuBackendName, &detail::identity_cpu_kernel);
    FRAME_CHECK(side_effect_kernel_status.is_ok());

    // kNoKernelOpName 故意只注册 schema,不注册 cpu kernel(专用于
    // constant_folding "无 cpu kernel 跳过不折" 用例)。
    ops::OpRegistry::instance()
        .register_op(kNoKernelOpName)
        .input("in", "identity input")
        .output("out", "identity output (no cpu kernel registered)")
        .shape_infer(&detail::infer_pass_test_identity_shape);
    return true;
  }();
  (void)registered;
}

}  // namespace frame::compiler::testing
