#pragma once
// constant 算子共用工具(M8):注册名锚常量 + attrs<->Tensor 物化/编码 helper。
// 物化逻辑单份(REUSE-002):cpu constant kernel
// (src/backends/cpu/kernels/constant.cpp)与 constant_folding pass 的编译期
// 求值(src/compiler/passes/constant_folding.cpp)均调用本文件两个函数,禁止
// 各自复制第二份。见 docs/architecture/operator-system.md、
// docs/architecture/compiler-passes.md §3.3。

#include <cstdint>
#include <string_view>
#include <unordered_map>

#include <frame/core/macros.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/ir/attribute.h>

namespace frame::ops {

// constant 算子注册名(五处共用锚:schema/kernel/canonicalize/constant_folding/
// common_subexpression_elimination,均引用本常量,禁止各处另行硬编码字面量
// "constant")。
inline constexpr std::string_view kConstantOpName = "constant";

// double 精确表示整数的上界(53 位尾数,2^53)。M22(批4,决议点A,
// docs/plan/2026-07-19-batch4-m22-seq.md §1.1)常量算子 dtype 白名单扩至
// int32/int64 后,value 属性(kDoubleArray)编码整数值时若量级超出本界,
// double 表示会静默失精;src/ops/schemas/constant.cpp 的 shape_infer(decode
// 方向:double -> 目标整数 dtype)与本文件 encode_tensor_to_attrs(encode 方向:
// 整数 dtype -> double)均以本常量为唯一数据源做逐元素 fail-loud 校验,禁止
// 各自另行硬编码该量级。
inline constexpr int64_t kMaxDoubleExactInteger = 9007199254740992LL;  // 2^53

// 从 constant 算子的属性字典填充 out(调用方预分配、out 的 dtype/shape 须与
// attrs 描述一致)。attrs 三键:value(kDoubleArray,行优先展平)/
// shape(kShape)/dtype(kDType)。校验(逐条,违例返回英文错误):三键存在且
// 类型正确;value 元素数 == shape.numel();dtype 属 v0 白名单
// (float32/float16/bfloat16/int32/int64);out 的 dtype/shape 与 attrs 描述
// 一致。精度论证:double 的 53 位尾数精确覆盖 float32(24 位尾数)及其值域
// 子集 float16/bfloat16 的全部有限值,double -> 目标 dtype 因此不存在无法
// 表示的中间值,只需一次 round-to-nearest-even 转换(fp32 为恒等拷贝,
// fp16/bf16 复用 include/frame/core/dtype.h 的位级转换 float_to_float16/
// float_to_bfloat16);int32/int64 的逐元素整值/值域/2^53 精度界校验由调用方
// (src/ops/schemas/constant.cpp::infer_constant_shape)先行把关,本函数按
// double -> 目标整数 dtype 做恒等数值转换(static_cast,不重复校验)。
FRAME_API Status fill_tensor_from_constant_attrs(
    const std::unordered_map<std::string, ir::AttrValue>& attrs, Tensor& out);

// fill_tensor_from_constant_attrs 的逆向:读 in(宿主内存,dtype 属 v0 白名单)
// 逐元素升 double,覆盖式写入 attrs 的 value(kDoubleArray)/shape(kShape)/
// dtype(kDType)三键(已存在的同名键会被替换,其余键不受影响)。int32/int64
// 源值逐元素校验 |value| <= kMaxDoubleExactInteger(fail-loud,防未来整数
// 折叠静默失精,设计门建议项)。
FRAME_API Status encode_tensor_to_attrs(const Tensor& in,
                                        std::unordered_map<std::string, ir::AttrValue>& attrs);

// v0 常量可编码 dtype 白名单谓词(float32/float16/bfloat16/int32/int64)。
// 单一事实来源:此前 constant_utils.cpp/schemas/constant.cpp/
// compiler/autograd.cpp 各持一份文件局部副本,M18 收敛为本函数单份
// (REUSE-002,白名单扩容只改一处);M22 扩容同样只改本函数(决议点A)。
FRAME_API bool is_constant_dtype_supported(DType dtype);

}  // namespace frame::ops
