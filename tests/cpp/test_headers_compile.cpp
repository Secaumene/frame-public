// 骨架期唯一"真测试":包含全部公共头,并以头文件实际声明的符号做编译期断言。
// 全部检查为 static_assert(编译期可判定),故本测试"通过"即证明公共 API 头自洽可编译。
// 说明:定义了 FRAME_TESTS_HAVE_GTEST 时挂到 GoogleTest;否则降级为独立可执行
// (main 直接返回 0),static_assert 仍在编译期完成验证(见 tests/CMakeLists.txt)。

#include <cstdint>
#include <string_view>
#include <type_traits>

#include <frame/frame.h>

namespace {

// ---- core/dtype:概念 ScalarType 与 dtype_traits 特化 ----
static_assert(frame::ScalarType<float>, "float must be a valid ScalarType");
static_assert(frame::ScalarType<std::int32_t>, "int32_t must be a valid ScalarType");
static_assert(!frame::ScalarType<void*>, "void* must not be a ScalarType");
static_assert(frame::dtype_traits<float>::size == 4, "float32 item size must be 4 bytes");
static_assert(frame::dtype_traits<double>::size == 8, "float64 item size must be 8 bytes");
static_assert(frame::dtype_traits<float>::code == frame::DTypeCode::kFloat32,
              "float must map to DTypeCode::kFloat32");
static_assert(frame::dtype_traits<frame::float16_t>::code == frame::DTypeCode::kFloat16,
              "float16_t must map to DTypeCode::kFloat16");
static_assert(frame::dtype_traits<frame::float16_t>::size == 2,
              "float16_t item size must be 2 bytes");
static_assert(frame::dtype_traits<frame::float16_t>::name == "float16",
              "float16_t dtype name must be \"float16\"");
static_assert(frame::dtype_traits<frame::bfloat16_t>::code == frame::DTypeCode::kBFloat16,
              "bfloat16_t must map to DTypeCode::kBFloat16");
static_assert(frame::dtype_traits<frame::bfloat16_t>::size == 2,
              "bfloat16_t item size must be 2 bytes");
static_assert(frame::dtype_traits<frame::bfloat16_t>::name == "bfloat16",
              "bfloat16_t dtype name must be \"bfloat16\"");

// ---- core/tensor:值语义(浅拷贝)----
static_assert(std::is_copy_constructible_v<frame::Tensor>,
              "Tensor must be copy constructible (shallow copy)");

// ---- core/device:纯值类型 + 常量后端键 ----
static_assert(std::is_copy_constructible_v<frame::Device>, "Device must be a value type");
static_assert(frame::cpu_device(0) == frame::Device{frame::kCpuBackendName, 0},
              "cpu_device factory must equal aggregate-constructed Device");

// ---- compiler/pass:定义满足 PassType 概念的玩具 pass,验证 CRTP + concept 扩展点自洽 ----
class ToyPass final : public frame::compiler::PassBase<ToyPass> {
 public:
  static constexpr std::string_view kName = "toy_pass";
  frame::Status run_impl(frame::ir::Graph& /*graph*/) { return frame::Status::ok(); }
};
static_assert(frame::compiler::PassType<ToyPass>, "ToyPass must satisfy the PassType concept");

// ---- ops/op_schema:OpTrait 封闭枚举成员存在(k 前缀,首成员值为 0)----
static_assert(static_cast<int>(frame::ops::OpTrait::kElementwise) == 0,
              "OpTrait::kElementwise must exist as the first enumerator");

// ---- 枚举底型锁定:封闭取值集用最小底型承载(performance-enum-size)----
static_assert(sizeof(frame::ErrorCode) == 1, "ErrorCode underlying type must stay uint8_t");
static_assert(sizeof(frame::ops::OpTrait) == 1, "OpTrait underlying type must stay uint8_t");

}  // namespace

#if defined(FRAME_TESTS_HAVE_GTEST)

#include <gtest/gtest.h>

// 编译期 static_assert 已完成全部校验;此处提供一个恒真运行期用例,使 ctest 可发现并运行。
TEST(HeadersCompile, PublicApiStaticAssertions) {
  SUCCEED() << "all compile-time header static_assert checks passed";
}

#else

// 无 GTest 降级路径:static_assert 已在编译期验证,main 仅需成功退出。
int main() { return 0; }

#endif
