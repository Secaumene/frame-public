#pragma once
// 算子测试共用设施:shape_infer 用 TensorType 构造、eager 数值路径共用
// fixture(取 cpu 后端真实 Allocator + 按值列表/shape 构造 Tensor —— 对任意
// ScalarType T 一视同仁,fp16/bf16 用例只需传入 T{bits} 位级字面量构造的向量
// 即复用同一套逻辑,无需另写位级构造专用重载)、kernel 防御性拒绝路径共用
// fixture 模板。三个共用设施均只关心"取 cpu 后端/allocator"与"按值列表/shape
// 构造 Tensor",与具体算子是逐元素(elementwise:add/mul/relu)、归约
// (reduction:sum)还是矩阵乘(matmul)无关,故不以具体算子门类限定类名/文件名
// (REUSE-002:同一份样板,禁止各自维护第二份复制)。本文件名沿用
// "elementwise" 字样是历史命名(最初只服务 add/mul/relu 三个逐元素算子),现已
// 扩展服务 sum(归约)、matmul(矩阵乘)——改名留待后续批次,不影响当前内容的
// 通用性。参照 tests/cpp/ir/ir_test_helpers.h 的先例抽取,供
// tests/cpp/ops/test_op_add.cpp、tests/cpp/ops/test_op_mul.cpp、
// tests/cpp/ops/test_op_relu.cpp、tests/cpp/ops/test_op_sum.cpp、
// tests/cpp/ops/test_op_matmul.cpp 复用。
//
// 与 tests/cpp/common/test_tolerance.cpp 内的 FakeAllocator 版 MakeTensor1D
// 用途不同,不合并:那里全程不依赖任何已注册后端(纯 host 内存,tolerance.h
// 工具自身单测);这里的用例需要真实驱动已注册 cpu 后端的
// Backend::launch/KernelFn,必须用 BackendRegistry 取得的真实 Allocator。

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <initializer_list>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/ir/node.h>
#include <frame/ops/kernel_registry.h>

namespace frame::ops::testing {

// 构造给定 dtype/shape 的 TensorType(shape_infer 单测专用,不涉及实际内存,
// 与 tests/cpp/ir/ir_test_helpers.h::MakeFloat32Type 同思路,但支持任意 dtype
// 以覆盖 dtype 不一致用例)。
inline frame::ir::TensorType MakeType(frame::DType dtype, std::initializer_list<int64_t> dims) {
  frame::ir::TensorType type;
  type.dtype = dtype;
  type.shape = frame::Shape(dims);
  type.layout = frame::ir::Layout::kRowMajor;
  type.device = frame::cpu_device();
  return type;
}

// eager 数值路径共用 fixture:SetUp 取 cpu 后端真实 Allocator(经
// BackendRegistry,hal 已实化);MakeTensor1D<T> 按值列表构造 1D Tensor。各算子
// 测试文件继承本类得到同名派生 fixture(如
// `class AddOpEagerTest : public frame::ops::testing::ElementwiseEagerTestBase {};`),
// TEST_F 打印的测试套件名即派生类名;具体数值(输入值、期望结果)因算子而异,
// 不属可共用逻辑,留在各自文件内。
class ElementwiseEagerTestBase : public ::testing::Test {
 protected:
  void SetUp() override {
    const frame::Result<frame::hal::Backend*> result =
        frame::hal::BackendRegistry::instance().get(frame::kCpuBackendName);
    ASSERT_TRUE(result.is_ok());
    backend_ = result.value();
    device_ = frame::cpu_device();
    allocator_ = backend_->allocator(device_);
    ASSERT_NE(allocator_, nullptr);
  }

  template <typename T>
  frame::Tensor MakeTensor1D(const std::vector<T>& values) {
    frame::Result<frame::Tensor> result =
        frame::Tensor::empty(frame::Shape({static_cast<int64_t>(values.size())}),
                             frame::DType::of<T>(), device_, *allocator_);
    EXPECT_TRUE(result.is_ok());
    frame::Tensor tensor = result.value();
    T* data = tensor.data<T>();
    for (size_t i = 0; i < values.size(); ++i) data[i] = values[i];
    return tensor;
  }

  // 按显式 shape + 展平(行优先)值列表构造 Tensor;MakeTensor1D<T> 只支持
  // 1D,归约(sum)/矩阵乘(matmul)等算子的输入/输出普遍是多维张量,故提供
  // 本通用版本(同样复用 device_/allocator_,不重复"取 cpu 后端"样板)。
  template <typename T>
  frame::Tensor MakeTensorWithShape(const frame::Shape& shape, const std::vector<T>& flat_values) {
    frame::Result<frame::Tensor> result =
        frame::Tensor::empty(shape, frame::DType::of<T>(), device_, *allocator_);
    EXPECT_TRUE(result.is_ok());
    frame::Tensor tensor = result.value();
    EXPECT_EQ(static_cast<int64_t>(flat_values.size()), tensor.numel());
    T* data = tensor.data<T>();
    for (size_t i = 0; i < flat_values.size(); ++i) data[i] = flat_values[i];
    return tensor;
  }

  frame::hal::Backend* backend_ = nullptr;
  frame::Device device_{};
  frame::hal::Allocator* allocator_ = nullptr;
};

// kernel 防御性拒绝路径共用 fixture 模板:SetUp 经
// KernelRegistry::find(OpNameTag::kOpName, cpu) 取 KernelFn,并取 cpu 后端真实
// Allocator。用模板参数(而非虚函数)传递算子名 —— 编译期机制优先于运行时机制
// (铁律 #1②)。调用方在各自测试文件内定义 1 行 tag 结构体 + 类型别名,如:
//   struct AddOpNameTag { static constexpr std::string_view kOpName = "add"; };
//   using AddOpKernelTest = frame::ops::testing::ElementwiseKernelTestBase<AddOpNameTag>;
template <typename OpNameTag>
class ElementwiseKernelTestBase : public ::testing::Test {
 protected:
  void SetUp() override {
    const frame::Result<frame::ops::KernelFn> found =
        frame::ops::KernelRegistry::instance().find(OpNameTag::kOpName, frame::kCpuBackendName);
    ASSERT_TRUE(found.is_ok());
    kernel_ = found.value();

    const frame::Result<frame::hal::Backend*> backend_result =
        frame::hal::BackendRegistry::instance().get(frame::kCpuBackendName);
    ASSERT_TRUE(backend_result.is_ok());
    allocator_ = backend_result.value()->allocator(frame::cpu_device());
    ASSERT_NE(allocator_, nullptr);
  }

  template <typename T>
  frame::Tensor MakeTensor(const frame::Shape& shape) {
    frame::Result<frame::Tensor> result =
        frame::Tensor::empty(shape, frame::DType::of<T>(), frame::cpu_device(), *allocator_);
    EXPECT_TRUE(result.is_ok());
    return result.value();
  }

  frame::ops::KernelFn kernel_ = nullptr;
  frame::hal::Allocator* allocator_ = nullptr;
};

}  // namespace frame::ops::testing
