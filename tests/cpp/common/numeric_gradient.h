#pragma once
// 数值微分测试公共件(M17,ARCH-066/BUILD-011「解析梯度 ≡ 数值微分校验」专款
// 的唯一实现载体):中心差分 (f(x+h)-f(x-h))/(2h) 逐元素求标量函数 f 对 x 的
// 偏导数。与 tests/cpp/common/tolerance.h(数值容差比较)职责分离、互不重复
// ——本文件只产出数值梯度,不做容差判定,调用方按 BUILD-011 惯例自行转换为
// Tensor 后经 tolerance.h 的 tensor_all_close + relaxed_tolerance 比较(数值
// 微分自带 O(h²) 截断误差,BUILD-011 明文放宽一档)。
//
// h(步长)由调用方显式传入,不内置默认值——不同算子/输入量级的数值条件不同,
// 不应共享同一硬编码步长;本仓 h 实测取值见
// tests/cpp/compiler/test_autograd.cpp 对应用例注释(fp32,BUILD-011 建议
// 1e-3 量级起调)。

#include <cstdint>
#include <functional>
#include <vector>

#include <frame/core/dtype.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>

namespace frame::testing {

// 标量损失函数:给定当前 x(调用方原地扰动后的张量,dtype/shape/device 均与
// 调用方传入 numeric_gradient 的 x 一致),返回标量 loss(double 精度)。典型
// 实现是"跑一次已编译训练图并读回 loss 输出元素"的薄封装闭包。允许返回错误
// (如编译/执行失败),numeric_gradient 遇错误立即透传、不产出局部结果。
using ScalarLossFn = std::function<Result<double>(const Tensor& x)>;

// 对 x 的每个标量分量独立做中心差分,依次对该分量做 +h/-h 原地扰动、各调一
// 次 f,差分后立即原地复原该分量(不额外分配 Tensor、不依赖 Allocator)。x
// 须为 fp32、host 内存可直接解引用的张量(cpu 参考后端的真实 Allocator 分配
// 的张量满足此前提;ARCH-066 要求 fp16/bf16 梯度经 fp32 解析参照验证,故本
// 函数不需要覆盖 fp16/bf16 直接扰动路径)。返回值与 x numel 对应,顺序即线性
// 下标顺序(行优先,与 x 自身的展平顺序一致)。
inline Result<std::vector<double>> numeric_gradient(const ScalarLossFn& f, Tensor& x, double h) {
  if (!(x.dtype() == DType::of<float>())) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "numeric_gradient: x must be float32 (ARCH-066: fp16/bf16 gradients are "
                        "verified against fp32 analytic references, not directly against "
                        "numeric differentiation), got dtype '" +
                            std::string(x.dtype().name()) + "'");
  }

  const int64_t numel = x.numel();
  float* data = x.data<float>();
  std::vector<double> gradient(static_cast<size_t>(numel), 0.0);

  for (int64_t i = 0; i < numel; ++i) {
    const float original = data[static_cast<size_t>(i)];

    data[static_cast<size_t>(i)] = original + static_cast<float>(h);
    const Result<double> plus = f(x);
    if (!plus.is_ok()) {
      data[static_cast<size_t>(i)] = original;
      return plus.status();
    }

    data[static_cast<size_t>(i)] = original - static_cast<float>(h);
    const Result<double> minus = f(x);
    if (!minus.is_ok()) {
      data[static_cast<size_t>(i)] = original;
      return minus.status();
    }

    data[static_cast<size_t>(i)] = original;
    gradient[static_cast<size_t>(i)] = (plus.value() - minus.value()) / (2.0 * h);
  }

  return gradient;
}

}  // namespace frame::testing
