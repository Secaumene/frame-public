#pragma once
// frame::data 测试公共件(BUILD-012 目录内复用,同
// tests/cpp/ops/elementwise_op_test_helpers.h、
// tests/cpp/compiler/mlp_forward_graph_helper.h 先例):真实 cpu 后端夹具 +
// 张量构造辅助函数 + DataLoader 洗牌索引序独立 oracle。
//
// 分配器选取:经 BackendRegistry 取真实 cpu 后端分配器(测试是调用方,
// ADR-0020 裁定 b——frame::data 源码本身不查 BackendRegistry、不 include hal
// 头,但测试目标链接 frame::frame 聚合库,在 hal 依赖面内取用合法,同
// tests/cpp/nn/test_training_smoke.cpp::NnTrainingSmokeTest、
// tests/cpp/interop/test_onnx_weights.cpp::OnnxWeightsTest 同款夹具)。
//
// 数值比较口径:本目录全部数值断言均为 memcpy 语义下的精确值搬运(DataLoader
// 批组装按行整段 memcpy,不涉及任何浮点运算),测试数据本身取小整数值(fp32
// 可精确表示),故用 EXPECT_EQ 逐元素精确比较,不接入
// tests/cpp/common/tolerance.h 的近似容差工具(该工具面向"计算结果"的近似浮点
// 比较,BUILD-011;本目录场景不适用,亦不属于"手写 EXPECT_NEAR/自造阈值")。

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>

// FRAME_ASSERT_HAS_VALUE_OR_RETURN(opt, msg):对 std::optional 做
// has_value() 断言并在失败时 return 的替代写法,行为与
// `ASSERT_TRUE((opt).has_value()) << msg;` 等价(致命失败、从当前函数
// return),仅用于紧随其后需要解引用同一 opt 变量的场景。不用 gtest 的
// ASSERT_TRUE 宏是因为已用最小复现验证:gtest ASSERT_TRUE 内部
// switch(0){case 0: default: ...} 控制流技巧会使 clang-tidy
// bugprone-unchecked-optional-access 的数据流分析失去对被检查变量的追踪,
// 对"同一具名 optional 变量先 has_value() 检查、再解引用"这一本应合规的写法
// 仍会误报(EXPECT_TRUE 或普通 if-return 均不受影响,唯独 ASSERT_TRUE 触发),
// 属该 checker 与 gtest ASSERT_TRUE 宏的已知交互限制、非真实缺陷。
#define FRAME_ASSERT_HAS_VALUE_OR_RETURN(opt, msg) \
  do {                                             \
    if (!(opt).has_value()) {                      \
      ADD_FAILURE() << msg;                        \
      return;                                      \
    }                                              \
  } while (false)

namespace frame::testing::data {

// 取真实 cpu 后端分配器的公共夹具。
class DataLoaderTestBase : public ::testing::Test {
 protected:
  void SetUp() override {
    const frame::Result<frame::hal::Backend*> backend_result =
        frame::hal::BackendRegistry::instance().get(frame::kCpuBackendName);
    ASSERT_TRUE(backend_result.is_ok());
    backend_ = backend_result.value();
    device_ = frame::cpu_device();
    allocator_ = backend_->allocator(device_);
    ASSERT_NE(allocator_, nullptr);
  }

  // 按 values 逐元素填充一个 cpu float32 张量(行主序,与 shape 元素总数一致
  // 由调用方保证);.value() 前置条件不满足即 fatal,同
  // tests/cpp/nn/test_training_smoke.cpp::MakeTensorFromFloats、
  // tests/cpp/interop/test_onnx_weights.cpp::MakeTensorFromValues 同款写法
  // (测试夹具内的分配失败视为环境异常,不是被测行为,不需要 Result 分支)。
  frame::Tensor MakeFloatTensor(const std::vector<float>& values, const frame::Shape& shape) {
    frame::Tensor tensor =
        frame::Tensor::empty(shape, frame::DType::of<float>(), device_, *allocator_).value();
    float* data = tensor.data<float>();
    for (size_t i = 0; i < values.size(); ++i) data[i] = values[i];
    return tensor;
  }

  frame::hal::Backend* backend_ = nullptr;
  frame::Device device_{};
  frame::hal::Allocator* allocator_ = nullptr;
};

// 独立复算 DataLoader 洗牌索引序:严格照抄
// include/frame/data/dataloader.h 头注释②文档化的公开机械契约——
// std::mt19937_64(seed ^ epoch_index) 驱动 std::shuffle,对 [0, num_samples)
// 恒等序打底。本函数与被测实现(src/data/dataloader.cpp::RebuildIndices)各自
// 独立编写,用作测试侧 oracle:验证被测实现是否遵守文档契约,而非仅仅"调用
// 被测实现两次互相对照"造成的自证循环(该契约本身是 ARCH-076/dataloader.h
// 明文规定的机械可判定算法,不是内部实现细节)。
inline std::vector<int64_t> ComputeShuffledIndices(int64_t num_samples, uint64_t seed,
                                                   uint64_t epoch_index) {
  std::vector<int64_t> indices(static_cast<size_t>(num_samples));
  std::iota(indices.begin(), indices.end(), int64_t{0});
  std::mt19937_64 rng(seed ^ epoch_index);
  std::shuffle(indices.begin(), indices.end(), rng);
  return indices;
}

}  // namespace frame::testing::data
