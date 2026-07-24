// frame::nn 骨架编译冒烟占位测试(M20 批2 Task 2)。构图纯度(ARCH-072)/
// 参数清单确定性与路径命名 golden(ARCH-073)/Linear-Sequential 构图与手工
// 构图 dump_text 逐字节等/端到端训练冒烟等完整测试组见
// docs/architecture/nn-design.md §8,由 test-writer 后续交付;本文件仅验证
// Linear 工厂可用且 parameters() 计数符合 with_bias 预期,防止本骨架 PR 是
// 空测。

#include <cstddef>
#include <gtest/gtest.h>

#include <frame/core/dtype.h>
#include <frame/nn/layers.h>

namespace {

using frame::DType;
using frame::DTypeCode;
using frame::nn::Linear;
using frame::nn::Module;

TEST(NnModuleSmoke, LinearWithBiasHasTwoParameters) {
  const Module linear = Linear("fc1", /*batch=*/8, /*in_dim=*/4, /*out_dim=*/8,
                               /*with_bias=*/true, DType(DTypeCode::kFloat32));
  EXPECT_EQ(linear.parameters().size(), static_cast<size_t>(2));
}

TEST(NnModuleSmoke, LinearWithoutBiasHasOneParameter) {
  const Module linear = Linear("fc2", /*batch=*/8, /*in_dim=*/8, /*out_dim=*/1,
                               /*with_bias=*/false, DType(DTypeCode::kFloat32));
  EXPECT_EQ(linear.parameters().size(), static_cast<size_t>(1));
}

}  // namespace
