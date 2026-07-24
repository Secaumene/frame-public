// shape 模块单测:verify() 对动态维的拒绝(ARCH-013/ARCH-044)与静态 shape 的接受、
// row_major_strides 正确性、numel。
#include <gtest/gtest.h>
#include <string_view>

#include <frame/core/shape.h>

namespace {

using frame::kDynamicDim;
using frame::Shape;
using frame::Strides;

TEST(ShapeTest, VerifyRejectsDynamicDim) {
  const Shape shape({2, kDynamicDim, 4});
  const frame::Status status = shape.verify();
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), frame::ErrorCode::kInvalidArgument);
  // 错误消息须为英文(LANG-005):只做子串存在性检查,不绑死措辞全文。
  EXPECT_NE(status.message().find("dynamic dimension"), std::string_view::npos);
}

TEST(ShapeTest, VerifyRejectsNegativeDimensionOtherThanDynamicSentinel) {
  // -2 不是动态维哨兵(kDynamicDim == -1),须走 shape.cpp 中"其余负维"分支,
  // 与 kDynamicDim 分支的错误消息不同(不含 "dynamic dimension",而含 "negative")。
  const Shape shape({2, -2, 3});
  const frame::Status status = shape.verify();
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), frame::ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find("negative"), std::string_view::npos);
}

TEST(ShapeTest, VerifyAcceptsStaticShape) {
  const Shape shape({2, 3, 4});
  EXPECT_TRUE(shape.verify().is_ok());
}

TEST(ShapeTest, VerifyAcceptsScalarShape) {
  const Shape shape;  // rank 0(标量)
  EXPECT_TRUE(shape.verify().is_ok());
}

TEST(ShapeTest, HasDynamicDimDetectsSentinel) {
  EXPECT_TRUE(Shape({2, kDynamicDim}).has_dynamic_dim());
  EXPECT_FALSE(Shape({2, 3}).has_dynamic_dim());
}

TEST(ShapeTest, NumelComputesProductOfDims) {
  EXPECT_EQ(Shape({2, 3, 4}).numel(), 24);
  EXPECT_EQ(Shape().numel(), 1);        // 标量:空维列表连乘为 1(空积恒等元)
  EXPECT_EQ(Shape({0, 3}).numel(), 0);  // 含 0 维:元素总数为 0
  EXPECT_EQ(Shape({2, kDynamicDim, 4}).numel(), kDynamicDim);
}

TEST(ShapeTest, RowMajorStridesMatchesExpected) {
  const Strides strides = frame::row_major_strides(Shape({2, 3, 4}));
  EXPECT_EQ(strides, Strides({12, 4, 1}));
}

TEST(ShapeTest, RowMajorStridesRank1) {
  const Strides strides = frame::row_major_strides(Shape({5}));
  EXPECT_EQ(strides, Strides({1}));
}

TEST(ShapeTest, RowMajorStridesRank0) {
  const Strides strides = frame::row_major_strides(Shape());
  EXPECT_EQ(strides, Strides());
}

TEST(ShapeTest, ToStringFormatsDimsCommaSeparated) {
  EXPECT_EQ(Shape().to_string(), "[]");            // 空 shape(标量)
  EXPECT_EQ(Shape({5}).to_string(), "[5]");        // 一维
  EXPECT_EQ(Shape({2, 3}).to_string(), "[2, 3]");  // 二维,逗号+空格分隔
}

}  // namespace
