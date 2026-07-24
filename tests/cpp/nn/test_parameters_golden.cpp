// Module::parameters() 确定性 + 路径命名 golden(ARCH-073,
// docs/architecture/nn-design.md §3/§8)。
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/ir/graph.h>
#include <frame/nn/layers.h>
#include <frame/nn/module.h>

namespace {

using frame::cpu_device;
using frame::DType;
using frame::Result;
using frame::Shape;
using frame::ir::Graph;
using frame::ir::TensorType;
using frame::ir::Value;
using frame::nn::Linear;
using frame::nn::Module;
using frame::nn::ParamSpec;
using frame::nn::Relu;
using frame::nn::Sequential;

// ARCH-073 判定用例(nn-design.md §8):
// Sequential("mlp", {Linear("0",...,with_bias=true), Relu("1"),
//                    Linear("2",...,with_bias=false)})。
Module BuildMlpModuleForParameterNamingCase() {
  return Sequential("mlp", {Linear("0", /*batch=*/4, /*in_dim=*/3, /*out_dim=*/5,
                                   /*with_bias=*/true, DType::of<float>()),
                            Relu("1"),
                            Linear("2", /*batch=*/4, /*in_dim=*/5, /*out_dim=*/2,
                                   /*with_bias=*/false, DType::of<float>())});
}

bool TensorTypeEqual(const TensorType& a, const TensorType& b) {
  return a.dtype == b.dtype && a.shape == b.shape && a.layout == b.layout && a.device == b.device;
}

bool ParamSpecEqual(const ParamSpec& a, const ParamSpec& b) {
  return a.name == b.name && TensorTypeEqual(a.type, b.type) && a.init.kind == b.init.kind &&
         a.init.lo == b.init.lo && a.init.hi == b.init.hi && a.init.values == b.init.values;
}

TEST(ModuleParametersGolden, ParametersIsDeterministicAcrossRepeatedCalls) {
  const Module model = BuildMlpModuleForParameterNamingCase();
  const std::vector<ParamSpec> first = model.parameters();
  const std::vector<ParamSpec> second = model.parameters();

  ASSERT_EQ(first.size(), second.size());
  for (size_t i = 0; i < first.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_TRUE(ParamSpecEqual(first[i], second[i]));
  }
}

TEST(ModuleParametersGolden, ParametersNamesMatchGoldenPathPrefixedList) {
  const Module model = BuildMlpModuleForParameterNamingCase();
  const std::vector<ParamSpec> params = model.parameters();

  // golden(先序遍历,src/nn/module.cpp::CollectParameters):Sequential
  // 自身无直接参数 → 子0(Linear "0",with_bias=true → weight,bias)
  // → 子1(Relu "1",无参数)→ 子2(Linear "2",with_bias=false → weight)。
  // 路径分隔符固定 "."。
  const std::vector<std::string> expected_names{"mlp.0.weight", "mlp.0.bias", "mlp.2.weight"};
  ASSERT_EQ(params.size(), expected_names.size());
  for (size_t i = 0; i < params.size(); ++i) {
    EXPECT_EQ(params[i].name, expected_names[i]);
  }
}

TEST(ModuleParametersGolden, DirectParamsPrecedeChildSubtreeParamsInPreorder) {
  // 独立于上面 golden 名字之外,再核对"先序遍历"本身的结构性质:一个带自身
  // 直接参数的合成 Module,其 parameters() 首段必须是自身参数(名字前缀恰为
  // "根名.参数名",不含任何子模块名分段),随后才是子模块子树参数。
  Module root;
  root.name = "root";

  ParamSpec self_param;
  self_param.name = "self_weight";
  self_param.type.dtype = DType::of<float>();
  self_param.type.shape = Shape({2, 2});
  self_param.type.device = cpu_device();
  root.params = {self_param};

  root.children = {Linear("child", /*batch=*/2, /*in_dim=*/2, /*out_dim=*/2, /*with_bias=*/false,
                          DType::of<float>())};
  // 本用例只核对 parameters(),不调用 build(),build_fn 留空占位即可
  // (Module::build() 不在本测试路径上被调用)。
  root.build_fn = [](Graph&, std::span<Value* const>,
                     std::span<Value* const>) -> Result<std::vector<Value*>> {
    return std::vector<Value*>{};
  };

  const std::vector<ParamSpec> params = root.parameters();
  ASSERT_EQ(params.size(), 2u);
  EXPECT_EQ(params[0].name, "root.self_weight");
  EXPECT_EQ(params[1].name, "root.child.weight");
}

}  // namespace
