// ARCH-071 params 切片不变式测试(docs/architecture/nn-design.md §2/§8):
// 传入某 Module::build 的 params 恰为该 Module parameters() 的同序全集;
// 组合模块按 [自身直接参数…, 子0 子树参数…, 子1 子树参数…] 先序分段切片,
// 每段长度 = 对应子 parameters().size()。
//
// 手工拼装一个「带自身直接参数 + 多子模块」的合成 Module(Module 是值语义聚合
// 类型,允许直接手工拼装字段,ARCH-071 头注释),其 build_fn 在切片的同时把
// 实际收到的三段切片记录下来,供测试断言与独立计算的期望切片边界比对。
#include <cstddef>
#include <gtest/gtest.h>
#include <span>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
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
using frame::nn::add_parameter_inputs;
using frame::nn::Linear;
using frame::nn::Module;
using frame::nn::ParamSpec;

// 合成 Module:自身持 1 个直接参数,子模块为两个 Linear(参数数各不相同,便于
// 区分切片边界:child0 = Linear(with_bias=true,2 参数),child1 =
// Linear(with_bias=false,1 参数))。build_fn 按 ARCH-071 不变式对 params 分段
// 切片,分段结果同时记入调用方提供的三个输出参数,供测试断言。
Module BuildCompositeWithSelfAndTwoChildren(const Module& child0, const Module& child1,
                                            std::vector<Value*>* recorded_self_slice,
                                            std::vector<Value*>* recorded_child0_slice,
                                            std::vector<Value*>* recorded_child1_slice) {
  Module composite;
  composite.name = "composite";

  ParamSpec self_param;
  self_param.name = "self_weight";
  self_param.type.dtype = DType::of<float>();
  self_param.type.shape = Shape({2, 2});
  self_param.type.device = cpu_device();
  composite.params = {self_param};

  composite.children = {child0, child1};

  // 直接按值捕获 child0/child1(不经中间局部拷贝再捕获——避免
  // performance-unnecessary-copy-initialization:捕获本身已产生闭包内的那份
  // 拷贝,中间局部变量是多余的第二份拷贝)。inputs/params 是 BuildFn 契约固定
  // 的相邻同型 span 形参(同 src/nn/layers.cpp::Sequential 先例),语义上不可
  // 合并/重排,误置换会在切片尺寸校验与 shape 推断立即暴露。
  // NOLINTBEGIN(bugprone-easily-swappable-parameters)
  composite.build_fn = [child0, child1, recorded_self_slice, recorded_child0_slice,
                        recorded_child1_slice](
                           Graph& graph, std::span<Value* const> inputs,
                           std::span<Value* const> params) -> Result<std::vector<Value*>> {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    constexpr size_t kSelfParamCount = 1;
    const size_t child0_param_count = child0.parameters().size();
    const size_t child1_param_count = child1.parameters().size();

    // ARCH-071 切片不变式:[自身…, 子0 子树…, 子1 子树…],每段长度 = 对应
    // parameters().size()。
    const std::span<Value* const> self_slice = params.subspan(0, kSelfParamCount);
    const std::span<Value* const> child0_slice =
        params.subspan(kSelfParamCount, child0_param_count);
    const std::span<Value* const> child1_slice =
        params.subspan(kSelfParamCount + child0_param_count, child1_param_count);

    recorded_self_slice->assign(self_slice.begin(), self_slice.end());
    recorded_child0_slice->assign(child0_slice.begin(), child0_slice.end());
    recorded_child1_slice->assign(child1_slice.begin(), child1_slice.end());

    const Result<std::vector<Value*>> child0_result = child0.build(graph, inputs, child0_slice);
    if (!child0_result.is_ok()) return child0_result.status();
    const Result<std::vector<Value*>> child1_result =
        child1.build(graph, child0_result.value(), child1_slice);
    if (!child1_result.is_ok()) return child1_result.status();
    return child1_result.value();
  };

  return composite;
}

TEST(ModuleParamsSlicingInvariant, CompositeBuildReceivesSelfThenChildSubtreeSlicesInOrder) {
  std::vector<Value*> recorded_self;
  std::vector<Value*> recorded_child0;
  std::vector<Value*> recorded_child1;

  const Module child0 = Linear("childA", /*batch=*/2, /*in_dim=*/2, /*out_dim=*/3,
                               /*with_bias=*/true, DType::of<float>());  // 2 参数:weight+bias
  const Module child1 = Linear("childB", /*batch=*/2, /*in_dim=*/3, /*out_dim=*/1,
                               /*with_bias=*/false, DType::of<float>());  // 1 参数:weight only

  const Module composite = BuildCompositeWithSelfAndTwoChildren(child0, child1, &recorded_self,
                                                                &recorded_child0, &recorded_child1);

  const std::vector<ParamSpec> flat_params = composite.parameters();
  // 先序遍历:自身(1)+ 子0 子树(2)+ 子1 子树(1)= 4。
  ASSERT_EQ(flat_params.size(), 4u);
  EXPECT_EQ(flat_params[0].name, "composite.self_weight");
  EXPECT_EQ(flat_params[1].name, "composite.childA.weight");
  EXPECT_EQ(flat_params[2].name, "composite.childA.bias");
  EXPECT_EQ(flat_params[3].name, "composite.childB.weight");

  Graph graph("params_slicing");
  TensorType x_type;
  x_type.dtype = DType::of<float>();
  x_type.shape = Shape({2, 2});
  x_type.device = cpu_device();
  Value* x = graph.add_graph_input(x_type).value();

  const Result<std::vector<Value*>> param_inputs = add_parameter_inputs(graph, flat_params);
  ASSERT_TRUE(param_inputs.is_ok()) << param_inputs.status().message();

  const Result<std::vector<Value*>> build_result =
      composite.build(graph, std::vector<Value*>{x}, param_inputs.value());
  ASSERT_TRUE(build_result.is_ok()) << build_result.status().message();

  // ①三段切片长度分别等于对应段的 parameters().size()。
  EXPECT_EQ(recorded_self.size(), 1u);
  EXPECT_EQ(recorded_child0.size(), 2u);
  EXPECT_EQ(recorded_child1.size(), 1u);

  // ②三段依序拼接后与整体传入 build() 的 params 逐位(指针)相等——即 build()
  // 收到的 params 确实按 [自身…, 子0 子树…, 子1 子树…] 被正确切分,不多不少、
  // 不错位(ARCH-071 判定核心)。
  std::vector<Value*> reconstructed;
  reconstructed.insert(reconstructed.end(), recorded_self.begin(), recorded_self.end());
  reconstructed.insert(reconstructed.end(), recorded_child0.begin(), recorded_child0.end());
  reconstructed.insert(reconstructed.end(), recorded_child1.begin(), recorded_child1.end());
  EXPECT_EQ(reconstructed, param_inputs.value());
}

}  // namespace
