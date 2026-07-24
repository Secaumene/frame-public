// compiler 子系统"转正"测试:standard_pipeline(include/frame/compiler/
// pipeline.h)装配的 pass 顺序与 ARCH-053 固定全序的一致性校验。
//
// 观测面:PassManager::pass_names()(include/frame/compiler/pass_manager.h)
// 按装配序返回已加入各 pass 的 name(),可直接做逐位全序断言——早前版本因该
// 只读访问器尚不存在,退而采用"成员资格(set_dump_ir_after 逐名探测)+ 首位
// (V3 非法图短路)"两层旁证,现访问器已补齐,直接改为 EXPECT_EQ 九元素向量
// 一次性证明完整排列,原两层旁证一并移除(不再需要,亦避免与
// tests/cpp/compiler/test_pass_manager.cpp::
// PassProducingIllegalGraphMakesRunFailWithPassAndV3PrefixOnce 重复覆盖
// "pass 产出非法图时 run() 报错前缀格式"这一点——该点已在那条用例里用手工
// 构造的 PassManager 验证过,机制与用哪个 PassManager 实例无关)。
#include <array>
#include <gtest/gtest.h>
#include <string_view>
#include <vector>

#include <frame/compiler/pass_manager.h>
#include <frame/compiler/pipeline.h>

namespace {

using frame::compiler::PassManager;
using frame::compiler::standard_pipeline;

// ARCH-053 固定全序的九个标准 pass 名(唯一权威副本见
// include/frame/compiler/pipeline.h 头注释;本文件独立誊写一份用于逐位比对
// —— 若文档全序被改动而实现未同步,本用例会直接失败于 EXPECT_EQ,如实反映
// "文档与实现不一致",而非静默通过)。
constexpr std::array<std::string_view, 9> kDocumentedOrder = {
    "canonicalize",          "shape_inference",
    "constant_folding",      "common_subexpression_elimination",
    "dead_node_elimination", "layout_assignment",
    "operator_fusion",       "memory_planning",
    "backend_lowering",
};

TEST(CompilerStub, StandardPipelineOrderMatchesDoc) {
  frame::Result<PassManager> pipeline_result = standard_pipeline("cpu");
  ASSERT_TRUE(pipeline_result.is_ok()) << pipeline_result.status().message();
  const PassManager& manager = pipeline_result.value();

  const std::vector<std::string_view> actual_order = manager.pass_names();
  const std::vector<std::string_view> expected_order(kDocumentedOrder.begin(),
                                                     kDocumentedOrder.end());
  EXPECT_EQ(actual_order, expected_order);
}

}  // namespace
