// ops 子系统"转正"测试:OpRegistry::register_op 三类 fatal 违例之一 —— ②重名
// (ARCH-040,src/ops/op_registry.cpp::fatal_registration_error)。①非法名/
// ③保留名两类死亡测试见 tests/cpp/ops/test_op_registry.cpp;OpSchema builder
// 与 NodeContext::attr<T> 见 test_op_schema.cpp;KernelRegistry 见
// test_kernel_registry.cpp;make_op_query 见 test_op_query.cpp。骨架桩转正后
// tests/cpp/{compiler,backends,hal_conformance}/ 三个桩仍待各自子系统实化。
#include <gtest/gtest.h>

#include <frame/ops/op_registry.h>

TEST(OpsStub, OpRegistryRejectsDuplicateName) {
  // EXPECT_DEATH 默认 "fast"(fork)风格:子进程是当前进程内存的完整拷贝,故先在
  // "父进程"(fork 之前)成功注册一次,EXPECT_DEATH 语句内(fork 之后的子进程)
  // 重注册同名算子触发 fatal,诊断串 "duplicate op name" 与①非法名/③保留名
  // 两类可区分。
  const frame::ops::OpSchema& first =
      frame::ops::OpRegistry::instance().register_op("test_ops_stub_duplicate_target");
  EXPECT_EQ(first.name(), "test_ops_stub_duplicate_target");

  EXPECT_DEATH(
      { frame::ops::OpRegistry::instance().register_op("test_ops_stub_duplicate_target"); },
      "duplicate op name");
}
