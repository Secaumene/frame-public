# ADR-0016:GoogleTest 升级至 v1.17.0 并移除 FIND_PACKAGE_ARGS

- 状态:已接受
- 日期:2026-07-13
- 关联铁律:#5 两级复用
- 关联规则:REUSE-010、REUSE-012、REUSE-013、REUSE-014

## 背景

GoogleTest 现锁 v1.15.2 且带 `FIND_PACKAGE_ARGS NAMES GTest`(系统包
优先)。ADR-0014/0015 已把 Google Benchmark 与 pybind11 统一为「锁定
即锁定」口径,并明确 GoogleTest 的 FIND_PACKAGE_ARGS 取舍留待其自身
升级触点——本次升级即该触点。REUSE-010 规定版本变更须 ADR;
REUSE-013 要求升级时采用最新稳定版并记载核实来源与日期。

## 决策

- 锁定版本升级 **v1.15.2 → v1.17.0**(REUSE-013 记载:核实来源
  github.com/google/googletest/releases,核实日期 2026-07-13,
  v1.17.0 为当时最新稳定发行版,2025-04-30 发布;许可证
  BSD-3-Clause,REUSE-014 白名单内)。
- 兼容性核实:v1.17.0 要求 ≥C++17(本仓 C++20 基线,ADR-0004,
  满足);v1.16/1.17 相对 v1.15 的行为变更为增量选项(如
  `--gtest_fail_if_no_test_linked`,默认不启用);本仓 60 个测试
  文件仅使用 `GTest::gtest_main` 与 `gtest_discover_tests`、无
  gmock,预期零源码适配(以两 preset 全量测试实证,见判定方法)。
- **移除 googletest FetchContent 的 `FIND_PACKAGE_ARGS`**:锁定即
  锁定,不再允许系统包静默替代锁定 tag;至此三个 FetchContent 依赖
  (GoogleTest/pybind11/Google Benchmark)防漂移口径统一
  (ADR-0014/0015 同款,pybind11 漂移事故同类风险就此关闭)。
- 判定方法:`cmake/frame_dependencies.cmake` 中 googletest GIT_TAG
  为 v1.17.0 且该段无 FIND_PACKAGE_ARGS;全新配置(删除构建缓存)
  日志中 googletest 走 FetchContent 源码构建、无 `Found GTest`;
  cpu-only 与 dev 两 preset 全量构建 + ctest 零失败。

## 备选方案

- 仅升版、保留 FIND_PACKAGE_ARGS:系统包静默替代面残留,三依赖
  锁定口径分裂,pybind11 同类漂移风险重现——否决。
- 维持 v1.15.2:锁定停留在非最新稳定版,违 REUSE-013——否决。

## 后果

- 正面:全部 FetchContent 依赖「唯一版本来源 = 本文件锁定 GIT_TAG」
  不变式成立;版本回到当前受维护线。
- 负面与代价:离线且无 FetchContent 缓存的环境不再能以系统 GTest
  兜底(tests/CMakeLists.txt 的无 GTest 降级路径保留,其适用语义
  收窄为「离线且无缓存」);首次配置需重新拉取源码。
- 跟进:同批更新 frame_dependencies.cmake 头部版本锁定表与依赖
  决策表类别 B 行、Benchmark 段「与上方 GoogleTest 不同」对照注释、
  tests/CMakeLists.txt 头注释、docs/plan/build-order.md 第 4 节、
  PLAN.md、docs/decisions/README.md 索引。
