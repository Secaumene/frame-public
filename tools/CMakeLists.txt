# =============================================================================
# tools/CMakeLists.txt —— 命令行工具 frame_dslc(FRAME_BUILD_TOOLS=ON 时接入,
# 见顶层 CMakeLists.txt 的 EXISTS 保护与 cmake/frame_options.cmake;
# ADR-0017、ADR-0018)。
#
# 【TOP-006】target 链接 frame::compiler_flags(告警/ASAN 统一接线)。目标命名
# frame_dslc(BUILD-002:tools/ 下命令行工具命名 frame_<tool>)。链聚合库
# frame::frame(而非 frame::frontend)——静态注册翻译单元(FRAME_REGISTER_OP/
# FRAME_REGISTER_PASS/FRAME_REGISTER_BACKEND 等)仅聚合库经 WHOLE_ARCHIVE 保号
# (见 src/CMakeLists.txt),链 frame::frontend 会得到空注册表(design-reviewer
# 裁决)。JSON 解析仅存在于本工具层(nlohmann_json::nlohmann_json,ADR-0018),
# frame_frontend 库不 include nlohmann。
#
# e2e 冒烟测试(随 FRAME_BUILD_TESTS 接入,与 tests/CMakeLists.txt 同一开关;
# 字面 BUILD_TESTING 变量本仓从未 include(CTest) 定义,恒为假,故不可用——
# 用 FRAME_BUILD_TESTS 才能让下方五条测试真正随 ctest --preset cpu-only 跑
# 起来):--run/--check 各跑一次权威样例 tiny_mlp.json,另跑三次 --check 分别
# 拒绝违例 spec(缺 target、schema_version 非法、dtype 非法枚举取值,均
# WILL_FAIL)。原生 add_test(非 GoogleTest 可执行,故不经 gtest_discover_tests),
# LABELS "tools" 供按模块过滤。
# =============================================================================

add_executable(frame_dslc
  frame_dslc/main.cpp
  frame_dslc/json_loader.cpp
)
target_link_libraries(frame_dslc PRIVATE frame::frame nlohmann_json::nlohmann_json frame::compiler_flags)

if(FRAME_BUILD_TESTS)
  add_test(NAME frame_dslc_run_smoke
           COMMAND frame_dslc --run ${CMAKE_CURRENT_SOURCE_DIR}/frame_dslc/testdata/tiny_mlp.json)
  set_tests_properties(frame_dslc_run_smoke PROPERTIES LABELS "tools")

  add_test(NAME frame_dslc_check_ok
           COMMAND frame_dslc --check ${CMAKE_CURRENT_SOURCE_DIR}/frame_dslc/testdata/tiny_mlp.json)
  set_tests_properties(frame_dslc_check_ok PROPERTIES LABELS "tools")

  add_test(NAME frame_dslc_check_rejects_bad_spec
           COMMAND frame_dslc --check ${CMAKE_CURRENT_SOURCE_DIR}/frame_dslc/testdata/invalid_missing_target.json)
  set_tests_properties(frame_dslc_check_rejects_bad_spec PROPERTIES WILL_FAIL TRUE LABELS "tools")

  add_test(NAME frame_dslc_check_rejects_bad_schema_version
           COMMAND frame_dslc --check ${CMAKE_CURRENT_SOURCE_DIR}/frame_dslc/testdata/invalid_schema_version.json)
  set_tests_properties(frame_dslc_check_rejects_bad_schema_version PROPERTIES WILL_FAIL TRUE LABELS "tools")

  add_test(NAME frame_dslc_check_rejects_bad_dtype
           COMMAND frame_dslc --check ${CMAKE_CURRENT_SOURCE_DIR}/frame_dslc/testdata/invalid_bad_dtype.json)
  set_tests_properties(frame_dslc_check_rejects_bad_dtype PROPERTIES WILL_FAIL TRUE LABELS "tools")
endif()
