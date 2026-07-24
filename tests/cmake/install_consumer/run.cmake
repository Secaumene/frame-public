if(NOT DEFINED FRAME_TEST_SOURCE_DIR)
  message(FATAL_ERROR "FRAME_TEST_SOURCE_DIR is required")
endif()
if(NOT DEFINED FRAME_TEST_BINARY_DIR)
  message(FATAL_ERROR "FRAME_TEST_BINARY_DIR is required")
endif()
if(NOT DEFINED FRAME_BUILD_DIR)
  message(FATAL_ERROR "FRAME_BUILD_DIR is required")
endif()
if(NOT DEFINED CMAKE_CTEST_COMMAND)
  message(FATAL_ERROR "CMAKE_CTEST_COMMAND is required")
endif()

# 先转绝对规范路径，再证明测试目录是构建树的严格子路径；不信任调用方原始输入。
cmake_path(ABSOLUTE_PATH FRAME_BUILD_DIR NORMALIZE OUTPUT_VARIABLE _frame_build_dir)
cmake_path(ABSOLUTE_PATH FRAME_TEST_BINARY_DIR NORMALIZE OUTPUT_VARIABLE _frame_test_binary_dir)
set(_frame_tests_build_dir "${_frame_build_dir}/tests")
cmake_path(IS_PREFIX _frame_build_dir "${_frame_test_binary_dir}" NORMALIZE
           _frame_test_is_in_build_dir)
cmake_path(IS_PREFIX _frame_tests_build_dir "${_frame_test_binary_dir}" NORMALIZE
           _frame_test_is_in_tests_build_dir)
if(NOT EXISTS "${_frame_build_dir}/CMakeCache.txt")
  message(FATAL_ERROR "FRAME_BUILD_DIR must contain CMakeCache.txt")
endif()
if(NOT _frame_test_is_in_build_dir
   OR NOT _frame_test_is_in_tests_build_dir
   OR _frame_test_binary_dir STREQUAL _frame_build_dir
   OR _frame_test_binary_dir STREQUAL _frame_tests_build_dir)
  message(FATAL_ERROR
    "FRAME_TEST_BINARY_DIR must be a strict child of FRAME_BUILD_DIR/tests")
endif()

# 专用测试根已验证为构建树内的严格子路径；只清理规范化后的该目录。
file(REMOVE_RECURSE "${_frame_test_binary_dir}")

set(_frame_install_prefix "${_frame_test_binary_dir}/prefix")
set(_frame_consumer_build_dir "${_frame_test_binary_dir}/consumer-build")
set(_frame_install_command "${CMAKE_COMMAND}" --install "${_frame_build_dir}" --prefix
                           "${_frame_install_prefix}")
set(_frame_configure_command "${CMAKE_COMMAND}" -E env
                             "CMAKE_PREFIX_PATH=${_frame_install_prefix}" "${CMAKE_COMMAND}" -S
                             "${FRAME_TEST_SOURCE_DIR}/project" -B "${_frame_consumer_build_dir}")
set(_frame_build_command "${CMAKE_COMMAND}" --build "${_frame_consumer_build_dir}")
set(_frame_ctest_command "${CMAKE_CTEST_COMMAND}" --test-dir "${_frame_consumer_build_dir}"
                         --output-on-failure)

if(DEFINED FRAME_TEST_CONFIG AND NOT FRAME_TEST_CONFIG STREQUAL "")
  list(APPEND _frame_install_command --config "${FRAME_TEST_CONFIG}")
  list(APPEND _frame_configure_command "-DCMAKE_BUILD_TYPE=${FRAME_TEST_CONFIG}")
  list(APPEND _frame_build_command --config "${FRAME_TEST_CONFIG}")
  list(APPEND _frame_ctest_command --build-config "${FRAME_TEST_CONFIG}")
endif()

execute_process(
  COMMAND ${_frame_install_command}
  RESULT_VARIABLE _frame_install_result
  OUTPUT_VARIABLE _frame_install_stdout
  ERROR_VARIABLE _frame_install_stderr
)
if(NOT _frame_install_result EQUAL 0)
  message(FATAL_ERROR
    "Frame installation failed (result ${_frame_install_result}).\nstdout:\n${_frame_install_stdout}\nstderr:\n${_frame_install_stderr}")
endif()

execute_process(
  COMMAND ${_frame_configure_command}
  RESULT_VARIABLE _frame_configure_result
  OUTPUT_VARIABLE _frame_configure_stdout
  ERROR_VARIABLE _frame_configure_stderr
)
if(NOT _frame_configure_result EQUAL 0)
  message(FATAL_ERROR
    "Consumer configuration failed (result ${_frame_configure_result}).\nstdout:\n${_frame_configure_stdout}\nstderr:\n${_frame_configure_stderr}")
endif()

execute_process(
  COMMAND ${_frame_build_command}
  RESULT_VARIABLE _frame_build_result
  OUTPUT_VARIABLE _frame_build_stdout
  ERROR_VARIABLE _frame_build_stderr
)
if(NOT _frame_build_result EQUAL 0)
  message(FATAL_ERROR
    "Consumer build failed (result ${_frame_build_result}).\nstdout:\n${_frame_build_stdout}\nstderr:\n${_frame_build_stderr}")
endif()

execute_process(
  COMMAND ${_frame_ctest_command}
  RESULT_VARIABLE _frame_ctest_result
  OUTPUT_VARIABLE _frame_ctest_stdout
  ERROR_VARIABLE _frame_ctest_stderr
)
if(NOT _frame_ctest_result EQUAL 0)
  message(FATAL_ERROR
    "Consumer test failed (result ${_frame_ctest_result}).\nstdout:\n${_frame_ctest_stdout}\nstderr:\n${_frame_ctest_stderr}")
endif()
