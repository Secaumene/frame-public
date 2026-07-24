# =============================================================================
# examples/CMakeLists.txt —— 可运行示例(FRAME_BUILD_EXAMPLES=ON 时接入,
# 该开关定义见 cmake/frame_options.cmake)。
#
# 【TOP-006】每个 target 均链接 frame::compiler_flags(告警/ASAN 统一接线)。
# 六个示例各自独立可执行,链接聚合库 frame::frame(含全部已启用后端的
# WHOLE_ARCHIVE 静态注册,见 src/CMakeLists.txt)。target 命名循测试目标既有
# 惯例 frame_test_<module>(BUILD-002/012),此处对应 frame_example_<NN>_<name>。
# =============================================================================

add_executable(frame_example_01_tensor_basics 01_tensor_basics/main.cpp)
target_link_libraries(frame_example_01_tensor_basics PRIVATE frame::frame frame::compiler_flags)

add_executable(frame_example_02_graph_compile 02_graph_compile/main.cpp)
target_link_libraries(frame_example_02_graph_compile PRIVATE frame::frame frame::compiler_flags)

add_executable(frame_example_03_custom_op 03_custom_op/main.cpp)
target_link_libraries(frame_example_03_custom_op PRIVATE frame::frame frame::compiler_flags)

add_executable(frame_example_04_nn_and_data 04_nn_and_data/main.cpp)
target_link_libraries(frame_example_04_nn_and_data PRIVATE frame::frame frame::compiler_flags)

add_executable(frame_example_05_training 05_training/main.cpp)
target_link_libraries(frame_example_05_training PRIVATE frame::frame frame::compiler_flags)

add_executable(frame_example_06_onnx_weights 06_onnx_weights/main.cpp)
target_link_libraries(frame_example_06_onnx_weights PRIVATE frame::frame frame::compiler_flags)

if(FRAME_BUILD_TESTS)
  add_test(NAME example_01_tensor_basics COMMAND frame_example_01_tensor_basics)
  if(FRAME_ENABLE_CUDA STREQUAL "ON" AND TARGET frame_backend_cuda)
    add_test(
      NAME example_02_graph_compile_cuda
      COMMAND frame_example_02_graph_compile cuda)
    set(frame_example_02_test_name example_02_graph_compile_cuda)
  else()
    add_test(
      NAME example_02_graph_compile_cpu
      COMMAND frame_example_02_graph_compile cpu)
    set(frame_example_02_test_name example_02_graph_compile_cpu)
  endif()
  add_test(NAME example_03_custom_op COMMAND frame_example_03_custom_op)
  add_test(NAME example_04_nn_and_data COMMAND frame_example_04_nn_and_data)
  add_test(NAME example_05_training COMMAND frame_example_05_training)
  add_test(NAME example_06_onnx_weights COMMAND frame_example_06_onnx_weights)

  set_tests_properties(
    example_01_tensor_basics
    ${frame_example_02_test_name}
    example_03_custom_op
    example_04_nn_and_data
    example_05_training
    example_06_onnx_weights
    PROPERTIES LABELS examples)
  set_tests_properties(
    example_06_onnx_weights
    PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()
