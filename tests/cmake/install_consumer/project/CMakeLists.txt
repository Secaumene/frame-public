cmake_minimum_required(VERSION 3.24)

project(frame_install_consumer LANGUAGES CXX)

find_package(frame CONFIG REQUIRED)

add_executable(
  frame_quickstart
  ../../../../examples/00_quickstart/main.cpp
)
target_compile_features(frame_quickstart PRIVATE cxx_std_20)
target_link_libraries(frame_quickstart PRIVATE frame::frame)

add_executable(
  frame_test_install_consumer
  ../../../cpp/cmake/test_install_consumer.cpp
)
target_compile_features(frame_test_install_consumer PRIVATE cxx_std_20)
target_link_libraries(frame_test_install_consumer PRIVATE frame::frame)

enable_testing()
add_test(NAME frame_quickstart COMMAND frame_quickstart)
add_test(NAME frame_test_install_consumer COMMAND frame_test_install_consumer)
set_tests_properties(frame_quickstart PROPERTIES PASS_REGULAR_EXPRESSION "^6")
