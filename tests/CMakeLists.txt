# =============================================================================
# tests/CMakeLists.txt —— C++ 测试(GoogleTest + gtest_discover_tests)。
#
# GoogleTest 由 cmake/frame_dependencies.cmake 经 FetchContent 锁 tag 提供
# (ADR-0016,不带 FIND_PACKAGE_ARGS)。若拉取失败(离线且无 FetchContent 缓存),
# 降级为不依赖 GTest 的独立可执行:frame_test_headers_compile 的 static_assert
# 在编译期即验证,main 直接返回 0,使 cpu-only 构建仍能配置/构建/测试通过。
#
# 【TOP-006】每个 target 均链接 frame::compiler_flags。
# =============================================================================

if(TARGET GTest::gtest_main)
  include(GoogleTest)

  # ---- 骨架期唯一"真测试":公共头 static_assert(target 命名按 BUILD-002/012)----
  add_executable(frame_test_headers_compile cpp/test_headers_compile.cpp)
  target_compile_definitions(frame_test_headers_compile PRIVATE FRAME_TESTS_HAVE_GTEST=1)
  target_link_libraries(frame_test_headers_compile
    PRIVATE frame::frame GTest::gtest_main frame::compiler_flags)
  gtest_discover_tests(frame_test_headers_compile)

  # ---- 曾经容纳各子系统骨架桩测试的合并可执行文件 frame_test_stubs 已随最后
  #      一个子系统(compiler)实化转正而整体退役(core/ir/ops/backends/
  #      hal_conformance 的骨架桩此前已分别迁出至下方对应 frame_test_<module>
  #      目标;compiler 的 test_compiler_stub.cpp 现随 frame_test_compiler 目标
  #      一并迁入,见下方)。未来新子系统的骨架桩测试不再进 frame_test_stubs
  #      这一合并壳(该壳已不存在),直接建各自的 frame_test_<module> 目标
  #      (BUILD-012)----

  # ---- core 模块单测(BUILD-002/012:frame_test_<module>)。全程 host 内存 +
  #      FakeAllocator(tests/cpp/core/fake_allocator.h),不依赖任何已注册后端
  #      (cpu 参考后端的真实 Allocator 要到 M4 才落地)----
  add_executable(frame_test_core
    cpp/core/test_dtype.cpp
    cpp/core/test_shape.cpp
    cpp/core/test_storage.cpp
    cpp/core/test_tensor.cpp
    cpp/core/test_core_stub.cpp
  )
  target_link_libraries(frame_test_core
    PRIVATE frame::frame GTest::gtest_main frame::compiler_flags)
  gtest_discover_tests(frame_test_core PROPERTIES LABELS "core")

  # ---- ir 模块单测(BUILD-002/012:frame_test_<module>)。全程 host 内存,不依赖
  #      任何已注册后端(TensorType::device 只是数据字段,见 ir_test_helpers.h)----
  add_executable(frame_test_ir
    cpp/ir/test_graph_construction.cpp
    cpp/ir/test_graph_erase.cpp
    cpp/ir/test_graph_mutation.cpp
    cpp/ir/test_graph_layout.cpp
    cpp/ir/test_attribute.cpp
    cpp/ir/test_graph_verify.cpp
    cpp/ir/test_serialization.cpp
    cpp/ir/test_memory_plan.cpp
    cpp/ir/test_ir_stub.cpp
  )
  target_link_libraries(frame_test_ir
    PRIVATE frame::frame GTest::gtest_main frame::compiler_flags)
  gtest_discover_tests(frame_test_ir PROPERTIES LABELS "ir")

  # ---- ops 模块单测(BUILD-002/012:frame_test_<module>)。全程不依赖任何已
  #      注册后端;OpRegistry/KernelRegistry 均为进程级 Meyer's singleton,各
  #      测试注册的 op 名/kernel 键以 "test_*" 前缀跨全体源文件保持进程级唯一
  #      (见 cpp/ops/test_ops_stub.cpp 头注释)----
  add_executable(frame_test_ops
    cpp/ops/test_ops_stub.cpp
    cpp/ops/test_op_schema.cpp
    cpp/ops/test_op_registry.cpp
    cpp/ops/test_kernel_registry.cpp
    cpp/ops/test_op_query.cpp
    cpp/ops/test_executable_plan.cpp
    cpp/ops/test_op_add.cpp
    cpp/ops/test_op_mul.cpp
    cpp/ops/test_op_relu.cpp
    cpp/ops/test_op_sum.cpp
    cpp/ops/test_op_matmul.cpp
    cpp/ops/test_op_square.cpp
    cpp/ops/test_op_constant.cpp
    cpp/ops/test_op_fused_elementwise.cpp
    cpp/ops/test_op_relu_grad_internal.cpp
    cpp/ops/test_op_matmul_grad_internal.cpp
    cpp/ops/test_op_sum_grad_internal.cpp
    cpp/ops/test_op_mse_loss.cpp
    cpp/ops/test_conv2d.cpp
    cpp/ops/test_conv1d.cpp
    cpp/ops/test_pool2d.cpp
    cpp/ops/test_op_reshape.cpp
    cpp/ops/test_op_sigmoid.cpp
    cpp/ops/test_op_heaviside_surrogate.cpp
    cpp/ops/test_op_tanh.cpp
    cpp/ops/test_op_rsqrt.cpp
    cpp/ops/test_op_softmax.cpp
    cpp/ops/test_op_layer_norm.cpp
    cpp/ops/test_op_transpose.cpp
    cpp/ops/test_op_concat.cpp
    cpp/ops/test_op_slice.cpp
    cpp/ops/test_op_gather.cpp
    cpp/ops/test_op_scatter_add.cpp
    cpp/ops/test_op_fft.cpp
    cpp/ops/test_op_selective_scan.cpp
    cpp/ops/test_gradient_closure.cpp
  )
  target_link_libraries(frame_test_ops
    PRIVATE frame::frame GTest::gtest_main frame::compiler_flags)
  gtest_discover_tests(frame_test_ops PROPERTIES LABELS "ops")

  # ---- backends 模块单测(BUILD-002/012:frame_test_<module>)。cpu 参考后端
  #      永远启用(FRAME_REGISTER_BACKEND 静态注册),BackendRegistry 是进程级
  #      Meyer's singleton,本目标各测试文件注册的后端名/kernel op 名以
  #      "test_backend_registry_" / "test_cpu_backend_" / "test_cuda_backend_"
  #      前缀跨全体源文件保持进程级唯一(见 cpp/backends/test_backend_registry.cpp、
  #      cpp/backends/test_cpu_backend.cpp、cpp/backends/test_cuda_backend.cpp
  #      头注释)。test_cuda_backend.cpp 全部 cuda 用例以 BackendRegistry 是否
  #      注册 "cuda" + device_count()>0 探测做 GTEST_SKIP(BUILD-010/M24),cpu-only
  #      构建下自动跳过、不影响本目标零回归 ----
  add_executable(frame_test_backends
    cpp/backends/test_backends_stub.cpp
    cpp/backends/test_backend_registry.cpp
    cpp/backends/test_cpu_backend.cpp
    cpp/backends/test_cuda_backend.cpp
  )
  target_link_libraries(frame_test_backends
    PRIVATE frame::frame GTest::gtest_main frame::compiler_flags)
  gtest_discover_tests(frame_test_backends PROPERTIES LABELS "backends")

  # ---- HAL 一致性测试套件(BUILD-002/012:frame_test_<module>)。以
  #      BackendRegistry::available() 参数化,对任意已注册后端跑同一组接口
  #      行为测试(backend-hal.md 第 5 章 checklist 第 6 项);当前
  #      available() 仅 "cpu",套件实现全程禁止对后端名做特判,新后端接入后
  #      自动纳入 ----
  add_executable(frame_test_hal_conformance
    cpp/hal_conformance/test_hal_conformance_stub.cpp
    cpp/hal_conformance/test_hal_conformance.cpp
  )
  target_link_libraries(frame_test_hal_conformance
    PRIVATE frame::frame GTest::gtest_main frame::compiler_flags)
  gtest_discover_tests(frame_test_hal_conformance PROPERTIES LABELS "hal")

  # ---- compiler 模块单测(BUILD-002/012:frame_test_<module>)。PassRegistry
  #      是进程级 Meyer's singleton,本目标各测试文件注册/使用的 pass 名以
  #      "test_pass_registry_"/"test_pass_manager_"/"test_compiler_stub_"
  #      前缀跨全体源文件保持进程级唯一(同 core/ir/ops 等模块既有纪律)。
  #      golden_test_helpers.h(tests/cpp/compiler/golden_test_helpers.h)读取
  #      tests/cpp/compiler/testdata/ 下的文本文件,路径解析以编译期常量
  #      FRAME_REPO_ROOT_DIR 为基准(ctest 实际工作目录不保证是仓库根,见该
  #      头文件头注释),故本目标须注入该定义 ----
  add_executable(frame_test_compiler
    cpp/compiler/test_compiler_stub.cpp
    cpp/compiler/test_pass_registry.cpp
    cpp/compiler/test_pass_manager.cpp
    cpp/compiler/test_golden_infrastructure.cpp
    cpp/compiler/test_shape_inference.cpp
    cpp/compiler/test_backend_lowering.cpp
    cpp/compiler/test_canonicalize.cpp
    cpp/compiler/test_constant_folding.cpp
    cpp/compiler/test_common_subexpression_elimination.cpp
    cpp/compiler/test_dead_node_elimination.cpp
    cpp/compiler/test_layout_assignment.cpp
    cpp/compiler/test_operator_fusion.cpp
    cpp/compiler/test_memory_planning.cpp
    cpp/compiler/test_autograd.cpp
    cpp/compiler/test_physics_training.cpp
    cpp/compiler/test_training_loop.cpp
  )
  target_compile_definitions(frame_test_compiler
    PRIVATE FRAME_REPO_ROOT_DIR="${PROJECT_SOURCE_DIR}")
  target_link_libraries(frame_test_compiler
    PRIVATE frame::frame GTest::gtest_main frame::compiler_flags)
  gtest_discover_tests(frame_test_compiler PROPERTIES LABELS "compiler")

  # ---- runtime 模块单测(BUILD-002/012:frame_test_<module>)。M7 端到端闭环
  #      验收(m7-design-brief 决议点 6):runtime::compile 编排入口 + 编译缓存
  #      判据;M10 新增 eager 回退链①②③单测(test_fallback_chain.cpp,
  #      m10-design-brief.md 决议点 E)。BackendRegistry/KernelRegistry 均为
  #      进程级 Meyer's singleton,本目标各源文件注册的 fake 后端名/kernel 键
  #      以 "test_compilation_cache_"/"test_fallback_chain_" 前缀跨全体测试
  #      文件保持进程级唯一(同 core/ir/ops/compiler 等模块既有纪律)----
  add_executable(frame_test_runtime
    cpp/runtime/test_runtime_compile.cpp
    cpp/runtime/test_compilation_cache.cpp
    cpp/runtime/test_fallback_chain.cpp
  )
  target_link_libraries(frame_test_runtime
    PRIVATE frame::frame GTest::gtest_main frame::compiler_flags)
  gtest_discover_tests(frame_test_runtime PROPERTIES LABELS "runtime")

  # ---- common 模块单测(BUILD-002/012:frame_test_<module>)。
  #      tests/cpp/common/tolerance.h 是 BUILD-011 数值容差表的唯一实现载体
  #      (header-only,namespace frame::testing);本目标是该头文件自身的单测。
  #      全程使用 host 内存的 FakeAllocator(tests/cpp/core/fake_allocator.h),
  #      不依赖任何已注册后端 ----
  add_executable(frame_test_common
    cpp/common/test_tolerance.cpp
    cpp/common/test_numeric_gradient.cpp
  )
  target_link_libraries(frame_test_common
    PRIVATE frame::frame GTest::gtest_main frame::compiler_flags)
  gtest_discover_tests(frame_test_common PROPERTIES LABELS "common")

  # ---- frontend 模块单测(BUILD-002/012:frame_test_<module>)。前端 DSL
  #      (JSON 模型描述,ADR-0017)对应的纯 C++ ModelSpec/lowering/runner/
  #      emitter 四头单测:spec 校验(FE-002~005)、lowering golden(手法同
  #      compiler 模块,复用 golden_test_helpers.h::read_file_contents,見
  #      cpp/frontend/test_lowering.cpp 头注释)、训练冒烟(收敛阈值断言,
  #      BUILD-011 禁止精确数值 golden)、emit_cpp 产物 golden(同 lowering 手法)。
  #      testdata/ 与 lowering golden 均需 FRAME_REPO_ROOT_DIR(同 compiler 模块)----
  add_executable(frame_test_frontend
    cpp/frontend/test_model_spec.cpp
    cpp/frontend/test_lowering.cpp
    cpp/frontend/test_runner.cpp
    cpp/frontend/test_emitter.cpp
  )
  target_compile_definitions(frame_test_frontend
    PRIVATE FRAME_REPO_ROOT_DIR="${PROJECT_SOURCE_DIR}")
  target_link_libraries(frame_test_frontend
    PRIVATE frame::frame GTest::gtest_main frame::compiler_flags)
  gtest_discover_tests(frame_test_frontend PROPERTIES LABELS "frontend")

  # ---- interop 模块单测(BUILD-002/012:frame_test_<module>)。ONNX 权重
  #      导入导出(ADR-0013)。interop 本身仅依赖 core(hal::Allocator 仅前向
  #      声明的引用形参,见 include/frame/interop/onnx_weights.h);本目标链接
  #      frame::frame 聚合库,故在 hal 依赖面内,经
  #      BackendRegistry::instance().get("cpu")->allocator(...) 取得真实 cpu
  #      分配器后按维护者裁决(方案 b)注入 load_onnx_weights。含
  #      train-save-import-infer 链路用例(ADR 判定②),复用
  #      tests/cpp/compiler/mlp_forward_graph_helper.h 的 MLP 前向图构造 ----
  add_executable(frame_test_interop
    cpp/interop/test_onnx_weights.cpp
  )
  target_link_libraries(frame_test_interop
    PRIVATE frame::frame GTest::gtest_main frame::compiler_flags)
  gtest_discover_tests(frame_test_interop PROPERTIES LABELS "interop")

  # ---- nn 模块单测(BUILD-002/012:frame_test_<module>)。frame::nn(ADR-0020)
  #      C++ 编译期构图组合子层:test_nn_module.cpp 是最小编译冒烟占位用例
  #      (Linear parameters() 计数);批2 Task 2 补齐完整测试组(见
  #      docs/architecture/nn-design.md §8):构图纯度(ARCH-072)/参数清单
  #      确定性与路径命名 golden(ARCH-073)/params 切片不变式(ARCH-071)/
  #      Linear-Sequential 构图与手工构图 dump_text 逐字节等(ARCH-074)/
  #      端到端训练冒烟(判定⑤佐证:nn 不依赖 compiler/runtime 仍可被调用方
  #      组合训练线消费);M21 批3 T6(docs/plan/2026-07-18-batch3-m21-conv.md
  #      第1.4节)新增卷积批次工厂(Conv2d/Conv1d/MaxPool2d/AvgPool2d/Sigmoid/
  #      Flatten/AFF/Dwt2d/Dwt1d)测试组:构图纯度+负例
  #      (test_conv_batch_construction_purity.cpp)、parameters() golden +
  #      AFF params 切片不变式(test_conv_batch_parameters_and_slicing.cpp)、
  #      Dwt 滤波器常量正确性(test_dwt_filter_constants.cpp)、AFF 端到端数值
  #      冒烟(test_aff_smoke.cpp);M22 批4 T5(docs/plan/2026-07-19-batch4-m22-seq.md
  #      §1.7)新增序列批次工厂(LayerNorm/LSTM/MultiheadAttention/
  #      TransformerEncoderBlock)测试组:构图纯度+负例
  #      (test_seq_batch_construction_purity.cpp)、parameters() golden +
  #      MultiheadAttention/TransformerEncoderBlock params 切片不变式
  #      (test_seq_batch_parameters_and_slicing.cpp)、LSTM/MultiheadAttention
  #      端到端数值冒烟(test_lstm_smoke.cpp、test_mha_smoke.cpp);M22 批4 T6
  #      (docs/plan/2026-07-19-batch4-m22-seq.md §2 T6/§3 验收硬门第 5 条)新增
  #      LSTM/Transformer 两网络固定种子收敛测试 + golden IR 快照
  #      (test_lstm_convergence.cpp、test_transformer_convergence.cpp);M23 批5
  #      T5(docs/plan/2026-07-21-batch5-m23-fft.md §1.5)新增频域批次工厂
  #      (SpectralConv1d/FourierFilter1d/Fno1dBlock)测试组:构图纯度+负例
  #      (test_fft_batch_construction_purity.cpp)、parameters() golden +
  #      Fno1dBlock params 切片不变式(test_fft_batch_parameters_and_slicing.cpp);
  #      M23 批5 T6(同计划 §1.5/§3 验收硬门第 5 条)新增 FNO/FourierFilter1d
  #      前向数值 golden + IR dump_text golden + CPU/CUDA 前向一致
  #      (test_fft_batch_forward_numeric.cpp)、SpectralConv1d 单块小回归收敛
  #      测试 + golden IR 快照(test_spectral_conv1d_convergence.cpp)----
  add_executable(frame_test_nn
    cpp/nn/test_nn_module.cpp
    cpp/nn/test_construction_purity.cpp
    cpp/nn/test_parameters_golden.cpp
    cpp/nn/test_params_slicing_invariant.cpp
    cpp/nn/test_linear_sequential_golden.cpp
    cpp/nn/test_training_smoke.cpp
    cpp/nn/test_conv_batch_construction_purity.cpp
    cpp/nn/test_conv_batch_parameters_and_slicing.cpp
    cpp/nn/test_dwt_filter_constants.cpp
    cpp/nn/test_aff_smoke.cpp
    cpp/nn/test_cnn_convergence.cpp
    cpp/nn/test_aff_convergence.cpp
    cpp/nn/test_wavelet_convergence.cpp
    cpp/nn/test_seq_batch_construction_purity.cpp
    cpp/nn/test_seq_batch_parameters_and_slicing.cpp
    cpp/nn/test_lstm_smoke.cpp
    cpp/nn/test_mha_smoke.cpp
    cpp/nn/test_lstm_convergence.cpp
    cpp/nn/test_transformer_convergence.cpp
    cpp/nn/test_fft_batch_construction_purity.cpp
    cpp/nn/test_fft_batch_parameters_and_slicing.cpp
    cpp/nn/test_fft_batch_forward_numeric.cpp
    cpp/nn/test_spectral_conv1d_convergence.cpp
    cpp/nn/test_ssm_batch_construction.cpp
    cpp/nn/test_ssm_convergence.cpp
    cpp/nn/test_snn_batch_construction.cpp
    cpp/nn/test_snn_convergence.cpp
    cpp/nn/test_gnn_batch_construction.cpp
    cpp/nn/test_gnn_convergence.cpp
  )
  # M21 批3 T7(docs/plan/2026-07-18-batch3-m21-conv.md 第1.4/3节)+ M22 批4 T6
  # (docs/plan/2026-07-19-batch4-m22-seq.md §2 T6):网络收敛测试的 golden IR
  # 快照用例经 convergence_test_helpers.h 复用
  # tests/cpp/compiler/golden_test_helpers.h::read_file_contents 读取
  # tests/cpp/nn/testdata/ 下的期望文本,该 helper 头以 FRAME_REPO_ROOT_DIR 为
  # 路径解析基准(同 frame_test_compiler/frame_test_frontend 既有纪律)。
  target_compile_definitions(frame_test_nn
    PRIVATE FRAME_REPO_ROOT_DIR="${PROJECT_SOURCE_DIR}")
  target_link_libraries(frame_test_nn
    PRIVATE frame::frame GTest::gtest_main frame::compiler_flags)
  gtest_discover_tests(frame_test_nn PROPERTIES LABELS "nn")

  # ---- data 模块单测(BUILD-002/012:frame_test_<module>)。frame::data
  #      (ADR-0020)批迭代层:test_dataset_construction.cpp 是最小编译冒烟
  #      占位用例(4 样本 2 列 TensorDataset::create 的 size() 校验)。批2
  #      Task 3 完整测试组(docs/architecture/nn-design.md §5/§8)三件:
  #      test_dataloader_shuffle_determinism.cpp(洗牌确定性,含
  #      reset()/顺序推进一致性)、test_dataloader_batch_boundaries.cpp(批
  #      切分边界:整除/不整除 × drop_last 两态、两个极端 batch_size、epoch
  #      哨兵语义)、test_dataloader_batch_values.cpp(批组装数值正确 +
  #      TensorDataset 负例)。三者共用 data_test_helpers.h(真实 cpu 后端
  #      夹具经 BackendRegistry,ADR-0020 裁定 b;洗牌索引序独立 oracle)----
  add_executable(frame_test_data
    cpp/data/test_dataset_construction.cpp
    cpp/data/test_dataloader_shuffle_determinism.cpp
    cpp/data/test_dataloader_batch_boundaries.cpp
    cpp/data/test_dataloader_batch_values.cpp
  )
  target_link_libraries(frame_test_data
    PRIVATE frame::frame GTest::gtest_main frame::compiler_flags)
  gtest_discover_tests(frame_test_data PROPERTIES LABELS "data")
else()
  message(WARNING
    "GoogleTest not available (offline and no FetchContent cache). Building "
    "frame_test_headers_compile as a standalone executable without GTest; header "
    "static_asserts are still verified at compile time.")
  add_executable(frame_test_headers_compile cpp/test_headers_compile.cpp)
  target_link_libraries(frame_test_headers_compile PRIVATE frame::frame frame::compiler_flags)
  add_test(NAME frame_test_headers_compile COMMAND frame_test_headers_compile)
endif()

# ---- CPU-only 安装消费回归：真实安装到专用前缀，再由仓外 CMake 项目配置、链接并运行。
#      当前包配置仅保证纯 CPU 导出自包含，任一硬件后端启用时不注册本测试。----
if(FRAME_INSTALL_CPP
   AND NOT FRAME_BACKEND_CUDA_ENABLED
   AND NOT FRAME_BACKEND_INTEL_GPU_ENABLED
   AND NOT FRAME_BACKEND_INTEL_NPU_ENABLED
   AND NOT FRAME_BACKEND_ASCEND_ENABLED)
  add_test(
    NAME frame_test_install_consumer
    COMMAND "${CMAKE_COMMAND}"
      "-DFRAME_TEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/cmake/install_consumer"
      "-DFRAME_TEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/cmake/install_consumer"
      "-DFRAME_BUILD_DIR=${PROJECT_BINARY_DIR}"
      "-DFRAME_TEST_CONFIG=$<CONFIG>"
      "-DCMAKE_CTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/install_consumer/run.cmake"
  )
  set_tests_properties(frame_test_install_consumer PROPERTIES
    LABELS "cmake;install;cpu"
    TIMEOUT 180
  )
endif()
