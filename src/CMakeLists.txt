# =============================================================================
# src/CMakeLists.txt —— 模块 target 与聚合库定义(履行顶层 CMakeLists.txt 的
# TOP-001 ~ TOP-006 契约)。
#
# 骨架期全部实现为占位桩;每个 .cpp = include 对应头 + 一条 FRAME-* 类别待办标注。
# =============================================================================

# ---------------------------------------------------------------------------
# 【TOP-001】六个模块 target(全部 STATIC)+ frame::* 别名:五个挂在主链上
#   (core/ir/ops/compiler/runtime),外加 frame_interop 这一仅依赖 core 的旁支
#   (ADR-0013,宿主侧互操作,不参与主链)、frame_nn 这一依赖集为 core+ir+ops
#   的旁支(ADR-0020,C++ 编译期构图组合子层,不参与主链、不依赖
#   compiler/runtime/frontend)与 frame_data 这一仅依赖 core 的旁支
#   (ADR-0020,批迭代/DataLoader 层,同 frame_interop 口径)。
# 【TOP-002】主链依赖方向固定为 runtime -> compiler -> ops -> ir -> core
#   (仅沿链向下;与 docs/architecture/overview.md ARCH-001 一致——
#   ops 依赖 ir:include/frame/ops/op_schema.h include 了 frame/ir/attribute.h);
#   frame_interop 仅依赖 core、frame_nn 仅依赖 core+ir+ops、frame_data 仅依赖
#   core,均不接入该链(ADR-0013/ADR-0020,overview.md §2.9)。
# 【TOP-003】公共头挂在 frame_core(PUBLIC,BUILD_INTERFACE)。
# 【TOP-006】所有自有 target 链接 frame::compiler_flags(告警/ASAN 统一接线)。
# ---------------------------------------------------------------------------

# ---- core:基础设施(错误模型/类型系统/形状/存储/张量) ----
add_library(frame_core STATIC
  core/status.cpp
  core/dtype.cpp
  core/shape.cpp
  core/storage.cpp
  core/tensor.cpp
)
add_library(frame::core ALIAS frame_core)
target_include_directories(frame_core PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
# C++20 要求经导出包传递给外部消费方(find_package(frame) 工程无需自设标准;
# 缺此行时消费工程在默认标准下编译失败,ADR-0017 批实测发现);构建树内与
# 顶层 CMAKE_CXX_STANDARD 20 一致无行为变化。
target_compile_features(frame_core PUBLIC cxx_std_20)
target_link_libraries(frame_core PRIVATE frame::compiler_flags)

# ---- interop:宿主侧互操作(ONNX 权重导入/导出,ADR-0013)。仅依赖 core,
#   不接入 runtime->compiler->ops->ir->core 主链;不含 FRAME_REGISTER_*
#   静态初始化器,不进 WHOLE_ARCHIVE(下方聚合库处普通链接即可)。----
add_library(frame_interop STATIC
  interop/onnx_weights.cpp
)
add_library(frame::interop ALIAS frame_interop)
target_link_libraries(frame_interop PUBLIC frame::core)
target_link_libraries(frame_interop PRIVATE frame::compiler_flags)

# ---- data:C++ 数据加载层(ADR-0020,M20 批2 Task 3)。TensorDataset +
#   DataLoader(批迭代/固定种子洗牌/drop_last),仅依赖 core(hal::Allocator
#   仅前向声明借用,同 frame_interop 口径);不含 FRAME_REGISTER_* 静态初始
#   化器,不进 WHOLE_ARCHIVE(下方聚合库处普通链接即可)。----
add_library(frame_data STATIC
  data/dataset.cpp
  data/dataloader.cpp
)
add_library(frame::data ALIAS frame_data)
target_link_libraries(frame_data PUBLIC frame::core)
target_link_libraries(frame_data PRIVATE frame::compiler_flags)

# ---- ir:图 IR(属性/节点/图) ----
add_library(frame_ir STATIC
  ir/attribute.cpp
  ir/node.cpp
  ir/graph.cpp
  ir/serialization.cpp
  ir/memory_plan.cpp
)
add_library(frame::ir ALIAS frame_ir)
target_link_libraries(frame_ir PUBLIC frame::core)
target_link_libraries(frame_ir PRIVATE frame::compiler_flags)

# ---- ops:算子/内核注册表 + 内置 schema 桩(依赖 ir:schema 以 ir::AttrType 描述属性) ----
add_library(frame_ops STATIC
  ops/op_registry.cpp
  ops/kernel_registry.cpp
  ops/op_schema.cpp
  ops/constant_utils.cpp
  ops/fused_elementwise_utils.cpp
  ops/executable_plan.cpp
  ops/graph_builder.cpp
  ops/schemas/elementwise.cpp
  ops/schemas/matmul.cpp
  ops/schemas/reduction.cpp
  ops/schemas/constant.cpp
  ops/schemas/fused_elementwise.cpp
  ops/schemas/loss.cpp
  ops/schemas/conv.cpp
  ops/schemas/pool.cpp
  ops/schemas/shape.cpp
  ops/schemas/sequence.cpp
  ops/schemas/gather.cpp
  ops/schemas/fft.cpp
)
add_library(frame::ops ALIAS frame_ops)
target_link_libraries(frame_ops PUBLIC frame::ir)
target_link_libraries(frame_ops PRIVATE frame::compiler_flags)

# ---- nn:C++ 编译期构图组合子层(ADR-0020,M20)。依赖集 = core+ir+ops,不
#   接入 runtime->compiler->ops->ir->core 主链(nn 只产出前向子图 + 确定性
#   参数声明,build_backward_graph/build_sgd_update_graph/compile 的调用留在
#   调用方);不含 FRAME_REGISTER_* 静态初始化器,不进 WHOLE_ARCHIVE(下方
#   聚合库处普通链接即可,同 frame_frontend/frame_interop 口径)。 ----
add_library(frame_nn STATIC
  nn/module.cpp
  nn/layers.cpp
)
add_library(frame::nn ALIAS frame_nn)
target_link_libraries(frame_nn PUBLIC frame::ops)
target_link_libraries(frame_nn PRIVATE frame::compiler_flags)

# ---- compiler:pass 扩展点/管线/内置 pass 桩(依赖 ops:pass 依 OpTrait 决策) ----
add_library(frame_compiler STATIC
  compiler/pass_manager.cpp
  compiler/pass_registry.cpp
  compiler/pipeline.cpp
  compiler/passes/canonicalize.cpp
  compiler/passes/shape_inference.cpp
  compiler/passes/constant_folding.cpp
  compiler/passes/common_subexpression_elimination.cpp
  compiler/passes/dead_node_elimination.cpp
  compiler/passes/layout_assignment.cpp
  compiler/passes/operator_fusion.cpp
  compiler/passes/memory_planning.cpp
  compiler/passes/backend_lowering.cpp
  compiler/autograd.cpp
)
add_library(frame::compiler ALIAS frame_compiler)
target_link_libraries(frame_compiler PUBLIC frame::ops)
target_link_libraries(frame_compiler PRIVATE frame::compiler_flags)

# ---- runtime:后端注册表/可执行体/编译编排入口(链顶,依赖 compiler 及其
#   传递闭包;compile.cpp 是 include/frame/runtime/compile.h 的实现单元,跑
#   标准管线 + 调 Backend::compile + 编译缓存,M7 已实化) ----
add_library(frame_runtime STATIC
  runtime/backend_registry.cpp
  runtime/executable.cpp
  runtime/warn_log.cpp
  runtime/fallback_stats.cpp
  runtime/fallback_executable.cpp
  runtime/compile.cpp
)
add_library(frame::runtime ALIAS frame_runtime)
target_link_libraries(frame_runtime PUBLIC frame::compiler)
target_link_libraries(frame_runtime PRIVATE frame::compiler_flags)

# ---- frontend:「模型描述 → 可运行」用户面入口(ADR-0017),链顶消费层
#   (依赖方向 core←ir←ops←compiler←runtime←frontend)。依赖 frame::runtime
#   及其传递闭包;不含 FRAME_REGISTER_* 静态初始化器,普通链接即可(不进
#   下方聚合库的 WHOLE_ARCHIVE 清单)。frontend/lowering.cpp 网络结构构图段
#   改经 frame::nn 模块(ADR-0020/M20 批2 Task4),故另 PRIVATE 链 frame::nn
#   ——nn 用法是 lowering.cpp 内部实现细节,不出现在 frontend 任何公开头,
#   无需 PUBLIC 传递。 ----
add_library(frame_frontend STATIC
  frontend/model_spec.cpp
  frontend/lowering.cpp
  frontend/runner.cpp
  frontend/emitter.cpp
)
add_library(frame::frontend ALIAS frame_frontend)
target_link_libraries(frame_frontend PUBLIC frame::runtime)
target_link_libraries(frame_frontend PRIVATE frame::nn)
target_link_libraries(frame_frontend PRIVATE frame::compiler_flags)

# ---------------------------------------------------------------------------
# 【TOP-005】后端子目录:cpu 无条件启用(参考后端,是其余后端样板);
#   其余后端仅当对应 FRAME_BACKEND_*_ENABLED 为真时加入。
#   后端子目录内只调 frame_add_backend()(BKD-001),禁止手写 add_library。
#   frame_add_backend 前置条件:frame::core 与 frame::compiler_flags 已存在(见上)。
# ---------------------------------------------------------------------------
add_subdirectory(backends/cpu)

if(FRAME_BACKEND_CUDA_ENABLED)
  add_subdirectory(backends/cuda)
endif()
if(FRAME_BACKEND_INTEL_GPU_ENABLED)
  add_subdirectory(backends/intel_gpu)
endif()
if(FRAME_BACKEND_INTEL_NPU_ENABLED)
  add_subdirectory(backends/intel_npu)
endif()
if(FRAME_BACKEND_ASCEND_ENABLED)
  add_subdirectory(backends/ascend)
endif()

# ---------------------------------------------------------------------------
# 【TOP-004】聚合库 frame(别名 frame::frame):链接全部模块 target;
#   读取全局属性 FRAME_ENABLED_BACKENDS 链接已启用后端,后端静态库必须用
#   $<LINK_LIBRARY:WHOLE_ARCHIVE,...> 链接(静态注册符号防裁剪,
#   缘由见 cmake/frame_backend.cmake 头部注释)。
# 采用 INTERFACE 库聚合:自身无翻译单元,只传递链接与用法需求。
# ---------------------------------------------------------------------------
add_library(frame INTERFACE)
add_library(frame::frame ALIAS frame)
target_link_libraries(frame INTERFACE
  frame::core
  frame::ir
  frame::runtime
  frame::interop
  frame::frontend
  frame::nn
  frame::data
  frame::compiler_flags
)

# frame_ops 含 FRAME_REGISTER_OP/FRAME_REGISTER_KERNEL 静态初始化器(落地于
# src/ops/schemas/ 等翻译单元);普通静态链接下,若最终可执行文件未直接引用
# 该翻译单元导出的符号,链接器会把整个目标文件连同其静态初始化器一起裁剪,
# 导致算子/内核注册丢失(OpRegistry::find 返回 nullptr)——缘由与
# cmake/frame_backend.cmake 头部注释所述的后端库场景同构,故同样改用
# WHOLE_ARCHIVE 链接。
# 构建树/安装树双分支:install(EXPORT)(cmake/frame_install.cmake)不会改写
# $<LINK_LIBRARY:...> 生成表达式内嵌的裸 target 名,故导出侧需带 frame::
# 命名空间——直接写成导出后 frame-targets.cmake 里实际存在的 imported
# target 名字面量;BUILD_INTERFACE 分支保留裸名供本工程内部(python/tests)
# 构建期链接,INSTALL_INTERFACE 分支供 find_package(frame) 的消费方链接。
target_link_libraries(frame INTERFACE
  $<BUILD_INTERFACE:$<LINK_LIBRARY:WHOLE_ARCHIVE,frame_ops>>
  $<INSTALL_INTERFACE:$<LINK_LIBRARY:WHOLE_ARCHIVE,frame::frame_ops>>)

# frame_compiler 含 FRAME_REGISTER_PASS 静态初始化器(落地于
# src/compiler/passes/ 各翻译单元);与 frame_ops 同构地会被链接器裁剪(见上
# 一条注释),故同样从普通链接清单撤下、改用 WHOLE_ARCHIVE 直接链接(严格镜像
# frame_ops 处理,含 BUILD_INTERFACE/INSTALL_INTERFACE 双分支缘由)。
target_link_libraries(frame INTERFACE
  $<BUILD_INTERFACE:$<LINK_LIBRARY:WHOLE_ARCHIVE,frame_compiler>>
  $<INSTALL_INTERFACE:$<LINK_LIBRARY:WHOLE_ARCHIVE,frame::frame_compiler>>)

get_property(_frame_enabled_backends GLOBAL PROPERTY FRAME_ENABLED_BACKENDS)
foreach(_frame_backend_target IN LISTS _frame_enabled_backends)
  # WHOLE_ARCHIVE:防止 FRAME_REGISTER_BACKEND 静态初始化器符号被链接器裁剪;
  # BUILD_INTERFACE/INSTALL_INTERFACE 双分支缘由同上(导出侧需带 frame::
  # 命名空间字面量)。
  target_link_libraries(frame INTERFACE
    $<BUILD_INTERFACE:$<LINK_LIBRARY:WHOLE_ARCHIVE,${_frame_backend_target}>>
    $<INSTALL_INTERFACE:$<LINK_LIBRARY:WHOLE_ARCHIVE,frame::${_frame_backend_target}>>)
endforeach()
unset(_frame_enabled_backends)
