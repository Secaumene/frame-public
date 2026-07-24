# Python 绑定:pybind11 扩展模块 _core。
# pybind11 由 cmake/frame_dependencies.cmake 提供(类别 B:FetchContent 锁 tag,自 ADR-0015 起不带 FIND_PACKAGE_ARGS)。
# 铁律 #2:python/ 仅做绑定,核心逻辑一律在 C++ 侧(链接 frame::frame)。
#
# 绑定源码按 PY-002 预判超 500 行直接模块化(M12):module.cpp 为
# PYBIND11_MODULE 入口,聚合 bind_core/bind_graph/bind_ops/bind_runtime/
# bind_interop/bind_autograd/bind_nn/bind_data 八个子绑定单元(bind_interop.cpp:
# ONNX 权重导入/导出,ADR-0013;bind_autograd.cpp:训练 API,反向图/SGD 更新图
# 构建,积压批次②;bind_nn.cpp/bind_data.cpp:frame::nn Module 工厂与
# frame::data TensorDataset/DataLoader,M20 批2 Task5);translate_status.{h,cpp}
# 是 Status -> Python 异常转换的唯一入口(PY-030)。
pybind11_add_module(_core
  src/module.cpp
  src/bind_core.cpp
  src/bind_graph.cpp
  src/bind_ops.cpp
  src/bind_runtime.cpp
  src/bind_interop.cpp
  src/bind_autograd.cpp
  src/bind_nn.cpp
  src/bind_data.cpp
  src/translate_status.cpp
)
# 【TOP-006】链接 frame::compiler_flags(告警/ASAN 统一接线);frame::frame 为聚合库
# (含 WHOLE_ARCHIVE 后端注册,见 src/CMakeLists.txt)。
target_link_libraries(_core PRIVATE frame::frame frame::compiler_flags)

# 安装规则(M12,决议点 G):scikit-build-core 约定——pyproject.toml 的
# `wheel.packages = ["python/frame"]` 把本仓 python/frame/ 映射为 wheel 内的
# 顶层 frame 包目录,故编译产物须落位到该目录下,使 `import frame._core` 在
# 安装后的 wheel 中可解析(与 python/frame/__init__.py 的 `from . import
# _core` 相对导入路径一致)。
# 仅 scikit-build-core 驱动的 wheel 构建才装该扩展(其调用 cmake 时定义
# SKBUILD);系统安装(cmake --install)禁止携带 Python 扩展(BUILD-041,
# docs/standards/build-and-test.md 第 10 节)。
if(SKBUILD)
  install(TARGETS _core DESTINATION frame)
endif()
