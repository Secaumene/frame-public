"""Frame:编译优先的深度学习框架(骨架期)。

本包为薄层纯 Python 门面(铁律 #2):计算/调度/内存管理全部在 C++ 核心,
Python 侧仅做 pybind11 绑定与门面转发。C++ 扩展模块为 frame._core。
"""

from __future__ import annotations

from ._version import __version__

# 骨架期容错:_core 扩展可能尚未编译(例如未启用 FRAME_BUILD_PYTHON,或仅跑纯
# Python 工具)。此时门面仍可导入,依赖 _core 的功能留待调用点再行报错。
try:
    from . import _core  # noqa: F401  C++ 扩展产物,导出核心绑定
    _core_available = True
except ImportError:
    _core = None  # type: ignore[assignment]
    _core_available = False

# M12:构图→编译→执行最小闭环的再导出(七算子 + Graph/Tensor/DType/compile/
# from_numpy,PY-021 暂缓期清单销项)。M17 追加 mse_loss(ARCH-064,PY-021
# 自 M12 起全量执法,随注册同变更绑定)。ADR-0013 追加
# save_onnx_weights/load_onnx_weights(ONNX 权重导入/导出,非 OpRegistry
# 算子、不受 PY-021 约束,再导出纯为用户可用性)。积压批次②追加
# build_backward_graph/build_sgd_update_graph(训练 API,同样非 OpRegistry 算子、
# 不受 PY-021 约束,再导出纯为用户可用性)。_core 缺失时不导出这些名字,依上方
# _core_available 容错策略——调用点访问 `frame.add` 等会得到常规
# AttributeError,而非在导入期硬失败(PY-041)。
if _core_available:
    from ._core import (  # noqa: A004  compile/sum 有意再导出,与内置同名(numpy 同款先例)
        DType,
        Graph,
        Tensor,
        add,
        build_backward_graph,
        build_sgd_update_graph,
        compile,
        constant,
        from_numpy,
        load_onnx_weights,
        matmul,
        mse_loss,
        mul,
        relu,
        save_onnx_weights,
        square,
        sum,
    )

    # M20 批2 Task5:frame.nn / frame.data 子命名空间门面(纯转发,零逻辑,
    # docs/architecture/nn-design.md §6)——底层 pybind11 绑定(bind_nn.cpp/
    # bind_data.cpp)按既有 bind_* 惯例平铺注册进 _core,子命名空间外观由本包
    # nn.py/data.py 两个薄门面模块整理(PY-003)。同上,_core 缺失时不导入,
    # 显式 `import frame.nn`/`frame.data` 会得到 ImportError,而非在顶层
    # `import frame` 期硬失败(PY-041)。
    from . import data as data  # noqa: F401
    from . import nn as nn  # noqa: F401

__all__ = ["__version__", "_core", "_core_available"]
if _core_available:
    __all__ += [
        "DType",
        "Graph",
        "Tensor",
        "add",
        "build_backward_graph",
        "build_sgd_update_graph",
        "compile",
        "constant",
        "data",
        "from_numpy",
        "load_onnx_weights",
        "matmul",
        "mse_loss",
        "mul",
        "nn",
        "relu",
        "save_onnx_weights",
        "square",
        "sum",
    ]
