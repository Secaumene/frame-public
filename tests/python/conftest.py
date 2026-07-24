"""pytest 配置:确保开发期可直接 import frame(纯 Python 门面包),并提供
tests/python/ 下全部 test_*.py 共用的 fixture(BUILD-020)。

共用逻辑集中于本文件而非另建帮助模块的原因:BUILD-020 判定方法要求
`tests/python/` 下除 `conftest.py` 外的全部 `*.py` 文件名匹配 `^test_.*\\.py$`
(docs/standards/build-and-test.md 第7章),`conftest.py` 是唯一豁免项,故公共
fixture/常量只能放在这里,不新建 `_helpers.py` 之类的第二个豁免入口。
"""

import gc
import os
import sys

import numpy as np
import pytest

# tests/python/ -> 仓库根 -> python/
_here = os.path.dirname(os.path.abspath(__file__))
_python_root = os.path.abspath(os.path.join(_here, "..", "..", "python"))
if _python_root not in sys.path:
    sys.path.insert(0, _python_root)


# 数值容差:全仓库唯一权威来源 = BUILD-011(docs/standards/build-and-test.md
# 第6章);C++ 侧经 tests/cpp/common/ 容差工具消费同一张表,Python 侧无同构
# 工具可复用(该工具是 C++ 头文件),故在此照抄同值一份,供下方 allclose
# fixture 与全部 test_*.py 共用,不在各测试文件里各自手写阈值。
_BUILD_011_TOLERANCE = {
    "float32": {"rtol": 1e-5, "atol": 1e-6},
    "float16": {"rtol": 1e-2, "atol": 1e-3},
    "bfloat16": {"rtol": 2e-2, "atol": 2e-3},
}


@pytest.fixture(scope="session")
def core():
    """frame._core 扩展模块;未编译时按 BUILD-020 SKIP 政策整会话统一跳过
    (英文原因),不视为测试失败——与既有 test_import.py 的骨架期容错口径一致。
    """
    frame = pytest.importorskip("frame")
    if not frame._core_available:
        pytest.skip(
            "frame._core C++ extension is not built (FRAME_BUILD_PYTHON=OFF, "
            "or the package was not installed via `pip install -e .`)"
        )
    from frame import _core

    return _core


@pytest.fixture(scope="session")
def nn(core):
    """frame.nn 子命名空间(M20 批2 Task5,python/frame/nn.py 门面);依赖 core
    fixture 已触发的 skip-if-unavailable 逻辑,不重复判断。
    """
    del core  # 仅借其 skip 时序,自身不直接使用
    import frame

    return frame.nn


@pytest.fixture(scope="session")
def data(core):
    """frame.data 子命名空间(M20 批2 Task5,python/frame/data.py 门面);依赖
    core fixture 已触发的 skip-if-unavailable 逻辑,不重复判断。
    """
    del core  # 仅借其 skip 时序,自身不直接使用
    import frame

    return frame.data


@pytest.fixture(scope="session")
def allclose():
    """按 BUILD-011 数值容差表(docs/standards/build-and-test.md 第6章,唯一
    权威来源,本文件顶部 `_BUILD_011_TOLERANCE` 照抄同值并注明来源)断言
    numpy 数组数值一致。dtype_name 取值 {"float32","float16","bfloat16"}。
    """

    def _assert_allclose(actual, expected, dtype_name):
        tol = _BUILD_011_TOLERANCE[dtype_name]
        np.testing.assert_allclose(actual, expected, rtol=tol["rtol"], atol=tol["atol"])

    return _assert_allclose


@pytest.fixture(scope="session")
def cuda_available(core):
    """cuda 后端是否已在本次安装中注册。wheel 构建期 `FRAME_ENABLE_CUDA=AUTO`
    (pyproject.toml cmake.args 未显式覆盖,见 docs/standards/build-and-test.md
    BUILD-004)只有探测到 CUDA Toolkit >= 12.0 才会把 cuda 后端链入
    `frame._core`;用 `Device("cuda", 0)` 构造探测——与
    python/src/bind_core.cpp::make_device 走同一条 BackendRegistry 查找路径,
    未注册时该调用抛 KeyError(PY-030 kNotFound 映射)。
    """
    try:
        core.Device("cuda", 0)
        return True
    except KeyError:
        return False


@pytest.fixture(scope="session")
def bf16_tensor(core):
    """一份 dtype=bfloat16 的 Tensor。numpy 无 bfloat16 原生 dtype,无法经
    `from_numpy` 构造(决议点 D 边界),只能借道构图→编译→执行路径,用
    `constant` 算子物化(v0 白名单含 bfloat16,见
    src/ops/schemas/constant.cpp)。会话级共享:该 Tensor 只读使用
    (`.numpy()` 触发错误、`.dtype` 读属性),多个测试文件复用同一份不会互相
    污染。
    """
    graph = core.Graph("bf16_tensor_fixture")
    value = core.constant(graph, [1.0, 2.0], [2], core.DType.bfloat16, backend="cpu")
    graph.mark_output(value)
    executable = core.compile(graph, "cpu")
    return executable.run([])[0]


@pytest.fixture()
def collect_garbage():
    """跨图句柄生命周期测试用:`del` 之后强制立即回收,排除引用计数延迟造成
    的假阴性(CPython 的引用计数本应立即回收,`gc.collect()` 仅用于兜底排除
    循环引用等边缘情况,使测试结论更稳)。"""
    return gc.collect
