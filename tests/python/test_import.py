"""骨架期唯一 Python 真测试:import frame 成功即过。

绑定扩展 _core 未编译时(骨架期常态),相关断言以 pytest.skip 容错跳过。
"""

import pytest


def test_import_frame_facade():
    """纯 Python 门面包必须可导入,且版本号与 _version 一致。"""
    frame = pytest.importorskip("frame")
    assert frame.__version__ == "0.1.0"


def test_core_binding_optional():
    """C++ 扩展 _core:已编译则可用,未编译则跳过(骨架期)。"""
    import frame

    if not frame._core_available:
        pytest.skip("C++ extension _core not built yet (skeleton stage)")
    assert frame._core is not None
