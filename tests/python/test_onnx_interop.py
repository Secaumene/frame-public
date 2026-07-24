"""ONNX 权重交换 pytest(ADR-0013 判定③:用户可用性 + 生态互操作实证)。
① Python 侧 save→load 往返(numpy 数值比对);
② onnx.checker 校验导出文件合法(pytest.importorskip("onnx"),包缺失时按
   既有 skip 口径整会话跳过,与 conftest.py::core fixture 同款容错);
③ onnx 侧读 initializer 数值(onnx.numpy_helper)与 frame 侧一致。
"""

import numpy as np
import pytest


def test_save_load_roundtrip_float32(core, allclose, tmp_path):
    """float32 权重 save→load 往返,numpy 数值一致(BUILD-011 fp32 容差)。"""
    w1 = np.array([[1.0, -2.0], [3.0, 4.5]], dtype=np.float32)
    b1 = np.array([0.5, -0.5, 1.5], dtype=np.float32)
    path = str(tmp_path / "weights_fp32.onnx")

    core.save_onnx_weights(path, {"w1": core.from_numpy(w1), "b1": core.from_numpy(b1)})
    loaded = core.load_onnx_weights(path)

    assert set(loaded.keys()) == {"w1", "b1"}
    allclose(loaded["w1"].numpy(), w1, "float32")
    allclose(loaded["b1"].numpy(), b1, "float32")


def test_save_load_roundtrip_float16(core, allclose, tmp_path):
    """float16 权重 save→load 往返(BUILD-011 fp16 容差)。"""
    w = np.array([1.5, -2.25, 3.0, 0.0], dtype=np.float16).reshape(2, 2)
    path = str(tmp_path / "weights_fp16.onnx")

    core.save_onnx_weights(path, {"w": core.from_numpy(w)})
    loaded = core.load_onnx_weights(path)

    assert loaded["w"].dtype == core.DType.float16
    allclose(loaded["w"].numpy(), w, "float16")


def test_load_missing_file_raises_key_error(core, tmp_path):
    """文件不存在 -> C++ 侧 kNotFound -> KeyError(PY-030 映射表)。"""
    path = str(tmp_path / "does_not_exist.onnx")
    with pytest.raises(KeyError):
        core.load_onnx_weights(path)


def test_save_load_preserves_insertion_order(core, tmp_path):
    """save_onnx_weights 按 dict 插入序写出,load_onnx_weights 返回顺序与之
    一致(CPython 3.7+ dict 保序语言保证,C++ 侧按文件出现顺序读回)。"""
    values = {
        "first": np.array([1.0], dtype=np.float32),
        "second": np.array([2.0], dtype=np.float32),
        "third": np.array([3.0], dtype=np.float32),
    }
    path = str(tmp_path / "ordered.onnx")
    core.save_onnx_weights(path, {name: core.from_numpy(v) for name, v in values.items()})
    loaded = core.load_onnx_weights(path)
    assert list(loaded.keys()) == ["first", "second", "third"]


def test_onnx_checker_accepts_exported_file(core, tmp_path):
    """② onnx.checker 校验导出文件合法(ADR-0013 待查证 (c) 实证销项)。"""
    onnx = pytest.importorskip("onnx")
    w = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    path = str(tmp_path / "checker.onnx")
    core.save_onnx_weights(path, {"w": core.from_numpy(w)})

    model = onnx.load(path)
    onnx.checker.check_model(model)


def test_onnx_side_reads_matching_initializer_values(core, allclose, tmp_path):
    """③ onnx 侧(onnx.numpy_helper)读出的 initializer 数值与 frame 侧一致。"""
    onnx = pytest.importorskip("onnx")
    from onnx import numpy_helper

    w = np.array([[1.5, -2.5, 3.5], [4.5, -5.5, 6.5]], dtype=np.float32)
    path = str(tmp_path / "values.onnx")
    core.save_onnx_weights(path, {"w": core.from_numpy(w)})

    model = onnx.load(path)
    assert len(model.graph.initializer) == 1
    initializer = model.graph.initializer[0]
    assert initializer.name == "w"
    onnx_array = numpy_helper.to_array(initializer)
    allclose(onnx_array, w, "float32")
