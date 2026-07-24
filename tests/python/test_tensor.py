"""Tensor/DType/Shape/Device 与 numpy 互操作用例(M12 决议点 B/D)。

数值容差经 conftest.py 的 `allclose` fixture 取 BUILD-011(docs/standards/
build-and-test.md 第6章)唯一权威表值,本文件不手写 rtol/atol。
"""

import numpy as np
import pytest


def test_dtype_enum_members(core):
    """DType 白名单值与 .pyi 声明一致(float32=0/float16=2/bfloat16=3/int32=6/
    int64=7,与 include/frame/core/dtype.h::DTypeCode 的枚举序号一致;int32/
    int64 是 M22 批4 决议点A新增,docs/plan/2026-07-19-batch4-m22-seq.md
    §1.1/§3 验收硬门第6条)。"""
    assert int(core.DType.float32) == 0
    assert int(core.DType.float16) == 2
    assert int(core.DType.bfloat16) == 3
    assert int(core.DType.int32) == 6
    assert int(core.DType.int64) == 7


def test_shape_roundtrip_and_repr(core):
    """Shape 与 list[int] 互转;__repr__ 含各维尺寸(非本测试的机械覆盖要求,
    顺带验证)。"""
    shape = core.Shape([2, 3, 4])
    assert shape.dims == [2, 3, 4]
    assert "2" in repr(shape)
    assert "3" in repr(shape)
    assert "4" in repr(shape)


def test_device_backend_and_index_properties(core):
    """Device 构造 + backend/index 只读属性;index 缺省值 0。"""
    device = core.Device("cpu", 1)
    assert device.backend == "cpu"
    assert device.index == 1

    default_index_device = core.Device("cpu")
    assert default_index_device.index == 0


def test_from_numpy_float32_roundtrip(core, allclose):
    """float32 from_numpy -> numpy() 往返数值一致(BUILD-011 fp32 容差)。"""
    array = np.array([[1.0, -2.5, 3.25], [4.0, -5.5, 6.75]], dtype=np.float32)
    tensor = core.from_numpy(array)
    assert isinstance(tensor, core.Tensor)
    allclose(tensor.numpy(), array, "float32")


def test_from_numpy_float16_roundtrip(core, allclose):
    """float16 from_numpy -> numpy() 往返数值一致(BUILD-011 fp16 容差)。"""
    array = np.array([1.5, -2.25, 3.0, 0.0], dtype=np.float16).reshape(2, 2)
    tensor = core.from_numpy(array)
    assert tensor.numpy().dtype == np.float16
    allclose(tensor.numpy(), array, "float16")


def test_tensor_shape_dtype_device_properties(core):
    """Tensor.shape/dtype/device 只读属性与构造输入一致。"""
    array = np.zeros((2, 5), dtype=np.float32)
    tensor = core.from_numpy(array)
    assert tensor.shape.dims == [2, 5]
    assert tensor.dtype == core.DType.float32
    assert tensor.device.backend == "cpu"
    assert tensor.device.index == 0


def test_from_numpy_rejects_non_contiguous_array(core):
    """非 C-contiguous 数组(切片视图)报 ValueError(from_numpy 显式拒绝,
    见 python/src/bind_core.cpp::from_numpy)。"""
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    view = base[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="C-contiguous"):
        core.from_numpy(view)


def test_from_numpy_rejects_unsupported_dtype(core):
    """v0 支持 float32/float16/int32/int64(M22 批4 决议点A扩项);int8/float64
    等一律 ValueError(bfloat16 的 NotImplementedError 边界见
    test_bfloat16_tensor_numpy_raises_not_implemented,numpy 本身无 bfloat16
    dtype,故 from_numpy 侧无法单独区分 bf16,落入本"unsupported numpy dtype"
    通用分支——与 python/src/bind_core.cpp 头注释一致)。"""
    with pytest.raises(ValueError, match="unsupported numpy dtype"):
        core.from_numpy(np.zeros(4, dtype=np.int8))
    with pytest.raises(ValueError, match="unsupported numpy dtype"):
        core.from_numpy(np.zeros(4, dtype=np.float64))


def test_from_numpy_int32_int64_roundtrip(core):
    """int32/int64 numpy 往返(np -> frame tensor -> np)逐字节一致(M22 批4
    §1.1 决议点A + §3 验收硬门第6条:整数 dtype 无损搬运,精确比较无需容差,
    tests/cpp/common/tolerance.h 整数档 rtol=atol=0 同口径)。"""
    array32 = np.array([[1, -2, 3], [4, -5, 6]], dtype=np.int32)
    tensor32 = core.from_numpy(array32)
    assert tensor32.dtype == core.DType.int32
    np.testing.assert_array_equal(tensor32.numpy(), array32)

    array64 = np.array([1, -2, 3, 1 << 40], dtype=np.int64)
    tensor64 = core.from_numpy(array64)
    assert tensor64.dtype == core.DType.int64
    np.testing.assert_array_equal(tensor64.numpy(), array64)


def test_tensor_to_cpu_returns_independent_copy(core, allclose):
    """Tensor.to('cpu') 在同后端场景下仍返回数值一致的独立拷贝(v0 拷贝语义,
    决议点 D)。跨设备搬运(H2D/D2H)由 test_cuda.py 在真实 GPU 上验证。"""
    array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    tensor = core.from_numpy(array)
    moved = tensor.to("cpu")
    assert moved.device.backend == "cpu"
    allclose(moved.numpy(), array, "float32")


def test_bfloat16_tensor_numpy_raises_not_implemented(core, bf16_tensor):
    """bfloat16 numpy 无原生表示,Tensor.numpy() 报 NotImplementedError(决议
    点 D)。bf16_tensor fixture(conftest.py)经 constant 算子物化——numpy 无
    bfloat16 dtype,无法经 from_numpy 直接构造出这个场景。"""
    assert bf16_tensor.dtype == core.DType.bfloat16
    with pytest.raises(NotImplementedError, match="no numpy representation"):
        bf16_tensor.numpy()
