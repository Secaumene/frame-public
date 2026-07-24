"""面向用户算子逐个用例(add/mul/selective_scan/heaviside_surrogate/
scatter_add/relu/square/matmul/sum/constant 等，PY-021 全量执法；M12 决议点 C)
+ Graph/Value/Node 构图入口覆盖。

每个算子:构图 -> compile("cpu") -> run,数值与 numpy 参考比较,容差经
conftest.py 的 `allclose` fixture 取 BUILD-011 唯一权威表值。
"""

import numpy as np
import pytest


def _run_single_output(core, graph, output_value, inputs):
    """构图公共尾段:mark_output 单输出 -> compile("cpu") -> run,返回 numpy
    数组。各算子用例结构高度同构(REUSE-002),抽出本 helper 避免每个测试
    重复相同的四行编排代码;helper 本身不含任何数值计算逻辑,只是调用序列的
    折叠。"""
    graph.mark_output(output_value)
    executable = core.compile(graph, "cpu")
    assert isinstance(executable, core.Executable)
    outputs = executable.run(inputs)
    return outputs[0].numpy()


def test_add(core, allclose):
    """逐元素加法,v0 无广播,shape 相同。"""
    graph = core.Graph("add_graph")
    lhs = graph.add_graph_input([2, 3], core.DType.float32)
    rhs = graph.add_graph_input([2, 3], core.DType.float32)
    assert isinstance(lhs, core.Value)
    out = core.add(graph, lhs, rhs)
    assert isinstance(out, core.Value)

    lhs_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    rhs_np = np.array([[0.5, -0.5, 1.5], [-1.5, 2.5, -2.5]], dtype=np.float32)
    result = _run_single_output(core, graph, out, [core.from_numpy(lhs_np), core.from_numpy(rhs_np)])
    allclose(result, lhs_np + rhs_np, "float32")


def test_mul(core, allclose):
    """逐元素乘法。"""
    graph = core.Graph("mul_graph")
    lhs = graph.add_graph_input([2, 2], core.DType.float32)
    rhs = graph.add_graph_input([2, 2], core.DType.float32)
    out = core.mul(graph, lhs, rhs)

    lhs_np = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    rhs_np = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    result = _run_single_output(core, graph, out, [core.from_numpy(lhs_np), core.from_numpy(rhs_np)])
    allclose(result, lhs_np * rhs_np, "float32")


def test_selective_scan(core, allclose):
    """五输入直接 op 绑定:构图后沿末轴执行手算状态递推。"""
    graph = core.Graph("selective_scan_graph")
    values = [graph.add_graph_input([3], core.DType.float32) for _ in range(5)]
    out = core.selective_scan(graph, *values)
    assert isinstance(out, core.Value)

    inputs_np = [
        np.array([1.0, 2.0, -1.0], dtype=np.float32),
        np.array([0.5, 0.25, -0.5], dtype=np.float32),
        np.array([2.0, 1.0, 3.0], dtype=np.float32),
        np.array([1.0, 2.0, 0.5], dtype=np.float32),
        np.array([0.1, -1.0, 2.0], dtype=np.float32),
    ]
    tensors = [core.from_numpy(value) for value in inputs_np]
    result = _run_single_output(core, graph, out, tensors)
    allclose(result, np.array([2.1, 3.0, -4.125], dtype=np.float32), "float32")


def test_heaviside_surrogate(core, allclose):
    """M27 代理阶跃薄绑定:前向仍是精确阶跃，-0/+0 均归正支。"""
    graph = core.Graph("heaviside_surrogate_graph")
    x = graph.add_graph_input([5], core.DType.float32)
    out = core.heaviside_surrogate(graph, x, 2.0)
    assert isinstance(out, core.Value)
    values = np.array([-1.0, -0.0, 0.0, 0.25, 2.0], dtype=np.float32)
    result = _run_single_output(core, graph, out, [core.from_numpy(values)])
    allclose(result, np.array([0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32), "float32")


def test_heaviside_surrogate_rejects_invalid_alpha(core):
    """Python 入口原样暴露 alpha fail-loud 合同。"""
    graph = core.Graph("heaviside_surrogate_invalid_alpha")
    x = graph.add_graph_input([2], core.DType.float32)
    with pytest.raises(ValueError):
        core.heaviside_surrogate(graph, x, 0.0)


def test_scatter_add(core, allclose):
    """M28 公共散加绑定:重复 int64 索引累加并保留空输出行。"""
    graph = core.Graph("scatter_add_graph")
    updates = graph.add_graph_input([3, 2], core.DType.float32)
    indices = graph.add_graph_input([3], core.DType.int64)
    out = core.scatter_add(graph, updates, indices, [4, 2])
    updates_np = np.array([[10.0, 20.0], [1.0, 2.0], [100.0, 200.0]], dtype=np.float32)
    indices_np = np.array([2, 0, 2], dtype=np.int64)
    result = _run_single_output(
        core, graph, out, [core.from_numpy(updates_np), core.from_numpy(indices_np)]
    )
    expected = np.array([[1.0, 2.0], [0.0, 0.0], [110.0, 220.0], [0.0, 0.0]], dtype=np.float32)
    allclose(result, expected, "float32")


def test_relu(core, allclose):
    """逐元素 ReLU:含正负与零值,验证分段行为。"""
    graph = core.Graph("relu_graph")
    x = graph.add_graph_input([2, 3], core.DType.float32)
    out = core.relu(graph, x)

    x_np = np.array([[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]], dtype=np.float32)
    result = _run_single_output(core, graph, out, [core.from_numpy(x_np)])
    allclose(result, np.maximum(x_np, 0.0), "float32")


def test_square(core, allclose):
    """逐元素平方(cpu kernel;不依赖 mul 分解产物同样的数值结果)。"""
    graph = core.Graph("square_graph")
    x = graph.add_graph_input([3], core.DType.float32)
    out = core.square(graph, x)

    x_np = np.array([-2.0, 0.0, 3.5], dtype=np.float32)
    result = _run_single_output(core, graph, out, [core.from_numpy(x_np)])
    allclose(result, x_np**2, "float32")


def test_matmul(core, allclose):
    """矩阵乘法(rank-2):[m,k] x [k,n] -> [m,n]。"""
    graph = core.Graph("matmul_graph")
    lhs = graph.add_graph_input([2, 3], core.DType.float32)
    rhs = graph.add_graph_input([3, 4], core.DType.float32)
    out = core.matmul(graph, lhs, rhs)

    lhs_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    rhs_np = np.array(
        [[0.1, -0.2, 0.3, -0.4], [0.5, -0.6, 0.7, -0.8], [0.9, -1.0, 1.1, -1.2]], dtype=np.float32
    )
    result = _run_single_output(core, graph, out, [core.from_numpy(lhs_np), core.from_numpy(rhs_np)])
    allclose(result, lhs_np @ rhs_np, "float32")


def test_sum_full_reduction_with_empty_axes(core, allclose):
    """sum 的 axes 为空数组 = 全维归约(design-reviewer 决议,m5-design-brief
    决议点 3),输出 shape 应为 []。"""
    graph = core.Graph("sum_full_graph")
    x = graph.add_graph_input([2, 3], core.DType.float32)
    out = core.sum(graph, x, [])

    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    graph.mark_output(out)
    executable = core.compile(graph, "cpu")
    output_tensor = executable.run([core.from_numpy(x_np)])[0]
    assert output_tensor.shape.dims == []
    allclose(output_tensor.numpy(), np.sum(x_np), "float32")


def test_sum_axis_reduction(core, allclose):
    """sum 沿单一 axis 归约,keepdims 缺省 False,消去该维。"""
    graph = core.Graph("sum_axis_graph")
    x = graph.add_graph_input([2, 3], core.DType.float32)
    out = core.sum(graph, x, [1])

    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    result = _run_single_output(core, graph, out, [core.from_numpy(x_np)])
    assert result.shape == (2,)
    allclose(result, np.sum(x_np, axis=1), "float32")


def test_mse_loss(core, allclose):
    """均方误差损失(M17,ARCH-064):mean((pred-target)**2),标量(rank-0)
    输出。"""
    graph = core.Graph("mse_loss_graph")
    pred = graph.add_graph_input([2, 3], core.DType.float32)
    target = graph.add_graph_input([2, 3], core.DType.float32)
    out = core.mse_loss(graph, pred, target)
    assert isinstance(out, core.Value)

    pred_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    target_np = np.array([[1.5, 1.5, 2.5], [3.5, 6.0, 5.0]], dtype=np.float32)

    graph.mark_output(out)
    executable = core.compile(graph, "cpu")
    output_tensor = executable.run([core.from_numpy(pred_np), core.from_numpy(target_np)])[0]
    assert output_tensor.shape.dims == []
    allclose(output_tensor.numpy(), np.mean((pred_np - target_np) ** 2), "float32")


def test_constant_float32(core, allclose):
    """constant:0 输入节点,把常量值物化进图(dtype=float32)。"""
    graph = core.Graph("constant_f32_graph")
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    out = core.constant(graph, values, [2, 3], core.DType.float32, backend="cpu")
    result = _run_single_output(core, graph, out, [])
    allclose(result, np.array(values, dtype=np.float32).reshape(2, 3), "float32")


def test_constant_float16(core, allclose):
    """constant dtype=float16(fill_tensor_from_constant_attrs 的 double ->
    fp16 转换路径,与 float32 分支不同代码路径,单独覆盖)。"""
    graph = core.Graph("constant_f16_graph")
    values = [0.5, -1.25, 2.0]
    out = core.constant(graph, values, [3], core.DType.float16, backend="cpu")
    result = _run_single_output(core, graph, out, [])
    assert result.dtype == np.float16
    allclose(result, np.array(values, dtype=np.float16), "float16")


def test_graph_name_property(core):
    """Graph.name 只读属性与构造实参一致。"""
    graph = core.Graph("my_named_graph")
    assert graph.name == "my_named_graph"

    default_named_graph = core.Graph()
    assert default_named_graph.name == ""


def test_node_type_is_exported_but_currently_unreachable(core):
    """Node 类已导出(python/src/bind_graph.cpp 注册 py::class_<ir::Node>),
    但当前公开算子绑定与 add_graph_input 均只返回 Value(见 bind_ops.cpp/
    bind_graph.cpp),且 Node 未绑定任何构造函数——纯 Python 侧没有任何途径
    获得一个 Node 实例。本用例只能验证符号本身存在;深层行为覆盖不可达,
    记入 test-writer 报告的"疑似实现问题"(而非本测试放宽验收线)。"""
    assert hasattr(core, "Node")
    assert isinstance(core.Node, type)
    with pytest.raises(TypeError):
        core.Node()
