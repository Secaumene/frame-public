"""错误映射逐个异常(PY-030,translate_status 唯一转换点)+ 跨图句柄误用的
生命周期安全性(裁决修订 2 的 keep_alive 契约)。"""

import numpy as np
import pytest


def test_unregistered_backend_name_raises_key_error(core):
    """未注册后端名 -> KeyError(Status kNotFound,BackendRegistry.get 报错;
    PY-030 映射表)。覆盖代表性的四个调用点:Device 构造、add_graph_input、
    compile、Tensor.to。

    compile() 取空图(zero graph_input、zero 节点)而非带输入的图:
    `runtime::compile` 会先校验 backend_name 与图的 device 后端是否一致
    (include/frame/runtime/compile.h 头注释"①"),图非空时不一致先报
    ValueError("does not match graph device backend"),根本走不到更深处的
    BackendRegistry::get——而 add_graph_input 本身就要求 backend 已注册
    (否则立即 KeyError),所以带节点的图不可能出现"图 device 已知但该
    backend 未注册"的组合。只有空图跳过该一致性校验(头注释"图无算子节点、
    无从判定 device 时跳过本校验"),才能让 backend_name 一路传到
    BackendRegistry::get 触发本测试要断言的 KeyError。"""
    with pytest.raises(KeyError):
        core.Device("no-such-backend")

    graph = core.Graph("bad_backend_input")
    with pytest.raises(KeyError):
        graph.add_graph_input([2], core.DType.float32, backend="no-such-backend")

    empty_graph = core.Graph("empty_for_compile")
    with pytest.raises(KeyError):
        core.compile(empty_graph, "no-such-backend")

    tensor = core.from_numpy(np.zeros(2, dtype=np.float32))
    with pytest.raises(KeyError):
        tensor.to("no-such-backend")


def test_compile_rejects_backend_name_mismatching_graph_device(core):
    """compile() 自身的第一道校验:backend_name 实参与图的 device 后端不一致
    (但均已注册)-> ValueError,而非试图用 backend_name 编译一个设备不匹配
    的图(include/frame/runtime/compile.h 头注释"①")。与上一个用例的空图
    KeyError 分支互补,合起来覆盖 compile() 的两条错误路径。"""
    graph = core.Graph("mismatched_backend")
    x = graph.add_graph_input([2], core.DType.float32, backend="cpu")
    graph.mark_output(x)
    # 用明显不是任何已注册后端名的字符串:该一致性校验是纯字符串比较
    # (backend_name 实参 vs 图里已固化的 device 后端名),与 backend_name 自身
    # 是否已注册无关,故取值不依赖本机实际启用了哪些后端,保证跨机器可移植。
    with pytest.raises(ValueError, match="does not match graph device backend"):
        core.compile(graph, "totally-bogus-backend-name")


def test_elementwise_shape_mismatch_raises_value_error(core):
    """add/mul 等二元逐元素算子 v0 无广播,shape 不一致 -> ValueError(Status
    kInvalidArgument,src/ops/schemas/elementwise.cpp 的
    infer_binary_elementwise_shape)。"""
    graph = core.Graph("bad_elementwise_shape")
    lhs = graph.add_graph_input([2, 2], core.DType.float32)
    rhs = graph.add_graph_input([3, 3], core.DType.float32)
    with pytest.raises(ValueError, match="same shape"):
        core.add(graph, lhs, rhs)


def test_matmul_contraction_dimension_mismatch_raises_value_error(core):
    """matmul 收缩维不一致 -> ValueError(src/ops/schemas/matmul.cpp)。"""
    graph = core.Graph("bad_matmul_shape")
    lhs = graph.add_graph_input([2, 3], core.DType.float32)
    rhs = graph.add_graph_input([4, 5], core.DType.float32)
    with pytest.raises(ValueError, match="contraction dimension mismatch"):
        core.matmul(graph, lhs, rhs)


def test_sum_axis_out_of_range_raises_value_error(core):
    """sum 的 axes 条目越界(不做负索引归一化,v0 显式优于隐式)-> ValueError
    (src/ops/schemas/reduction.cpp::infer_sum_shape)。"""
    graph = core.Graph("bad_sum_axis")
    x = graph.add_graph_input([2, 3], core.DType.float32)
    with pytest.raises(ValueError, match="out of range"):
        core.sum(graph, x, [5])


def test_bfloat16_numpy_raises_not_implemented_error(bf16_tensor):
    """bfloat16 Tensor.numpy() -> NotImplementedError(Status kUnimplemented,
    PY-030 映射表;bf16_tensor fixture 见 conftest.py)。"""
    with pytest.raises(NotImplementedError, match="no numpy representation"):
        bf16_tensor.numpy()


def test_cross_graph_value_handle_raises_instead_of_crashing(core, collect_garbage):
    """跨图句柄误用 + `del` 原图后仍持有句柄的生命周期安全性,两点合一验证
    (裁决修订 2:`py::return_value_policy::reference_internal` 等价
    `keep_alive<0,1>`,把返回的 Value 句柄反向拴住其所属 Graph 的 Python
    包装对象)。

    取舍(按任务要求写明,未另起子进程做隔离式崩溃探测):
    1) `reference_internal` 保证只要 Python 侧还持有 `value`,`graph` 的
       Python 包装对象就不会被析构;因此 `del graph` 之后经由 `value` 触碰
       底层 C++ 对象(哪怕只是走到"value 不属于该图"这条校验分支)读取的
       仍是合法内存,不是 UAF —— 能拿到一个内容良好的 `ValueError`(而非
       段错误或垃圾数据)本身就是该生命周期契约成立的证据。
    2) 若该契约被破坏(真悬垂指针),这里会发生的是进程级 segfault 而非可
       捕获的 Python 异常;那种情况下 try/except 完全捕获不到任何东西,整
       个 pytest 进程会被信号直接杀死——这本身就是可观测的失败信号(CI 上
       表现为该测试"异常终止"而非常规"FAIL"),子进程隔离能多提供的只是
       "崩溃不拖累同一进程内其余用例继续运行",不能提供本方法探测不到的
       额外能力;为保持简单直接在本进程内断言。
    """
    graph = core.Graph("will_be_deleted")
    value = graph.add_graph_input([2, 2], core.DType.float32)
    del graph
    collect_garbage()

    other_graph = core.Graph("other_graph")
    with pytest.raises(ValueError, match="does not belong to this graph"):
        other_graph.mark_output(value)
    with pytest.raises(ValueError, match="does not belong to this graph"):
        core.add(other_graph, value, value)

    # 进程仍存活、后续断言仍可正常求值 —— 上面两处异常均为可捕获的 ValueError
    # 而非进程崩溃,是本用例的核心判据。
    assert True
