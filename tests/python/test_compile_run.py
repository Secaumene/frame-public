"""端到端:构图(matmul+add+relu)-> compile("cpu") -> run ≡ numpy 参考;
二次 compile+run 结果一致(编译缓存路径,见 include/frame/runtime/compile.h
头注释"缓存键计算"一节)。"""

import numpy as np


def _build_matmul_add_relu_graph(core, backend):
    """matmul(x, w) + bias 后 relu,rank-2 x[2,3] @ w[3,4] + bias[2,4]。与
    tests/cpp/backends/test_cuda_backend.cpp::BuildMatmulAddReluGraph 同构
    (同一份 x/w/bias 数值素材,便于交叉核对),Python 侧另建一份是因为图构造
    走的是 Python 绑定 API 而非 C++ ir::Graph 直接调用,非重复实现
    (REUSE-002 意图规避的是"同一语言内的重复算法",跨语言薄壳转发不算)。"""
    graph = core.Graph(f"matmul_add_relu_{backend}")
    x = graph.add_graph_input([2, 3], core.DType.float32, backend=backend)
    w = graph.add_graph_input([3, 4], core.DType.float32, backend=backend)
    bias = graph.add_graph_input([2, 4], core.DType.float32, backend=backend)
    matmul_out = core.matmul(graph, x, w)
    added = core.add(graph, matmul_out, bias)
    relu_out = core.relu(graph, added)
    graph.mark_output(relu_out)
    return graph


def _reference_inputs():
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    w_np = np.array(
        [[0.1, -0.2, 0.3, -0.4], [0.5, -0.6, 0.7, -0.8], [0.9, -1.0, 1.1, -1.2]], dtype=np.float32
    )
    bias_np = np.array([[0.25, -0.5, 0.75, -1.0], [-0.25, 0.5, -0.75, 1.0]], dtype=np.float32)
    return x_np, w_np, bias_np


def test_end_to_end_matmul_add_relu_matches_numpy_reference(core, allclose):
    """matmul+add+relu 整图结果与 numpy 参考(x@w+bias 再 clip 到 0)一致。"""
    graph = _build_matmul_add_relu_graph(core, "cpu")
    executable = core.compile(graph, "cpu")

    x_np, w_np, bias_np = _reference_inputs()
    inputs = [core.from_numpy(x_np), core.from_numpy(w_np), core.from_numpy(bias_np)]
    result = executable.run(inputs)[0].numpy()

    reference = np.maximum(x_np @ w_np + bias_np, 0.0)
    allclose(result, reference, "float32")


def test_second_run_on_same_executable_is_consistent(core, allclose):
    """同一 Executable 二次 run(),两次结果与 numpy 参考一致(run 幂等,不
    因重复调用累积副作用)。"""
    graph = _build_matmul_add_relu_graph(core, "cpu")
    executable = core.compile(graph, "cpu")

    x_np, w_np, bias_np = _reference_inputs()
    inputs = [core.from_numpy(x_np), core.from_numpy(w_np), core.from_numpy(bias_np)]
    reference = np.maximum(x_np @ w_np + bias_np, 0.0)

    first_result = executable.run(inputs)[0].numpy()
    second_result = executable.run(inputs)[0].numpy()
    allclose(first_result, reference, "float32")
    allclose(second_result, reference, "float32")


def test_second_compile_on_same_graph_hits_cache_and_matches_reference(core, allclose):
    """同一 graph 对象二次调用 compile("cpu"):命中 runtime::compile 的编译
    缓存(缓存键 = backend 名 + 编译选项指纹 + dump_text(graph),不因调用
    次数/Executable 对象身份而异,见 include/frame/runtime/compile.h 头注释)
    ——本用例的可观测面是"两次 compile 各自 run() 均与参考一致",不直接断言
    C++ 内部的缓存命中计数(纯 Python 侧无法插桩该私有状态)。"""
    graph = _build_matmul_add_relu_graph(core, "cpu")
    x_np, w_np, bias_np = _reference_inputs()
    inputs = [core.from_numpy(x_np), core.from_numpy(w_np), core.from_numpy(bias_np)]
    reference = np.maximum(x_np @ w_np + bias_np, 0.0)

    first_executable = core.compile(graph, "cpu")
    second_executable = core.compile(graph, "cpu")

    allclose(first_executable.run(inputs)[0].numpy(), reference, "float32")
    allclose(second_executable.run(inputs)[0].numpy(), reference, "float32")


def test_output_tensor_usable_after_executable_garbage_collected(core, allclose):
    """回归(批1-Task4b):临时 Executable 用法 compile(...).run(...) 的输出
    张量,在 Executable 包装对象被回收后必须仍然可用——修复前输出张量的
    device.backend 视图指向包装对象成员字符串,对象回收后视图悬垂,
    numpy()/二次执行按注册表查名报错。"""
    import gc

    graph = _build_matmul_add_relu_graph(core, "cpu")
    x_np, w_np, bias_np = _reference_inputs()
    inputs = [core.from_numpy(x_np), core.from_numpy(w_np), core.from_numpy(bias_np)]

    # 临时 Executable:run 返回后包装对象立即不可达。
    out = core.compile(graph, "cpu").run(inputs)[0]
    gc.collect()
    # 制造堆复用压力,提高修复前悬垂视图被覆写的确定性(修复后无影响)。
    _churn = ["x" * 64 for _ in range(4096)]

    reference = np.maximum(x_np @ w_np + bias_np, 0.0)
    allclose(out.numpy(), reference, "float32")

    # 输出张量二次入图执行(relu 幂等:relu(out) == out)。
    relu_graph = core.Graph("relu_of_out")
    v = relu_graph.add_graph_input([2, 4], core.DType.float32, backend="cpu")
    relu_graph.mark_output(core.relu(relu_graph, v))
    second = core.compile(relu_graph, "cpu")
    allclose(second.run([out])[0].numpy(), reference, "float32")
