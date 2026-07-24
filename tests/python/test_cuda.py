"""cuda 后端可选端到端用例(M12 决议点 D/E;GTEST 同款 SKIP 口径,
BUILD-010/BUILD-020,M24 政策)。

`cuda_available` fixture(conftest.py)经 `Device("cuda", 0)` 探测本次安装是否
链入了 cuda 后端(wheel 构建期 `FRAME_ENABLE_CUDA=AUTO` 探测到 CUDA Toolkit
时才会启用,见 docs/standards/build-and-test.md BUILD-004)。无卡/未探测到
CUDA Toolkit 的环境按此 skip,原因为英文;**目标后端可用的环境不得 SKIP**
(M24 政策),故本文件在探测到 cuda 后端的机器上必须真实执行本用例。
"""

import numpy as np
import pytest


def test_cuda_from_numpy_to_device_run_matches_cpu_reference(core, cuda_available, allclose):
    """from_numpy -> to("cuda") -> 构图(device=cuda)compile("cuda") -> run ->
    numpy() 与同一图在 cpu 后端执行的结果一致(D2H 拷出后比较)。"""
    if not cuda_available:
        pytest.skip("no CUDA backend registered in this frame._core build (CUDA Toolkit not detected)")

    def build_graph(backend):
        graph = core.Graph(f"cuda_vs_cpu_{backend}")
        lhs = graph.add_graph_input([2, 3], core.DType.float32, backend=backend)
        rhs = graph.add_graph_input([2, 3], core.DType.float32, backend=backend)
        out = core.relu(graph, core.add(graph, lhs, rhs))
        graph.mark_output(out)
        return graph

    lhs_np = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float32)
    rhs_np = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=np.float32)

    # cpu 参考路径:同一图结构、纯 cpu 后端执行。
    cpu_executable = core.compile(build_graph("cpu"), "cpu")
    cpu_result = cpu_executable.run([core.from_numpy(lhs_np), core.from_numpy(rhs_np)])[0].numpy()

    # cuda 路径:显式 H2D 搬运(to("cuda"))后在 cuda 设备上执行。
    cuda_executable = core.compile(build_graph("cuda"), "cuda")
    lhs_cuda = core.from_numpy(lhs_np).to("cuda")
    rhs_cuda = core.from_numpy(rhs_np).to("cuda")
    assert lhs_cuda.device.backend == "cuda"

    cuda_output = cuda_executable.run([lhs_cuda, rhs_cuda])[0]
    assert cuda_output.device.backend == "cuda"
    cuda_result = cuda_output.numpy()  # D2H 拷出(v0 拷贝语义)

    # 与直接用 numpy 计算的公式参考比对(独立于 cpu 后端实现的第二条证据)。
    formula_reference = np.maximum(lhs_np + rhs_np, 0.0)
    allclose(cuda_result, formula_reference, "float32")
    # 与同图 cpu 执行结果比对(任务验收口径:numpy() ≡ cpu 结果)。
    allclose(cuda_result, cpu_result, "float32")
