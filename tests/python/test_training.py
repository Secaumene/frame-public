"""训练 API(build_backward_graph/build_sgd_update_graph)pytest(积压批次②,
python/src/bind_autograd.cpp;权威契约见 include/frame/compiler/autograd.h、
docs/architecture/autograd.md 第2/5/6章)。

1. build_sgd_update_graph 单参数一步更新数值 ≡ 手算 p - lr·g(numpy 对照)。
2. python 侧端到端小训练循环:复用 test_compile_run.py 风格构 MLP,固定种子
   numpy 数据,build_backward_graph + build_sgd_update_graph 各 compile 一次,
   循环内只调 run(),不重复 compile(docs/architecture/autograd.md 第6章
   训练循环形态)。
3. M26 多输出 forward 契约与错误映射(PY-030 映射表):原输出前缀保留、
   wrt 梯度追加/越界下标/非有限学习率。
4. 训练后 save_onnx_weights -> load_onnx_weights -> 推理一致(ADR-0013 串联,
   C++ 侧已有 tests/cpp/interop/test_onnx_weights.cpp 对应链路,本用例补
   python 侧一例)。
5. fp16 训练链路数值证据(M19 Task 4):matmul->mse_loss 图全程 dtype=
   float16,build_backward_graph + build_sgd_update_graph 各 compile 一次,
   循环 30 步只调 run(),断言 loss 有限且末值 < 首值、参数/梯度 dtype 全程
   保持 float16。bf16 无 numpy 桥(numpy 无 bfloat16 原生 dtype),本文件不加
   bf16 用例——对应数值证据在
   tests/cpp/compiler/test_training_loop.cpp。
"""

import numpy as np
import pytest

# 网络维度(小规模,加速测试执行,v0 无广播故不带 bias,见 spec 交付要求):
# x[4,2] -> matmul(w1[2,4]) -> relu -> matmul(w2[4,1]) -> mse_loss(., target[4,1])。
_BATCH = 4
_IN_DIM = 2
_HIDDEN_DIM = 4
_OUT_DIM = 1


def _build_mlp_forward_graph(core, backend="cpu"):
    """forward.inputs() 按位 = [x, w1, w2, target];本训练 helper 主动只把
    loss 标为输出,故派生图仍保持既有 [loss, grad...] 布局。"""
    graph = core.Graph("mlp_forward")
    x = graph.add_graph_input([_BATCH, _IN_DIM], core.DType.float32, backend=backend)
    w1 = graph.add_graph_input([_IN_DIM, _HIDDEN_DIM], core.DType.float32, backend=backend)
    w2 = graph.add_graph_input([_HIDDEN_DIM, _OUT_DIM], core.DType.float32, backend=backend)
    target = graph.add_graph_input([_BATCH, _OUT_DIM], core.DType.float32, backend=backend)
    hidden = core.relu(graph, core.matmul(graph, x, w1))
    pred = core.matmul(graph, hidden, w2)
    loss = core.mse_loss(graph, pred, target)
    graph.mark_output(loss)
    return graph


def _mlp_training_material(seed=20260713):
    """固定种子生成 x/target/初始 w1/w2(numpy),target 取一个同架构"教师网络"
    的输出(relu(x@w1_true)@w2_true),保证学生网络可学习、训练曲线稳定下降。"""
    rng = np.random.default_rng(seed)
    x_np = rng.uniform(-1.0, 1.0, size=(_BATCH, _IN_DIM)).astype(np.float32)
    w1_true = rng.uniform(-1.0, 1.0, size=(_IN_DIM, _HIDDEN_DIM)).astype(np.float32)
    w2_true = rng.uniform(-1.0, 1.0, size=(_HIDDEN_DIM, _OUT_DIM)).astype(np.float32)
    target_np = np.maximum(x_np @ w1_true, 0.0) @ w2_true
    w1_np = rng.uniform(-0.3, 0.3, size=(_IN_DIM, _HIDDEN_DIM)).astype(np.float32)
    w2_np = rng.uniform(-0.3, 0.3, size=(_HIDDEN_DIM, _OUT_DIM)).astype(np.float32)
    return x_np, target_np, w1_np, w2_np


def test_sgd_single_param_one_step_matches_manual_computation(core, allclose):
    """单参数一步更新数值 ≡ 手算 p - lr·g(BUILD-011 float32 容差)。"""
    learning_rate = 0.1
    update_graph = core.build_sgd_update_graph([([2, 3], core.DType.float32)], learning_rate)
    executable = core.compile(update_graph, "cpu")

    p_np = np.array([[1.0, 2.0, -3.0], [0.5, -0.5, 4.0]], dtype=np.float32)
    g_np = np.array([[0.1, -0.2, 0.3], [1.0, -1.0, 2.0]], dtype=np.float32)
    outputs = executable.run([core.from_numpy(p_np), core.from_numpy(g_np)])

    allclose(outputs[0].numpy(), p_np - learning_rate * g_np, "float32")


def test_mlp_training_loop_converges(core):
    """python 侧端到端小训练循环:build_backward_graph + build_sgd_update_graph
    各 compile 一次,循环 100 步只调 run(),末 loss < 首 loss * 0.2 且全程有限。
    """
    x_np, target_np, w1_np, w2_np = _mlp_training_material()

    forward = _build_mlp_forward_graph(core)
    training = core.build_backward_graph(forward, 0, [1, 2])
    train_executable = core.compile(training, "cpu")

    update = core.build_sgd_update_graph(
        [([_IN_DIM, _HIDDEN_DIM], core.DType.float32), ([_HIDDEN_DIM, _OUT_DIM], core.DType.float32)],
        0.1,
        backend="cpu",
    )
    update_executable = core.compile(update, "cpu")

    x_t = core.from_numpy(x_np)
    target_t = core.from_numpy(target_np)
    w1_t = core.from_numpy(w1_np)
    w2_t = core.from_numpy(w2_np)

    losses = []
    for _ in range(100):
        train_outputs = train_executable.run([x_t, w1_t, w2_t, target_t])
        loss_value = float(train_outputs[0].numpy())
        assert np.isfinite(loss_value), f"loss is not finite: {loss_value}"
        losses.append(loss_value)

        update_outputs = update_executable.run([w1_t, w2_t, train_outputs[1], train_outputs[2]])
        w1_t, w2_t = update_outputs[0], update_outputs[1]

    # 阈值经本机实测校准(seed=20260713、100 步、lr=0.1):实测 initial≈0.01276、
    # final≈0.00143,比值约 0.112,留有安全边际,不是刚好卡阈值。
    assert losses[-1] < losses[0] * 0.2, f"initial={losses[0]} final={losses[-1]}"


def _linear_regression_material_fp16(seed=20260713):
    """固定种子生成 fp16 训练材料:x/初始 w 为 float16,target 取"教师权重"
    w_true(float32 生成、结果下采样为 float16)对 x 的线性预测——与
    `_mlp_training_material` 同一"教师网络"思路,但网络退化为单个 matmul(无
    relu/第二层),对齐本用例只需覆盖 fp16 训练链路数值证据、不重复端到端
    MLP 收敛覆盖(那已有 fp32 版本 `test_mlp_training_loop_converges`)。
    x_np/target_np/w_np 三者均已是 np.float16。"""
    rng = np.random.default_rng(seed)
    x_np = rng.uniform(-1.0, 1.0, size=(_BATCH, _IN_DIM)).astype(np.float32)
    w_true = rng.uniform(-1.0, 1.0, size=(_IN_DIM, _OUT_DIM)).astype(np.float32)
    target_np = (x_np @ w_true).astype(np.float16)
    w_np = rng.uniform(-0.3, 0.3, size=(_IN_DIM, _OUT_DIM)).astype(np.float32)
    return x_np.astype(np.float16), target_np, w_np.astype(np.float16)


def test_fp16_training_loop_loss_decreases_and_dtype_preserved(core):
    """fp16 训练链路数值证据(M19 Task 4):matmul(x,w)->mse_loss 图(全程
    dtype=float16),build_backward_graph + build_sgd_update_graph 各 compile
    一次,循环 30 步只调 run()。断言:①loss 全程有限;②末值 < 首值(fp16
    训练收敛性证据,阈值经本机实测校准:seed=20260713、30 步、lr=0.1,line
    below 注释给出实测量级);③参数与梯度 dtype 全程保持 float16(v0 fp16
    训练链路不发生静默升精度,呼应 build_sgd_update_graph/mse_loss 等 cpu
    kernel 白名单本就含 float16,src/backends/cpu/kernels/loss.cpp)。bf16
    无 numpy 桥(numpy 无 bfloat16 原生 dtype,见 conftest.py bf16_tensor
    fixture 头注释),故本文件 python 侧不加 bf16 训练用例——bf16 训练链路的
    数值证据由 tests/cpp/compiler/test_training_loop.cpp 的
    SgdSingleStepBf16MatchesFp32AnalyticReference 覆盖。"""
    x_np, target_np, w_np = _linear_regression_material_fp16()

    forward = core.Graph("fp16_linear_forward")
    x = forward.add_graph_input([_BATCH, _IN_DIM], core.DType.float16, backend="cpu")
    w = forward.add_graph_input([_IN_DIM, _OUT_DIM], core.DType.float16, backend="cpu")
    target = forward.add_graph_input([_BATCH, _OUT_DIM], core.DType.float16, backend="cpu")
    pred = core.matmul(forward, x, w)
    loss = core.mse_loss(forward, pred, target)
    forward.mark_output(loss)

    training = core.build_backward_graph(forward, 0, [1])
    train_executable = core.compile(training, "cpu")

    update = core.build_sgd_update_graph(
        [([_IN_DIM, _OUT_DIM], core.DType.float16)], 0.1, backend="cpu"
    )
    update_executable = core.compile(update, "cpu")

    x_t = core.from_numpy(x_np)
    target_t = core.from_numpy(target_np)
    w_t = core.from_numpy(w_np)
    assert w_t.dtype == core.DType.float16

    losses = []
    for _ in range(30):
        train_outputs = train_executable.run([x_t, w_t, target_t])
        loss_value = float(train_outputs[0].numpy())
        assert np.isfinite(loss_value), f"loss is not finite: {loss_value}"
        losses.append(loss_value)

        grad_t = train_outputs[1]
        assert grad_t.dtype == core.DType.float16, f"grad dtype degraded to {grad_t.dtype}"

        update_outputs = update_executable.run([w_t, grad_t])
        w_t = update_outputs[0]
        assert w_t.dtype == core.DType.float16, f"param dtype degraded to {w_t.dtype}"

    # 阈值经本机实测校准(seed=20260713、30 步、lr=0.1):实测 initial≈0.75、
    # final≈1.7e-03,比值远小于 1,留有安全边际,不是刚好卡阈值。
    assert losses[-1] < losses[0] * 0.5, f"initial={losses[0]} final={losses[-1]}"


def test_build_backward_graph_preserves_multi_output_prefix_and_appends_gradient(core, allclose):
    """M26 ARCH-061:多输出 forward 选择第二个标量 loss 后,编译执行结果保持
    原输出顺序并在尾部追加 wrt 梯度。"""
    graph = core.Graph("multi_output_forward")
    x = graph.add_graph_input([2, 2], core.DType.float32)
    squared = core.square(graph, x)
    loss = core.sum(graph, squared, [])
    graph.mark_output(squared)
    graph.mark_output(loss)

    training = core.build_backward_graph(graph, 1, [0])
    executable = core.compile(training, "cpu")
    x_np = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float32)
    outputs = executable.run([core.from_numpy(x_np)])

    assert len(outputs) == 3
    allclose(outputs[0].numpy(), np.square(x_np), "float32")
    allclose(outputs[1].numpy(), np.sum(np.square(x_np)), "float32")
    allclose(outputs[2].numpy(), 2.0 * x_np, "float32")


def test_build_backward_graph_rejects_out_of_range_loss_output_index(core):
    """loss_output_index 越界 -> ValueError。"""
    graph = core.Graph("single_output_forward_loss_index")
    x = graph.add_graph_input([2, 2], core.DType.float32)
    loss = core.sum(graph, x, [])
    graph.mark_output(loss)
    with pytest.raises(ValueError, match="out of range"):
        core.build_backward_graph(graph, 5, [0])


def test_build_backward_graph_rejects_out_of_range_wrt_index(core):
    """wrt_input_indices 条目越界 -> ValueError。"""
    graph = core.Graph("single_output_forward_wrt_index")
    x = graph.add_graph_input([2, 2], core.DType.float32)
    loss = core.sum(graph, x, [])
    graph.mark_output(loss)
    with pytest.raises(ValueError, match="out of range"):
        core.build_backward_graph(graph, 0, [7])


def test_build_sgd_update_graph_rejects_non_finite_learning_rate(core):
    """learning_rate 非有限值(NaN/inf)-> ValueError。"""
    param_types = [([2, 2], core.DType.float32)]
    with pytest.raises(ValueError, match="must be finite"):
        core.build_sgd_update_graph(param_types, float("nan"))
    with pytest.raises(ValueError, match="must be finite"):
        core.build_sgd_update_graph(param_types, float("inf"))


def test_training_then_onnx_roundtrip_inference_matches(core, allclose, tmp_path):
    """训练若干步后 save_onnx_weights -> load_onnx_weights 往返权重,用往返后
    的权重推理与用训练所得权重直接推理结果一致(ADR-0013 串联,python 侧
    一例)。"""
    x_np, target_np, w1_np, w2_np = _mlp_training_material()

    forward = _build_mlp_forward_graph(core)
    training = core.build_backward_graph(forward, 0, [1, 2])
    train_executable = core.compile(training, "cpu")
    update = core.build_sgd_update_graph(
        [([_IN_DIM, _HIDDEN_DIM], core.DType.float32), ([_HIDDEN_DIM, _OUT_DIM], core.DType.float32)],
        0.05,
        backend="cpu",
    )
    update_executable = core.compile(update, "cpu")

    x_t = core.from_numpy(x_np)
    target_t = core.from_numpy(target_np)
    w1_t = core.from_numpy(w1_np)
    w2_t = core.from_numpy(w2_np)
    for _ in range(20):
        train_outputs = train_executable.run([x_t, w1_t, w2_t, target_t])
        update_outputs = update_executable.run([w1_t, w2_t, train_outputs[1], train_outputs[2]])
        w1_t, w2_t = update_outputs[0], update_outputs[1]

    path = str(tmp_path / "trained_weights.onnx")
    core.save_onnx_weights(path, {"w1": w1_t, "w2": w2_t})
    loaded = core.load_onnx_weights(path)

    infer_graph = core.Graph("mlp_infer")
    xi = infer_graph.add_graph_input([_BATCH, _IN_DIM], core.DType.float32)
    w1i = infer_graph.add_graph_input([_IN_DIM, _HIDDEN_DIM], core.DType.float32)
    w2i = infer_graph.add_graph_input([_HIDDEN_DIM, _OUT_DIM], core.DType.float32)
    hidden = core.relu(infer_graph, core.matmul(infer_graph, xi, w1i))
    pred = core.matmul(infer_graph, hidden, w2i)
    infer_graph.mark_output(pred)
    infer_executable = core.compile(infer_graph, "cpu")

    pred_from_trained = infer_executable.run([x_t, w1_t, w2_t])[0].numpy()
    pred_from_loaded = infer_executable.run([x_t, loaded["w1"], loaded["w2"]])[0].numpy()
    allclose(pred_from_loaded, pred_from_trained, "float32")
