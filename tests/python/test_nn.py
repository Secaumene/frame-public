"""frame.nn pybind11 绑定 pytest(M20 批2 Task5,python/src/bind_nn.cpp;权威
契约见 include/frame/nn/{module.h,layers.h}、docs/architecture/nn-design.md
§6)。PY-021「绑定即测」:Module.name/parameters()/build()、Linear/Relu/
Sequential/MseLoss 工厂、add_parameter_inputs 便捷面均至少 1 例。

数值对照口径:Linear/Sequential 构图后手工 from_numpy 注入权重(不经
InitSpec——数值物化非本批交付范围,module.h 头注释),与 numpy 手算矩阵乘/加/
relu/mse 对照(BUILD-011 float32 容差,经 conftest allclose)。
"""

import numpy as np
import pytest


def test_linear_with_bias_forward_matches_numpy(core, nn, allclose):
    """Linear(with_bias=True):matmul(x,w)+b,与 numpy 手算对照。"""
    batch, in_dim, out_dim = 4, 3, 5
    graph = core.Graph("linear_with_bias")
    x = graph.add_graph_input([batch, in_dim], core.DType.float32)

    linear = nn.Linear("fc", batch, in_dim, out_dim, True, core.DType.float32)
    params = nn.add_parameter_inputs(graph, linear)
    outputs = linear.build(graph, [x], params)
    assert len(outputs) == 1
    graph.mark_output(outputs[0])

    executable = core.compile(graph, "cpu")

    rng = np.random.default_rng(20260718)
    x_np = rng.uniform(-1.0, 1.0, size=(batch, in_dim)).astype(np.float32)
    w_np = rng.uniform(-1.0, 1.0, size=(in_dim, out_dim)).astype(np.float32)
    b_np = rng.uniform(-1.0, 1.0, size=(batch, out_dim)).astype(np.float32)
    result = executable.run(
        [core.from_numpy(x_np), core.from_numpy(w_np), core.from_numpy(b_np)]
    )[0].numpy()

    allclose(result, x_np @ w_np + b_np, "float32")


def test_linear_without_bias_forward_matches_numpy(core, nn, allclose):
    """Linear(with_bias=False):matmul(x,w),无 bias 图输入。"""
    batch, in_dim, out_dim = 4, 3, 5
    graph = core.Graph("linear_without_bias")
    x = graph.add_graph_input([batch, in_dim], core.DType.float32)

    linear = nn.Linear("fc", batch, in_dim, out_dim, False, core.DType.float32)
    params = nn.add_parameter_inputs(graph, linear)
    assert len(params) == 1
    outputs = linear.build(graph, [x], params)
    graph.mark_output(outputs[0])

    executable = core.compile(graph, "cpu")

    rng = np.random.default_rng(20260718)
    x_np = rng.uniform(-1.0, 1.0, size=(batch, in_dim)).astype(np.float32)
    w_np = rng.uniform(-1.0, 1.0, size=(in_dim, out_dim)).astype(np.float32)
    result = executable.run([core.from_numpy(x_np), core.from_numpy(w_np)])[0].numpy()

    allclose(result, x_np @ w_np, "float32")


def test_sequential_linear_relu_linear_forward_matches_numpy(core, nn, allclose):
    """Sequential([Linear(bias), Relu, Linear(no bias)]):与 numpy 手算对照,
    覆盖 Relu 工厂与多层参数切片拼接。"""
    batch, in_dim, hidden_dim, out_dim = 4, 3, 6, 2
    graph = core.Graph("mlp")
    x = graph.add_graph_input([batch, in_dim], core.DType.float32)

    model = nn.Sequential(
        "mlp",
        [
            nn.Linear("l0", batch, in_dim, hidden_dim, True, core.DType.float32),
            nn.Relu("relu0"),
            nn.Linear("l1", batch, hidden_dim, out_dim, False, core.DType.float32),
        ],
    )
    params = nn.add_parameter_inputs(graph, model)
    assert len(params) == 3  # l0.weight, l0.bias, l1.weight
    outputs = model.build(graph, [x], params)
    graph.mark_output(outputs[0])

    executable = core.compile(graph, "cpu")

    rng = np.random.default_rng(20260718)
    x_np = rng.uniform(-1.0, 1.0, size=(batch, in_dim)).astype(np.float32)
    w0_np = rng.uniform(-1.0, 1.0, size=(in_dim, hidden_dim)).astype(np.float32)
    b0_np = rng.uniform(-1.0, 1.0, size=(batch, hidden_dim)).astype(np.float32)
    w1_np = rng.uniform(-1.0, 1.0, size=(hidden_dim, out_dim)).astype(np.float32)
    tensors = [core.from_numpy(t) for t in (x_np, w0_np, b0_np, w1_np)]
    result = executable.run(tensors)[0].numpy()

    expected = np.maximum(x_np @ w0_np + b0_np, 0.0) @ w1_np
    allclose(result, expected, "float32")


def test_mse_loss_module_matches_manual_computation(core, nn, allclose):
    """MseLoss(pred, target):mean((pred-target)^2),标量输出,与 numpy 手算对照。"""
    shape = [4, 2]
    graph = core.Graph("mse")
    pred = graph.add_graph_input(shape, core.DType.float32)
    target = graph.add_graph_input(shape, core.DType.float32)

    loss_module = nn.MseLoss("loss")
    outputs = loss_module.build(graph, [pred, target], [])
    assert len(outputs) == 1
    graph.mark_output(outputs[0])

    executable = core.compile(graph, "cpu")

    rng = np.random.default_rng(20260718)
    pred_np = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    target_np = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    result = executable.run([core.from_numpy(pred_np), core.from_numpy(target_np)])[0].numpy()

    expected = np.mean((pred_np - target_np) ** 2, dtype=np.float32)
    allclose(result, expected, "float32")


def test_module_parameters_metadata_linear_with_bias(core, nn):
    """parameters() 元信息:(name, shape, dtype) 三元组,名字带模块名前缀。"""
    linear = nn.Linear("fc", 4, 3, 5, True, core.DType.float32)
    infos = linear.parameters()
    assert infos == [
        ("fc.weight", [3, 5], core.DType.float32),
        ("fc.bias", [4, 5], core.DType.float32),
    ]


def test_module_parameters_metadata_linear_without_bias(core, nn):
    """with_bias=False 时 parameters() 仅含 weight 一项。"""
    linear = nn.Linear("fc2", 8, 8, 1, False, core.DType.float32)
    infos = linear.parameters()
    assert infos == [("fc2.weight", [8, 1], core.DType.float32)]


def test_module_parameters_metadata_sequential_prefixes_child_names(core, nn):
    """Sequential 的 parameters() 名字前缀为「顶层名.子模块名.局部名」先序遍历。"""
    model = nn.Sequential(
        "seq",
        [
            nn.Linear("0", 2, 2, 3, True, core.DType.float32),
            nn.Relu("1"),
        ],
    )
    names = [name for name, _, _ in model.parameters()]
    assert names == ["seq.0.weight", "seq.0.bias"]


def test_module_name_property(core, nn):
    """Module.name 只读属性回读构造时的名字。"""
    linear = nn.Linear("my_layer", 2, 2, 2, False, core.DType.float32)
    assert linear.name == "my_layer"


def test_add_parameter_inputs_returns_values_matching_parameters_count(core, nn):
    """add_parameter_inputs(graph, module) 返回的 Value 句柄数 == parameters() 数。"""
    graph = core.Graph("param_inputs")
    linear = nn.Linear("fc", 4, 3, 5, True, core.DType.float32)
    values = nn.add_parameter_inputs(graph, linear)
    assert len(values) == len(linear.parameters())


# ---------------------------------------------------------------------------
# M21 批3 T6:卷积批次工厂 Conv2d/Conv1d/MaxPool2d/AvgPool2d/Sigmoid/Flatten/
# AFF/Dwt2d/Dwt1d(docs/plan/2026-07-18-batch3-m21-conv.md 第1.4节)。每个新
# 工厂至少落在下列一条构图-编译-run 链内(BUILD-021/PY-021)。
# ---------------------------------------------------------------------------


def test_conv2d_sigmoid_maxpool2d_flatten_forward_matches_numpy(core, nn, allclose):
    """Conv2d(1x1)->Sigmoid->MaxPool2d->Flatten 链,与 numpy 手算对照。
    kernel_hw=[1,1] 使卷积退化为逐像素通道线性变换(np.einsum 手算),规避
    手写滑窗卷积;MaxPool2d(kernel=stride=[2,2]) 恰整除输入空间维,窗口不
    重叠,可用 reshape+max 手算。"""
    n, cin, cout, h, w = 2, 2, 3, 4, 4
    graph = core.Graph("cnn_chain")
    x = graph.add_graph_input([n, cin, h, w], core.DType.float32)

    conv = nn.Conv2d("conv", cin, cout, [1, 1], [1, 1], [0, 0], 1, True, core.DType.float32)
    conv_params = nn.add_parameter_inputs(graph, conv)
    conv_out = conv.build(graph, [x], conv_params)
    assert len(conv_out) == 1

    sigmoid_out = nn.Sigmoid("act").build(graph, conv_out, [])
    pool_out = nn.MaxPool2d("pool", [2, 2], [2, 2], [0, 0]).build(graph, sigmoid_out, [])
    flatten_out = nn.Flatten("flatten").build(graph, pool_out, [])
    assert len(flatten_out) == 1
    graph.mark_output(flatten_out[0])

    executable = core.compile(graph, "cpu")

    rng = np.random.default_rng(20260719)
    x_np = rng.uniform(-1.0, 1.0, size=(n, cin, h, w)).astype(np.float32)
    w_np = rng.uniform(-1.0, 1.0, size=(cout, cin, 1, 1)).astype(np.float32)
    b_np = rng.uniform(-1.0, 1.0, size=(cout,)).astype(np.float32)
    tensors = [core.from_numpy(t) for t in (x_np, w_np, b_np)]
    result = executable.run(tensors)[0].numpy()

    conv_expected = np.einsum("nchw,oc->nohw", x_np, w_np[:, :, 0, 0]) + b_np[:, None, None]
    sigmoid_expected = 1.0 / (1.0 + np.exp(-conv_expected))
    pooled = sigmoid_expected.reshape(n, cout, h // 2, 2, w // 2, 2).max(axis=(3, 5))
    expected = pooled.reshape(n, -1)

    allclose(result, expected.astype(np.float32), "float32")


def test_conv1d_forward_matches_numpy(core, nn, allclose):
    """Conv1d(1x1)->与 numpy 手算对照(与 Conv2d 用例同款 1x1 退化技巧,
    规避手写滑窗卷积)。"""
    n, cin, cout, length = 2, 3, 2, 5
    graph = core.Graph("conv1d_chain")
    x = graph.add_graph_input([n, cin, length], core.DType.float32)

    conv = nn.Conv1d("conv", cin, cout, 1, 1, 0, 1, True, core.DType.float32)
    conv_params = nn.add_parameter_inputs(graph, conv)
    conv_out = conv.build(graph, [x], conv_params)
    graph.mark_output(conv_out[0])

    executable = core.compile(graph, "cpu")

    rng = np.random.default_rng(20260719)
    x_np = rng.uniform(-1.0, 1.0, size=(n, cin, length)).astype(np.float32)
    w_np = rng.uniform(-1.0, 1.0, size=(cout, cin, 1)).astype(np.float32)
    b_np = rng.uniform(-1.0, 1.0, size=(cout,)).astype(np.float32)
    tensors = [core.from_numpy(t) for t in (x_np, w_np, b_np)]
    result = executable.run(tensors)[0].numpy()

    expected = np.einsum("ncl,oc->nol", x_np, w_np[:, :, 0]) + b_np[None, :, None]
    allclose(result, expected.astype(np.float32), "float32")


def test_avg_pool2d_forward_matches_numpy(core, nn, allclose):
    """AvgPool2d(kernel=stride=[2,2],padding=[0,0])——窗口不重叠,与
    reshape+mean 手算对照(分母恒 KH*KW,含 padding,本例 padding=0 故与
    简单均值一致)。"""
    n, c, h, w = 2, 2, 4, 4
    graph = core.Graph("avg_pool2d_chain")
    x = graph.add_graph_input([n, c, h, w], core.DType.float32)
    pool_out = nn.AvgPool2d("pool", [2, 2], [2, 2], [0, 0]).build(graph, [x], [])
    graph.mark_output(pool_out[0])

    executable = core.compile(graph, "cpu")

    rng = np.random.default_rng(20260719)
    x_np = rng.uniform(-1.0, 1.0, size=(n, c, h, w)).astype(np.float32)
    result = executable.run([core.from_numpy(x_np)])[0].numpy()

    expected = x_np.reshape(n, c, h // 2, 2, w // 2, 2).mean(axis=(3, 5))
    allclose(result, expected.astype(np.float32), "float32")


def test_aff_forward_matches_hand_computed_reference(core, nn, allclose):
    """nn.AFF(local-only 变体):M=sigmoid(c2(relu(c1(X+Y)))),
    out=M*X+(1-M)*Y,c1/c2 均 1x1 卷积,numpy 手算对照(与
    tests/cpp/nn/test_aff_smoke.cpp 同一公式,规模稍大用 einsum 批量手算)。"""
    n, channels, h, w = 1, 2, 2, 2
    graph = core.Graph("aff_chain")
    x = graph.add_graph_input([n, channels, h, w], core.DType.float32)
    y = graph.add_graph_input([n, channels, h, w], core.DType.float32)

    aff = nn.AFF("aff", channels, core.DType.float32)
    aff_params = nn.add_parameter_inputs(graph, aff)
    assert len(aff_params) == 4  # c1.weight, c1.bias, c2.weight, c2.bias
    aff_out = aff.build(graph, [x, y], aff_params)
    graph.mark_output(aff_out[0])

    executable = core.compile(graph, "cpu")

    rng = np.random.default_rng(20260719)
    x_np = rng.uniform(-1.0, 1.0, size=(n, channels, h, w)).astype(np.float32)
    y_np = rng.uniform(-1.0, 1.0, size=(n, channels, h, w)).astype(np.float32)
    w1_np = rng.uniform(-0.5, 0.5, size=(channels, channels, 1, 1)).astype(np.float32)
    b1_np = rng.uniform(-0.5, 0.5, size=(channels,)).astype(np.float32)
    w2_np = rng.uniform(-0.5, 0.5, size=(channels, channels, 1, 1)).astype(np.float32)
    b2_np = rng.uniform(-0.5, 0.5, size=(channels,)).astype(np.float32)
    tensors = [
        core.from_numpy(t) for t in (x_np, y_np, w1_np, b1_np, w2_np, b2_np)
    ]
    result = executable.run(tensors)[0].numpy()

    t1 = np.einsum("nchw,oc->nohw", x_np + y_np, w1_np[:, :, 0, 0]) + b1_np[:, None, None]
    r1 = np.maximum(t1, 0.0)
    t2 = np.einsum("nchw,oc->nohw", r1, w2_np[:, :, 0, 0]) + b2_np[:, None, None]
    m = 1.0 / (1.0 + np.exp(-t2))
    expected = m * x_np + (1.0 - m) * y_np

    allclose(result, expected.astype(np.float32), "float32")


def test_dwt2d_and_dwt1d_build_compile_run_smoke(core, nn):
    """Dwt2d/Dwt1d:固定滤波器不参与训练(parameters() 为空),构图-编译-run
    一条链验证输出形状与数值有限(数值正确性已由 tests/cpp/nn/
    test_dwt_filter_constants.cpp 的图内常量断言覆盖,本用例只验证 Python
    绑定面可用)。"""
    n, channels, hh, ww = 1, 2, 4, 4
    graph = core.Graph("dwt2d_chain")
    x = graph.add_graph_input([n, channels, hh, ww], core.DType.float32)
    dwt2d = nn.Dwt2d("dwt2d", channels, "haar")
    assert len(dwt2d.parameters()) == 0
    out2d = dwt2d.build(graph, [x], [])
    graph.mark_output(out2d[0])
    executable2d = core.compile(graph, "cpu")
    rng = np.random.default_rng(20260719)
    x_np = rng.uniform(-1.0, 1.0, size=(n, channels, hh, ww)).astype(np.float32)
    result2d = executable2d.run([core.from_numpy(x_np)])[0].numpy()
    assert result2d.shape == (n, 4 * channels, hh // 2, ww // 2)
    assert np.all(np.isfinite(result2d))

    length = 8
    graph1d = core.Graph("dwt1d_chain")
    x1d = graph1d.add_graph_input([n, channels, length], core.DType.float32)
    dwt1d = nn.Dwt1d("dwt1d", channels, "db4")
    assert len(dwt1d.parameters()) == 0
    out1d = dwt1d.build(graph1d, [x1d], [])
    graph1d.mark_output(out1d[0])
    executable1d = core.compile(graph1d, "cpu")
    x1d_np = rng.uniform(-1.0, 1.0, size=(n, channels, length)).astype(np.float32)
    result1d = executable1d.run([core.from_numpy(x1d_np)])[0].numpy()
    # db4 滤波器 K=4、stride=2、padding=0(计划 1.4 节口径),floor 输出长度=
    # (length-4)//2+1(不同于 haar 的 K=2、length//2)。
    expected_out_length = (length - 4) // 2 + 1
    assert result1d.shape == (n, 2 * channels, expected_out_length)
    assert np.all(np.isfinite(result1d))


# ---------------------------------------------------------------------------
# M22 批4 T5:序列批次工厂 LayerNorm/LSTM/MultiheadAttention/
# TransformerEncoderBlock(docs/plan/2026-07-19-batch4-m22-seq.md §1.7)。每个
# 新工厂至少落在下列一条构图-编译-run 链内(BUILD-021/PY-021)。
# ---------------------------------------------------------------------------


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _lstm_reference(x, w_ih, w_hh, bias, hidden_dim):
    """numpy 手算 LSTM 前向(门序 i,f,g,o 固定,同 src/nn/layers.cpp::LSTM):
    h0=c0=0;z=x_t@W_ih+h@W_hh+bias;c=f*c+i*g;h=o*tanh(c)。"""
    batch, num_steps, _ = x.shape
    h = np.zeros((batch, hidden_dim), dtype=np.float64)
    c = np.zeros((batch, hidden_dim), dtype=np.float64)
    for t in range(num_steps):
        x_t = x[:, t, :]
        z = x_t @ w_ih + h @ w_hh + bias
        i = _sigmoid(z[:, 0:hidden_dim])
        f = _sigmoid(z[:, hidden_dim : 2 * hidden_dim])
        g = np.tanh(z[:, 2 * hidden_dim : 3 * hidden_dim])
        o = _sigmoid(z[:, 3 * hidden_dim : 4 * hidden_dim])
        c = f * c + i * g
        h = o * np.tanh(c)
    return h


def _mha_reference(x, wq, bq, wk, bk, wv, bv, wo, bo, batch, seq_len, num_heads):
    """numpy 手算多头自注意力前向(per-(b,h) 静态展开,同
    src/nn/layers.cpp::MultiheadAttention)。"""
    embed_dim = x.shape[1]
    dh = embed_dim // num_heads
    scale = 1.0 / np.sqrt(dh)
    q = x @ wq + bq
    k = x @ wk + bk
    v = x @ wv + bv
    attn_out = np.zeros_like(x, dtype=np.float64)
    for b in range(batch):
        rows = slice(b * seq_len, (b + 1) * seq_len)
        head_outs = []
        for h in range(num_heads):
            cols = slice(h * dh, (h + 1) * dh)
            q_bh = q[rows, cols]
            k_bh = k[rows, cols]
            v_bh = v[rows, cols]
            scores = (q_bh @ k_bh.T) * scale
            scores = scores - scores.max(axis=1, keepdims=True)
            weights = np.exp(scores)
            weights = weights / weights.sum(axis=1, keepdims=True)
            head_outs.append(weights @ v_bh)
        attn_out[rows, :] = np.concatenate(head_outs, axis=1)
    return attn_out @ wo + bo


def test_layer_norm_forward_matches_numpy(core, nn, allclose):
    """LayerNorm:与 numpy 手算行归一化+仿射对照(总体方差,无 Bessel 修正,
    同 src/backends/cpu/kernels/sequence.cpp::layer_norm_cpu_kernel)。"""
    rows, dim, eps = 3, 4, 1e-5
    graph = core.Graph("layer_norm_chain")
    x = graph.add_graph_input([rows, dim], core.DType.float32)

    ln = nn.LayerNorm("ln", dim, eps, core.DType.float32)
    params = ln.parameters()
    assert params == [("ln.gamma", [dim], core.DType.float32), ("ln.beta", [dim], core.DType.float32)]
    param_inputs = nn.add_parameter_inputs(graph, ln)
    outputs = ln.build(graph, [x], param_inputs)
    assert len(outputs) == 1
    graph.mark_output(outputs[0])

    executable = core.compile(graph, "cpu")

    rng = np.random.default_rng(20260720)
    x_np = rng.uniform(-1.0, 1.0, size=(rows, dim)).astype(np.float32)
    gamma_np = rng.uniform(0.5, 1.5, size=(dim,)).astype(np.float32)
    beta_np = rng.uniform(-0.5, 0.5, size=(dim,)).astype(np.float32)
    result = executable.run(
        [core.from_numpy(x_np), core.from_numpy(gamma_np), core.from_numpy(beta_np)]
    )[0].numpy()

    mean = x_np.mean(axis=1, keepdims=True)
    var = ((x_np - mean) ** 2).mean(axis=1, keepdims=True)
    xhat = (x_np - mean) / np.sqrt(var + eps)
    expected = gamma_np[None, :] * xhat + beta_np[None, :]

    allclose(result, expected.astype(np.float32), "float32")


def test_layer_norm_rejects_non_positive_eps(core, nn):
    """eps<=0 由 layer_norm 算子 shape_infer 在 build() 期拒绝(ValueError),
    LayerNorm 工厂自身不重复校验(include/frame/nn/layers.h 头注释)。"""
    graph = core.Graph("layer_norm_bad_eps")
    x = graph.add_graph_input([2, 3], core.DType.float32)
    ln = nn.LayerNorm("ln", 3, 0.0, core.DType.float32)
    param_inputs = nn.add_parameter_inputs(graph, ln)
    with pytest.raises(ValueError):
        ln.build(graph, [x], param_inputs)


def test_lstm_forward_matches_numpy(core, nn, allclose):
    """LSTM:与 numpy 手算门控循环对照(门序 i,f,g,o 固定)。"""
    batch, num_steps, input_dim, hidden_dim = 2, 3, 2, 3
    graph = core.Graph("lstm_chain")
    x = graph.add_graph_input([batch, num_steps, input_dim], core.DType.float32)

    lstm = nn.LSTM("lstm", batch, num_steps, input_dim, hidden_dim, core.DType.float32)
    params = lstm.parameters()
    assert [name for name, _, _ in params] == ["lstm.W_ih", "lstm.W_hh", "lstm.bias"]
    param_inputs = nn.add_parameter_inputs(graph, lstm)
    outputs = lstm.build(graph, [x], param_inputs)
    assert len(outputs) == 1
    graph.mark_output(outputs[0])

    executable = core.compile(graph, "cpu")

    rng = np.random.default_rng(20260720)
    x_np = rng.uniform(-1.0, 1.0, size=(batch, num_steps, input_dim)).astype(np.float32)
    w_ih_np = rng.uniform(-0.5, 0.5, size=(input_dim, 4 * hidden_dim)).astype(np.float32)
    w_hh_np = rng.uniform(-0.5, 0.5, size=(hidden_dim, 4 * hidden_dim)).astype(np.float32)
    bias_np = rng.uniform(-0.5, 0.5, size=(batch, 4 * hidden_dim)).astype(np.float32)
    result = executable.run(
        [
            core.from_numpy(x_np),
            core.from_numpy(w_ih_np),
            core.from_numpy(w_hh_np),
            core.from_numpy(bias_np),
        ]
    )[0].numpy()
    assert result.shape == (batch, hidden_dim)

    expected = _lstm_reference(
        x_np.astype(np.float64),
        w_ih_np.astype(np.float64),
        w_hh_np.astype(np.float64),
        bias_np.astype(np.float64),
        hidden_dim,
    )
    allclose(result, expected.astype(np.float32), "float32")


def test_multihead_attention_forward_matches_numpy(core, nn, allclose):
    """MultiheadAttention:与 numpy 手算逐头注意力对照(per-(b,h) 静态展开)。"""
    batch, seq_len, embed_dim, num_heads = 2, 2, 4, 2
    rows = batch * seq_len
    graph = core.Graph("mha_chain")
    x = graph.add_graph_input([rows, embed_dim], core.DType.float32)

    mha = nn.MultiheadAttention("mha", batch, seq_len, embed_dim, num_heads, True, core.DType.float32)
    params = mha.parameters()
    assert [name for name, _, _ in params] == [
        "mha.q.weight",
        "mha.q.bias",
        "mha.k.weight",
        "mha.k.bias",
        "mha.v.weight",
        "mha.v.bias",
        "mha.o.weight",
        "mha.o.bias",
    ]
    param_inputs = nn.add_parameter_inputs(graph, mha)
    outputs = mha.build(graph, [x], param_inputs)
    assert len(outputs) == 1
    graph.mark_output(outputs[0])

    executable = core.compile(graph, "cpu")

    rng = np.random.default_rng(20260720)
    x_np = rng.uniform(-1.0, 1.0, size=(rows, embed_dim)).astype(np.float32)
    wq_np = rng.uniform(-0.5, 0.5, size=(embed_dim, embed_dim)).astype(np.float32)
    bq_np = rng.uniform(-0.5, 0.5, size=(rows, embed_dim)).astype(np.float32)
    wk_np = rng.uniform(-0.5, 0.5, size=(embed_dim, embed_dim)).astype(np.float32)
    bk_np = rng.uniform(-0.5, 0.5, size=(rows, embed_dim)).astype(np.float32)
    wv_np = rng.uniform(-0.5, 0.5, size=(embed_dim, embed_dim)).astype(np.float32)
    bv_np = rng.uniform(-0.5, 0.5, size=(rows, embed_dim)).astype(np.float32)
    wo_np = rng.uniform(-0.5, 0.5, size=(embed_dim, embed_dim)).astype(np.float32)
    bo_np = rng.uniform(-0.5, 0.5, size=(rows, embed_dim)).astype(np.float32)
    tensors = [
        core.from_numpy(t)
        for t in (x_np, wq_np, bq_np, wk_np, bk_np, wv_np, bv_np, wo_np, bo_np)
    ]
    result = executable.run(tensors)[0].numpy()
    assert result.shape == (rows, embed_dim)

    expected = _mha_reference(
        x_np.astype(np.float64),
        wq_np.astype(np.float64),
        bq_np.astype(np.float64),
        wk_np.astype(np.float64),
        bk_np.astype(np.float64),
        wv_np.astype(np.float64),
        bv_np.astype(np.float64),
        wo_np.astype(np.float64),
        bo_np.astype(np.float64),
        batch,
        seq_len,
        num_heads,
    )
    allclose(result, expected.astype(np.float32), "float32")


def test_multihead_attention_rejects_embed_dim_not_divisible_by_num_heads(core, nn):
    """embed_dim % num_heads != 0 由 build() 期显式校验拒绝(ValueError),
    见 src/nn/layers.cpp::MultiheadAttention。"""
    batch, seq_len, embed_dim, num_heads = 1, 2, 5, 2
    graph = core.Graph("mha_bad_heads")
    x = graph.add_graph_input([batch * seq_len, embed_dim], core.DType.float32)
    mha = nn.MultiheadAttention(
        "mha", batch, seq_len, embed_dim, num_heads, False, core.DType.float32
    )
    param_inputs = nn.add_parameter_inputs(graph, mha)
    with pytest.raises(ValueError):
        mha.build(graph, [x], param_inputs)


def test_transformer_encoder_block_build_compile_run_smoke(core, nn):
    """TransformerEncoderBlock(Post-LN):构图-编译-run 一条链验证输出 shape
    与 parameters() 计数(mha 4 个 Linear + ln1 + ffn1 + ffn2 + ln2,均
    with_bias=True:(2+2+2+2)+2+2+2+2=16);数值正确性已由 LayerNorm/LSTM/
    MultiheadAttention/Linear 各自的手算对照用例覆盖(TransformerEncoderBlock
    是四者的既有构图组合,无新数值语义)。"""
    batch, seq_len, embed_dim, num_heads, ffn_dim = 1, 2, 4, 2, 6
    rows = batch * seq_len
    graph = core.Graph("transformer_block_chain")
    x = graph.add_graph_input([rows, embed_dim], core.DType.float32)

    block = nn.TransformerEncoderBlock(
        "block", batch, seq_len, embed_dim, num_heads, ffn_dim, True, core.DType.float32
    )
    params = block.parameters()
    assert len(params) == 16

    param_inputs = nn.add_parameter_inputs(graph, block)
    outputs = block.build(graph, [x], param_inputs)
    assert len(outputs) == 1
    graph.mark_output(outputs[0])

    executable = core.compile(graph, "cpu")

    rng = np.random.default_rng(20260720)
    tensors = [core.from_numpy(rng.uniform(-1.0, 1.0, size=(rows, embed_dim)).astype(np.float32))]
    for _, shape, _ in params:
        tensors.append(core.from_numpy(rng.uniform(-0.5, 0.5, size=shape).astype(np.float32)))
    result = executable.run(tensors)[0].numpy()

    assert result.shape == (rows, embed_dim)
    assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# M23 批5 T5:频域批次工厂 SpectralConv1d/FourierFilter1d/Fno1dBlock
# (docs/plan/2026-07-21-batch5-m23-fft.md §1.5)。每个新工厂至少落在下列一条
# 构图-编译-run 链内(BUILD-021/PY-021),与 numpy np.fft.rfft/irfft 手算对照
# (numpy 口径:正变换不归一化、逆变换归一化 1/n,同 rfft/irfft 算子契约,
# §1.2;已由 tests/cpp/ops/test_op_fft.cpp 的解析谱 golden 独立佐证 rfft/irfft
# 自身与 numpy 口径一致,故本文件可放心以 np.fft 为 oracle)。
# ---------------------------------------------------------------------------


def _spectral_conv1d_reference(x, w_re, w_im, modes, out_channels):
    """numpy 手算谱卷积前向(逐模态复数矩阵乘 + 隐式零补,同
    src/nn/layers.cpp::SpectralConv1d)。"""
    batch, in_channels, n = x.shape
    k = n // 2 + 1
    x_hat = np.fft.rfft(x, axis=-1)  # [batch, in_channels, k] complex128
    w_complex = (w_re + 1j * w_im).reshape(in_channels, modes, out_channels)
    y_hat = np.zeros((batch, out_channels, k), dtype=np.complex128)
    for j in range(modes):
        y_hat[:, :, j] = x_hat[:, :, j] @ w_complex[:, j, :]
    return np.fft.irfft(y_hat, n=n, axis=-1)


def test_spectral_conv1d_forward_matches_numpy_rfft_reference(core, nn, allclose):
    """SpectralConv1d:与 numpy np.fft.rfft/irfft + 逐模态复数矩阵乘手算对照
    (modes<k=n/2+1,覆盖零补分支)。"""
    batch, in_channels, out_channels, n, modes = 2, 3, 4, 8, 2
    graph = core.Graph("spectral_conv1d_chain")
    x = graph.add_graph_input([batch, in_channels, n], core.DType.float32)

    sc = nn.SpectralConv1d("sc", batch, in_channels, out_channels, n, modes, core.DType.float32)
    params = sc.parameters()
    assert [name for name, _, _ in params] == ["sc.W_re", "sc.W_im"]
    param_inputs = nn.add_parameter_inputs(graph, sc)
    outputs = sc.build(graph, [x], param_inputs)
    assert len(outputs) == 1
    graph.mark_output(outputs[0])

    executable = core.compile(graph, "cpu")

    rng = np.random.default_rng(20260721)
    x_np = rng.uniform(-1.0, 1.0, size=(batch, in_channels, n)).astype(np.float32)
    w_re_np = rng.uniform(-0.5, 0.5, size=(in_channels, modes * out_channels)).astype(np.float32)
    w_im_np = rng.uniform(-0.5, 0.5, size=(in_channels, modes * out_channels)).astype(np.float32)
    result = executable.run(
        [core.from_numpy(x_np), core.from_numpy(w_re_np), core.from_numpy(w_im_np)]
    )[0].numpy()
    assert result.shape == (batch, out_channels, n)

    expected = _spectral_conv1d_reference(
        x_np.astype(np.float64),
        w_re_np.astype(np.float64),
        w_im_np.astype(np.float64),
        modes,
        out_channels,
    )
    allclose(result, expected.astype(np.float32), "float32")


def test_spectral_conv1d_rejects_modes_exceeding_half_plus_one(core, nn):
    """modes>n/2+1 由 build() 期显式校验拒绝(ValueError),见
    src/nn/layers.cpp::SpectralConv1d。"""
    batch, in_channels, out_channels, n = 2, 3, 4, 8  # k=5
    graph = core.Graph("spectral_conv1d_bad_modes")
    x = graph.add_graph_input([batch, in_channels, n], core.DType.float32)
    sc = nn.SpectralConv1d("sc", batch, in_channels, out_channels, n, 6, core.DType.float32)
    param_inputs = nn.add_parameter_inputs(graph, sc)
    with pytest.raises(ValueError):
        sc.build(graph, [x], param_inputs)


def test_fourier_filter1d_forward_matches_numpy_rfft_reference(core, nn, allclose):
    """FourierFilter1d:与 numpy np.fft.rfft/irfft + 逐元素复乘手算对照
    (**注意逐样本参数语义**:w_re/w_im 形状含 batch 维,是每样本独立滤波器,
    见 include/frame/nn/layers.h 头注释)。"""
    batch, channels, n = 2, 3, 8
    k = n // 2 + 1
    graph = core.Graph("fourier_filter1d_chain")
    x = graph.add_graph_input([batch, channels, n], core.DType.float32)

    ff = nn.FourierFilter1d("ff", batch, channels, n, core.DType.float32)
    params = ff.parameters()
    assert params == [
        ("ff.w_re", [batch, channels, k, 1], core.DType.float32),
        ("ff.w_im", [batch, channels, k, 1], core.DType.float32),
    ]
    param_inputs = nn.add_parameter_inputs(graph, ff)
    outputs = ff.build(graph, [x], param_inputs)
    assert len(outputs) == 1
    graph.mark_output(outputs[0])

    executable = core.compile(graph, "cpu")

    rng = np.random.default_rng(20260721)
    x_np = rng.uniform(-1.0, 1.0, size=(batch, channels, n)).astype(np.float32)
    w_re_np = rng.uniform(-0.5, 0.5, size=(batch, channels, k, 1)).astype(np.float32)
    w_im_np = rng.uniform(-0.5, 0.5, size=(batch, channels, k, 1)).astype(np.float32)
    result = executable.run(
        [core.from_numpy(x_np), core.from_numpy(w_re_np), core.from_numpy(w_im_np)]
    )[0].numpy()
    assert result.shape == (batch, channels, n)

    x_hat = np.fft.rfft(x_np.astype(np.float64), axis=-1)
    w_complex = (w_re_np.astype(np.float64) + 1j * w_im_np.astype(np.float64))[..., 0]
    y_hat = w_complex * x_hat
    expected = np.fft.irfft(y_hat, n=n, axis=-1)
    allclose(result, expected.astype(np.float32), "float32")


def test_fno1d_block_forward_matches_numpy_reference(core, nn, allclose):
    """Fno1dBlock:children=[SpectralConv1d, Conv1d] 组合
    y=tanh(spectral(x)+conv1x1(x)),与 numpy 手算对照(conv1x1 同
    test_conv1d_forward_matches_numpy 的 einsum 手法)。"""
    batch, in_channels, out_channels, n, modes = 2, 3, 4, 8, 2
    graph = core.Graph("fno1d_block_chain")
    x = graph.add_graph_input([batch, in_channels, n], core.DType.float32)

    block = nn.Fno1dBlock("fno", batch, in_channels, out_channels, n, modes, core.DType.float32)
    params = block.parameters()
    assert [name for name, _, _ in params] == [
        "fno.spectral.W_re",
        "fno.spectral.W_im",
        "fno.bypass.weight",
        "fno.bypass.bias",
    ]
    param_inputs = nn.add_parameter_inputs(graph, block)
    outputs = block.build(graph, [x], param_inputs)
    assert len(outputs) == 1
    graph.mark_output(outputs[0])

    executable = core.compile(graph, "cpu")

    rng = np.random.default_rng(20260721)
    x_np = rng.uniform(-1.0, 1.0, size=(batch, in_channels, n)).astype(np.float32)
    w_re_np = rng.uniform(-0.5, 0.5, size=(in_channels, modes * out_channels)).astype(np.float32)
    w_im_np = rng.uniform(-0.5, 0.5, size=(in_channels, modes * out_channels)).astype(np.float32)
    bypass_w_np = rng.uniform(-0.5, 0.5, size=(out_channels, in_channels, 1)).astype(np.float32)
    bypass_b_np = rng.uniform(-0.5, 0.5, size=(out_channels,)).astype(np.float32)
    result = executable.run(
        [
            core.from_numpy(x_np),
            core.from_numpy(w_re_np),
            core.from_numpy(w_im_np),
            core.from_numpy(bypass_w_np),
            core.from_numpy(bypass_b_np),
        ]
    )[0].numpy()
    assert result.shape == (batch, out_channels, n)

    spectral = _spectral_conv1d_reference(
        x_np.astype(np.float64),
        w_re_np.astype(np.float64),
        w_im_np.astype(np.float64),
        modes,
        out_channels,
    )
    conv1x1 = np.einsum(
        "bcl,oc->bol", x_np.astype(np.float64), bypass_w_np[:, :, 0].astype(np.float64)
    ) + bypass_b_np.astype(np.float64)[None, :, None]
    expected = np.tanh(spectral + conv1x1)
    allclose(result, expected.astype(np.float32), "float32")


# ---------------------------------------------------------------------------
# M25 状态空间模型工厂薄绑定:参数元信息、构图、编译和执行链。
# ---------------------------------------------------------------------------


def _run_ssm_factory(core, nn, factory_name):
    """Mamba/FourierMamba 公共 Python 薄绑定执行骨架。"""
    batch, channels, steps, kernel_size = 1, 2, 4, 2
    factory = getattr(nn, factory_name)
    model = factory("ssm", batch, channels, steps, kernel_size, core.DType.float32)
    graph = core.Graph(f"{factory_name}_python_binding")
    x = graph.add_graph_input([batch, channels, steps], core.DType.float32)
    param_values = nn.add_parameter_inputs(graph, model)
    outputs = model.build(graph, [x], param_values)
    assert len(outputs) == 1
    graph.mark_output(outputs[0])
    executable = core.compile(graph, "cpu")

    rng = np.random.default_rng(20260723)
    x_np = rng.uniform(-0.5, 0.5, size=(batch, channels, steps)).astype(np.float32)
    tensors = [core.from_numpy(x_np)]
    for _, shape, _ in model.parameters():
        tensors.append(core.from_numpy(rng.uniform(-0.2, 0.2, size=shape).astype(np.float32)))
    result = executable.run(tensors)[0].numpy()
    assert result.shape == (batch, channels, steps)
    assert np.all(np.isfinite(result))
    return model.parameters()


def test_mamba_python_factory_builds_and_runs(core, nn):
    """Mamba 公开 factory 绑定保持参数先序并可执行整图。"""
    params = _run_ssm_factory(core, nn, "Mamba")
    assert len(params) == 16
    assert params[0] == ("ssm.conv.weight", [2, 1, 2], core.DType.float32)
    assert params[-1] == ("ssm.out.bias", [4, 2], core.DType.float32)


def test_fourier_mamba_python_factory_builds_and_runs(core, nn):
    """FourierMamba 公开 factory 绑定追加复用频域分支参数并可执行。"""
    params = _run_ssm_factory(core, nn, "FourierMamba")
    assert len(params) == 18
    assert params[-2] == ("ssm.fourier.w_re", [1, 2, 3, 1], core.DType.float32)
    assert params[-1] == ("ssm.fourier.w_im", [1, 2, 3, 1], core.DType.float32)


# ---------------------------------------------------------------------------
# M27 固定时间步 SNN 工厂薄绑定。
# ---------------------------------------------------------------------------


def test_lif_cell_python_factory_builds_and_runs(core, nn, allclose):
    """LIFCell 无参数，静态递推输出完整时间轴 spike。"""
    graph = core.Graph("lif_cell_python_binding")
    x = graph.add_graph_input([1, 3, 1], core.DType.float32)
    lif = nn.LIFCell("lif", 1, 3, 1, 0.5, 1.0, 2.0, core.DType.float32)
    assert lif.parameters() == []
    outputs = lif.build(graph, [x], [])
    assert len(outputs) == 1
    values = np.array([[[0.6], [0.7], [0.2]]], dtype=np.float32)
    graph.mark_output(outputs[0])
    result = core.compile(graph, "cpu").run([core.from_numpy(values)])[0].numpy()
    allclose(result, np.array([[[0.0], [1.0], [0.0]]], dtype=np.float32), "float32")


def test_snn_classifier_python_factory_parameter_order_and_execution(core, nn):
    """SnnClassifier 镜像参数先序并执行 Linear-LIF-Linear-时间和流水线。"""
    batch, steps, input_dim, hidden_dim, classes = 1, 2, 2, 3, 2
    graph = core.Graph("snn_classifier_python_binding")
    x = graph.add_graph_input([batch, steps, input_dim], core.DType.float32)
    model = nn.SnnClassifier(
        "snn",
        batch,
        steps,
        input_dim,
        hidden_dim,
        classes,
        0.5,
        0.5,
        2.0,
        False,
        core.DType.float32,
    )
    params = model.parameters()
    assert params == [
        ("snn.input.weight", [input_dim, hidden_dim], core.DType.float32),
        ("snn.output.weight", [hidden_dim, classes], core.DType.float32),
    ]
    param_values = nn.add_parameter_inputs(graph, model)
    outputs = model.build(graph, [x], param_values)
    assert len(outputs) == 1
    graph.mark_output(outputs[0])
    executable = core.compile(graph, "cpu")
    inputs = [
        core.from_numpy(np.array([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32)),
        core.from_numpy(
            np.array([[0.8, -0.6, 0.7], [-0.5, 0.8, -0.4]], dtype=np.float32)
        ),
        core.from_numpy(np.array([[0.2, -0.1], [0.3, 0.4], [-0.2, 0.5]], dtype=np.float32)),
    ]
    result = executable.run(inputs)[0].numpy()
    assert result.shape == (batch, classes)
    assert np.all(np.isfinite(result))


def test_graph_conv_python_factory_builds_and_runs(core, nn):
    """GraphConv 重导出、唯一参数与固定拓扑执行。"""
    graph = core.Graph("graph_conv_python")
    x = graph.add_graph_input([3, 2], core.DType.float32)
    model = nn.GraphConv("g", 3, 2, 2, [0, 1, 2], [1, 2, 0], core.DType.float32)
    assert model.parameters() == [("g.linear.weight", [2, 2], core.DType.float32)]
    params = nn.add_parameter_inputs(graph, model)
    outputs = model.build(graph, [x], params)
    graph.mark_output(outputs[0])
    result = core.compile(graph, "cpu").run(
        [
            core.from_numpy(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)),
            core.from_numpy(np.eye(2, dtype=np.float32)),
        ]
    )[0].numpy()
    np.testing.assert_array_equal(result, np.array([[5, 6], [1, 2], [3, 4]], dtype=np.float32))


def test_hypergraph_conv_python_factory_builds_and_runs(core, nn):
    """HypergraphConv 重导出、唯一参数与节点-超边-节点执行。"""
    graph = core.Graph("hypergraph_conv_python")
    x = graph.add_graph_input([3, 1], core.DType.float32)
    model = nn.HypergraphConv("h", 3, 2, 1, 1, [0, 1, 1, 2], [0, 0, 1, 1], core.DType.float32)
    assert model.parameters() == [("h.linear.weight", [1, 1], core.DType.float32)]
    params = nn.add_parameter_inputs(graph, model)
    outputs = model.build(graph, [x], params)
    graph.mark_output(outputs[0])
    result = core.compile(graph, "cpu").run(
        [core.from_numpy(np.array([[1], [2], [3]], dtype=np.float32)),
         core.from_numpy(np.ones((1, 1), dtype=np.float32))]
    )[0].numpy()
    assert result.shape == (3, 1)
    assert np.all(np.isfinite(result))
