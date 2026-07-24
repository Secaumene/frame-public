"""frame._core 类型存根(PY-020,手写全量,随 python/src/ 绑定变更同步维护)。

M12:构图→编译→执行最小闭环 + 七个面向用户算子(PY-021 暂缓期清单销项)。
M17:新增 mse_loss(ARCH-064,PY-021 自 M12 起全量执法,随注册同变更绑定)。
ADR-0013:新增 save_onnx_weights/load_onnx_weights(ONNX 权重导入/导出)。
积压批次②:新增 build_backward_graph/build_sgd_update_graph(训练 API,
docs/architecture/autograd.md 第2/5/6章)。
M20 批2 Task5:新增 Module/Linear/Relu/Sequential/MseLoss/add_parameter_inputs
(frame::nn,docs/architecture/nn-design.md §6)与
TensorDataset/DataLoaderOptions/DataLoader(frame::data,同章);frame.nn/
frame.data 子命名空间门面见 python/frame/nn.py、python/frame/data.py。
M22 批4 T5:DType 新增 int32/int64(决议点A最小接触面,docs/plan/
2026-07-19-batch4-m22-seq.md §1.1);frame.nn 新增 LayerNorm/LSTM/
MultiheadAttention/TransformerEncoderBlock 四工厂(同计划 §1.7)。
M23 批5 T5:frame.nn 新增 SpectralConv1d/FourierFilter1d/Fno1dBlock 三工厂
(docs/plan/2026-07-21-batch5-m23-fft.md §1.5)。
M25 批6 T4:frame.nn 新增 Mamba/FourierMamba 两工厂
(docs/plan/2026-07-23-batch6-m25-ssm.md §1.4)。
M27 批8:新增 heaviside_surrogate 与 frame.nn LIFCell/SnnClassifier
(docs/plan/2026-07-23-batch8-m27-snn.md §1.1~1.3)。
M28 批9:新增 scatter_add 与 frame.nn GraphConv/HypergraphConv
(docs/plan/2026-07-23-batch9-m28-gnn.md §1.1~1.3)。
"""

from __future__ import annotations

import enum
from typing import Any, Sequence

try:
    from numpy.typing import NDArray
except ImportError:  # numpy 为可选测试依赖(pyproject.toml [project.optional-dependencies].test)
    NDArray = Any  # type: ignore[assignment,misc]

class DType(enum.IntEnum):
    """张量元素类型(v0 白名单:float32/float16/bfloat16 三浮点档 + M22 批4
    决议点A新增的 int32/int64 两整数档,docs/plan/2026-07-19-batch4-m22-seq.md
    §1.1)。"""

    float32 = 0
    float16 = 2
    bfloat16 = 3
    int32 = 6
    int64 = 7

class Shape:
    """张量形状,与 list[int] 互转。"""

    def __init__(self, dims: Sequence[int]) -> None:
        """按各维尺寸列表构造。"""

    @property
    def dims(self) -> list[int]:
        """各维尺寸列表。"""

    def __repr__(self) -> str: ...

class Device:
    """设备寻址句柄:后端注册键字符串 + 设备序号。"""

    def __init__(self, backend: str, index: int = 0) -> None:
        """构造设备句柄(backend 须已在 BackendRegistry 注册)。"""

    @property
    def backend(self) -> str:
        """后端注册键。"""

    @property
    def index(self) -> int:
        """设备序号。"""

class Tensor:
    """张量句柄(值语义,实际数据由 C++ 核心持有,铁律 #2)。"""

    @property
    def shape(self) -> Shape:
        """张量形状。"""

    @property
    def dtype(self) -> DType:
        """张量元素类型。"""

    @property
    def device(self) -> Device:
        """张量归属设备。"""

    def numpy(self) -> NDArray[Any]:
        """拷出为一份新 numpy 数组(D2H,v0 拷贝语义;bfloat16 报
        NotImplementedError,numpy 无原生表示;支持 float32/float16/
        int32/int64)。"""

    def to(self, backend: str) -> Tensor:
        """显式搬运到指定后端设备(H2D/D2H/D2D,v0 拷贝语义)。"""

class Value:
    """SSA 值句柄(不透明,仅供构图接线)。"""

class Node:
    """算子节点句柄(不透明;v0 预留符号——当前无任何 API 产出 Node 实例,
    不可构造;M17 反向图/梯度里程碑将提供 Value 到 producer 的只读访问面)。"""

class Graph:
    """静态计算图句柄(构图入口,铁律 #1①)。"""

    def __init__(self, name: str = "") -> None:
        """按名字构造一个空图。"""

    @property
    def name(self) -> str:
        """图名字。"""

    def add_graph_input(
        self, shape: Sequence[int], dtype: DType, backend: str = "cpu"
    ) -> Value:
        """新增一个图输入,返回其唯一输出 Value 句柄(生命周期绑定本图)。"""

    def mark_output(self, value: Value) -> None:
        """登记图输出(value 须属于本图,否则报错)。"""

class Executable:
    """整图编译产物的执行句柄(compile() 的返回值)。"""

    def run(self, inputs: Sequence[Tensor]) -> list[Tensor]:
        """按编译期签名执行整图,预分配并返回输出张量列表。"""

def compile(graph: Graph, backend: str) -> Executable:
    """把图编译为可执行句柄(整图编译路径,铁律 #1①)。"""

def from_numpy(array: NDArray[Any]) -> Tensor:
    """从 numpy 数组拷入一份 cpu Tensor(v0 拷贝语义,仅支持
    float32/float16/int32/int64 且须 C-contiguous)。"""

def add(graph: Graph, lhs: Value, rhs: Value) -> Value:
    """逐元素加法。"""

def mul(graph: Graph, lhs: Value, rhs: Value) -> Value:
    """逐元素乘法。"""

def selective_scan(
    graph: Graph, x: Value, a: Value, b: Value, c: Value, d: Value
) -> Value:
    """沿最后一轴执行选择性状态扫描;五个输入须具有相同静态 shape 与 dtype。"""

def heaviside_surrogate(graph: Graph, x: Value, alpha: float) -> Value:
    """逐元素代理阶跃:x>=0 输出 1,否则输出 0;alpha 控制平滑代理梯度。"""

def scatter_add(
    graph: Graph, updates: Value, indices: Value, output_shape: Sequence[int]
) -> Value:
    """按 indices 将 updates 行累加到静态 output_shape。"""

def relu(graph: Graph, x: Value) -> Value:
    """逐元素 ReLU。"""

def square(graph: Graph, x: Value) -> Value:
    """逐元素平方。"""

def matmul(graph: Graph, lhs: Value, rhs: Value) -> Value:
    """矩阵乘法(rank-2)。"""

def sum(graph: Graph, x: Value, axes: Sequence[int]) -> Value:
    """沿 axes 求和归约(axes 为空表示全维归约)。"""

def mse_loss(graph: Graph, pred: Value, target: Value) -> Value:
    """均方误差损失 mean((pred-target)^2),标量(rank-0)输出。"""

def constant(
    graph: Graph,
    values: Sequence[float],
    shape: Sequence[int],
    dtype: DType,
    backend: str = "cpu",
) -> Value:
    """把常量值物化为图中一个 0 输入节点。"""

def build_backward_graph(
    forward: Graph, loss_output_index: int, wrt_input_indices: Sequence[int]
) -> Graph:
    """由前向图生成反向训练图:loss_output_index 选择一个标量输出;
    图输出依次为 [forward_outputs..., grad(wrt_input_indices[0]),
    grad(wrt_input_indices[1]), ...](原输出前缀保持原序,梯度按给定顺序追加);
    不在 wrt_input_indices 中的 forward 输入按停止梯度处理;
    wrt_input_indices 中不在 loss 依赖链上的输入按惯例补零梯度
    (docs/architecture/autograd.md 第2章)。"""

def build_sgd_update_graph(
    param_types: Sequence[tuple[Sequence[int], DType]],
    learning_rate: float,
    backend: str = "cpu",
) -> Graph:
    """构建 SGD 参数更新图:param_types 为 (shape, dtype) 二元组列表,图输入按位 =
    [param_0..param_{n-1}, grad_0..grad_{n-1}],图输出按位 =
    [new_param_0..new_param_{n-1}],与 param 顺序一一对应;learning_rate 经
    constant 烘焙进图(v0 固定学习率,docs/architecture/autograd.md 第6章)。"""

def save_onnx_weights(path: str, weights: dict[str, Tensor]) -> None:
    """把 dict[str, Tensor] 写为一份最小合法 ONNX ModelProto(仅权重
    initializer 子集,ADR-0013;每张量须 device='cpu' 且 dtype 属
    float32/float16/bfloat16 白名单)。"""

def load_onnx_weights(path: str) -> dict[str, Tensor]:
    """从 path 读取 ONNX 权重 initializer 子集,返回 dict[str, Tensor]
    (顺序与文件中出现顺序一致,ADR-0013)。"""

# ---------------------------------------------------------------------------
# M20 批2 Task5:frame::nn(docs/architecture/nn-design.md §6)。
# 子命名空间门面见 python/frame/nn.py(纯 re-export)。
# ---------------------------------------------------------------------------

class Module:
    """编译期构图组合子(不透明句柄):持 name 与先序遍历得到的参数元信息;
    数值构图经 build() 完成,自身不含任何数值。仅经 Linear/Relu/Sequential/
    MseLoss 等工厂取得。"""

    @property
    def name(self) -> str:
        """模块名字。"""

    def parameters(self) -> list[tuple[str, list[int], DType]]:
        """先序遍历扁平参数清单,元素为 (name, shape, dtype) 三元组(不含初始化
        声明——数值物化非本绑定职责)。"""

    def build(
        self, graph: Graph, inputs: Sequence[Value], params: Sequence[Value]
    ) -> list[Value]:
        """按 parameters() 校验 params 尺寸后构图,返回输出 Value 句柄列表
        (生命周期绑定 graph)。"""

def Linear(
    name: str, batch: int, in_dim: int, out_dim: int, with_bias: bool, dtype: DType
) -> Module:
    """全连接层:matmul(x, weight[in_dim,out_dim])[+ add(., bias[batch,out_dim])]
    (nn-design.md §2)。"""

def Relu(name: str) -> Module:
    """逐元素 ReLU 激活包装(无参数,恰 1 输入 1 输出)。"""

def Sequential(name: str, children: Sequence[Module]) -> Module:
    """顺序组合:按 children 声明序逐个转发 inputs/outputs,params 按子模块
    parameters().size() 先序分段切片。"""

def MseLoss(name: str) -> Module:
    """均方误差损失(无参数,恰 2 输入即 pred/target、恰 1 标量输出)。"""

# M21 批3 T6:卷积批次工厂(docs/plan/2026-07-18-batch3-m21-conv.md 第1.4节)。

def Conv2d(
    name: str,
    in_channels: int,
    out_channels: int,
    kernel_hw: Sequence[int],
    stride_hw: Sequence[int],
    padding_hw: Sequence[int],
    groups: int,
    with_bias: bool,
    dtype: DType,
) -> Module:
    """二维卷积层:ParamSpec weight[out_channels,in_channels/groups,KH,KW] +
    (with_bias 时) bias[out_channels];单个 conv2d 节点(bias 为算子内可选第三
    输入,计划 1.4 节裁决点①)。kernel_hw/stride_hw/padding_hw 均为长度 2 的
    [H, W] 序列。"""

def Conv1d(
    name: str,
    in_channels: int,
    out_channels: int,
    kernel: int,
    stride: int,
    padding: int,
    groups: int,
    with_bias: bool,
    dtype: DType,
) -> Module:
    """一维卷积层,同构 Conv2d(kernel/stride/padding 均标量)。"""

def MaxPool2d(
    name: str, kernel_hw: Sequence[int], stride_hw: Sequence[int], padding_hw: Sequence[int]
) -> Module:
    """二维最大池化(无参数,恰 1 输入 1 输出)。"""

def AvgPool2d(
    name: str, kernel_hw: Sequence[int], stride_hw: Sequence[int], padding_hw: Sequence[int]
) -> Module:
    """二维平均池化(无参数,恰 1 输入 1 输出;分母恒 KH*KW,含 padding)。"""

def Sigmoid(name: str) -> Module:
    """逐元素 sigmoid 激活包装(无参数,恰 1 输入 1 输出,镜像 Relu)。"""

def Flatten(name: str) -> Module:
    """展平层(无参数):从输入静态 shape 算出 [N, prod(其余维)] 并发单个
    reshape 节点。"""

def AFF(name: str, channels: int, dtype: DType) -> Module:
    """AFF 注意力特征融合(local-only 变体,无全局池化分支,计划 1.4 节偏差
    声明):两个 1x1 Conv2d 子模块 c1/c2,恰 2 输入 X/Y、恰 1 输出。"""

def Dwt2d(name: str, channels: int, wavelet_kind: str) -> Module:
    """二维离散小波变换(固定滤波器经 constant 节点物化,不参与训练);
    wavelet_kind 目前仅接受 'haar'。"""

def Dwt1d(name: str, channels: int, wavelet_kind: str) -> Module:
    """一维离散小波变换(固定滤波器经 constant 节点物化,不参与训练);
    wavelet_kind 接受 'haar' 或 'db4'。"""

# M22 批4 T5:序列批次工厂(docs/plan/2026-07-19-batch4-m22-seq.md §1.7)。

def LayerNorm(name: str, dim: int, eps: float, dtype: DType) -> Module:
    """层归一化:ParamSpec gamma[dim]/beta[dim](单 layer_norm 节点内沿行广播);
    输入限 rank-2 [rows, dim],恰 1 输入 1 输出。"""

def LSTM(
    name: str, batch: int, num_steps: int, input_dim: int, hidden_dim: int, dtype: DType
) -> Module:
    """长短期记忆网络:输入 x[batch, num_steps, input_dim],静态展开 num_steps
    步(门序 i,f,g,o 固定),输出末步隐状态 h_T[batch, hidden_dim]。"""

def MultiheadAttention(
    name: str,
    batch: int,
    seq_len: int,
    embed_dim: int,
    num_heads: int,
    with_bias: bool,
    dtype: DType,
) -> Module:
    """多头自注意力:输入 x[batch*seq_len, embed_dim](Linear 2-D 口径),
    children=4 个 Linear(q/k/v/o);经 per-(b,h) 静态展开表达(matmul 维持
    rank-2,不扩批量 matmul);恰 1 输入 1 输出,输出与输入同形。"""

def TransformerEncoderBlock(
    name: str,
    batch: int,
    seq_len: int,
    embed_dim: int,
    num_heads: int,
    ffn_dim: int,
    with_bias: bool,
    dtype: DType,
) -> Module:
    """Transformer 编码器块(Post-LN):children=[mha, ln1, ffn1, ffn2, ln2];
    y1=ln1(x+mha(x));y=ln2(y1+ffn2(relu(ffn1(y1))));恰 1 输入 1 输出。"""

# M23 批5 T5:频域批次工厂(docs/plan/2026-07-21-batch5-m23-fft.md §1.5)。

def SpectralConv1d(
    name: str,
    batch: int,
    in_channels: int,
    out_channels: int,
    n: int,
    modes: int,
    dtype: DType,
) -> Module:
    """谱卷积(FNO 频域算子):输入 x[batch,in_channels,n] -> rfft ->
    slice(前 modes 模态) -> 逐模态静态展开(matmul 维持 rank-2 复乘组合) ->
    零补到 k=n/2+1 -> irfft -> 输出 [batch,out_channels,n]。ParamSpec
    W_re/W_im 各 [in_channels, modes*out_channels];build() 期校验
    modes <= n/2+1;恰 1 输入 1 输出。"""

def FourierFilter1d(name: str, batch: int, channels: int, n: int, dtype: DType) -> Module:
    """频域可学习复滤波器(Fourier Mamba 频域支线):输入
    x[batch,channels,n] -> rfft -> 逐元素复乘(y_re=w_re*x_re-w_im*x_im,
    y_im=w_re*x_im+w_im*x_re) -> irfft -> 输出 [batch,channels,n]。ParamSpec
    w_re/w_im 各 [batch,channels,k=n/2+1,1]——**注意逐样本参数语义**:形状含
    batch 维,是「每样本独立滤波器」而非跨样本共享的滤波器;恰 1 输入 1 输出。"""

def Fno1dBlock(
    name: str,
    batch: int,
    in_channels: int,
    out_channels: int,
    n: int,
    modes: int,
    dtype: DType,
) -> Module:
    """FNO 前向块:children=[SpectralConv1d, Conv1d(kernel=1 逐点旁路)];
    y=tanh(add(spectral(x), conv1x1(x)));恰 1 输入 1 输出。"""

# M25 批6 T4:状态空间模型工厂(docs/plan/
# 2026-07-23-batch6-m25-ssm.md §1.4)。

def Mamba(
    name: str,
    batch: int,
    channels: int,
    steps: int,
    kernel_size: int,
    dtype: DType,
) -> Module:
    """Mamba 块:输入/输出 [batch,channels,steps];depthwise causal Conv1d
    后产生 input/a/b/c/d/gate 六路投影,input/a/b/c/d 进入
    selective_scan,结果经 gate 融合与 out Linear 投影。"""

def FourierMamba(
    name: str,
    batch: int,
    channels: int,
    steps: int,
    kernel_size: int,
    dtype: DType,
) -> Module:
    """Fourier Mamba 融合块:children=[Mamba, FourierFilter1d];两分支输出
    相加后取 tanh,输入/输出均为 [batch,channels,steps]。"""

# M27 批8:固定时间步脉冲神经网络工厂。

def LIFCell(
    name: str,
    batch: int,
    num_steps: int,
    features: int,
    decay: float,
    threshold: float,
    alpha: float,
    dtype: DType,
) -> Module:
    """LIF 单元:输入/输出 [batch,num_steps,features],静态展开并用代理阶跃
    重置膜电位;无参数。"""

def SnnClassifier(
    name: str,
    batch: int,
    num_steps: int,
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    decay: float,
    threshold: float,
    alpha: float,
    with_bias: bool,
    dtype: DType,
) -> Module:
    """固定时间步 SNN 分类器:children=[input,lif,output],输出 logits
    [batch,num_classes]。"""

# M28 批9:固定拓扑图网络工厂。

def GraphConv(
    name: str,
    num_nodes: int,
    in_features: int,
    out_features: int,
    source_indices: Sequence[int],
    target_indices: Sequence[int],
    dtype: DType,
) -> Module:
    """有向图卷积:固定边表归一化消息并 scatter_add 到目标节点；索引须在界且 <=2^53。"""

def HypergraphConv(
    name: str,
    num_nodes: int,
    num_hyperedges: int,
    in_features: int,
    out_features: int,
    node_indices: Sequence[int],
    hyperedge_indices: Sequence[int],
    dtype: DType,
) -> Module:
    """超图卷积:执行 Dv^-1/2 H De^-1 H^T Dv^-1/2 后接 Linear；索引须在界且 <=2^53。"""

def add_parameter_inputs(graph: Graph, module: Module) -> list[Value]:
    """便捷面:取 module.parameters() 批量物化为图输入,返回 Value 句柄列表
    (生命周期绑定 graph)。"""

# ---------------------------------------------------------------------------
# M20 批2 Task5:frame::data(docs/architecture/nn-design.md §6)。
# 子命名空间门面见 python/frame/data.py(纯 re-export)。
# ---------------------------------------------------------------------------

class TensorDataset:
    """一组等长「列」张量(样本维为 axis0),各列须驻 cpu 后端。"""

    def __init__(self, columns: Sequence[Tensor]) -> None:
        """按列张量列表构造(逐条校验:非空/各列 rank>=1/各列驻 cpu 后端/各列
        axis0 尺寸一致)。"""

class DataLoaderOptions:
    """DataLoader 构造选项。"""

    def __init__(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        """构造批迭代选项。"""

class DataLoader:
    """绑定一个 TensorDataset + DataLoaderOptions 的批迭代器(Python 迭代协议,
    __next__ 产出 list[Tensor])。"""

    def __init__(self, dataset: TensorDataset, options: DataLoaderOptions) -> None:
        """校验 options.batch_size >= 1 后就绪 epoch 0。"""

    def __iter__(self) -> DataLoader:
        """返回自身。"""

    def __next__(self) -> list[Tensor]:
        """产出下一批 list[Tensor];当前 epoch 批序耗尽时抛 StopIteration 并推进
        到下一 epoch(dataloader.h 头注释①)。"""
