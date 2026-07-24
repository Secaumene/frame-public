"""frame.nn:frame::nn 编译期构图组合子层的薄门面(铁律 #2:纯转发,零逻辑)。

暴露 Module(不透明句柄)与首批模块工厂 Linear/Relu/Sequential/MseLoss,以及
M21 批3 T6 新增的卷积批次工厂 Conv2d/Conv1d/MaxPool2d/AvgPool2d/Sigmoid/
Flatten/AFF/Dwt2d/Dwt1d(docs/plan/2026-07-18-batch3-m21-conv.md 第1.4节)、
M22 批4 T5 新增的序列批次工厂 LayerNorm/LSTM/MultiheadAttention/
TransformerEncoderBlock(docs/plan/2026-07-19-batch4-m22-seq.md §1.7)与
M23 批5 T5 新增的频域批次工厂 SpectralConv1d/FourierFilter1d/Fno1dBlock
(docs/plan/2026-07-21-batch5-m23-fft.md §1.5)、M25 批6 T4 新增的状态空间
模型工厂 Mamba/FourierMamba(docs/plan/2026-07-23-batch6-m25-ssm.md
§1.4)、M27 批8新增的 LIFCell/SnnClassifier(docs/plan/
2026-07-23-batch8-m27-snn.md §1.2/1.3)、M28 批9新增的 GraphConv/
HypergraphConv(docs/plan/2026-07-23-batch9-m28-gnn.md §1.2/1.3),以及
add_parameter_inputs
便捷面(docs/architecture/nn-design.md §6)。全部符号实际定义于 C++ 扩展
frame._core(python/src/bind_nn.cpp),本模块仅做 re-export。

本模块依赖 frame._core 扩展;C++ 扩展未编译时(FRAME_BUILD_PYTHON=OFF)显式
`import frame.nn` 将得到 ImportError——这与顶层 `import frame` 的容错策略
(python/frame/__init__.py 的 _core_available 分支)不同:顶层包始终可导入,
仅在调用点触发 AttributeError(PY-041);本模块属用户主动选择依赖该扩展的
子命名空间,失败即时暴露更符合预期。
"""

from __future__ import annotations

from ._core import (
    AFF,
    AvgPool2d,
    Conv1d,
    Conv2d,
    Dwt1d,
    Dwt2d,
    Flatten,
    Fno1dBlock,
    FourierFilter1d,
    FourierMamba,
    GraphConv,
    HypergraphConv,
    LayerNorm,
    LIFCell,
    Linear,
    LSTM,
    Mamba,
    MaxPool2d,
    Module,
    MseLoss,
    MultiheadAttention,
    Relu,
    Sequential,
    Sigmoid,
    SnnClassifier,
    SpectralConv1d,
    TransformerEncoderBlock,
    add_parameter_inputs,
)

__all__ = [
    "AFF",
    "AvgPool2d",
    "Conv1d",
    "Conv2d",
    "Dwt1d",
    "Dwt2d",
    "Flatten",
    "Fno1dBlock",
    "FourierFilter1d",
    "FourierMamba",
    "GraphConv",
    "HypergraphConv",
    "LayerNorm",
    "LIFCell",
    "Linear",
    "LSTM",
    "Mamba",
    "MaxPool2d",
    "Module",
    "MseLoss",
    "MultiheadAttention",
    "Relu",
    "Sequential",
    "Sigmoid",
    "SnnClassifier",
    "SpectralConv1d",
    "TransformerEncoderBlock",
    "add_parameter_inputs",
]
