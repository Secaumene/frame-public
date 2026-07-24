"""frame.data:frame::data 数据加载层的薄门面(铁律 #2:纯转发,零逻辑)。

暴露 TensorDataset/DataLoaderOptions/DataLoader(docs/architecture/nn-design.md
§6)。全部符号实际定义于 C++ 扩展 frame._core(python/src/bind_data.cpp),本
模块仅做 re-export;导入失败模式说明同 frame/nn.py 头注释。
"""

from __future__ import annotations

from ._core import DataLoader, DataLoaderOptions, TensorDataset

__all__ = ["DataLoader", "DataLoaderOptions", "TensorDataset"]
