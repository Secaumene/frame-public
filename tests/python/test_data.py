"""frame.data pybind11 绑定 pytest(M20 批2 Task5,python/src/bind_data.cpp;
权威契约见 include/frame/data/{dataset.h,dataloader.h}、
docs/architecture/nn-design.md §6)。PY-021「绑定即测」:TensorDataset 构造、
DataLoaderOptions 构造、DataLoader 迭代协议(批数/形状/数值/shuffle 确定性/
drop_last 两态/StopIteration 语义)均至少 1 例。

数值口径:DataLoader 批组装为逐行 memcpy(不涉及浮点运算,见
tests/cpp/data/data_test_helpers.h 头注释),故用精确相等(np.array_equal),
不接入 BUILD-011 近似容差工具。
"""

import numpy as np

_NUM_SAMPLES = 10
_FEATURE_COLS = 3
_TARGET_COLS = 1
_BATCH_SIZE = 4  # 10 / 4 -> 批长 4, 4, 2


def _build_dataset(core, data):
    """features[row,col] = row*10+col+1;targets[row,0] = 1000+row(与
    tests/cpp/data/test_dataloader_batch_values.cpp 同一源行公式,便于交叉核对)。
    """
    rows = np.arange(_NUM_SAMPLES, dtype=np.float32).reshape(-1, 1)
    cols = np.arange(_FEATURE_COLS, dtype=np.float32).reshape(1, -1)
    features_np = rows * 10.0 + cols + 1.0
    targets_np = 1000.0 + rows
    dataset = data.TensorDataset([core.from_numpy(features_np), core.from_numpy(targets_np)])
    return dataset, features_np, targets_np


def test_dataloader_shuffle_false_batch_count_shapes_and_values_match_numpy_slices(core, data):
    """shuffle=False:批数/形状/数值均与「按源行顺序切片」的 numpy 结果精确相等。"""
    dataset, features_np, targets_np = _build_dataset(core, data)
    options = data.DataLoaderOptions(batch_size=_BATCH_SIZE, shuffle=False, seed=0, drop_last=False)
    loader = data.DataLoader(dataset, options)

    batches = list(loader)
    assert len(batches) == 3  # ceil(10/4) = 3,批长 4,4,2

    next_row = 0
    for features_batch, targets_batch in batches:
        batch_len = features_batch.shape.dims[0]
        assert targets_batch.shape.dims[0] == batch_len
        expected_features = features_np[next_row : next_row + batch_len]
        expected_targets = targets_np[next_row : next_row + batch_len]
        assert np.array_equal(features_batch.numpy(), expected_features)
        assert np.array_equal(targets_batch.numpy(), expected_targets)
        next_row += batch_len
    assert next_row == _NUM_SAMPLES


def test_dataloader_shuffle_true_same_seed_two_loaders_produce_identical_batches(core, data):
    """shuffle=True 且同 seed:两个独立 DataLoader 实例逐批内容(数值)完全相等
    (洗牌确定性)。"""
    dataset, _, _ = _build_dataset(core, data)
    options = data.DataLoaderOptions(batch_size=_BATCH_SIZE, shuffle=True, seed=777, drop_last=False)

    loader_a = data.DataLoader(dataset, options)
    loader_b = data.DataLoader(dataset, options)
    batches_a = list(loader_a)
    batches_b = list(loader_b)

    assert len(batches_a) == len(batches_b) == 3
    for (features_a, targets_a), (features_b, targets_b) in zip(batches_a, batches_b):
        assert np.array_equal(features_a.numpy(), features_b.numpy())
        assert np.array_equal(targets_a.numpy(), targets_b.numpy())

    # 洗牌确须真正打乱了顺序(与 shuffle=False 的结果不同,排除"恒等序"假阳性)。
    unshuffled_first_row = batches_a[0][0].numpy()[0]
    assert not np.array_equal(unshuffled_first_row, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_dataloader_drop_last_false_keeps_short_tail_batch(core, data):
    """drop_last=False:尾批长度 = 10 % 4 = 2,不丢弃。"""
    dataset, _, _ = _build_dataset(core, data)
    options = data.DataLoaderOptions(batch_size=_BATCH_SIZE, shuffle=False, seed=0, drop_last=False)
    loader = data.DataLoader(dataset, options)
    batches = list(loader)
    assert [b[0].shape.dims[0] for b in batches] == [4, 4, 2]


def test_dataloader_drop_last_true_discards_short_tail_batch(core, data):
    """drop_last=True:尾批(长度 2 < batch_size)被整批丢弃,只剩 2 个满批。"""
    dataset, _, _ = _build_dataset(core, data)
    options = data.DataLoaderOptions(batch_size=_BATCH_SIZE, shuffle=False, seed=0, drop_last=True)
    loader = data.DataLoader(dataset, options)
    batches = list(loader)
    assert [b[0].shape.dims[0] for b in batches] == [4, 4]


def test_dataloader_iteration_raises_stop_iteration_at_epoch_end_then_advances_epoch(core, data):
    """StopIteration 语义:显式调用 __next__ 耗尽本 epoch 后抛 StopIteration;
    下一轮 for 循环(隐式重新 __iter__ + __next__)直接产出新 epoch 首批
    (dataloader.h 头注释①:next() 已在返回哨兵的同时把状态推进到下一 epoch)。
    """
    dataset, features_np, _ = _build_dataset(core, data)
    options = data.DataLoaderOptions(batch_size=_BATCH_SIZE, shuffle=False, seed=0, drop_last=False)
    loader = data.DataLoader(dataset, options)

    it = iter(loader)
    for _ in range(3):
        next(it)
    try:
        next(it)
        raise AssertionError("expected StopIteration at epoch end")
    except StopIteration:
        pass

    # 下一轮迭代:新 epoch 首批与 epoch 0 首批(shuffle=False 恒等序)内容相同。
    second_epoch_first_batch = next(iter(loader))
    assert np.array_equal(second_epoch_first_batch[0].numpy(), features_np[0:4])
