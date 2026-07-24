// DataLoader 的批迭代/洗牌/批组装实现(ARCH-076,见
// include/frame/data/dataloader.h 头注释①~③给出的机械可判定契约)。

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <utility>

#include <frame/data/dataloader.h>

namespace frame::data {

namespace {

// 单样本(axis0 之外全部维)的元素个数;rank() == 1 时一行即单个标量,返回 1。
int64_t RowElementCount(const Shape& shape) {
  int64_t count = 1;
  for (int64_t axis = 1; axis < shape.rank(); ++axis) {
    count *= shape.dim(axis);
  }
  return count;
}

}  // namespace

Result<DataLoader> DataLoader::create(TensorDataset dataset, DataLoaderOptions options) {
  if (options.batch_size < 1) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "DataLoader.create: options.batch_size must be >= 1, got " +
                            std::to_string(options.batch_size));
  }
  DataLoader loader(std::move(dataset), options);
  loader.reset(0);
  return loader;
}

DataLoader::DataLoader(TensorDataset dataset, DataLoaderOptions options)
    : dataset_(std::move(dataset)), options_(options) {
  const int64_t num_samples = dataset_.size();
  batches_per_epoch_ = options_.drop_last
                           ? num_samples / options_.batch_size
                           : (num_samples + options_.batch_size - 1) / options_.batch_size;
}

void DataLoader::reset(uint64_t epoch_index) {
  epoch_index_ = epoch_index;
  next_batch_index_ = 0;
  RebuildIndices();
}

void DataLoader::RebuildIndices() {
  const auto num_samples = static_cast<size_t>(dataset_.size());
  indices_.resize(num_samples);
  std::iota(indices_.begin(), indices_.end(), int64_t{0});
  if (options_.shuffle) {
    std::mt19937_64 rng(options_.seed ^ epoch_index_);
    std::shuffle(indices_.begin(), indices_.end(), rng);
  }
}

Result<std::optional<std::vector<Tensor>>> DataLoader::next(hal::Allocator& allocator) {
  if (next_batch_index_ >= batches_per_epoch_) {
    // 当前 epoch 批序耗尽:推进到下一 epoch 并以空 optional 作边界哨兵返回
    // (头注释①),调用方下一次 next() 调用即取得新 epoch 首批。
    reset(epoch_index_ + 1);
    return std::optional<std::vector<Tensor>>();
  }

  const int64_t batch_start = next_batch_index_ * options_.batch_size;
  const int64_t num_samples = dataset_.size();
  const int64_t batch_len = std::min(options_.batch_size, num_samples - batch_start);

  std::vector<Tensor> batch;
  batch.reserve(dataset_.columns().size());
  for (const Tensor& column : dataset_.columns()) {
    std::vector<int64_t> batch_dims = column.shape().dims();
    batch_dims[0] = batch_len;
    const Shape batch_shape(std::move(batch_dims));
    Result<Tensor> out_result =
        Tensor::empty(batch_shape, column.dtype(), column.device(), allocator);
    if (!out_result.is_ok()) {
      return out_result.status();
    }
    Tensor out = std::move(out_result.value());

    const int64_t row_elems = RowElementCount(column.shape());
    const size_t row_bytes = static_cast<size_t>(row_elems) * column.dtype().itemsize();
    auto* dst = static_cast<std::byte*>(out.raw_data());
    const auto* src = static_cast<const std::byte*>(column.raw_data());
    for (int64_t i = 0; i < batch_len; ++i) {
      const int64_t sample_index = indices_[static_cast<size_t>(batch_start + i)];
      std::memcpy(dst + static_cast<size_t>(i) * row_bytes,
                  src + static_cast<size_t>(sample_index) * row_bytes, row_bytes);
    }
    batch.push_back(std::move(out));
  }

  ++next_batch_index_;
  return std::optional<std::vector<Tensor>>(std::move(batch));
}

}  // namespace frame::data
