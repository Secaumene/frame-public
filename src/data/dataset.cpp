// TensorDataset::create 的校验实现(ARCH-076,见 include/frame/data/dataset.h
// 头注释①~④)。

#include <string>
#include <utility>

#include <frame/data/dataset.h>

namespace frame::data {

Result<TensorDataset> TensorDataset::create(std::vector<Tensor> columns) {
  if (columns.empty()) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "TensorDataset.create: columns must not be empty");
  }

  int64_t num_samples = -1;
  for (size_t i = 0; i < columns.size(); ++i) {
    const Tensor& column = columns[i];
    if (column.shape().rank() < 1) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "TensorDataset.create: column " + std::to_string(i) + " has rank " +
                              std::to_string(column.shape().rank()) + ", expected rank >= 1");
    }
    if (column.device().backend != kCpuBackendName) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "TensorDataset.create: column " + std::to_string(i) +
                              " resides on backend '" + std::string(column.device().backend) +
                              "', expected '" + std::string(kCpuBackendName) + "'");
    }
    const int64_t column_samples = column.shape().dim(0);
    if (column_samples < 1) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "TensorDataset.create: column " + std::to_string(i) +
                              " has axis0 sample count " + std::to_string(column_samples) +
                              ", expected >= 1");
    }
    if (num_samples < 0) {
      num_samples = column_samples;
    } else if (column_samples != num_samples) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "TensorDataset.create: column " + std::to_string(i) +
                              " has axis0 sample count " + std::to_string(column_samples) +
                              ", expected " + std::to_string(num_samples) + " to match column 0");
    }
  }

  TensorDataset dataset;
  dataset.columns_ = std::move(columns);
  dataset.num_samples_ = num_samples;
  return dataset;
}

}  // namespace frame::data
