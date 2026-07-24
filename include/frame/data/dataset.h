#pragma once
// frame::data 的 Dataset 契约(docs/architecture/nn-design.md ARCH-076,
// ADR-0020 决策 2)。v0 唯一形态 = TensorDataset(内存驻留张量列);Dataset
// 抽象接口非目标(仅一种实现,抽象徒增白名单外 virtual 风险,CPP-010)。
//
// 依赖纪律(ARCH-070):data 目录只依赖 core,不 include
// frame/ir|ops|compiler|runtime|hal|frontend 任何头(hal::Allocator 若需要
// 仅前向声明,同 include/frame/core/tensor.h、
// include/frame/interop/onnx_weights.h 先例;本文件不涉及分配器,故连前向
// 声明都不需要)。

#include <cstdint>
#include <span>
#include <vector>

#include <frame/core/macros.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>

namespace frame::data {

// TensorDataset:一组等长「列」张量(ARCH-076)。各列 axis0 为样本维,要求
// (create() 逐条校验,任一违反返回 kInvalidArgument、英文消息含违例列下标
// 与具体字段):
//   ①columns 非空(至少一列,零列无样本维可言);
//   ②各列 rank() >= 1;
//   ③各列须驻 cpu 后端(Tensor::device().backend == kCpuBackendName)——
//     DataLoader 批组装按行主序直接 memcpy,要求列数据以 host 内存承载;
//   ④各列 axis0 尺寸(Shape::dim(0))相等且 >= 1(以第 0 列为基准比对)。
// 校验通过后为不可变值语义句柄(列张量本身仍是 Tensor 既有的浅拷贝共享
// Storage 语义,拷贝 TensorDataset 不复制底层数值内存)。
class FRAME_API TensorDataset {
 public:
  // 校验 columns(见类头注释①~④)后构造;失败原样返回 Status,不部分构造。
  static Result<TensorDataset> create(std::vector<Tensor> columns);

  // 样本数(= 各列 axis0 尺寸,create() 已保证全列一致)。
  int64_t size() const { return num_samples_; }

  // 只读列视图,顺序与传入 create() 时一致。
  std::span<const Tensor> columns() const { return columns_; }

 private:
  TensorDataset() = default;

  std::vector<Tensor> columns_;
  int64_t num_samples_ = 0;
};

}  // namespace frame::data
