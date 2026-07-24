#pragma once
// frame::data 的 DataLoader(docs/architecture/nn-design.md ARCH-076,
// ADR-0020 决策 2)。依赖纪律同 dataset.h(仅依赖 core);hal::Allocator 仅
// 前向声明借用(同 include/frame/core/tensor.h、
// include/frame/interop/onnx_weights.h 先例),本文件不 include hal 头。
//
// 批迭代契约(供调用方与测试对照,机械可判定):
// ①「epoch 边界哨兵」驱动:next() 在当前 epoch 的批序耗尽时返回
//   std::nullopt,并把内部状态原子推进到下一 epoch(epoch 序号自增、按新
//   epoch 序号重新生成样本索引序、批计数归零);下一次 next() 调用即产出新
//   epoch 首批。惯用消费写法:
//     while (true) {
//       Result<std::optional<std::vector<Tensor>>> result = loader.next(allocator);
//       if (!result.is_ok()) { /* 处理错误 */ break; }
//       if (!result.value().has_value()) break;  // 本 epoch 结束(哨兵)
//       std::vector<Tensor>& batch = *result.value();
//       ...
//     }
// ②洗牌确定性:options.shuffle 为真时,每个 epoch 的样本索引序由
//   std::mt19937_64(options.seed ^ epoch_index) 驱动 std::shuffle 生成——
//   同 seed、同 epoch_index 恒产出同一索引序,且与「如何到达该
//   epoch_index」无关(reset() 直接跳转与逐 epoch 顺序推进 next() 结果一致,
//   均只取决于 seed ^ epoch_index);shuffle 为假时索引序恒为
//   [0, 1, ..., size()-1](恒等序)。
// ③批组装:第 b 批(b 从 0 起)覆盖(经②索引序重排后的)样本索引区间
//   [b*batch_size, b*batch_size + 本批长度) 的行切片,按列逐样本 memcpy 进
//   新分配的批张量(行主序 axis0 切片连续,要求列数据为 host 内存——
//   TensorDataset::create 已校验各列驻 cpu 后端);尾批不足 batch_size 时,
//   options.drop_last 为真则整批丢弃(不计入 batches_per_epoch()),为假则
//   原样保留短批(该批长度 = size() % batch_size)。

#include <cstdint>
#include <optional>
#include <vector>

#include <frame/core/macros.h>
#include <frame/core/status.h>
#include <frame/core/tensor.h>
#include <frame/data/dataset.h>

namespace frame::hal {
class Allocator;  // 前向声明:data 公共头不 include hal 头(依赖纪律见上)
}  // namespace frame::hal

namespace frame::data {

// DataLoader 构造选项(ARCH-076,字段与顺序照抄 spec 字面量)。
struct DataLoaderOptions {
  int64_t batch_size = 1;
  bool shuffle = false;
  uint64_t seed = 0;
  bool drop_last = false;
};

// DataLoader:绑定一个 TensorDataset + DataLoaderOptions 的批迭代器(值语义)。
// 仅经 create() 静态工厂取得(校验 options.batch_size >= 1)。
class FRAME_API DataLoader {
 public:
  // 校验 options.batch_size >= 1(否则 kInvalidArgument,英文消息含实际
  // 值);校验通过后绑定 dataset(浅拷贝,同 Tensor 值语义)并就绪 epoch 0
  // 的样本索引序(等价于构造后立即 reset(0))。
  static Result<DataLoader> create(TensorDataset dataset, DataLoaderOptions options);

  // 当前 epoch 序号(create() 后为 0;每次 next() 触发的 epoch 边界哨兵会
  // 使其自增 1,reset(epoch_index) 直接置为给定值)。
  uint64_t epoch() const { return epoch_index_; }

  // 每 epoch 的批次总数(样本数与 options 恒定后此值不随 epoch 变化;
  // drop_last 为真且 size() < batch_size 时可为 0)。
  int64_t batches_per_epoch() const { return batches_per_epoch_; }

  // 产出下一批:std::vector<Tensor>,每列一个批张量,列序与
  // dataset.columns() 一致、批张量 device 与对应列 device 一致(cpu 后端);
  // 当前 epoch 批序耗尽时返回 std::nullopt 并推进到下一 epoch(见头注释①)。
  // allocator 须产出 host 可解引用内存(批组装为直接 memcpy),契约同
  // include/frame/interop/onnx_weights.h::load_onnx_weights——调用方常见
  // 取法为对已注册 cpu 后端取其 allocator(data 目录本身不查
  // BackendRegistry、不 include hal 头)。
  Result<std::optional<std::vector<Tensor>>> next(hal::Allocator& allocator);

  // 显式跳转到指定 epoch(重新按 seed ^ epoch_index 生成样本索引序、批计数
  // 归零);不改变 dataset/options。供需要重放某 epoch 或从任意 epoch 起步
  // 的调用方使用(见头注释②:结果与逐 epoch 顺序推进 next() 一致)。
  void reset(uint64_t epoch_index = 0);

 private:
  DataLoader(TensorDataset dataset, DataLoaderOptions options);

  // 按当前 epoch_index_ 重建 indices_(create()/reset() 共用)。
  void RebuildIndices();

  TensorDataset dataset_;
  DataLoaderOptions options_;
  int64_t batches_per_epoch_ = 0;
  std::vector<int64_t> indices_;
  uint64_t epoch_index_ = 0;
  int64_t next_batch_index_ = 0;
};

}  // namespace frame::data
