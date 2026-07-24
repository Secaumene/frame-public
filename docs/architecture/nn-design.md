# nn 模块层与 data 数据加载层设计(M20,ADR-0020)

> **强制等级**:规范(MUST)——本章条文(ARCH-070~076;067~069 刻意跳空
> 预留,防后续实现者误判缺条)是 frame::nn 与 frame::data 的实现契约;
> 与 ADR-0020 冲突时以 ADR 为准并修订本章。
> **相关铁律**:#1 编译优先 / #2 C++ 核心 / #3 后端隔离 / #5 两级复用
> **面向读者**:实现代码的工程师
> **最后更新**:2026-07-23(M22/M23/M25/M27/M28 公共工厂与 Python 面交付注记,
> docs/plan/2026-07-18-batch3-m21-conv.md 第1.4节)

规则条文格式:`【编号】【MUST/MUST NOT/SHOULD】正文。判定方法:<可机械执行的检查>。`

## 1. 定位与分层

- 【ARCH-070】【MUST】`frame::nn`(include/frame/nn/ + src/nn/)依赖集 =
  core + ir + ops,禁止依赖 compiler / runtime / hal / frontend;`frame::data`
  (include/frame/data/ + src/data/)仅依赖 core(hal::Allocator 仅前向声明
  借用,同 include/frame/core/tensor.h 先例)。nn 只产出**前向子图 + 确定性
  有序参数声明**;build_backward_graph / build_sgd_update_graph / compile 的
  调用一律留在调用方(frontend runner、示例、测试)。判定方法:
  `grep -rn "frame/compiler\|frame/runtime\|frame/frontend" include/frame/nn/ src/nn/` 零命中;
  `grep -rn "frame/ir\|frame/ops\|frame/compiler\|frame/runtime\|frame/frontend" include/frame/data/ src/data/` 零命中;
  `bash scripts/check_iron_rules.sh`(core_dirs 已含 src/nn、src/data)OK。
- ARCH-001 分层图增两分支(overview.md 同步):nn 挂 ops 之侧翼(依赖集
  core+ir+ops),data 挂 core 之侧翼(类比 interop);frontend → nn 单向。

## 2. Module 数据模型与组合机制

- 【ARCH-071】【MUST】Module 是**值语义聚合类型**(非虚基类):持
  `name`(ASCII 标识)、直接参数声明 `std::vector<ParamSpec>`、子模块
  `std::vector<Module>`(值语义树)与类型擦除构建器
  `BuildFn = std::function<Result<std::vector<ir::Value*>>(ir::Graph&,
  std::span<ir::Value* const> inputs, std::span<ir::Value* const> params)>`。
  组合禁止白名单外 virtual / dynamic_cast / typeid(CPP-010/011);异构容器
  与 Python 动态组合一律经 BuildFn 闭包(同 KernelFn/GradientFn 先例)。
  **params 切片不变式**:传入某 Module::build 的 params 恰为该 Module
  `parameters()` 的同序全集;组合模块按 [自身直接参数…, 子0 子树参数…,
  子1 子树参数…] 先序分段切片,每段长度 = 对应子 `parameters().size()`。
  判定方法:check_iron_rules.sh 检查 1/2 对 nn 目录通过;切片不变式测试
  (带自身参数 + 多子模块的合成 Module)在仓。
- 【ARCH-072】【MUST】构图一律经 `ops::create_node_with_inferred_types`
  (REUSE-002,与 frontend lowering 同一份 helper);Module::build 及其闭包
  **零 eager、不触数值**——不得创建 Tensor、不得调 kernel、不得读写数值
  内存(参数初值仅作声明,见第 3 节)。判定方法:
  `grep -rn "Tensor::empty\|raw_data\|KernelContext" include/frame/nn/ src/nn/` 零命中
  + 构图纯度测试(build 前后进程无设备内存分配,以 cpu allocator 计数或
  「build 仅产 ir 对象」的白盒断言实现,批2 test-writer 定形)。
- 首批模块工厂(自由函数,返回 Module 值)。**形状口径**:v0 全静态形状
  世界(无动态维),模块工厂显式收其**作用形状**(首批 = `batch` 标量;
  M21+ 需要时扩为 leading shape span,约定不变、无需改章):
  `Linear(std::string name, int64_t batch, int64_t in_dim, int64_t out_dim,
  bool with_bias, DType dtype)`——matmul(x, weight[in,out]);with_bias 时
  add(·, bias),**bias 的 TensorType = [batch, out_dim],照抄现 lowering
  (与输出同形)**——如此参数声明在工厂期即静态确定(ARCH-073)且 golden
  逐字节判据可守(ARCH-074)。广播形态 bias 待 ADR-0009 在 shape_infer
  落地后另批启用并为 bias 减参(签名演化经本章修订;本章第 9 节非目标)。
  `Relu(std::string name)`;`Sequential(std::string name,
  std::vector<Module> children)`——闭包内按子模块参数计数切 params 片段、
  逐子转发 inputs/outputs。

## 3. 参数管理与初始化声明

- 【ARCH-073】【MUST】`ParamSpec{ std::string name; ir::TensorType type;
  InitSpec init; }`;`Module::parameters()` 返回**先序遍历**(自身直接参数
  按声明序,再逐子模块递归)的扁平清单,名字带路径前缀(如
  `mlp.0.weight`,分隔符 `.`,全 ASCII);同一 Module 树任意两次调用结果
  逐位相同(确定性)。`InitSpec` = kUniformSeeded{lo,hi} | kInline{values}
  (对齐 frontend ModelSpec 现有两种初始化;仅声明,数值物化归调用方——
  M20 内 runner 仍走既有 GenerateHostTensorValues 路径不改)。判定方法:
  参数清单确定性测试(两次 parameters() 逐位相等 + golden 名单)。
- 【ARCH-074】【MUST】参数图输入的物化 helper
  `nn::add_parameter_inputs(ir::Graph&, std::span<const ParamSpec>)
  -> Result<std::vector<ir::Value*>>`:按清单序逐个
  `graph.add_graph_input(type)`,返回 Value* 序与清单逐位对应。职责边界
  (design-reviewer 裁定):**损失子图的构图原语归 nn**(如
  `nn::MseLoss(name)` 模块,与其他组合子同级——ADR-0020 判定③字面量含
  mse_loss 的前提);**装配决策归调用方**——是否装配损失、target 从哪来、
  图输入总序([数据输入…, 参数…, target],frontend 维持既有契约)与
  mark_output 均由调用方定。判定方法(lowering 重构验收,含批量参数前置
  带来的发射序重排勘误):①LoweredModel 三元组(param_names/param_types/
  wrt_input_indices)与图输入总序逐位不变;②forward 图与旧实现**拓扑等价**
  ——忽略 value 重编号后,算子节点集(op 类型+属性)与数据流边集逐一对应,
  仅 graph_input 与计算节点的交错位置允许平移(批量参数前置的固有重排);
  ③frame_dslc --run 同 seed 同 final_loss 轨迹;④frontend 四件套与更新后
  golden 全绿(golden 文本随重构一次性重定基并在文件内注明依据)。

## 4. 与 frontend / 训练线的组装关系(Task 4 蓝图)

- frontend `lower_to_graph` 重构为:构造 Module 树(ModelSpec 逐层 →
  Sequential{Linear[,Relu]…})→ 数据图输入 → `add_parameter_inputs` →
  `module.build(...)` → 经 nn::MseLoss 追加损失节点(target)与 mark_output——
  LoweredModel 三元组与图输入总序**逐位不变**,forward 图与旧实现拓扑等价
  (见 ARCH-074 判定方法;dump_text 发射序因批量参数前置而重排,golden
  重定基对照)。
- 【ARCH-075】【MUST】重构完成判据(ADR-0020 判定③):
  `grep -En 'create_node_with_inferred_types\(graph, *"(matmul|add|mul|relu|square|sum|mse_loss|conv1d|conv2d|max_pool2d|avg_pool2d|reshape|sigmoid|tanh|rsqrt|softmax|layer_norm|transpose|concat|slice|gather|rfft|irfft|selective_scan|heaviside_surrogate|scatter_add)"' src/frontend/lowering.cpp`
  命中数 == 0;**算子字面量清单随批次维护**(M21+ 新增网络结构算子经 nn 后
  同步扩清单,维护点=本条 + ADR-0020 判定③;M21 批3 T6 已扩入
  conv1d/conv2d/max_pool2d/avg_pool2d/reshape/sigmoid 六项;M22 批4 T6 扩入
  tanh/rsqrt/softmax/layer_norm/transpose/concat/slice/gather 八项;M23 批5
  T6 扩入 rfft/irfft 两项;M25/M27/M28 分别扩入 selective_scan、
  heaviside_surrogate、scatter_add)。loss 节点经 nn
  的损失构图原语(ARCH-074 裁定归属)构图,不在 lowering 内联字面调用。

## 5. frame::data 设计

- 【ARCH-076】【MUST】v0 形态:`TensorDataset{std::vector<Tensor> columns}`
  ——各列 axis0 为样本维且样本数相等,张量必须驻 cpu 后端
  (`device.backend == kCpuBackendName`,构造时校验,消息英文);
  `DataLoaderOptions{int64_t batch_size; bool shuffle; uint64_t seed;
  bool drop_last;}`;`DataLoader` 每 epoch 产出批序列,`next(hal::Allocator&)`
  逐批组装:按(可洗牌的)样本索引把各列行切片 memcpy 进新批张量(行主序
  axis0 切片连续,逐样本一次 memcpy);洗牌 = std::mt19937_64(seed ^ epoch
  序号) + std::shuffle,同种子同 epoch 同序(确定性);尾批不足 batch_size
  时按 drop_last 丢弃或保留短批。多线程 prefetch / 磁盘格式为非目标
  (v2.0,spec §11)。判定方法:洗牌确定性、批边界(整除/不整除 ×
  drop_last 两态)、批数值正确三组测试在仓。
- 分配器经 `hal::Allocator&` 形参注入(ADR-0020 裁定 b),data 不查
  BackendRegistry、不含任何后端头。

## 6. Python 绑定面

- `frame.nn`:暴露 Linear/Relu/Sequential 工厂与 Module(不透明句柄 +
  parameters() 元信息只读视图);`frame.data`:TensorDataset/DataLoader
  (迭代协议产出张量列表,allocator 由绑定层内部取 cpu 后端注入——绑定层
  属调用方,可依赖 runtime)。迭代语义提示:DataLoader 耗尽抛
  StopIteration 后自动进入下一 epoch(跨 epoch 可再迭代,忠实转发 C++
  契约),与「耗尽后恒终态」的常规 Python 迭代器不同,使用侧按 epoch
  重新 iter 即可。逻辑全在 C++,绑定仅参数/错误转换(铁律 #2,
  PY-010/021 既有纪律);`.pyi` 与 pytest 随批交付。

- **M21 批3 T6 交付注记**(docs/plan/2026-07-18-batch3-m21-conv.md 第1.4节):
  `frame.nn` 另暴露 Conv2d/Conv1d/MaxPool2d/AvgPool2d/Sigmoid/Flatten/AFF/
  Dwt2d/Dwt1d 九个工厂。其中两项口径需使用侧留意,防误期:
  ①`nn.AFF` 是 **local-only 变体**——原文 MS-CAM 的全局分支(全局池化→跨
  空间广播)因 v0 无图级广播(ADR-0009 待实现落地)未实现,`nn.AFF` 仅含
  局部注意力路径(两个 1x1 Conv2d + sigmoid 门控),不是完整 MS-CAM;
  全局分支留待 ADR-0009 广播落地后另批补齐。
  ②`nn.Dwt2d`/`nn.Dwt1d` 的滤波器是**固定系数、不参与训练**的
  constant 节点(非 ParamSpec)——`parameters()` 对这两个工厂返回空列表,
  不会被优化器更新;`wavelet_kind` 取值经 Python 侧字符串
  `"haar"`/`"db4"` 传入(C++ 侧对应 `nn::WaveletKind` 封闭枚举),`Dwt2d`
  在 v1 仅接受 `"haar"`(`"db4"` 需要可分离两趟卷积,留后续批次)。

- **M22/M23/M25/M27/M28 交付注记**:除既有工厂外，`frame.nn` 已提供
  LayerNorm、LSTM、MultiheadAttention、TransformerEncoderBlock、SpectralConv1d、
  FourierFilter1d、Fno1dBlock、Mamba、FourierMamba、LIFCell、SnnClassifier、
  GraphConv 与 HypergraphConv；`frame._core` 另提供 selective_scan、
  heaviside_surrogate 与 scatter_add 的薄绑定，`.pyi` 同步公开签名。M25–M28
  已完成 CPU/CUDA/Python 最终验收；这些工厂均只构建静态 IR，不改变
  ARCH-070~076 的契约。

## 7. 构建接线与机械检查

- CMake:`frame_nn` STATIC 链 `frame::ops`,`frame_data` STATIC 链
  `frame::core`;二者无静态注册 TU,**不进 WHOLE_ARCHIVE**(同
  frame_frontend/interop 口径);聚合库 frame::frame 与 install 导出集
  (cmake/frame_install.cmake,BUILD-040/041)同步纳入;TOP-001/002 模块
  清单表述同步修订(属规范修订,随实现同 PR)。
- scripts/check_iron_rules.sh 第 128 行 core_dirs 循环增 `src/nn src/data`
  并追注 ADR-0020(与实现同 PR,ADR-0020 放行条件)。

## 8. 测试要求

nn:构图纯度(ARCH-072)/参数清单确定性与路径命名 golden(ARCH-073)/
Linear-Sequential 构图与手工构图 dump_text 逐字节等/端到端:nn 构图 +
调用方组合 build_backward_graph + SGD 单步收敛冒烟(证明 nn 不依赖
compiler 仍可被训练线消费)。data:第 5 节三组。frontend:Task 4 golden
回归 + frame_dslc --run 同 seed 同轨迹。容差一律经 BUILD-011 载体。

## 9. 非目标(启动前置 = 新 ADR 或对应批次)

广播形态 bias(前置:ADR-0009 在 elementwise shape_infer 落地——该 ADR
已接受但实现待批,本章不代拉);卷积/序列等模块(M21+ 逐批);优化器状态
对象(v1.x 议题池);DataLoader prefetch/磁盘格式(v2.0);Module 序列化。
