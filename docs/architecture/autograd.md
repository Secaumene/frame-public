# 自动微分与训练架构(编译期反向图)

> **强制等级**:规范(MUST)
> **相关铁律**:#1 编译优先 / #5 两级复用
> **面向读者**:实现代码的工程师
> **最后更新**:2026-07-23(M26 高阶自动微分设计;M27 代理梯度窄例外)

本文档是 ADR-0008(训练能力纳入产品范围,已接受)的架构细化:反向模式自动
微分实现为**编译期图→图变换**,运行时 tape/逐算子记录不作为主线(铁律 #1、
ADR-0001)。规则编号前缀 `ARCH-06x`。

## 1. 总体形态:训练图 = 前向图的派生图

```
用户前向图 forward(inputs..., params...) → outputs
        │ compiler::build_backward_graph(forward, loss_index, wrt_indices)
        ▼
训练图 training(inputs..., params...) → [forward_outputs..., grad_0, grad_1, ...]
        │ runtime::compile(training, backend)   ← 复用九段管线与全部既有基建
        ▼
Executable(每步 run 产出 loss 与全部梯度)
```

- 【ARCH-060】【MUST】反向图生成是**独立的图→图变换函数**(构图期入口),
  不是标准管线 pass:训练图构建由用户意图显式触发,而优化 pass 对一切图
  无条件生效,二者语义不同;标准管线九段顺序不动(ARCH-053 不触发,无 ADR
  情形 7)。产出的训练图经既有 `runtime::compile` 走完整管线与缓存,对
  runtime/后端完全透明。判定方法:`include/frame/compiler/` 下的
  build_backward_graph 声明不出现在 `pipeline.h` 九段清单;code-reviewer
  对把反向生成挂入 PassManager 的 diff 按本条打回。
- 落点:`include/frame/compiler/autograd.h` + `src/compiler/autograd.cpp`
  (compiler 层图→图变换,依赖 ir/ops 合法;不新增一级子目录,ARCH-003 不
  触发)。

## 2. 反向图生成入口契约

```cpp
// compiler 层(M17 交付)。forward 不被修改(纯函数,ARCH-021 精神)。
Result<ir::Graph> build_backward_graph(
    const ir::Graph& forward,
    int32_t loss_output_index,              // 前向图输出列表中的 loss 下标
    std::span<const int32_t> wrt_input_indices);  // 对哪些 graph_input 求梯度
```

- 【ARCH-061】【MUST】loss 必须是**标量**输出(shape rank 0 或
  numel==1;非标量返回错误含实际 shape);`wrt_input_indices` 显式给出求导
  对象(**参数/停止梯度不引入任何 IR 标记**——不在列表中即停止梯度;避免
  触碰 Value/Attribute 对象模型,ADR-010 情形 3 不触发)。判定方法:ir 层
  diff 不出现 requires_grad/trainable 类字段;非标量 loss 的错误路径有用例。
  M26 起允许 `forward` 有一个或多个已标记输出,由
  `loss_output_index` 选择其中的标量目标;`ir::clone_graph` 保真复制全部
  原输出,生成器只在其后追加梯度,故结果固定为
  `[forward.outputs()..., grad(wrt[0]), grad(wrt[1]), ...]`。单输出调用的
  既有 `[loss, grad...]` 布局逐字不变。该扩展不新增输出裁剪 API、不修改
  IR 对象模型,同时使一阶梯度可作为后续标量化节点的输入再执行同一变换。
  判定方法:tests/cpp/ 存在「多输出 forward 选择其中任一标量 loss → 原输出
  顺序保留且梯度追加」用例。(M26 设计门)
- 生成算法(实现约定,M17):①按拓扑序克隆前向图——**克隆工具单份共享
  (REUSE-002)**:把 `runtime::compile` 现私有 clone_graph 提升为 ir 层公开
  工具 `ir::clone_graph(const Graph&, std::unordered_map<const Value*,
  Value*>* value_map = nullptr)`(可选输出源→克隆 Value 映射,反向生成器
  据此挂梯度节点;runtime 与 compiler 共用这一份,禁止第二份复制);②loss 输出接种子梯度
  `constant(1)`(同 dtype,标量);③按**逆拓扑序**对每个参与 loss 依赖链
  的节点查其 OpSchema 的 GradientFn(见第 3 章),把梯度微图**内联展开**进
  训练图(微图 graph_input 按第 3 章约定绑定实际 Value);④同一 Value 被
  多消费者使用时,各分支梯度经 `add` 累加(SSA 下的多路径梯度求和);
  ⑤`mark_output`:clone 阶段登记的全部 `forward.outputs()` 原序保留且不重复
  标记,`wrt_input_indices` 各梯度按给定顺序随后;某 wrt 不在 loss 依赖
  链上(其克隆值无累加梯度)时,补一份**同型 `constant(0)`** 作为该位梯度
  ——标准反向模式惯例,且保持「每个 wrt 恰好一个梯度输出、按 wrt 顺序
  按位对齐」的输出契约(否则输出个数随连通性浮动,破坏
  `[forward_outputs..., grad_0, grad_1, ...]` 按位约定)。(M17 实现裁决,
  M26 向后兼容扩展)
- 不参与 loss 依赖链的节点不生成反向(死代码由既有 DNE pass 兜底清理)。
- 【ARCH-062】【MUST】链上任一算子的 OpSchema 未注册 GradientFn 即整体
  返回错误(消息含算子名;不静默跳过——静默会产出错误梯度)。带
  kHasSideEffect trait 的算子不可微,注册 GradientFn 时启动期 fail-fast。
  判定方法:tests/cpp/ 存在「链上含无梯度算子 → 报错含算子名」用例。

## 3. 梯度注册面:OpSchema 扩展字段 GradientFn

- 【ARCH-063】【MUST】梯度规则经 `OpSchema` 新字段注册(与 shape_infer/
  decomposition 同族——梯度是算子级语义知识,归属 schema;**不新建第五个
  注册表**,M6 注册表家族收敛冻结继续有效):

```cpp
// include/frame/ops/op_schema.h(M17 交付;与 DecomposeFn 同形态先例)
using GradientFn = Result<ir::Graph> (*)(const NodeContext&);
OpSchema& gradient(GradientFn fn) noexcept;   // builder
GradientFn gradient() const;                  // 访问器,未设置为 nullptr
```

  判定方法:`grep -n "GradientFn" include/frame/ops/op_schema.h` 命中;无
  独立 GradientRegistry 头文件。
- **梯度微图按位契约**(与 DecomposeFn 的「graph_inputs 按位对应」同纪律):
  设前向算子 n 输入 m 输出,微图 graph_inputs 按位 =
  `[x_0..x_{n-1}, y_0..y_{m-1}, gy_0..gy_{m-1}]`(前向输入、前向输出、输出
  梯度;不需要的位也必须占位声明,生成器内联时按需绑定——恒定布局免除
  逐算子协商);图输出按位 = `[gx_0..gx_{n-1}]`(对每个前向输入的梯度,
  v0 要求全部输入位均产出梯度)。同一 Value 可重复 mark_output(M2 既有
  能力,如 add 的两个输入梯度都是 gy 本身)。
  **类型来源与有效性模型**:y/gy 位的 TensorType 在 GradientFn 内经本算子
  自身 `shape_infer(ctx)` 重算获得(ARCH-041 保证该函数存在;gy 类型恒等于
  对应 y;dtype/device 沿用第 0 输入,与 shape_inference 校验模式口径一致)
  ——**不扩展 NodeContext**,GradientFn 所需的全部上下文由既有 NodeContext
  + 既有 shape_infer 闭合,这正是「与 DecomposeFn 同形态」论证的完整版:
  二者入参同型,GradientFn 只是多消费一次本 schema 已有的推断函数。微图是
  **独立合法可执行图**(与 DecomposeFn 同纪律:具体类型、可过 verify),
  非占位模板;生成器内联时按 Value 映射改接线,不改类型。
- 微图内可使用任何已注册算子,包括为梯度专设的 `*_grad_internal` 算子
  (`_internal` 后缀:不面向用户、不入绑定面,PY-021 天然豁免;每个都按
  ARCH-041 携带 shape_infer 与 cpu 参考 kernel)。

## 4. v0 梯度清单(M17 交付物;数学定义随实现写入各 schema 注释)

| 前向算子 | 梯度构成 | 需新增的 internal 算子 |
|---|---|---|
| add(a,b) | ga=gy;gb=gy(重复 mark_output) | 无 |
| mul(a,b) | ga=mul(gy,b);gb=mul(gy,a) | 无 |
| square(x) | gx=mul(gy, mul(constant(2), x)) | 无 |
| relu(x) | gx=relu_grad_internal(x, gy)(x>0 处透传 gy,余 0) | relu_grad_internal |
| matmul(a,b) | ga=matmul_grad_lhs_internal(gy,b)=gy·bᵀ;gb=matmul_grad_rhs_internal(a,gy)=aᵀ·gy(kernel 内转置索引,不物化 transpose) | matmul_grad_{lhs,rhs}_internal |
| sum(x,axes) | gx=sum_grad_internal(gy)(沿归约轴复制展开回输入 shape;输入 shape 经 kShape attr、归约 axes 经 kInt64Array attr 携带——gy 已丢失被归约维,仅凭输入 shape 不足以消除同尺寸维歧义) | sum_grad_internal |
| mse_loss(pred,target) | gpred=mse_loss_grad_internal(pred,target,gy)=2·(pred−target)/N·gy;gtarget 同式反号 | mse_loss_grad_internal |

- 【ARCH-064】【MUST】`mse_loss(pred, target) → 标量`是**独立注册算子**
  (schema + shape_infer + cpu kernel,面向用户、入绑定追加清单),不以
  sub/mean 组合表达——v0 算子集无 sub/mean,为损失专门引入通用算子属
  过度设计,损失核心化实现更短更稳(REUSE-011 参考实现口径)。判定方法:
  M17 后 `OpRegistry::find("mse_loss")` 非空且带 GradientFn。
- 新增算子的梯度覆盖义务(ADR-0008 后果条):自 M17 起,新增面向用户算子
  时 PR 必须声明「已注册 GradientFn」或「不可微/暂不支持训练,原因」;
  code-reviewer 按本行检查。

## 5. 高阶自动微分(M26)

- 【ARCH-067】【MUST】高阶导数不引入 tape、运行时特例或第二套自动微分
  引擎;它由**同一个 `build_backward_graph` 对派生图再次执行**得到。设原图
  有 p 个输出、首次 wrt 列表有 q 项,则一阶图的第 `p+j` 个输出是
  `grad(wrt[j])`。若该梯度非标量,调用方须先在一阶图中以已注册可微算子构造
  一个标量收缩(通常是 `sum`)并 `mark_output`,再以这个新输出的下标调用
  `build_backward_graph`。判定方法:实现中不存在 `build_second_backward_*`
  专用遍历或 runtime tape;二阶用例连续调用同一公开函数两次。
- 标量收缩得到的是向量-雅可比积。对配点间独立的 PINN(算子只混合每个样本
  内部特征),`sum(u)` 的一阶梯度及 `sum(du/dx)` 的二阶梯度逐位就是点式
  导数。一般耦合函数的完整 Hessian 不由一次全一收缩给出;调用方须按所需
  方向/基向量分别构造标量收缩,禁止把 Hessian 行和静默宣称为完整 Hessian。
  **PINO/FNO 不适用该坐标输入技巧**:其 `[B,C,N]` 输入是离散场,对输入求导
  得到的是离散算子敏感度而非空间导数。PINO 的固定周期网格物理残差必须复用
  rfft/irfft,以常量波数在打包实虚轴上组合谱微分(二阶乘 `-k²`)后再 irfft;
  该链自身可微,参数梯度仍由本章同一反向变换生成,不新增前向算子。
- 【ARCH-068】【MUST】可被再次变换的一阶图必须满足**梯度闭包**:其 loss
  依赖链上的每个非零输入节点均注册 GradientFn,且该 GradientFn 微图内的每个
  非零输入节点也满足同一条件;零输入 constant 按既有豁免。M26 补齐 M17
  遗留的五个 internal GradientFn,数学契约如下:

| internal 算子 | 二阶微图输出(按原输入位置) |
|---|---|
| `sum_grad_internal(gy; input_shape, axes)` | `ggy=sum(ggx, axes, keepdims)`;`keepdims` 由 gy shape 与 input_shape/axes 恢复并校验 |
| `matmul_grad_lhs_internal(gy,b)=gy·bᵀ` | `g_gy=matmul(gga,b)`;`g_b=matmul_grad_rhs_internal(gga,gy)` |
| `matmul_grad_rhs_internal(a,gy)=aᵀ·gy` | `g_a=matmul_grad_lhs_internal(gy,ggb)`;`g_gy=matmul(a,ggb)` |
| `mse_loss_grad_internal(pred,target,gy)` | `g_pred=(2/N)·expand(gy)·ggpred`;`g_target=-g_pred`;`g_gy=sum((2/N)·(pred-target)·ggpred)` |
| `relu_grad_internal(x,gy)` | `g_x=0`(ReLU 二阶导在 kink 外为零、kink 处亦取零);`g_gy=relu_grad_internal(x,ggx)` |

  `expand(gy)` 复用 `sum_grad_internal` 的全轴展开,不新增广播算子。M21–M25
  已有 conv/pool/shape/sequence/gather/FFT/selective_scan 内部闭包继续由
  `tests/cpp/ops/test_gradient_closure.cpp` 统一机械枚举;M26 后该测试还须覆盖
  上表五项并遍历其微图。`fused_elementwise_internal` 是标准编译管线在自动
  微分变换**之后**才产生的优化产物,不是可接受的用户/GradientFn 构图入口;
  手工把它放入待求导图仍按 ARCH-062 fail-loud,本批不为编码后的融合链另造
  一套符号求导器。
- `sum_grad_internal` 的微图必须同时覆盖原 `sum` 的 keepdims=false/true;
  前者 gy rank 删除归约轴,后者 gy rank 与 input rank 相同且归约轴尺寸为 1。
  由这两种合法 shape 恢复 keepdims,不新增或猜测默认属性。
- 【ARCH-069】【MUST】二阶数值验收至少含:解析多项式的二阶导数、包含
  matmul/sum/mse_loss 的组合链、ReLU 远离 kink 的零二阶导、一个断开 wrt 的
  零梯度路径,以及 1D PINN 固定配点二阶物理残差与 PINO 固定周期网格谱微分
  物理残差训练闭环。解析二阶结果须按 BUILD-011 与二次中心差分对照;谱微分
  须与固定解析模态对照;所有派生图均须通过 verify 与完整编译管线。

## 6. 优化器与训练闭环(M18 交付)

- 【ARCH-065】【MUST】v0 优化器 = **独立更新图**(拒绝图内原位更新:v0 无
  原位/别名语义——M9 memory_planning 口径「所有 kernel 分配新输出」,图内
  更新违 SSA)。SGD 更新图纯用既有算子组合,无需新 kernel:
  `new_param = add(param, mul(constant(-lr), grad))`。
  判定方法:M18 的更新图构建 helper 不注册任何新 kernel。
- 训练循环形态(host 驱动,调用方持有参数张量):
  每步 ①`run(训练图, [inputs, params...]) → [loss, grads...]`;
  ②`run(更新图, [params..., grads...]) → [new_params...]`;③参数指针轮换。
  两图 shape 恒定 ⇒ 编译缓存恒命中,每步零重编译(铁律 #1④ 收益落地)。
- 更新图构建 helper:`compiler::build_sgd_update_graph(param_types, lr)`
  (M18 定签名细节);学习率经 constant 烘焙进图(v0;变 lr 是后续议题,
  届时按「lr 作为图输入」扩展,不动本契约)。
- M18 验收基准(ADR-0008):小规模 MLP 回归任务在 cpu 参考后端经编译路径
  训练,固定种子下 loss 单调下降并低于阈值;阈值与种子写入测试注释。

## 7. 数值验证纪律(M17/M18/M26 测试契约)

- 【ARCH-066】【MUST】每个 GradientFn 必须配**解析梯度 ≡ 数值微分**用例:
  中心差分 `(f(x+h)−f(x−h))/2h`;h 与容差**按 BUILD-011 的「解析梯度 ≡
  数值微分校验」专款执行**(该款是数值微分放宽的唯一授权来源,本文档仅
  引用不另设口径;fp16/bf16 梯度经 fp32 解析参照验证)。逐算子单测 +
  组合图(matmul+add+relu+mse_loss 全链)两级覆盖。判定方法:tests/cpp/
  对应用例存在且通过(ADR-0008 判定方法)。
  唯一窄例外:`heaviside_surrogate` 的 GradientFn 是 M27 明定的平滑代理导数,
  不等于离散阶跃前向的分布导数;它必须按 BUILD-011「代理梯度窄例外」同时
  校验闭式公式与 `sigmoid(alpha*x)` 平滑代理中心差分,高阶导数继续变换该
  可微代理微图。该例外不适用于任何其他 op,也不改变未命中算子的本条要求。
- 训练图自身经既有 verify/管线全绿(反向节点是普通节点,V1–V7 无特殊化)。

## 8. 实现待办(M17/M18/M26 销项锚点)

- 已完成(M17):OpSchema::GradientFn 字段与 builder/访问器(include/frame/ops/op_schema.h;注册读回与 kHasSideEffect fail-fast 用例在 tests/cpp/ops/)。
- 已完成(M17):ir::clone_graph 提升共享(全仓单份)且 compiler::build_backward_graph 落地(src/compiler/autograd.cpp);两级数值微分用例见 tests/cpp/compiler/test_autograd.cpp。
- 已完成(M17):第 4 章清单六个新算子与七个 GradientFn 全部注册(mse_loss 面向用户并已绑定 Python,PY-021);数值用例全绿。
- 已完成(M18):build_sgd_update_graph(纯组合,零新 kernel,ARCH-065 判定满足)与端到端 MLP 收敛用例(tests/cpp/compiler/test_training_loop.cpp,seed=20260713,300 步 loss 0.2123→5.63e-05);README/execution-model 训练路径已回写。
- 已完成(M26):放宽多输出图的 loss 选择并保持原输出前缀;补齐第 5 章五个
  internal GradientFn;同一 `build_backward_graph` 连续变换的多项式/非方阵
  matmul/断开支路数值验收、PINN 三次变换训练与 PINO 周期谱二阶导训练闭环
  均落在 `tests/cpp/compiler/`，梯度闭包清单同步扩项。

## 9. 明确不做什么(防蔓延;启动前置均为新 ADR 或后续里程碑裁决)

完整 Hessian/Jacobian 的一次性物化 API、动态 lr/优化器状态(momentum/Adam)、
梯度检查点、混合精度训练、广播语义
(方向已裁 ADR-0009,实施另立里程碑;梯度累加依赖 v0 同 shape 约束成立)、
输出裁剪/重排 API——以上任何一项不在 M26 范围。

`build_backward_graph` 与 `build_sgd_update_graph` 的 Python 薄绑定已随 M18/M26
公共训练图能力同步提供；它们只暴露 C++ 图变换，不在 Python 层承载训练逻辑。
