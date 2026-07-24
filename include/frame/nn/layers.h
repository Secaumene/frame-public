#pragma once
// frame::nn 首批模块工厂(docs/architecture/nn-design.md §2):Linear/Relu/
// Sequential/MseLoss;以及 M21 批3 T6 新增的卷积批次工厂(docs/plan/
// 2026-07-18-batch3-m21-conv.md 第1.4节,design-reviewer 两轮 APPROVE):
// Conv2d/Conv1d/MaxPool2d/AvgPool2d/Sigmoid/Flatten/AFF/Dwt2d/Dwt1d；M22/M23
// 序列与频域工厂；M25 的 Mamba/FourierMamba；M27 的 LIFCell/
// SnnClassifier；M28 的 GraphConv/HypergraphConv。均为
// 返回 Module 值的自由函数,构图统一经 ops::create_node_with_inferred_types
// (REUSE-002,与 frontend lowering 同一份 helper);零 eager、不触数值
// (ARCH-072)。

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <frame/core/dtype.h>
#include <frame/core/macros.h>
#include <frame/nn/module.h>

namespace frame::nn {

// 全连接层:matmul(x, weight[in_dim, out_dim])[+ add(matmul 输出, bias)]。
// 形状口径(v0 全静态形状世界,无动态维;design-reviewer 定点裁决,
// nn-design.md §2 修订版):batch 是工厂显式收的**作用形状**标量(M21+ 需要时
// 扩为 leading shape span,约定不变、无需改章);bias 的 TensorType =
// [batch, out_dim],与输出(matmul 结果)同形——照抄现 lowering
// (src/frontend/lowering.cpp::AppendLayer 的 bias_shape 用法),使参数声明
// 在工厂期即静态确定(ARCH-073)且 golden 逐字节判据可守(ARCH-074)。广播
// 形态 bias(仅 [out_dim]、依赖 add 未来的广播支持)待 ADR-0009 在 shape_infer
// 落地后另批启用并为本工厂减参(本章第 9 节非目标)。
FRAME_API Module Linear(std::string name, int64_t batch, int64_t in_dim, int64_t out_dim,
                        bool with_bias, DType dtype);

// 逐元素 relu 激活包装:无参数、无子模块,恰 1 输入恰 1 输出。
FRAME_API Module Relu(std::string name);

// 顺序组合:按 children 声明序逐个转发 inputs/outputs——前一子模块的输出即
// 下一子模块的输入,首个子模块消费 Sequential::build 收到的原始 inputs;
// params 按 [子0 参数…, 子1 参数…, …] 先序分段切片(ARCH-071 params 切片
// 不变式,每段长度 = 对应子 parameters().size())。Sequential 自身无直接
// 参数。
FRAME_API Module Sequential(std::string name, std::vector<Module> children);

// 均方误差损失:mse_loss(pred, target),恰 2 输入(先 pred 后 target)、
// 恰 1 输出;无参数、无子模块。是否装配损失、target 图输入从哪来等装配决策
// 一律归调用方(ARCH-074 职责边界裁定)。
FRAME_API Module MseLoss(std::string name);

// ---------------------------------------------------------------------------
// M21 批3 T6:卷积批次工厂(docs/plan/2026-07-18-batch3-m21-conv.md 第1.4节)。
// 形状口径同 Linear(v0 全静态形状世界,build 期从输入静态类型取形);Conv2d/
// Conv1d 无需显式 batch 形参(bias 是算子内可选第三输入 [Cout],通道维在算子
// 语义内广播,非图级广播,裁决点①)。kernel_hw/stride_hw/padding_hw 用
// std::array<int64_t, 2>(layers.h 现状无既有 HW 对偶类型,按计划 1.4 节口径
// 选定,不发明新公共类型——数组本身是标准库类型)。
// ---------------------------------------------------------------------------

// 二维卷积层:ParamSpec weight[out_channels, in_channels/groups, KH, KW] +
// (with_bias 时) bias[out_channels];BuildFn 构单个 conv2d 节点(bias 作为
// 该节点第三输入,内联在算子语义中广播,不产生额外 add 节点,裁决点①)。
FRAME_API Module Conv2d(std::string name, int64_t in_channels, int64_t out_channels,
                        std::array<int64_t, 2> kernel_hw, std::array<int64_t, 2> stride_hw,
                        std::array<int64_t, 2> padding_hw, int64_t groups, bool with_bias,
                        DType dtype);

// 一维卷积层:同构 Conv2d,ParamSpec weight[out_channels, in_channels/groups,
// kernel] + (with_bias 时) bias[out_channels];kernel/stride/padding 均标量
// (conv1d 算子属性口径,裁决点②:CUDA 侧经 decomposition 落 cuDNN,CPU 侧另有
// 直循环参考 kernel,本工厂不感知该差异)。
FRAME_API Module Conv1d(std::string name, int64_t in_channels, int64_t out_channels, int64_t kernel,
                        int64_t stride, int64_t padding, int64_t groups, bool with_bias,
                        DType dtype);

// 二维最大池化:无参数、无子模块,恰 1 输入恰 1 输出,BuildFn 构单个
// max_pool2d 节点。
FRAME_API Module MaxPool2d(std::string name, std::array<int64_t, 2> kernel_hw,
                           std::array<int64_t, 2> stride_hw, std::array<int64_t, 2> padding_hw);

// 二维平均池化:同 MaxPool2d,BuildFn 构单个 avg_pool2d 节点(分母恒
// KH*KW,含 padding,算子语义,本工厂不感知该差异)。
FRAME_API Module AvgPool2d(std::string name, std::array<int64_t, 2> kernel_hw,
                           std::array<int64_t, 2> stride_hw, std::array<int64_t, 2> padding_hw);

// 逐元素 sigmoid 激活包装:无参数、无子模块,恰 1 输入恰 1 输出(镜像 Relu)。
FRAME_API Module Sigmoid(std::string name);

// 展平层:无参数、无子模块,恰 1 输入恰 1 输出;BuildFn 从输入 Value 的静态
// TensorType 算出 [N, prod(其余维)](N = 输入第 0 维,其余维连乘),发单个
// reshape 节点(target_shape = 该计算结果)。
FRAME_API Module Flatten(std::string name);

// AFF(Attentional Feature Fusion,计划 1.4 节 local-only 变体——v0 无图级
// 广播,不含原文 MS-CAM 的全局池化分支,该分支留待 ADR-0009 广播落地后另批
// 补齐,docs/architecture/nn-design.md §6 交付注记):两个 1x1 Conv2d 子模块
// c1/c2(children,先序 [c1 子树参数, c2 子树参数],ARCH-071 切片不变式对组合
// 模块自动成立)。恰 2 输入 X/Y,恰 1 输出:
// M = sigmoid(c2(relu(c1(add(X, Y)))));
// out = add(mul(M, X), mul(one_minus_M, Y)),one_minus_M 经两个 constant
// splat(系数 1 与 -1)与 mul/add 组合得到(无 sub 算子,§1.4)。
FRAME_API Module AFF(std::string name, int64_t channels, DType dtype);

// 小波族类型(封闭枚举):Dwt2d/Dwt1d 工厂选用固定滤波器系数的依据。v1 覆盖
// 范围:kHaar(2d/1d 均支持)、kDb4(仅 1d;2d 侧需要可分离两趟卷积,v1 暂不
// 支持,Dwt2d 传入 kDb4 在 build() 期返回错误状态,计划 1.4 节)。
enum class WaveletKind : uint8_t {
  kHaar,
  kDb4,
};

// 二维离散小波变换(v1 kind=kHaar):无参数、无子模块(固定滤波器经 BuildFn
// 内 constant 节点物化,非 ParamSpec,不参与训练,§1.4);恰 1 输入恰 1 输出。
// 滤波器 [4*channels, 1, 2, 2],groups=channels、stride=[2,2]、padding=[0,0];
// 四子带(LL/LH/HL/HH)经分组卷积的输出通道天然交织(每组 4 个输出通道对应
// 同一输入通道),免额外 concat 节点。
FRAME_API Module Dwt2d(std::string name, int64_t channels, WaveletKind wavelet_kind);

// 一维离散小波变换(v1 kind={kHaar, kDb4}):无参数、无子模块,恰 1 输入恰 1
// 输出。滤波器 [2*channels, 1, K],groups=channels、stride=2、padding=0
// (Haar K=2,Db4 K=4);低通/高通两子带经分组卷积的输出通道交织(每组 2 个
// 输出通道对应同一输入通道,先低通后高通)。
FRAME_API Module Dwt1d(std::string name, int64_t channels, WaveletKind wavelet_kind);

// ---------------------------------------------------------------------------
// M22 批4 T5:序列批次工厂(docs/plan/2026-07-19-batch4-m22-seq.md §1.7,
// design-reviewer 两轮 APPROVE)。四工厂全部只用既有公开算子经
// ops::create_node_with_inferred_types 构图(唯一构图 helper),不新增算子。
// ---------------------------------------------------------------------------

// 层归一化:ParamSpec gamma[dim]/beta[dim](单 layer_norm 节点内沿行广播,
// §1.2/1.7,同 conv bias 算子内广播先例——参数形状免 batch,优于 Linear bias
// 同形旧例);输入限 rank-2 [rows, dim](layer_norm 算子自身 shape_infer 校验,
// YAGNI,高秩折叠留需求出现时);恰 1 输入恰 1 输出。eps 经 kDouble 属性透传,
// eps<=0 由算子 shape_infer 在 build() 期拒绝(不在本工厂重复校验)。
FRAME_API Module LayerNorm(std::string name, int64_t dim, double eps, DType dtype);

// 长短期记忆网络:输入 x[batch, num_steps, input_dim];ParamSpec
// W_ih[input_dim,4H]/W_hh[hidden_dim,4H]/bias[batch,4H](Linear bias 同形
// 先例);静态展开 num_steps 步,门序 i,f,g,o 固定(§1.7 原文):
// x_t=reshape(slice(x,axis=1,t,t+1),[B,E]);z=x_t·W_ih+h·W_hh+bias;
// i/f/g/o=slice(z,axis=1,·)+sigmoid/sigmoid/tanh/sigmoid(按该门序切列);
// c=f⊙c+i⊙g;h=o⊙tanh(c);h0/c0=constant(0) splat[B,H]。恰 1 输入恰 1 输出
// (h_T[B,H])。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
FRAME_API Module LSTM(std::string name, int64_t batch, int64_t num_steps, int64_t input_dim,
                      int64_t hidden_dim, DType dtype);

// 多头自注意力:输入 x[batch·seq_len, embed_dim](Linear 2-D 口径);children=4
// 个 Linear(先序 q/k/v/o,batch 形参=batch·seq_len、in_dim=out_dim=embed_dim、
// with_bias 透传);build() 期校验 embed_dim % num_heads==0,Dh=embed_dim/
// num_heads;matmul 维持 rank-2(§1.3,不扩批量 matmul),经 per-(b,h) 静态展开
// (§1.7)表达:行 slice[bS,(b+1)S) → 列 slice[hDh,(h+1)Dh) 得 Q/K/V_bh[S,Dh];
// scores=matmul(Q_bh,transpose(K_bh,[1,0]));scaled=scores·constant(1/√Dh)
// splat;A=softmax(scaled);O_bh=matmul(A,V_bh);头沿 axis=1 concat → 批沿
// axis=0 concat → o 投影。恰 1 输入恰 1 输出(自注意力,输出与输入同形)。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
FRAME_API Module MultiheadAttention(std::string name, int64_t batch, int64_t seq_len,
                                    int64_t embed_dim, int64_t num_heads, bool with_bias,
                                    DType dtype);

// Transformer 编码器块(Post-LN,§1.7 固定):children=[mha, ln1, ffn1, ffn2,
// ln2](先序=构建执行序,ARCH-071 params 切片按此序分段);
// y1=ln1(x+mha(x));y=ln2(y1+ffn2(relu(ffn1(y1)))),FFN=Linear→relu→Linear
// (relu 无参数,内联构图、非独立 child,同 AFF 先例);残差 add 要求两操作数同形
// (v0 无图级广播)。恰 1 输入恰 1 输出。两个 LayerNorm 子模块的 eps 固定为
// 1e-5(§1.7 原文本工厂签名未透传该形参,采用 transformer 文献惯用默认值)。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
FRAME_API Module TransformerEncoderBlock(std::string name, int64_t batch, int64_t seq_len,
                                         int64_t embed_dim, int64_t num_heads, int64_t ffn_dim,
                                         bool with_bias, DType dtype);

// ---------------------------------------------------------------------------
// M23 批5 T5:频域批次工厂(docs/plan/2026-07-21-batch5-m23-fft.md §1.5,
// design-reviewer APPROVE)。三工厂全部只用既有公开算子(含 M23 批5 T3 新增的
// rfft/irfft)经 ops::create_node_with_inferred_types 构图,不新增算子。复数
// 打包表示(决议点A,末轴 2=(re,im) 交错)与复乘组合均无 sub 算子,经
// constant(-1)+mul+add 表达(M22 AFF 先例)。
// ---------------------------------------------------------------------------

// 谱卷积(Fourier Neural Operator 的频域算子):输入 x[batch,in_channels,n] →
// rfft → [batch,in_channels,k,2](k=n/2+1)→ slice 前 modes 模态 → 逐模态
// 静态展开(§1.5,M22 MHA per-(b,h) 展开同款;matmul 维持 rank-2,不扩批量
// matmul):每模态 slice 出 re/im[batch,in_channels] 二维张量,与该模态的权重
// 列切片 W_re/W_im[in_channels,out_channels] 做 4 次 matmul + 复乘组合
// (re_out=re@W_re-im@W_im;im_out=re@W_im+im@W_re)得 [batch,out_channels] 的
// re/im → reshape/concat 回拼 [batch,out_channels,1,2] → 各模态沿模态轴
// concat 回 [batch,out_channels,modes,2] → 零补到 k(modes==k 时跳过零补
// concat,slice_gradient 零宽度跳过同款先例)→ irfft(n) → 输出
// [batch,out_channels,n]。ParamSpec 2 件:W_re/W_im 各
// [in_channels, modes*out_channels],逐模态列切片取 [in_channels,out_channels]
// (第 j 模态占列 [j*out_channels, (j+1)*out_channels))。build() 期校验
// modes <= k=n/2+1(k 由 n 静态算出),恰 1 输入恰 1 输出。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
FRAME_API Module SpectralConv1d(std::string name, int64_t batch, int64_t in_channels,
                                int64_t out_channels, int64_t n, int64_t modes, DType dtype);

// 频域可学习复滤波器(Fourier Mamba 频域支线,待 M25 合体,§1.5):输入
// x[batch,channels,n] → rfft → [batch,channels,k,2](k=n/2+1)→ 逐元素复乘
// (y_re=w_re⊙x_re-w_im⊙x_im;y_im=w_re⊙x_im+w_im⊙x_re,x_re/x_im 经 slice
// 末轴取得)→ concat 回 [batch,channels,k,2] → irfft(n) → 输出
// [batch,channels,n]。ParamSpec w_re/w_im 各 [batch,channels,k,1]
// (Linear bias 同形先例)——**注意逐样本参数语义**:w_re/w_im 形状含 batch
// 维,是「每样本独立滤波器」而非跨样本共享的滤波器,调用方组装 batch 维输入
// 与参数时须与此语义自洽(设计门建议 3,§1.5)。恰 1 输入恰 1 输出。
FRAME_API Module FourierFilter1d(std::string name, int64_t batch, int64_t channels, int64_t n,
                                 DType dtype);

// FNO 前向块(FNO1d block,§1.5):children=[SpectralConv1d, Conv1d](后者
// kernel=1/stride=1/padding=0/groups=1 的逐点旁路,M21 既有工厂;with_bias
// 固定为 true,是计划未明确处的工程取舍——常规卷积基线配置,无 BN 层可吸收
// bias,同 AFF 头注释先例);y=tanh(add(spectral(x), conv1x1(x)))。params
// 按 [spectral 子树参数…, conv1x1 子树参数…] 先序分段切片(ARCH-071 切片不
// 变式对组合模块自动成立)。恰 1 输入恰 1 输出。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
FRAME_API Module Fno1dBlock(std::string name, int64_t batch, int64_t in_channels,
                            int64_t out_channels, int64_t n, int64_t modes, DType dtype);

// ---------------------------------------------------------------------------
// M25 批6 T4:状态空间模型工厂(docs/plan/
// 2026-07-23-batch6-m25-ssm.md §1.4,design-reviewer T3 补审 APPROVE)。
// 两工厂只组合既有 Conv1d/Linear/FourierFilter1d 工厂与公开算子,
// 全部计算均在静态图内表达。
// ---------------------------------------------------------------------------

// Mamba 块:输入/输出均为 [batch,channels,steps]。children 固定为
// [conv,input,a,b,c,d,gate,out]:conv 为带 bias 的 depthwise causal
// Conv1d(C->C,kernel_size,stride=1,padding=kernel_size-1,groups=C),裁出前
// steps 步后转为 [batch*steps,channels];六路带 bias Linear(C->C)
// 投影依次施加 tanh/sigmoid/tanh/tanh/tanh/sigmoid,恢复为
// [batch,channels,steps]。input/a/b/c/d 送入 selective_scan,结果与 gate
// 相乘后由带 bias 的 out Linear(C->C)投影并恢复原形。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
FRAME_API Module Mamba(std::string name, int64_t batch, int64_t channels, int64_t steps,
                       int64_t kernel_size, DType dtype);

// FourierMamba 融合块:输入/输出均为 [batch,channels,steps],
// children 固定为 [mamba,fourier];时域 Mamba 分支与复用既有
// FourierFilter1d 的频域分支相加后取 tanh。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
FRAME_API Module FourierMamba(std::string name, int64_t batch, int64_t channels, int64_t steps,
                              int64_t kernel_size, DType dtype);

// ---------------------------------------------------------------------------
// M27 批8:脉冲神经网络工厂(docs/plan/
// 2026-07-23-batch8-m27-snn.md §1.2/1.3,design-reviewer APPROVE)。
// 两工厂仅组合公开算子与既有 Linear,静态展开固定时间步,零 eager。
// ---------------------------------------------------------------------------

// LIF 单元:输入/输出 [batch,num_steps,features],无参数。v0=0,每步
// v_pre=decay*v_prev+x_t;spike=heaviside_surrogate(v_pre-threshold;alpha);
// v_next=v_pre*(1-spike)。reset 不 detach,代理梯度完整穿过更新式。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
FRAME_API Module LIFCell(std::string name, int64_t batch, int64_t num_steps, int64_t features,
                         double decay, double threshold, double alpha, DType dtype);

// 固定时间步 SNN 分类器:输入 [batch,num_steps,input_dim],输出 logits
// [batch,num_classes]。children=[input,lif,output],流水线为两次 Linear 包围
// LIF,末尾沿时间轴求和。两个 Linear 的作用 batch 均为 batch*num_steps。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
FRAME_API Module SnnClassifier(std::string name, int64_t batch, int64_t num_steps,
                               int64_t input_dim, int64_t hidden_dim, int64_t num_classes,
                               double decay, double threshold, double alpha, bool with_bias,
                               DType dtype);

// ---------------------------------------------------------------------------
// M28 批9:固定拓扑图网络工厂(docs/plan/
// 2026-07-23-batch9-m28-gnn.md §1.2/1.3,design-reviewer APPROVE)。
// 拓扑向量按值持有,build() 只构造静态 IR,不触碰 eager 数值。
// ---------------------------------------------------------------------------

// 有向图卷积:固定 source/target 边表,按源出度与目标入度对消息归一化后
// scatter_add 到目标节点。索引须在界且 <=2^53(kDoubleArray 精确整数界)。
// children=[linear],唯一参数 weight[Fin,Fout]。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
FRAME_API Module GraphConv(std::string name, int64_t num_nodes, int64_t in_features,
                           int64_t out_features, std::vector<int64_t> source_indices,
                           std::vector<int64_t> target_indices, DType dtype);

// 超图卷积:固定二值关联表,执行 Dv^-1/2 H De^-1 H^T Dv^-1/2 后接 Linear。
// 索引须在界且 <=2^53；children=[linear],唯一参数 weight[Fin,Fout]；重复
// 关联在 build() 期拒绝。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
FRAME_API Module HypergraphConv(std::string name, int64_t num_nodes, int64_t num_hyperedges,
                                int64_t in_features, int64_t out_features,
                                std::vector<int64_t> node_indices,
                                std::vector<int64_t> hyperedge_indices, DType dtype);

}  // namespace frame::nn
