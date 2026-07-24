// bind_nn:frame::nn 的 Module 与首批模块工厂(Linear/Relu/Sequential/MseLoss)
// + add_parameter_inputs 便捷面的 pybind11 绑定(M20 批2 Task5,
// docs/architecture/nn-design.md §6)。薄壳纪律(PY-001):函数体只做参数转换
// (ParamSpec -> (name, shape, dtype) 元组、DTypeCode <-> DType 包装)+ 调用
// C++ 公开 API(Module::parameters()/build()、nn::Linear/Relu/Sequential/
// MseLoss、nn::add_parameter_inputs)+ 错误转换(经 translate_status,PY-030)。
//
// Module.build()/add_parameter_inputs() 返回的 Value* 句柄生命周期均须绑定
// 所属 Graph(同 bind_graph.cpp/bind_ops.cpp 先例,防止 Graph 析构后句柄悬垂)。
// add_parameter_inputs 是自由函数、graph 为首个 Python 侧实参,pybind11 的
// return_value_policy::reference_internal(对容器返回值按元素展开、逐元素绑定
// 到"parent"——固定取自底层调用的第 1 个 Python 实参,pybind11.h dispatcher
// 硬编码 `args_in_arr[0]`,与 process_attribute<keep_alive<>> 的函数级
// postcall 机制不同,后者仅对返回值整体生效、无法穿透 std::vector→list 的逐
// 元素转换)可直接沿用(=既有 kToGraph 惯例)。Module.build() 绑定为实例方法后
// 该"parent"固定取 self(Module),而不是 graph 形参,故不能照搬同一写法;
// 改为显式收 graph 的原始 py::object 手柄,函数体内逐元素调用公开 API
// `py::cast(value, return_value_policy::reference_internal, parent)`(与
// type_caster_base 内部对 reference_internal 的处理同一路径,parent 显式传
// graph 手柄,不依赖隐式推导),手工拼装返回列表,达到与
// reference_internal/keep_alive<0,1> 等价的"Graph 随任一存活 Value 续命"效果。

#include <array>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <frame/core/dtype.h>
#include <frame/core/status.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/nn/layers.h>
#include <frame/nn/module.h>

#include "translate_status.h"

namespace frame::python_bindings {

namespace py = pybind11;

namespace {

// 单个 ParamSpec -> (name, shape, dtype) 三元组:不含 InitSpec(数值物化非本
// 绑定职责,include/frame/nn/module.h 头注释)。
std::tuple<std::string, std::vector<int64_t>, DTypeCode> to_param_info(const nn::ParamSpec& spec) {
  return {spec.name, spec.type.shape.dims(), spec.type.dtype.code()};
}

// Module.parameters():先序遍历扁平参数清单的只读元信息视图。
std::vector<std::tuple<std::string, std::vector<int64_t>, DTypeCode>> module_parameters(
    const nn::Module& module) {
  const std::vector<nn::ParamSpec> params = module.parameters();
  std::vector<std::tuple<std::string, std::vector<int64_t>, DTypeCode>> infos;
  infos.reserve(params.size());
  for (const nn::ParamSpec& spec : params) {
    infos.push_back(to_param_info(spec));
  }
  return infos;
}

// Module.build(graph, inputs, params):校验 params 尺寸后转发构图,错误转换经
// translate_result;返回列表逐元素手工绑定到 graph 的生命周期(见本文件头
// 注释——self 占据 pybind11 隐式 parent 位,故不能靠函数级
// return_value_policy 自动完成,须显式收 graph 原始 py::object 手柄)。
py::list module_build(const nn::Module& module, const py::object& graph_obj,
                      std::vector<ir::Value*> inputs, std::vector<ir::Value*> params) {
  ir::Graph& graph = graph_obj.cast<ir::Graph&>();
  const std::vector<ir::Value*> outputs = translate_result(module.build(graph, inputs, params));
  py::list result;
  for (ir::Value* value : outputs) {
    result.append(py::cast(value, py::return_value_policy::reference_internal, graph_obj));
  }
  return result;
}

// add_parameter_inputs(graph, module) 便捷面:取 module.parameters() 转发给
// nn::add_parameter_inputs(graph, span<const ParamSpec>),省去调用方手工铺开
// 参数清单这一步。
std::vector<ir::Value*> py_add_parameter_inputs(ir::Graph& graph, const nn::Module& module) {
  return translate_result(nn::add_parameter_inputs(graph, module.parameters()));
}

// Linear 工厂薄封装:Python 侧 dtype 走既有 DTypeCode 枚举(与
// bind_graph.cpp::add_graph_input、bind_ops.cpp::constant 同惯例),此处转换为
// nn::Linear 所需的 DType 包装类型。
nn::Module py_linear(std::string name, int64_t batch, int64_t in_dim, int64_t out_dim,
                     bool with_bias, DTypeCode dtype) {
  return nn::Linear(std::move(name), batch, in_dim, out_dim, with_bias, DType(dtype));
}

// Conv2d/Conv1d/AFF 工厂薄封装(M21 批3 T6,docs/plan/2026-07-18-batch3-m21-
// conv.md 第1.4节):dtype 转换同 py_linear 惯例;kernel_hw/stride_hw/
// padding_hw 走 pybind11/stl.h 对 std::array<int64_t, 2> 的内建转换(Python
// 侧传入长度不为 2 的 list/tuple 时 pybind11 自动抛 TypeError,不另写校验)。
nn::Module py_conv2d(std::string name, int64_t in_channels, int64_t out_channels,
                     std::array<int64_t, 2> kernel_hw, std::array<int64_t, 2> stride_hw,
                     std::array<int64_t, 2> padding_hw, int64_t groups, bool with_bias,
                     DTypeCode dtype) {
  return nn::Conv2d(std::move(name), in_channels, out_channels, kernel_hw, stride_hw, padding_hw,
                    groups, with_bias, DType(dtype));
}

nn::Module py_conv1d(std::string name, int64_t in_channels, int64_t out_channels, int64_t kernel,
                     int64_t stride, int64_t padding, int64_t groups, bool with_bias,
                     DTypeCode dtype) {
  return nn::Conv1d(std::move(name), in_channels, out_channels, kernel, stride, padding, groups,
                    with_bias, DType(dtype));
}

nn::Module py_aff(std::string name, int64_t channels, DTypeCode dtype) {
  return nn::AFF(std::move(name), channels, DType(dtype));
}

// wavelet_kind 字符串 -> nn::WaveletKind 解析:bind_nn 当前无枚举绑定先例
// (Module.build 收发的 Value*/Graph 等均非枚举),按计划 1.4 节口径选用字符串
// 参数 + C++ 侧解析,保持薄壳(不新增 py::enum_ 绑定面)。
Result<nn::WaveletKind> parse_wavelet_kind(const std::string& wavelet_kind) {
  if (wavelet_kind == "haar") return nn::WaveletKind::kHaar;
  if (wavelet_kind == "db4") return nn::WaveletKind::kDb4;
  return Status::make(ErrorCode::kInvalidArgument, "unrecognized wavelet_kind '" + wavelet_kind +
                                                       "' (expected 'haar' or 'db4')");
}

nn::Module py_dwt2d(std::string name, int64_t channels, const std::string& wavelet_kind) {
  return nn::Dwt2d(std::move(name), channels, translate_result(parse_wavelet_kind(wavelet_kind)));
}

nn::Module py_dwt1d(std::string name, int64_t channels, const std::string& wavelet_kind) {
  return nn::Dwt1d(std::move(name), channels, translate_result(parse_wavelet_kind(wavelet_kind)));
}

// M22 批4 T5:序列批次工厂薄封装(docs/plan/2026-07-19-batch4-m22-seq.md
// §1.7)。dtype 转换同 py_linear 惯例。
nn::Module py_layer_norm(std::string name, int64_t dim, double eps, DTypeCode dtype) {
  return nn::LayerNorm(std::move(name), dim, eps, DType(dtype));
}

nn::Module py_lstm(std::string name, int64_t batch, int64_t num_steps, int64_t input_dim,
                   int64_t hidden_dim, DTypeCode dtype) {
  return nn::LSTM(std::move(name), batch, num_steps, input_dim, hidden_dim, DType(dtype));
}

nn::Module py_multihead_attention(std::string name, int64_t batch, int64_t seq_len,
                                  int64_t embed_dim, int64_t num_heads, bool with_bias,
                                  DTypeCode dtype) {
  return nn::MultiheadAttention(std::move(name), batch, seq_len, embed_dim, num_heads, with_bias,
                                DType(dtype));
}

nn::Module py_transformer_encoder_block(std::string name, int64_t batch, int64_t seq_len,
                                        int64_t embed_dim, int64_t num_heads, int64_t ffn_dim,
                                        bool with_bias, DTypeCode dtype) {
  return nn::TransformerEncoderBlock(std::move(name), batch, seq_len, embed_dim, num_heads, ffn_dim,
                                     with_bias, DType(dtype));
}

// M23 批5 T5:频域批次工厂薄封装(docs/plan/2026-07-21-batch5-m23-fft.md
// §1.5)。dtype 转换同 py_linear 惯例。
nn::Module py_spectral_conv1d(std::string name, int64_t batch, int64_t in_channels,
                              int64_t out_channels, int64_t n, int64_t modes, DTypeCode dtype) {
  return nn::SpectralConv1d(std::move(name), batch, in_channels, out_channels, n, modes,
                            DType(dtype));
}

nn::Module py_fourier_filter1d(std::string name, int64_t batch, int64_t channels, int64_t n,
                               DTypeCode dtype) {
  return nn::FourierFilter1d(std::move(name), batch, channels, n, DType(dtype));
}

nn::Module py_fno1d_block(std::string name, int64_t batch, int64_t in_channels,
                          int64_t out_channels, int64_t n, int64_t modes, DTypeCode dtype) {
  return nn::Fno1dBlock(std::move(name), batch, in_channels, out_channels, n, modes, DType(dtype));
}

// M25 批6 T4:状态空间模型工厂薄封装(docs/plan/
// 2026-07-23-batch6-m25-ssm.md §1.4)。dtype 转换同 py_linear 先例。
nn::Module py_mamba(std::string name, int64_t batch, int64_t channels, int64_t steps,
                    int64_t kernel_size, DTypeCode dtype) {
  return nn::Mamba(std::move(name), batch, channels, steps, kernel_size, DType(dtype));
}

nn::Module py_fourier_mamba(std::string name, int64_t batch, int64_t channels, int64_t steps,
                            int64_t kernel_size, DTypeCode dtype) {
  return nn::FourierMamba(std::move(name), batch, channels, steps, kernel_size, DType(dtype));
}

// M27 批8:固定时间步脉冲网络工厂薄封装,dtype 转换同 py_linear 先例。
nn::Module py_lif_cell(std::string name, int64_t batch, int64_t num_steps, int64_t features,
                       double decay, double threshold, double alpha, DTypeCode dtype) {
  return nn::LIFCell(std::move(name), batch, num_steps, features, decay, threshold, alpha,
                     DType(dtype));
}

nn::Module py_snn_classifier(std::string name, int64_t batch, int64_t num_steps, int64_t input_dim,
                             int64_t hidden_dim, int64_t num_classes, double decay,
                             double threshold, double alpha, bool with_bias, DTypeCode dtype) {
  return nn::SnnClassifier(std::move(name), batch, num_steps, input_dim, hidden_dim, num_classes,
                           decay, threshold, alpha, with_bias, DType(dtype));
}

// M28 批9:固定拓扑图网络工厂薄封装,dtype 转换同 py_linear 先例。
nn::Module py_graph_conv(std::string name, int64_t num_nodes, int64_t in_features,
                         int64_t out_features, std::vector<int64_t> source_indices,
                         std::vector<int64_t> target_indices, DTypeCode dtype) {
  return nn::GraphConv(std::move(name), num_nodes, in_features, out_features,
                       std::move(source_indices), std::move(target_indices), DType(dtype));
}

nn::Module py_hypergraph_conv(std::string name, int64_t num_nodes, int64_t num_hyperedges,
                              int64_t in_features, int64_t out_features,
                              std::vector<int64_t> node_indices,
                              std::vector<int64_t> hyperedge_indices, DTypeCode dtype) {
  return nn::HypergraphConv(std::move(name), num_nodes, num_hyperedges, in_features, out_features,
                            std::move(node_indices), std::move(hyperedge_indices), DType(dtype));
}

}  // namespace

void bind_nn(py::module_& m) {
  py::class_<nn::Module>(m, "Module",
                         "编译期构图组合子(不透明句柄):持 name 与先序遍历得到的参数元信息;"
                         "数值构图经 build() 完成,自身不含任何数值(nn-design.md ARCH-071/072)。仅经"
                         "Linear/Relu/Sequential/MseLoss 等工厂取得。")
      .def_readonly("name", &nn::Module::name, "模块名字。")
      .def("parameters", &module_parameters,
           "先序遍历扁平参数清单,元素为 (name, shape, dtype) 三元组"
           "(不含初始化声明——数值物化非本绑定职责)。")
      .def("build", &module_build, py::arg("graph"), py::arg("inputs"), py::arg("params"),
           "按 parameters() 校验 params 尺寸后构图,返回输出 Value 句柄列表"
           "(生命周期绑定 graph)。");

  m.def("Linear", &py_linear, py::arg("name"), py::arg("batch"), py::arg("in_dim"),
        py::arg("out_dim"), py::arg("with_bias"), py::arg("dtype"),
        "全连接层:matmul(x, weight[in_dim,out_dim])[+ add(., bias[batch,out_dim])]"
        "(nn-design.md §2)。");
  m.def("Relu", &nn::Relu, py::arg("name"), "逐元素 ReLU 激活包装(无参数,恰 1 输入 1 输出)。");
  m.def("Sequential", &nn::Sequential, py::arg("name"), py::arg("children"),
        "顺序组合:按 children 声明序逐个转发 inputs/outputs,params 按子模块"
        "parameters().size() 先序分段切片。");
  m.def("MseLoss", &nn::MseLoss, py::arg("name"),
        "均方误差损失(无参数,恰 2 输入即 pred/target、恰 1 标量输出)。");

  m.def("Conv2d", &py_conv2d, py::arg("name"), py::arg("in_channels"), py::arg("out_channels"),
        py::arg("kernel_hw"), py::arg("stride_hw"), py::arg("padding_hw"), py::arg("groups"),
        py::arg("with_bias"), py::arg("dtype"),
        "二维卷积层:ParamSpec weight[out_channels,in_channels/groups,KH,KW] + "
        "(with_bias 时) bias[out_channels];单个 conv2d 节点(bias 为算子内可选第三输入,"
        "计划 2026-07-18-batch3-m21-conv 第1.4节裁决点①)。");
  m.def("Conv1d", &py_conv1d, py::arg("name"), py::arg("in_channels"), py::arg("out_channels"),
        py::arg("kernel"), py::arg("stride"), py::arg("padding"), py::arg("groups"),
        py::arg("with_bias"), py::arg("dtype"),
        "一维卷积层,同构 Conv2d(kernel/stride/padding 均标量)。");
  m.def("MaxPool2d", &nn::MaxPool2d, py::arg("name"), py::arg("kernel_hw"), py::arg("stride_hw"),
        py::arg("padding_hw"), "二维最大池化(无参数,恰 1 输入 1 输出)。");
  m.def("AvgPool2d", &nn::AvgPool2d, py::arg("name"), py::arg("kernel_hw"), py::arg("stride_hw"),
        py::arg("padding_hw"), "二维平均池化(无参数,恰 1 输入 1 输出;分母恒 KH*KW,含 padding)。");
  m.def("Sigmoid", &nn::Sigmoid, py::arg("name"),
        "逐元素 sigmoid 激活包装(无参数,恰 1 输入 1 输出,镜像 Relu)。");
  m.def("Flatten", &nn::Flatten, py::arg("name"),
        "展平层(无参数):从输入静态 shape 算出 [N, prod(其余维)] 并发单个 reshape 节点。");
  m.def("AFF", &py_aff, py::arg("name"), py::arg("channels"), py::arg("dtype"),
        "AFF 注意力特征融合(local-only 变体,无全局池化分支,计划 1.4 节偏差声明):"
        "两个 1x1 Conv2d 子模块 c1/c2,恰 2 输入 X/Y、恰 1 输出。");
  m.def("Dwt2d", &py_dwt2d, py::arg("name"), py::arg("channels"), py::arg("wavelet_kind"),
        "二维离散小波变换(固定滤波器经 constant 节点物化,不参与训练);"
        "wavelet_kind 目前仅接受 'haar'。");
  m.def("Dwt1d", &py_dwt1d, py::arg("name"), py::arg("channels"), py::arg("wavelet_kind"),
        "一维离散小波变换(固定滤波器经 constant 节点物化,不参与训练);"
        "wavelet_kind 接受 'haar' 或 'db4'。");

  // M22 批4 T5:序列批次工厂(docs/plan/2026-07-19-batch4-m22-seq.md §1.7)。
  m.def("LayerNorm", &py_layer_norm, py::arg("name"), py::arg("dim"), py::arg("eps"),
        py::arg("dtype"),
        "层归一化:ParamSpec gamma[dim]/beta[dim](单 layer_norm 节点内沿行广播);"
        "输入限 rank-2 [rows,dim],恰 1 输入 1 输出。");
  m.def("LSTM", &py_lstm, py::arg("name"), py::arg("batch"), py::arg("num_steps"),
        py::arg("input_dim"), py::arg("hidden_dim"), py::arg("dtype"),
        "长短期记忆网络:输入 x[batch,num_steps,input_dim],静态展开 num_steps 步"
        "(门序 i,f,g,o 固定),输出末步隐状态 h_T[batch,hidden_dim]。");
  m.def("MultiheadAttention", &py_multihead_attention, py::arg("name"), py::arg("batch"),
        py::arg("seq_len"), py::arg("embed_dim"), py::arg("num_heads"), py::arg("with_bias"),
        py::arg("dtype"),
        "多头自注意力:输入 x[batch*seq_len,embed_dim](Linear 2-D 口径),"
        "children=4 个 Linear(q/k/v/o);经 per-(b,h) 静态展开表达"
        "(matmul 维持 rank-2,不扩批量 matmul);恰 1 输入 1 输出,输出与输入同形。");
  m.def("TransformerEncoderBlock", &py_transformer_encoder_block, py::arg("name"), py::arg("batch"),
        py::arg("seq_len"), py::arg("embed_dim"), py::arg("num_heads"), py::arg("ffn_dim"),
        py::arg("with_bias"), py::arg("dtype"),
        "Transformer 编码器块(Post-LN):children=[mha,ln1,ffn1,ffn2,ln2];"
        "y1=ln1(x+mha(x));y=ln2(y1+ffn2(relu(ffn1(y1))));恰 1 输入 1 输出。");

  // M23 批5 T5:频域批次工厂(docs/plan/2026-07-21-batch5-m23-fft.md §1.5)。
  m.def("SpectralConv1d", &py_spectral_conv1d, py::arg("name"), py::arg("batch"),
        py::arg("in_channels"), py::arg("out_channels"), py::arg("n"), py::arg("modes"),
        py::arg("dtype"),
        "谱卷积(FNO 频域算子):rfft->slice(modes)->逐模态复乘展开->零补->irfft;"
        "ParamSpec W_re/W_im 各 [in_channels,modes*out_channels];恰 1 输入 1 输出。");
  m.def("FourierFilter1d", &py_fourier_filter1d, py::arg("name"), py::arg("batch"),
        py::arg("channels"), py::arg("n"), py::arg("dtype"),
        "频域可学习复滤波器(Fourier Mamba 频域支线):rfft->逐元素复乘->irfft;"
        "ParamSpec w_re/w_im 各 [batch,channels,k=n/2+1,1](注意逐样本参数语义,"
        "非跨样本共享滤波器);恰 1 输入 1 输出。");
  m.def("Fno1dBlock", &py_fno1d_block, py::arg("name"), py::arg("batch"), py::arg("in_channels"),
        py::arg("out_channels"), py::arg("n"), py::arg("modes"), py::arg("dtype"),
        "FNO 前向块:children=[SpectralConv1d,Conv1d(kernel=1 逐点旁路)];"
        "y=tanh(add(spectral(x),conv1x1(x)));恰 1 输入 1 输出。");

  // M25 批6 T4:状态空间模型工厂(docs/plan/
  // 2026-07-23-batch6-m25-ssm.md §1.4)。
  m.def("Mamba", &py_mamba, py::arg("name"), py::arg("batch"), py::arg("channels"),
        py::arg("steps"), py::arg("kernel_size"), py::arg("dtype"),
        "Mamba 块:depthwise causal Conv1d 后分别产生 input/a/b/c/d/gate 投影;"
        "selective_scan 结果经 gate 融合与 out Linear 投影后恢复 [B,C,L]。");
  m.def("FourierMamba", &py_fourier_mamba, py::arg("name"), py::arg("batch"), py::arg("channels"),
        py::arg("steps"), py::arg("kernel_size"), py::arg("dtype"),
        "Fourier Mamba 融合块:children=[Mamba,FourierFilter1d];两分支相加后取 tanh。");

  // M27 批8:固定时间步脉冲神经网络工厂。
  m.def("LIFCell", &py_lif_cell, py::arg("name"), py::arg("batch"), py::arg("num_steps"),
        py::arg("features"), py::arg("decay"), py::arg("threshold"), py::arg("alpha"),
        py::arg("dtype"),
        "LIF 单元:输入/输出 [batch,num_steps,features],静态展开并用代理阶跃重置膜电位。");
  m.def("SnnClassifier", &py_snn_classifier, py::arg("name"), py::arg("batch"),
        py::arg("num_steps"), py::arg("input_dim"), py::arg("hidden_dim"), py::arg("num_classes"),
        py::arg("decay"), py::arg("threshold"), py::arg("alpha"), py::arg("with_bias"),
        py::arg("dtype"),
        "固定时间步 SNN 分类器:children=[input,lif,output],末尾沿时间轴求和得到 logits。");

  // M28 批9:固定拓扑图网络工厂。
  m.def("GraphConv", &py_graph_conv, py::arg("name"), py::arg("num_nodes"), py::arg("in_features"),
        py::arg("out_features"), py::arg("source_indices"), py::arg("target_indices"),
        py::arg("dtype"),
        "有向图卷积:固定边表归一化消息并 scatter_add 到目标节点;索引须在界且 <=2^53。");
  m.def("HypergraphConv", &py_hypergraph_conv, py::arg("name"), py::arg("num_nodes"),
        py::arg("num_hyperedges"), py::arg("in_features"), py::arg("out_features"),
        py::arg("node_indices"), py::arg("hyperedge_indices"), py::arg("dtype"),
        "超图卷积:执行 Dv^-1/2 H De^-1 H^T Dv^-1/2 后接 Linear;索引须在界且 <=2^53。");

  m.def("add_parameter_inputs", &py_add_parameter_inputs, py::arg("graph"), py::arg("module"),
        py::return_value_policy::reference_internal,
        "便捷面:取 module.parameters() 批量物化为图输入,返回 Value 句柄列表"
        "(生命周期绑定 graph)。");
}

}  // namespace frame::python_bindings
