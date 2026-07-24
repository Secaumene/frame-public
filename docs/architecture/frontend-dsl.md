# 前端 DSL(JSON 模型描述)规范

> **强制等级**:规范(MUST)
> **相关铁律**:#1 编译优先 / #2 C++ 核心 / #5 两级复用
> **面向读者**:实现代码的工程师
> **最后更新**:2026-07-13(随 ADR-0017 初版)

本文档是 JSON 模型描述格式(`schema_version` 0)的**唯一权威**;范围裁决见
ADR-0017,依赖裁决见 ADR-0018。层职责见 `overview.md` §2.10;工具入口
`tools/frame_dslc`(`--check` 校验 / `--run` 进程内训练 / `--emit` 生成 C++
源码工程)。JSON 解析仅存在于工具层;frontend 库只消费纯 C++ `ModelSpec`。

## 1. 顶层结构(v0)

| 字段 | 类型 | 必填 | 语义 |
|---|---|---|---|
| `schema_version` | int | 是 | 当前恒 `0` |
| `model` | object | 是 | `name`(ASCII 标识符风格)、`dtype`(恒 `"float32"`)、`batch`(int > 0) |
| `inputs` | array | 是 | v0 恰一个数据输入:`{name, shape}`,`shape[0] == batch` |
| `layers` | array | 是 | 逐层 `{name, kind, input, weight_shape, bias_shape?, activation?}` |
| `loss` | object | 是 | `{kind: "mse", prediction: <末层名>, target_shape}` |
| `optimizer` | object | 是 | `{kind: "sgd", learning_rate > 0}` |
| `training` | object | 是 | `{steps > 0, seed(uint32), log_every >= 0}` |
| `data` | object | 是 | 数据与初始化,见第 2 节 |

`layers[].kind` v0 仅 `"linear"`:lower 为 `matmul`(+ 可选 `add` bias)
(+ 可选 `relu`);`activation ∈ {"relu", "none"}`(缺省 `"none"`);
`weight_shape = [in, out]`;`bias_shape` 可省(无 bias),给出时必须为
**全形状** `[batch, out]`(v0 无广播,共享 bias 的自然形状 `[out]` 留待
ADR-0009 广播落地后升 `schema_version` 引入)。

## 2. data 段

- `data.<输入名>` 与 `data.target`:`{kind: "inline", values: [...]}`
  (元素数 == 对应形状 numel)或 `{kind: "uniform_seeded", range: [lo, hi]}`。
- `data.params`:`{kind: "uniform_seeded", weight_range: [lo, hi],
  bias_range: [lo, hi]}`(全部参数的初始化;v0 不支持参数 inline)。
- 随机源:`std::mt19937(training.seed)`,抽取顺序 = 数据输入 → target →
  逐层参数(weight 先于 bias),与生成代码/进程内执行一致(同 seed 同轨迹)。

## 3. 校验规则(frontend::validate 逐条执行)

- 【FE-001】【MUST】`schema_version` 非 0 一律拒绝,错误消息含收到的版本号。
- 【FE-002】【MUST】名字引用闭包:`layers[].input` 必须是某 input 名或前序
  layer 名;`loss.prediction` 必须是某 layer 名;全部名字唯一且非空。
- 【FE-003】【MUST】形状链一致:layer 输入末维 == `weight_shape[0]`;
  `bias_shape == [batch, weight_shape[1]]`;`loss.target_shape` == 末层输出
  形状;所有维度 > 0。
- 【FE-004】【MUST】枚举白名单:`kind`/`activation`/`loss.kind`/
  `optimizer.kind`/`model.dtype` 取值超出本文档枚举一律拒绝(前向兼容:
  不认识 = 报错,禁止静默忽略)。
- 【FE-005】【MUST】data 完整性:每个输入与 target 必须有 data 项;
  `inline` 的 `values` 元素数与形状 numel 相等;`uniform` 范围 lo < hi;
  必须有 `data.params` 项。
- 判定方法:`tests/cpp/frontend/test_model_spec.cpp` 对以上每条至少一个
  否定用例;`frame_dslc --check` 对违例 spec 退出码非 0。

## 4. lowering 契约

- 图输入序:`[数据输入..., 逐层 weight,bias..., target]`(与
  `tests/cpp/compiler/mlp_forward_graph_helper.h` 的 `[x,w1,b1,w2,target]`
  口径一致);单一图输出 = loss。
- 全部节点经 `ops::create_node_with_inferred_types` 创建(shape 推断复用
  OpRegistry,REUSE-002);`LoweredModel` 携带 `param_names` /
  `param_types` / `wrt_input_indices` 供训练与代码生成共用。
- 底座注记(M20,ADR-0020):网络结构构图段自 M20 起经 `frame::nn` 模块
  (逐层 `nn::Linear`/`nn::Relu` + `nn::add_parameter_inputs` 批量参数
  前置 + `nn::MseLoss`)实现,FE-002 全部合法拓扑(跳连/数据输入复用/
  非末层 prediction)语义不变;dump 发射序为「参数图输入前置」形态,
  验收口径见 docs/architecture/nn-design.md ARCH-074 判定方法。

## 5. --emit 产物形态

`<out>/main.cpp`(自包含:spec 烘焙常量 → 构图 → `build_backward_graph`
→ `build_sgd_update_graph` → `runtime::compile` ×2 → 训练循环(参数轮换)
→ 推理 → 报告;风格镜像 `examples/02_graph_compile/main.cpp`,注释中文)
+ `<out>/CMakeLists.txt`(`cmake_minimum_required(3.25)`、
`find_package(frame REQUIRED)`、链 `frame::frame`;前置:先用
`scripts/install.sh --prefix <dir>` 安装,再以 `CMAKE_PREFIX_PATH=<dir>`
配置)。

## 6. 完整示例(tiny_mlp,权威样例)

权威样例文件:`tools/frame_dslc/testdata/tiny_mlp.json`(8×4 输入 →
linear(4→8, bias, relu) → linear(8→1) → mse,sgd lr=0.05,300 步,
seed=20260713——与 `test_training_loop.cpp` 收敛用例同款拓扑与种子)。
本文档不复制其内容,以该文件为准(单一事实来源)。
