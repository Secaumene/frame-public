# ADR-0013:ONNX 权重交换采用自研最小 wire-format 编解码

- 状态:已接受
- 日期:2026-07-13
- 关联铁律:#5 两级复用
- 关联规则:REUSE-010、ARCH-003、BUILD-011

## 背景

训练闭环(v1.1)落地后,项目所有者提出(2026-07-13):权重需可持久化并
与生态互操作——ONNX 权重导入/导出,以「训练→保存→加载→推理」全链路
验收。ONNX 官方 C++ 路径传递依赖 protobuf 运行时;本项目所需仅为
ModelProto 中 graph.initializer 的 TensorProto 读写,须裁决依赖取舍。

## 决策

- **范围 = 权重交换子集**:读/写 `graph.initializer` 的 TensorProto
  (name/dims/data_type/raw_data;dtype 覆盖 FLOAT/FLOAT16/BFLOAT16)+
  合法 ModelProto 骨架(ir_version/opset_import/graph.name)。**不做算子
  图导入导出**(图级互操作另案)。导入遇子集外 data_type(如 INT64)
  拒绝并返回含该枚举值的英文错误,不静默跳过。
- **实现 = 自研最小 protobuf wire-format 编解码**:编码侧仅产生 varint
  与 length-delimited;解码侧为正确跳过未知字段另识别 wire 1(64-bit)/
  wire 5(32-bit,proto2 非 packed repeated float 即此类)定长推进,
  已废弃 groups(wire 3/4)显式拒绝——小型单文件编解码器量级。拒绝
  引入 onnx + protobuf 库(为 initializer 子集拉入整库,成本远超需求);
  自研审批即本 ADR(铁律 #5)。互操作经测试侧 Python `onnx` 包校验
  (pytest 可选依赖,importorskip;Apache-2.0;仅测试侧不进 CMake——
  REUSE-010 判定限 find_package/FetchContent,循 numpy 先例)。
- 【待查证】判定①③承重的 ONNX 标准事实,实现时逐一核实,禁止凭记忆
  编码 —— 来源:onnx/onnx.proto 与 onnx.checker 文档:(a) 各字段的
  protobuf 字段号/wire type 与 Model→Graph→initializer 嵌套;(b)
  BFLOAT16 枚举值、raw_data 编码(宽度/字节序)与最小 ir_version/opset;
  (c) checker 对「仅 initializer、无 node」最小 ModelProto 的合法性
  (历史版本可能要求 graph.output,以最小合法骨架为准)。
- **落点 = 新增一级子目录** `include/frame/interop/` + `src/interop/`
  (onnx_weights.{h,cpp}:save/load,host 内存,仅依赖 core);互操作
  不属既有五层职责,**ADR-010 情形 4 的新增目录授权即本 ADR**。
- 验收(判定方法):①字节级往返(save→load 逐张量 dtype/shape/位模式
  相等);②train-save-import-infer 链路(MLP 训练后导出,重建同构图、
  加载权重、编译推理,与训练末态推理一致,BUILD-011 容差);③pytest
  侧 `onnx.checker` 校验导出文件(包缺失按既有 skip 口径);④
  `grep -rin "protobuf" cmake/` 为空且无新增 find_package/FetchContent。

## 备选方案

- 官方 onnx C++ 库:字段全、随 opset 演进;但传递依赖 protobuf 运行时,
  为权重子集付出整库成本——否决(图级互操作立项时重评)。
- 自定义二进制格式:实现最简;但零生态互操作,违背需求——否决。

## 后果

- 正面:零新构建依赖;导出文件可被标准 ONNX 工具读取(checker 校验)。
- 负面与代价:wire-format 子集随所用字段演进维护;子集外字段读取跳过
  (dtype 除外,显式拒)、写入不产生。
- 跟进(与新层落地同批):overview.md 分层节登记 interop 职责边界;
  src/CMakeLists.txt TOP-001 目标数与 TOP-002 依赖 DAG 纳入 frame_interop
  (位于 core 之上、不依赖其余各层);build-order.md §2 分层表增行。
  图级 ONNX 导入/导出立项时重评本 ADR。
