#pragma once
// 图 IR 确定性文本序列化:dump_text 序列化,parse_text 反序列化(仅接受
// dump_text 产出的规范形态)。基础格式见 docs/architecture/ir-design.md 第3章
// (单行一节点:`%<id> = <op>(%<in0>, ...) {<attr>=<value>, ...} :
// <dtype>[<shape>]@<backend>:<index>`, 结尾每个图输出一行 `graph_output(%<id>)`)。
//
// 本头注释是该文档未尽格式细节的唯一权威(不改 ir-design.md 第3章正文):
//
// 1. 多输出/零输出节点(文档示例只覆盖单输出算子)。等号左侧 id 列表与冒号
//    右侧类型列表均按输出序号以 ", " 分隔;N=1 时与文档格式完全一致。
//    零输出节点省略 " = " 前缀与 " : " 后缀,只剩 `<op>(%<in0>, ...) {...}`。
// 2. 属性文本 `{<attr>=<value>, ...}`:零属性时整个 `{}` 段省略(不输出空
//    大括号)。属性按名字典序(`std::string` 字典序)输出,值按 AttrType
//    (ARCH-020)分派:
//      - kInt64:十进制整数,如 `-3`。
//      - kDouble:`std::to_chars`(`<charconv>`)产出的最短往返(shortest
//        round-trip)十进制文本,locale 无关,与 parse 侧 `std::from_chars`
//        对称;若结果不含 '.'/'e'/'E'(即看起来像整数,如 `to_chars(5.0)`
//        产出 `"5"`),强制追加 `.0`,以便与 kInt64 的文本无歧义区分。
//      - kBool:`true` / `false`。
//      - kString:双引号包裹,反斜杠/双引号/换行分别转义为 `\\`、`\"`、`\n`,
//        其余字符原样输出。
//      - kInt64Array / kDoubleArray:`[e0, e1, ...]`(逗号+空格分隔,元素格式
//        同标量 kInt64/kDouble);空数组为 `[]`。
//      - kDType:`dtype(<name>)`,`<name>` 取自 `DType::name()`(英文全称,如
//        `float32`,复用既有 dtype.h 基础设施,不新增缩写表)。
//      - kShape:`shape[<d0>,<d1>,...]`(维度间逗号**不带空格**,与下条张量
//        类型后缀的 shape 记法一致;空 shape 为 `shape[]`)。
// 3. 张量类型后缀 `<dtype>[<shape>]@<backend>:<index>`:`<dtype>` 同样取自
//    `DType::name()`;`<shape>` 维度间逗号不带空格(如 `[32,784]`)。
//
//    3a. layout 尾缀(M9,决议点 A,本条为该 token 位置与 parse 规则的唯一
//    权威——`docs/architecture/ir-design.md` 第3章仅引用本条,不重复定义):
//    当 `TensorType::layout != Layout::kUnknown` 时,在上述后缀末尾再追加一个
//    冒号段 `:row_major`(v0 唯一具体 layout 是 `Layout::kRowMajor`,故该 token
//    固定为字面量 `row_major`),得到
//    `<dtype>[<shape>]@<backend>:<index>:row_major`;`layout == Layout::kUnknown`
//    时不追加任何内容,输出与本条描述的基础格式逐字节一致(向后兼容:M9 之前
//    产出的全部文本与 golden 文件不受影响)。parse 侧:`<backend>:<index>` 之后
//    若还存在第二个 `:` 分隔的尾段且其值恰为 `row_major`,解析为
//    `Layout::kRowMajor`,并将该尾段从用于定位 `<backend>:<index>` 的原有解析
//    逻辑中剥离(按段数区分,不与 `@backend:index` 的单个冒号混淆,无歧义);
//    否则(不存在该尾段)解析为 `Layout::kUnknown`。除 `row_major` 外的其他
//    尾段文本视为格式错误(v0 尚无第二个具体 layout)。
// 4. `<backend>` 仅接受 `include/frame/core/device.h` 声明的内置后端注册键
//    常量集合(`kCpuBackendName`/`kCudaBackendName`/`kIntelGpuBackendName`/
//    `kIntelNpuBackendName`/`kAscendBackendName`)——`Device::backend` 是非
//    拥有型 `string_view`,parse_text 反序列化时必须映射回这些静态存储期的
//    常量,不能持有指向输入 `text` 缓冲区的悬挂视图;因此当前 parse_text
//    不支持任意自定义后端注册键文本往返(dump_text 侧不受影响,可原样输出
//    任意 `Device::backend` 取值,仅 parse_text 侧有此限制)。
// 5. id 分配:按 `topological_order()` 遍历节点、节点内按输出序号(0..N-1)
//    递增分配,构成全局单调递增序列;引用侧(`%<id>`)与产出侧共用同一序列。
//
// parse_text 仅经公开构图 API(`Graph::create_node`/`add_graph_input`/
// `mark_output`/`Node::set_attr`)重建图,不直接触碰 Graph/Node 的私有状态,
// 因而天然复用这些 API 自身的合法性校验;适用场景为 golden 测试读入与调试,
// 不是通用文本 IR 解析器。

#include <string>
#include <string_view>

#include <frame/core/status.h>
#include <frame/ir/graph.h>

namespace frame::ir {

// 单个属性值的确定性文本化(按 AttrType 分派,格式细节见上方头注释第 2 条:
// kInt64/kDouble/kString/kBool/kInt64Array/kDoubleArray/kDType/kShape 各自
// 的文本形态)。M8 起提升为公开 API(原为 src/ir/serialization.cpp 内的匿名
// 命名空间函数):dump_text 与 common_subexpression_elimination pass 的等价键
// 序列化共用同一份实现(REUSE-002 单份,禁止第二份属性文本化复制)。
std::string format_attr_value(const AttrValue& value);

// 确定性文本序列化:同一图任何两次调用输出逐字节相同(id 按拓扑序分配、
// 属性按名字典序输出,细节见上方头注释)。
std::string dump_text(const Graph& graph);

// 最小反序列化:仅接受 dump_text 产出的规范形态(golden 测试/调试用途,非
// 通用文本 IR 解析器)。失败返回英文错误消息,含 1-based 行号。
Result<Graph> parse_text(std::string_view text);

}  // namespace frame::ir
