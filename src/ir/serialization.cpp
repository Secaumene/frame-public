// 图 IR 确定性文本序列化的实现单元(dump_text/parse_text)。
// 格式规范见 include/frame/ir/serialization.h 头注释(唯一权威)。

#include <array>
#include <charconv>
#include <map>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include <frame/core/macros.h>
#include <frame/ir/serialization.h>

namespace frame::ir {

namespace {

// dump_text/parse_text 共用的图输出标记行前缀 "graph_output(",在编译期从
// kGraphOutputMarker(见 graph.h)拼接派生而非另行硬编码字面量,避免 dump
// 侧与 parse 侧(含行首前缀匹配用的长度)三处各自维护同一常量而漂移。用定长
// std::array 存储、constexpr string_view 引用——不用 std::string(其静态
// 存储期动态初始化可能抛出无法捕获的异常,bugprone-throwing-static-
// initialization,项目 WarningsAsErrors 强制项)。
constexpr std::array<char, kGraphOutputMarker.size() + 1> kGraphOutputLinePrefixStorage = [] {
  std::array<char, kGraphOutputMarker.size() + 1> chars{};
  for (size_t i = 0; i < kGraphOutputMarker.size(); ++i) {
    chars[i] = kGraphOutputMarker[i];
  }
  chars[kGraphOutputMarker.size()] = '(';
  return chars;
}();
constexpr std::string_view kGraphOutputLinePrefix(kGraphOutputLinePrefixStorage.data(),
                                                  kGraphOutputLinePrefixStorage.size());

// =============================================================================
// dump_text 侧:各类型 -> 文本
// =============================================================================

// double -> 往返安全文本:std::to_chars(<charconv>)最短往返(shortest
// round-trip)表示,locale 无关,与 parse 侧 std::from_chars 对称;结果不含
// '.'/'e'/'E' 时追加 ".0"(如 to_chars(5.0) 产出 "5"),避免与 kInt64 的
// 文本表示混淆。
std::string format_double(double value) {
  // 64 字节足以容纳 double 的 to_chars 最短往返表示(含符号/小数点/指数),
  // 留有余量。
  char buffer[64];
  const auto [ptr, ec] = std::to_chars(buffer, buffer + sizeof(buffer), value);
  FRAME_CHECK(ec == std::errc());  // 缓冲区按最坏情形预留,不应失败
  std::string text(buffer, ptr);
  const bool looks_like_float =
      text.find('.') != std::string::npos || text.find('e') != std::string::npos ||
      text.find('E') != std::string::npos || text.find("inf") != std::string::npos ||
      text.find("nan") != std::string::npos;
  if (!looks_like_float) {
    text += ".0";
  }
  return text;
}

// string -> 双引号包裹并转义反斜杠/双引号/换行。
std::string format_quoted_string(std::string_view value) {
  std::string out = "\"";
  for (const char c : value) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      default:
        out += c;
    }
  }
  out += '"';
  return out;
}

// Shape 维度列表 -> "d0,d1,..."(不带空格,复用于张量类型后缀与 kShape 属性)。
std::string format_shape_dims(const Shape& shape) {
  std::string out;
  const std::vector<int64_t>& dims = shape.dims();
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i != 0) out += ',';
    out += std::to_string(dims[i]);
  }
  return out;
}

}  // namespace

// 公开 API(声明见 include/frame/ir/serialization.h):提升自本文件原匿名
// 命名空间内的同名函数(M8,REUSE-002 单份,详见头文件声明处注释)。定义须
// 落在 format_double/format_quoted_string/format_shape_dims(上方匿名命名空间
// 内,内部链接但本翻译单元内可见)之后,故不能移到文件顶部。
std::string format_attr_value(const AttrValue& value) {
  switch (attr_type_of(value)) {
    case AttrType::kInt64:
      return std::to_string(std::get<int64_t>(value));
    case AttrType::kDouble:
      return format_double(std::get<double>(value));
    case AttrType::kString:
      return format_quoted_string(std::get<std::string>(value));
    case AttrType::kBool:
      return std::get<bool>(value) ? "true" : "false";
    case AttrType::kInt64Array: {
      const std::vector<int64_t>& values = std::get<std::vector<int64_t>>(value);
      std::string out = "[";
      for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out += ", ";
        out += std::to_string(values[i]);
      }
      out += ']';
      return out;
    }
    case AttrType::kDoubleArray: {
      const std::vector<double>& values = std::get<std::vector<double>>(value);
      std::string out = "[";
      for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out += ", ";
        out += format_double(values[i]);
      }
      out += ']';
      return out;
    }
    case AttrType::kDType:
      return "dtype(" + std::string(std::get<DType>(value).name()) + ")";
    case AttrType::kShape:
      return "shape[" + format_shape_dims(std::get<Shape>(value)) + "]";
  }
  FRAME_CHECK(false);  // AttrType 是封闭枚举(ARCH-020),switch 已穷举全部取值
  return {};
}

namespace {

// layout 尾缀 token 文本(M9,决议点 A):唯一权威见
// include/frame/ir/serialization.h 头注释第3a条。
constexpr std::string_view kRowMajorLayoutSuffix = ":row_major";

std::string format_tensor_type(const TensorType& type) {
  std::string out(type.dtype.name());
  out += '[';
  out += format_shape_dims(type.shape);
  out += "]@";
  out += std::string(type.device.backend);
  out += ':';
  out += std::to_string(type.device.index);
  // layout != kUnknown 时追加 ":row_major" 尾缀;kUnknown 不追加,与 M9 前
  // 格式逐字节兼容(头注释第3a条)。
  switch (type.layout) {
    case Layout::kUnknown:
      break;
    case Layout::kRowMajor:
      out += kRowMajorLayoutSuffix;
      break;
  }
  return out;
}

// 取 value 在本次 dump 中分配到的 id;value_ids 由调用方保证已覆盖图内全部
// 已产出 Value,命中失败即违反内部不变量(非用户输入错误),走 FRAME_CHECK。
int64_t lookup_value_id(const std::unordered_map<const Value*, int64_t>& value_ids,
                        const Value* value) {
  const auto it = value_ids.find(value);
  FRAME_CHECK(it != value_ids.end());
  return it->second;
}

std::string format_node_line(const Node& node,
                             const std::unordered_map<const Value*, int64_t>& value_ids) {
  std::string line;

  const std::vector<Value>& outputs = node.outputs();
  if (!outputs.empty()) {
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (i != 0) line += ", ";
      line += '%';
      line += std::to_string(lookup_value_id(value_ids, &outputs[i]));
    }
    line += " = ";
  }

  line += node.op();
  line += '(';
  const std::vector<Value*>& inputs = node.inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (i != 0) line += ", ";
    line += '%';
    line += std::to_string(lookup_value_id(value_ids, inputs[i]));
  }
  line += ')';

  // 属性按名字典序输出:node.attrs() 本身枚举顺序不确定(见 node.h 头注释
  // 契约),此处用 std::map<std::string, ...> 显式按名排序,满足序列化层的
  // 确定性契约。
  if (!node.attrs().empty()) {
    std::map<std::string, const AttrValue*> sorted_attrs;
    for (const auto& [name, value] : node.attrs()) {
      sorted_attrs.emplace(name, &value);
    }
    line += " {";
    bool first = true;
    for (const auto& [name, value] : sorted_attrs) {
      if (!first) line += ", ";
      first = false;
      line += name;
      line += '=';
      line += format_attr_value(*value);
    }
    line += '}';
  }

  if (!outputs.empty()) {
    line += " : ";
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (i != 0) line += ", ";
      line += format_tensor_type(outputs[i].type());
    }
  }
  return line;
}

// =============================================================================
// parse_text 侧:文本 -> 各类型
// =============================================================================

Status parse_error(int64_t line_number, const std::string& message) {
  return Status::make(ErrorCode::kInvalidArgument,
                      "parse_text: line " + std::to_string(line_number) + ": " + message);
}

// 按 '\n' 切分为行;不产生末尾多余的空行(dump_text 每行均以 '\n' 结尾)。
std::vector<std::string_view> split_lines(std::string_view text) {
  std::vector<std::string_view> lines;
  size_t start = 0;
  while (start < text.size()) {
    const size_t newline_pos = text.find('\n', start);
    if (newline_pos == std::string_view::npos) {
      lines.push_back(text.substr(start));
      break;
    }
    lines.push_back(text.substr(start, newline_pos - start));
    start = newline_pos + 1;
  }
  return lines;
}

// 跳过双引号字符串内容(含反斜杠转义),在 s 中找第一个不在引号内的 target
// 字符位置;找不到返回 npos。用于在属性/输入/类型列表中定位分隔符,避免被
// 引号字符串内部的同名字符(如字符串值内的逗号)误判为分隔符。
// (仅两个参数、类型互不可隐式转换,规避 bugprone-easily-swappable-parameters;
// 需要从非 0 位置起搜索的调用方自行 substr 后按偏移量换算结果,见调用处。)
size_t find_unquoted(std::string_view s, char target) {
  bool in_quotes = false;
  for (size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (in_quotes) {
      if (c == '\\' && i + 1 < s.size()) {
        ++i;
        continue;
      }
      if (c == '"') in_quotes = false;
      continue;
    }
    if (c == '"') {
      in_quotes = true;
      continue;
    }
    if (c == target) return i;
  }
  return std::string_view::npos;
}

// 按顶层 ", " 切分列表(嵌套 []/() 与引号字符串内的 ", " 不算分隔符)。
// 用于输入列表/属性列表/类型列表/数组属性值内部元素列表。
std::vector<std::string_view> split_top_level(std::string_view s) {
  std::vector<std::string_view> parts;
  if (s.empty()) return parts;
  size_t start = 0;
  int depth = 0;
  bool in_quotes = false;
  size_t i = 0;
  while (i < s.size()) {
    const char c = s[i];
    if (in_quotes) {
      if (c == '\\' && i + 1 < s.size()) {
        i += 2;
        continue;
      }
      if (c == '"') in_quotes = false;
      ++i;
      continue;
    }
    if (c == '"') {
      in_quotes = true;
      ++i;
      continue;
    }
    if (c == '[' || c == '(') {
      ++depth;
      ++i;
      continue;
    }
    if (c == ']' || c == ')') {
      --depth;
      ++i;
      continue;
    }
    if (depth == 0 && c == ',' && i + 1 < s.size() && s[i + 1] == ' ') {
      parts.push_back(s.substr(start, i - start));
      i += 2;
      start = i;
      continue;
    }
    ++i;
  }
  parts.push_back(s.substr(start));
  return parts;
}

bool looks_like_double_token(std::string_view token) {
  return token.find('.') != std::string_view::npos || token.find('e') != std::string_view::npos ||
         token.find('E') != std::string_view::npos || token.find("inf") != std::string_view::npos ||
         token.find("nan") != std::string_view::npos;
}

Result<int64_t> parse_int64_token(std::string_view token, int64_t line_number) {
  int64_t value = 0;
  const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
  if (ec != std::errc() || ptr != token.data() + token.size()) {
    return parse_error(line_number, "invalid int64 literal '" + std::string(token) + "'");
  }
  return value;
}

Result<double> parse_double_token(std::string_view token, int64_t line_number) {
  double value = 0.0;
  const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
  if (ec != std::errc() || ptr != token.data() + token.size()) {
    return parse_error(line_number, "invalid double literal '" + std::string(token) + "'");
  }
  return value;
}

// "d0,d1,..." -> 维度列表(不带空格,shape 记法专用);空串 -> 空列表(rank 0)。
Result<std::vector<int64_t>> parse_shape_dims(std::string_view inner, int64_t line_number) {
  std::vector<int64_t> dims;
  if (inner.empty()) return dims;
  size_t start = 0;
  while (start <= inner.size()) {
    const size_t comma = inner.find(',', start);
    const std::string_view token = (comma == std::string_view::npos)
                                       ? inner.substr(start)
                                       : inner.substr(start, comma - start);
    const Result<int64_t> value = parse_int64_token(token, line_number);
    if (!value.is_ok()) return value.status();
    dims.push_back(value.value());
    if (comma == std::string_view::npos) break;
    start = comma + 1;
  }
  return dims;
}

// English dtype 名 -> DType(穷举 DTypeCode 合法取值,反查 DType::name();
// 复用既有 dtype.h 基础设施,不新增缩写表)。
Result<DType> dtype_from_name(std::string_view name, int64_t line_number) {
  for (uint8_t code = 0; code < static_cast<uint8_t>(DTypeCode::kCount); ++code) {
    const DType candidate(static_cast<DTypeCode>(code));
    if (candidate.name() == name) return candidate;
  }
  return parse_error(line_number, "unknown dtype name '" + std::string(name) + "'");
}

// 后端注册键文本 -> device.h 声明的静态存储期常量(Device::backend 是非拥有
// string_view,禁止持有指向输入 text 缓冲区的悬挂视图,细节见
// serialization.h 头注释第4条)。
Result<std::string_view> resolve_backend_name(std::string_view name, int64_t line_number) {
  if (name == kCpuBackendName) return kCpuBackendName;
  if (name == kCudaBackendName) return kCudaBackendName;
  if (name == kIntelGpuBackendName) return kIntelGpuBackendName;
  if (name == kIntelNpuBackendName) return kIntelNpuBackendName;
  if (name == kAscendBackendName) return kAscendBackendName;
  return parse_error(line_number, "unknown backend name '" + std::string(name) + "'");
}

Result<AttrValue> parse_quoted_string_value(std::string_view value, int64_t line_number) {
  if (value.size() < 2 || value.front() != '"' || value.back() != '"') {
    return parse_error(line_number,
                       "malformed string attribute value '" + std::string(value) + "'");
  }
  const std::string_view inner = value.substr(1, value.size() - 2);
  std::string out;
  for (size_t i = 0; i < inner.size(); ++i) {
    const char c = inner[i];
    if (c == '\\' && i + 1 < inner.size()) {
      const char next = inner[i + 1];
      switch (next) {
        case '\\':
          out += '\\';
          break;
        case '"':
          out += '"';
          break;
        case 'n':
          out += '\n';
          break;
        default:
          return parse_error(line_number, "unknown escape sequence in string attribute value");
      }
      ++i;
      continue;
    }
    out += c;
  }
  return AttrValue{std::move(out)};
}

Result<AttrValue> parse_dtype_attr_value(std::string_view value, int64_t line_number) {
  if (value.size() < 7 || value.substr(0, 6) != "dtype(" || value.back() != ')') {
    return parse_error(line_number, "malformed dtype attribute value '" + std::string(value) + "'");
  }
  const std::string_view name = value.substr(6, value.size() - 7);
  const Result<DType> dtype = dtype_from_name(name, line_number);
  if (!dtype.is_ok()) return dtype.status();
  return AttrValue{dtype.value()};
}

Result<AttrValue> parse_shape_attr_value(std::string_view value, int64_t line_number) {
  if (value.size() < 7 || value.substr(0, 6) != "shape[" || value.back() != ']') {
    return parse_error(line_number, "malformed shape attribute value '" + std::string(value) + "'");
  }
  const std::string_view inner = value.substr(6, value.size() - 7);
  const Result<std::vector<int64_t>> dims = parse_shape_dims(inner, line_number);
  if (!dims.is_ok()) return dims.status();
  return AttrValue{Shape(dims.value())};
}

Result<AttrValue> parse_array_attr_value(std::string_view value, int64_t line_number) {
  if (value.size() < 2 || value.back() != ']') {
    return parse_error(line_number, "malformed array attribute value '" + std::string(value) + "'");
  }
  const std::string_view inner = value.substr(1, value.size() - 2);
  const std::vector<std::string_view> tokens = split_top_level(inner);
  const bool empty_array = tokens.size() == 1 && tokens[0].empty();

  bool has_double = false;
  if (!empty_array) {
    for (const std::string_view token : tokens) {
      if (looks_like_double_token(token)) {
        has_double = true;
        break;
      }
    }
  }

  if (has_double) {
    std::vector<double> values;
    values.reserve(tokens.size());
    for (const std::string_view token : tokens) {
      const Result<double> parsed = parse_double_token(token, line_number);
      if (!parsed.is_ok()) return parsed.status();
      values.push_back(parsed.value());
    }
    return AttrValue{std::move(values)};
  }

  std::vector<int64_t> values;
  if (!empty_array) {
    values.reserve(tokens.size());
    for (const std::string_view token : tokens) {
      const Result<int64_t> parsed = parse_int64_token(token, line_number);
      if (!parsed.is_ok()) return parsed.status();
      values.push_back(parsed.value());
    }
  }
  return AttrValue{std::move(values)};
}

Result<AttrValue> parse_scalar_attr_value(std::string_view value, int64_t line_number) {
  if (looks_like_double_token(value)) {
    const Result<double> parsed = parse_double_token(value, line_number);
    if (!parsed.is_ok()) return parsed.status();
    return AttrValue{parsed.value()};
  }
  const Result<int64_t> parsed = parse_int64_token(value, line_number);
  if (!parsed.is_ok()) return parsed.status();
  return AttrValue{parsed.value()};
}

Result<AttrValue> parse_attr_value(std::string_view value, int64_t line_number) {
  if (!value.empty() && value.front() == '"') return parse_quoted_string_value(value, line_number);
  if (value.substr(0, 6) == "dtype(") return parse_dtype_attr_value(value, line_number);
  if (value.substr(0, 6) == "shape[") return parse_shape_attr_value(value, line_number);
  if (value == "true") return AttrValue{true};
  if (value == "false") return AttrValue{false};
  if (!value.empty() && value.front() == '[') return parse_array_attr_value(value, line_number);
  return parse_scalar_attr_value(value, line_number);
}

Result<TensorType> parse_tensor_type(std::string_view token, int64_t line_number) {
  // layout 尾缀剥离(M9,头注释第3a条):先剥离末尾的 ":row_major"(若存在),
  // 剩余部分按既有 "<dtype>[<shape>]@<backend>:<index>" 逻辑解析(剥离后
  // token.rfind(':') 重新定位到 <backend>:<index> 之间的冒号,不再与
  // layout 尾缀的冒号混淆,按段数区分、无歧义)。
  Layout layout = Layout::kUnknown;
  if (token.size() > kRowMajorLayoutSuffix.size() &&
      token.substr(token.size() - kRowMajorLayoutSuffix.size()) == kRowMajorLayoutSuffix) {
    layout = Layout::kRowMajor;
    token = token.substr(0, token.size() - kRowMajorLayoutSuffix.size());
  }

  const size_t bracket_pos = token.find('[');
  const size_t at_pos = token.find('@');
  const size_t colon_pos = token.rfind(':');
  if (bracket_pos == std::string_view::npos || at_pos == std::string_view::npos ||
      colon_pos == std::string_view::npos || at_pos < bracket_pos || colon_pos < at_pos) {
    return parse_error(line_number, "malformed tensor type '" + std::string(token) + "'");
  }
  const size_t close_bracket_pos = token.find(']', bracket_pos);
  if (close_bracket_pos == std::string_view::npos || close_bracket_pos > at_pos) {
    return parse_error(line_number, "malformed tensor type '" + std::string(token) + "'");
  }

  const Result<DType> dtype = dtype_from_name(token.substr(0, bracket_pos), line_number);
  if (!dtype.is_ok()) return dtype.status();
  const Result<std::vector<int64_t>> dims = parse_shape_dims(
      token.substr(bracket_pos + 1, close_bracket_pos - bracket_pos - 1), line_number);
  if (!dims.is_ok()) return dims.status();
  const Result<std::string_view> backend =
      resolve_backend_name(token.substr(at_pos + 1, colon_pos - at_pos - 1), line_number);
  if (!backend.is_ok()) return backend.status();
  const Result<int64_t> index = parse_int64_token(token.substr(colon_pos + 1), line_number);
  if (!index.is_ok()) return index.status();

  TensorType type;
  type.dtype = dtype.value();
  type.shape = Shape(dims.value());
  type.layout = layout;
  type.device = Device{backend.value(), static_cast<int32_t>(index.value())};
  return type;
}

Result<int64_t> parse_value_ref_id(std::string_view token, int64_t line_number) {
  if (token.empty() || token.front() != '%') {
    return parse_error(line_number,
                       "value reference '" + std::string(token) + "' must start with '%'");
  }
  return parse_int64_token(token.substr(1), line_number);
}

Result<Value*> resolve_value_ref(std::string_view token, int64_t line_number,
                                 const std::unordered_map<int64_t, Value*>& id_to_value) {
  const Result<int64_t> id = parse_value_ref_id(token, line_number);
  if (!id.is_ok()) return id.status();
  const auto it = id_to_value.find(id.value());
  if (it == id_to_value.end()) {
    return parse_error(line_number, "reference to undefined value %" + std::to_string(id.value()));
  }
  return it->second;
}

Status parse_graph_output_line(std::string_view line, int64_t line_number,
                               const std::unordered_map<int64_t, Value*>& id_to_value,
                               Graph& graph) {
  if (line.back() != ')') {
    return parse_error(line_number, "malformed graph_output line (missing ')')");
  }
  const std::string_view inner =
      line.substr(kGraphOutputLinePrefix.size(), line.size() - kGraphOutputLinePrefix.size() - 1);
  const Result<Value*> value = resolve_value_ref(inner, line_number, id_to_value);
  if (!value.is_ok()) return value.status();
  return graph.mark_output(value.value());
}

// 登记 id -> Value 映射;id 已存在视为"重复 %id 定义"错误(而非静默保留旧
// 映射)。dump_text 产出的规范形态中 id 全局单调递增、每个 id 只应作为恰好
// 一行的左侧出现一次,重复定义只可能来自非规范输入。
Status register_value_id(int64_t id, Value* value, int64_t line_number,
                         std::unordered_map<int64_t, Value*>& id_to_value) {
  const auto [it, inserted] = id_to_value.emplace(id, value);
  (void)it;
  if (!inserted) {
    return parse_error(line_number, "duplicate definition of value %" + std::to_string(id));
  }
  return Status::ok();
}

// 解析单个节点定义行,经公开构图 API(create_node/add_graph_input/set_attr)
// 落地节点,并把该行产出的 id 登记进 id_to_value,供后续行引用。
Status parse_node_line(std::string_view line, int64_t line_number,
                       std::unordered_map<int64_t, Value*>& id_to_value, Graph& graph) {
  const size_t first_paren = line.find('(');
  if (first_paren == std::string_view::npos) {
    return parse_error(line_number, "missing '(' in node definition");
  }
  const size_t eq_pos = line.find(" = ");
  std::string_view lhs;
  std::string_view rest = line;
  bool has_lhs = false;
  if (eq_pos != std::string_view::npos && eq_pos < first_paren) {
    lhs = line.substr(0, eq_pos);
    rest = line.substr(eq_pos + 3);
    has_lhs = true;
  }

  const size_t op_paren = rest.find('(');
  if (op_paren == std::string_view::npos) {
    return parse_error(line_number, "missing '(' after op name");
  }
  const std::string_view op = rest.substr(0, op_paren);
  const size_t close_paren = rest.find(')', op_paren);
  if (close_paren == std::string_view::npos) {
    return parse_error(line_number, "missing ')' after inputs");
  }
  const std::string_view inputs_text = rest.substr(op_paren + 1, close_paren - op_paren - 1);
  std::string_view remainder = rest.substr(close_paren + 1);

  std::vector<Value*> inputs;
  if (!inputs_text.empty()) {
    for (const std::string_view token : split_top_level(inputs_text)) {
      const Result<Value*> input = resolve_value_ref(token, line_number, id_to_value);
      if (!input.is_ok()) return input.status();
      inputs.push_back(input.value());
    }
  }

  std::vector<std::pair<std::string, AttrValue>> attrs;
  if (remainder.size() >= 2 && remainder[0] == ' ' && remainder[1] == '{') {
    // find_unquoted 只在 "{" 之后的子串内搜索,结果需加回 2 的偏移量换算回
    // remainder 坐标系。
    const size_t close_brace_rel = find_unquoted(remainder.substr(2), '}');
    if (close_brace_rel == std::string_view::npos) {
      return parse_error(line_number, "missing '}' after attributes");
    }
    const size_t close_brace = close_brace_rel + 2;
    const std::string_view attrs_text = remainder.substr(2, close_brace - 2);
    if (!attrs_text.empty()) {
      for (const std::string_view pair_text : split_top_level(attrs_text)) {
        const size_t eq = find_unquoted(pair_text, '=');
        if (eq == std::string_view::npos) {
          return parse_error(line_number, "malformed attribute (missing '=')");
        }
        const std::string_view name = pair_text.substr(0, eq);
        const Result<AttrValue> value = parse_attr_value(pair_text.substr(eq + 1), line_number);
        if (!value.is_ok()) return value.status();
        attrs.emplace_back(std::string(name), value.value());
      }
    }
    remainder = remainder.substr(close_brace + 1);
  }

  std::vector<TensorType> output_types;
  if (!remainder.empty()) {
    if (remainder.size() < 3 || remainder.substr(0, 3) != " : ") {
      return parse_error(line_number, "malformed trailing output type section");
    }
    for (const std::string_view token : split_top_level(remainder.substr(3))) {
      const Result<TensorType> type = parse_tensor_type(token, line_number);
      if (!type.is_ok()) return type.status();
      output_types.push_back(type.value());
    }
  }

  std::vector<int64_t> lhs_ids;
  if (has_lhs) {
    for (const std::string_view token : split_top_level(lhs)) {
      const Result<int64_t> id = parse_value_ref_id(token, line_number);
      if (!id.is_ok()) return id.status();
      lhs_ids.push_back(id.value());
    }
  }
  if (lhs_ids.size() != output_types.size()) {
    return parse_error(line_number, "output id count does not match output type count");
  }

  if (op == kGraphInputOp) {
    if (output_types.size() != 1 || !inputs.empty()) {
      return parse_error(line_number,
                         "graph_input node must have zero inputs and exactly one output");
    }
    const Result<Value*> value = graph.add_graph_input(output_types[0]);
    if (!value.is_ok()) return value.status();
    FRAME_RETURN_IF_ERROR(register_value_id(lhs_ids[0], value.value(), line_number, id_to_value));
    return Status::ok();
  }

  Result<Node*> node =
      graph.create_node(std::string(op), std::move(inputs), std::move(output_types));
  if (!node.is_ok()) return node.status();
  for (auto& [name, value] : attrs) {
    node.value()->set_attr(name, std::move(value));
  }
  for (size_t i = 0; i < lhs_ids.size(); ++i) {
    // Node::outputs() 非 const 重载已移除(决议点 5-②),按下标取可变指针改经
    // Node::output(int32_t)——parse_node_line 刚经 create_node 建好该节点,i
    // 必然落在其输出个数区间内,不会取到 nullptr。
    FRAME_RETURN_IF_ERROR(register_value_id(
        lhs_ids[i], node.value()->output(static_cast<int32_t>(i)), line_number, id_to_value));
  }
  return Status::ok();
}

}  // namespace

std::string dump_text(const Graph& graph) {
  // id 按拓扑序分配:遍历 topological_order(),节点内按输出序号(0..N-1)
  // 递增分配,构成全局单调递增序列。
  std::unordered_map<const Value*, int64_t> value_ids;
  int64_t next_id = 0;
  for (const Node* node : graph.topological_order()) {
    for (const Value& output : node->outputs()) {
      value_ids.emplace(&output, next_id);
      ++next_id;
    }
  }

  std::string text;
  for (const Node* node : graph.topological_order()) {
    text += format_node_line(*node, value_ids);
    text += '\n';
  }
  for (const Value* output : graph.outputs()) {
    text += kGraphOutputLinePrefix;
    text += '%';
    text += std::to_string(lookup_value_id(value_ids, output));
    text += ")\n";
  }
  return text;
}

Result<Graph> parse_text(std::string_view text) {
  Graph graph;
  std::unordered_map<int64_t, Value*> id_to_value;

  const std::vector<std::string_view> lines = split_lines(text);
  for (size_t line_index = 0; line_index < lines.size(); ++line_index) {
    const int64_t line_number = static_cast<int64_t>(line_index) + 1;
    const std::string_view line = lines[line_index];
    if (line.empty()) {
      return parse_error(line_number, "empty line is not allowed");
    }
    if (line.substr(0, kGraphOutputLinePrefix.size()) == kGraphOutputLinePrefix) {
      const Status status = parse_graph_output_line(line, line_number, id_to_value, graph);
      if (!status.is_ok()) return status;
      continue;
    }
    const Status status = parse_node_line(line, line_number, id_to_value, graph);
    if (!status.is_ok()) return status;
  }
  return graph;
}

}  // namespace frame::ir
