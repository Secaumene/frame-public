// ONNX 权重交换编解码实现(ADR-0013)。自研最小 protobuf wire-format 编解码
// (proto2,varint / length-delimited 两种 wire type 写入;读取侧另识别 wire
// 1/5 定长跳过、显式拒绝已废弃 groups wire 3/4),不引入 onnx/protobuf 库
// (ADR-0013 决策段)。字段号/wire type/枚举值来源:onnx/onnx 仓 main 分支
// onnx.proto(维护者 2026-07-13 核实,见实现 spec 附注)。

#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/interop/onnx_weights.h>

namespace frame::interop {

namespace {

// raw_data 编码假设:host 为小端(浮点位模式按 IEEE 754 固定宽度小端写出,
// x86_64 下与本仓 float/float16_t/bfloat16_t 的内存位型直接 memcpy 一致,
// ADR-0013 已核实事实)。大端主机需要显式逐字节转换,本文件未实现,
// static_assert 在编译期拒绝该平台组合而非运行时给出错误结果。
static_assert(std::endian::native == std::endian::little,
              "onnx_weights raw_data encoding assumes a little-endian host platform "
              "(see docs/decisions/0013-onnx-weight-exchange-minimal-codec.md); big-endian "
              "hosts require explicit byte-swapping that this file does not implement");

// ---- protobuf wire type 常量 ----
constexpr int kWireVarint = 0;
constexpr int kWireFixed64 = 1;
constexpr int kWireLengthDelimited = 2;
constexpr int kWireGroupStart = 3;
constexpr int kWireGroupEnd = 4;
constexpr int kWireFixed32 = 5;

// ---- 本文件覆盖的 ONNX proto2 字段号(onnx.proto,已核实) ----
constexpr int kFieldModelIrVersion = 1;    // ModelProto.ir_version(int64, varint)
constexpr int kFieldModelGraph = 7;        // ModelProto.graph(GraphProto)
constexpr int kFieldModelOpsetImport = 8;  // ModelProto.opset_import(repeated OperatorSetIdProto)

constexpr int kFieldOpsetDomain = 1;   // OperatorSetIdProto.domain(string)
constexpr int kFieldOpsetVersion = 2;  // OperatorSetIdProto.version(int64, varint)

constexpr int kFieldGraphName = 2;         // GraphProto.name(string)
constexpr int kFieldGraphInitializer = 5;  // GraphProto.initializer(repeated TensorProto)

constexpr int kFieldTensorDims = 1;      // TensorProto.dims(repeated int64,非 packed)
constexpr int kFieldTensorDataType = 2;  // TensorProto.data_type(int32, varint)
constexpr int kFieldTensorName = 8;      // TensorProto.name(string)
constexpr int kFieldTensorRawData = 9;   // TensorProto.raw_data(bytes)

// ModelProto 骨架取值(ADR-0013 已核实/待 pytest onnx.checker 实证收敛)。
constexpr int64_t kOnnxIrVersion = 8;
constexpr int64_t kOnnxOpsetVersion = 13;
constexpr std::string_view kOnnxDefaultDomain = "";  // 主 ai.onnx 域
constexpr std::string_view kOnnxGraphName = "frame_weights";

// TensorProto.DataType 枚举子集(v0 三 dtype 白名单对应值,ADR-0013 已核实)。
// 底型 uint8_t 足以承载三值(wire 上仍按 ONNX 规定的 int32 varint 编解码,
// 编解码两侧均显式与具名枚举值比较、从不将任意宽整数直接 static_cast 进本
// 枚举类型,故缩小底型不引入截断安全隐患,performance-enum-size 判定成立)。
enum class OnnxDataType : uint8_t {
  kFloat = 1,
  kFloat16 = 10,
  kBFloat16 = 16,
};

// ---------------------------------------------------------------------------
// 编码侧:嵌套消息先序列化子缓冲(ByteBuffer)再由外层写 length-delimited。
// ---------------------------------------------------------------------------
using ByteBuffer = std::string;

void append_varint(ByteBuffer& out, uint64_t value) {
  while (value >= 0x80) {
    out.push_back(static_cast<char>((value & 0x7f) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<char>(value));
}

void append_tag(ByteBuffer& out, int field_number, int wire_type) {
  append_varint(out, (static_cast<uint64_t>(field_number) << 3) | static_cast<uint64_t>(wire_type));
}

// value 按 uint64_t 的位模式原样写出:负 int64 调用方须先 static_cast<uint64_t>
// 传入(两者补码位模式相同),自然产生 protobuf 规范的 10 字节 varint
// (dims 恒非负,此处仅防御性支持通用 int64 语义,不做符号假设)。field_number
// 全部来自本文件顶部具名 kField* 常量,调用点无裸整数字面量可被误传,误用
// 风险低。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void append_varint_field(ByteBuffer& out, int field_number, uint64_t value) {
  append_tag(out, field_number, kWireVarint);
  append_varint(out, value);
}

void append_string_field(ByteBuffer& out, int field_number, std::string_view value) {
  append_tag(out, field_number, kWireLengthDelimited);
  append_varint(out, value.size());
  out.append(value.data(), value.size());
}

void append_bytes_field(ByteBuffer& out, int field_number, const void* data, size_t size) {
  append_tag(out, field_number, kWireLengthDelimited);
  append_varint(out, size);
  out.append(static_cast<const char*>(data), size);
}

void append_message_field(ByteBuffer& out, int field_number, const ByteBuffer& message) {
  append_tag(out, field_number, kWireLengthDelimited);
  append_varint(out, message.size());
  out.append(message);
}

// frame dtype -> ONNX DataType(v0 三值白名单;子集外 dtype 返回错误,消息含
// dtype 名)。每个 case 分支直接 return 一个具名枚举值,不存在
// "先默认/值初始化再回填" 的中间态(避免 OnnxDataType 出现无对应枚举项的
// 0 值,呼应 bugprone-invalid-enum-default-initialization)。
Result<OnnxDataType> dtype_to_onnx(DType dtype) {
  switch (dtype.code()) {
    case DTypeCode::kFloat32:
      return OnnxDataType::kFloat;
    case DTypeCode::kFloat16:
      return OnnxDataType::kFloat16;
    case DTypeCode::kBFloat16:
      return OnnxDataType::kBFloat16;
    default:
      return Status::make(ErrorCode::kInvalidArgument,
                          "save_onnx_weights: dtype '" + std::string(dtype.name()) +
                              "' outside the v0 whitelist (float32/float16/bfloat16)");
  }
}

// 单张量 -> TensorProto 子缓冲。校验:device 须为 cpu、dtype 属三值白名单
// (interop 与 ops 分属不同依赖层,不复用 ops::is_constant_dtype_supported——
// 该判定函数位于 interop 依赖面之外,本地三值判定属分层约束下的独立小逻辑,
// 不构成 REUSE 重复)。
Result<ByteBuffer> encode_tensor_proto(const NamedTensor& item) {
  const Tensor& tensor = item.tensor;
  if (tensor.device().backend != kCpuBackendName) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "save_onnx_weights: tensor '" + item.name +
                            "' must reside on the cpu device (found backend '" +
                            std::string(tensor.device().backend) + "')");
  }

  const Result<OnnxDataType> onnx_dtype_result = dtype_to_onnx(tensor.dtype());
  if (!onnx_dtype_result.is_ok()) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "save_onnx_weights: tensor '" + item.name + "' has dtype '" +
                            std::string(tensor.dtype().name()) +
                            "' outside the v0 whitelist (float32/float16/bfloat16)");
  }
  const OnnxDataType onnx_dtype = onnx_dtype_result.value();

  ByteBuffer message;
  for (const int64_t dim : tensor.shape().dims()) {
    append_varint_field(message, kFieldTensorDims, static_cast<uint64_t>(dim));
  }
  append_varint_field(message, kFieldTensorDataType,
                      static_cast<uint64_t>(static_cast<int32_t>(onnx_dtype)));
  append_string_field(message, kFieldTensorName, item.name);
  const size_t nbytes = static_cast<size_t>(tensor.numel()) * tensor.dtype().itemsize();
  append_bytes_field(message, kFieldTensorRawData, tensor.raw_data(), nbytes);
  return message;
}

// ---------------------------------------------------------------------------
// 解码侧:游标式读取。未知字段按 wire type 跳过(0/1/5/2 定长或
// length-delimited 推进);已废弃 groups(wire 3/4)显式拒绝
// kUnimplemented;越界/截断一律 kInvalidArgument(消息含偏移)。
// ---------------------------------------------------------------------------
class ByteCursor {
 public:
  explicit ByteCursor(std::string_view data) : data_(data) {}

  bool at_end() const { return pos_ >= data_.size(); }

  Result<uint64_t> read_varint() {
    uint64_t result = 0;
    for (int shift = 0; shift < 70; shift += 7) {
      if (pos_ >= data_.size()) {
        return Status::make(
            ErrorCode::kInvalidArgument,
            "load_onnx_weights: truncated varint at offset " + std::to_string(pos_));
      }
      const uint8_t byte = static_cast<uint8_t>(data_[pos_++]);
      result |= static_cast<uint64_t>(byte & 0x7f) << shift;
      if ((byte & 0x80) == 0) return result;
    }
    return Status::make(
        ErrorCode::kInvalidArgument,
        "load_onnx_weights: varint exceeds 64 bits at offset " + std::to_string(pos_));
  }

  Result<std::string_view> read_bytes(size_t count) {
    if (count > data_.size() - pos_) {
      return Status::make(
          ErrorCode::kInvalidArgument,
          "load_onnx_weights: field extends past end of buffer at offset " + std::to_string(pos_));
    }
    const std::string_view result = data_.substr(pos_, count);
    pos_ += count;
    return result;
  }

  Result<std::string_view> read_length_delimited() {
    const Result<uint64_t> length = read_varint();
    if (!length.is_ok()) return length.status();
    return read_bytes(static_cast<size_t>(length.value()));
  }

  Result<std::pair<int, int>> read_tag() {
    const Result<uint64_t> tag = read_varint();
    if (!tag.is_ok()) return tag.status();
    const int field_number = static_cast<int>(tag.value() >> 3);
    const int wire_type = static_cast<int>(tag.value() & 0x7);
    return std::make_pair(field_number, wire_type);
  }

  // 跳过一个未知字段(按 wire type);wire 3/4(groups)显式拒绝。
  Status skip_field(int wire_type) {
    switch (wire_type) {
      case kWireVarint:
        return read_varint().status();
      case kWireFixed64:
        return read_bytes(8).status();
      case kWireFixed32:
        return read_bytes(4).status();
      case kWireLengthDelimited:
        return read_length_delimited().status();
      case kWireGroupStart:
      case kWireGroupEnd:
        return Status::make(ErrorCode::kUnimplemented,
                            "load_onnx_weights: deprecated group wire type " +
                                std::to_string(wire_type) + " is not supported at offset " +
                                std::to_string(pos_));
      default:
        return Status::make(ErrorCode::kInvalidArgument, "load_onnx_weights: unknown wire type " +
                                                             std::to_string(wire_type) +
                                                             " at offset " + std::to_string(pos_));
    }
  }

 private:
  std::string_view data_;
  size_t pos_ = 0;
};

DType onnx_to_dtype(OnnxDataType value) {
  switch (value) {
    case OnnxDataType::kFloat:
      return DType(DTypeCode::kFloat32);
    case OnnxDataType::kFloat16:
      return DType(DTypeCode::kFloat16);
    case OnnxDataType::kBFloat16:
      return DType(DTypeCode::kBFloat16);
  }
  FRAME_CHECK(false);  // 调用方已校验 value 落在三值白名单内,不可达
  return DType(DTypeCode::kFloat32);
}

// TensorProto 子缓冲 -> 单张量(allocator 分配 host cpu 内存承载 raw_data)。
Result<NamedTensor> parse_tensor_proto(std::string_view message, hal::Allocator& allocator) {
  ByteCursor cursor(message);
  std::vector<int64_t> dims;
  std::optional<OnnxDataType> onnx_dtype;
  std::string name;
  std::string_view raw_data;
  bool has_raw_data = false;

  while (!cursor.at_end()) {
    const Result<std::pair<int, int>> tag = cursor.read_tag();
    if (!tag.is_ok()) return tag.status();
    const auto [field_number, wire_type] = tag.value();

    if (field_number == kFieldTensorDims && wire_type == kWireVarint) {
      const Result<uint64_t> value = cursor.read_varint();
      if (!value.is_ok()) return value.status();
      dims.push_back(static_cast<int64_t>(value.value()));
      continue;
    }
    if (field_number == kFieldTensorDataType && wire_type == kWireVarint) {
      const Result<uint64_t> value = cursor.read_varint();
      if (!value.is_ok()) return value.status();
      // 不把任意宽的 raw_value 直接 static_cast 进 OnnxDataType(其底型现为
      // uint8_t,窄范围截断可能让越界值假性匹配某个合法枚举项);改为逐一与
      // 具名枚举值的宽整数表示比较,OnnxDataType 变量全程只被赋具名枚举值。
      const int64_t raw_value = static_cast<int64_t>(value.value());
      if (raw_value == static_cast<int64_t>(OnnxDataType::kFloat)) {
        onnx_dtype = OnnxDataType::kFloat;
      } else if (raw_value == static_cast<int64_t>(OnnxDataType::kFloat16)) {
        onnx_dtype = OnnxDataType::kFloat16;
      } else if (raw_value == static_cast<int64_t>(OnnxDataType::kBFloat16)) {
        onnx_dtype = OnnxDataType::kBFloat16;
      } else {
        return Status::make(ErrorCode::kInvalidArgument,
                            "load_onnx_weights: TensorProto.data_type " +
                                std::to_string(raw_value) +
                                " is outside the v0 whitelist (FLOAT=1/FLOAT16=10/BFLOAT16=16)");
      }
      continue;
    }
    if (field_number == kFieldTensorName && wire_type == kWireLengthDelimited) {
      const Result<std::string_view> value = cursor.read_length_delimited();
      if (!value.is_ok()) return value.status();
      name = std::string(value.value());
      continue;
    }
    if (field_number == kFieldTensorRawData && wire_type == kWireLengthDelimited) {
      const Result<std::string_view> value = cursor.read_length_delimited();
      if (!value.is_ok()) return value.status();
      raw_data = value.value();
      has_raw_data = true;
      continue;
    }
    const Status skip_status = cursor.skip_field(wire_type);
    if (!skip_status.is_ok()) return skip_status;
  }

  if (!onnx_dtype.has_value()) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "load_onnx_weights: TensorProto '" + name + "' is missing data_type");
  }
  const DType dtype = onnx_to_dtype(*onnx_dtype);

  const Shape shape(dims);
  Result<Tensor> tensor_result = Tensor::empty(shape, dtype, cpu_device(), allocator);
  if (!tensor_result.is_ok()) return tensor_result.status();
  Tensor tensor = std::move(tensor_result.value());

  const size_t expected_bytes = static_cast<size_t>(tensor.numel()) * dtype.itemsize();
  if (!has_raw_data) {
    if (expected_bytes != 0) {
      return Status::make(ErrorCode::kInvalidArgument,
                          "load_onnx_weights: TensorProto '" + name + "' is missing raw_data");
    }
  } else if (raw_data.size() != expected_bytes) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "load_onnx_weights: TensorProto '" + name + "' raw_data size " +
                            std::to_string(raw_data.size()) + " does not match expected " +
                            std::to_string(expected_bytes) + " byte(s)");
  }
  if (expected_bytes != 0) {
    std::memcpy(tensor.raw_data(), raw_data.data(), expected_bytes);
  }

  return NamedTensor{std::move(name), std::move(tensor)};
}

// GraphProto 子缓冲 -> initializer 列表(按文件中出现顺序)。
Result<std::vector<NamedTensor>> parse_graph_proto(std::string_view message,
                                                   hal::Allocator& allocator) {
  ByteCursor cursor(message);
  std::vector<NamedTensor> result;
  while (!cursor.at_end()) {
    const Result<std::pair<int, int>> tag = cursor.read_tag();
    if (!tag.is_ok()) return tag.status();
    const auto [field_number, wire_type] = tag.value();

    if (field_number == kFieldGraphInitializer && wire_type == kWireLengthDelimited) {
      const Result<std::string_view> sub_message = cursor.read_length_delimited();
      if (!sub_message.is_ok()) return sub_message.status();
      Result<NamedTensor> tensor = parse_tensor_proto(sub_message.value(), allocator);
      if (!tensor.is_ok()) return tensor.status();
      result.push_back(std::move(tensor.value()));
      continue;
    }
    const Status skip_status = cursor.skip_field(wire_type);
    if (!skip_status.is_ok()) return skip_status;
  }
  return result;
}

// ModelProto(整份文件内容)-> initializer 列表。
Result<std::vector<NamedTensor>> parse_model_proto(std::string_view message,
                                                   hal::Allocator& allocator) {
  ByteCursor cursor(message);
  std::optional<std::vector<NamedTensor>> graph_result;
  while (!cursor.at_end()) {
    const Result<std::pair<int, int>> tag = cursor.read_tag();
    if (!tag.is_ok()) return tag.status();
    const auto [field_number, wire_type] = tag.value();

    if (field_number == kFieldModelGraph && wire_type == kWireLengthDelimited) {
      const Result<std::string_view> sub_message = cursor.read_length_delimited();
      if (!sub_message.is_ok()) return sub_message.status();
      Result<std::vector<NamedTensor>> parsed = parse_graph_proto(sub_message.value(), allocator);
      if (!parsed.is_ok()) return parsed.status();
      graph_result = std::move(parsed.value());
      continue;
    }
    const Status skip_status = cursor.skip_field(wire_type);
    if (!skip_status.is_ok()) return skip_status;
  }
  if (!graph_result.has_value()) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "load_onnx_weights: ModelProto is missing the graph field");
  }
  return std::move(*graph_result);
}

}  // namespace

Status save_onnx_weights(const std::string& path, std::span<const NamedTensor> weights) {
  ByteBuffer graph_message;
  append_string_field(graph_message, kFieldGraphName, kOnnxGraphName);
  for (const NamedTensor& item : weights) {
    Result<ByteBuffer> tensor_message = encode_tensor_proto(item);
    if (!tensor_message.is_ok()) return tensor_message.status();
    append_message_field(graph_message, kFieldGraphInitializer, tensor_message.value());
  }

  ByteBuffer opset_message;
  append_string_field(opset_message, kFieldOpsetDomain, kOnnxDefaultDomain);
  append_varint_field(opset_message, kFieldOpsetVersion, static_cast<uint64_t>(kOnnxOpsetVersion));

  ByteBuffer model_message;
  append_varint_field(model_message, kFieldModelIrVersion, static_cast<uint64_t>(kOnnxIrVersion));
  append_message_field(model_message, kFieldModelOpsetImport, opset_message);
  append_message_field(model_message, kFieldModelGraph, graph_message);

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "save_onnx_weights: cannot open '" + path + "' for writing");
  }
  out.write(model_message.data(), static_cast<std::streamsize>(model_message.size()));
  if (!out.good()) {
    return Status::make(ErrorCode::kInternal, "save_onnx_weights: failed to write '" + path + "'");
  }
  return Status::ok();
}

Result<std::vector<NamedTensor>> load_onnx_weights(const std::string& path,
                                                   hal::Allocator& allocator) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    return Status::make(ErrorCode::kNotFound, "load_onnx_weights: cannot open '" + path + "'");
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return parse_model_proto(buffer.str(), allocator);
}

}  // namespace frame::interop
