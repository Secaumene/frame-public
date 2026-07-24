// constant 算子共用工具的实现单元(见 include/frame/ops/constant_utils.h)。

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/ops/constant_utils.h>

namespace frame::ops {

// v0 常量可编码 dtype 白名单(公开单份,M18 收敛;见头文件精度论证)。M22
// (批4,决议点A)扩至 int32/int64(整数索引最小接触面,gather 族 indices 的
// 形式零梯度经 constant 整数 splat 表达,docs/plan/2026-07-19-batch4-m22-seq.md
// §1.1);float64 不引入(AttrType 不扩,ARCH-020 不动)。
bool is_constant_dtype_supported(DType dtype) {
  const DTypeCode code = dtype.code();
  return code == DTypeCode::kFloat32 || code == DTypeCode::kFloat16 ||
         code == DTypeCode::kBFloat16 || code == DTypeCode::kInt32 || code == DTypeCode::kInt64;
}

Status fill_tensor_from_constant_attrs(const std::unordered_map<std::string, ir::AttrValue>& attrs,
                                       Tensor& out) {
  const auto value_it = attrs.find("value");
  if (value_it == attrs.end()) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'constant' is missing required attribute 'value' (double array)");
  }
  const std::vector<double>* value = std::get_if<std::vector<double>>(&value_it->second);
  if (value == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'constant' attribute 'value' has wrong type, expected double array");
  }

  const auto shape_it = attrs.find("shape");
  if (shape_it == attrs.end()) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'constant' is missing required attribute 'shape'");
  }
  const Shape* shape = std::get_if<Shape>(&shape_it->second);
  if (shape == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'constant' attribute 'shape' has wrong type, expected shape");
  }

  const auto dtype_it = attrs.find("dtype");
  if (dtype_it == attrs.end()) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'constant' is missing required attribute 'dtype'");
  }
  const DType* dtype = std::get_if<DType>(&dtype_it->second);
  if (dtype == nullptr) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'constant' attribute 'dtype' has wrong type, expected dtype");
  }

  const int64_t expected_numel = shape->numel();
  if (static_cast<int64_t>(value->size()) != expected_numel) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'constant' attribute 'value' has " + std::to_string(value->size()) +
                            " element(s), attribute 'shape' " + shape->to_string() + " expects " +
                            std::to_string(expected_numel));
  }

  if (!is_constant_dtype_supported(*dtype)) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'constant' does not support dtype '" + std::string(dtype->name()) +
                            "' (supported: float32/float16/bfloat16/int32/int64)");
  }

  if (!(out.dtype() == *dtype)) {
    return Status::make(ErrorCode::kInvalidArgument, "op 'constant' output dtype '" +
                                                         std::string(out.dtype().name()) +
                                                         "' does not match attribute 'dtype' '" +
                                                         std::string(dtype->name()) + "'");
  }
  if (!(out.shape() == *shape)) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'constant' output shape " + out.shape().to_string() +
                            " does not match attribute 'shape' " + shape->to_string());
  }

  // double -> 目标 dtype:fp32 为恒等窄化拷贝,fp16/bf16 经 double -> float
  // -> 半精度两步(复用 core/dtype.h 位级转换),精度论证见头文件声明处;
  // int32/int64 直接 static_cast(调用方 infer_constant_shape 已逐元素校验
  // 整值/值域/2^53 界,本函数不重复校验)。dispatch_dtype 对 DTypeCode 全体
  // 成员编译期穷举,故本 lambda 也会为白名单之外的 dtype(如 int8_t)实例化;
  // 这些分支本体留空——白名单校验已在上方拒绝,运行时不可达。
  return dispatch_dtype(dtype->code(), [&]<typename T>() -> Status {
    if constexpr (std::is_same_v<T, float>) {
      float* out_data = out.data<float>();
      for (int64_t i = 0; i < expected_numel; ++i) {
        out_data[i] = static_cast<float>((*value)[static_cast<size_t>(i)]);
      }
    } else if constexpr (std::is_same_v<T, float16_t>) {
      float16_t* out_data = out.data<float16_t>();
      for (int64_t i = 0; i < expected_numel; ++i) {
        out_data[i] = float_to_float16(static_cast<float>((*value)[static_cast<size_t>(i)]));
      }
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
      bfloat16_t* out_data = out.data<bfloat16_t>();
      for (int64_t i = 0; i < expected_numel; ++i) {
        out_data[i] = float_to_bfloat16(static_cast<float>((*value)[static_cast<size_t>(i)]));
      }
    } else if constexpr (std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::int64_t>) {
      T* out_data = out.data<T>();
      for (int64_t i = 0; i < expected_numel; ++i) {
        out_data[i] = static_cast<T>((*value)[static_cast<size_t>(i)]);
      }
    }
    return Status::ok();
  });
}

Status encode_tensor_to_attrs(const Tensor& in,
                              std::unordered_map<std::string, ir::AttrValue>& attrs) {
  if (!is_constant_dtype_supported(in.dtype())) {
    return Status::make(ErrorCode::kInvalidArgument,
                        "op 'constant' encode does not support dtype '" +
                            std::string(in.dtype().name()) +
                            "' (supported: float32/float16/bfloat16/int32/int64)");
  }

  const int64_t numel = in.numel();
  std::vector<double> value(static_cast<size_t>(numel));
  // fp32 恒等升宽拷贝;fp16/bf16 经既有位级转换精确加宽到 float 后再升 double
  // (双精度往返,零信息损失,精度论证同上);int32/int64 逐元素校验
  // |value| <= kMaxDoubleExactInteger(2^53)后再升 double——超界值经
  // kDoubleArray 编码会静默失精,fail-loud 拒绝而非默默截断(设计门建议项,
  // 防未来整数折叠静默失精)。dispatch_dtype 全体穷举下的空分支同款理由见
  // fill_tensor_from_constant_attrs。
  const Status status = dispatch_dtype(in.dtype().code(), [&]<typename T>() -> Status {
    if constexpr (std::is_same_v<T, float>) {
      const float* in_data = static_cast<const float*>(in.raw_data());
      for (int64_t i = 0; i < numel; ++i) {
        value[static_cast<size_t>(i)] = static_cast<double>(in_data[i]);
      }
    } else if constexpr (std::is_same_v<T, float16_t>) {
      const float16_t* in_data = static_cast<const float16_t*>(in.raw_data());
      for (int64_t i = 0; i < numel; ++i) {
        value[static_cast<size_t>(i)] = static_cast<double>(float16_to_float(in_data[i]));
      }
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
      const bfloat16_t* in_data = static_cast<const bfloat16_t*>(in.raw_data());
      for (int64_t i = 0; i < numel; ++i) {
        value[static_cast<size_t>(i)] = static_cast<double>(bfloat16_to_float(in_data[i]));
      }
    } else if constexpr (std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::int64_t>) {
      const T* in_data = static_cast<const T*>(in.raw_data());
      for (int64_t i = 0; i < numel; ++i) {
        // 先升宽到 int64_t 再与 kMaxDoubleExactInteger 比较(T 为 int32_t 时
        // 若反过来把该 2^53 常量窄化到 int32_t 比较会先行溢出,比较基准本身
        // 就不对;int32_t/int64_t 升宽到 int64_t 恒无信息损失)。
        const int64_t raw_wide = static_cast<int64_t>(in_data[i]);
        if (raw_wide > kMaxDoubleExactInteger || raw_wide < -kMaxDoubleExactInteger) {
          return Status::make(ErrorCode::kInvalidArgument,
                              "op 'constant' encode: value " + std::to_string(raw_wide) +
                                  " at index " + std::to_string(i) +
                                  " exceeds the double-exact integer bound 2^53=" +
                                  std::to_string(kMaxDoubleExactInteger) +
                                  ", would lose precision when encoded as kDoubleArray");
        }
        value[static_cast<size_t>(i)] = static_cast<double>(raw_wide);
      }
    }
    return Status::ok();
  });
  FRAME_RETURN_IF_ERROR(status);

  attrs["value"] = ir::AttrValue{std::move(value)};
  attrs["shape"] = ir::AttrValue{in.shape()};
  attrs["dtype"] = ir::AttrValue{in.dtype()};
  return Status::ok();
}

}  // namespace frame::ops
