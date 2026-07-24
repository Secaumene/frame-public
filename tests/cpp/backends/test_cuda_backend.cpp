// CudaBackend 正式测试(M11,src/backends/cuda/):以 BackendRegistry 取真实
// "cuda" 注册键驱动,不 mock 任何 HAL 类型。覆盖范围:
//   0. 冒烟 S1-S4:设备枚举(device_count>=1)/allocate-deallocate 往返/
//      H2D<->D2H 字节往返(经 Backend::copy + 显式 stream->synchronize,与
//      backend-hal.md 2.1"异步拷贝"契约一致)/流同步与 event 跨流依赖(record/
//      wait/query,真实用一个异步 kernel 的产出验证依赖生效,而非仅断言 API
//      返回码)。
//   1. kernel 数值 vs cpu 参考(BUILD-011 容差,tolerance.h):add/mul/relu/
//      square 各 fp32+fp16+bf16 一例;sum 全归约(fp32+fp16+bf16,fp16/bf16
//      各驱动 reduction.cu CUB 全归约路径的 transform_iterator 升 float 累加
//      子分支)+轴归约+大规模(>=2^20 累加,放宽一档);matmul fp32
//      (严格 32F)与 fp16/bf16(半精度输入 32F 累加),统一经 cublasLtMatmul;
//      constant 物化;fused_elementwise_internal 组合调用(经 add->relu 融合链
//      图,由 runtime::compile 标准管线的 operator_fusion pass 产出真实融合
//      节点,而非手工拼 attrs)。数据流全程遵循 docs/backends/cuda.md 第7章 +
//      cuda_backend.h 头注释既定口径:host 构造 -> cuda allocator 分配 device
//      张量 -> Backend::copy H2D -> kernel/executable -> Backend::copy D2H ->
//      与 cpu 参考结果 tensor_all_close(不直接用 data<T>() 读写 device
//      张量——CudaAllocator 底层经 cudaMalloc 分配,不可 host 端直接解引用,
//      全程走 Backend::copy 排入 stream 的正规路径)。
//   2. 端到端:matmul->add->relu 图 device=cuda 经 runtime::compile("cuda")
//      编译执行 vs 同图 device=cpu 数值一致;同一 graph 对象二次调用
//      runtime::compile 命中编译缓存(同一 shared_ptr<Executable>),两次
//      run() 结果一致,FallbackStats 对图内算子零计数(全程未发生回退)。
//   3. 回退跨设备:构造仅注册 cpu kernel(无 cuda kernel、无 decomposition)的
//      测试算子,夹在 add(cuda)/relu(cuda) 之间(三明治),经 runtime::compile
//      触发整图级 FallbackExecutable,验证其内部 D2H(cuda 产出喂给 cpu
//      kernel)/H2D(cpu 产出喂回 cuda kernel)编排数值正确 + FallbackStats
//      仅对该算子计数一次(add/relu 经①直接命中不计数)。
//   4. device 校验负例:给 cuda 图编译出的 Executable::run 喂 cpu 设备张量,
//      报错消息含双方 backend 名("cuda"与"cpu")。
//
// 无 GPU 环境口径(BUILD-010/M24):BackendRegistry 未注册 "cuda"(构建时
// FRAME_ENABLE_CUDA 未开启)或 device_count()==0 时,全部用例经 fixture
// SetUp() 统一 GTEST_SKIP(英文原因);本机 RTX 5070 Ti 真实可测环境下不得
// SKIP,须真实执行(cuda.md 第7章)。本文件注册的测试专用算子/kernel 以
// "test_cuda_backend_" 前缀跨全体测试文件保持进程级唯一(同
// tests/cpp/backends/test_cpu_backend.cpp 头注释纪律)。
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <frame/compiler/autograd.h>
#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/storage.h>
#include <frame/core/tensor.h>
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/event.h>
#include <frame/hal/executable.h>
#include <frame/hal/stream.h>
#include <frame/ir/attribute.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>
#include <frame/ops/constant_utils.h>
#include <frame/ops/graph_builder.h>
#include <frame/ops/kernel_registry.h>
#include <frame/ops/op_registry.h>
#include <frame/runtime/compile.h>
#include <frame/runtime/fallback_stats.h>

#include "../common/numeric_gradient.h"
#include "../common/tolerance.h"
#include "../ir/ir_test_helpers.h"

namespace {

using frame::bfloat16_t;
using frame::Device;
using frame::DType;
using frame::DTypeCode;
using frame::ErrorCode;
using frame::float16_t;
using frame::Result;
using frame::Shape;
using frame::Status;
using frame::Tensor;
using frame::hal::Backend;
using frame::hal::BackendRegistry;
using frame::hal::CompileOptions;
using frame::hal::Executable;
using frame::hal::KernelInvocation;
using frame::ir::AttrValue;
using frame::ir::Graph;
using frame::ir::Node;
using frame::ir::Value;
using frame::ir::testing::MakeFloat32Type;
using frame::ops::AttrMap;
using frame::ops::create_node_with_inferred_types;
using frame::ops::kConstantOpName;
using frame::ops::KernelContext;
using frame::ops::NodeContext;
using frame::runtime::FallbackStats;
using frame::testing::default_tolerance;
using frame::testing::relaxed_tolerance;
using frame::testing::tensor_all_close;
using frame::testing::tf32_tolerance;

// ---------------------------------------------------------------------------
// fp16/bf16 输入构造辅助:从 float 字面量列表经既有位级转换函数构造(与
// test_runtime_compile.cpp::MakeFilledFloat16Tensor 同一手法:调用被测转换
// 函数生成"自洽"输入,不追求"已知位模式手算期望值"——本文件数值断言的基准是
// cpu 参考 kernel 的真实计算结果,不是独立手算值,故无需 bit-exact 已知量,
// REUSE-002:比 test_op_add.cpp 等文件的位级字面量惯例更简洁,取舍已在头
// 注释说明)。
std::vector<float16_t> ToFloat16(std::initializer_list<float> values) {
  std::vector<float16_t> out;
  out.reserve(values.size());
  for (float v : values) out.push_back(frame::float_to_float16(v));
  return out;
}

std::vector<bfloat16_t> ToBFloat16(std::initializer_list<float> values) {
  std::vector<bfloat16_t> out;
  out.reserve(values.size());
  for (float v : values) out.push_back(frame::float_to_bfloat16(v));
  return out;
}

// ---------------------------------------------------------------------------
// M19 Task 5 素材:elementwise 向量化访存边界用例专用。上面两个 ToFloat16/
// ToBFloat16 走 initializer_list,面向少量字面量输入;这里按 numel 生成长
// 序列,需要 vector 入参重载(重载而非改名——braced-init-list 调用仍精确匹配
// initializer_list 版本,不产生二义性,标准重载决议:identity 转换优先于经
// vector 的 initializer_list 构造这一 user-defined 转换)。
// ---------------------------------------------------------------------------
std::vector<float16_t> ToFloat16(const std::vector<float>& values) {
  std::vector<float16_t> out;
  out.reserve(values.size());
  for (float v : values) out.push_back(frame::float_to_float16(v));
  return out;
}

std::vector<bfloat16_t> ToBFloat16(const std::vector<float>& values) {
  std::vector<bfloat16_t> out;
  out.reserve(values.size());
  for (float v : values) out.push_back(frame::float_to_bfloat16(v));
  return out;
}

// add(二元代表)固定输入序列(非随机):lhs/rhs 各自独立周期,覆盖正负值。
std::vector<float> MakeVectorizationBoundaryAddLhs(int64_t numel) {
  std::vector<float> values(static_cast<size_t>(numel));
  for (int64_t i = 0; i < numel; ++i) {
    values[static_cast<size_t>(i)] = static_cast<float>((i % 7) - 3) * 0.25F;  // 覆盖 [-0.75, 1.0]
  }
  return values;
}

std::vector<float> MakeVectorizationBoundaryAddRhs(int64_t numel) {
  std::vector<float> values(static_cast<size_t>(numel));
  for (int64_t i = 0; i < numel; ++i) {
    values[static_cast<size_t>(i)] = static_cast<float>((i % 5) - 2) * 0.75F;  // 覆盖 [-1.5, 2.25]
  }
  return values;
}

// relu(一元代表)固定输入序列(非随机):含 0/正/负,确保 relu 的截断分支
// (x<=0)与透传分支(x>0)在整段序列上都被真实触发,而非退化为恒定同号输入。
std::vector<float> MakeVectorizationBoundaryReluInput(int64_t numel) {
  std::vector<float> values(static_cast<size_t>(numel));
  for (int64_t i = 0; i < numel; ++i) {
    values[static_cast<size_t>(i)] = static_cast<float>((i % 9) - 4) * 0.5F;  // 覆盖 [-2.0, 2.0]
  }
  return values;
}

// 按 T 分派到对应 dtype 的转换(float 恒等;fp16/bf16 走上面的 vector 版
// ToFloat16/ToBFloat16 位级转换)。
template <typename T>
std::vector<T> ToVectorizationBoundaryDType(const std::vector<float>& values) {
  if constexpr (std::is_same_v<T, float>) {
    return values;
  } else if constexpr (std::is_same_v<T, float16_t>) {
    return ToFloat16(values);
  } else {
    static_assert(std::is_same_v<T, bfloat16_t>,
                  "ToVectorizationBoundaryDType only supports float/float16_t/bfloat16_t");
    return ToBFloat16(values);
  }
}

// ---------------------------------------------------------------------------
// M19 Task 6 素材:matmul TF32 专项(ADR-0019)。固定序列(非随机),恒正、幅度
// 适中(约 [0.03, 0.35]):matmul 的"灾难性消去"发生在收缩维累加层面——若
// lhs/rhs 含正负混合值,K=512 项累加中大量正负项相互抵消会使某些输出元素的
// cpu 参考值本身趋近 0,导致相对偏差 |a-e|/|e| 因分母骤减而虚高,这是抵消
// 误差而非 TF32 尾数截断本身引入的偏差,会干扰容差校准判读;故本套输入恒正,
// 使每个输出元素的部分和随 K 单调累积、稳定远离 0(与本文件其余 elementwise
// 用例"覆盖正负值"的设计目标不同——那些用例覆盖的是逐元素算子的分支/符号
// 处理,不涉及累加抵消问题,见 MakeVectorizationBoundary* 系列注释)。lhs/rhs
// 各自独立周期,按 numel 参数化以复用于 64x64x64 与 64x512x64 两档矩阵规模。
// ---------------------------------------------------------------------------
std::vector<float> MakeMatmulTf32Lhs(int64_t numel) {
  std::vector<float> values(static_cast<size_t>(numel));
  for (int64_t i = 0; i < numel; ++i) {
    values[static_cast<size_t>(i)] =
        0.05F + static_cast<float>(i % 7) * 0.05F;  // 覆盖 [0.05, 0.35],恒正
  }
  return values;
}

std::vector<float> MakeMatmulTf32Rhs(int64_t numel) {
  std::vector<float> values(static_cast<size_t>(numel));
  for (int64_t i = 0; i < numel; ++i) {
    values[static_cast<size_t>(i)] =
        0.03F + static_cast<float>(i % 11) * 0.03F;  // 覆盖 [0.03, 0.33],恒正
  }
  return values;
}

// TF32 校准诊断:逐元素求最大绝对/相对偏差,仅用于打印观测数据供 ADR-0019
// 容差终值回填判断——不是判等断言,不参与任何 EXPECT_*/ASSERT_*,不受
// BUILD-011"禁止手写 EXPECT_NEAR/自造阈值"管控(该条款约束参与 PASS/FAIL
// 判定的容差,本结构体纯诊断打印)。相对偏差分母恰为 0 的元素跳过相对偏差
// 统计(避免除零),其绝对偏差仍计入 max_abs_diff。
struct MatmulTf32Deviation {
  double max_abs_diff = 0.0;
  double max_rel_diff = 0.0;
};

MatmulTf32Deviation ComputeMatmulTf32Deviation(const Tensor& actual, const Tensor& expected) {
  MatmulTf32Deviation deviation;
  const float* actual_data = static_cast<const float*>(actual.raw_data());
  const float* expected_data = static_cast<const float*>(expected.raw_data());
  const int64_t numel = actual.numel();
  for (int64_t i = 0; i < numel; ++i) {
    const double a = static_cast<double>(actual_data[i]);
    const double e = static_cast<double>(expected_data[i]);
    const double abs_diff = std::fabs(a - e);
    deviation.max_abs_diff = std::max(deviation.max_abs_diff, abs_diff);
    if (e != 0.0) {
      deviation.max_rel_diff = std::max(deviation.max_rel_diff, abs_diff / std::fabs(e));
    }
  }
  return deviation;
}

// ---------------------------------------------------------------------------
// M21 T5 素材:conv2d/pool2d 系一致性用例的固定输入序列(非随机,覆盖正负值,
// REUSE-002:与 MakeVectorizationBoundary* 系列同一动机)。本节全部用例的
// PASS/FAIL 判据是"cuda 结果 vs cpu 参考"(cpu kernel 为受信参考实现,同本
// 文件第1节既有 add/relu/matmul 等用例的既定方法论),故不要求输入值经手算
// 验证——与 tests/cpp/ops/test_conv2d.cpp 等"验证 cpu kernel 自身正确性"的
// 已知值用例分工不同(那类用例已在 T4 交付,不在本文件重复)。modulus 控制
// 周期、scale 控制幅度;max_pool2d 系用例的 argmax 平局约定由 cpu/cuda 两侧
// 共享同一钉死规则(kh 外 kw 内、严格 `>`),故重复取值(平局)不影响
// cuda==cpu 一致性判据的有效性。
// ---------------------------------------------------------------------------
// 相邻可转换形参 (count, scale, modulus) 为本生成器固定契约序,调用点均以
// 具名字面量传入,误置换会使张量元素数瞬间失配并在形状校验报错(同
// nchw_index 先例论证)。
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::vector<float> MakeM21PatternValues(int64_t count, float scale, int64_t modulus) {
  std::vector<float> values(static_cast<size_t>(count));
  // 整数除法为本意:图样以 modulus 的整数半程为中心对称偏移。
  const int64_t half_modulus = modulus / 2;
  for (int64_t i = 0; i < count; ++i) {
    values[static_cast<size_t>(i)] = scale * static_cast<float>((i % modulus) - half_modulus);
  }
  return values;
}

// ---------------------------------------------------------------------------
// 交付点3 素材:仅注册 cpu kernel(无 cuda kernel、无 decomposition)的测试
// 算子,恰1输入1输出,shape 恒等——驱动 FallbackExecutable 的③跳
// (docs/architecture/execution-model.md 第5章),夹在 add(cuda)/relu(cuda)
// 之间构成三明治场景(同 m10-design-brief 决议点 E 先例,tests/cpp/runtime/
// test_fallback_chain.cpp::kOp1Name 同款结构)。schema/kernel 注册见文件尾。
// ---------------------------------------------------------------------------
constexpr std::string_view kCpuOnlySandwichOpName = "test_cuda_backend_cpu_only_sandwich_op";

Result<std::vector<Shape>> InferCpuOnlySandwichOpShape(const NodeContext& ctx) {
  if (ctx.input_types.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument, "op '" + std::string(kCpuOnlySandwichOpName) +
                                                         "' expects 1 input, got " +
                                                         std::to_string(ctx.input_types.size()));
  }
  return std::vector<Shape>{ctx.input_types[0].shape};
}

// out = in * 2 - 1(非平凡、与 add/relu 数值可区分的变换,cpu-only,便于观察
// 该步是否真的在 host 端执行)。
Status CpuOnlySandwichOpKernel(KernelContext& ctx) {
  if (ctx.inputs.size() != 1 || ctx.outputs.size() != 1) {
    return Status::make(ErrorCode::kInvalidArgument, "op '" + std::string(kCpuOnlySandwichOpName) +
                                                         "' cpu kernel expects 1 input and 1 "
                                                         "output");
  }
  const float* in = static_cast<const float*>(ctx.inputs[0].raw_data());
  float* out = ctx.outputs[0].data<float>();
  const int64_t numel = ctx.outputs[0].numel();
  for (int64_t i = 0; i < numel; ++i) out[i] = in[i] * 2.0F - 1.0F;
  return Status::ok();
}

// ---------------------------------------------------------------------------
// fixture:真实 cuda/cpu 后端 + 分配器 + 一条常驻 cuda stream。SetUp()
// 统一处理 BUILD-010/M24 SKIP 口径:cuda 后端未注册(cpu-only 构建)或
// device_count()==0(构建了 cuda 后端但本机无卡)均 GTEST_SKIP,英文原因。
// GTEST_SKIP() 宏自身即含 return(gtest 定义:展开为 `return
// AssertHelper(...) = Message()`,函数返回 void 时合法),故各分支内无需额外
// return 语句。
// ---------------------------------------------------------------------------
class CudaBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const Result<Backend*> cuda_result = BackendRegistry::instance().get(frame::kCudaBackendName);
    if (!cuda_result.is_ok()) {
      GTEST_SKIP() << "CUDA backend 'cuda' is not registered in this build (configured without "
                      "FRAME_ENABLE_CUDA or CUDA Toolkit was not detected at configure time)";
    }
    cuda_backend_ = cuda_result.value();

    const Result<int32_t> count = cuda_backend_->device_count();
    ASSERT_TRUE(count.is_ok()) << count.status().message();
    if (count.value() < 1) {
      GTEST_SKIP() << "no CUDA device available on this machine (CudaBackend::device_count() "
                      "returned 0)";
    }

    const Result<Backend*> cpu_result = BackendRegistry::instance().get(frame::kCpuBackendName);
    ASSERT_TRUE(cpu_result.is_ok()) << cpu_result.status().message();
    cpu_backend_ = cpu_result.value();

    device_ = Device{frame::kCudaBackendName, 0};
    cpu_device_ = frame::cpu_device();

    cuda_allocator_ = cuda_backend_->allocator(device_);
    ASSERT_NE(cuda_allocator_, nullptr);
    cpu_allocator_ = cpu_backend_->allocator(cpu_device_);
    ASSERT_NE(cpu_allocator_, nullptr);

    Result<std::unique_ptr<frame::hal::Stream>> stream_result =
        cuda_backend_->create_stream(device_);
    ASSERT_TRUE(stream_result.is_ok()) << stream_result.status().message();
    stream_ = std::move(stream_result.value());
  }

  // ---- host Tensor 构造(cpu 后端真实 Allocator;同
  //      elementwise_op_test_helpers.h::MakeTensor1D/MakeTensorWithShape 同一
  //      思路,独立实现——本 fixture 的 device_/allocator_ 是 cuda 侧而非 cpu
  //      侧,该头文件的共用 fixture 假设不适用,过于琐碎不值得为此再抽取共享
  //      头,同 test_fallback_chain.cpp::MakeFilledTensor 头注释先例)----
  template <typename T>
  Tensor MakeHostTensor1D(const std::vector<T>& values) {
    Result<Tensor> result = Tensor::empty(Shape({static_cast<int64_t>(values.size())}),
                                          DType::of<T>(), cpu_device_, *cpu_allocator_);
    EXPECT_TRUE(result.is_ok()) << result.status().message();
    Tensor tensor = result.value();
    T* data = tensor.data<T>();
    for (size_t i = 0; i < values.size(); ++i) data[i] = values[i];
    return tensor;
  }

  template <typename T>
  Tensor MakeHostTensorWithShape(const Shape& shape, const std::vector<T>& flat_values) {
    Result<Tensor> result = Tensor::empty(shape, DType::of<T>(), cpu_device_, *cpu_allocator_);
    EXPECT_TRUE(result.is_ok()) << result.status().message();
    Tensor tensor = result.value();
    EXPECT_EQ(static_cast<int64_t>(flat_values.size()), tensor.numel());
    T* data = tensor.data<T>();
    for (size_t i = 0; i < flat_values.size(); ++i) data[i] = flat_values[i];
    return tensor;
  }

  // ---- H2D/D2H:经 Backend::copy 排入 stream_(cuda_backend.h 头注释既定
  //      口径,不直接 data<T>() 读写 device 张量)----
  Tensor CopyToDevice(const Tensor& host) {
    Result<Tensor> device_result =
        Tensor::empty(host.shape(), host.dtype(), device_, *cuda_allocator_);
    EXPECT_TRUE(device_result.is_ok()) << device_result.status().message();
    Tensor device_tensor = device_result.value();
    const size_t bytes = static_cast<size_t>(host.numel()) * host.dtype().itemsize();
    if (bytes > 0) {
      const Status status = cuda_backend_->copy(device_tensor.raw_data(), device_, host.raw_data(),
                                                cpu_device_, bytes, stream_.get());
      EXPECT_TRUE(status.is_ok()) << status.message();
    }
    return device_tensor;
  }

  Tensor CopyToHost(const Tensor& device_tensor) {
    Result<Tensor> host_result =
        Tensor::empty(device_tensor.shape(), device_tensor.dtype(), cpu_device_, *cpu_allocator_);
    EXPECT_TRUE(host_result.is_ok()) << host_result.status().message();
    Tensor host_tensor = host_result.value();
    const size_t bytes =
        static_cast<size_t>(device_tensor.numel()) * device_tensor.dtype().itemsize();
    if (bytes > 0) {
      const Status status =
          cuda_backend_->copy(host_tensor.raw_data(), cpu_device_, device_tensor.raw_data(),
                              device_, bytes, stream_.get());
      EXPECT_TRUE(status.is_ok()) << status.message();
    }
    return host_tensor;
  }

  // ---- kernel 数值比对通用驱动:host_inputs -> H2D -> Backend::launch(cuda)
  //      -> D2H -> stream_->synchronize() -> 返回 host 张量。仅支持单输出
  //      算子(本文件全部数值用例恰好如此)----
  Tensor RunOnCuda(std::string_view op, const std::vector<Tensor>& host_inputs,
                   const Shape& output_shape, DType output_dtype,
                   const std::unordered_map<std::string, AttrValue>* attrs = nullptr) {
    std::vector<Tensor> device_inputs;
    device_inputs.reserve(host_inputs.size());
    for (const Tensor& host : host_inputs) device_inputs.push_back(CopyToDevice(host));

    Result<Tensor> device_output_result =
        Tensor::empty(output_shape, output_dtype, device_, *cuda_allocator_);
    EXPECT_TRUE(device_output_result.is_ok()) << device_output_result.status().message();
    std::vector<Tensor> device_outputs{device_output_result.value()};

    KernelInvocation invocation;
    invocation.op = op;
    invocation.inputs = device_inputs;
    invocation.outputs = device_outputs;
    invocation.attrs = attrs;
    invocation.device = device_;
    const Status launch_status = cuda_backend_->launch(invocation, stream_.get());
    EXPECT_TRUE(launch_status.is_ok()) << launch_status.message();

    Tensor host_output = CopyToHost(device_outputs[0]);
    const Status sync_status = stream_->synchronize();
    EXPECT_TRUE(sync_status.is_ok()) << sync_status.message();
    return host_output;
  }

  Tensor RunOnCpu(std::string_view op, const std::vector<Tensor>& host_inputs,
                  const Shape& output_shape, DType output_dtype,
                  const std::unordered_map<std::string, AttrValue>* attrs = nullptr) {
    Result<Tensor> output_result =
        Tensor::empty(output_shape, output_dtype, cpu_device_, *cpu_allocator_);
    EXPECT_TRUE(output_result.is_ok()) << output_result.status().message();
    std::vector<Tensor> outputs{output_result.value()};

    KernelInvocation invocation;
    invocation.op = op;
    invocation.inputs = host_inputs;
    invocation.outputs = outputs;
    invocation.attrs = attrs;
    invocation.device = cpu_device_;
    const Status status = cpu_backend_->launch(invocation, nullptr);
    EXPECT_TRUE(status.is_ok()) << status.message();
    return outputs[0];
  }

  template <typename T>
  void ExpectBinaryElementwiseMatchesCpu(std::string_view op, DTypeCode code,
                                         const std::vector<T>& lhs_values,
                                         const std::vector<T>& rhs_values) {
    const Tensor lhs = MakeHostTensor1D<T>(lhs_values);
    const Tensor rhs = MakeHostTensor1D<T>(rhs_values);
    const Shape shape({static_cast<int64_t>(lhs_values.size())});
    const Tensor cuda_out = RunOnCuda(op, {lhs, rhs}, shape, DType::of<T>());
    const Tensor cpu_out = RunOnCpu(op, {lhs, rhs}, shape, DType::of<T>());
    EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, default_tolerance(code)));
  }

  template <typename T>
  void ExpectUnaryElementwiseMatchesCpu(std::string_view op, DTypeCode code,
                                        const std::vector<T>& values) {
    const Tensor in = MakeHostTensor1D<T>(values);
    const Shape shape({static_cast<int64_t>(values.size())});
    const Tensor cuda_out = RunOnCuda(op, {in}, shape, DType::of<T>());
    const Tensor cpu_out = RunOnCpu(op, {in}, shape, DType::of<T>());
    EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, default_tolerance(code)));
  }

  // M19 Task 5 守护:add(二元代表)向量化访存边界,按 numel 生成固定序列后
  // 复用上面的 ExpectBinaryElementwiseMatchesCpu(不重复 H2D/launch/D2H 骨架)。
  template <typename T>
  void ExpectAddVectorizationBoundaryMatchesCpu(DTypeCode code, int64_t numel) {
    const std::vector<T> lhs =
        ToVectorizationBoundaryDType<T>(MakeVectorizationBoundaryAddLhs(numel));
    const std::vector<T> rhs =
        ToVectorizationBoundaryDType<T>(MakeVectorizationBoundaryAddRhs(numel));
    ExpectBinaryElementwiseMatchesCpu<T>("add", code, lhs, rhs);
  }

  // M19 Task 5 守护:relu(一元代表)向量化访存边界,同上复用
  // ExpectUnaryElementwiseMatchesCpu。
  template <typename T>
  void ExpectReluVectorizationBoundaryMatchesCpu(DTypeCode code, int64_t numel) {
    const std::vector<T> values =
        ToVectorizationBoundaryDType<T>(MakeVectorizationBoundaryReluInput(numel));
    ExpectUnaryElementwiseMatchesCpu<T>("relu", code, values);
  }

  // add->relu 两节点链(满足 kElementwise+kFusable 融合条件,同
  // tests/cpp/compiler/testdata/operator_fusion_two_node_chain_input.txt 图
  // 形态):供融合链数值用例与跨设备回退三明治用例复用图构造骨架。
  static Graph BuildAddReluGraph(Device device) {
    Graph graph("test_cuda_backend_add_relu");
    Value* a = graph.add_graph_input(MakeFloat32Type({4}, device)).value();
    Value* b = graph.add_graph_input(MakeFloat32Type({4}, device)).value();
    Node* add_node = graph.create_node("add", {a, b}, {MakeFloat32Type({4}, device)}).value();
    Node* relu_node =
        graph.create_node("relu", {add_node->output(0)}, {MakeFloat32Type({4}, device)}).value();
    graph.mark_output(relu_node, 0);
    return graph;
  }

  static Graph BuildMatmulAddReluGraph(Device device) {
    Graph graph("test_cuda_backend_matmul_add_relu");
    Value* x = graph.add_graph_input(MakeFloat32Type({2, 3}, device)).value();
    Value* w = graph.add_graph_input(MakeFloat32Type({3, 4}, device)).value();
    Value* bias = graph.add_graph_input(MakeFloat32Type({2, 4}, device)).value();
    Node* matmul_node =
        graph.create_node("matmul", {x, w}, {MakeFloat32Type({2, 4}, device)}).value();
    Node* add_node =
        graph.create_node("add", {matmul_node->output(0), bias}, {MakeFloat32Type({2, 4}, device)})
            .value();
    Node* relu_node =
        graph.create_node("relu", {add_node->output(0)}, {MakeFloat32Type({2, 4}, device)}).value();
    graph.mark_output(relu_node, 0);
    return graph;
  }

  // M19 Task 6:纯 matmul 单节点图(TF32 专项用,shape 由调用方参数化
  // M/K/N),复用 BuildMatmulAddReluGraph 同一图构造手法。
  static Graph BuildMatmulOnlyGraph(Device device, int64_t m, int64_t k, int64_t n) {
    Graph graph("test_cuda_backend_matmul_tf32");
    Value* lhs = graph.add_graph_input(MakeFloat32Type({m, k}, device)).value();
    Value* rhs = graph.add_graph_input(MakeFloat32Type({k, n}, device)).value();
    Node* matmul_node =
        graph.create_node("matmul", {lhs, rhs}, {MakeFloat32Type({m, n}, device)}).value();
    graph.mark_output(matmul_node, 0);
    return graph;
  }

  // M19 Task 6:matmul 专项编译执行驱动。不同于 RunOnCuda/RunOnCpu——那两个
  // 经 Backend::launch 直接调用 kernel,KernelInvocation 不携带
  // CompileOptions(cuda_backend.cpp::launch 构造 KernelContext 时
  // compile_options 恒为 nullptr),allow_tf32 在该路径下恒不生效。本方法改走
  // frame::runtime::compile + Executable::run(CudaExecutable::compile 把
  // options 按值存入 options_,run() 期经 &options_ 下传 KernelContext,见
  // src/backends/cuda/cuda_executable.h/.cpp 头注释),CompileOptions 由调用方
  // 显式传入才会真正传导到 matmul_cuda_kernel 的 compute_type 选择。
  Tensor RunMatmulGraphOnCuda(const Graph& graph, const Tensor& lhs_host, const Tensor& rhs_host,
                              const Shape& out_shape, const CompileOptions& options) {
    const Result<std::shared_ptr<Executable>> executable =
        frame::runtime::compile(graph, frame::kCudaBackendName, options);
    EXPECT_TRUE(executable.is_ok()) << executable.status().message();
    const Tensor lhs_dev = CopyToDevice(lhs_host);
    const Tensor rhs_dev = CopyToDevice(rhs_host);
    std::vector<Tensor> inputs{lhs_dev, rhs_dev};
    Result<Tensor> out_result =
        Tensor::empty(out_shape, DType::of<float>(), device_, *cuda_allocator_);
    EXPECT_TRUE(out_result.is_ok()) << out_result.status().message();
    std::vector<Tensor> outputs{out_result.value()};
    const Status run_status = executable.value()->run(inputs, outputs, *stream_);
    EXPECT_TRUE(run_status.is_ok()) << run_status.message();
    const Tensor host_out = CopyToHost(outputs[0]);
    EXPECT_TRUE(stream_->synchronize().is_ok());
    return host_out;
  }

  Tensor RunMatmulGraphOnCpu(const Graph& graph, const Tensor& lhs_host, const Tensor& rhs_host,
                             const Shape& out_shape) {
    const Result<std::shared_ptr<Executable>> executable =
        frame::runtime::compile(graph, frame::kCpuBackendName, CompileOptions{});
    EXPECT_TRUE(executable.is_ok()) << executable.status().message();
    std::vector<Tensor> inputs{lhs_host, rhs_host};
    Result<Tensor> out_result =
        Tensor::empty(out_shape, DType::of<float>(), cpu_device_, *cpu_allocator_);
    EXPECT_TRUE(out_result.is_ok()) << out_result.status().message();
    std::vector<Tensor> outputs{out_result.value()};
    Result<std::unique_ptr<frame::hal::Stream>> cpu_stream_result =
        cpu_backend_->create_stream(cpu_device_);
    EXPECT_TRUE(cpu_stream_result.is_ok()) << cpu_stream_result.status().message();
    const Status run_status = executable.value()->run(inputs, outputs, *cpu_stream_result.value());
    EXPECT_TRUE(run_status.is_ok()) << run_status.message();
    return outputs[0];
  }

  // 构造给定 dtype/shape/device 的 TensorType(M21 T5:与
  // tests/cpp/ir/ir_test_helpers.h::MakeFloat32Type 同思路,但 dtype/device
  // 均可参数化——该文件的 MakeFloat32Type 固定 float32、本文件已有的
  // frame::ops::testing::MakeType 固定 cpu_device,两者均不满足本文件"任意
  // dtype 的图节点跑在 cuda 设备"的需要,故补一个局部小工具,不改动那两处)。
  static frame::ir::TensorType MakeDeviceType(DType dtype, const Shape& shape, Device device) {
    frame::ir::TensorType type;
    type.dtype = dtype;
    type.shape = shape;
    type.layout = frame::ir::Layout::kRowMajor;
    type.device = device;
    return type;
  }

  // 直接编译并执行 schema 注册的 GradientFn 微图；用于覆盖空张量在 CUDA
  // executable 中的零字节中间 arena entry，而非仅验证前向单算子早退。
  std::vector<Tensor> RunGradientMicrographOnCuda(
      std::string_view op, const std::vector<frame::ir::TensorType>& forward_input_types,
      const AttrMap* attrs, const std::vector<Tensor>& host_micrograph_inputs) {
    const frame::ops::OpSchema* schema = frame::ops::OpRegistry::instance().find(op);
    EXPECT_NE(schema, nullptr);
    NodeContext ctx;
    ctx.op = op;
    ctx.input_types = forward_input_types;
    ctx.attrs = attrs;
    const Result<Graph> micrograph = schema->gradient()(ctx);
    EXPECT_TRUE(micrograph.is_ok()) << micrograph.status().message();
    const Result<std::shared_ptr<Executable>> executable =
        frame::runtime::compile(micrograph.value(), frame::kCudaBackendName, CompileOptions{});
    EXPECT_TRUE(executable.is_ok()) << executable.status().message();

    std::vector<Tensor> device_inputs;
    device_inputs.reserve(host_micrograph_inputs.size());
    for (const Tensor& host : host_micrograph_inputs) {
      device_inputs.push_back(CopyToDevice(host));
    }
    const Result<std::vector<Tensor>> device_outputs = frame::runtime::run_with_allocated_outputs(
        *executable.value(), frame::kCudaBackendName, device_inputs);
    EXPECT_TRUE(device_outputs.is_ok()) << device_outputs.status().message();

    std::vector<Tensor> host_outputs;
    host_outputs.reserve(device_outputs.value().size());
    for (const Tensor& device_output : device_outputs.value()) {
      host_outputs.push_back(CopyToHost(device_output));
    }
    EXPECT_TRUE(stream_->synchronize().is_ok());
    return host_outputs;
  }

  // M21 T5:任意 op 的 cuda 侧图编译执行驱动(泛化 matmul TF32 专项的
  // RunMatmulGraphOnCuda/RunMatmulGraphOnCpu 为多输入/任意属性/输出形状经
  // shape_infer 自动推断——不再要求调用方手算 out_shape)。经
  // frame::runtime::compile + run_with_allocated_outputs 执行(不走 eager
  // Backend::launch 旁路:KernelInvocation 不携带 CompileOptions,
  // allow_tf32 唯经图编译路径才真正传导到 kernel,理由同
  // RunMatmulGraphOnCuda 头注释)。host_inputs 逐个 CopyToDevice 后跑图;
  // run_with_allocated_outputs 内部自建 stream 并在返回前
  // synchronize(),故结果在 host 端 CopyToHost 前已确定就绪(happens-before
  // 关系由该次 host 端阻塞同步建立,与本 fixture 的 stream_ 是否为同一
  // stream 无关)。
  Tensor RunOpGraphOnCuda(std::string_view op, const std::vector<Tensor>& host_inputs,
                          const AttrMap& attrs, const CompileOptions& options) {
    Graph graph("test_cuda_backend_graph_" + std::string(op));
    std::vector<Value*> graph_inputs;
    graph_inputs.reserve(host_inputs.size());
    for (const Tensor& host : host_inputs) {
      const Result<Value*> v =
          graph.add_graph_input(MakeDeviceType(host.dtype(), host.shape(), device_));
      EXPECT_TRUE(v.is_ok()) << v.status().message();
      graph_inputs.push_back(v.value());
    }
    const Result<Node*> node =
        create_node_with_inferred_types(graph, std::string(op), graph_inputs, attrs);
    EXPECT_TRUE(node.is_ok()) << node.status().message();
    EXPECT_TRUE(graph.mark_output(node.value(), 0).is_ok());

    const Result<std::shared_ptr<Executable>> executable =
        frame::runtime::compile(graph, frame::kCudaBackendName, options);
    EXPECT_TRUE(executable.is_ok()) << executable.status().message();

    std::vector<Tensor> device_inputs;
    device_inputs.reserve(host_inputs.size());
    for (const Tensor& host : host_inputs) device_inputs.push_back(CopyToDevice(host));

    const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
        *executable.value(), frame::kCudaBackendName, device_inputs);
    EXPECT_TRUE(outputs.is_ok()) << outputs.status().message();

    const Tensor host_out = CopyToHost(outputs.value()[0]);
    EXPECT_TRUE(stream_->synchronize().is_ok());
    return host_out;
  }

  // M21 T5:cpu 侧对照(与 RunOpGraphOnCuda 同参数形态);cpu Device 即主机
  // 内存,host_inputs 无需搬运,直接喂给 run_with_allocated_outputs。
  Tensor RunOpGraphOnCpu(std::string_view op, const std::vector<Tensor>& host_inputs,
                         const AttrMap& attrs) {
    Graph graph("test_cuda_backend_graph_cpu_" + std::string(op));
    std::vector<Value*> graph_inputs;
    graph_inputs.reserve(host_inputs.size());
    for (const Tensor& host : host_inputs) {
      const Result<Value*> v =
          graph.add_graph_input(MakeDeviceType(host.dtype(), host.shape(), cpu_device_));
      EXPECT_TRUE(v.is_ok()) << v.status().message();
      graph_inputs.push_back(v.value());
    }
    const Result<Node*> node =
        create_node_with_inferred_types(graph, std::string(op), graph_inputs, attrs);
    EXPECT_TRUE(node.is_ok()) << node.status().message();
    EXPECT_TRUE(graph.mark_output(node.value(), 0).is_ok());

    const Result<std::shared_ptr<Executable>> executable =
        frame::runtime::compile(graph, frame::kCpuBackendName, CompileOptions{});
    EXPECT_TRUE(executable.is_ok()) << executable.status().message();

    const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
        *executable.value(), frame::kCpuBackendName, host_inputs);
    EXPECT_TRUE(outputs.is_ok()) << outputs.status().message();
    return outputs.value()[0];
  }

  // ---------------------------------------------------------------------------
  // M21 T5:conv2d/pool2d/sigmoid/reshape 一致性用例专用 Expect* 封装(镜像本
  // 文件第1节 ExpectBinaryElementwiseMatchesCpu/ExpectUnaryElementwiseMatchesCpu
  // 风格,经 RunOpGraphOnCuda/RunOpGraphOnCpu 驱动图编译路径)。dtype 转换复用
  // 既有 ToVectorizationBoundaryDType<T>(REUSE-002)。
  // ---------------------------------------------------------------------------

  template <typename T>
  void ExpectConv2dMatchesCpu(DTypeCode code, const Shape& x_shape,
                              const std::vector<float>& x_values_f, const Shape& w_shape,
                              const std::vector<float>& w_values_f,
                              const std::vector<float>* bias_values_f,
                              const std::vector<int64_t>& stride,
                              const std::vector<int64_t>& padding, int64_t groups) {
    const Tensor x =
        MakeHostTensorWithShape<T>(x_shape, ToVectorizationBoundaryDType<T>(x_values_f));
    const Tensor w =
        MakeHostTensorWithShape<T>(w_shape, ToVectorizationBoundaryDType<T>(w_values_f));
    std::vector<Tensor> inputs{x, w};
    if (bias_values_f != nullptr) {
      inputs.push_back(MakeHostTensorWithShape<T>(Shape({w_shape.dim(0)}),
                                                  ToVectorizationBoundaryDType<T>(*bias_values_f)));
    }
    const AttrMap attrs{{"stride", stride}, {"padding", padding}, {"groups", groups}};
    const Tensor cuda_out = RunOpGraphOnCuda("conv2d", inputs, attrs, CompileOptions{});
    const Tensor cpu_out = RunOpGraphOnCpu("conv2d", inputs, attrs);
    EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, default_tolerance(code)));
  }

  // 相邻同型 Shape/vector 形参为算子输入契约序(与 schema 输入序一致),调用
  // 点均以具名局部变量传入,误置换会在 shape 校验立即报错(同
  // compute_conv2d_geometry 先例论证)。以下三个 Expect* 同理。
  // NOLINTBEGIN(bugprone-easily-swappable-parameters)
  template <typename T>
  void ExpectConv2dGradInputMatchesCpu(DTypeCode code, const Shape& input_shape,
                                       const Shape& dy_shape, const std::vector<float>& dy_values_f,
                                       const Shape& w_shape, const std::vector<float>& w_values_f,
                                       const std::vector<int64_t>& stride,
                                       const std::vector<int64_t>& padding, int64_t groups) {
    const Tensor dy =
        MakeHostTensorWithShape<T>(dy_shape, ToVectorizationBoundaryDType<T>(dy_values_f));
    const Tensor w =
        MakeHostTensorWithShape<T>(w_shape, ToVectorizationBoundaryDType<T>(w_values_f));
    std::vector<Tensor> inputs{dy, w};
    const AttrMap attrs{
        {"input_shape", input_shape}, {"stride", stride}, {"padding", padding}, {"groups", groups}};
    const Tensor cuda_out =
        RunOpGraphOnCuda("conv2d_grad_input_internal", inputs, attrs, CompileOptions{});
    const Tensor cpu_out = RunOpGraphOnCpu("conv2d_grad_input_internal", inputs, attrs);
    EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, default_tolerance(code)));
  }

  template <typename T>
  void ExpectConv2dGradFilterMatchesCpu(DTypeCode code, const Shape& x_shape,
                                        const std::vector<float>& x_values_f, const Shape& dy_shape,
                                        const std::vector<float>& dy_values_f,
                                        const Shape& filter_shape,
                                        const std::vector<int64_t>& stride,
                                        const std::vector<int64_t>& padding, int64_t groups) {
    const Tensor x =
        MakeHostTensorWithShape<T>(x_shape, ToVectorizationBoundaryDType<T>(x_values_f));
    const Tensor dy =
        MakeHostTensorWithShape<T>(dy_shape, ToVectorizationBoundaryDType<T>(dy_values_f));
    std::vector<Tensor> inputs{x, dy};
    const AttrMap attrs{{"filter_shape", filter_shape},
                        {"stride", stride},
                        {"padding", padding},
                        {"groups", groups}};
    const Tensor cuda_out =
        RunOpGraphOnCuda("conv2d_grad_filter_internal", inputs, attrs, CompileOptions{});
    const Tensor cpu_out = RunOpGraphOnCpu("conv2d_grad_filter_internal", inputs, attrs);
    EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, default_tolerance(code)));
  }

  template <typename T>
  void ExpectPool2dForwardMatchesCpu(std::string_view op, DTypeCode code, const Shape& x_shape,
                                     const std::vector<float>& x_values_f,
                                     const std::vector<int64_t>& kernel,
                                     const std::vector<int64_t>& stride,
                                     const std::vector<int64_t>& padding) {
    const Tensor x =
        MakeHostTensorWithShape<T>(x_shape, ToVectorizationBoundaryDType<T>(x_values_f));
    std::vector<Tensor> inputs{x};
    const AttrMap attrs{{"kernel", kernel}, {"stride", stride}, {"padding", padding}};
    const Tensor cuda_out = RunOpGraphOnCuda(op, inputs, attrs, CompileOptions{});
    const Tensor cpu_out = RunOpGraphOnCpu(op, inputs, attrs);
    EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, default_tolerance(code)));
  }

  template <typename T>
  void ExpectMaxPool2dGradInternalMatchesCpu(DTypeCode code, const Shape& input_shape,
                                             const Shape& dy_shape,
                                             const std::vector<float>& dy_values_f,
                                             const std::vector<float>& x_values_f,
                                             const std::vector<int64_t>& kernel,
                                             const std::vector<int64_t>& stride,
                                             const std::vector<int64_t>& padding) {
    const Tensor dy =
        MakeHostTensorWithShape<T>(dy_shape, ToVectorizationBoundaryDType<T>(dy_values_f));
    const Tensor x =
        MakeHostTensorWithShape<T>(input_shape, ToVectorizationBoundaryDType<T>(x_values_f));
    std::vector<Tensor> inputs{dy, x};
    const AttrMap attrs{
        {"input_shape", input_shape}, {"kernel", kernel}, {"stride", stride}, {"padding", padding}};
    const Tensor cuda_out =
        RunOpGraphOnCuda("max_pool2d_grad_internal", inputs, attrs, CompileOptions{});
    const Tensor cpu_out = RunOpGraphOnCpu("max_pool2d_grad_internal", inputs, attrs);
    EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, default_tolerance(code)));
  }

  template <typename T>
  void ExpectMaxPool2dSelectInternalMatchesCpu(DTypeCode code, const Shape& x_shape,
                                               const std::vector<float>& g_values_f,
                                               const std::vector<float>& x_values_f,
                                               const std::vector<int64_t>& kernel,
                                               const std::vector<int64_t>& stride,
                                               const std::vector<int64_t>& padding) {
    const Tensor g =
        MakeHostTensorWithShape<T>(x_shape, ToVectorizationBoundaryDType<T>(g_values_f));
    const Tensor x =
        MakeHostTensorWithShape<T>(x_shape, ToVectorizationBoundaryDType<T>(x_values_f));
    std::vector<Tensor> inputs{g, x};
    const AttrMap attrs{{"kernel", kernel}, {"stride", stride}, {"padding", padding}};
    const Tensor cuda_out =
        RunOpGraphOnCuda("max_pool2d_select_internal", inputs, attrs, CompileOptions{});
    const Tensor cpu_out = RunOpGraphOnCpu("max_pool2d_select_internal", inputs, attrs);
    EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, default_tolerance(code)));
  }

  template <typename T>
  void ExpectAvgPool2dGradInternalMatchesCpu(DTypeCode code, const Shape& input_shape,
                                             const Shape& dy_shape,
                                             const std::vector<float>& dy_values_f,
                                             const std::vector<int64_t>& kernel,
                                             const std::vector<int64_t>& stride,
                                             const std::vector<int64_t>& padding) {
    const Tensor dy =
        MakeHostTensorWithShape<T>(dy_shape, ToVectorizationBoundaryDType<T>(dy_values_f));
    std::vector<Tensor> inputs{dy};
    const AttrMap attrs{
        {"input_shape", input_shape}, {"kernel", kernel}, {"stride", stride}, {"padding", padding}};
    const Tensor cuda_out =
        RunOpGraphOnCuda("avg_pool2d_grad_internal", inputs, attrs, CompileOptions{});
    const Tensor cpu_out = RunOpGraphOnCpu("avg_pool2d_grad_internal", inputs, attrs);
    EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, default_tolerance(code)));
  }
  // NOLINTEND(bugprone-easily-swappable-parameters)

  template <typename T>
  void ExpectSigmoidGraphMatchesCpu(DTypeCode code, const Shape& x_shape,
                                    const std::vector<float>& x_values_f) {
    const Tensor x =
        MakeHostTensorWithShape<T>(x_shape, ToVectorizationBoundaryDType<T>(x_values_f));
    std::vector<Tensor> inputs{x};
    const AttrMap attrs{};
    const Tensor cuda_out = RunOpGraphOnCuda("sigmoid", inputs, attrs, CompileOptions{});
    const Tensor cpu_out = RunOpGraphOnCpu("sigmoid", inputs, attrs);
    EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, default_tolerance(code)));
  }

  template <typename T>
  void ExpectReshapeGraphMatchesCpu(DTypeCode code, const Shape& x_shape,
                                    const std::vector<float>& x_values_f,
                                    const Shape& target_shape) {
    const Tensor x =
        MakeHostTensorWithShape<T>(x_shape, ToVectorizationBoundaryDType<T>(x_values_f));
    std::vector<Tensor> inputs{x};
    const AttrMap attrs{{"target_shape", target_shape}};
    const Tensor cuda_out = RunOpGraphOnCuda("reshape", inputs, attrs, CompileOptions{});
    const Tensor cpu_out = RunOpGraphOnCpu("reshape", inputs, attrs);
    EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, default_tolerance(code)));
  }

  // ---------------------------------------------------------------------------
  // M22(批4 T4,§1.6 决议点F)一致性用例专用:单输出图算子的通用 Expect*
  // (REUSE-002,泛化上面 ExpectSigmoidGraphMatchesCpu/ExpectReshapeGraphMatchesCpu
  // 的骨架为"任意 op/任意已构造好的 inputs/attrs",供
  // tanh/rsqrt/transpose/concat/slice/gather/gather_grad_internal 共用,免逐
  // 算子重复 H2D/图编译/D2H 骨架)。
  // ---------------------------------------------------------------------------
  void ExpectGraphOpMatchesCpu(std::string_view op, DTypeCode code,
                               const std::vector<Tensor>& inputs, const AttrMap& attrs) {
    const Tensor cuda_out = RunOpGraphOnCuda(op, inputs, attrs, CompileOptions{});
    const Tensor cpu_out = RunOpGraphOnCpu(op, inputs, attrs);
    EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, default_tolerance(code)));
  }

  // M22 T4:softmax/layer_norm 的图编译训练路径(前向 + 反向微图)专用驱动
  // ——device 由调用方传入的 forward graph 内 TensorType.device 参数化(该图
  // 须已用 MakeDeviceType(..., device) 构造 graph inputs);经
  // frame::compiler::build_backward_graph 生成训练图(输出=[loss,
  // grad(wrt_input_indices[0]), ...])+ frame::runtime::compile 编译执行。
  // cuda 版:host_inputs 逐个 CopyToDevice 后跑图,输出逐个 CopyToHost 搬回。
  std::vector<Tensor> RunTrainingGraphOnCuda(const Graph& forward, int32_t loss_output_index,
                                             const std::vector<int32_t>& wrt_input_indices,
                                             const std::vector<Tensor>& host_inputs) {
    const Result<Graph> training_graph =
        frame::compiler::build_backward_graph(forward, loss_output_index, wrt_input_indices);
    EXPECT_TRUE(training_graph.is_ok()) << training_graph.status().message();

    const Result<std::shared_ptr<Executable>> executable =
        frame::runtime::compile(training_graph.value(), frame::kCudaBackendName, CompileOptions{});
    EXPECT_TRUE(executable.is_ok()) << executable.status().message();

    std::vector<Tensor> device_inputs;
    device_inputs.reserve(host_inputs.size());
    for (const Tensor& host : host_inputs) device_inputs.push_back(CopyToDevice(host));

    const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
        *executable.value(), frame::kCudaBackendName, device_inputs);
    EXPECT_TRUE(outputs.is_ok()) << outputs.status().message();

    std::vector<Tensor> host_outputs;
    host_outputs.reserve(outputs.value().size());
    for (const Tensor& device_out : outputs.value()) host_outputs.push_back(CopyToHost(device_out));
    EXPECT_TRUE(stream_->synchronize().is_ok());
    return host_outputs;
  }

  // M22 T4:cpu 侧对照(与 RunTrainingGraphOnCuda 同参数形态);cpu Device 即
  // 主机内存,host_inputs 无需搬运。
  std::vector<Tensor> RunTrainingGraphOnCpu(const Graph& forward, int32_t loss_output_index,
                                            const std::vector<int32_t>& wrt_input_indices,
                                            const std::vector<Tensor>& host_inputs) {
    const Result<Graph> training_graph =
        frame::compiler::build_backward_graph(forward, loss_output_index, wrt_input_indices);
    EXPECT_TRUE(training_graph.is_ok()) << training_graph.status().message();

    const Result<std::shared_ptr<Executable>> executable =
        frame::runtime::compile(training_graph.value(), frame::kCpuBackendName, CompileOptions{});
    EXPECT_TRUE(executable.is_ok()) << executable.status().message();

    const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
        *executable.value(), frame::kCpuBackendName, host_inputs);
    EXPECT_TRUE(outputs.is_ok()) << outputs.status().message();
    return outputs.value();
  }

  // M23 T4:cuda 版"解析梯度 ≡ 数值微分"(BUILD-011 容差)。与
  // tests/cpp/ops/gradient_check_test_helpers.h::CheckGradientMatchesNumeric
  // 同一中心差分手法(ARCH-066),但 loss_fn 经本类既有的
  // RunTrainingGraphOnCuda 在 cuda 后端真实执行,而非该头文件固定驱动的 cpu
  // 后端——用于验证 rfft/irfft 梯度微图不仅"cuda 与 cpu 结果一致"(较弱断言,
  // 见 Softmax/LayerNormTrainingGraphBackwardMicrographRunsOnCuda 两例),而是
  // 在 cuda 真实执行路径上本身就数值收敛到解析梯度(REUSE-002:管线结构相同
  // 但驱动的 executable 后端不同,该头文件是 cpu-only 实现,不可直接复用)。x
  // 须为 host 内存可直接解引用的 fp32 张量(numeric_gradient 前提),每次扰动
  // 后经 RunTrainingGraphOnCuda 内部的 CopyToDevice 重新搬运。
  ::testing::AssertionResult CheckGradientMatchesNumericOnCuda(
      const Graph& forward, int32_t loss_output_index, const std::vector<int32_t>& wrt_indices,
      size_t wrt_position, Tensor& x, double h) {
    auto loss_fn = [&](const Tensor& current_x) -> Result<double> {
      const std::vector<Tensor> outputs =
          RunTrainingGraphOnCuda(forward, loss_output_index, wrt_indices, {current_x});
      if (outputs.empty()) {
        return Status::make(ErrorCode::kInternal,
                            "RunTrainingGraphOnCuda produced no outputs for loss readback");
      }
      return static_cast<double>(*static_cast<const float*>(outputs[0].raw_data()));
    };

    const Result<std::vector<double>> numeric = frame::testing::numeric_gradient(loss_fn, x, h);
    if (!numeric.is_ok()) {
      return ::testing::AssertionFailure()
             << "numeric_gradient failed: " << numeric.status().message();
    }

    const std::vector<Tensor> outputs =
        RunTrainingGraphOnCuda(forward, loss_output_index, wrt_indices, {x});
    if (outputs.size() <= 1 + wrt_position) {
      return ::testing::AssertionFailure()
             << "RunTrainingGraphOnCuda produced insufficient outputs for analytic gradient "
                "readback";
    }
    const Tensor& analytic = outputs[1 + wrt_position];

    const Result<Tensor> numeric_tensor_result =
        Tensor::empty(analytic.shape(), DType::of<float>(), cpu_device_, *cpu_allocator_);
    if (!numeric_tensor_result.is_ok()) {
      return ::testing::AssertionFailure() << numeric_tensor_result.status().message();
    }
    Tensor numeric_tensor = numeric_tensor_result.value();
    float* numeric_data = numeric_tensor.data<float>();
    for (size_t i = 0; i < numeric.value().size(); ++i) {
      numeric_data[i] = static_cast<float>(numeric.value()[i]);
    }

    return tensor_all_close(analytic, numeric_tensor, relaxed_tolerance(DTypeCode::kFloat32));
  }

  Backend* cuda_backend_ = nullptr;
  Backend* cpu_backend_ = nullptr;
  Device device_{};
  Device cpu_device_{};
  frame::hal::Allocator* cuda_allocator_ = nullptr;
  frame::hal::Allocator* cpu_allocator_ = nullptr;
  std::unique_ptr<frame::hal::Stream> stream_;
};

// =============================================================================
// 0. 冒烟 S1-S4。
// =============================================================================

TEST_F(CudaBackendTest, S1DeviceCountIsAtLeastOne) {
  const Result<int32_t> count = cuda_backend_->device_count();
  ASSERT_TRUE(count.is_ok()) << count.status().message();
  EXPECT_GE(count.value(), 1);
}

TEST_F(CudaBackendTest, S2AllocatorAllocateDeallocateRoundTripAtDefaultAlignment) {
  constexpr size_t kBytes = 256;
  const Result<void*> allocated = cuda_allocator_->allocate(kBytes, frame::kDefaultAlignment);
  ASSERT_TRUE(allocated.is_ok()) << allocated.status().message();
  void* ptr = allocated.value();
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % frame::kDefaultAlignment, 0u);

  // CudaAllocator 经 cudaMalloc 分配(纯设备驻留内存,不可 host 端直接
  // 解引用),往返经 cuda_backend_->copy(H2D 写入、D2H 读回)中转,与
  // tests/cpp/hal_conformance/test_hal_conformance.cpp::AllocatorRoundTripAtAlignment64
  // 泛化后的通用断言同一手法。
  std::vector<unsigned char> expected(kBytes, 0x5A);
  const Status h2d =
      cuda_backend_->copy(ptr, device_, expected.data(), cpu_device_, kBytes, stream_.get());
  ASSERT_TRUE(h2d.is_ok()) << h2d.message();
  std::vector<unsigned char> readback(kBytes, 0);
  const Status d2h =
      cuda_backend_->copy(readback.data(), cpu_device_, ptr, device_, kBytes, stream_.get());
  ASSERT_TRUE(d2h.is_ok()) << d2h.message();
  const Status sync = stream_->synchronize();
  ASSERT_TRUE(sync.is_ok()) << sync.message();
  EXPECT_EQ(std::memcmp(readback.data(), expected.data(), kBytes), 0);

  cuda_allocator_->deallocate(ptr);
}

TEST_F(CudaBackendTest, S3HostToDeviceToHostRoundTripPreservesBytesAfterSynchronize) {
  constexpr size_t kBytes = 128;
  std::vector<unsigned char> pattern(kBytes);
  for (size_t i = 0; i < kBytes; ++i) pattern[i] = static_cast<unsigned char>(i * 11 + 3);

  Result<Tensor> host_src_result =
      Tensor::empty(Shape({static_cast<int64_t>(kBytes)}), DType::of<std::uint8_t>(), cpu_device_,
                    *cpu_allocator_);
  ASSERT_TRUE(host_src_result.is_ok()) << host_src_result.status().message();
  // 按值持有(浅句柄,复制廉价):后续作为拷贝目的端需要非 const raw_data()。
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  Tensor host_src = host_src_result.value();
  std::memcpy(host_src.raw_data(), pattern.data(), kBytes);

  Result<Tensor> device_buf_result = Tensor::empty(
      Shape({static_cast<int64_t>(kBytes)}), DType::of<std::uint8_t>(), device_, *cuda_allocator_);
  ASSERT_TRUE(device_buf_result.is_ok()) << device_buf_result.status().message();
  // 按值持有(浅句柄,复制廉价):后续作为拷贝目的端需要非 const raw_data()。
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  Tensor device_buf = device_buf_result.value();

  const Status h2d = cuda_backend_->copy(device_buf.raw_data(), device_, host_src.raw_data(),
                                         cpu_device_, kBytes, stream_.get());
  ASSERT_TRUE(h2d.is_ok()) << h2d.message();

  Result<Tensor> host_dst_result =
      Tensor::empty(Shape({static_cast<int64_t>(kBytes)}), DType::of<std::uint8_t>(), cpu_device_,
                    *cpu_allocator_);
  ASSERT_TRUE(host_dst_result.is_ok()) << host_dst_result.status().message();
  // 按值持有(浅句柄,复制廉价):后续作为拷贝目的端需要非 const raw_data()。
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  Tensor host_dst = host_dst_result.value();

  const Status d2h = cuda_backend_->copy(host_dst.raw_data(), cpu_device_, device_buf.raw_data(),
                                         device_, kBytes, stream_.get());
  ASSERT_TRUE(d2h.is_ok()) << d2h.message();

  // backend-hal.md 2.1:copy 是"异步拷贝",两次 copy 排入 stream_ 后必须先
  // synchronize() 再读取结果——呼应任务0对 hal_conformance 同类缺口的修复
  // (CopyHostToHostRoundTripPreservesBytes 此前漏了这一步),本用例从一开始
  // 就补齐,不留同一坑。
  const Status sync = stream_->synchronize();
  ASSERT_TRUE(sync.is_ok()) << sync.message();

  EXPECT_EQ(std::memcmp(host_dst.raw_data(), pattern.data(), kBytes), 0);
}

TEST_F(CudaBackendTest, S4EventEstablishesCrossStreamDependencyForAsyncKernelResult) {
  Result<std::unique_ptr<frame::hal::Stream>> stream_a_result =
      cuda_backend_->create_stream(device_);
  ASSERT_TRUE(stream_a_result.is_ok()) << stream_a_result.status().message();
  std::unique_ptr<frame::hal::Stream> stream_a = std::move(stream_a_result.value());

  Result<std::unique_ptr<frame::hal::Stream>> stream_b_result =
      cuda_backend_->create_stream(device_);
  ASSERT_TRUE(stream_b_result.is_ok()) << stream_b_result.status().message();
  std::unique_ptr<frame::hal::Stream> stream_b = std::move(stream_b_result.value());

  Result<std::unique_ptr<frame::hal::Event>> event_result = cuda_backend_->create_event(device_);
  ASSERT_TRUE(event_result.is_ok()) << event_result.status().message();
  std::unique_ptr<frame::hal::Event> event = std::move(event_result.value());

  const Tensor lhs_host = MakeHostTensor1D<float>({1.0F, 2.0F, 3.0F, 4.0F});
  const Tensor rhs_host = MakeHostTensor1D<float>({10.0F, 20.0F, 30.0F, 40.0F});

  Result<Tensor> lhs_dev_result =
      Tensor::empty(Shape({4}), DType::of<float>(), device_, *cuda_allocator_);
  ASSERT_TRUE(lhs_dev_result.is_ok());
  Tensor lhs_dev = lhs_dev_result.value();
  Result<Tensor> rhs_dev_result =
      Tensor::empty(Shape({4}), DType::of<float>(), device_, *cuda_allocator_);
  ASSERT_TRUE(rhs_dev_result.is_ok());
  Tensor rhs_dev = rhs_dev_result.value();
  Result<Tensor> out_dev_result =
      Tensor::empty(Shape({4}), DType::of<float>(), device_, *cuda_allocator_);
  ASSERT_TRUE(out_dev_result.is_ok());
  Tensor out_dev = out_dev_result.value();

  ASSERT_TRUE(cuda_backend_
                  ->copy(lhs_dev.raw_data(), device_, lhs_host.raw_data(), cpu_device_,
                         4 * sizeof(float), stream_a.get())
                  .is_ok());
  ASSERT_TRUE(cuda_backend_
                  ->copy(rhs_dev.raw_data(), device_, rhs_host.raw_data(), cpu_device_,
                         4 * sizeof(float), stream_a.get())
                  .is_ok());

  KernelInvocation invocation;
  invocation.op = "add";
  std::vector<Tensor> add_inputs{lhs_dev, rhs_dev};
  std::vector<Tensor> add_outputs{out_dev};
  invocation.inputs = add_inputs;
  invocation.outputs = add_outputs;
  invocation.device = device_;
  const Status launch_status = cuda_backend_->launch(invocation, stream_a.get());
  ASSERT_TRUE(launch_status.is_ok()) << launch_status.message();

  // stream_a 上 record 事件(add kernel 排队之后的位置);stream_b wait 该
  // 事件建立跨流依赖,再在 stream_b 上把 out_dev D2H 拷贝——验证的是"显式建立
  // 依赖后数值正确"这一 HAL 契约(backend-hal.md 2.2/2.3),不依赖"kernel 恰好
  // 已经跑完"的运气。
  ASSERT_TRUE(stream_a->record(*event).is_ok());
  ASSERT_TRUE(stream_b->wait(*event).is_ok());

  Result<Tensor> out_host_result =
      Tensor::empty(Shape({4}), DType::of<float>(), cpu_device_, *cpu_allocator_);
  ASSERT_TRUE(out_host_result.is_ok());
  // 按值持有(浅句柄,复制廉价):后续作为拷贝目的端需要非 const raw_data()。
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  Tensor out_host = out_host_result.value();
  const Status d2h = cuda_backend_->copy(out_host.raw_data(), cpu_device_, out_dev.raw_data(),
                                         device_, 4 * sizeof(float), stream_b.get());
  ASSERT_TRUE(d2h.is_ok()) << d2h.message();
  ASSERT_TRUE(stream_b->synchronize().is_ok());

  const Tensor expected = MakeHostTensor1D<float>({11.0F, 22.0F, 33.0F, 44.0F});
  EXPECT_TRUE(tensor_all_close(out_host, expected, default_tolerance(DTypeCode::kFloat32)));

  EXPECT_TRUE(event->query());  // 全部相关工作已完成,query() 应为 true。
  EXPECT_TRUE(stream_a->synchronize().is_ok());
}

// =============================================================================
// 1. kernel 数值:add/mul/relu/square,fp32+fp16+bf16 各一例。
// =============================================================================

TEST_F(CudaBackendTest, AddFloat32MatchesCpuReference) {
  ExpectBinaryElementwiseMatchesCpu<float>("add", DTypeCode::kFloat32, {1.5F, -2.0F, 0.0F, 3.25F},
                                           {2.25F, 4.5F, 3.0F, -1.0F});
}

TEST_F(CudaBackendTest, AddFloat16MatchesCpuReference) {
  ExpectBinaryElementwiseMatchesCpu<float16_t>("add", DTypeCode::kFloat16,
                                               ToFloat16({1.5F, -2.0F, 0.5F, 3.0F}),
                                               ToFloat16({2.25F, 4.5F, -1.0F, 0.75F}));
}

TEST_F(CudaBackendTest, AddBFloat16MatchesCpuReference) {
  ExpectBinaryElementwiseMatchesCpu<bfloat16_t>("add", DTypeCode::kBFloat16,
                                                ToBFloat16({1.5F, -2.0F, 0.5F, 3.0F}),
                                                ToBFloat16({2.25F, 4.5F, -1.0F, 0.75F}));
}

TEST_F(CudaBackendTest, MulFloat32MatchesCpuReference) {
  ExpectBinaryElementwiseMatchesCpu<float>("mul", DTypeCode::kFloat32, {1.5F, -2.0F, 0.0F, 3.25F},
                                           {2.25F, 4.5F, 3.0F, -1.0F});
}

TEST_F(CudaBackendTest, MulFloat16MatchesCpuReference) {
  ExpectBinaryElementwiseMatchesCpu<float16_t>("mul", DTypeCode::kFloat16,
                                               ToFloat16({1.5F, -2.0F, 0.5F, 3.0F}),
                                               ToFloat16({2.25F, 4.5F, -1.0F, 0.75F}));
}

TEST_F(CudaBackendTest, MulBFloat16MatchesCpuReference) {
  ExpectBinaryElementwiseMatchesCpu<bfloat16_t>("mul", DTypeCode::kBFloat16,
                                                ToBFloat16({1.5F, -2.0F, 0.5F, 3.0F}),
                                                ToBFloat16({2.25F, 4.5F, -1.0F, 0.75F}));
}

TEST_F(CudaBackendTest, ReluFloat32MatchesCpuReference) {
  ExpectUnaryElementwiseMatchesCpu<float>("relu", DTypeCode::kFloat32, {-1.5F, 2.0F, -0.5F, 3.25F});
}

TEST_F(CudaBackendTest, ReluFloat16MatchesCpuReference) {
  ExpectUnaryElementwiseMatchesCpu<float16_t>("relu", DTypeCode::kFloat16,
                                              ToFloat16({-1.5F, 2.0F, -0.5F, 3.25F}));
}

TEST_F(CudaBackendTest, ReluBFloat16MatchesCpuReference) {
  ExpectUnaryElementwiseMatchesCpu<bfloat16_t>("relu", DTypeCode::kBFloat16,
                                               ToBFloat16({-1.5F, 2.0F, -0.5F, 3.25F}));
}

TEST_F(CudaBackendTest, SquareFloat32MatchesCpuReference) {
  ExpectUnaryElementwiseMatchesCpu<float>("square", DTypeCode::kFloat32,
                                          {-1.5F, 2.0F, -0.5F, 3.25F});
}

TEST_F(CudaBackendTest, SquareFloat16MatchesCpuReference) {
  ExpectUnaryElementwiseMatchesCpu<float16_t>("square", DTypeCode::kFloat16,
                                              ToFloat16({-1.5F, 2.0F, -0.5F, 3.25F}));
}

TEST_F(CudaBackendTest, SquareBFloat16MatchesCpuReference) {
  ExpectUnaryElementwiseMatchesCpu<bfloat16_t>("square", DTypeCode::kBFloat16,
                                               ToBFloat16({-1.5F, 2.0F, -0.5F, 3.25F}));
}

// =============================================================================
// 1续. kernel 数值:elementwise 向量化访存边界(M19 Task 5 守护用例)。
// src/backends/cuda/kernels/elementwise.cu 即将新增"16B 对齐 + numel%4==0 走
// float4/__half2/__nv_bfloat162 宽访存,否则标量回退"的运行期分支(M19 计划,
// 本次改动前 elementwise.cu 仍是纯标量实现)。numel=1024(可被4整除)驱动宽
// 访存路径,numel=1023(不可被4整除)驱动标量回退路径——本套用例在当前标量
// 实现下即应 PASS,向量化改造落地后仍必须 PASS,用于守护"宽访存不破坏数值"
// 与"非4整除输入正确回退到标量路径"两件事,防止向量化改造引入静默数值错误。
// device 侧张量经 Tensor::empty -> cuda_allocator_ 分配,恒以
// kDefaultAlignment=64 字节对齐(include/frame/core/storage.h),远超 16B 门槛,
// 故两档用例的路径分野只取决于 numel%4,与"指针是否 16B 对齐"这一判据无关——
// 本框架下测试侧无法构造出 Tensor 持有的"不对齐"设备指针,该判据的反例不属
// 本文件覆盖范围。add 为二元代表、relu 为一元代表,输入固定序列(非随机)
// 覆盖正负值(relu 分支覆盖见 MakeVectorizationBoundaryReluInput 注释)。
// =============================================================================

TEST_F(CudaBackendTest, AddVectorizationBoundaryAlignedNumel1024Float32MatchesCpuReference) {
  ExpectAddVectorizationBoundaryMatchesCpu<float>(DTypeCode::kFloat32, 1024);
}

TEST_F(CudaBackendTest, AddVectorizationBoundaryScalarFallbackNumel1023Float32MatchesCpuReference) {
  ExpectAddVectorizationBoundaryMatchesCpu<float>(DTypeCode::kFloat32, 1023);
}

TEST_F(CudaBackendTest, AddVectorizationBoundaryAlignedNumel1024Float16MatchesCpuReference) {
  ExpectAddVectorizationBoundaryMatchesCpu<float16_t>(DTypeCode::kFloat16, 1024);
}

TEST_F(CudaBackendTest, AddVectorizationBoundaryScalarFallbackNumel1023Float16MatchesCpuReference) {
  ExpectAddVectorizationBoundaryMatchesCpu<float16_t>(DTypeCode::kFloat16, 1023);
}

TEST_F(CudaBackendTest, AddVectorizationBoundaryAlignedNumel1024BFloat16MatchesCpuReference) {
  ExpectAddVectorizationBoundaryMatchesCpu<bfloat16_t>(DTypeCode::kBFloat16, 1024);
}

TEST_F(CudaBackendTest,
       AddVectorizationBoundaryScalarFallbackNumel1023BFloat16MatchesCpuReference) {
  ExpectAddVectorizationBoundaryMatchesCpu<bfloat16_t>(DTypeCode::kBFloat16, 1023);
}

TEST_F(CudaBackendTest, ReluVectorizationBoundaryAlignedNumel1024Float32MatchesCpuReference) {
  ExpectReluVectorizationBoundaryMatchesCpu<float>(DTypeCode::kFloat32, 1024);
}

TEST_F(CudaBackendTest,
       ReluVectorizationBoundaryScalarFallbackNumel1023Float32MatchesCpuReference) {
  ExpectReluVectorizationBoundaryMatchesCpu<float>(DTypeCode::kFloat32, 1023);
}

TEST_F(CudaBackendTest, ReluVectorizationBoundaryAlignedNumel1024Float16MatchesCpuReference) {
  ExpectReluVectorizationBoundaryMatchesCpu<float16_t>(DTypeCode::kFloat16, 1024);
}

TEST_F(CudaBackendTest,
       ReluVectorizationBoundaryScalarFallbackNumel1023Float16MatchesCpuReference) {
  ExpectReluVectorizationBoundaryMatchesCpu<float16_t>(DTypeCode::kFloat16, 1023);
}

TEST_F(CudaBackendTest, ReluVectorizationBoundaryAlignedNumel1024BFloat16MatchesCpuReference) {
  ExpectReluVectorizationBoundaryMatchesCpu<bfloat16_t>(DTypeCode::kBFloat16, 1024);
}

TEST_F(CudaBackendTest,
       ReluVectorizationBoundaryScalarFallbackNumel1023BFloat16MatchesCpuReference) {
  ExpectReluVectorizationBoundaryMatchesCpu<bfloat16_t>(DTypeCode::kBFloat16, 1023);
}

// =============================================================================
// 1续. kernel 数值:sum(全归约 + 轴归约 + 大规模放宽一档)。
// =============================================================================

TEST_F(CudaBackendTest, SumFullReductionMatchesCpuReference) {
  const Tensor x =
      MakeHostTensorWithShape<float>(Shape({2, 3}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const std::unordered_map<std::string, AttrValue> attrs{{"axes", std::vector<int64_t>{}}};
  const Tensor cuda_out = RunOnCuda("sum", {x}, Shape(), DType::of<float>(), &attrs);
  const Tensor cpu_out = RunOnCpu("sum", {x}, Shape(), DType::of<float>(), &attrs);
  EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(CudaBackendTest, SumAxisReductionMatchesCpuReference) {
  const Tensor x =
      MakeHostTensorWithShape<float>(Shape({2, 3}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const std::unordered_map<std::string, AttrValue> attrs{{"axes", std::vector<int64_t>{1}}};
  const Tensor cuda_out = RunOnCuda("sum", {x}, Shape({2}), DType::of<float>(), &attrs);
  const Tensor cpu_out = RunOnCpu("sum", {x}, Shape({2}), DType::of<float>(), &attrs);
  EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(CudaBackendTest, SumLargeFullReductionAtBuildEleven2Pow20AccumulationsUsesRelaxedTolerance) {
  // BUILD-011:"单个输出元素的累加次数 >= 2^20 的 reduction/matmul 类用例"允许
  // 放宽一档。本用例单输出元素(全归约,rank-0)累加次数恰为 2^20=1048576,
  // 触发该条款,改用 relaxed_tolerance(fp32 放宽到 fp16 档)而非
  // default_tolerance;数值分布沿用
  // tests/cpp/ops/test_op_sum.cpp::Float32LargeReductionAtBuildEleven2Pow20AccumulationsUsesRelaxedTolerance
  // 同一构造手法(0.1F*(i%7+1)循环取值,非退化输入,真实触发可观测舍入误差
  // 累积)。
  constexpr int64_t kLargeReductionNumel = int64_t{1} << 20;
  std::vector<float> values(static_cast<size_t>(kLargeReductionNumel));
  for (int64_t i = 0; i < kLargeReductionNumel; ++i) {
    values[static_cast<size_t>(i)] = 0.1F * static_cast<float>(i % 7 + 1);
  }
  const Tensor x = MakeHostTensorWithShape<float>(Shape({kLargeReductionNumel}), values);
  const std::unordered_map<std::string, AttrValue> attrs{{"axes", std::vector<int64_t>{}}};
  const Tensor cuda_out = RunOnCuda("sum", {x}, Shape(), DType::of<float>(), &attrs);
  const Tensor cpu_out = RunOnCpu("sum", {x}, Shape(), DType::of<float>(), &attrs);
  EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, relaxed_tolerance(DTypeCode::kFloat32)));
}

// 全归约(axes 为空)驱动 src/backends/cuda/kernels/reduction.cu::
// run_full_reduction_sum 的非 float 分支(thrust::transform_iterator 升 float
// 累加 + 1 元素 float scratch + sum_finalize_kernel 转回,ADR-0010 核实结论
// ②):把该核实阶段的手工探针(2^16 个 fp16 元素求和,transform_iterator 路径
// 误差量级 1e-4 对比逐元素 half 累加误差量级 1e4)固化为回归测试,防止未来改动
// 悄悄退化为半精度累加。规模取 2^12=4096——足以体现累加语义(远超几元素的
// 退化用例),未达 BUILD-011 放宽阈值 2^20,用 default_tolerance;数值分布沿用
// SumLargeFullReductionAtBuildEleven2Pow20AccumulationsUsesRelaxedTolerance
// 同一套 0.1F*(i%7+1)循环取值手法(REUSE-002)。
TEST_F(CudaBackendTest, SumFloat16FullReductionMatchesCpuReference) {
  constexpr int64_t kNumel = int64_t{1} << 12;
  std::vector<float16_t> values(static_cast<size_t>(kNumel));
  for (int64_t i = 0; i < kNumel; ++i) {
    values[static_cast<size_t>(i)] = frame::float_to_float16(0.1F * static_cast<float>(i % 7 + 1));
  }
  const Tensor x = MakeHostTensorWithShape<float16_t>(Shape({kNumel}), values);
  const std::unordered_map<std::string, AttrValue> attrs{{"axes", std::vector<int64_t>{}}};
  const Tensor cuda_out = RunOnCuda("sum", {x}, Shape(), DType::of<float16_t>(), &attrs);
  const Tensor cpu_out = RunOnCpu("sum", {x}, Shape(), DType::of<float16_t>(), &attrs);
  EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, default_tolerance(DTypeCode::kFloat16)));
}

// 同上,bf16 子分支(UpcastToFloat<bfloat16_t> 经 __bfloat162float 升 float)。
TEST_F(CudaBackendTest, SumBFloat16FullReductionMatchesCpuReference) {
  constexpr int64_t kNumel = int64_t{1} << 12;
  std::vector<bfloat16_t> values(static_cast<size_t>(kNumel));
  for (int64_t i = 0; i < kNumel; ++i) {
    values[static_cast<size_t>(i)] = frame::float_to_bfloat16(0.1F * static_cast<float>(i % 7 + 1));
  }
  const Tensor x = MakeHostTensorWithShape<bfloat16_t>(Shape({kNumel}), values);
  const std::unordered_map<std::string, AttrValue> attrs{{"axes", std::vector<int64_t>{}}};
  const Tensor cuda_out = RunOnCuda("sum", {x}, Shape(), DType::of<bfloat16_t>(), &attrs);
  const Tensor cpu_out = RunOnCpu("sum", {x}, Shape(), DType::of<bfloat16_t>(), &attrs);
  EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, default_tolerance(DTypeCode::kBFloat16)));
}

// =============================================================================
// 1续. kernel 数值:matmul,三精度统一经 cublasLtMatmul(ADR-0019)。
// =============================================================================

TEST_F(CudaBackendTest, MatmulFloat32MatchesCpuReference) {
  const Tensor lhs =
      MakeHostTensorWithShape<float>(Shape({2, 3}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor rhs =
      MakeHostTensorWithShape<float>(Shape({3, 2}), {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});
  const Tensor cuda_out = RunOnCuda("matmul", {lhs, rhs}, Shape({2, 2}), DType::of<float>());
  const Tensor cpu_out = RunOnCpu("matmul", {lhs, rhs}, Shape({2, 2}), DType::of<float>());
  EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(CudaBackendTest, MatmulFloat16MatchesCpuReferenceViaCublasLtMatmul) {
  const Tensor lhs = MakeHostTensorWithShape<float16_t>(
      Shape({2, 3}), ToFloat16({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}));
  const Tensor rhs = MakeHostTensorWithShape<float16_t>(
      Shape({3, 2}), ToFloat16({7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F}));
  const Tensor cuda_out = RunOnCuda("matmul", {lhs, rhs}, Shape({2, 2}), DType::of<float16_t>());
  const Tensor cpu_out = RunOnCpu("matmul", {lhs, rhs}, Shape({2, 2}), DType::of<float16_t>());
  EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, default_tolerance(DTypeCode::kFloat16)));
}

TEST_F(CudaBackendTest, MatmulBFloat16MatchesCpuReferenceViaCublasLtMatmul) {
  const Tensor lhs = MakeHostTensorWithShape<bfloat16_t>(
      Shape({2, 3}), ToBFloat16({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}));
  const Tensor rhs = MakeHostTensorWithShape<bfloat16_t>(
      Shape({3, 2}), ToBFloat16({7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F}));
  const Tensor cuda_out = RunOnCuda("matmul", {lhs, rhs}, Shape({2, 2}), DType::of<bfloat16_t>());
  const Tensor cpu_out = RunOnCpu("matmul", {lhs, rhs}, Shape({2, 2}), DType::of<bfloat16_t>());
  EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, default_tolerance(DTypeCode::kBFloat16)));
}

// =============================================================================
// 1续. kernel 数值:matmul TF32(ADR-0019,BUILD-011 fp32(allow_tf32)档,M19
// Task 6)。上面三例(MatmulFloat32/Float16/BFloat16MatchesCpuReference...)经
// Backend::launch 直接调用 kernel(KernelInvocation 不携带 CompileOptions,
// src/backends/cuda/cuda_backend.cpp::launch 构造 KernelContext 时
// compile_options 恒为 nullptr——见 matmul_cuda_kernel 对 ctx.compile_options
// 判空的用法),该路径下 allow_tf32 恒不生效,不适用于本节;必须改走
// frame::runtime::compile + Executable::run(CudaExecutable::compile 把
// options 按值存入 options_,run() 期经 &options_ 下传 KernelContext,见
// src/backends/cuda/cuda_executable.h/.cpp 头注释),CompileOptions::
// allow_tf32=true 才会真正传导到 matmul_cuda_kernel 的 compute_type 选择
// (CUBLAS_COMPUTE_32F_FAST_TF32,src/backends/cuda/kernels/matmul.cpp)。cpu
// 参考实现恒为严格 fp32(ARCH-041),不受 allow_tf32 影响,故 cuda(TF32)与
// cpu(严格 fp32)比对须用本文件专用的 tf32_tolerance() 档(BUILD-011,防放松
// 洗白判据:本文件同时含字面 allow_tf32 = true)。
// =============================================================================

TEST_F(CudaBackendTest, MatmulFloat32AllowTf32MatchesCpuWithinTf32Tolerance) {
  constexpr int64_t kM = 64;
  constexpr int64_t kK = 64;
  constexpr int64_t kN = 64;
  const Graph cuda_graph = BuildMatmulOnlyGraph(device_, kM, kK, kN);
  const Graph cpu_graph = BuildMatmulOnlyGraph(cpu_device_, kM, kK, kN);
  const Tensor lhs_host =
      MakeHostTensorWithShape<float>(Shape({kM, kK}), MakeMatmulTf32Lhs(kM * kK));
  const Tensor rhs_host =
      MakeHostTensorWithShape<float>(Shape({kK, kN}), MakeMatmulTf32Rhs(kK * kN));

  CompileOptions allow_tf32_options;
  allow_tf32_options.allow_tf32 = true;
  const Tensor cuda_out =
      RunMatmulGraphOnCuda(cuda_graph, lhs_host, rhs_host, Shape({kM, kN}), allow_tf32_options);
  const Tensor cpu_out = RunMatmulGraphOnCpu(cpu_graph, lhs_host, rhs_host, Shape({kM, kN}));

  // cpu 参考恒严格 fp32(ARCH-041 不变),本档仅限 TF32 开启用例取用(BUILD-011
  // fp32(allow_tf32)档)。
  EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, tf32_tolerance()));
}

// ADR-0019 暂定容差(rtol 1e-3/atol 1e-4)的校准用例:K=512 比上例 K=64 更接近
// 真实模型的收缩维规模,借此观察 TF32 尾数截断经更长累加链后的实测偏差量级。
// 无论 PASS/FAIL 均打印实测最大相对/绝对偏差(ComputeMatmulTf32Deviation),
// 供维护者按 ADR-0019 回填流程判定容差终值;本用例本身不因超差而放宽容差、
// 删除或加 DISABLED_ 前缀。
TEST_F(CudaBackendTest, MatmulFloat32AllowTf32LargeKMatchesCpuWithinTf32Tolerance) {
  constexpr int64_t kM = 64;
  constexpr int64_t kK = 512;
  constexpr int64_t kN = 64;
  const Graph cuda_graph = BuildMatmulOnlyGraph(device_, kM, kK, kN);
  const Graph cpu_graph = BuildMatmulOnlyGraph(cpu_device_, kM, kK, kN);
  const Tensor lhs_host =
      MakeHostTensorWithShape<float>(Shape({kM, kK}), MakeMatmulTf32Lhs(kM * kK));
  const Tensor rhs_host =
      MakeHostTensorWithShape<float>(Shape({kK, kN}), MakeMatmulTf32Rhs(kK * kN));

  CompileOptions allow_tf32_options;
  allow_tf32_options.allow_tf32 = true;
  const Tensor cuda_out =
      RunMatmulGraphOnCuda(cuda_graph, lhs_host, rhs_host, Shape({kM, kN}), allow_tf32_options);
  const Tensor cpu_out = RunMatmulGraphOnCpu(cpu_graph, lhs_host, rhs_host, Shape({kM, kN}));

  const MatmulTf32Deviation deviation = ComputeMatmulTf32Deviation(cuda_out, cpu_out);
  std::cout << "[TF32 calibration] "
               "MatmulFloat32AllowTf32LargeKMatchesCpuWithinTf32Tolerance M="
            << kM << " K=" << kK << " N=" << kN << " max_abs_diff=" << deviation.max_abs_diff
            << " max_rel_diff=" << deviation.max_rel_diff
            << " (ADR-0019 finalized tolerance: rtol=1e-3 atol=1e-4)" << '\n';

  EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, tf32_tolerance()));
}

// 守护:同 K=512 图形状,默认 CompileOptions(allow_tf32=false)编译,与 cpu
// 用严格 fp32 档 default_tolerance 比对——防止 cublasLt 严格路径悄然变成 TF32
// 数学模式(matmul_cuda_kernel 的 use_tf32 判据须 ctx.compile_options 非空且
// allow_tf32 显式为 true 才成立,src/backends/cuda/kernels/matmul.cpp)。K=512
// 单输出元素累加次数为 512,远低于 BUILD-011"放宽一档"门槛 2^20(1048576),故
// 沿用严格 fp32 主档而非放宽档。
TEST_F(CudaBackendTest, MatmulFloat32StrictDefaultLargeKMatchesCpuWithinFp32Tolerance) {
  constexpr int64_t kM = 64;
  constexpr int64_t kK = 512;
  constexpr int64_t kN = 64;
  const Graph cuda_graph = BuildMatmulOnlyGraph(device_, kM, kK, kN);
  const Graph cpu_graph = BuildMatmulOnlyGraph(cpu_device_, kM, kK, kN);
  const Tensor lhs_host =
      MakeHostTensorWithShape<float>(Shape({kM, kK}), MakeMatmulTf32Lhs(kM * kK));
  const Tensor rhs_host =
      MakeHostTensorWithShape<float>(Shape({kK, kN}), MakeMatmulTf32Rhs(kK * kN));

  const Tensor cuda_out =
      RunMatmulGraphOnCuda(cuda_graph, lhs_host, rhs_host, Shape({kM, kN}), CompileOptions{});
  const Tensor cpu_out = RunMatmulGraphOnCpu(cpu_graph, lhs_host, rhs_host, Shape({kM, kN}));

  EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, default_tolerance(DTypeCode::kFloat32)));
}

// =============================================================================
// 1续. kernel 数值:constant 物化(0 输入,host staging -> cudaMemcpy H2D)。
// =============================================================================

TEST_F(CudaBackendTest, ConstantMaterializationMatchesCpuReference) {
  std::unordered_map<std::string, AttrValue> attrs;
  attrs["value"] = AttrValue{std::vector<double>{1.5, -2.25, 0.0, 4.75}};
  attrs["shape"] = AttrValue{Shape({4})};
  attrs["dtype"] = AttrValue{DType::of<float>()};

  const Tensor cuda_out = RunOnCuda(kConstantOpName, {}, Shape({4}), DType::of<float>(), &attrs);
  const Tensor cpu_out = RunOnCpu(kConstantOpName, {}, Shape({4}), DType::of<float>(), &attrs);
  EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, default_tolerance(DTypeCode::kFloat32)));
}

// =============================================================================
// 1续. kernel 数值:fused_elementwise_internal 组合调用(经含融合链的图,由
// runtime::compile 标准管线的 operator_fusion pass 产出真实融合节点)。
// =============================================================================

TEST_F(CudaBackendTest, FusedElementwiseChainAddThenReluMatchesCpuReference) {
  const Graph cuda_graph = BuildAddReluGraph(device_);
  const Graph cpu_graph = BuildAddReluGraph(cpu_device_);

  const Result<std::shared_ptr<Executable>> cuda_executable =
      frame::runtime::compile(cuda_graph, frame::kCudaBackendName, CompileOptions{});
  ASSERT_TRUE(cuda_executable.is_ok()) << cuda_executable.status().message();
  const Result<std::shared_ptr<Executable>> cpu_executable =
      frame::runtime::compile(cpu_graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(cpu_executable.is_ok()) << cpu_executable.status().message();

  // a+b 含负值部分和,确保 relu 实际发生截断(而非恒等透传)。
  const Tensor a_host = MakeHostTensor1D<float>({1.0F, 1.5F, 2.0F, 2.5F});
  const Tensor b_host = MakeHostTensor1D<float>({-3.0F, -2.5F, -2.0F, -1.5F});

  const Tensor a_dev = CopyToDevice(a_host);
  const Tensor b_dev = CopyToDevice(b_host);
  std::vector<Tensor> cuda_inputs{a_dev, b_dev};
  Result<Tensor> cuda_out_result =
      Tensor::empty(Shape({4}), DType::of<float>(), device_, *cuda_allocator_);
  ASSERT_TRUE(cuda_out_result.is_ok());
  std::vector<Tensor> cuda_outputs{cuda_out_result.value()};
  const Status cuda_run_status = cuda_executable.value()->run(cuda_inputs, cuda_outputs, *stream_);
  ASSERT_TRUE(cuda_run_status.is_ok()) << cuda_run_status.message();
  const Tensor cuda_result_host = CopyToHost(cuda_outputs[0]);
  ASSERT_TRUE(stream_->synchronize().is_ok());

  std::vector<Tensor> cpu_inputs{a_host, b_host};
  Result<Tensor> cpu_out_result =
      Tensor::empty(Shape({4}), DType::of<float>(), cpu_device_, *cpu_allocator_);
  ASSERT_TRUE(cpu_out_result.is_ok());
  std::vector<Tensor> cpu_outputs{cpu_out_result.value()};
  Result<std::unique_ptr<frame::hal::Stream>> cpu_stream_result =
      cpu_backend_->create_stream(cpu_device_);
  ASSERT_TRUE(cpu_stream_result.is_ok());
  const Status cpu_run_status =
      cpu_executable.value()->run(cpu_inputs, cpu_outputs, *cpu_stream_result.value());
  ASSERT_TRUE(cpu_run_status.is_ok()) << cpu_run_status.message();

  EXPECT_TRUE(
      tensor_all_close(cuda_result_host, cpu_outputs[0], default_tolerance(DTypeCode::kFloat32)));
}

// =============================================================================
// 2. 端到端:matmul->add->relu,device=cuda vs device=cpu;二次执行命中缓存。
// =============================================================================

TEST_F(CudaBackendTest, EndToEndMatmulAddReluMatchesCpuReferenceAndSecondCompileHitsCache) {
  const Graph cuda_graph = BuildMatmulAddReluGraph(device_);

  const Result<std::shared_ptr<Executable>> first =
      frame::runtime::compile(cuda_graph, frame::kCudaBackendName, CompileOptions{});
  ASSERT_TRUE(first.is_ok()) << first.status().message();
  // 同一 graph 对象二次调用 runtime::compile(compile() 形参为 const
  // ir::Graph&,不消费/不移动该图):缓存键仅取决于 backend 名 +
  // options.fingerprint() + dump_text(graph),与调用次数/对象身份无关,命中
  // 缓存不再重新走管线与 Backend::compile ——用"同一 shared_ptr"断言取代
  // FallbackChainTest 里 HostFakeBackend::compile_call_count 那套计数思路
  // (真实 cuda 后端不受测试代码控制、无法插桩计数,故改断言"两次结果一致 +
  // 返回同一 Executable 对象",是等价力度更弱但可行的观测面,取舍见本文件
  // 头注释第2条)。
  const Result<std::shared_ptr<Executable>> second =
      frame::runtime::compile(cuda_graph, frame::kCudaBackendName, CompileOptions{});
  ASSERT_TRUE(second.is_ok()) << second.status().message();
  EXPECT_EQ(second.value(), first.value());

  const Graph cpu_graph = BuildMatmulAddReluGraph(cpu_device_);
  const Result<std::shared_ptr<Executable>> cpu_executable =
      frame::runtime::compile(cpu_graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(cpu_executable.is_ok()) << cpu_executable.status().message();

  const Tensor x_host =
      MakeHostTensorWithShape<float>(Shape({2, 3}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor w_host = MakeHostTensorWithShape<float>(
      Shape({3, 4}),
      {0.1F, -0.2F, 0.3F, -0.4F, 0.5F, -0.6F, 0.7F, -0.8F, 0.9F, -1.0F, 1.1F, -1.2F});
  const Tensor bias_host = MakeHostTensorWithShape<float>(
      Shape({2, 4}), {0.25F, -0.5F, 0.75F, -1.0F, -0.25F, 0.5F, -0.75F, 1.0F});

  const Tensor x_dev = CopyToDevice(x_host);
  const Tensor w_dev = CopyToDevice(w_host);
  const Tensor bias_dev = CopyToDevice(bias_host);
  std::vector<Tensor> cuda_inputs{x_dev, w_dev, bias_dev};

  Result<Tensor> cuda_out1_result =
      Tensor::empty(Shape({2, 4}), DType::of<float>(), device_, *cuda_allocator_);
  ASSERT_TRUE(cuda_out1_result.is_ok());
  std::vector<Tensor> cuda_outputs_1{cuda_out1_result.value()};
  ASSERT_TRUE(first.value()->run(cuda_inputs, cuda_outputs_1, *stream_).is_ok());
  const Tensor cuda_result_1 = CopyToHost(cuda_outputs_1[0]);
  ASSERT_TRUE(stream_->synchronize().is_ok());

  Result<Tensor> cuda_out2_result =
      Tensor::empty(Shape({2, 4}), DType::of<float>(), device_, *cuda_allocator_);
  ASSERT_TRUE(cuda_out2_result.is_ok());
  std::vector<Tensor> cuda_outputs_2{cuda_out2_result.value()};
  ASSERT_TRUE(second.value()->run(cuda_inputs, cuda_outputs_2, *stream_).is_ok());
  const Tensor cuda_result_2 = CopyToHost(cuda_outputs_2[0]);
  ASSERT_TRUE(stream_->synchronize().is_ok());

  std::vector<Tensor> cpu_inputs{x_host, w_host, bias_host};
  Result<Tensor> cpu_out_result =
      Tensor::empty(Shape({2, 4}), DType::of<float>(), cpu_device_, *cpu_allocator_);
  ASSERT_TRUE(cpu_out_result.is_ok());
  std::vector<Tensor> cpu_outputs{cpu_out_result.value()};
  Result<std::unique_ptr<frame::hal::Stream>> cpu_stream_result =
      cpu_backend_->create_stream(cpu_device_);
  ASSERT_TRUE(cpu_stream_result.is_ok());
  ASSERT_TRUE(
      cpu_executable.value()->run(cpu_inputs, cpu_outputs, *cpu_stream_result.value()).is_ok());

  EXPECT_TRUE(
      tensor_all_close(cuda_result_1, cpu_outputs[0], default_tolerance(DTypeCode::kFloat32)));
  EXPECT_TRUE(
      tensor_all_close(cuda_result_2, cuda_result_1, default_tolerance(DTypeCode::kFloat32)));

  // 全程未发生任何回退决策(matmul/add/relu 融合后的 fused_elementwise_internal
  // 均有 cuda kernel):FallbackStats 对本图涉及的算子键零计数。
  EXPECT_EQ(FallbackStats::instance().count("matmul", frame::kCudaBackendName), 0);
  EXPECT_EQ(FallbackStats::instance().count("add", frame::kCudaBackendName), 0);
  EXPECT_EQ(FallbackStats::instance().count("relu", frame::kCudaBackendName), 0);
  EXPECT_EQ(FallbackStats::instance().count("fused_elementwise_internal", frame::kCudaBackendName),
            0);
}

// =============================================================================
// 3. 回退跨设备:add(cuda) -> 三明治cpu-only算子 -> relu(cuda),验证
// FallbackExecutable 内部 D2H/H2D 编排数值正确 + FallbackStats 仅对三明治算子
// 计数一次。
// =============================================================================

TEST_F(CudaBackendTest, CrossDeviceFallbackSandwichOrchestratesCopiesCorrectly) {
  Graph cuda_graph("test_cuda_backend_fallback_sandwich");
  Value* a = cuda_graph.add_graph_input(MakeFloat32Type({4}, device_)).value();
  Value* b = cuda_graph.add_graph_input(MakeFloat32Type({4}, device_)).value();
  Node* add_node = cuda_graph.create_node("add", {a, b}, {MakeFloat32Type({4}, device_)}).value();
  Node* sandwich_node = cuda_graph
                            .create_node(std::string(kCpuOnlySandwichOpName), {add_node->output(0)},
                                         {MakeFloat32Type({4}, device_)})
                            .value();
  Node* relu_node =
      cuda_graph.create_node("relu", {sandwich_node->output(0)}, {MakeFloat32Type({4}, device_)})
          .value();
  cuda_graph.mark_output(relu_node, 0);

  Graph cpu_graph("test_cuda_backend_fallback_sandwich_cpu_reference");
  Value* cpu_a = cpu_graph.add_graph_input(MakeFloat32Type({4}, cpu_device_)).value();
  Value* cpu_b = cpu_graph.add_graph_input(MakeFloat32Type({4}, cpu_device_)).value();
  Node* cpu_add_node =
      cpu_graph.create_node("add", {cpu_a, cpu_b}, {MakeFloat32Type({4}, cpu_device_)}).value();
  Node* cpu_sandwich_node =
      cpu_graph
          .create_node(std::string(kCpuOnlySandwichOpName), {cpu_add_node->output(0)},
                       {MakeFloat32Type({4}, cpu_device_)})
          .value();
  Node* cpu_relu_node =
      cpu_graph
          .create_node("relu", {cpu_sandwich_node->output(0)}, {MakeFloat32Type({4}, cpu_device_)})
          .value();
  cpu_graph.mark_output(cpu_relu_node, 0);

  const Result<std::shared_ptr<Executable>> cuda_result =
      frame::runtime::compile(cuda_graph, frame::kCudaBackendName, CompileOptions{});
  ASSERT_TRUE(cuda_result.is_ok()) << cuda_result.status().message();
  const Result<std::shared_ptr<Executable>> cpu_result =
      frame::runtime::compile(cpu_graph, frame::kCpuBackendName, CompileOptions{});
  ASSERT_TRUE(cpu_result.is_ok()) << cpu_result.status().message();

  const Tensor a_host = MakeHostTensor1D<float>({1.0F, -2.0F, 3.0F, 0.5F});
  const Tensor b_host = MakeHostTensor1D<float>({2.0F, 1.0F, -4.0F, 0.5F});

  const Tensor a_dev = CopyToDevice(a_host);
  const Tensor b_dev = CopyToDevice(b_host);
  std::vector<Tensor> cuda_inputs{a_dev, b_dev};
  Result<Tensor> cuda_out_result =
      Tensor::empty(Shape({4}), DType::of<float>(), device_, *cuda_allocator_);
  ASSERT_TRUE(cuda_out_result.is_ok());
  std::vector<Tensor> cuda_outputs{cuda_out_result.value()};
  const Status cuda_run_status = cuda_result.value()->run(cuda_inputs, cuda_outputs, *stream_);
  ASSERT_TRUE(cuda_run_status.is_ok()) << cuda_run_status.message();
  const Tensor cuda_result_host = CopyToHost(cuda_outputs[0]);
  ASSERT_TRUE(stream_->synchronize().is_ok());

  std::vector<Tensor> cpu_inputs{a_host, b_host};
  Result<Tensor> cpu_out_result =
      Tensor::empty(Shape({4}), DType::of<float>(), cpu_device_, *cpu_allocator_);
  ASSERT_TRUE(cpu_out_result.is_ok());
  std::vector<Tensor> cpu_outputs{cpu_out_result.value()};
  Result<std::unique_ptr<frame::hal::Stream>> cpu_stream_result =
      cpu_backend_->create_stream(cpu_device_);
  ASSERT_TRUE(cpu_stream_result.is_ok());
  const Status cpu_run_status =
      cpu_result.value()->run(cpu_inputs, cpu_outputs, *cpu_stream_result.value());
  ASSERT_TRUE(cpu_run_status.is_ok()) << cpu_run_status.message();

  EXPECT_TRUE(
      tensor_all_close(cuda_result_host, cpu_outputs[0], default_tolerance(DTypeCode::kFloat32)));

  // kCpuOnlySandwichOpName 在 "cuda" 后端全进程唯一使用(仅本文件本用例),
  // add/relu 经①(KernelRegistry::find(op,"cuda") 命中真实 cuda kernel)直接
  // eager launch、不计入 FallbackStats(见 fallback_executable.cpp
  // log_fallback 头注释:仅"需要②或③才能解决"才计一次)。
  EXPECT_EQ(FallbackStats::instance().count(kCpuOnlySandwichOpName, frame::kCudaBackendName), 1);
  EXPECT_EQ(FallbackStats::instance().count("add", frame::kCudaBackendName), 0);
  EXPECT_EQ(FallbackStats::instance().count("relu", frame::kCudaBackendName), 0);
}

// =============================================================================
// 4. device 校验负例:cuda Executable::run 喂 cpu 设备张量,报错含双方 backend
// 名(hal::validate_tensor_devices,M11 裁决修订3)。
// =============================================================================

TEST_F(CudaBackendTest, RunRejectsCpuTensorsWithErrorMessageContainingBothBackendNames) {
  Graph graph("test_cuda_backend_device_mismatch");
  Value* x = graph.add_graph_input(MakeFloat32Type({4}, device_)).value();
  Node* relu_node = graph.create_node("relu", {x}, {MakeFloat32Type({4}, device_)}).value();
  graph.mark_output(relu_node, 0);

  const Result<std::shared_ptr<Executable>> executable_result =
      frame::runtime::compile(graph, frame::kCudaBackendName, CompileOptions{});
  ASSERT_TRUE(executable_result.is_ok()) << executable_result.status().message();

  // 故意提供 cpu 设备张量(而非本图编译目标 cuda)驱动 device 独立校验。
  const Tensor cpu_input = MakeHostTensor1D<float>({1.0F, -1.0F, 2.0F, -2.0F});
  std::vector<Tensor> inputs{cpu_input};
  Result<Tensor> output_result =
      Tensor::empty(Shape({4}), DType::of<float>(), device_, *cuda_allocator_);
  ASSERT_TRUE(output_result.is_ok());
  std::vector<Tensor> outputs{output_result.value()};

  const Status status = executable_result.value()->run(inputs, outputs, *stream_);
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(status.message().find(std::string(frame::kCudaBackendName)), std::string::npos);
  EXPECT_NE(status.message().find(std::string(frame::kCpuBackendName)), std::string::npos);
}

// =============================================================================
// 5. M21 批3 T5:conv2d/conv2d 两个反向/max_pool2d/avg_pool2d 前反向/select/
//    sigmoid/reshape 一致性(cuda vs cpu,经 runtime::compile 图编译路径
//    执行,不走 eager Backend::launch 旁路,理由见 RunOpGraphOnCuda 头注释)。
//    几何 A(fp32,含 bias/groups/stride/padding 全覆盖):x[1,2,5,5] +
//    w[4,1,3,3](Cin_per_group=1,groups=2)+ bias[4],stride2 padding1 ->
//    out[1,4,3,3]。几何 B(fp16/bf16,较简单):x[1,1,4,4] + w[2,1,3,3] +
//    bias[2],stride1 padding0 groups1 -> out[1,2,2,2]。pool 系用例统一取
//    x[1,2,5,5],kernel3 stride2 padding1 -> out[1,2,3,3]。
// =============================================================================

TEST_F(CudaBackendTest, Conv2dFloat32WithBiasGroupsStridePaddingMatchesCpuReference) {
  const std::vector<float> bias_values = MakeM21PatternValues(4, 0.2F, 5);
  ExpectConv2dMatchesCpu<float>(DTypeCode::kFloat32, Shape({1, 2, 5, 5}),
                                MakeM21PatternValues(50, 0.05F, 9), Shape({4, 1, 3, 3}),
                                MakeM21PatternValues(36, 0.1F, 7), &bias_values,
                                std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 1},
                                /*groups=*/2);
}

TEST_F(CudaBackendTest, Conv2dFloat16MatchesCpuReference) {
  const std::vector<float> bias_values = MakeM21PatternValues(2, 0.3F, 3);
  ExpectConv2dMatchesCpu<float16_t>(DTypeCode::kFloat16, Shape({1, 1, 4, 4}),
                                    MakeM21PatternValues(16, 0.1F, 7), Shape({2, 1, 3, 3}),
                                    MakeM21PatternValues(18, 0.15F, 5), &bias_values,
                                    std::vector<int64_t>{1, 1}, std::vector<int64_t>{0, 0},
                                    /*groups=*/1);
}

TEST_F(CudaBackendTest, Conv2dBFloat16MatchesCpuReference) {
  const std::vector<float> bias_values = MakeM21PatternValues(2, 0.3F, 3);
  ExpectConv2dMatchesCpu<bfloat16_t>(DTypeCode::kBFloat16, Shape({1, 1, 4, 4}),
                                     MakeM21PatternValues(16, 0.1F, 7), Shape({2, 1, 3, 3}),
                                     MakeM21PatternValues(18, 0.15F, 5), &bias_values,
                                     std::vector<int64_t>{1, 1}, std::vector<int64_t>{0, 0},
                                     /*groups=*/1);
}

TEST_F(CudaBackendTest, Conv2dGradInputInternalFloat32MatchesCpuReference) {
  ExpectConv2dGradInputMatchesCpu<float>(
      DTypeCode::kFloat32, Shape({1, 2, 5, 5}), Shape({1, 4, 3, 3}),
      MakeM21PatternValues(36, 0.2F, 7), Shape({4, 1, 3, 3}), MakeM21PatternValues(36, 0.1F, 7),
      std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 1}, /*groups=*/2);
}

TEST_F(CudaBackendTest, Conv2dGradInputInternalFloat16MatchesCpuReference) {
  ExpectConv2dGradInputMatchesCpu<float16_t>(
      DTypeCode::kFloat16, Shape({1, 1, 4, 4}), Shape({1, 2, 2, 2}),
      MakeM21PatternValues(8, 0.2F, 5), Shape({2, 1, 3, 3}), MakeM21PatternValues(18, 0.1F, 5),
      std::vector<int64_t>{1, 1}, std::vector<int64_t>{0, 0}, /*groups=*/1);
}

TEST_F(CudaBackendTest, Conv2dGradInputInternalBFloat16MatchesCpuReference) {
  ExpectConv2dGradInputMatchesCpu<bfloat16_t>(
      DTypeCode::kBFloat16, Shape({1, 1, 4, 4}), Shape({1, 2, 2, 2}),
      MakeM21PatternValues(8, 0.2F, 5), Shape({2, 1, 3, 3}), MakeM21PatternValues(18, 0.1F, 5),
      std::vector<int64_t>{1, 1}, std::vector<int64_t>{0, 0}, /*groups=*/1);
}

TEST_F(CudaBackendTest, Conv2dGradFilterInternalFloat32MatchesCpuReference) {
  ExpectConv2dGradFilterMatchesCpu<float>(
      DTypeCode::kFloat32, Shape({1, 2, 5, 5}), MakeM21PatternValues(50, 0.05F, 9),
      Shape({1, 4, 3, 3}), MakeM21PatternValues(36, 0.2F, 7), Shape({4, 1, 3, 3}),
      std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 1}, /*groups=*/2);
}

TEST_F(CudaBackendTest, Conv2dGradFilterInternalFloat16MatchesCpuReference) {
  ExpectConv2dGradFilterMatchesCpu<float16_t>(
      DTypeCode::kFloat16, Shape({1, 1, 4, 4}), MakeM21PatternValues(16, 0.1F, 7),
      Shape({1, 2, 2, 2}), MakeM21PatternValues(8, 0.2F, 5), Shape({2, 1, 3, 3}),
      std::vector<int64_t>{1, 1}, std::vector<int64_t>{0, 0}, /*groups=*/1);
}

TEST_F(CudaBackendTest, Conv2dGradFilterInternalBFloat16MatchesCpuReference) {
  ExpectConv2dGradFilterMatchesCpu<bfloat16_t>(
      DTypeCode::kBFloat16, Shape({1, 1, 4, 4}), MakeM21PatternValues(16, 0.1F, 7),
      Shape({1, 2, 2, 2}), MakeM21PatternValues(8, 0.2F, 5), Shape({2, 1, 3, 3}),
      std::vector<int64_t>{1, 1}, std::vector<int64_t>{0, 0}, /*groups=*/1);
}

TEST_F(CudaBackendTest, MaxPool2dFloat32MatchesCpuReference) {
  ExpectPool2dForwardMatchesCpu<float>(
      "max_pool2d", DTypeCode::kFloat32, Shape({1, 2, 5, 5}), MakeM21PatternValues(50, 0.3F, 11),
      std::vector<int64_t>{3, 3}, std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 1});
}

TEST_F(CudaBackendTest, MaxPool2dFloat16MatchesCpuReference) {
  ExpectPool2dForwardMatchesCpu<float16_t>(
      "max_pool2d", DTypeCode::kFloat16, Shape({1, 2, 5, 5}), MakeM21PatternValues(50, 0.3F, 11),
      std::vector<int64_t>{3, 3}, std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 1});
}

TEST_F(CudaBackendTest, MaxPool2dBFloat16MatchesCpuReference) {
  ExpectPool2dForwardMatchesCpu<bfloat16_t>(
      "max_pool2d", DTypeCode::kBFloat16, Shape({1, 2, 5, 5}), MakeM21PatternValues(50, 0.3F, 11),
      std::vector<int64_t>{3, 3}, std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 1});
}

TEST_F(CudaBackendTest, AvgPool2dFloat32MatchesCpuReference) {
  ExpectPool2dForwardMatchesCpu<float>(
      "avg_pool2d", DTypeCode::kFloat32, Shape({1, 2, 5, 5}), MakeM21PatternValues(50, 0.3F, 11),
      std::vector<int64_t>{3, 3}, std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 1});
}

TEST_F(CudaBackendTest, AvgPool2dFloat16MatchesCpuReference) {
  ExpectPool2dForwardMatchesCpu<float16_t>(
      "avg_pool2d", DTypeCode::kFloat16, Shape({1, 2, 5, 5}), MakeM21PatternValues(50, 0.3F, 11),
      std::vector<int64_t>{3, 3}, std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 1});
}

TEST_F(CudaBackendTest, AvgPool2dBFloat16MatchesCpuReference) {
  ExpectPool2dForwardMatchesCpu<bfloat16_t>(
      "avg_pool2d", DTypeCode::kBFloat16, Shape({1, 2, 5, 5}), MakeM21PatternValues(50, 0.3F, 11),
      std::vector<int64_t>{3, 3}, std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 1});
}

TEST_F(CudaBackendTest, MaxPool2dGradInternalFloat32MatchesCpuReference) {
  ExpectMaxPool2dGradInternalMatchesCpu<float>(
      DTypeCode::kFloat32, Shape({1, 2, 5, 5}), Shape({1, 2, 3, 3}),
      MakeM21PatternValues(18, 0.4F, 6), MakeM21PatternValues(50, 0.3F, 11),
      std::vector<int64_t>{3, 3}, std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 1});
}

TEST_F(CudaBackendTest, MaxPool2dGradInternalFloat16MatchesCpuReference) {
  ExpectMaxPool2dGradInternalMatchesCpu<float16_t>(
      DTypeCode::kFloat16, Shape({1, 2, 5, 5}), Shape({1, 2, 3, 3}),
      MakeM21PatternValues(18, 0.4F, 6), MakeM21PatternValues(50, 0.3F, 11),
      std::vector<int64_t>{3, 3}, std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 1});
}

TEST_F(CudaBackendTest, MaxPool2dGradInternalBFloat16MatchesCpuReference) {
  ExpectMaxPool2dGradInternalMatchesCpu<bfloat16_t>(
      DTypeCode::kBFloat16, Shape({1, 2, 5, 5}), Shape({1, 2, 3, 3}),
      MakeM21PatternValues(18, 0.4F, 6), MakeM21PatternValues(50, 0.3F, 11),
      std::vector<int64_t>{3, 3}, std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 1});
}

TEST_F(CudaBackendTest, MaxPool2dSelectInternalFloat32MatchesCpuReference) {
  ExpectMaxPool2dSelectInternalMatchesCpu<float>(
      DTypeCode::kFloat32, Shape({1, 2, 5, 5}), MakeM21PatternValues(50, 0.5F, 13),
      MakeM21PatternValues(50, 0.3F, 11), std::vector<int64_t>{3, 3}, std::vector<int64_t>{2, 2},
      std::vector<int64_t>{1, 1});
}

TEST_F(CudaBackendTest, MaxPool2dSelectInternalFloat16MatchesCpuReference) {
  ExpectMaxPool2dSelectInternalMatchesCpu<float16_t>(
      DTypeCode::kFloat16, Shape({1, 2, 5, 5}), MakeM21PatternValues(50, 0.5F, 13),
      MakeM21PatternValues(50, 0.3F, 11), std::vector<int64_t>{3, 3}, std::vector<int64_t>{2, 2},
      std::vector<int64_t>{1, 1});
}

TEST_F(CudaBackendTest, MaxPool2dSelectInternalBFloat16MatchesCpuReference) {
  ExpectMaxPool2dSelectInternalMatchesCpu<bfloat16_t>(
      DTypeCode::kBFloat16, Shape({1, 2, 5, 5}), MakeM21PatternValues(50, 0.5F, 13),
      MakeM21PatternValues(50, 0.3F, 11), std::vector<int64_t>{3, 3}, std::vector<int64_t>{2, 2},
      std::vector<int64_t>{1, 1});
}

TEST_F(CudaBackendTest, AvgPool2dGradInternalFloat32MatchesCpuReference) {
  ExpectAvgPool2dGradInternalMatchesCpu<float>(
      DTypeCode::kFloat32, Shape({1, 2, 5, 5}), Shape({1, 2, 3, 3}),
      MakeM21PatternValues(18, 0.4F, 6), std::vector<int64_t>{3, 3}, std::vector<int64_t>{2, 2},
      std::vector<int64_t>{1, 1});
}

TEST_F(CudaBackendTest, AvgPool2dGradInternalFloat16MatchesCpuReference) {
  ExpectAvgPool2dGradInternalMatchesCpu<float16_t>(
      DTypeCode::kFloat16, Shape({1, 2, 5, 5}), Shape({1, 2, 3, 3}),
      MakeM21PatternValues(18, 0.4F, 6), std::vector<int64_t>{3, 3}, std::vector<int64_t>{2, 2},
      std::vector<int64_t>{1, 1});
}

TEST_F(CudaBackendTest, AvgPool2dGradInternalBFloat16MatchesCpuReference) {
  ExpectAvgPool2dGradInternalMatchesCpu<bfloat16_t>(
      DTypeCode::kBFloat16, Shape({1, 2, 5, 5}), Shape({1, 2, 3, 3}),
      MakeM21PatternValues(18, 0.4F, 6), std::vector<int64_t>{3, 3}, std::vector<int64_t>{2, 2},
      std::vector<int64_t>{1, 1});
}

TEST_F(CudaBackendTest, SigmoidFloat32GraphMatchesCpuReference) {
  ExpectSigmoidGraphMatchesCpu<float>(DTypeCode::kFloat32, Shape({2, 3, 4}),
                                      MakeM21PatternValues(24, 0.5F, 9));
}

TEST_F(CudaBackendTest, SigmoidFloat16GraphMatchesCpuReference) {
  ExpectSigmoidGraphMatchesCpu<float16_t>(DTypeCode::kFloat16, Shape({2, 3, 4}),
                                          MakeM21PatternValues(24, 0.5F, 9));
}

TEST_F(CudaBackendTest, SigmoidBFloat16GraphMatchesCpuReference) {
  ExpectSigmoidGraphMatchesCpu<bfloat16_t>(DTypeCode::kBFloat16, Shape({2, 3, 4}),
                                           MakeM21PatternValues(24, 0.5F, 9));
}

TEST_F(CudaBackendTest, ReshapeFloat32GraphMatchesCpuReference) {
  ExpectReshapeGraphMatchesCpu<float>(DTypeCode::kFloat32, Shape({2, 3, 4}),
                                      MakeM21PatternValues(24, 1.0F, 11), Shape({4, 6}));
}

TEST_F(CudaBackendTest, ReshapeFloat16GraphMatchesCpuReference) {
  ExpectReshapeGraphMatchesCpu<float16_t>(DTypeCode::kFloat16, Shape({2, 3, 4}),
                                          MakeM21PatternValues(24, 1.0F, 11), Shape({4, 6}));
}

TEST_F(CudaBackendTest, ReshapeBFloat16GraphMatchesCpuReference) {
  ExpectReshapeGraphMatchesCpu<bfloat16_t>(DTypeCode::kBFloat16, Shape({2, 3, 4}),
                                           MakeM21PatternValues(24, 1.0F, 11), Shape({4, 6}));
}

// =============================================================================
// 5续. conv2d TF32(ADR-0021 决策 4 沿用 ADR-0019 单开关):fp32(allow_tf32)档 +
// TF32 生效性实测(计划 T5 硬项)。两条用例均经图编译路径(RunOpGraphOnCuda
// 内部走 runtime::compile + run_with_allocated_outputs),CompileOptions::
// allow_tf32=true 才会真正传导到 conv2d_cuda_kernel 的 mathType 选择
// (CUDNN_DEFAULT_MATH,src/backends/cuda/kernels/conv.cpp)。
// =============================================================================

TEST_F(CudaBackendTest, Conv2dFloat32AllowTf32MatchesCpuWithinTf32Tolerance) {
  constexpr int64_t kN = 2;
  constexpr int64_t kCin = 16;
  constexpr int64_t kCout = 16;
  constexpr int64_t kH = 8;
  constexpr int64_t kW = 8;
  constexpr int64_t kKh = 3;
  constexpr int64_t kKw = 3;
  const Shape x_shape({kN, kCin, kH, kW});
  const Shape w_shape({kCout, kCin, kKh, kKw});
  const Tensor x = MakeHostTensorWithShape<float>(x_shape, MakeMatmulTf32Lhs(kN * kCin * kH * kW));
  const Tensor w =
      MakeHostTensorWithShape<float>(w_shape, MakeMatmulTf32Rhs(kCout * kCin * kKh * kKw));
  std::vector<Tensor> inputs{x, w};
  const AttrMap attrs{{"stride", std::vector<int64_t>{1, 1}},
                      {"padding", std::vector<int64_t>{1, 1}},
                      {"groups", int64_t{1}}};

  CompileOptions allow_tf32_options;
  allow_tf32_options.allow_tf32 = true;
  const Tensor cuda_out = RunOpGraphOnCuda("conv2d", inputs, attrs, allow_tf32_options);
  const Tensor cpu_out = RunOpGraphOnCpu("conv2d", inputs, attrs);

  // cpu 参考恒严格 fp32(ARCH-041 不变),本档仅限 TF32 开启用例取用
  // (BUILD-011 fp32(allow_tf32)档)。
  EXPECT_TRUE(tensor_all_close(cuda_out, cpu_out, tf32_tolerance()));
}

// TF32 生效性实测(计划 T5 硬项):同输入 fp32 严格(CompileOptions{}默认
// allow_tf32=false,conv2d_cuda_kernel 选 CUDNN_FMA_MATH)与 allow_tf32=true
// (选 CUDNN_DEFAULT_MATH)两次跑 cuda conv2d,逐位比较。判据:实测最大绝对
// 偏差 > 0 视为"确认生效"并打印偏差量级;若为 0(位级不可区分),不判用例
// 失败——cuDNN 是否真正切到 TensorOp 数学路径依赖启发式算法选择结果,与输入
// 规模/GPU 架构相关,不可由本仓强行断言必然发生,仅如实记录并回填 BE-000
// (见本次交付报告,docs/backends/cuda.md 第 8 章后续登记)。规模刻意取得比
// 上一用例更大(Cin=Cout=32、H=W=16),增大触发 TensorOp 算法路径的概率。
TEST_F(CudaBackendTest, Conv2dFloat32Tf32EffectivenessRecordsDeviationOrIndistinguishable) {
  constexpr int64_t kN = 2;
  constexpr int64_t kCin = 32;
  constexpr int64_t kCout = 32;
  constexpr int64_t kH = 16;
  constexpr int64_t kW = 16;
  constexpr int64_t kKh = 3;
  constexpr int64_t kKw = 3;
  const Shape x_shape({kN, kCin, kH, kW});
  const Shape w_shape({kCout, kCin, kKh, kKw});
  const Tensor x = MakeHostTensorWithShape<float>(x_shape, MakeMatmulTf32Lhs(kN * kCin * kH * kW));
  const Tensor w =
      MakeHostTensorWithShape<float>(w_shape, MakeMatmulTf32Rhs(kCout * kCin * kKh * kKw));
  std::vector<Tensor> inputs{x, w};
  const AttrMap attrs{{"stride", std::vector<int64_t>{1, 1}},
                      {"padding", std::vector<int64_t>{1, 1}},
                      {"groups", int64_t{1}}};

  const Tensor strict_out = RunOpGraphOnCuda("conv2d", inputs, attrs, CompileOptions{});
  CompileOptions allow_tf32_options;
  allow_tf32_options.allow_tf32 = true;
  const Tensor tf32_out = RunOpGraphOnCuda("conv2d", inputs, attrs, allow_tf32_options);

  const MatmulTf32Deviation deviation = ComputeMatmulTf32Deviation(tf32_out, strict_out);
  std::cout << "[TF32 effectiveness] conv2d N=" << kN << " Cin=" << kCin << " Cout=" << kCout
            << " H=" << kH << " W=" << kW << " KH=" << kKh << " KW=" << kKw
            << " max_abs_diff=" << deviation.max_abs_diff
            << " max_rel_diff=" << deviation.max_rel_diff
            << (deviation.max_abs_diff > 0.0
                    ? " (TF32 confirmed effective: strict CUDNN_FMA_MATH vs allow_tf32 "
                      "CUDNN_DEFAULT_MATH outputs differ)"
                    : " (bit-identical: CUDNN_DEFAULT_MATH indistinguishable from "
                      "CUDNN_FMA_MATH on this input/GPU; recorded as inconclusive, needs "
                      "BE-000 follow-up per M21 T5)")
            << '\n';

  // 两条路径均须产出与 cpu 参考数值合理接近的结果;本用例的核心断言是上面
  // 的诊断打印本身,不因偏差为 0 判失败(理由见头注释)。
  const Tensor cpu_out = RunOpGraphOnCpu("conv2d", inputs, attrs);
  EXPECT_TRUE(tensor_all_close(strict_out, cpu_out, default_tolerance(DTypeCode::kFloat32)));
  EXPECT_TRUE(tensor_all_close(tf32_out, cpu_out, tf32_tolerance()));
}

// =============================================================================
// 6. M22(批4 T4,§1.6 决议表):tanh/rsqrt/softmax/layer_norm/transpose/concat/
// slice/gather/gather_grad_internal 一致性用例 + gather 越界负例 +
// softmax/layer_norm 图编译训练路径(反向微图在 cuda 上跑通)。add 的
// int32/int64 扩容(§1.1 决议点A)一并补两例。
// =============================================================================

TEST_F(CudaBackendTest, AddInt32MatchesCpuReference) {
  ExpectBinaryElementwiseMatchesCpu<int32_t>("add", DTypeCode::kInt32,
                                             std::vector<int32_t>{1, -2, 3, 100000},
                                             std::vector<int32_t>{10, 20, -30, 5});
}

TEST_F(CudaBackendTest, AddInt64MatchesCpuReference) {
  ExpectBinaryElementwiseMatchesCpu<int64_t>("add", DTypeCode::kInt64,
                                             std::vector<int64_t>{1, -2, 3, 9000000000LL},
                                             std::vector<int64_t>{10, 20, -30, 5});
}

TEST_F(CudaBackendTest, TanhFloat32GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensor1D<float>(MakeM21PatternValues(24, 0.5F, 9));
  ExpectGraphOpMatchesCpu("tanh", DTypeCode::kFloat32, {x}, AttrMap{});
}

TEST_F(CudaBackendTest, TanhFloat16GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensor1D<float16_t>(ToFloat16(MakeM21PatternValues(24, 0.5F, 9)));
  ExpectGraphOpMatchesCpu("tanh", DTypeCode::kFloat16, {x}, AttrMap{});
}

TEST_F(CudaBackendTest, TanhBFloat16GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensor1D<bfloat16_t>(ToBFloat16(MakeM21PatternValues(24, 0.5F, 9)));
  ExpectGraphOpMatchesCpu("tanh", DTypeCode::kBFloat16, {x}, AttrMap{});
}

// rsqrt 定义域要求 x>0(1/sqrt(x));复用 M19 Task 6 素材 MakeMatmulTf32Lhs(恒
// 正,覆盖 [0.05, 0.35],REUSE-002)而非以 0 为中心的 MakeM21PatternValues。
TEST_F(CudaBackendTest, RsqrtFloat32GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensor1D<float>(MakeMatmulTf32Lhs(24));
  ExpectGraphOpMatchesCpu("rsqrt", DTypeCode::kFloat32, {x}, AttrMap{});
}

TEST_F(CudaBackendTest, RsqrtFloat16GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensor1D<float16_t>(ToFloat16(MakeMatmulTf32Lhs(24)));
  ExpectGraphOpMatchesCpu("rsqrt", DTypeCode::kFloat16, {x}, AttrMap{});
}

TEST_F(CudaBackendTest, RsqrtBFloat16GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensor1D<bfloat16_t>(ToBFloat16(MakeMatmulTf32Lhs(24)));
  ExpectGraphOpMatchesCpu("rsqrt", DTypeCode::kBFloat16, {x}, AttrMap{});
}

TEST_F(CudaBackendTest, SoftmaxFloat32GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensorWithShape<float>(Shape({3, 4}), MakeM21PatternValues(12, 0.4F, 9));
  ExpectGraphOpMatchesCpu("softmax", DTypeCode::kFloat32, {x}, AttrMap{});
}

TEST_F(CudaBackendTest, SoftmaxFloat16GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensorWithShape<float16_t>(Shape({3, 4}),
                                                      ToFloat16(MakeM21PatternValues(12, 0.4F, 9)));
  ExpectGraphOpMatchesCpu("softmax", DTypeCode::kFloat16, {x}, AttrMap{});
}

TEST_F(CudaBackendTest, SoftmaxBFloat16GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensorWithShape<bfloat16_t>(
      Shape({3, 4}), ToBFloat16(MakeM21PatternValues(12, 0.4F, 9)));
  ExpectGraphOpMatchesCpu("softmax", DTypeCode::kBFloat16, {x}, AttrMap{});
}

// layer_norm(x[N,D], gamma[D], beta[D]; eps) 一致性:3 输入 + eps 属性。
TEST_F(CudaBackendTest, LayerNormFloat32GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensorWithShape<float>(Shape({3, 4}), MakeM21PatternValues(12, 0.3F, 7));
  const Tensor gamma = MakeHostTensorWithShape<float>(Shape({4}), MakeM21PatternValues(4, 0.2F, 5));
  const Tensor beta = MakeHostTensorWithShape<float>(Shape({4}), MakeM21PatternValues(4, 0.1F, 3));
  const AttrMap attrs{{"eps", 1e-5}};
  ExpectGraphOpMatchesCpu("layer_norm", DTypeCode::kFloat32, {x, gamma, beta}, attrs);
}

TEST_F(CudaBackendTest, LayerNormFloat16GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensorWithShape<float16_t>(Shape({3, 4}),
                                                      ToFloat16(MakeM21PatternValues(12, 0.3F, 7)));
  const Tensor gamma =
      MakeHostTensorWithShape<float16_t>(Shape({4}), ToFloat16(MakeM21PatternValues(4, 0.2F, 5)));
  const Tensor beta =
      MakeHostTensorWithShape<float16_t>(Shape({4}), ToFloat16(MakeM21PatternValues(4, 0.1F, 3)));
  const AttrMap attrs{{"eps", 1e-5}};
  ExpectGraphOpMatchesCpu("layer_norm", DTypeCode::kFloat16, {x, gamma, beta}, attrs);
}

TEST_F(CudaBackendTest, LayerNormBFloat16GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensorWithShape<bfloat16_t>(
      Shape({3, 4}), ToBFloat16(MakeM21PatternValues(12, 0.3F, 7)));
  const Tensor gamma =
      MakeHostTensorWithShape<bfloat16_t>(Shape({4}), ToBFloat16(MakeM21PatternValues(4, 0.2F, 5)));
  const Tensor beta =
      MakeHostTensorWithShape<bfloat16_t>(Shape({4}), ToBFloat16(MakeM21PatternValues(4, 0.1F, 3)));
  const AttrMap attrs{{"eps", 1e-5}};
  ExpectGraphOpMatchesCpu("layer_norm", DTypeCode::kBFloat16, {x, gamma, beta}, attrs);
}

TEST_F(CudaBackendTest, HeavisideSurrogateFloat32GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensor1D<float>({-2.0F, -0.0F, 0.0F, 0.25F, 2.0F});
  const AttrMap attrs{{"alpha", 2.0}};
  ExpectGraphOpMatchesCpu("heaviside_surrogate", DTypeCode::kFloat32, {x}, attrs);
}

TEST_F(CudaBackendTest, HeavisideSurrogateFloat16GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensor1D<float16_t>(ToFloat16({-2.0F, -0.0F, 0.0F, 0.25F, 2.0F}));
  const AttrMap attrs{{"alpha", 2.0}};
  ExpectGraphOpMatchesCpu("heaviside_surrogate", DTypeCode::kFloat16, {x}, attrs);
}

TEST_F(CudaBackendTest, HeavisideSurrogateBFloat16GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensor1D<bfloat16_t>(ToBFloat16({-2.0F, -0.0F, 0.0F, 0.25F, 2.0F}));
  const AttrMap attrs{{"alpha", 2.0}};
  ExpectGraphOpMatchesCpu("heaviside_surrogate", DTypeCode::kBFloat16, {x}, attrs);
}

TEST_F(CudaBackendTest, HeavisideSurrogateEmptyTensorMatchesCpuReference) {
  const Tensor x = MakeHostTensorWithShape<float>(Shape({0}), {});
  const AttrMap attrs{{"alpha", 2.0}};
  ExpectGraphOpMatchesCpu("heaviside_surrogate", DTypeCode::kFloat32, {x}, attrs);
}

TEST_F(CudaBackendTest, HeavisideSurrogateEmptyGradientMicrographRunsOnCuda) {
  const Shape shape({0});
  const AttrMap attrs{{"alpha", 2.0}};
  const std::vector<frame::ir::TensorType> forward_types{
      MakeDeviceType(DType::of<float>(), shape, device_)};
  const Tensor empty = MakeHostTensorWithShape<float>(shape, {});
  const std::vector<Tensor> outputs = RunGradientMicrographOnCuda(
      "heaviside_surrogate", forward_types, &attrs, {empty, empty, empty});
  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_EQ(outputs[0].shape(), shape);
  EXPECT_EQ(outputs[0].raw_data(), nullptr);
}

TEST_F(CudaBackendTest, EmptyAddAndMulGraphsMatchCpuReference) {
  const Tensor empty = MakeHostTensorWithShape<float>(Shape({0, 3}), {});
  ExpectGraphOpMatchesCpu("add", DTypeCode::kFloat32, {empty, empty}, AttrMap{});
  ExpectGraphOpMatchesCpu("mul", DTypeCode::kFloat32, {empty, empty}, AttrMap{});
}

// selective_scan 五输入沿最后一轴扫描;rank-3 输入同时覆盖多条前导序列。
TEST_F(CudaBackendTest, SelectiveScanFloat32GraphMatchesCpuReference) {
  const Shape shape({2, 2, 3});
  const Tensor x = MakeHostTensorWithShape<float>(shape, MakeM21PatternValues(12, 0.3F, 11));
  const Tensor a = MakeHostTensorWithShape<float>(shape, MakeM21PatternValues(12, 0.05F, 7));
  const Tensor b = MakeHostTensorWithShape<float>(shape, MakeM21PatternValues(12, 0.2F, 9));
  const Tensor c = MakeHostTensorWithShape<float>(shape, MakeM21PatternValues(12, 0.15F, 5));
  const Tensor d = MakeHostTensorWithShape<float>(shape, MakeM21PatternValues(12, 0.1F, 13));
  ExpectGraphOpMatchesCpu("selective_scan", DTypeCode::kFloat32, {x, a, b, c, d}, AttrMap{});
}

TEST_F(CudaBackendTest, SelectiveScanFloat16GraphMatchesCpuReference) {
  const Shape shape({2, 2, 3});
  const Tensor x =
      MakeHostTensorWithShape<float16_t>(shape, ToFloat16(MakeM21PatternValues(12, 0.3F, 11)));
  const Tensor a =
      MakeHostTensorWithShape<float16_t>(shape, ToFloat16(MakeM21PatternValues(12, 0.05F, 7)));
  const Tensor b =
      MakeHostTensorWithShape<float16_t>(shape, ToFloat16(MakeM21PatternValues(12, 0.2F, 9)));
  const Tensor c =
      MakeHostTensorWithShape<float16_t>(shape, ToFloat16(MakeM21PatternValues(12, 0.15F, 5)));
  const Tensor d =
      MakeHostTensorWithShape<float16_t>(shape, ToFloat16(MakeM21PatternValues(12, 0.1F, 13)));
  ExpectGraphOpMatchesCpu("selective_scan", DTypeCode::kFloat16, {x, a, b, c, d}, AttrMap{});
}

TEST_F(CudaBackendTest, SelectiveScanBFloat16GraphMatchesCpuReference) {
  const Shape shape({2, 2, 3});
  const Tensor x =
      MakeHostTensorWithShape<bfloat16_t>(shape, ToBFloat16(MakeM21PatternValues(12, 0.3F, 11)));
  const Tensor a =
      MakeHostTensorWithShape<bfloat16_t>(shape, ToBFloat16(MakeM21PatternValues(12, 0.05F, 7)));
  const Tensor b =
      MakeHostTensorWithShape<bfloat16_t>(shape, ToBFloat16(MakeM21PatternValues(12, 0.2F, 9)));
  const Tensor c =
      MakeHostTensorWithShape<bfloat16_t>(shape, ToBFloat16(MakeM21PatternValues(12, 0.15F, 5)));
  const Tensor d =
      MakeHostTensorWithShape<bfloat16_t>(shape, ToBFloat16(MakeM21PatternValues(12, 0.1F, 13)));
  ExpectGraphOpMatchesCpu("selective_scan", DTypeCode::kBFloat16, {x, a, b, c, d}, AttrMap{});
}

TEST_F(CudaBackendTest, SelectiveScanEmptyLeadingDimensionMatchesCpuReference) {
  const Shape shape({0, 3});
  const Tensor empty = MakeHostTensorWithShape<float>(shape, {});
  ExpectGraphOpMatchesCpu("selective_scan", DTypeCode::kFloat32,
                          {empty, empty, empty, empty, empty}, AttrMap{});
}

TEST_F(CudaBackendTest, SelectiveScanEmptyGradientMicrographRunsOnCuda) {
  const Shape shape({0, 3});
  const frame::ir::TensorType type = MakeDeviceType(DType::of<float>(), shape, device_);
  const std::vector<frame::ir::TensorType> forward_types{type, type, type, type, type};
  const Tensor empty = MakeHostTensorWithShape<float>(shape, {});
  const std::vector<Tensor> outputs = RunGradientMicrographOnCuda(
      "selective_scan", forward_types, nullptr, {empty, empty, empty, empty, empty, empty, empty});
  ASSERT_EQ(outputs.size(), 5U);
  for (const Tensor& output : outputs) {
    EXPECT_EQ(output.shape(), shape);
    EXPECT_EQ(output.raw_data(), nullptr);
  }
}

// transpose(x; perm):非自逆 rank-3 perm(spec 硬项)——perm=[2,0,1] 应用两次
// 得 [1,2,0]≠恒等,故非自逆。
TEST_F(CudaBackendTest, TransposeNonSelfInverseRank3PermGraphMatchesCpuReference) {
  const Tensor x =
      MakeHostTensorWithShape<float>(Shape({2, 3, 4}), MakeM21PatternValues(24, 0.3F, 7));
  const AttrMap attrs{{"perm", std::vector<int64_t>{2, 0, 1}}};
  ExpectGraphOpMatchesCpu("transpose", DTypeCode::kFloat32, {x}, attrs);
}

TEST_F(CudaBackendTest, TransposeFloat16GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensorWithShape<float16_t>(Shape({2, 3, 4}),
                                                      ToFloat16(MakeM21PatternValues(24, 0.3F, 7)));
  const AttrMap attrs{{"perm", std::vector<int64_t>{2, 0, 1}}};
  ExpectGraphOpMatchesCpu("transpose", DTypeCode::kFloat16, {x}, attrs);
}

TEST_F(CudaBackendTest, TransposeBFloat16GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensorWithShape<bfloat16_t>(
      Shape({2, 3, 4}), ToBFloat16(MakeM21PatternValues(24, 0.3F, 7)));
  const AttrMap attrs{{"perm", std::vector<int64_t>{2, 0, 1}}};
  ExpectGraphOpMatchesCpu("transpose", DTypeCode::kBFloat16, {x}, attrs);
}

// concat(xs...; axis):两输入常规例(fp32/fp16/bf16)+ 单输入退化例(min_count=1,
// 恒等拷贝,§1.4 表)。
TEST_F(CudaBackendTest, ConcatTwoInputsFloat32GraphMatchesCpuReference) {
  const Tensor a = MakeHostTensorWithShape<float>(Shape({2, 3}), MakeM21PatternValues(6, 0.3F, 7));
  const Tensor b = MakeHostTensorWithShape<float>(Shape({2, 2}), MakeM21PatternValues(4, 0.4F, 5));
  const AttrMap attrs{{"axis", int64_t{1}}};
  ExpectGraphOpMatchesCpu("concat", DTypeCode::kFloat32, {a, b}, attrs);
}

TEST_F(CudaBackendTest, ConcatTwoInputsFloat16GraphMatchesCpuReference) {
  const Tensor a = MakeHostTensorWithShape<float16_t>(Shape({2, 3}),
                                                      ToFloat16(MakeM21PatternValues(6, 0.3F, 7)));
  const Tensor b = MakeHostTensorWithShape<float16_t>(Shape({2, 2}),
                                                      ToFloat16(MakeM21PatternValues(4, 0.4F, 5)));
  const AttrMap attrs{{"axis", int64_t{1}}};
  ExpectGraphOpMatchesCpu("concat", DTypeCode::kFloat16, {a, b}, attrs);
}

TEST_F(CudaBackendTest, ConcatTwoInputsBFloat16GraphMatchesCpuReference) {
  const Tensor a = MakeHostTensorWithShape<bfloat16_t>(
      Shape({2, 3}), ToBFloat16(MakeM21PatternValues(6, 0.3F, 7)));
  const Tensor b = MakeHostTensorWithShape<bfloat16_t>(
      Shape({2, 2}), ToBFloat16(MakeM21PatternValues(4, 0.4F, 5)));
  const AttrMap attrs{{"axis", int64_t{1}}};
  ExpectGraphOpMatchesCpu("concat", DTypeCode::kBFloat16, {a, b}, attrs);
}

TEST_F(CudaBackendTest, ConcatSingleInputDegenerateIdentityGraphMatchesCpuReference) {
  const Tensor a = MakeHostTensorWithShape<float>(Shape({2, 3}), MakeM21PatternValues(6, 0.3F, 7));
  const AttrMap attrs{{"axis", int64_t{0}}};
  ExpectGraphOpMatchesCpu("concat", DTypeCode::kFloat32, {a}, attrs);
}

// slice(x; axis, start, stop):[2,5] 沿 axis=1 切 [1,4) -> [2,3]。
TEST_F(CudaBackendTest, SliceFloat32GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensorWithShape<float>(Shape({2, 5}), MakeM21PatternValues(10, 0.3F, 7));
  const AttrMap attrs{{"axis", int64_t{1}}, {"start", int64_t{1}}, {"stop", int64_t{4}}};
  ExpectGraphOpMatchesCpu("slice", DTypeCode::kFloat32, {x}, attrs);
}

TEST_F(CudaBackendTest, SliceFloat16GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensorWithShape<float16_t>(Shape({2, 5}),
                                                      ToFloat16(MakeM21PatternValues(10, 0.3F, 7)));
  const AttrMap attrs{{"axis", int64_t{1}}, {"start", int64_t{1}}, {"stop", int64_t{4}}};
  ExpectGraphOpMatchesCpu("slice", DTypeCode::kFloat16, {x}, attrs);
}

TEST_F(CudaBackendTest, SliceBFloat16GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensorWithShape<bfloat16_t>(
      Shape({2, 5}), ToBFloat16(MakeM21PatternValues(10, 0.3F, 7)));
  const AttrMap attrs{{"axis", int64_t{1}}, {"start", int64_t{1}}, {"stop", int64_t{4}}};
  ExpectGraphOpMatchesCpu("slice", DTypeCode::kBFloat16, {x}, attrs);
}

// gather(x[N,F], indices[K]):N=4,F=3,K=4,indices=[0,2,3,1](int32/int64 双档,
// §1.5 硬项)。
TEST_F(CudaBackendTest, GatherInt32IndicesFloat32GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensorWithShape<float>(Shape({4, 3}), MakeM21PatternValues(12, 0.2F, 5));
  const Tensor indices = MakeHostTensor1D<int32_t>({0, 2, 3, 1});
  ExpectGraphOpMatchesCpu("gather", DTypeCode::kFloat32, {x, indices}, AttrMap{});
}

TEST_F(CudaBackendTest, GatherInt32IndicesFloat16GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensorWithShape<float16_t>(Shape({4, 3}),
                                                      ToFloat16(MakeM21PatternValues(12, 0.2F, 5)));
  const Tensor indices = MakeHostTensor1D<int32_t>({0, 2, 3, 1});
  ExpectGraphOpMatchesCpu("gather", DTypeCode::kFloat16, {x, indices}, AttrMap{});
}

TEST_F(CudaBackendTest, GatherInt32IndicesBFloat16GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensorWithShape<bfloat16_t>(
      Shape({4, 3}), ToBFloat16(MakeM21PatternValues(12, 0.2F, 5)));
  const Tensor indices = MakeHostTensor1D<int32_t>({0, 2, 3, 1});
  ExpectGraphOpMatchesCpu("gather", DTypeCode::kBFloat16, {x, indices}, AttrMap{});
}

TEST_F(CudaBackendTest, GatherInt64IndicesFloat32GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensorWithShape<float>(Shape({4, 3}), MakeM21PatternValues(12, 0.2F, 5));
  const Tensor indices = MakeHostTensor1D<int64_t>({3, 0, 0, 2});
  ExpectGraphOpMatchesCpu("gather", DTypeCode::kFloat32, {x, indices}, AttrMap{});
}

// gather_grad_internal(gy[K,F], indices[K]; input_shape):重复索引累加例
// (N=3,F=2,K=3,indices=[0,0,2],§1.5 硬项:行0累加 gy 的第0、1两行,行1保持
// 零梯度,行2等于 gy 第2行)。
TEST_F(CudaBackendTest, GatherGradInternalDuplicateIndicesFloat32GraphMatchesCpuReference) {
  const Tensor gy = MakeHostTensorWithShape<float>(Shape({3, 2}), MakeM21PatternValues(6, 0.3F, 5));
  const Tensor indices = MakeHostTensor1D<int32_t>({0, 0, 2});
  const AttrMap attrs{{"input_shape", Shape({3, 2})}};
  ExpectGraphOpMatchesCpu("gather_grad_internal", DTypeCode::kFloat32, {gy, indices}, attrs);
}

TEST_F(CudaBackendTest, GatherGradInternalDuplicateIndicesFloat16GraphMatchesCpuReference) {
  const Tensor gy = MakeHostTensorWithShape<float16_t>(Shape({3, 2}),
                                                       ToFloat16(MakeM21PatternValues(6, 0.3F, 5)));
  const Tensor indices = MakeHostTensor1D<int32_t>({0, 0, 2});
  const AttrMap attrs{{"input_shape", Shape({3, 2})}};
  ExpectGraphOpMatchesCpu("gather_grad_internal", DTypeCode::kFloat16, {gy, indices}, attrs);
}

TEST_F(CudaBackendTest, GatherGradInternalDuplicateIndicesBFloat16GraphMatchesCpuReference) {
  const Tensor gy = MakeHostTensorWithShape<bfloat16_t>(
      Shape({3, 2}), ToBFloat16(MakeM21PatternValues(6, 0.3F, 5)));
  const Tensor indices = MakeHostTensor1D<int32_t>({0, 0, 2});
  const AttrMap attrs{{"input_shape", Shape({3, 2})}};
  ExpectGraphOpMatchesCpu("gather_grad_internal", DTypeCode::kBFloat16, {gy, indices}, attrs);
}

TEST_F(CudaBackendTest, GatherGradInternalInt64IndicesDuplicateFloat32GraphMatchesCpuReference) {
  const Tensor gy = MakeHostTensorWithShape<float>(Shape({3, 2}), MakeM21PatternValues(6, 0.3F, 5));
  const Tensor indices = MakeHostTensor1D<int64_t>({1, 1, 0});
  const AttrMap attrs{{"input_shape", Shape({3, 2})}};
  ExpectGraphOpMatchesCpu("gather_grad_internal", DTypeCode::kFloat32, {gy, indices}, attrs);
}

TEST_F(CudaBackendTest, ScatterAddFloat32Int32DuplicateMatchesCpuReference) {
  const Tensor updates =
      MakeHostTensorWithShape<float>(Shape({3, 2}), MakeM21PatternValues(6, 0.3F, 5));
  const Tensor indices = MakeHostTensor1D<int32_t>({0, 0, 2});
  const AttrMap attrs{{"output_shape", Shape({4, 2})}};
  ExpectGraphOpMatchesCpu("scatter_add", DTypeCode::kFloat32, {updates, indices}, attrs);
}

TEST_F(CudaBackendTest, ScatterAddFloat32Int64DuplicateMatchesCpuReference) {
  const Tensor updates =
      MakeHostTensorWithShape<float>(Shape({3, 2}), MakeM21PatternValues(6, 0.3F, 5));
  const Tensor indices = MakeHostTensor1D<int64_t>({1, 1, 3});
  const AttrMap attrs{{"output_shape", Shape({4, 2})}};
  ExpectGraphOpMatchesCpu("scatter_add", DTypeCode::kFloat32, {updates, indices}, attrs);
}

TEST_F(CudaBackendTest, ScatterAddFloat16Int32DuplicateMatchesCpuReference) {
  const Tensor updates = MakeHostTensorWithShape<float16_t>(
      Shape({3, 2}), ToFloat16(MakeM21PatternValues(6, 0.3F, 5)));
  const Tensor indices = MakeHostTensor1D<int32_t>({0, 0, 2});
  const AttrMap attrs{{"output_shape", Shape({4, 2})}};
  ExpectGraphOpMatchesCpu("scatter_add", DTypeCode::kFloat16, {updates, indices}, attrs);
}

TEST_F(CudaBackendTest, ScatterAddFloat16Int64DuplicateMatchesCpuReference) {
  const Tensor updates = MakeHostTensorWithShape<float16_t>(
      Shape({3, 2}), ToFloat16(MakeM21PatternValues(6, 0.3F, 5)));
  const Tensor indices = MakeHostTensor1D<int64_t>({1, 1, 3});
  const AttrMap attrs{{"output_shape", Shape({4, 2})}};
  ExpectGraphOpMatchesCpu("scatter_add", DTypeCode::kFloat16, {updates, indices}, attrs);
}

TEST_F(CudaBackendTest, ScatterAddBFloat16Int32DuplicateMatchesCpuReference) {
  const Tensor updates = MakeHostTensorWithShape<bfloat16_t>(
      Shape({3, 2}), ToBFloat16(MakeM21PatternValues(6, 0.3F, 5)));
  const Tensor indices = MakeHostTensor1D<int32_t>({0, 0, 2});
  const AttrMap attrs{{"output_shape", Shape({4, 2})}};
  ExpectGraphOpMatchesCpu("scatter_add", DTypeCode::kBFloat16, {updates, indices}, attrs);
}

TEST_F(CudaBackendTest, ScatterAddBFloat16Int64DuplicateMatchesCpuReference) {
  const Tensor updates = MakeHostTensorWithShape<bfloat16_t>(
      Shape({3, 2}), ToBFloat16(MakeM21PatternValues(6, 0.3F, 5)));
  const Tensor indices = MakeHostTensor1D<int64_t>({1, 1, 3});
  const AttrMap attrs{{"output_shape", Shape({4, 2})}};
  ExpectGraphOpMatchesCpu("scatter_add", DTypeCode::kBFloat16, {updates, indices}, attrs);
}

TEST_F(CudaBackendTest, ScatterAddOutOfRangeIndexReturnsErrorOnCuda) {
  const Tensor updates =
      MakeHostTensorWithShape<float>(Shape({2, 2}), MakeM21PatternValues(4, 0.3F, 5));
  const Tensor indices = MakeHostTensor1D<int64_t>({0, 4});
  const Tensor updates_dev = CopyToDevice(updates);
  const Tensor indices_dev = CopyToDevice(indices);
  const Result<Tensor> out_dev_result =
      Tensor::empty(Shape({4, 2}), DType::of<float>(), device_, *cuda_allocator_);
  ASSERT_TRUE(out_dev_result.is_ok()) << out_dev_result.status().message();
  std::vector<Tensor> inputs{updates_dev, indices_dev};
  std::vector<Tensor> outputs{out_dev_result.value()};
  const AttrMap attrs{{"output_shape", Shape({4, 2})}};
  KernelInvocation invocation;
  invocation.op = "scatter_add";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;
  const Status status = cuda_backend_->launch(invocation, stream_.get());
  EXPECT_FALSE(status.is_ok());
  EXPECT_EQ(status.message(),
            "op 'scatter_add' cuda kernel indices[1]=4 is out of range for out's first "
            "dimension V=4");
}

// gather 越界索引负例(§1.5:ARCH-031,拒绝 device 侧静默 clamp)——直接经
// Backend::launch(非图编译路径)喂一个越界 indices,断言返回错误且消息含
// "out of range"。
TEST_F(CudaBackendTest, GatherOutOfRangeIndexReturnsErrorOnCuda) {
  const Shape x_shape({4, 3});
  const Tensor x = MakeHostTensorWithShape<float>(x_shape, MakeM21PatternValues(12, 0.2F, 5));
  const Tensor indices = MakeHostTensor1D<int32_t>({0, 1, 10, 2});  // 10 越界(N=4)
  const Tensor x_dev = CopyToDevice(x);
  const Tensor indices_dev = CopyToDevice(indices);
  const Result<Tensor> out_dev_result =
      Tensor::empty(Shape({4, 3}), DType::of<float>(), device_, *cuda_allocator_);
  ASSERT_TRUE(out_dev_result.is_ok()) << out_dev_result.status().message();
  std::vector<Tensor> inputs{x_dev, indices_dev};
  std::vector<Tensor> outputs{out_dev_result.value()};

  KernelInvocation invocation;
  invocation.op = "gather";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = cuda_backend_->launch(invocation, stream_.get());
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("out of range"), std::string_view::npos);
}

// softmax/layer_norm 的图编译训练路径(前向 + 反向微图,反向由既有
// mul/sum/reshape/matmul/add/rsqrt cuda kernel 承载,§1.2 表):
// loss=sum(op(x)^2),wrt x,device=cuda 与 device=cpu 分别构图(build_backward_graph
// 按输入图自带的 device 生成同 device 训练图)、对比 loss 与 grad_x。
TEST_F(CudaBackendTest, SoftmaxTrainingGraphBackwardMicrographRunsOnCuda) {
  const Shape x_shape({2, 3});
  const std::vector<float> x_values_f = MakeM21PatternValues(6, 0.4F, 7);
  const std::vector<int32_t> wrt{0};

  auto build_graph = [&](Device device) {
    Graph graph("test_cuda_backend_softmax_squared_loss");
    Value* x = graph.add_graph_input(MakeDeviceType(DType::of<float>(), x_shape, device)).value();
    Node* softmax_node = create_node_with_inferred_types(graph, "softmax", {x}).value();
    Node* square_node =
        create_node_with_inferred_types(graph, "square", {softmax_node->output(0)}).value();
    const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
    Node* sum_node =
        create_node_with_inferred_types(graph, "sum", {square_node->output(0)}, sum_attrs).value();
    EXPECT_TRUE(graph.mark_output(sum_node, 0).is_ok());
    return graph;
  };

  const Tensor x_cuda = MakeHostTensorWithShape<float>(x_shape, x_values_f);
  const Graph cuda_forward = build_graph(device_);
  const std::vector<Tensor> cuda_outputs = RunTrainingGraphOnCuda(cuda_forward, 0, wrt, {x_cuda});

  const Tensor x_cpu = MakeHostTensorWithShape<float>(x_shape, x_values_f);
  const Graph cpu_forward = build_graph(cpu_device_);
  const std::vector<Tensor> cpu_outputs = RunTrainingGraphOnCpu(cpu_forward, 0, wrt, {x_cpu});

  ASSERT_EQ(cuda_outputs.size(), 2U);
  ASSERT_EQ(cpu_outputs.size(), 2U);
  EXPECT_TRUE(
      tensor_all_close(cuda_outputs[0], cpu_outputs[0], default_tolerance(DTypeCode::kFloat32)));
  EXPECT_TRUE(
      tensor_all_close(cuda_outputs[1], cpu_outputs[1], default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(CudaBackendTest, LayerNormTrainingGraphBackwardMicrographRunsOnCuda) {
  const Shape x_shape({2, 3});
  const Shape param_shape({3});
  const std::vector<float> x_values_f = MakeM21PatternValues(6, 0.3F, 7);
  const std::vector<float> gamma_values_f = MakeM21PatternValues(3, 0.2F, 5);
  const std::vector<float> beta_values_f = MakeM21PatternValues(3, 0.1F, 3);
  const std::vector<int32_t> wrt{0};

  auto build_graph = [&](Device device) {
    Graph graph("test_cuda_backend_layer_norm_squared_loss");
    Value* x = graph.add_graph_input(MakeDeviceType(DType::of<float>(), x_shape, device)).value();
    Value* gamma =
        graph.add_graph_input(MakeDeviceType(DType::of<float>(), param_shape, device)).value();
    Value* beta =
        graph.add_graph_input(MakeDeviceType(DType::of<float>(), param_shape, device)).value();
    const AttrMap ln_attrs{{"eps", 1e-5}};
    Node* ln_node =
        create_node_with_inferred_types(graph, "layer_norm", {x, gamma, beta}, ln_attrs).value();
    Node* square_node =
        create_node_with_inferred_types(graph, "square", {ln_node->output(0)}).value();
    const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
    Node* sum_node =
        create_node_with_inferred_types(graph, "sum", {square_node->output(0)}, sum_attrs).value();
    EXPECT_TRUE(graph.mark_output(sum_node, 0).is_ok());
    return graph;
  };

  const Tensor x_cuda = MakeHostTensorWithShape<float>(x_shape, x_values_f);
  const Tensor gamma_cuda = MakeHostTensorWithShape<float>(param_shape, gamma_values_f);
  const Tensor beta_cuda = MakeHostTensorWithShape<float>(param_shape, beta_values_f);
  const Graph cuda_forward = build_graph(device_);
  const std::vector<Tensor> cuda_outputs =
      RunTrainingGraphOnCuda(cuda_forward, 0, wrt, {x_cuda, gamma_cuda, beta_cuda});

  const Tensor x_cpu = MakeHostTensorWithShape<float>(x_shape, x_values_f);
  const Tensor gamma_cpu = MakeHostTensorWithShape<float>(param_shape, gamma_values_f);
  const Tensor beta_cpu = MakeHostTensorWithShape<float>(param_shape, beta_values_f);
  const Graph cpu_forward = build_graph(cpu_device_);
  const std::vector<Tensor> cpu_outputs =
      RunTrainingGraphOnCpu(cpu_forward, 0, wrt, {x_cpu, gamma_cpu, beta_cpu});

  ASSERT_EQ(cuda_outputs.size(), 2U);
  ASSERT_EQ(cpu_outputs.size(), 2U);
  EXPECT_TRUE(
      tensor_all_close(cuda_outputs[0], cpu_outputs[0], default_tolerance(DTypeCode::kFloat32)));
  EXPECT_TRUE(
      tensor_all_close(cuda_outputs[1], cpu_outputs[1], default_tolerance(DTypeCode::kFloat32)));
}

// =============================================================================
// M23(批5 T4,§1.2/1.6 决议点B/F,ADR-0022)rfft/irfft cuda kernel 一致性套件:
//   1. CPU/CUDA 数值一致(经图编译路径,ExpectGraphOpMatchesCpu 驱动
//      frame::runtime::compile("cuda") vs ("cpu")):rfft/irfft 各奇偶 n +
//      批量 leading 维三组;
//   2. roundtrip:irfft(rfft(x), n)≡x,纯 cuda 执行(经图编译路径),偶/奇 n +
//      批量 leading 维;
//   3. dtype fail-loud 负例(非 fp32,eager Backend::launch 直接触发,同
//      GatherOutOfRangeIndexReturnsErrorOnCuda 先例);
//   4. 梯度微图在 cuda 后端跑通:解析梯度 ≡ 中心差分(经
//      CheckGradientMatchesNumericOnCuda,rfft/irfft 各一组)。
// =============================================================================

TEST_F(CudaBackendTest, RfftEvenNRank1GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensorWithShape<float>(Shape({4}), MakeM21PatternValues(4, 0.5F, 7));
  ExpectGraphOpMatchesCpu("rfft", DTypeCode::kFloat32, {x}, AttrMap{});
}

TEST_F(CudaBackendTest, RfftOddNRank1GraphMatchesCpuReference) {
  const Tensor x = MakeHostTensorWithShape<float>(Shape({5}), MakeM21PatternValues(5, 0.4F, 9));
  ExpectGraphOpMatchesCpu("rfft", DTypeCode::kFloat32, {x}, AttrMap{});
}

TEST_F(CudaBackendTest, RfftBatchedLeadingDimGraphMatchesCpuReference) {
  // leading 维=2,验证 cufftPlanMany 的 batch 形参(决议点F:batch=前导维
  // 乘积)与 cpu 侧 pocketfft 完整 shape 遍历数值一致。
  const Tensor x = MakeHostTensorWithShape<float>(Shape({2, 6}), MakeM21PatternValues(12, 0.3F, 9));
  ExpectGraphOpMatchesCpu("rfft", DTypeCode::kFloat32, {x}, AttrMap{});
}

TEST_F(CudaBackendTest, IrfftEvenNRank1GraphMatchesCpuReference) {
  const Tensor z = MakeHostTensorWithShape<float>(Shape({3, 2}), MakeM21PatternValues(6, 0.4F, 7));
  const AttrMap attrs{{"n", int64_t{4}}};
  ExpectGraphOpMatchesCpu("irfft", DTypeCode::kFloat32, {z}, attrs);
}

TEST_F(CudaBackendTest, IrfftOddNRank1GraphMatchesCpuReference) {
  const Tensor z = MakeHostTensorWithShape<float>(Shape({3, 2}), MakeM21PatternValues(6, 0.3F, 9));
  const AttrMap attrs{{"n", int64_t{5}}};
  ExpectGraphOpMatchesCpu("irfft", DTypeCode::kFloat32, {z}, attrs);
}

TEST_F(CudaBackendTest, IrfftBatchedLeadingDimGraphMatchesCpuReference) {
  const Tensor z =
      MakeHostTensorWithShape<float>(Shape({2, 4, 2}), MakeM21PatternValues(16, 0.25F, 9));
  const AttrMap attrs{{"n", int64_t{6}}};
  ExpectGraphOpMatchesCpu("irfft", DTypeCode::kFloat32, {z}, attrs);
}

// roundtrip(纯 cuda 执行,rfft->irfft 两节点图,经图编译路径,与
// tests/cpp/ops/test_op_fft.cpp::FftRoundtripTest 同款构造手法,device 换为
// cuda,判据也从"回读 cpu 参考"改为"直接比对原始输入")。
Graph BuildFftRoundtripGraphForCuda(Device device, const Shape& x_shape, int64_t n) {
  Graph graph("test_cuda_backend_fft_roundtrip");
  frame::ir::TensorType x_type;
  x_type.dtype = DType::of<float>();
  x_type.shape = x_shape;
  x_type.layout = frame::ir::Layout::kRowMajor;
  x_type.device = device;
  Value* x = graph.add_graph_input(x_type).value();
  Node* rfft_node = create_node_with_inferred_types(graph, "rfft", {x}).value();
  const AttrMap irfft_attrs{{"n", n}};
  Node* irfft_node =
      create_node_with_inferred_types(graph, "irfft", {rfft_node->output(0)}, irfft_attrs).value();
  EXPECT_TRUE(graph.mark_output(irfft_node, 0).is_ok());
  return graph;
}

TEST_F(CudaBackendTest, FftRoundtripEvenNRank1MatchesInputOnCuda) {
  const int64_t n = 6;
  const Graph graph = BuildFftRoundtripGraphForCuda(device_, Shape({n}), n);
  const Result<std::shared_ptr<Executable>> executable =
      frame::runtime::compile(graph, frame::kCudaBackendName, CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  const Tensor x_host =
      MakeHostTensorWithShape<float>(Shape({n}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor x_dev = CopyToDevice(x_host);
  std::vector<Tensor> inputs{x_dev};
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCudaBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  const Tensor host_out = CopyToHost(outputs.value()[0]);
  ASSERT_TRUE(stream_->synchronize().is_ok());
  EXPECT_TRUE(tensor_all_close(host_out, x_host, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(CudaBackendTest, FftRoundtripOddNRank1MatchesInputOnCuda) {
  const int64_t n = 5;
  const Graph graph = BuildFftRoundtripGraphForCuda(device_, Shape({n}), n);
  const Result<std::shared_ptr<Executable>> executable =
      frame::runtime::compile(graph, frame::kCudaBackendName, CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  const Tensor x_host = MakeHostTensorWithShape<float>(Shape({n}), {1.0F, -2.0F, 3.0F, 0.5F, 4.0F});
  const Tensor x_dev = CopyToDevice(x_host);
  std::vector<Tensor> inputs{x_dev};
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCudaBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  const Tensor host_out = CopyToHost(outputs.value()[0]);
  ASSERT_TRUE(stream_->synchronize().is_ok());
  EXPECT_TRUE(tensor_all_close(host_out, x_host, default_tolerance(DTypeCode::kFloat32)));
}

TEST_F(CudaBackendTest, FftRoundtripBatchedLeadingDimMatchesInputOnCuda) {
  const int64_t n = 6;
  const Graph graph = BuildFftRoundtripGraphForCuda(device_, Shape({2, n}), n);
  const Result<std::shared_ptr<Executable>> executable =
      frame::runtime::compile(graph, frame::kCudaBackendName, CompileOptions{});
  ASSERT_TRUE(executable.is_ok()) << executable.status().message();

  const Tensor x_host = MakeHostTensorWithShape<float>(
      Shape({2, n}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, -1.0F, 0.5F, 2.5F, -3.0F, 4.5F, 0.0F});
  const Tensor x_dev = CopyToDevice(x_host);
  std::vector<Tensor> inputs{x_dev};
  const Result<std::vector<Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
      *executable.value(), frame::kCudaBackendName, inputs);
  ASSERT_TRUE(outputs.is_ok()) << outputs.status().message();
  const Tensor host_out = CopyToHost(outputs.value()[0]);
  ASSERT_TRUE(stream_->synchronize().is_ok());
  EXPECT_TRUE(tensor_all_close(host_out, x_host, default_tolerance(DTypeCode::kFloat32)));
}

// dtype fail-loud 负例(ADR-0022 决策 5:cuFFT 半精度不在 v0 范围内,kernel 须
// fail-loud 拒绝非 fp32),eager Backend::launch 直接触发,同
// GatherOutOfRangeIndexReturnsErrorOnCuda 先例。
TEST_F(CudaBackendTest, RfftRejectsNonFp32DtypeOnCuda) {
  const Tensor x = MakeHostTensor1D<int32_t>({1, 2, 3, 4});
  const Tensor x_dev = CopyToDevice(x);
  const Result<Tensor> out_dev_result =
      Tensor::empty(Shape({3, 2}), DType::of<int32_t>(), device_, *cuda_allocator_);
  ASSERT_TRUE(out_dev_result.is_ok()) << out_dev_result.status().message();
  std::vector<Tensor> inputs{x_dev};
  std::vector<Tensor> outputs{out_dev_result.value()};

  KernelInvocation invocation;
  invocation.op = "rfft";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.device = device_;
  const Status status = cuda_backend_->launch(invocation, stream_.get());
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("requires x/out to be float32"), std::string_view::npos);
}

TEST_F(CudaBackendTest, IrfftRejectsNonFp32DtypeOnCuda) {
  const Tensor z = MakeHostTensor1D<int32_t>({1, 2, 3, 4, 5, 6});
  const Tensor z_dev = CopyToDevice(z);
  const Result<Tensor> out_dev_result =
      Tensor::empty(Shape({4}), DType::of<int32_t>(), device_, *cuda_allocator_);
  ASSERT_TRUE(out_dev_result.is_ok()) << out_dev_result.status().message();
  std::vector<Tensor> inputs{z_dev};
  std::vector<Tensor> outputs{out_dev_result.value()};
  const AttrMap attrs{{"n", int64_t{4}}};

  KernelInvocation invocation;
  invocation.op = "irfft";
  invocation.inputs = inputs;
  invocation.outputs = outputs;
  invocation.attrs = &attrs;
  invocation.device = device_;
  const Status status = cuda_backend_->launch(invocation, stream_.get());
  EXPECT_FALSE(status.is_ok());
  EXPECT_NE(status.message().find("requires z/out to be float32"), std::string_view::npos);
}

// 梯度微图在 cuda 后端跑通:解析梯度 ≡ 中心差分(BUILD-011 容差,rfft/irfft
// 各一组;h 取值同 tests/cpp/ops/test_op_fft.cpp::kFftCentralDifferenceH,两
// 算子均为线性变换,数值微分截断误差理论上为零)。
constexpr double kFftCudaCentralDifferenceH = 1e-2;

TEST_F(CudaBackendTest, RfftGradientMatchesNumericOnCuda) {
  const int64_t n = 4;
  auto build_graph = [&](Device device) {
    Graph graph("test_cuda_backend_rfft_loss");
    Value* x =
        graph.add_graph_input(MakeDeviceType(DType::of<float>(), Shape({n}), device)).value();
    Node* rfft_node = create_node_with_inferred_types(graph, "rfft", {x}).value();
    const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
    Node* sum_node =
        create_node_with_inferred_types(graph, "sum", {rfft_node->output(0)}, sum_attrs).value();
    EXPECT_TRUE(graph.mark_output(sum_node, 0).is_ok());
    return graph;
  };
  const Graph forward = build_graph(device_);
  Tensor x = MakeHostTensorWithShape<float>(Shape({n}), {1.0F, 2.0F, 3.0F, 4.0F});
  const std::vector<int32_t> wrt{0};
  EXPECT_TRUE(CheckGradientMatchesNumericOnCuda(forward, 0, wrt, 0, x, kFftCudaCentralDifferenceH));
}

TEST_F(CudaBackendTest, IrfftGradientMatchesNumericOnCuda) {
  const int64_t n = 5;
  const int64_t k = n / 2 + 1;
  auto build_graph = [&](Device device) {
    Graph graph("test_cuda_backend_irfft_loss");
    Value* z =
        graph.add_graph_input(MakeDeviceType(DType::of<float>(), Shape({k, 2}), device)).value();
    const AttrMap irfft_attrs{{"n", n}};
    Node* irfft_node = create_node_with_inferred_types(graph, "irfft", {z}, irfft_attrs).value();
    const AttrMap sum_attrs{{"axes", std::vector<int64_t>{}}};
    Node* sum_node =
        create_node_with_inferred_types(graph, "sum", {irfft_node->output(0)}, sum_attrs).value();
    EXPECT_TRUE(graph.mark_output(sum_node, 0).is_ok());
    return graph;
  };
  const Graph forward = build_graph(device_);
  Tensor z =
      MakeHostTensorWithShape<float>(Shape({k, 2}), {0.5F, -1.0F, 2.0F, 0.75F, -0.25F, 1.5F});
  const std::vector<int32_t> wrt{0};
  EXPECT_TRUE(CheckGradientMatchesNumericOnCuda(forward, 0, wrt, 0, z, kFftCudaCentralDifferenceH));
}

}  // namespace

FRAME_REGISTER_OP(kCpuOnlySandwichOpName)
    .input("x", "sandwich op input")
    .output("out",
            "test_cuda_backend cpu-only sandwich op (M11 cross-device fallback orchestration "
            "test; deliberately never given a cuda kernel or decomposition)")
    .shape_infer(&InferCpuOnlySandwichOpShape);

FRAME_REGISTER_KERNEL(kCpuOnlySandwichOpName, frame::kCpuBackendName, CpuOnlySandwichOpKernel);
