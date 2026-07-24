// =============================================================================
// benchmarks/bench_conv2d.cpp —— conv2d 编译执行基准(BENCH-020 登记,M21 批3
// T7,docs/plan/2026-07-18-batch3-m21-conv.md 第3节)。
//
// 测量对象:runtime::compile 编译执行主路径(ARCH-010,BENCH-011)—— 单
// conv2d 节点图(NCHW,固定形状 N=8, Cin=Cout=16, H=W=32, K=3, stride=1,
// padding=1,groups=1,带 bias)。编译(runtime::compile)只做一次(不计时,
// setup 阶段);循环内仅执行 Executable::run(计时区间)。
// 后端/dtype 档位:cpu 恒定注册(仅 fp32);cuda 后端在运行期探测可用
// (BackendRegistry 已注册该键)时同源注册 fp32 严格 / fp32 allow_tf32(ADR-0021
// 决策4)/ fp16 / bf16 四档,不可用则跳过(见文件尾 main(),手法同
// bench_matmul.cpp 的精度对照基准族)。
// 对照组:本批不提供 eager 对照组(理由同 bench_matmul.cpp 文件头注释——
// eager 唯一合法入口 Backend::launch 的调用点白名单当前仅覆盖 src/runtime/,
// benchmarks/ 未列入其中)。
//
// 与 bench_matmul.cpp 的复用关系:本文件与 bench_matmul.cpp 分属不同
// target(benchmarks/CMakeLists.txt 逐文件建 executable),彼此不 include
// 对方(该文件头注释已声明"独立实现"取舍,同一 benchmarks/ 目录当前无共享
// 头);故图构造/填充张量等 helper 在本文件内独立实现一份,结构与命名手法
// 照抄 bench_matmul.cpp(REUSE-002 既有取舍的延续,非重复实现新逻辑)。
//
// 语言纪律(铁律 #4):标识符/日志英文,注释中文。计时/统计/重复次数控制
// 全部经 Google Benchmark(ADR-0014),不手写系统时钟计时循环(BENCH-002);
// 不对性能数字做断言(BENCH-012)。
// =============================================================================

#include <benchmark/benchmark.h>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <frame/frame.h>
#include <frame/ops/graph_builder.h>

namespace {

constexpr int64_t kBatch = 8;
constexpr int64_t kInChannels = 16;
constexpr int64_t kOutChannels = 16;
constexpr int64_t kHeight = 32;
constexpr int64_t kWidth = 32;
constexpr int64_t kKernel = 3;
constexpr int64_t kStride = 1;
constexpr int64_t kPadding = 1;

// 构造 [dims] TensorType(device/dtype 均由调用方显式指定,支持 cpu/cuda 多档
// 后端与 fp32/fp16/bf16 多档精度——手法同 bench_matmul.cpp::MakeSquareTensorType,
// 本文件形状非方阵故改用通用 dims 形参)。
frame::ir::TensorType MakeConv2dTensorType(std::vector<int64_t> dims, frame::Device device,
                                           frame::DType dtype) {
  frame::ir::TensorType type;
  type.dtype = dtype;
  type.shape = frame::Shape(std::move(dims));
  type.device = device;
  return type;
}

// 建单 conv2d 节点图(x[N,Cin,H,W], w[Cout,Cin,KH,KW], bias[Cout] ->
// y[N,Cout,H,W]),输入按序 [x, w, bias](stride=1/padding=1 令输出空间维与
// 输入相等,compute_conv2d_geometry floor 口径,src/ops/schemas/conv.cpp)。
// conv2d 有 stride/padding/groups 三个必填属性,故经
// ops::create_node_with_inferred_types(经 OpRegistry 查 schema 做 shape 推断,
// 免去手工重算输出空间维——同 src/nn/layers.cpp::Conv2d 构图手法)而非
// ir::Graph::create_node(该方法不支持属性,bench_matmul.cpp 的 matmul/add/relu
// 三算子均无属性故可直接用之,本文件因 conv2d 需要属性而改用此 helper)。
frame::Result<frame::ir::Graph> BuildConv2dGraph(frame::Device device, frame::DType dtype) {
  frame::ir::Graph graph("bench_conv2d");

  const frame::Result<frame::ir::Value*> x = graph.add_graph_input(
      MakeConv2dTensorType({kBatch, kInChannels, kHeight, kWidth}, device, dtype));
  if (!x.is_ok()) return x.status();
  const frame::Result<frame::ir::Value*> w = graph.add_graph_input(
      MakeConv2dTensorType({kOutChannels, kInChannels, kKernel, kKernel}, device, dtype));
  if (!w.is_ok()) return w.status();
  const frame::Result<frame::ir::Value*> bias =
      graph.add_graph_input(MakeConv2dTensorType({kOutChannels}, device, dtype));
  if (!bias.is_ok()) return bias.status();

  const frame::ops::AttrMap attrs{
      {"stride", std::vector<int64_t>{kStride, kStride}},
      {"padding", std::vector<int64_t>{kPadding, kPadding}},
      {"groups", int64_t{1}},
  };
  const frame::Result<frame::ir::Node*> conv_node = frame::ops::create_node_with_inferred_types(
      graph, "conv2d", {x.value(), w.value(), bias.value()}, attrs);
  if (!conv_node.is_ok()) return conv_node.status();

  const frame::Status mark_status = graph.mark_output(conv_node.value(), 0);
  if (!mark_status.is_ok()) return mark_status;
  return graph;
}

// 按 [dims] 形状、dtype T、给定填充值 fill_value 构造一个位于 device 上的输入
// 张量——手法照抄 bench_matmul.cpp::MakeFilledTensorT(cpu 后端直接经主机指针
// 写入;非 cpu 后端先在 cpu 侧构造同形张量,再经 backend.copy() H2D 搬运,
// setup 阶段不计时)。
template <frame::ScalarType T>
frame::Result<frame::Tensor> MakeFilledTensorT(std::vector<int64_t> dims, T fill_value,
                                               frame::Device device, frame::hal::Backend& backend,
                                               frame::hal::Allocator& allocator) {
  const frame::Shape shape(std::move(dims));
  const frame::DType dtype = frame::DType::of<T>();

  if (device.backend == frame::kCpuBackendName) {
    const frame::Result<frame::Tensor> tensor_result =
        frame::Tensor::empty(shape, dtype, device, allocator);
    if (!tensor_result.is_ok()) return tensor_result.status();
    frame::Tensor tensor = tensor_result.value();
    T* data = tensor.data<T>();
    for (int64_t i = 0; i < tensor.numel(); ++i) data[i] = fill_value;
    return tensor;
  }

  const frame::Result<frame::hal::Backend*> cpu_backend_result =
      frame::hal::BackendRegistry::instance().get(frame::kCpuBackendName);
  if (!cpu_backend_result.is_ok()) return cpu_backend_result.status();
  frame::hal::Allocator* cpu_allocator = cpu_backend_result.value()->allocator(frame::cpu_device());

  const frame::Result<frame::Tensor> host_tensor_result =
      frame::Tensor::empty(shape, dtype, frame::cpu_device(), *cpu_allocator);
  if (!host_tensor_result.is_ok()) return host_tensor_result.status();
  frame::Tensor host_tensor = host_tensor_result.value();
  T* host_data = host_tensor.data<T>();
  for (int64_t i = 0; i < host_tensor.numel(); ++i) host_data[i] = fill_value;

  const frame::Result<frame::Tensor> device_tensor_result =
      frame::Tensor::empty(shape, dtype, device, allocator);
  if (!device_tensor_result.is_ok()) return device_tensor_result.status();
  frame::Tensor device_tensor = device_tensor_result.value();

  const size_t bytes = static_cast<size_t>(host_tensor.numel()) * dtype.itemsize();
  if (bytes > 0) {
    const frame::Status copy_status =
        backend.copy(device_tensor.raw_data(), device, host_tensor.raw_data(), frame::cpu_device(),
                     bytes, nullptr);
    if (!copy_status.is_ok()) return copy_status;
  }
  return device_tensor;
}

// 按运行时 frame::DType 分派到 MakeFilledTensorT<T>(唯一允许的 dtype 运行时
// 分派点是 frame::dispatch_dtype,见 include/frame/core/dtype.h 头注释;本基准
// 仅消费 fp32/fp16/bf16 三档,其余 dtype 落 default 分支返回错误,该分支在
// 本基准实际运行时不可达——手法同 bench_matmul.cpp::MakeDTypePrecisionFilledTensor)。
frame::Result<frame::Tensor> MakeDTypeFilledTensor(std::vector<int64_t> dims, frame::DType dtype,
                                                   frame::Device device,
                                                   frame::hal::Backend& backend,
                                                   frame::hal::Allocator& allocator) {
  return frame::dispatch_dtype(dtype.code(), [&]<typename T>() -> frame::Result<frame::Tensor> {
    if constexpr (std::is_same_v<T, float>) {
      return MakeFilledTensorT<T>(std::move(dims), 1.0F, device, backend, allocator);
    } else if constexpr (std::is_same_v<T, frame::float16_t>) {
      return MakeFilledTensorT<T>(std::move(dims), frame::float_to_float16(1.0F), device, backend,
                                  allocator);
    } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
      return MakeFilledTensorT<T>(std::move(dims), frame::float_to_bfloat16(1.0F), device, backend,
                                  allocator);
    } else {
      return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                                 "unsupported dtype for conv2d benchmark");
    }
  });
}

// 基准主体:对给定后端/dtype/compile_options 一次性建图 + runtime::compile
// (setup,不计时,BENCH-011);循环内仅执行 run_with_allocated_outputs
// (计时区间)。cpu 与 cuda 多精度档共用同一份实现(REUSE-002)。
void RunConv2dBenchmark(benchmark::State& state, std::string_view backend_name, frame::DType dtype,
                        const frame::hal::CompileOptions& compile_options) {
  const frame::Device device{backend_name, 0};

  const frame::Result<frame::hal::Backend*> backend_result =
      frame::hal::BackendRegistry::instance().get(backend_name);
  if (!backend_result.is_ok()) {
    state.SkipWithError(std::string(backend_result.status().message()));
    return;
  }
  frame::hal::Backend* backend = backend_result.value();
  frame::hal::Allocator* allocator = backend->allocator(device);
  if (allocator == nullptr) {
    state.SkipWithError("allocator is null");
    return;
  }

  const frame::Result<frame::ir::Graph> graph = BuildConv2dGraph(device, dtype);
  if (!graph.is_ok()) {
    state.SkipWithError(std::string(graph.status().message()));
    return;
  }

  // 主路径:runtime::compile 一次(setup,不计时,BENCH-011)。
  const frame::Result<std::shared_ptr<frame::hal::Executable>> executable =
      frame::runtime::compile(graph.value(), backend_name, compile_options);
  if (!executable.is_ok()) {
    state.SkipWithError(std::string(executable.status().message()));
    return;
  }

  const frame::Result<frame::Tensor> x_input = MakeDTypeFilledTensor(
      {kBatch, kInChannels, kHeight, kWidth}, dtype, device, *backend, *allocator);
  if (!x_input.is_ok()) {
    state.SkipWithError(std::string(x_input.status().message()));
    return;
  }
  const frame::Result<frame::Tensor> w_input = MakeDTypeFilledTensor(
      {kOutChannels, kInChannels, kKernel, kKernel}, dtype, device, *backend, *allocator);
  if (!w_input.is_ok()) {
    state.SkipWithError(std::string(w_input.status().message()));
    return;
  }
  const frame::Result<frame::Tensor> bias_input =
      MakeDTypeFilledTensor({kOutChannels}, dtype, device, *backend, *allocator);
  if (!bias_input.is_ok()) {
    state.SkipWithError(std::string(bias_input.status().message()));
    return;
  }
  const std::vector<frame::Tensor> inputs{x_input.value(), w_input.value(), bias_input.value()};

  for (auto _ : state) {
    // 非 const 局部变量:绑定 DoNotOptimize(Tp&) 非弃用重载(const Tp& 重载对
    // 非平凡可拷贝类型已弃用,同 bench_matmul.cpp 既有手法)。
    frame::Result<std::vector<frame::Tensor>> outputs =
        frame::runtime::run_with_allocated_outputs(*executable.value(), backend_name, inputs);
    benchmark::DoNotOptimize(outputs);
  }
}

// cpu 档:恒定注册,仅 fp32(cpu 后端无 fp16/bf16 Tensor Core 语义可对照)。
void BM_Conv2d_Cpu(benchmark::State& state) {
  RunConv2dBenchmark(state, frame::kCpuBackendName, frame::DType::of<float>(),
                     frame::hal::CompileOptions{});
}
// BENCHMARK() 宏展开出的静态注册对象由 Google Benchmark 内部实现构造(可能
// 分配内存),tidy 无法证明其构造函数不抛;第三方宏产出的既有模式,非本文件
// 引入的风险(同 bench_matmul.cpp 既有抑制)。
// NOLINTNEXTLINE(bugprone-throwing-static-initialization)
BENCHMARK(BM_Conv2d_Cpu);

// cuda 精度对照四档(fp32 严格 / fp32 allow_tf32 / fp16 / bf16),仅 cuda 后端
// 可用时经 main() 运行期条件注册(理由同 bench_matmul.cpp 尾部注释:cuda 后端
// 在未装 CUDA Toolkit 的机器上不存在,静态注册会产出必然 SkipWithError 的
// 基准项而非"不存在该基准")。
void BM_Conv2dPrecision_Fp32Strict(benchmark::State& state) {
  RunConv2dBenchmark(state, frame::kCudaBackendName, frame::DType::of<float>(),
                     frame::hal::CompileOptions{});
}

void BM_Conv2dPrecision_Fp32Tf32(benchmark::State& state) {
  frame::hal::CompileOptions options;
  options.allow_tf32 = true;
  RunConv2dBenchmark(state, frame::kCudaBackendName, frame::DType::of<float>(), options);
}

void BM_Conv2dPrecision_Fp16(benchmark::State& state) {
  RunConv2dBenchmark(state, frame::kCudaBackendName, frame::DType::of<frame::float16_t>(),
                     frame::hal::CompileOptions{});
}

void BM_Conv2dPrecision_Bf16(benchmark::State& state) {
  RunConv2dBenchmark(state, frame::kCudaBackendName, frame::DType::of<frame::bfloat16_t>(),
                     frame::hal::CompileOptions{});
}

}  // namespace

// 自定义 main(不链接 benchmark::benchmark_main):cuda 档基准须在注册前完成
// 运行期探测,手法同 bench_matmul.cpp 尾部(理由不复述)。
int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;

  if (frame::hal::BackendRegistry::instance().get(frame::kCudaBackendName).is_ok()) {
    benchmark::RegisterBenchmark("BM_Conv2dPrecision_Fp32Strict", BM_Conv2dPrecision_Fp32Strict);
    benchmark::RegisterBenchmark("BM_Conv2dPrecision_Fp32Tf32", BM_Conv2dPrecision_Fp32Tf32);
    benchmark::RegisterBenchmark("BM_Conv2dPrecision_Fp16", BM_Conv2dPrecision_Fp16);
    benchmark::RegisterBenchmark("BM_Conv2dPrecision_Bf16", BM_Conv2dPrecision_Bf16);
  }

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
