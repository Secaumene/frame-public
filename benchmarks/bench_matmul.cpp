// =============================================================================
// benchmarks/bench_matmul.cpp —— matmul+add+relu 编译执行基准(BENCH-020 登记)。
//
// 测量对象:runtime::compile 编译执行主路径(ARCH-010,BENCH-011)—— matmul
// ([N,N]x[N,N]) -> add(+bias[N,N]) -> relu 三算子图。编译(runtime::compile)
// 只做一次(不计时,setup 阶段);循环内仅执行 Executable::run(计时区间)。
// shape/dtype 档位:float32 方阵,N ∈ {64, 256, 512}(state.range(0))。
// 后端档位:cpu 恒定注册;cuda 后端在运行期探测可用(BackendRegistry 已注册
// 该键)时同源注册同一批 shape,不可用则跳过(见文件尾 main())。
// 对照组:本批不提供 eager 对照组(如 BM_matmul_eager_reference)——eager
// 唯一合法入口 Backend::launch 的调用点白名单(ARCH-011,
// scripts/check_iron_rules.sh 检查 7)当前仅覆盖 src/runtime/,benchmarks/
// 未列入其中;是否扩充该白名单留待另案讨论,本批只交付编译执行主路径。
//
// 精度对照基准族(BM_MatmulPrecision_*,M19 Task 7,与上述 BM_MatmulAddRelu_*
// 并存互不影响):测量对象同为 runtime::compile 编译执行主路径,但图形状精简为
// 纯单 matmul([N,N]x[N,N]) 方阵图(不含 add/relu),仅 cuda 后端。四变体——
// Fp32Strict(默认 CompileOptions,严格 fp32)、Fp32Tf32(显式 allow_tf32=true,
// ADR-0019)、Fp16、Bf16;shape 档位 N ∈ {256, 512, 1024}。对照目的 = ADR-0019
// Tensor Core 收益落档:四变体同形状对比,量化 TF32/fp16/bf16 相对严格 fp32 的
// 加速比。
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
#include <vector>

#include <frame/frame.h>

namespace {

// 构造 [n,n] TensorType(device 显式传入,支持 cpu/cuda 两档后端;dtype 默认
// float32,精度对照基准族显式传入 fp32/fp16/bf16——REUSE-002,BM_MatmulAddRelu_*
// 既有调用点省略该实参,行为与改动前完全一致)。
frame::ir::TensorType MakeSquareTensorType(int64_t n, frame::Device device,
                                           frame::DType dtype = frame::DType::of<float>()) {
  frame::ir::TensorType type;
  type.dtype = dtype;
  type.shape = frame::Shape({n, n});
  type.device = device;
  return type;
}

// 建 matmul(x,w) -> add(+bias) -> relu 三算子图,输入按序 [x, w, bias]
// (device 决定图归属后端,须与 runtime::compile 的 backend_name 一致,见
// include/frame/runtime/compile.h 头注释)。手法仿
// tests/cpp/compiler/mlp_forward_graph_helper.h 与 examples/02_graph_compile/
// main.cpp 的构图流程:本文件与二者分属不同 target(benchmarks/ 不链接
// gtest,不能直接 include 该 tests-only helper 头),故为同构图形的独立实现。
frame::Result<frame::ir::Graph> BuildMatmulAddReluGraph(int64_t n, frame::Device device) {
  frame::ir::Graph graph("bench_matmul_add_relu");

  const frame::Result<frame::ir::Value*> x = graph.add_graph_input(MakeSquareTensorType(n, device));
  if (!x.is_ok()) return x.status();
  const frame::Result<frame::ir::Value*> w = graph.add_graph_input(MakeSquareTensorType(n, device));
  if (!w.is_ok()) return w.status();
  const frame::Result<frame::ir::Value*> bias =
      graph.add_graph_input(MakeSquareTensorType(n, device));
  if (!bias.is_ok()) return bias.status();

  const frame::Result<frame::ir::Node*> matmul_node =
      graph.create_node("matmul", {x.value(), w.value()}, {MakeSquareTensorType(n, device)});
  if (!matmul_node.is_ok()) return matmul_node.status();

  const frame::Result<frame::ir::Node*> add_node = graph.create_node(
      "add", {matmul_node.value()->output(0), bias.value()}, {MakeSquareTensorType(n, device)});
  if (!add_node.is_ok()) return add_node.status();

  const frame::Result<frame::ir::Node*> relu_node =
      graph.create_node("relu", {add_node.value()->output(0)}, {MakeSquareTensorType(n, device)});
  if (!relu_node.is_ok()) return relu_node.status();

  const frame::Status mark_status = graph.mark_output(relu_node.value(), 0);
  if (!mark_status.is_ok()) return mark_status;
  return graph;
}

// 建纯 matmul(x,w) 单算子图([n,n]x[n,n]->[n,n]),dtype 由调用方显式指定
// (精度对照基准族专用,BM_MatmulAddRelu_* 沿用上方三算子图不变)。手法仿
// tests/cpp/backends/test_cuda_backend.cpp::BuildMatmulOnlyGraph,本文件与该
// 测试文件不同 target(理由同 BuildMatmulAddReluGraph 头注释),故为独立实现。
frame::Result<frame::ir::Graph> BuildMatmulOnlyGraph(int64_t n, frame::Device device,
                                                     frame::DType dtype) {
  frame::ir::Graph graph("bench_matmul_precision");

  const frame::Result<frame::ir::Value*> x =
      graph.add_graph_input(MakeSquareTensorType(n, device, dtype));
  if (!x.is_ok()) return x.status();
  const frame::Result<frame::ir::Value*> w =
      graph.add_graph_input(MakeSquareTensorType(n, device, dtype));
  if (!w.is_ok()) return w.status();

  const frame::Result<frame::ir::Node*> matmul_node =
      graph.create_node("matmul", {x.value(), w.value()}, {MakeSquareTensorType(n, device, dtype)});
  if (!matmul_node.is_ok()) return matmul_node.status();

  const frame::Status mark_status = graph.mark_output(matmul_node.value(), 0);
  if (!mark_status.is_ok()) return mark_status;
  return graph;
}

// 按 [n,n] 形状、dtype T、给定填充值 fill_value 构造一个位于 device 上的输入
// 张量:基准只关心执行耗时,不关心数值内容;固定值避免引入随机数生成器造成的
// 额外抖动。cpu 后端 allocator 分配的是主机可写内存,可直接经主机指针写入;
// 非 cpu 后端(如 cuda)allocator 分配的是设备内存,不可经主机指针直接写入
// (直接写会 SIGSEGV)——须先在 cpu 侧构造一份同形张量,再经 backend.copy()
// (H2D,stream=nullptr 走同步拷贝,setup 阶段不计时,cuda_backend.cpp 头注释
// 明确 stream 为空即同步语义)搬运,手法仿
// tests/cpp/backends/test_cuda_backend.cpp::CopyToDevice(本文件不链接
// gtest,不能直接复用该 fixture 私有方法,故为同构逻辑的独立实现)。dtype 泛型
// 化为精度对照基准族(fp16/bf16)与既有 float 版共用本实现(REUSE-002)。
template <frame::ScalarType T>
frame::Result<frame::Tensor> MakeFilledTensorT(int64_t n, T fill_value, frame::Device device,
                                               frame::hal::Backend& backend,
                                               frame::hal::Allocator& allocator) {
  const frame::Shape shape({n, n});
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

// float32、全 1.0F 填充版本:BM_MatmulAddRelu_* 既有调用点不变,行为与改动前
// 完全一致(转调 MakeFilledTensorT<float>,REUSE-002)。
frame::Result<frame::Tensor> MakeFilledTensor(int64_t n, frame::Device device,
                                              frame::hal::Backend& backend,
                                              frame::hal::Allocator& allocator) {
  return MakeFilledTensorT<float>(n, 1.0F, device, backend, allocator);
}

// 按运行时 frame::DType 分派到 MakeFilledTensorT<T>(唯一允许的 dtype 运行时
// 分派点是 frame::dispatch_dtype,见 include/frame/core/dtype.h 头注释;精度
// 对照基准族仅消费 fp32/fp16/bf16 三档,其余 dtype 落 default 分支返回错误,
// 该分支在本基准族实际运行时不可达)。
frame::Result<frame::Tensor> MakeDTypePrecisionFilledTensor(int64_t n, frame::DType dtype,
                                                            frame::Device device,
                                                            frame::hal::Backend& backend,
                                                            frame::hal::Allocator& allocator) {
  return frame::dispatch_dtype(dtype.code(), [&]<typename T>() -> frame::Result<frame::Tensor> {
    if constexpr (std::is_same_v<T, float>) {
      return MakeFilledTensorT<T>(n, 1.0F, device, backend, allocator);
    } else if constexpr (std::is_same_v<T, frame::float16_t>) {
      return MakeFilledTensorT<T>(n, frame::float_to_float16(1.0F), device, backend, allocator);
    } else if constexpr (std::is_same_v<T, frame::bfloat16_t>) {
      return MakeFilledTensorT<T>(n, frame::float_to_bfloat16(1.0F), device, backend, allocator);
    } else {
      return frame::Status::make(frame::ErrorCode::kInvalidArgument,
                                 "unsupported dtype for precision matmul benchmark");
    }
  });
}

// 基准主体:对给定后端一次性建图 + runtime::compile(setup,不计时,
// BENCH-011);循环内仅执行 run_with_allocated_outputs(计时区间)。cpu/cuda
// 两档基准共用同一份实现(REUSE-002),backend_name 是唯一差异点。
void RunMatmulAddReluBenchmark(benchmark::State& state, std::string_view backend_name) {
  const int64_t n = state.range(0);
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

  const frame::Result<frame::ir::Graph> graph = BuildMatmulAddReluGraph(n, device);
  if (!graph.is_ok()) {
    state.SkipWithError(std::string(graph.status().message()));
    return;
  }

  // 主路径:runtime::compile 一次(setup,不计时,BENCH-011)。
  const frame::Result<std::shared_ptr<frame::hal::Executable>> executable =
      frame::runtime::compile(graph.value(), backend_name, frame::hal::CompileOptions{});
  if (!executable.is_ok()) {
    state.SkipWithError(std::string(executable.status().message()));
    return;
  }

  const frame::Result<frame::Tensor> x_input = MakeFilledTensor(n, device, *backend, *allocator);
  if (!x_input.is_ok()) {
    state.SkipWithError(std::string(x_input.status().message()));
    return;
  }
  const frame::Result<frame::Tensor> w_input = MakeFilledTensor(n, device, *backend, *allocator);
  if (!w_input.is_ok()) {
    state.SkipWithError(std::string(w_input.status().message()));
    return;
  }
  const frame::Result<frame::Tensor> bias_input = MakeFilledTensor(n, device, *backend, *allocator);
  if (!bias_input.is_ok()) {
    state.SkipWithError(std::string(bias_input.status().message()));
    return;
  }
  const std::vector<frame::Tensor> inputs{x_input.value(), w_input.value(), bias_input.value()};

  for (auto _ : state) {
    // 非 const 局部变量:绑定 DoNotOptimize(Tp&) 非弃用重载(const Tp& 重载对
    // 非平凡可拷贝类型已弃用,见 benchmark.h 对应注释)。
    frame::Result<std::vector<frame::Tensor>> outputs =
        frame::runtime::run_with_allocated_outputs(*executable.value(), backend_name, inputs);
    benchmark::DoNotOptimize(outputs);
  }
}

void BM_MatmulAddRelu_Cpu(benchmark::State& state) {
  RunMatmulAddReluBenchmark(state, frame::kCpuBackendName);
}
// BENCHMARK() 宏展开出的静态注册对象由 Google Benchmark 内部实现构造(可能
// 分配内存),tidy 无法证明其构造函数不抛;第三方宏产出的既有模式,非本文件
// 引入的风险。
// NOLINTNEXTLINE(bugprone-throwing-static-initialization)
BENCHMARK(BM_MatmulAddRelu_Cpu)->Arg(64)->Arg(256)->Arg(512);

void BM_MatmulAddRelu_Cuda(benchmark::State& state) {
  RunMatmulAddReluBenchmark(state, frame::kCudaBackendName);
}

// 精度对照基准体:纯单 matmul 方阵图,按 (dtype, compile_options) 变体对比编译
// 执行耗时——对照目的 = ADR-0019 Tensor Core 收益落档。仅 cuda 后端(fp16/bf16
// 与 TF32 数学模式均为 cuda 专属,不设 cpu 变体)。组织方式复用
// RunMatmulAddReluBenchmark(REUSE-002):runtime::compile 一次(setup,不计时,
// BENCH-011),循环内仅执行 run_with_allocated_outputs(计时区间)。
void RunPrecisionMatmulBenchmark(benchmark::State& state, frame::DType dtype,
                                 const frame::hal::CompileOptions& compile_options) {
  const int64_t n = state.range(0);
  const frame::Device device{frame::kCudaBackendName, 0};

  const frame::Result<frame::hal::Backend*> backend_result =
      frame::hal::BackendRegistry::instance().get(frame::kCudaBackendName);
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

  const frame::Result<frame::ir::Graph> graph = BuildMatmulOnlyGraph(n, device, dtype);
  if (!graph.is_ok()) {
    state.SkipWithError(std::string(graph.status().message()));
    return;
  }

  // 主路径:runtime::compile 一次(setup,不计时,BENCH-011)。compile_options
  // 由调用方显式传入——fp32 严格档默认构造,fp32+TF32 档显式 allow_tf32=true
  // (ADR-0019);fp16/bf16 档与 allow_tf32 无关,沿用默认构造。
  const frame::Result<std::shared_ptr<frame::hal::Executable>> executable =
      frame::runtime::compile(graph.value(), frame::kCudaBackendName, compile_options);
  if (!executable.is_ok()) {
    state.SkipWithError(std::string(executable.status().message()));
    return;
  }

  const frame::Result<frame::Tensor> x_input =
      MakeDTypePrecisionFilledTensor(n, dtype, device, *backend, *allocator);
  if (!x_input.is_ok()) {
    state.SkipWithError(std::string(x_input.status().message()));
    return;
  }
  const frame::Result<frame::Tensor> w_input =
      MakeDTypePrecisionFilledTensor(n, dtype, device, *backend, *allocator);
  if (!w_input.is_ok()) {
    state.SkipWithError(std::string(w_input.status().message()));
    return;
  }
  const std::vector<frame::Tensor> inputs{x_input.value(), w_input.value()};

  for (auto _ : state) {
    // 非 const 局部变量:绑定 DoNotOptimize(Tp&) 非弃用重载(const Tp& 重载对
    // 非平凡可拷贝类型已弃用,见 benchmark.h 对应注释)。
    frame::Result<std::vector<frame::Tensor>> outputs = frame::runtime::run_with_allocated_outputs(
        *executable.value(), frame::kCudaBackendName, inputs);
    benchmark::DoNotOptimize(outputs);
  }
}

// fp32 严格档:默认 CompileOptions(allow_tf32=false),cublasLt 严格 fp32 数学
// 模式,不经 Tensor Core。
void BM_MatmulPrecision_Fp32Strict(benchmark::State& state) {
  RunPrecisionMatmulBenchmark(state, frame::DType::of<float>(), frame::hal::CompileOptions{});
}

// fp32+TF32 档:显式 allow_tf32=true(ADR-0019),cublasLt 改用
// CUBLAS_COMPUTE_32F_FAST_TF32,经 Tensor Core 但尾数截断到 TF32 精度。
void BM_MatmulPrecision_Fp32Tf32(benchmark::State& state) {
  frame::hal::CompileOptions options;
  options.allow_tf32 = true;
  RunPrecisionMatmulBenchmark(state, frame::DType::of<float>(), options);
}

// fp16 档:全程半精度存储与计算,经 Tensor Core(与 allow_tf32 开关无关)。
void BM_MatmulPrecision_Fp16(benchmark::State& state) {
  RunPrecisionMatmulBenchmark(state, frame::DType::of<frame::float16_t>(),
                              frame::hal::CompileOptions{});
}

// bf16 档:全程 bfloat16 存储与计算,经 Tensor Core(与 allow_tf32 开关无关)。
void BM_MatmulPrecision_Bf16(benchmark::State& state) {
  RunPrecisionMatmulBenchmark(state, frame::DType::of<frame::bfloat16_t>(),
                              frame::hal::CompileOptions{});
}

}  // namespace

// 自定义 main(不链接 benchmark::benchmark_main):cuda 档基准须在注册前完成
// 运行期探测(BackendRegistry::get,不可用则跳过注册)——cuda 后端在
// cpu-only/未装 CUDA Toolkit 的机器上不存在,若像 cpu 档一样经 BENCHMARK()
// 宏做静态注册,会在这些机器上产出一个必然 SkipWithError 的基准项而非"不存在
// 该基准",故改为运行期条件注册。cpu 档经 BENCHMARK() 宏在静态初始化期完成
// 注册,与本处的运行期注册互不干扰(二者均先于 RunSpecifiedBenchmarks 完成)。
int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;

  if (frame::hal::BackendRegistry::instance().get(frame::kCudaBackendName).is_ok()) {
    benchmark::RegisterBenchmark("BM_MatmulAddRelu_Cuda", BM_MatmulAddRelu_Cuda)
        ->Arg(64)
        ->Arg(256)
        ->Arg(512);

    // 精度对照基准族(M19 Task 7):同一条件注册口径,cuda 不可用时随上一分支
    // 一并跳过。
    benchmark::RegisterBenchmark("BM_MatmulPrecision_Fp32Strict", BM_MatmulPrecision_Fp32Strict)
        ->Arg(256)
        ->Arg(512)
        ->Arg(1024);
    benchmark::RegisterBenchmark("BM_MatmulPrecision_Fp32Tf32", BM_MatmulPrecision_Fp32Tf32)
        ->Arg(256)
        ->Arg(512)
        ->Arg(1024);
    benchmark::RegisterBenchmark("BM_MatmulPrecision_Fp16", BM_MatmulPrecision_Fp16)
        ->Arg(256)
        ->Arg(512)
        ->Arg(1024);
    benchmark::RegisterBenchmark("BM_MatmulPrecision_Bf16", BM_MatmulPrecision_Bf16)
        ->Arg(256)
        ->Arg(512)
        ->Arg(1024);
  }

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
