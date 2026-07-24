# ADR-0021:采纳 cuDNN 承载卷积/池化 CUDA 内核(销 ADR-0011 推迟)

- 状态:已接受
- 日期:2026-07-18
- 关联铁律:#3 #5
- 关联规则:REUSE-010/012、ARCH-041、BUILD-011、docs/backends/cuda.md 第 3/8 章、
  ADR-0011(被本条取代)、ADR-0019(allow_tf32 沿用)

## 背景

M21 卷积批次(spec docs/plan/v1_2-v1_3-networks-design.md 第 7 节)新增
conv1d/conv2d/max_pool2d/avg_pool2d,ADR-0011「卷积/池化类算子立项」重评条件
触发。BE-000 核实(2026-07-18 本机):cudnn9-cuda-13-3 = 9.24.0.43 与
CUDA 13.3.73 精确配套;运行时 handle 在 RTX 5070 Ti(CC 12.0)创建成功;
legacy 即时 API(cudnnConvolutionForward/BackwardData/BackwardFilter、
cudnnPoolingForward/Backward)在 9.24 标记 CUDNN_DEPRECATED 但符号全部导出,graph API 并存。

## 决策

1. 采纳 cuDNN 9(本机 9.24.0.43)承载 conv/pool 的 CUDA 前向与全部反向内核;
   ADR-0011 状态改「已取代(由 ADR-0021)」,其禁止探测/绑定的约束随之解除。
2. API 路线:v1 用 legacy 即时 API(算法经 cudnnGetConvolution*Algorithm_v7
   启发式首选 + workspace 查询)。弃用风险以隔离对冲:cuDNN 调用面唯一落点
   src/backends/cuda/kernels/{conv,pool,sequence}.cpp(sequence=M22 增补
   2026-07-19:cudnnSoftmaxForward 同一 legacy 面,入迁移触发符号集)与
   cuda_backend 的 handle 管理;迁移触发条件(可判定)= 未来 cuDNN 版本 nm
   探测不到上述符号导出,届时以新 ADR 迁 graph API。
3. handle 管理镜像 cublasLt 模式(ADR-0019):CudaBackend 惰性单例 + mutex
   guard(acquire_cudnn_handle)。
4. 精度:数据类型映射 fp32/fp16/bf16,计算恒 fp32 累加;fp32 严格 =
   CUDNN_FMA_MATH,`CompileOptions::allow_tf32=true` 时 = CUDNN_DEFAULT_MATH
   (TF32 许可;ADR-0019 单开关纪律,禁止第二旋钮);fp16/bf16 =
   CUDNN_TENSOR_OP_MATH。容差沿用 BUILD-011 各档(含 fp32(allow_tf32) 档)。
5. CPU 侧维持自研直接循环参考实现作数值基准(ARCH-041 正确性路径而非性能
   路径);oneDNN 留 v2.0 议题(spec 第 11 节)。
6. 引入方式(REUSE-012):cuDNN 非 CUDAToolkit find_package 组件,以
   find_path/find_library 在 src/backends/cuda/CMakeLists.txt 内探测系统安装
   (apt cudnn9-cuda-13-3),核心层零感知(铁律 3)。版本选型依据(REUSE-013):
   取 NVIDIA 对 CUDA 13.3 的官方配套发行系(cudnn9-cuda-13-3 打包即配套声明),
   上游支持矩阵与本机运行时核实同录批3计划第 0 节(核实日期 2026-07-18)。

判定方法:`grep -rln "cudnn" src/ | grep -v backends/cuda` 命中为零;cuda
preset 本机 conv/pool 测试零 SKIP;cuda.md 第 3 章 cuDNN 行改「已引入
(ADR-0021)」、第 8 章登记 BE-000 核实记录;README 索引 0011 行状态联动。

## 备选方案

- graph(backend)API 直用:免弃用风险;描述符组装冗长,静态 shape v0 收益
  低——暂不采,迁移条件见决策 2。
- cudnn-frontend header 库:API 友好;新增第三方依赖且随 cuDNN 版本耦合——
  否决(YAGNI)。
- CPU 参考也换 oneDNN:性能非目标,参考实现要求可读可验——否决,留 v2.0。

## 后果

- 正面:conv/pool 训练全链(前向 + 反向 data/filter/bias)走厂商库;三精度与
  allow_tf32 策略与 matmul 同构,无第二套精度语义。
- 负面/代价:legacy API 有大版本移除风险(对冲见决策 2);heuristic 算法选择
  设备相关,bench 数字仅代表本机。
- 跟进:cuda.md 第 3/8 章回写;M23 cuFFT 引入沿用本 BE-000 先例。
