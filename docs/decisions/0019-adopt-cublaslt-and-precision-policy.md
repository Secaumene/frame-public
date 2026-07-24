# ADR-0019:采纳 cublasLt 统一 matmul 路径并确立精度策略(allow_tf32)

- 状态:已接受
- 日期:2026-07-18
- 关联铁律:#1 #3 #5
- 关联规则:BUILD-011、ARCH-041、REUSE-010/012/013、docs/backends/cuda.md 第 3/8 章

## 背景

M19(v1.2 性能基座,spec 见 docs/plan/v1_2-v1_3-networks-design.md 第 5 节)要求
NVIDIA Tensor Core 三精度路径。现状(2026-07-17 勘察):cuda matmul 的 fp32 走
cublasSgemm(严格 FP32,不经 Tensor Core),fp16/bf16 走 cublasGemmEx
COMPUTE_32F;fp32 图没有降精度加速入口;CompileOptions 仅 {opt_level}。cublasLt
为 CUDA SDK 自带扩展 API,提供启发式算法选择与 epilogue 扩展面(M21 前瞻)。

## 决策

1. matmul CUDA kernel 统一升级 cublasLt(cublasLtMatmul + heuristic 首选算法 +
   workspace),替换 cublasSgemm/cublasGemmEx 双轨;数据类型映射 fp32→32F、
   fp16→16F、bf16→16BF,计算类型恒 COMPUTE_32F(fp32 累加)。
2. `CompileOptions` 新增 `bool allow_tf32 = false`,语义为**图级 fp32 降精度数学
   模式许可**(允许而非强制;厂商中立):开启时后端可对 fp32 计算采用 TF32 级
   (尾数 ≥10 位、fp32 指数域、fp32 累加)数学模式,无等价模式的后端维持严格
   fp32。matmul 为首个消费者;M21 conv 沿用同一开关,禁止新增第二个精度旋钮。
   默认 false:fp32 参考语义不悄然改变。fp16/bf16 路径与本开关无关。
3. allow_tf32 必须计入 `CompileOptions::fingerprint()`(缓存键分量纪律,M4 决议
   延续:字段变更必须同步 fingerprint)。
4. BUILD-011 容差表增行 `fp32(allow_tf32)`,rtol 1e-3 / atol 1e-4,已定案
   (2026-07-18 本机 K=512 matmul 实测回填:最大相对偏差 1.58e-4、最大绝对
   偏差 2.93e-3,rtol 项覆盖留 6 倍裕度;TF32 尾数 10 位与 fp16 相同,优势仅在
   fp32 指数域与 fp32 累加)。载体:tests/cpp/common/tolerance.h 具名入口 `tf32_tolerance()`,
   与表行、fingerprint 测试同批落地。cpu 参考恒为严格 fp32(ARCH-041 不变);
   TF32 开启时的 cpu-cuda 比对用本档。
5. 引入方式(REUSE-010/012):新增链接目标 `CUDA::cublasLt`(CUDAToolkit 自带
   组件,随 SDK 版本锁定,无新 find_package/FetchContent)与 `#include <cublasLt.h>`;
   接线落 src/backends/cuda/CMakeLists.txt。

判定方法:`grep -rn "cublasSgemm\|cublasGemmEx" src/backends/cuda/` 命中为零;
测试 FingerprintDistinguishesAllowTf32 与默认值断言在仓且过;BUILD-011 表含
fp32(allow_tf32) 行;防洗白:`grep -rln "tf32_tolerance" tests/` 命中文件必须同时
含 `allow_tf32 = true`(code-reviewer 逐处核对该档不用于严格 fp32 比较)。

## 备选方案

- 维持 GemmEx + COMPUTE_32F_FAST_TF32:改动最小;无 heuristic/epilogue 扩展面,
  M21 epilogue(bias/激活)融合前瞻受限——否决。
- TF32 默认开启:fp32 参考语义悄变,与 BUILD-011 现表冲突——否决。
- 精度开关走环境变量:绕过 CompileOptions 与缓存键,同键不同数值污染缓存——否决。
- 字段用 `enum PrecisionMode{strict,tf32}`:更耐扩展;当前仅二值,按 YAGNI 取 bool,多模式需求出现时再以新 ADR 迁移——暂不采。

## 后果

- 正面:fp32 图可显式获得 Tensor Core 加速;三精度单一代码路径;为 M21
  epilogue 融合备好扩展面。
- 负面/代价:cublasLt 调用面更繁(desc/preference/workspace 生命周期);heuristic
  结果设备相关,bench 数字仅代表本机(RTX 5070 Ti,CC 12.0)。
- 跟进:cuda.md 第 8 章实测记录更新(第 3 章复用清单已列 cuBLASLt,无需增列);bench_matmul
  增精度对照;tf32 容差终值实测回填;sm_120 各精度组合无逐条官方矩阵,以本机
  一致性套件 + bench 实测为最终判定(BE-000,cuBLAS 13.3 文档核实 2026-07-17)。
