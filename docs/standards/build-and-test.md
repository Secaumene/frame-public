# 构建与测试规范

> **强制等级**:规范(MUST)
> **相关铁律**:#1 编译优先 / #3 后端矩阵
> **面向读者**:实现代码的工程师
> **最后更新**:2026-07-24(GPLv3+ 官方纯源码发布门禁)

本文档是构建系统(CMake/presets/选项/目标命名)与测试体系(C++/Python/容差/CI 门槛)
的唯一规范;**数值容差表的全仓库唯一来源 = BUILD-011**。

规则条文格式:`【编号】【MUST/MUST NOT/SHOULD】正文。判定方法:<可机械执行的检查>。`

---

## 1. CMake 基线

- CMake ≥ 3.25(`CMakePresets.json` 为 version 6 schema,3.24 无法解析该 preset
  文件,故下限取 3.25);顶层 `CMakeLists.txt` + 模块目录
  `cmake/{frame_options,frame_compiler_flags,frame_dependencies,frame_backend,find_cann}.cmake`
  + `CMakePresets.json`。
- C++/CUDA 标准设定见 CPP-001(C++20;`CMAKE_CUDA_STANDARD 20`、CUDAToolkit ≥ 12.0)。
- CMake 脚本风格:每行一条命令,不在同一行串联多条命令。判定方法:code review。
- include 顺序与代码格式的唯一事实来源是 `.clang-format`(CPP-030、CPP-050),构建
  系统不重复检查。

## 2. Presets(名字即规范)

- 【BUILD-001】【MUST】构建变体一律经 preset 表达与调用(`cmake --preset <name>`、
  `cmake --build --preset <name>`、`ctest --preset <name>`);文档中禁止教人
  手敲 `-D` 长命令。判定方法:
  `grep -rn 'cmake .*[-]D' docs/` 命中为空(检查命令的 `[-]` 字符类写法用于
  避免本条自身被命中;引用 preset 实现细节的说明除外)。

configure preset 清单(每个 configure preset 有配套同名 build/test preset,`wheel` 除外——
它关闭测试,仅有同名 build preset;preset 名与用途以本表为规范,字段级实现见
`CMakePresets.json`):

| preset | 用途 | 关键配置 |
|---|---|---|
| `dev` | 默认开发入口 | Debug;warnings-as-errors;`FRAME_BUILD_TESTS=ON`;`FRAME_BUILD_EXAMPLES=ON`;后端开关 AUTO |
| `dev-asan` | 内存诊断 | 同 `dev` + `FRAME_ENABLE_ASAN=ON` |
| `release` | 优化构建 | Release |
| `cpu-only` | 最小口径;**骨架期验收基准** | 四后端开关全 OFF(cpu 参考后端永远启用,无开关) |
| `cuda` | 强制启用 CUDA 后端 | `FRAME_ENABLE_CUDA=ON`;`FRAME_BUILD_EXAMPLES=ON`(SDK 缺失即配置失败,BUILD-004) |
| `intel-gpu` | 强制启用 Intel GPU 后端 | `FRAME_ENABLE_INTEL_GPU=ON` |
| `intel-npu` | 强制启用 Intel NPU 后端 | `FRAME_ENABLE_INTEL_NPU=ON` |
| `ascend` | 强制启用昇腾后端 | `FRAME_ENABLE_ASCEND=ON` |
| `wheel` | Python wheel 打包(scikit-build-core 使用) | Release;`FRAME_BUILD_PYTHON=ON` |

- 公共父 preset(base)**不指定 generator**,交由 CMake 平台默认。

## 3. 选项与目标命名

三态后端开关(实现于 `cmake/frame_options.cmake`,默认 **AUTO**):

| 取值 | 语义 |
|---|---|
| `AUTO` | 探测到 SDK 即启用该后端;未探测到则打印 STATUS 消息并跳过 |
| `ON` | SDK 必须可用,探测失败即 `FATAL_ERROR`(BUILD-004) |
| `OFF` | 不编译该后端 |

选项清单:`FRAME_ENABLE_{CUDA,INTEL_GPU,INTEL_NPU,ASCEND}`(三态,默认 AUTO);
`FRAME_ENABLE_MLIR`(三态,默认 OFF,依据 ADR-0002,当前骨架不启用);
`FRAME_ENABLE_ASAN`(BOOL);`FRAME_BUILD_{PYTHON,TESTS,EXAMPLES,BENCHMARKS}`(BOOL);
`FRAME_BUILD_SHARED_LIBS`(BOOL,插件化预留);`FRAME_INSTALL_CPP`(BOOL,默认 ON,
安装/导出门控,见第 10 节);`FRAME_BUILD_TOOLS`(BOOL,默认 ON,tools/ 命令行
工具,ADR-0017;wheel 构建显式 OFF)。

- 【BUILD-004】【MUST】后端开关为 ON 而 SDK 探测失败时必须 `FATAL_ERROR`,禁止静默
  跳过或降级。判定方法:在无对应 SDK 的环境执行 `cmake --preset <backend>`,配置必须
  失败并输出含后端名的错误消息。
- 【BUILD-002】【MUST】库目标命名 `frame_core / frame_ir / frame_ops /
  frame_compiler / frame_runtime / frame_frontend / frame_backend_<name>`,别名
  `frame::<name>`;测试可执行按 BUILD-012 命名 `frame_test_<module>`;tools/ 下
  命令行工具可执行命名 `frame_<tool>`(现有:`frame_dslc`,ADR-0017);全部目标名
  匹配 `^frame_[a-z0-9_]+$`,唯一例外是聚合库 `frame`(别名 `frame::frame`,命名由
  顶层 CMakeLists.txt 的 TOP-004 契约固定)。判定方法:
  `grep -rn 'add_library\|add_executable' CMakeLists.txt src/ tests/ tools/` 的目标名
  逐一匹配该正则(聚合库 `frame` 除外)。

## 4. 后端条件编译与隔离

- 后端 SDK 一律 `find_package` 探测(昇腾用 `cmake/find_cann.cmake`);后端源码以
  独立 CMake 目标(`frame_backend_<name>`)隔离,经 `FRAME_REGISTER_BACKEND` 字符串
  注册接入,核心层零改动。
- 【BUILD-003】【MUST NOT】`src/{core,ir,ops,compiler,runtime}` 与 `include/frame/`
  (含 hal/——HAL 是纯接口)中禁止出现 `FRAME_ENABLE_*` 条件编译与任何后端头文件
  include;后端差异一律经 CMake 目标隔离与 HAL 注册机制表达。判定方法:运行
  `scripts/check_iron_rules.sh`(核心层后端隔离检查)。

## 5. 测试布局(单一事实来源)

```text
tests/
├─ cpp/
│  ├─ test_headers_compile.cpp     头文件自包含 + static_assert 编译期测试
│  ├─ common/                      测试公共件(数值容差工具,BUILD-011 唯一载体)
│  ├─ core/                        核心模块单测
│  ├─ ir/                          图 IR 单测(verify/序列化/动态维拒绝)
│  ├─ compiler/                    pass 单测;golden 数据在 compiler/testdata/
│  ├─ ops/                         算子单测(参考实现 vs 后端实现)
│  ├─ backends/                    后端冒烟测试
│  ├─ frontend/                    前端层单测(spec 校验/lowering golden/训练冒烟/emit golden)
│  └─ hal_conformance/             HAL 一致性套件(任意注册后端跑同一组行为测试)
├─ python/                         pytest:绑定与 Python API 的 test_*.py
└─ scripts/                        unittest:机械发布门禁的 test_*.py
```

- 【BUILD-012】【MUST】C++ 测试文件命名 `tests/cpp/<module>/test_*.cpp`(文件名正则
  `^test_[a-z0-9_]+\.cpp$`);测试目标命名 `frame_test_<module>`;ctest label =
  模块名 + 后端名。判定方法:`find tests/cpp -name '*.cpp'` 输出的文件名全部匹配该
  正则。

## 6. C++ 测试规范

- 框架:GoogleTest(`FetchContent` 引入,REUSE 准入表「已批准」档)。
- 【BUILD-010】【MUST】依赖真实硬件的测试在设备缺失时 `GTEST_SKIP()` 并输出含后端名
  与缺失原因的英文消息,禁止 fail。SKIP 政策(M24):**在目标后端可用的环境**(带
  设备的 CI runner 或本地设备)该后端测试不得 SKIP;当前环境缺设备时允许 SKIP,但
  提交/PR 说明必须列出被 SKIP 的测试清单(LANG-011 测试证据段)。判定方法:ctest
  输出的 SKIP 项与 PR 清单一致;带设备环境复跑 `ctest --preset <backend>` 无该后端
  SKIP。
- 【BUILD-011】【MUST】数值对比测试统一使用 `tests/cpp/common/` 的容差工具,禁止
  手写 `EXPECT_NEAR`/自造阈值。默认容差表(**全仓库唯一来源=本条**;后端文档中的
  速查副本以本条为准;后端文档可加严、不可放松):

  | dtype | rtol | atol |
  |---|---|---|
  | fp32 | 1e-5 | 1e-6 |
  | fp16 | 1e-2 | 1e-3 |
  | bf16 | 2e-2 | 2e-3 |

  大规模归约(单个输出元素的累加次数 ≥ 2^20 的 reduction/matmul 类用例)允许**放宽
  一档**(取表中下一行的值;bf16 已是末行,其放宽值须个案给出),且必须在测试代码
  放宽处以中文注释写明依据。
  **fp32(allow_tf32) 档**(ADR-0019 增设;独立于上表主档位序列,不参与「放宽一档」
  的行序推导):rtol 1e-3 / atol 1e-4,已定案(2026-07-18 本机 K=512 matmul 实测
  回填:最大相对偏差 1.58e-4、最大绝对偏差 2.93e-3,合成判据 atol+rtol*|e| 下
  rtol 项覆盖)。近零抵消场景(参考值趋零)不放宽本档,按上文「个案显式构造
  Tolerance + 中文注释依据」口径处理。仅限以 `CompileOptions::allow_tf32 = true` 编译的用例,经
  `tests/cpp/common/` 的具名入口 `tf32_tolerance()` 取用;cpu 参考实现恒为严格
  fp32(ARCH-041 不变),TF32 开启时的 cpu-cuda 比对用本档。判定方法:
  `grep -rln 'tf32_tolerance' tests/cpp/` 命中文件必须同时含 `allow_tf32 = true`;
  code-reviewer 逐处核对该档不用于严格 fp32 比较(防放松洗白)。
  **解析梯度 ≡ 数值微分校验**(autograd,M16 设计增补):数值微分侧(中心差分)
  自带 O(h²) 截断误差,允许对该类比较**放宽一档**;步长 h 按 dtype 选取并随实测
  定案写入测试注释(fp32 建议 1e-3 量级起调);fp16/bf16 的梯度不直接与数值微分
  比较,一律经 fp32 解析参照验证。本款是数值微分场景的唯一放宽授权,其余场景仍按
  上两款执行。判定方法:code-reviewer 检查数值断言调用容差工具且放宽
  处有注释;`grep -rn 'EXPECT_NEAR' tests/cpp/` 命中即打回。
  **代理梯度窄例外(M27)**:`heaviside_surrogate` 的前向是离散阶跃,其注册
  GradientFn 按算子合同有意采用平滑代理 `sigmoid(alpha*x)` 的导数,因此禁止
  将该 GradientFn 与阶跃前向的中心差分比较。该算子的梯度验收必须同时满足:
  ①逐元素等于闭式 `gy*alpha*s*(1-s)`;②同一结果与平滑代理
  `sigmoid(alpha*x)` 的 fp32 中心差分按本条数值微分容差一致;③二阶及更高阶
  继续微分已注册的代理微图。此例外仅限精确 op 名 `heaviside_surrogate`,不得
  扩展到其他算子、不得放宽容差表或省略 GradientFn 数值验收。判定方法:
  `tests/cpp/ops/` 同时存在闭式与平滑代理中心差分用例;code-reviewer 核对未用
  阶跃前向伪造数值梯度。
- pass 的 golden 测试与 fusion 数值等价测试要求见 ARCH-051 / ARCH-052
  (docs/architecture/compiler-passes.md);golden 数据放 `tests/cpp/compiler/testdata/`。

## 7. Python 测试规范

- 【BUILD-020】【MUST】框架 pytest;文件命名 `tests/python/test_*.py`;硬件缺失时
  `pytest.skip`(政策同 BUILD-010,含 M24 条款)。判定方法:
  `find tests/python -name '*.py' ! -name 'conftest.py'` 输出全部匹配 `^test_.*\.py$`;
  pytest 输出的 skip 项列入 PR 说明。
- 【BUILD-021】【MUST】绑定层每个暴露 API 至少 1 个用例。判定方法:对 `.pyi` 导出
  清单(PY-020)与 `tests/python/` 中被引用符号求差集,差集应为空;由 code-reviewer
  在涉及 `python/` 的 PR 上执行。

### 7.1 机械脚本测试

- 【BUILD-022】【MUST】`tests/scripts/test_*.py` 只使用 Python 标准库
  `unittest`，由 `scripts/ci_check.sh` 直接执行，不依赖 pytest 或 CTest。
  发布门禁的每类高风险拒绝路径（节点类型、文本内容、产物后缀、工作流发布、
  许可证与元数据）至少有一个负例。判定方法：运行
  `python3 -m unittest discover -s tests/scripts -p 'test_*.py' -v` 全部通过。

## 8. 合入门槛(CI)

- 【BUILD-030】【MUST】合入前 `scripts/ci_check.sh` 全绿;该脚本是 CI 的唯一入口
  (`.github/workflows/ci.yml` 只调用它),汇总检查:
  1. `scripts/check_language_policy.py`(LANG-001/002/003);
  2. `scripts/check_iron_rules.sh`(CPP-010/011/012/020/070、BUILD-003、
     ARCH-002/011/030);
  3. `tests/scripts/` 单测 + `scripts/check_source_release.py`
     (BUILD-022/050/051/052);
  4. clang-format / clang-tidy(CPP-030;clang-tidy 用 configure 产出的
     `compile_commands.json`,在 configure 之后执行);
  5. 构建 + 测试(骨架期口径 `cpu-only` preset)。

  本地工具缺失时对应检查打印 SKIP 不失败;CI runner 上真实执行。改动到某后端时,该
  后端至少一个真实设备任务通过;无设备时按 BUILD-010 的 M24 政策标注 SKIP 并在 PR
  说明列清单。判定方法:CI 状态为绿 + PR 说明含 SKIP 清单。

## 9. 骨架期特殊规定

- 骨架阶段(仅头文件桩与 `FRAME_UNIMPLEMENTED()` 桩)测试目标允许为空壳或仅编译期
  测试(`tests/cpp/test_headers_compile.cpp` 的 static_assert 全套),但 CMake 结构、
  目标命名(BUILD-002)、preset 命名(BUILD-001)与测试文件命名(BUILD-012)从第一天
  起生效。
- 骨架验收口径:`cmake --preset cpu-only` → `cmake --build --preset cpu-only` →
  `ctest --preset cpu-only`,三步零失败。
- 本地缺 clang-format/clang-tidy 时相关检查 SKIP(`scripts/ci_check.sh`),在 CI 中
  真实执行。
- `FRAME_BUILD_EXAMPLES` 选项默认 OFF;仓库自带的 `dev` 与 `cuda` preset
  显式设为 ON,使官方 C++ 示例参与日常构建,并在 `FRAME_BUILD_TESTS=ON` 时以
  `examples` label 接入 CTest。

## 10. 安装与导出(系统安装)与便利脚本

`cmake --install build/<preset> [--prefix <dir>]` 安装 C++ 静态库、头文件与 CMake
包配置;外部工程经 `find_package(frame REQUIRED)` 消费并链接 `frame::frame`
(消费方 CMake ≥ 3.24——导出目标含 `WHOLE_ARCHIVE` 生成表达式,
`frame-config.cmake` 内置版本校验)。

- 【BUILD-040】【MUST】全部 C++ 安装/导出规则集中于 `cmake/frame_install.cmake`,
  受 `FRAME_INSTALL_CPP`(BOOL,默认 ON)门控;导出包名 `frame`、命名空间
  `frame::`、export set 名 `frame_targets`。判定方法:
  `grep -rn '^[^#]*install(' CMakeLists.txt cmake/ src/ tests/` 仅命中
  `cmake/frame_install.cmake`;`python/CMakeLists.txt` 的 wheel 专用 install 受
  `if(SKBUILD)` 门控。
- 【BUILD-041】【MUST NOT】wheel 与系统安装互不携带对方产物:wheel 不得含 C++
  安装件(静态库/头文件/CMake 包配置),系统安装前缀不得含 Python 扩展(`_core`)
  与第三方(googletest 等)安装件。判定方法:wheel 解包 `unzip -l` 无 `.a` 与
  `include/frame` 条目;`cmake --install build/<preset> --prefix <tmp>` 后 `<tmp>`
  内无 `_core*.so`、无 gtest/gmock 头与库。

便利脚本(`scripts/`,日常开发循环封装;不属门禁检查器,不进 `ci_check.sh`;
原生 preset 命令(第 2 节)仍是权威,脚本行为与本文档不一致时以本文档为准):

| 脚本 | 用途 |
|---|---|
| `scripts/build.sh <preset> [--fresh]` | configure+build(preset 白名单校验,BUILD-001 口径) |
| `scripts/test.sh <preset> [ctest 参数]` | ctest 封装(wheel/bench 无 test preset,提前报错) |
| `scripts/clean.sh <preset>\|all [--yes]` | 删除构建目录(删除前列目录并确认) |
| `scripts/setup_dev.sh` | git 钩子 + `pip install -e .` 一键初始化开发环境 |
| `scripts/install.sh [--preset <name>] [--prefix <dir>]` | 系统安装封装(默认 release,未构建先构建) |
| `scripts/format.sh [--check]` | clang-format 就地应用 / 只读检查(与门禁同语义) |

## 11. 官方纯源码发布

- 【BUILD-050】【MUST】Frame 官方公开出口只包含净化后的纯源码树，以及代码托管
  平台从同一提交自动生成的源码归档。CI 中的编译与测试产物只能临时使用，不得上传
  或附加到 Release。判定方法：运行
  `python3 scripts/check_source_release.py --mode working-tree`，并检查公开 Release
  无人工上传附件。

- 【BUILD-051】【MUST NOT】官方出口不得包含 executable、静态或共享库、object、
  bytecode、中间代码、wheel、conda/系统包、容器镜像、厂商 SDK/runtime/driver，
  尤其不得发布与 CUDA、cuDNN、oneAPI、OpenVINO、CANN 等可选专有 SDK 链接或组合
  的编译产物。判定方法：BUILD-050 检查器拒绝符号链接、gitlink、非 UTF-8 文件、NUL、
  禁止后缀和工作流上传/包索引/容器发布命令，任一命中即失败。

- 【BUILD-052】【MUST】BUILD-050/051 是 Frame 官方维护者的发布流程约束，不是对
  `GPL-3.0-or-later` 下游权利的附加限制；下游分发者自行确认 GPL 与第三方条款。
  Frame 官方若要启用或变更预编译分发，必须先按 ADR-010 第 12 项立新 ADR，明确
  目标产物、渠道、所链接依赖与许可证兼容证据。判定方法：官方工作流出现上传构建
  产物、包索引或容器发布命令而无对应已接受 ADR 时，门禁和 code review 均失败。
