# C++ 编码规范

> **强制等级**:规范(MUST)
> **相关铁律**:#1 编译优先 / #4 语言策略
> **面向读者**:实现代码的工程师
> **最后更新**:2026-07-04

本文档是仓库全部 C++ 代码(`src/`、`include/frame/`、`tests/cpp/`、`examples/`)的编码规范。
`python/src/` 绑定代码同样适用本规范,另受 PY- 系列规则约束;两者冲突处(仅 CPP-020 的
`throw` 例外)以 docs/standards/python-binding.md 为准。

规则条文格式:`【编号】【MUST/MUST NOT/SHOULD】正文。判定方法:<可机械执行的检查>。`

---

## 1. 语言标准

- 【CPP-001】【MUST】全仓库 C++ 标准为 **C++20**(依据 ADR-0004):顶层 CMake 设定
  `CMAKE_CXX_STANDARD 20`、`CMAKE_CXX_STANDARD_REQUIRED ON`、`CMAKE_CXX_EXTENSIONS OFF`;
  CUDA 后端要求 **CUDAToolkit ≥ 12.0** 且 `CMAKE_CUDA_STANDARD 20`(nvcc 自 12.0 起支持
  C++20)。判定方法:grep 顶层 `CMakeLists.txt` 与 `cmake/frame_compiler_flags.cmake`
  中上述变量取值;CI 构建通过。

- C++20 特性(concepts、ranges、`std::span`、`if consteval` 等)可直接使用,无需逐特性
  审批。若某后端工具链对具体特性不兼容,先在对应 docs/backends/*.md 的「已知限制与待查证
  清单」登记,再经 ADR 收窄使用范围;禁止未经 ADR 私自降回 C++17 写法约束他人。

- 【待查证】CANN toolkit 配套宿主编译器(GCC 版本矩阵)对 C++20 的支持范围 —— 来源:
  昇腾社区 hiascend.com 文档中心《CANN 软件安装指南》兼容性章节;登记与跟踪见
  docs/backends/ascend.md。

## 2. 编译期优先(铁律 #1② 的落地)

- 【CPP-010】【MUST NOT】白名单之外禁止出现 `virtual` 关键字。虚函数白名单的**唯一正文**
  = `include/frame/hal/backend.h` 头部注释所载 R1∧R2∧R3 联合判定 + 6 类
  `{Backend, Stream, Event, Allocator, Executable, Pass}`(∪ 各后端实现文件中这 6 类的
  派生类、tests/ 下的 GoogleTest fixture)。本文档只引用、不复制白名单正文;两处不一致时
  以该头部注释为准。白名单外新增 `virtual` 必须先有已接受 ADR(触发条款见
  docs/decisions/README.md)。判定方法:运行 `scripts/check_iron_rules.sh`。

- 【CPP-011】【MUST NOT】禁止 `dynamic_cast` 与 `typeid`;白名单例外:`python/` 绑定层
  (pybind11 依赖 RTTI,因此项目**不**全局 `-fno-rtti`,但绑定层之外禁用 RTTI 设施)。
  判定方法:运行 `scripts/check_iron_rules.sh`。

- 【CPP-012】【MUST】dtype/标量类型差异用模板参数 + `if constexpr` 表达:kernel 入口经
  `frame::dispatch_dtype` 做一次编译期展开(`FRAME_REGISTER_KERNEL(op, backend_string, fn)`
  的键不含 dtype),kernel 内层循环禁止按 dtype/设备做运行时分支(呼应 ARCH-042)。
  编译期常量用 `constexpr`,禁止用宏定义常量(注册宏、分发宏、`FRAME_CHECK` 类宏除外)。
  判定方法:运行 `scripts/check_iron_rules.sh`(kernels/ 目录 dtype 分支检查);
  code review 核对 kernel 模板签名。

- 【CPP-013】【SHOULD】固定候选集合上的多态用 `std::variant` + `std::visit` 或模板静态
  分派表达,而非继承。偏离时在 PR 描述写明理由。判定方法:PR 描述含理由段。

- 【CPP-014】【MUST】模板导致的编译时间问题用显式实例化(`.cpp` 中
  `template class Foo<float>;`)控制;禁止以「退回虚函数」的方式解决(那会违反 CPP-010)。
  判定方法:code review;`scripts/check_iron_rules.sh` 兜底。

## 3. 错误处理

- 【CPP-020】【MUST】核心库(`src/`、`include/frame/`)不使用异常:错误统一以
  `Status` / `Result<T>`(定义于 `include/frame/core/status.h`)返回;`throw` 仅允许
  出现在 `python/` 绑定层(集中于 `translate_status()`,见 PY-030)。不可恢复的程序缺陷
  (违反内部不变量)用 `FRAME_CHECK(cond)`(见 `include/frame/core/macros.h`,失败即
  abort);未实现的桩统一 `return FRAME_UNIMPLEMENTED();`。判定方法:运行
  `scripts/check_iron_rules.sh`(`throw` token 扫描,`python/` 为白名单)。

- 【CPP-021】【SHOULD】`Status` 与 `Result<T>` 类型标注 `[[nodiscard]]`,调用方不得丢弃
  返回值。判定方法:`grep -n 'nodiscard' include/frame/core/status.h` 非空;
  warnings-as-errors 构建通过。

- Status 消息、日志文本一律英文——此为 LANG-005 的规则,判定方法见
  docs/standards/language-policy.md。

## 4. 命名

- 【CPP-040】【MUST】标识符命名符合下表正则。判定方法:`.clang-tidy` 的
  `readability-identifier-naming` 检查 + code review。

| 元素 | 风格 | 正则 | 示例 |
|---|---|---|---|
| 类型(class/struct/enum/using 别名) | PascalCase | `^[A-Z][A-Za-z0-9]*$` | `KernelRegistry` |
| 函数/方法 | snake_case | `^[a-z][a-z0-9_]*$` | `topological_order` |
| 变量/参数 | snake_case | `^[a-z][a-z0-9_]*$` | `input_count` |
| 成员变量 | snake_case 尾下划线 | `^[a-z][a-z0-9_]*_$` | `nodes_` |
| 常量(constexpr/枚举值) | kPascalCase | `^k[A-Z][A-Za-z0-9]*$` | `kMaxRank` |
| 宏 | FRAME_ 前缀大写蛇形 | `^FRAME_[A-Z0-9_]+$` | `FRAME_REGISTER_OP` |
| 命名空间 | 全小写,根为 `frame` | `^[a-z][a-z0-9_]*$` | `frame::hal` |
| 源文件名 | snake_case | `^[a-z0-9_]+\.(h\|cpp\|cu)$` | `op_registry.h` |
| 测试文件名 | test_ 前缀 | `^test_[a-z0-9_]+\.cpp$`(BUILD-012) | `test_graph.cpp` |

## 5. 头文件纪律

- 【CPP-050】【MUST】头文件使用 `#pragma once`;头文件自包含(单独编译通过);禁止在
  头文件中 `using namespace`。判定方法:`grep -rn 'using namespace' include/` 为空;
  `tests/cpp/test_headers_compile.cpp` 编译通过(自包含验证)。

- include 顺序的**唯一事实来源** = 仓库根 `.clang-format` 的 `IncludeCategories` 配置;
  以下文字为其分组的照抄说明,两者不一致时以 `.clang-format` 为准并同 PR 修正本节:
  ① 对应头(`foo.cpp` 的 `foo.h`)→ ② C 系统头 → ③ C++ 标准库 → ④ 第三方库(含后端
  SDK)→ ⑤ 项目内 `frame/...`。判定方法:`clang-format --dry-run --Werror`(缺工具时
  由 CI 执行,见 CPP-030)。

- 【CPP-051】【SHOULD】头文件中能前向声明就不 include。偏离无需说明(此条为降低编译
  时间的指南)。判定方法:code review。

## 6. 所有权与内存

- 【CPP-060】【MUST】拥有权用 `std::unique_ptr` 表达;裸指针与引用仅表示非拥有借用;
  禁止裸 `new`/`delete`(placement new 必须在本节白名单登记后使用,当前白名单为空)。
  判定方法:`grep -rnE '\bnew \|\bdelete ' src/ include/` 输出经 code-reviewer 逐条核对,
  命中且不在白名单即打回。

- 【CPP-061】【MUST】跨接口传连续数组用 `std::span`(C++20 基线下直接可用,不设任何
  自研替代类型);新增接口禁止出现 `(T* ptr, size_t len)` 参数对(与第三方 SDK 交互的
  边界层除外)。判定方法:code review 检查新增函数签名。

## 7. 工具链强制

- 格式:仓库根 `.clang-format`(Google style 基础,`ColumnLimit: 100` 等改项以该文件为
  唯一事实来源)。
- 静态检查:仓库根 `.clang-tidy`,启用 `modernize-*`、`performance-*`、`bugprone-*`、
  `readability-identifier-naming`(命名配置按第 4 章)。
- 【CPP-030】【MUST】CI 中 clang-format 与 clang-tidy 通过是合入门槛;本地工具缺失时
  `scripts/ci_check.sh` 打印 SKIP 不失败,CI runner 上真实执行。本地安装与 CI 逐字
  一致的固定版本:`pipx install clang-format==22.1.5` 与
  `pipx install clang-tidy==22.1.7`(无 pipx 时可用 `python3 -m venv` + pip 装同名
  同版本包);版本号与 `.github/workflows/ci.yml` 保持一致,升级须同 PR 修改两处。
  判定方法:运行 `scripts/ci_check.sh`;CI 状态为绿。

## 8. 设备代码例外(CUDA / SYCL / Ascend C)

设备端代码允许以下偏离,**偏离仅限本清单,其余条款照常适用**(命名、编译期优先、
所有权规则不豁免):

1. 设备端函数可使用后端原语(如 `__syncthreads`、`sycl::nd_item` 方法)与 SDK 向量
   类型;SDK 派生标识符不受第 4 章正则约束。
2. 设备端可用的标准库为受限子集,以各后端文档(docs/backends/*.md 第 8 章「已知限制
   与待查证清单」)登记为准;禁止在正文外散落记录。
3. 设备端无异常与 RTTI(与 CPP-011/020 天然一致);`FRAME_CHECK` 在设备端不可用,
   设备端断言用后端机制表达,错误经宿主侧转换为 `Status` 上报(如 CUDA 的
   `CUDA_CHECK` 宏模式,见 docs/backends/cuda.md)。
4. `.cu` 等设备源文件的 include 顺序中,SDK 头归第三方组(第 5 章分组 ④)。

## 9. TODO 约定(全仓库强制)

- 【CPP-070】【MUST】未实现处的 TODO 统一格式:
  `TODO(FRAME-{IMPL|DESIGN|TEST|DOC|PERF|DEP}): 说明。参考:<路径>。完成判据:<可判定条件>`;
  禁止裸 `TODO` / `FIXME`。判定方法:运行 `scripts/check_iron_rules.sh`(TODO 标签格式
  检查)。
