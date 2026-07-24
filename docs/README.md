# Frame 文档导航

> **强制等级**:参考(INFO)
> **相关铁律**:#1 编译优先 / #2 语言支持 / #3 后端矩阵 / #4 语言策略 / #5 两级复用
> **面向读者**:实现代码的工程师
> **最后更新**:2026-07-24(GPLv3+ 与官方纯源码发布入口)

本页只做路由与全局约定,不承载规则条文。目标:进入 `docs/` 后 30 秒内找到该读哪篇。规则条文的单一事实来源在各专门文档中,引用时使用规则编号(如 CPP-010),不复制正文。

> **使用 Frame？** 请从[官方使用手册](../help/README.md)开始。手册以 C++20
> 为主线，并链接 `examples/` 下持续构建的对应源码；`help/` 是面向用户的非规范性
> 指南，本 `docs/` 目录仍是规范与架构的单一事实来源。

## 1. 按任务找文档

| 我要做什么 | 必读 | 补充 |
|---|---|---|
| 写 C++ 代码 | `standards/cpp-coding.md` | `architecture/overview.md`(先弄清代码属于哪层) |
| 新增一个算子 | `architecture/operator-system.md` | 目标后端 `backends/*.md` |
| 新增一个编译 pass | `architecture/compiler-passes.md` | `architecture/ir-design.md` |
| 接入一个新后端 | `architecture/backend-hal.md` | `backends/README.md` |
| 引入第三方库 | `standards/reuse-policy.md` | `decisions/README.md`(REUSE-010:先有已接受 ADR) |
| 理解执行模型(编译 vs eager) | `architecture/execution-model.md` | `decisions/`(ADR-0001) |
| 理解/修改图 IR | `architecture/ir-design.md` | `architecture/compiler-passes.md` |
| 写 Python 绑定 | `standards/python-binding.md` | `standards/cpp-coding.md` |
| 写 commit / PR | `standards/language-policy.md` | — |
| 构建、写测试 | `standards/build-and-test.md` | 对应后端文档的「测试要求」章 |
| 发布公开源码 | `standards/build-and-test.md` 第 11 章 | `decisions/0023-adopt-gpl3-source-only-release.md` |
| 做重大技术决策 | `decisions/README.md` + `decisions/adr-template.md` | — |

## 2. 阅读顺序建议(首次进入仓库)

`architecture/overview.md` → `architecture/execution-model.md` → `standards/cpp-coding.md` → `standards/reuse-policy.md`;之后按第 1 节的表格按需查阅。

## 3. 规则编号体系

规则条文统一格式:

```
【编号】【MUST/MUST NOT/SHOULD】规则正文。判定方法:<可机械执行的检查>。
```

| 前缀 | 所属文档 | 示例 |
|---|---|---|
| ARCH- | architecture/ 各文档 | ARCH-001 |
| CPP- | standards/cpp-coding.md | CPP-001 |
| PY- | standards/python-binding.md | PY-001 |
| LANG- | standards/language-policy.md | LANG-001 |
| REUSE- | standards/reuse-policy.md | REUSE-001 |
| BUILD- | standards/build-and-test.md | BUILD-001 |
| BE- | backends/ 各文档 | BE-CUDA-001 |

措辞纪律:

- **MUST / MUST NOT**:违反即 code review 打回,不容裁量。
- **SHOULD**:偏离时必须在 PR 描述中写明理由。
- 禁止「尽量」「优雅」「合理」等不可判定措辞;凡不可判定的意图,必须转写为判定方法或降级为 INFO。

## 4. 全局写作约定(适用于 docs/ 下所有文档)

- **头部元数据块**:每个文档第一屏必须给出「强制等级 / 相关铁律 / 面向读者 / 最后更新」四行(格式见本页页首)。
- **语言策略**:文档正文中文;标识符、文件名、API、日志与错误消息纯英文;专有名词(CUDA、SYCL、pass、kernel 等)保留英文原文(LANG-006)。
- **待查证标注**:不确定的第三方 API、版本号、行为,一律写 `【待查证】<问题> —— 来源:<官方文档名/URL>`(BE-000),禁止凭记忆编造。
- **TODO 格式**:`TODO(FRAME-{IMPL|DESIGN|TEST|DOC|PERF|DEP}): 说明。参考:<路径>。完成判据:<可判定条件>`;禁止裸 TODO/FIXME。判定方法:运行 `scripts/check_iron_rules.sh`(TODO 标签格式检查)。
- **跨文档引用**:引用规则用编号(如 ARCH-011、BUILD-011),引用文件用仓库相对路径;禁止「见第 N 节」式脆弱引用跨文档使用。

## 5. 文件清单(19 个)

```
docs/
├─ README.md                      本页:导航与全局约定
├─ architecture/                  架构规范(ARCH-)
│  ├─ overview.md                 分层架构总览与依赖铁则
│  ├─ execution-model.md          执行模型:编译优先与 eager 逃生舱
│  ├─ ir-design.md                图 IR 设计与验证器
│  ├─ backend-hal.md              后端 HAL 接口规范(扁平对象模型)
│  ├─ operator-system.md          算子注册、分发与自定义算子扩展
│  └─ compiler-passes.md          pass 管线与自定义 pass 接口
├─ standards/
│  ├─ cpp-coding.md               C++ 编码规范(CPP-)
│  ├─ python-binding.md           Python 绑定(pybind11)规范(PY-)
│  ├─ language-policy.md          语言策略与 commit message 规范(LANG-)
│  ├─ reuse-policy.md             两级复用规范与第三方库准入清单(REUSE-)
│  └─ build-and-test.md           构建与测试规范(BUILD-)
├─ backends/
│  ├─ README.md                   后端矩阵总览 + 统一九章文档模板(BE-)
│  ├─ cuda.md                     CUDA 后端指南
│  ├─ intel-gpu.md                Intel GPU(SYCL/oneAPI)后端指南
│  ├─ intel-npu.md                Intel NPU(OpenVINO)后端指南
│  └─ ascend.md                   昇腾 NPU(CANN)后端指南
└─ decisions/
   ├─ README.md                   ADR 说明、触发清单与索引
   └─ adr-template.md             ADR 模板
```
