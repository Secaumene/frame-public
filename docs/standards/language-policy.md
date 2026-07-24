# 语言策略与 commit message 规范

> **强制等级**:规范(MUST)
> **相关铁律**:#4 语言策略
> **面向读者**:实现代码的工程师
> **最后更新**:2026-07-23(help/ 非规范性文档纳入中文正文范围)

总原则一句话:**自然语言用中文,程序语言用英文**。展开:沟通/注释/文档/commit 摘要
用中文;标识符/文件名/API 名/日志与错误消息纯英文;专有名词保留英文原文。

本文档同时是 **commit message 规范的唯一权威来源**(第 3 章);其他文档只引用、不复制。

规则条文格式:`【编号】【MUST/MUST NOT/SHOULD】正文。判定方法:<可机械执行的检查>。`

---

## 1. 可判定规则清单

- 【LANG-001】【MUST】代码标识符仅含 `[A-Za-z0-9_]`,不得出现非 ASCII 字符。
  判定方法:运行 `scripts/check_language_policy.py`。

- 【LANG-002】【MUST】文件/目录名匹配 `^[a-z0-9_.-]+$`;生态惯例文件白名单
  (`CMakeLists.txt`、`CMakePresets.json`、`README.md`、`CONTRIBUTING.md`、
  `PULL_REQUEST_TEMPLATE.md`、`Dockerfile` 等)以
  `scripts/check_language_policy.py` 内的白名单常量为唯一事实来源,本文档仅举例。
  判定方法:运行 `scripts/check_language_policy.py`。

- 【LANG-003】【MUST】源码注释的自然语言部分为中文。判定粒度为**注释块**(连续注释行
  为一块),每块至少含一个汉字;白名单块:许可证头、纯 URL 行、纯代码示例、
  `NOLINT`/`NOLINTNEXTLINE` 等工具标记行、TODO 标签行中的英文字段。
  判定方法:运行 `scripts/check_language_policy.py`(块级判定)。

- 【LANG-004】【MUST】`docs/`、`help/` 与 README 正文全中文;对外 API docstring
  中文。英文段仅允许出现在代码块、命令示例与专有名词处。判定方法:code review;
  `docs/` 或 `help/` 变更中出现成段英文正文即打回。

- 【LANG-005】【MUST】日志与 `Status`/异常消息一律英文;Python docstring 与文档中文。
  判定方法:对 `src/ include/ python/src/` 执行
  `grep -rnP '[\x{4e00}-\x{9fff}]' --include='*.h' --include='*.cpp' src/ include/ python/src/`,
  命中行若属于字符串字面量(而非注释)即违规;由 code-reviewer 对 PR diff 执行并
  逐条归类。

- 【LANG-006】【MUST】专有名词(CUDA、SYCL、OpenVINO、CANN、pass、kernel、tensor、
  stream 等)保留英文原文,不强行翻译。判定方法:code review。

## 2. commit message 规范(唯一权威来源)

- 【LANG-010】【MUST】commit 首行格式:`type(scope): 中文摘要`,并满足:
  1. `type ∈ {feat, fix, perf, refactor, docs, test, build, ci, chore, revert}`;
  2. `scope ∈ {core, ir, compiler, runtime, hal, ops, frontend, backend-cpu, backend-cuda,
     backend-intel-gpu, backend-intel-npu, backend-ascend, python, cmake, docs, tests,
     examples, tools, scripts}`(封闭枚举,禁止开放正则;扩充枚举须修改本文档并同步
     钩子中的正则副本;frontend/tools 由 ADR-0017 扩入);
  3. 摘要为中文(至少含一个汉字);
  4. 首行总长 ≤ 72 字符(按字符计,一个汉字计 1 字符);
  5. 冒号为 ASCII `:` 且后接一个空格;
  6. 正文(可选):空行 + 中文正文(动机/方案/影响)+ 空行 + `Refs: #issue`;
  7. 钩子放行例外:以 `Merge ` 开头的自动生成合并提交、以 `Revert ` 开头的
     `git revert` 自动生成提交,以及 `fixup!`/`squash!` 前缀的临时提交(手写回滚
     提交仍须用 `revert(scope): 中文摘要` 格式)。

  判定方法:`scripts/git-hooks/commit-msg` 钩子(安装:
  `git config core.hooksPath scripts/git-hooks`,新克隆仓库后必须执行一次)。

参考正则(钩子实现须与上文文字一致,冲突时以文字为准并同 PR 修正钩子):

```text
^(feat|fix|perf|refactor|docs|test|build|ci|chore|revert)\((core|ir|compiler|runtime|hal|ops|frontend|backend-cpu|backend-cuda|backend-intel-gpu|backend-intel-npu|backend-ascend|python|cmake|docs|tests|examples|tools|scripts)\): .+$
```

外加两项检查:首行字符数 ≤72;摘要部分含至少一个汉字(`[\x{4e00}-\x{9fff}]`)。

### 正例(3 条)

```text
feat(ops): 新增 relu 算子 schema 与 CPU 参考实现
build(cmake): 搭建后端三态开关与 presets 骨架
fix(backend-cuda): 修复 allocator 忽略对齐参数的缺陷
```

### 反例(3 条,附打回原因)

```text
Add relu op
```
无 `type(scope)` 结构,且摘要非中文。

```text
feat: 新增 relu 算子
```
缺 scope。

```text
feat(cuda): 新增 relu kernel
```
scope 不在封闭枚举内(应为 `backend-cuda`)。

## 3. PR 描述规范

- 【LANG-011】【MUST】PR 描述用中文,并完整填写 `.github/PULL_REQUEST_TEMPLATE.md`
  的三个必填段:
  1. **复用检查**:REUSE-001 五步搜索的命令与结果摘要;
  2. **已读文档**:本次改动前阅读的 docs/ 路径清单(「已读文档」的执法即以此清单为准);
  3. **测试证据**:执行的命令与结果(含按 BUILD-010 被 SKIP 的测试清单)。

  判定方法:code-reviewer 检查三段齐全且非空;缺段或空段即打回。
