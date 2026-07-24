# Python 绑定(pybind11)规范

> **强制等级**:规范(MUST)
> **相关铁律**:#2 语言支持 / #4 语言策略
> **面向读者**:实现代码的工程师
> **最后更新**:2026-07-04

本文档约束 `python/` 目录下的全部代码(pybind11 绑定源码与纯 Python 薄层)。
C++ 通用条款(CPP- 系列)对绑定源码同样适用,唯一例外是 `throw`(见 CPP-020 与 PY-030)。

规则条文格式:`【编号】【MUST/MUST NOT/SHOULD】正文。判定方法:<可机械执行的检查>。`

---

## 1. 定位:薄壳原则

- 【PY-001】【MUST】绑定层是薄壳:绑定函数体只做「参数转换 + 调用 C++ 公开 API + 错误
  转换」三件事,禁止业务逻辑(数值计算、图变换、调度决策一律在 C++ 层)。绑定 `.cpp`
  中任何函数体超过 15 行须在 PR 描述说明原因。判定方法:code-reviewer 按行数检查;
  超限且无说明即打回。

## 2. 目录与模块布局

- Python 包名 `frame`;C++ 扩展模块名 `frame._core`。
- 绑定源码位于 `python/src/`;骨架期为单文件 `python/src/bindings.cpp`
  (`PYBIND11_MODULE` 桩)。
- 【PY-002】【SHOULD】当 `bindings.cpp` 超过 500 行时,按 C++ 模块一对一拆分文件
  (`bind_tensor.cpp`、`bind_graph.cpp`…),入口 `module.cpp` 汇总。偏离时 PR 说明。
  判定方法:`wc -l python/src/bindings.cpp`。
- 【PY-003】【MUST】纯 Python 代码放 `python/frame/`,且只允许薄层内容:再导出
  (`__init__.py`)、版本号(`_version.py`)、类型存根(`.pyi`)、不含计算/调度/内存
  管理逻辑的纯工具函数。禁止在纯 Python 层实现算子、图
  或调度逻辑(铁律 #2:核心必在 C++,Python 为绑定)。判定方法:code review;
  `python/frame/` 中出现数值计算/图操作/调度/内存管理实现即打回。

## 3. GIL 纪律(枚举法)

- 【PY-010】【MUST】下述**五类**绑定必须以 `py::call_guard<py::gil_scoped_release>()`
  释放 GIL:`compile` / `run` / `synchronize` / `memcpy` / `allocate`。
- 【PY-011】【MUST NOT】五类之外的绑定不得释放 GIL。

五类函数清单(**唯一事实来源=本表**;新增或变更绑定必须同 PR 更新本表;骨架期本表为
规划清单,绑定落地时逐行核对):

| 类别 | 绑定符号 | 对应 C++ 调用 | 落地状态(M12 起逐行核对) |
|---|---|---|---|
| compile | `frame._core.compile` | `runtime::compile`(经 `Backend::compile`) | 已落地(M12) |
| run | `frame._core.Executable.run` | `runtime::run_with_allocated_outputs`(经 `Executable::run`) | 已落地(M12;run 内联 stream synchronize,均在同一 GIL 释放域) |
| synchronize | `frame._core.Stream.synchronize` | `Stream::synchronize` | v0 不独立导出,随 `Executable.run` 内联执行(M12 登记) |
| synchronize | `frame._core.Event.synchronize` | `Event::synchronize` | v0 不独立导出(同上) |
| memcpy | `frame._core.Tensor.numpy`(D2H) | 设备到宿主拷贝 | 已落地(M12;注记:函数前后须持 GIL 触碰 py::array 等 Python C API,故以**窄域手动 `gil_scoped_release`** 覆盖纯 C++ 拷贝段替代整函数 `call_guard`——GIL 释放义务等效满足,判定时按此口径核对) |
| memcpy | `frame._core.Tensor.to`(H2D/D2D) | 宿主/设备间拷贝 | 已落地(M12) |
| allocate | `frame._core.from_numpy`(含宿主拷入) | `Allocator::allocate` + 宿主拷贝 | 已落地(M12 新增行;注记:同 `numpy` 行——窄域手动 `gil_scoped_release`,持 `py::buffer_info` 保证释放期源缓冲存活) |
| allocate | `frame._core.empty` / `frame._core.zeros` 等设备张量工厂 | `Allocator::allocate` | v0 未导出(张量入口 = `from_numpy`;导出时按本表补核对) |

判定方法:code-reviewer 对绑定文件中每个 `def`/`def_static`/方法绑定逐一对照本表:
表内函数缺 `call_guard` → 打回;表外函数出现 `gil_scoped_release` → 打回;新增五类
函数未更新本表 → 打回。

## 4. 数据交换

- numpy 互操作:宿主内存经 buffer protocol 零拷贝进出(愿景描述;v0/M12 实现
  采**拷贝语义**——`from_numpy` 拷入、`numpy()` 拷出,生命周期简单安全;
  零拷贝为后续偏离销项:TODO(FRAME-IMPL): buffer protocol 零拷贝互操作。参考:docs/standards/python-binding.md 本节。完成判据:tests/python/ 中往返不复制的零拷贝用例通过。)。
- 【PY-012】【MUST】任何触发设备↔宿主拷贝的 API 必须显式命名(`numpy()`、`to()` 等
  动词形式);属性访问不得隐式触发拷贝。判定方法:code review 检查绑定的属性/方法
  签名。
- 【PY-013】【SHOULD】对外张量互操作提供 DLPack(`__dlpack__` / `__dlpack_device__`);
  骨架期默认暂不提供,记为偏离并以 TODO 跟踪:
  TODO(FRAME-IMPL): 实现 DLPack 导入导出。参考:docs/standards/python-binding.md(PY-013)。完成判据:tests/python/test_dlpack.py 中与 numpy 的 from_dlpack 往返用例通过。
- 【待查证】DLPack 协议版本与 `max_version` 协商细节 —— 来源:dmlc/dlpack 仓库文档与
  pybind11 官方文档。

## 5. 错误转换

- 【PY-030】【MUST】`Status` → Python 异常的转换集中在唯一函数 `translate_status()`
  (位于 `python/src/`),禁止各绑定点各自转换;`throw` 仅存在于绑定层(CPP-020)。
  判定方法:`grep -n 'throw' python/src/*.cpp` 的命中应仅位于 `translate_status`
  及其包装宏内。

状态码 → 异常映射表(状态码枚举以 `include/frame/core/status.h` 为准;新增状态码必须
同 PR 更新本表;未列出的状态码一律映射 `RuntimeError`):

| Status code | Python 异常 |
|---|---|
| `InvalidArgument` | `ValueError` |
| `NotFound` | `KeyError` |
| `Unimplemented` | `NotImplementedError` |
| `OutOfMemory` | `MemoryError` |
| `Internal`(及未列出者) | `RuntimeError` |

异常消息 = `Status` message 原文(英文,LANG-005);不做翻译、不做二次拼接。

## 6. API 表面

- docstring 一律中文(LANG-004);参数名为 snake_case 英文(LANG-001)。
- 【PY-020】【MUST】提供 `.pyi` 类型存根(`python/frame/_core.pyi`,手写或
  pybind11-stubgen 生成后人工审校,进版本库);绑定 API 变更必须同 PR 更新 `.pyi`。
  判定方法:PR diff 中 `python/src/` 有变更而 `.pyi` 无对应变更即打回。
- 【PY-021】【MUST】「面向用户算子」必须绑定到 Python 并更新 `.pyi`。
  「面向用户算子」的判定 = 已在 OpRegistry 注册**且**算子名无 `_internal` 后缀。
  判定方法:对已注册算子清单(去除 `_internal` 后缀者)与 `.pyi` 导出清单求差集,
  差集应为空;由 code-reviewer 在涉及 `src/ops/` 的 PR 上执行。
  生效时点(用户裁决,2026-07-11):绑定模块(`frame._core`,里程碑 M12)建立之前
  本条暂缓执行;暂缓期内注册的算子全部列入 M12 交付清单逐个销项,M12 起按上述判定
  方法全量执法。本款为补时序遗漏,判定逻辑本身不变。
  **到期登记(M12 收口)**:暂缓期七算子(add/mul/relu/sum/matmul/square/constant)
  已随 M12 绑定销项,本条自 M12 起全量执法;暂缓条款仅作历史记录保留。

## 7. 构建与版本

- 经 CMake 集成:`FetchContent` 拉取 pybind11 并锁定版本 tag(REUSE-012);
  产物为 `frame._core` 扩展模块;wheel 打包经 scikit-build-core(`pyproject.toml`,
  构建 preset 用 `wheel`,见 BUILD-001)。
- 锁定版本与支持矩阵:pybind11 **v3.0.4**(ADR-0015;3.0 线 CPython 下限 3.8、
  支持至 3.14,覆盖 3.9–3.13 声明矩阵;核实来源 pybind/pybind11 Releases 的
  v3.0.0 发布说明,核实日期 2026-07-13)。原「推荐锁定版本」待查证据此销项。
- 【PY-040】【MUST】支持的 Python 版本下限为 **3.9**;理由:与 numpy 及主流发行版
  维护窗口对齐。判定方法:`pyproject.toml` 的 `requires-python = ">=3.9"`。
- RTTI 说明:pybind11 依赖 RTTI,因此项目**不**全局 `-fno-rtti`;绑定层之外的业务
  代码仍禁用 `dynamic_cast`/`typeid`(CPP-011)。
- 【PY-041】【MUST】`FRAME_BUILD_PYTHON=OFF` 时核心库构建与测试不受影响(绑定为
  可选层,铁律 #2「尽力提供」)。判定方法:`cmake --preset cpu-only` 构建通过
  (该 preset 不含 Python)。

## 8. 测试

- Python 测试规范(文件命名、skip 政策、覆盖要求)见 BUILD-020 / BUILD-021 /
  BUILD-010(docs/standards/build-and-test.md);绑定层每个暴露 API 至少 1 个
  pytest 用例。
