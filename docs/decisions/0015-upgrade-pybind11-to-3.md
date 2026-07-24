# ADR-0015:pybind11 升级至 v3.0.4 并收紧锁定语义

- 状态:已接受
- 日期:2026-07-13
- 关联铁律:#5 两级复用
- 关联规则:REUSE-010、REUSE-012、REUSE-013、PY-040

## 背景

pybind11 现锁 v2.13.6,但其 FetchContent 带 `FIND_PACKAGE_ARGS 2.13`
(系统包优先),导致 venv 内 pip 安装的 3.0.4 实际替代了锁定版参与
构建——锁定语义失效(漂移已实测发现并留档)。同时 2.13 线已终止维护,
3.0.x 为当前稳定线。REUSE-010 规定版本变更须 ADR;REUSE-013 要求优先
最新稳定版。

## 决策

- 锁定版本升级 **v2.13.6 → v3.0.4**(REUSE-013 记载:核实来源
  github.com/pybind/pybind11/releases,核实日期 2026-07-13,v3.0.4 为
  当时最新稳定发行版;2.13.6 为 2.x 线末版,已停止维护;许可证 BSD-3-
  Clause,REUSE-014 白名单内)。
- 兼容性证据(升级依据,非猜测):本仓全部绑定面(七算子 + interop +
  autograd 六个绑定单元)已在 pip 安装的 pybind11 3.0.4 下完整构建并
  通过 pytest 49/49(漂移期间的实测事实);唯一 3.x 相关适配
  (`unique_ptr<Graph>` 返回规避拷贝探测)已在训练绑定批落地。
- Python 支持矩阵核实(来源 github.com/pybind/pybind11/releases/tag/
  v3.0.0 发布说明,核实日期 2026-07-13):3.0 线仅移除 Python 3.7 与
  PyPy 3.8/3.9 支持(CPython 下限 3.8)并新增 3.14 支持——覆盖 PY-040
  下限 3.9 与项目声明矩阵 3.9–3.13;CMake 支持窗 3.15–4.0,含本仓下限。
- **移除 pybind11 FetchContent 的 `FIND_PACKAGE_ARGS`**:锁定即锁定,
  不再允许系统/pip 包静默替代(与 Google Benchmark 引入形态一致,
  ADR-0014 同款防漂移口径);pyproject `[build-system]` 的
  `pybind11>=2.13` 同步收紧为 `>=3.0`(scikit-build-core 路径与
  FetchContent 路径版本线一致)。GoogleTest 的 FIND_PACKAGE_ARGS 维持
  现状(无漂移事实,另案再议)。
- 判定方法:`cmake/frame_dependencies.cmake` 中 pybind11 GIT_TAG 为
  v3.0.4 且无 FIND_PACKAGE_ARGS;`pyproject.toml` requires 含
  `pybind11>=3.0`;wheel/venv 全量重建后 pytest 全绿、cpu-only 门禁
  全绿。

## 备选方案

- 维持 2.13.6 并仅修 FIND_PACKAGE_ARGS:锁定语义恢复,但锁在已停维护
  的版本线上,与 REUSE-013 相悖且放弃已实测可用的 3.x——否决。
- 跟随系统包(去锁定):可复现性丧失,违反 REUSE-012 锁 tag 纪律——
  否决。

## 后果

- 正面:锁定语义真实生效;版本线回到受维护通道;与实测运行环境一致。
- 负面与代价:构建首次需重新拉取 pybind11 源;3.x 后续升级仍按
  REUSE-010/013 流程。
- 跟进:随实现同批更新 build-order.md 第 4 节、frame_dependencies.cmake
  头部**版本锁定表与依赖决策表类别 B 行**(获取方式列把 FIND_PACKAGE_ARGS
  注记为「防漂移需要时可去除」,与 Google Benchmark 落地形态一致;实例列
  pybind11 版本同步);python-binding.md 第 7 节「推荐锁定版本」待查证由
  本 ADR 销项(维护者亲自改);GoogleTest 的 FIND_PACKAGE_ARGS 取舍留待
  其自身升级触点。
