# third_party

本目录**默认为空**(除本 README 外不应有任何文件)。

## 依赖策略指引

- 第三方依赖的获取方式由 `cmake/frame_dependencies.cmake` 顶部的类别决策表(A–D)裁决,
  完整政策见 `docs/standards/reuse-policy.md`;全部 `FetchContent_Declare` 集中在该
  cmake 文件,禁止散落各子目录。
- 新增任何 `find_package` / `FetchContent` 依赖必须先有 ADR(REUSE-010),并在
  `cmake/frame_dependencies.cmake` 的依赖表注释中登记类别与版本。

## 本目录规则

- 【THIRD-001】【MUST NOT】禁止手工复制(vendor)任何第三方源码进入本目录。
  判定方法:`git ls-files third_party/` 仅含本 README(以及经批准的 submodule 条目)。
- 【THIRD-002】【MUST】仅当出现离线构建需求时,才允许以 git submodule 形式在本目录
  放入依赖,且须同时满足:在下方登记表登记(名称/版本/用途/ADR 编号),并有对应 ADR。
  判定方法:`.gitmodules` 的每个条目在下方登记表有同名行,且 `docs/decisions/` 存在
  对应 ADR 文件。

## 登记表(当前为空)

| 名称 | 版本 | 用途 | ADR |
|------|------|------|-----|
