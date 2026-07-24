# ADR-0012:spdlog 推迟,维持薄 warn_log

- 状态:推迟(重评条件:日志点 ≥3 或分级/落盘/异步需求)
- 日期:2026-07-13
- 关联铁律:#5 两级复用
- 关联规则:REUSE-010、LANG-005、ARCH-011

## 背景

现日志面极小:回退链 WARN 的单一出口 `warn_log`(src/runtime/,
fprintf(stderr) 薄函数,头注释锁死「永不生长为日志框架」)+ 两处启动
期 fatal 诊断(注册 fail-fast,终止路径非日志)。spdlog 在复用准入表
为已批准档,但接入 = 新增 FetchContent,须先有已接受 ADR(REUSE-010)。

## 决策

- 推迟引入 spdlog。单调用点引入完整日志库,集成与版本维护成本大于
  收益;`warn_log` 薄函数继续作为唯一 WARN 出口,其「不加 level/sink/
  格式化 DSL」约束继续有效——需求一旦超出该约束即触发本 ADR 重评,
  而非在薄函数上加功能(防影子日志框架,铁律 #5 第一级)。
- 重评条件展开(纯机械):排除 src/runtime/warn_log.h 与 warn_log.cpp
  自身后,`grep -rln "warn_log(" src/` 命中的不同 .cpp 文件数 ≥ 3
  (现值 2:compile.cpp 与 fallback_executable.cpp);或出现分级
  (ERROR/INFO)/落盘/异步写出任一明确需求。重评通过前禁止接入。
- 判定方法:`grep -rn "spdlog" cmake/ src/ CMakeLists.txt` 仅命中文档
  性注释;src/runtime/warn_log.h 的依赖跟进标记指向本 ADR。

## 备选方案

- 现在接入 spdlog:一步到位;但为单 WARN 出口引入线程池/格式化引擎级
  依赖,违反最小依赖面——否决。
- 自研分级日志:重复造轮子,恰是 spdlog 覆盖的标准能力,违反铁律 #5
  第一级——否决(需求出现时直接重评接入,不自研)。

## 后果

- 正面:依赖面不变;日志需求增长有明确的触发阈值与升级路径。
- 负面与代价:重评前无分级/落盘能力(现无此需求,代价为零)。
- 跟进:src/runtime/warn_log.h 的 TODO(FRAME-DEP) 改指本 ADR 与其
  重评条件。
