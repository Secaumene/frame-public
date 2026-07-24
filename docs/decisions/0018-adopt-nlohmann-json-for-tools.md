# ADR-0018:引入 nlohmann/json 作为工具层 JSON 解析库

- 状态:已接受
- 日期:2026-07-13
- 关联铁律:#5 两级复用
- 关联规则:REUSE-010、REUSE-012、REUSE-013、REUSE-014

## 背景

ADR-0017 的 frame_dslc 工具需解析 JSON 模型描述。reuse-policy
准入表已批准 nlohmann/json(限定「仅限工具与测试代码,不得进
核心运行时路径」),但首次实际接线新增 FetchContent 仍触发
REUSE-010,须本 ADR。

## 决策

- FetchContent 锁 tag **v3.12.0**(REUSE-013 记载:核实来源
  github.com/nlohmann/json/releases,核实日期 2026-07-13,当时
  最新稳定发行版,2025-04-11 发布;许可证 MIT,REUSE-014 白名单
  内)。
- 不带 FIND_PACKAGE_ARGS(ADR-0014/0015/0016 同口径,锁定即
  锁定);仅 FRAME_BUILD_TOOLS=ON 时拉取;`JSON_Install=OFF`
  (装机隔离,BUILD-041 口径);声明集中于
  frame_dependencies.cmake(DEP-001)并登记头部版本锁定表与
  决策表类别 B 行。
- 使用面限定:仅 tools/ 与 tests/;核心库(include/frame/ +
  src/)零暴露。解析用免异常模式(`json::parse(...,
  /*allow_exceptions=*/false)` + `is_discarded()`)转 Status,
  与全仓无异常文化一致(CPP-020)。
- 判定方法:`grep -rn nlohmann include/ src/` 为空;
  frame_dependencies.cmake 该段无 FIND_PACKAGE_ARGS;
  FRAME_BUILD_TOOLS=OFF 配置不触发拉取。

## 备选方案

- 自研极小 JSON 解析器:重造成熟标准能力,违铁律 5——否决。
- yaml-cpp / toml++:用户已裁 JSON 载体,且各需独立 ADR 扩
  依赖面——否决。
- 单头文件拷入仓库(vendoring):违 REUSE-012 禁 vendoring
  ——否决。

## 后果

- 正面:声明式载体零自研解析负担;工具层限定使核心依赖面不变。
- 负面与代价:FRAME_BUILD_TOOLS=ON 首次配置需网络拉取源码
  (离线环境依赖 FetchContent 缓存)。
- 跟进:frame_dependencies.cmake 头部两表登记;
  docs/plan/build-order.md 依赖表加行。
