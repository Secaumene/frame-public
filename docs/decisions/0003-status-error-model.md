# ADR-0003:核心库 Status/Result&lt;T&gt; 错误模型

- 状态:已接受
- 日期:2026-07-04
- 关联铁律:#2、#4
- 关联规则:CPP-020、LANG-005、PY-001

## 背景

核心库跨越四个后端插件与多种设备工具链,C++ 异常跨插件/ABI 边界的行为不可靠,
部分设备编译环境惯例性关闭异常;各设备 SDK(CUDA、SYCL、OpenVINO、CANN)均为
错误码风格 API。需要一个统一、显式、可跨插件边界传递的错误模型。

## 决策

- 核心库(include/frame/、src/)禁用异常,统一以 `Status` 与 `Result<T>` 返回
  错误,定义于 `include/frame/core/status.h`;
- `throw` 仅允许出现在 python/ 绑定层:由集中函数 `translate_status()` 将
  `Status` 映射为 Python 异常(映射表见 docs/standards/python-binding.md);
- `Status` 消息与日志文本一律英文(LANG-005,评审 M16 裁决);
- 不可恢复的程序缺陷用 `FRAME_CHECK(cond)` 带英文消息直接 abort,不走 Status。

判定方法:`scripts/check_iron_rules.sh` 检查 src/、include/ 中 `throw` token
零命中(python/ 目录除外);消息语言由 code-reviewer 检查表逐条核对。

## 备选方案

- C++ 异常:优点是语言原生、调用代码简洁;缺点是跨插件/ABI 边界不安全,与设备
  工具链兼容性差,错误控制流不可见。否决。
- 裸错误码(int/enum):优点是零开销;缺点是无消息与上下文载荷,返回值易被
  静默忽略。否决。
- 引入 abseil 的 Status/StatusOr:优点是实现成熟;缺点是为单一组件引入大型
  依赖,不满足 REUSE-010 的引入门槛。否决:自研轻量 Status/Result&lt;T&gt;,
  确有需要时再以新 ADR 重评。

## 后果

- 正面:错误路径显式可审计;HAL 与插件边界安全;与设备 SDK 错误码自然对接
  (CUDA_CHECK / ACL_CHECK 等宏统一转换为 Status)。
- 负面:调用点需要 FRAME_RETURN_IF_ERROR 类样板宏;无异常栈展开信息。
- 跟进:include/frame/core/status.h 头文件桩;python/ 绑定层 translate_status()
  桩;Status 错误码与 Python 异常映射表落入 docs/standards/python-binding.md。
