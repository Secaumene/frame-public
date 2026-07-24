# ADR-0004:C++20 基线

- 状态:已接受
- 日期:2026-07-04
- 关联铁律:#1、#3
- 关联规则:CPP-001

## 背景

铁律 #1② 要求 C++ 层编译期机制(template/constexpr/concept/CRTP)优先于运行时
机制(virtual/RTTI)。concept 对模板接口的约束表达、`if constexpr`、`std::span`
等特性显著降低编译期多态的书写与诊断成本;但语言基线必须经四后端工具链兼容性
论证后才能固定。

## 决策

全仓 C++ 基线为 C++20:`CMAKE_CXX_STANDARD 20` + `CMAKE_CXX_STANDARD_REQUIRED ON`。
论证:

- concept 约束模板接口、CRTP + concept 替代虚函数,直接服务铁律 #1②;
  `std::span` 免去自研 Span 容器;
- host 工具链 g++ 13 与 Intel icpx(oneAPI DPC++)支持 C++20;
- nvcc 自 CUDA 12.0 起支持 C++20,故 CUDA 后端要求 CUDAToolkit ≥ 12.0 且
  `CMAKE_CUDA_STANDARD 20`;
- CANN 宿主编译器兼容性:【待查证】CANN toolkit 配套宿主编译器对 C++20 的支持
  版本范围 —— 来源:昇腾社区(hiascend.com)CANN 安装指南与版本配套说明。
  列为跟进项,登记于 docs/backends/ascend.md 待查证清单。

判定方法:顶层 CMakeLists.txt 含 `CMAKE_CXX_STANDARD 20`;更改标准版本必须
新立 ADR(README.md 触发清单第 5 项)。

## 备选方案

- C++17 基线:优点是工具链兼容面最大;缺点是无 concepts/span,编译期多态被迫
  依赖 SFINAE 与自研容器,直接削弱铁律 #1② 的落地质量。否决。
- C++23 基线:优点是 `std::expected` 可作 Result&lt;T&gt; 底座;缺点是四后端
  工具链支持不齐,设备编译器风险高。否决:待工具链成熟后另立 ADR 重评。

## 后果

- 正面:concept/CRTP/if constexpr/span 全量可用,编码规范可执行性强。
- 负面:锁定 CUDA ≥ 12.0,放弃 11.x 环境;CANN 宿主兼容性存在待验证风险。
- 跟进:TODO(FRAME-DEP): 验证 CANN 宿主编译器 C++20 兼容性。参考:
  docs/backends/ascend.md。完成判据:在装有 CANN toolkit 的环境以 C++20
  编译 include/frame/ 全部头文件通过。
