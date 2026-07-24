# 执行模型:编译优先与 eager 逃生舱

> **强制等级**:规范(MUST)
> **相关铁律**:#1 编译优先
> **面向读者**:实现代码的工程师
> **最后更新**:2026-07-12

本文档是铁律 #1①(图静态编译/AOT 优先于 eager 解释)的落地规范,决策记录见 ADR-0001(已接受)。

## 1. 为什么编译优先

以下理由清单可在 PR 讨论中直接引用:

1. **整图视角的优化不可替代**:跨算子融合(operator_fusion)与内存复用(memory_planning)只能在整图视角完成,eager 逐算子执行天然做不到。
2. **NPU 类后端本质是图编译设备**:Intel NPU(经 OpenVINO)与昇腾图模式以「整图编译 + 推理请求」为原生形态,eager 逐算子在其上性能不可接受甚至不可行。
3. **静态 shape 下可做 AOT 内存规划**:缓冲区大小与生命周期在编译期确定,消除运行时分配。
4. **编译产物可缓存**:同一图重复执行时编译成本被摊销(见第 4 章)。

## 2. 两条执行路径的定义

### 2.1 编译路径(默认)

```
构图 → graph.verify() → 标准 pass 管线 → Backend::compile()
    → Executable → 编译缓存 → 重复执行 Executable::run()
```

管线固定顺序见 `architecture/compiler-passes.md`;`Backend::compile` 与 `Executable` 接口见 `architecture/backend-hal.md`。上述"标准 pass 管线 → `Backend::compile` → 编译缓存"整条链路的唯一编排入口是 `runtime::compile`(`include/frame/runtime/compile.h`,M7 已实化,见第 4.1 节)——本入口不做入图前的 `graph.verify()`,非法图的错误挂在第一个 pass 名下返回(避免每次 `compile()` 多付出一次全量 verify 的开销,与 `PassManager::run()` 既定口径一致)。

### 2.2 eager 逃生舱

单算子打包为 `KernelInvocation`(op 名、输入/输出 Tensor、属性、目标 device)后,经 `Backend::launch(invocation, stream)` 直接查 KernelRegistry 执行:无图、无优化、无缓存。签名以 `architecture/backend-hal.md` 2.1 为唯一权威。`Backend::launch` 是 eager 的唯一入口,这一收敛使准入可机械判定(见 ARCH-011)。

## 3. eager 的准入条件

- 【ARCH-010】【MUST】编译路径是默认路径;所有示例、文档、测试的主路径必须使用编译执行(单算子单元测试除外)。判定方法:code-reviewer 对 `examples/`、`docs/`、`tests/` 中以 eager 为主路径的新增内容按本条打回。
- 【ARCH-011】【MUST】仅以下三类情况允许调用 `Backend::launch`(eager):
  1. 用户显式开启调试开关(API 上有明确的 `eager`/`debug` 标志);
  2. 编译路径对某算子/dtype 组合不支持时的自动回退——必须打 WARN 级日志,消息英文(LANG-005),内容至少含算子名、后端名、回退原因;
  3. 单算子的单元测试。

  判定方法:运行 `scripts/check_iron_rules.sh`(检查 7:`Backend::launch` 调用点白名单——成员调用语法的调用点仅允许出现在 `src/runtime/`,承载第 1、2 类入口;`tests/` 不在脚本扫描范围,即第 3 类单测天然放行);需要在其他位置新增调用点时,须同 PR 扩充该脚本的白名单目录,并在 PR 描述中声明属于上述哪一类。
- 【ARCH-012】【MUST NOT】禁止为「图 + eager 混合执行」等增加第三种执行模式;确有需要先走 ADR(触发清单见 `decisions/README.md`)。判定方法:design-reviewer 对触及执行模式的 PR 按本条打回。

## 4. 编译缓存与静态 shape 边界

### 4.1 缓存键

```
cache_key = (图结构哈希, 后端名, 输入 dtype/shape 签名, 编译选项哈希)
```

四个分量任一变化即缓存 miss、重新编译。缓存实现位于 `src/runtime/`。

- **实现口径(M7 已实化,m7-design-brief 决议点 4;M10 交付物①"编译缓存"已
  提前至本里程碑)**:`src/runtime/compile.cpp` 内的 `CompilationCache`(进程级
  Meyer's 单例,`std::mutex` 保护,粒度粗即可 v0)。缓存键**不引入哈希函数,
  键即字符串本身**:`backend_name + '\x1f' + options.fingerprint() + '\x1f' +
  ir::dump_text(graph)`(`'\x1f'` 为 ASCII Unit Separator 分隔符)。四元组
  对应关系:`backend_name` 对应"后端名"分量;`options.fingerprint()` 对应
  "编译选项哈希"分量;`ir::dump_text(graph)` 同时承载"图结构"与"输入
  dtype/shape 签名"两个分量——图输入的类型即输入签名,均已完整体现在
  `dump_text` 的确定性文本内容中,故不需要为其单独拼接字符串段。判据(同图
  同签名二次执行不触发 `Backend::compile`)由
  `tests/cpp/runtime/test_compilation_cache.cpp` 固定。
- **已知边界**:无淘汰策略,容量随不同图数线性增长;淘汰策略留待 M10 视实际
  需要补充(见 `docs/plan/milestones.md` M10 节)。M10 裁决:v0 维持不做淘汰,已知边界继续登记。
- 公共编排入口见 `include/frame/runtime/compile.h`(`runtime::compile`):命中
  缓存则不触发标准 pass 管线与 `Backend::compile`,返回值与缓存共持同一
  `Executable`(`shared_ptr`)。

### 4.2 静态 shape 边界(v0)

- 【ARCH-013】【MUST】v0 仅支持静态 shape:`graph.verify()` 对含动态维(unknown 维度)的图必须返回错误;禁止注册动态 shape 算子变体、禁止引入符号维度机制(呼应 ARCH-044)。动态 shape 支持是显式的 ADR 议题,任何实现动作之前必须先有已接受的 ADR。判定方法:code-reviewer 对引入动态维处理逻辑的 PR 按本条打回;verify 拒绝动态维的行为由单元测试固定(tests/cpp/ir/ 已覆盖)。

## 5. eager 回退链

编译路径遇到后端不支持的算子(`Backend::compile` 返回带算子名的错误——见 ARCH-031)时,按固定顺序回退:

```
① 目标后端 kernel(eager 单算子)
        ↓ 仍不支持
② decomposition:按 OpSchema 注册的分解函数拆为更细算子后重试
        ↓ 仍不支持
③ CPU 参考实现("cpu" 后端,永远可用,ARCH-041 保证存在)
```

回退要求:

1. **日志**:每发生一跳回退,打一条 WARN 级日志,消息英文(LANG-005),至少含:算子名、原后端名、回退目标(kernel/decomposition/cpu)、原因。
2. **统计**:回退次数按(算子名 × 后端名)维度计数,运行结束可查询,便于发现性能悬崖。

v0 实现口径(M10 设计裁决,design-reviewer 通过):

- **触发哨兵**:仅错误码 `ErrorCode::kUnimplemented`(ARCH-031 的两个能力协商
  落点——`backend_lowering` 逐节点判定与 `Backend::compile` 整图判定——统一
  产出)触发回退;`kNotFound`(后端名不存在等配置错误)与其余错误码一律硬
  失败透传,绝不静默回退。
- **产物形态**:回退产物 = runtime 层 `FallbackExecutable`(实现 `hal::
  Executable` 接口),从**未经管线的原图**逐节点解析执行方案(管线产物含
  融合节点,不适用逐算子回退);`Backend::launch` 调用点收敛于 src/runtime/
  (ARCH-011 类别 2,白名单已覆盖)。产物按正常键入编译缓存,对上层透明。
- **单层分解**:② 展开的微图内节点只再走 ①→③,不做二次 decomposition
  (防递归);微图节点目标后端与 cpu kernel 双缺 → 编译错误(含内外算子名)。
- **WARN 与计数时机**:每跳回退的 WARN/计数发生在**编译期回退决策时,一次
  性**(缓存命中不重复计);统计语义 = 回退决策次数,非执行次数。计数粒度
  以**肇因算子**为单位:仅当某节点在目标后端无 eager kernel、需经 ② 或 ③
  解决时记 WARN + 计数(键 = 该算子 × 原目标后端);① 直接命中的节点不计
  (逐节点全计会稀释「性能悬崖」信号)。另在**回退决策点**恒打一条整图级
  WARN(含后端名与原因,无算子键、不入统计)——保证任何一次回退决策至少
  产生一条日志:整图模式后端拒绝整图而逐节点全部命中 ① 时,肇因算子级
  记录为零,该条整图级 WARN 是此情形下唯一的可观测面。
- **v0 内存边界**:现注册面全为 host 内存后端,①/③ 混跑不做跨设备搬运;
  遇非 host 设备张量防御式硬失败(fail-fast,不静默错值);跨设备 copy 编排
  随 M11 真实设备后端接入补齐。

回退链终点是 CPU 参考实现,因此每个 OpSchema 必须自带 CPU 参考实现(ARCH-041,见 `architecture/operator-system.md`);后端不得在 `compile()` 内部对不支持算子静默降级(ARCH-031,见 `architecture/backend-hal.md`)——回退决策统一由上层(runtime)执行,保证日志与统计不被绕过。

## 6. 训练路径(M16–M18、M26;单一事实来源 = architecture/autograd.md)

训练同样走编译路径,无第三种执行模式(ARCH-012 不触发):训练图由
`compiler::build_backward_graph` 从前向图派生(编译期反向图,ARCH-060),
参数更新为独立 SGD 更新图(ARCH-065,纯既有算子组合);两图 shape 恒定,
编译缓存恒命中,训练循环每步零重编译(第 4 章收益的训练形态)。契约、
梯度注册面(OpSchema::GradientFn)与数值验证纪律全文见
`architecture/autograd.md`,本章仅登记路径存在,不复述条文。
M26 的高阶导数仍重复调用同一图变换:一阶非标量梯度先经可微归约标量化,
再派生下一阶图;PINN 的坐标二阶导与 PINO 的 rfft/irfft 谱物理导数都随
普通图进入同一编译管线，不新增 tape 或高阶专用运行模式。
