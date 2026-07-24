# 后端 HAL 接口规范

> **强制等级**:规范(MUST)
> **相关铁律**:#1 编译优先 / #3 后端矩阵 / #5 两级复用
> **面向读者**:实现代码的工程师
> **最后更新**:2026-07-04

## 1. 设计原则

1. **接口最小完备**:以五后端(cpu/cuda/intel_gpu/intel_npu/ascend)的交集为准,后端特有能力经 `compile()` 内部判定并按 ARCH-031 返回错误协商,而非接口膨胀。
2. **扁平对象模型,Device 是值类型**:HAL 不设 Device 虚接口类。`Device` 是核心层值类型(`include/frame/core/device.h`):

   ```cpp
   struct Device {
     std::string_view backend;  // 注册键字符串,如 "cuda"
     int32_t index;             // 设备序号
   };
   ```

   仅作寻址句柄;内置后端名常量随该头文件提供。所有设备操作由 `Backend` 直接提供(`allocator(Device)`、`create_stream(Device)` 等)。**不设 DeviceType 枚举**——后端以注册键字符串标识,新增后端零核心层改动。
3. **虚函数白名单区域**:HAL 是插件边界,允许抽象基类 + 虚函数;白名单 = {`Backend`, `Stream`, `Event`, `Allocator`, `Executable`, `Pass`} 六类(∪ 各后端实现文件、测试 fixture),白名单外新增 `virtual` 需 ADR(CPP-010)。
4. **错误模型**:一律以 `Status`/`Result<T>`(`include/frame/core/status.h`)返回,禁止异常跨 HAL 边界(CPP-020);日志与 Status 消息英文(LANG-005)。

## 2. 接口方法清单

签名级清单。头文件桩 `include/frame/hal/{backend,stream,event,allocator,executable}.h` 必须与本节逐一对应,不一致以本文档为准并同 PR 修正。

### 2.1 Backend(`include/frame/hal/backend.h`)

> 本表与 `include/frame/hal/backend.h` 的 `class Backend` 纯虚方法逐一对应,是接口的唯一权威。全部方法返回 `Status`/`Result<T>`,禁止异常跨 HAL 边界。

| 方法 | 说明 |
|---|---|
| `name() const -> std::string_view` | 返回注册键字符串(`"cpu"` 等) |
| `device_count() const -> Result<int32_t>` | 可用设备数 |
| `create_stream(Device device) -> Result<std::unique_ptr<Stream>>` | 创建流 |
| `create_event(Device device) -> Result<std::unique_ptr<Event>>` | 创建事件(设备域生命周期;可在流 A record、流 B wait) |
| `allocator(Device device) -> Allocator*` | 指定设备的分配器(非拥有指针) |
| `copy(void* dst, Device dst_device, const void* src, Device src_device, size_t bytes, Stream* stream) -> Status` | 异步拷贝;方向(H2D/D2H/D2D)由两端 Device 推导,主机内存以 cpu 后端的 Device 表示 |
| `compile(const ir::Graph& graph, const CompileOptions& options) -> Result<std::unique_ptr<Executable>>` | 编译路径唯一入口(铁律 1① 后端首要职责) |
| `launch(const KernelInvocation& invocation, Stream* stream) -> Status` | eager 单算子入口,查 `KernelRegistry` 执行;调用点受 ARCH-011 白名单约束。`KernelInvocation` 打包 op 名、输入/输出 `Tensor`、属性与目标 `device` |

- 能力查询不设独立方法:某算子/dtype 是否被支持,由 `compile()` 在 lowering 时判定并按 ARCH-031 返回带算子名的错误(见第 4 章)。`Event` 创建入口是 `Backend::create_event`(而非 `Stream`——同一 event 可在流 A record、流 B wait,创建入口绑定 Stream 会误示流亲和),经 `Stream::record` 记录、`Stream::wait` 建立跨流依赖(见 2.2/2.3)。

#### CompileOptions 字段集(`include/frame/hal/backend.h`)

`CompileOptions` 仅被 `Backend::compile` 消费,控制后端内部 lowering/codegen 优化档位;**不控制标准 pass 管线**(管线由 compiler 层固定,ARCH-053)。

| 字段 | 类型 | 说明 |
|---|---|---|
| `opt_level` | `int32_t`(默认 `1`) | `0` = 关闭后端内部优化(调试);`1` = 默认档位 |
| `fingerprint() const` | `-> std::string` | 确定性 `key=value` 串,充当编译缓存键的「编译选项哈希」分量(execution-model.md 4.1),样例 `"opt_level=1"`。纪律:任何新增字段必须同步进入 `fingerprint()`,否则缓存键失真——code-reviewer 按此判定 |

#### KernelInvocation 字段集(`include/frame/hal/backend.h`)

eager 单算子启动描述(execution-model.md 2.2)。借用契约与 `ops::NodeContext` 相同:`attrs` 指针仅在 `launch` 调用期间有效,可空 = 无属性;`inputs`/`outputs` 借用调用方存储。`launch` 不做形状推断与输出分配——`outputs` 由调用方按 `OpSchema::shape_infer` 结果预分配。

| 字段 | 类型 | 说明 |
|---|---|---|
| `op` | `std::string_view` | 已注册算子名 |
| `inputs` | `std::span<const Tensor>` | 输入张量 |
| `outputs` | `std::span<Tensor>` | 输出张量(调用方预分配) |
| `attrs` | `const std::unordered_map<std::string, ir::AttrValue>*`(默认 `nullptr`) | 可空 = 无属性 |
| `device` | `Device` | 目标设备,显式携带(不从张量推导——空输入算子亦有明确目标) |

### 2.2 Stream(`include/frame/hal/stream.h`)

`synchronize() -> Status`;`record(Event&) -> Status`;`wait(const Event&) -> Status`;`native_handle() -> void*`(仅供后端内部下沉)。

- 【ARCH-030】【MUST NOT】`src/{core,ops,ir,compiler,runtime}/` 与 `python/` 禁止调用 `native_handle`;该方法仅允许在 `src/backends/` 与测试中使用。判定方法:运行 `scripts/check_iron_rules.sh`(标识符 token 检查:`native_handle` 出现位置白名单)。

### 2.3 Event(`include/frame/hal/event.h`)

`query() -> bool`(非阻塞完成查询);`synchronize() -> Status`。

- 创建入口:`Backend::create_event(Device device) -> Result<std::unique_ptr<Event>>`(见 2.1)——设备域生命周期,与 CANN `aclrtCreateEvent` 语义一致;SYCL/OpenVINO 等无独立事件创建原语的后端可实现为空事件容器、由 `record()` 填充。
- 未 record 的 Event 语义定案:`query()` 恒为 `true`、`synchronize()` 恒返回 `Ok`——「视为已完成」,与 CUDA `cudaEventCreate` 后未 record 即成功的语义一致;HAL 一致性套件(`tests/cpp/hal_conformance/`)据此断言。

### 2.4 Allocator(`include/frame/hal/allocator.h`)

`allocate(size_t bytes, size_t alignment) -> Result<void*>`;`deallocate(void* ptr) -> void`。是否池化由后端自行决定,接口不变。

### 2.5 Executable(`include/frame/hal/executable.h`)

`run(std::span<const Tensor> inputs, std::span<Tensor> outputs, Stream& stream) -> Status`;`input_signature()` / `output_signature()`(dtype/shape 签名列表,供编译缓存与调用校验)。

## 3. 注册机制

- 注册宏:`FRAME_REGISTER_BACKEND(name_string, BackendClass)`——static 对象构造时向全局 `BackendRegistry` 注册;宏与 `BackendRegistry` 声明位于 `include/frame/hal/backend.h`,注册表实现位于 `src/runtime/backend_registry.cpp`。
- 查询:`BackendRegistry::instance().get("cuda") -> Result<Backend*>`(非静态成员,经单例访问;用法示例见 `examples/03_custom_op/main.cpp`);键为注册名字符串。
- 注册键封闭清单(当前):`"cpu"`、`"cuda"`、`"intel_gpu"`、`"intel_npu"`、`"ascend"`;目录名与注册键一致(`src/backends/<注册键>/`)。cpu 永远启用,为参考后端。该清单约束 in-tree 后端命名;`BackendRegistry` 运行时不强制此清单(外部插件可注册其他键,呼应第 1 章可替换性 R2)。
- 后端以独立库形式链接,是否编译由三态 CMake 选项 `FRAME_ENABLE_<NAME>`(AUTO|ON|OFF,默认 AUTO)控制,详见 `standards/build-and-test.md`。

## 4. 能力协商协议

- 能力协商内嵌于 `compile()`:后端在 lowering 时逐节点判定该算子/dtype 是否可实现,不支持即中止编译并返回带算子名的错误;上层(runtime)据此触发回退链(`architecture/execution-model.md` 第 5 章)。本设计不设独立的 `supports_op` 查询方法——单一入口避免「查询通过但编译失败」的两段式不一致。
- 【ARCH-031】【MUST NOT】后端不得在 `compile()` 内部对不支持的算子静默降级到 CPU;必须返回带算子名、错误码为 `ErrorCode::kUnimplemented` 的错误(英文消息;错误码加严自 M10 设计裁决:「不支持」是回退链的唯一触发哨兵,与 `kNotFound`(后端名不存在等配置错误,硬失败)机械可辨),由上层(runtime)统一决策回退——保证回退日志与统计不被绕过。同一错误码约定适用于 `backend_lowering` pass 的逐节点支持性判定落点。判定方法:HAL 一致性测试套件(`tests/cpp/hal_conformance/`)包含「不支持算子时 compile 返回含算子名、码为 kUnimplemented 的错误」的用例,对任意注册后端执行。

## 5. 新后端接入 checklist

逐项可勾选;缺项即打回。

1. [ ] 在 `src/backends/<name>/` 建目录,目录名 = 注册键字符串(全小写下划线)。
2. [ ] 实现五个 HAL 接口类:`Backend`、`Stream`、`Event`、`Allocator`、`Executable`(第 2 章清单逐一对应)。
3. [ ] `FRAME_REGISTER_BACKEND("<name>", <Name>Backend)` 完成注册。
4. [ ] CMake 接入:SDK 探测优先 `find_package`;官方无 config 包时自写 `cmake/find_<sdk>.cmake`(参照 `cmake/find_cann.cmake`);新增三态选项 `FRAME_ENABLE_<NAME>`(AUTO|ON|OFF,默认 AUTO;ON 且探测失败必须 FATAL_ERROR,禁止静默跳过)。
5. [ ] 按 `backends/README.md` 九章模板撰写 `docs/backends/<name>.md`(缺章打回);不确定处一律 `【待查证】—— 来源:<官方文档>`(BE-000)。
6. [ ] 通过 HAL 一致性测试套件 `tests/cpp/hal_conformance/`:对任意注册后端跑同一组接口行为测试,含「注册自定义算子/pass 并生效」的扩展点用例(REUSE-020)。
7. [ ] 后端冒烟测试落在 `tests/cpp/backends/`;目标设备缺失时 `GTEST_SKIP()` 并输出原因(BUILD-010)。
8. [ ] 错误一律经 `Status`/`Result<T>` 返回,禁止异常跨 HAL 边界(CPP-020)。
9. [ ] 全部日志与错误消息英文(LANG-005)。

TODO(FRAME-TEST): 充实 HAL 一致性测试套件用例。参考:tests/cpp/hal_conformance/。完成判据:套件覆盖第 2 章全部接口方法与 ARCH-031、REUSE-020 用例,且对 cpu 后端在 cpu-only preset 下全绿。
