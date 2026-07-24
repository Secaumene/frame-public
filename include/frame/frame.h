#pragma once
// 伞头文件:include Frame 全部公共头,别无内容。
// 外部使用者只需 #include <frame/frame.h> 即可获得完整公共 API。

// core:基础设施(宏、错误模型、类型系统、设备、形状、存储、张量)
#include <frame/core/device.h>
#include <frame/core/dtype.h>
#include <frame/core/macros.h>
#include <frame/core/shape.h>
#include <frame/core/status.h>
#include <frame/core/storage.h>
#include <frame/core/tensor.h>

// ir:图 IR(属性、节点、图)
#include <frame/ir/attribute.h>
#include <frame/ir/graph.h>
#include <frame/ir/node.h>

// compiler:pass 扩展点与标准管线
#include <frame/compiler/pass.h>
#include <frame/compiler/pass_manager.h>
#include <frame/compiler/pipeline.h>

// hal:硬件后端抽象层
#include <frame/hal/allocator.h>
#include <frame/hal/backend.h>
#include <frame/hal/event.h>
#include <frame/hal/executable.h>
#include <frame/hal/stream.h>

// ops:算子/内核扩展点
#include <frame/ops/kernel_registry.h>
#include <frame/ops/op_registry.h>
#include <frame/ops/op_schema.h>

// runtime:编译编排入口(标准管线 → Backend::compile,含编译缓存)
#include <frame/runtime/compile.h>

// frontend:「模型描述 → 可运行」用户面入口(校验/lower/训练执行/代码生成,
// ADR-0017)
#include <frame/frontend/emitter.h>
#include <frame/frontend/lowering.h>
#include <frame/frontend/model_spec.h>
#include <frame/frontend/runner.h>
