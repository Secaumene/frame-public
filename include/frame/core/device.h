#pragma once
// 设备寻址句柄:Device 是纯值类型,constexpr 友好,无虚函数。
// 关键决策:**不设 DeviceType 枚举** —— 后端以注册键字符串标识。
// 新增后端零核心层改动:只需注册一个字符串名(见 docs/architecture/backend-hal.md)。

#include <cstdint>
#include <string_view>

namespace frame {

// 内置后端注册键常量(封闭清单)。目录名/注册键/此常量三者一致。
inline constexpr std::string_view kCpuBackendName = "cpu";
inline constexpr std::string_view kCudaBackendName = "cuda";
inline constexpr std::string_view kIntelGpuBackendName = "intel_gpu";
inline constexpr std::string_view kIntelNpuBackendName = "intel_npu";
inline constexpr std::string_view kAscendBackendName = "ascend";

// Device:寻址句柄 = 后端注册键字符串 + 设备序号。聚合类型,== 缺省。
struct Device {
  std::string_view backend;  // 注册键字符串,如 "cuda"
  int32_t index = 0;         // 设备序号

  constexpr bool operator==(const Device&) const = default;
};

// cpu 参考后端的 Device 工厂(cpu 永远启用)。其余后端无需专用工厂:
// 直接用 Device{kCudaBackendName, index} 构造即可,呼应"新增后端零核心层改动"。
constexpr Device cpu_device(int32_t index = 0) { return Device{kCpuBackendName, index}; }

}  // namespace frame
