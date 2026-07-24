# 00 从源码安装

**适用版本：Frame v1.3 M28，C++20；Python 绑定 0.1.0（可选）**

**最后更新：2026-07-24**

本章是非规范性使用说明；构建、安装和导出契约以[构建与测试](../../docs/standards/build-and-test.md)为准。

## 学习目标

从已有 Frame 源码构建 CPU 版本，并安装供独立 C++ 消费工程使用的安装包。

## 前置

- CMake 源码构建需要 CMake 3.25 或更高版本，以及支持 C++20 的编译器。
- 消费方工程需要 CMake 3.24 或更高版本。
- 从公开仓库获取源码；以下用 `<frame-source>` 表示克隆后的源码目录。
- Frame 官方只发布纯源码，不提供 wheel、系统包、容器或预编译库；本章全部编译
  都在你的机器上完成。

## 可运行命令

先克隆公开源码，再使用最小的 `cpu-only` preset。选择用户可写的专用安装前缀：

```bash
git clone https://github.com/Secaumene/frame-public.git <frame-source>
export FRAME_PREFIX="$HOME/.local/frame/0.1.0"
cd <frame-source>
bash scripts/install.sh --preset cpu-only --prefix "$FRAME_PREFIX"
find "$FRAME_PREFIX" -name frame-config.cmake -print
```

便利脚本在尚未配置该 preset 时会先构建。等价的原生命令是：

```bash
cd <frame-source>
cmake --preset cpu-only
cmake --build --preset cpu-only
cmake --install build/cpu-only --prefix "$FRAME_PREFIX"
```

首次构建可能获取项目锁定的依赖；这取决于本机缓存和构建环境。安装结果应包含 `frame-config.cmake`，上面的 `find` 命令会打印其实际位置。

## 验收清单

- 安装脚本成功时输出 `OK install (cpu-only)`。
- `find` 至少打印一条位于 `$FRAME_PREFIX` 下的 `frame-config.cmake` 路径。
- 以上两项都出现，才表示安装和 CMake 包配置已落到目标前缀。

## PyTorch 对照

PyTorch 常通过官方 Python wheel 安装并在解释器中导入；Frame 官方发布的是纯源码，
本章在本机编译并安装供 CMake
`find_package(frame)` 使用的 C++ 安装包。两种分发和使用方式不同，PyTorch
并非本安装过程的依赖。

## 常见失败

- 配置失败时，先核对 CMake 版本、C++20 编译器和当前源码目录。
- 依赖获取失败时，检查网络、代理或本机依赖缓存。
- 前缀不可写时，改用用户可写的专用前缀后重试。

## 边界

- 系统安装只包含 C++ 件，不包含 Python 扩展。
- Frame CI 只临时构建和测试，不上传本章产生的安装件或其他编译产物。
- 当前没有 uninstall 脚本。若使用专用、确认无其他内容的安装前缀，用户可自行删除该专用目录；删除前应先核对目标路径和内容。

## 小结

- 用 `cpu-only` 建立首次可复现的构建基线。
- 用 `FRAME_PREFIX` 隔离安装位置，并以 `frame-config.cmake` 验收导出结果。

## 练习

1. 将 `FRAME_PREFIX` 改为自己的可写专用目录并完成一次安装。
2. 用 `find` 记录 `frame-config.cmake` 的实际路径，供下一章配置消费工程。

## 下一章

继续[01 仓外 C++ 项目](../01-quickstart/README.md)。
