# Contributing

Thanks for helping improve the pipeline. Bug reports, focused fixes, documentation improvements, and reproducible performance investigations are welcome.

## Before opening an issue

Include your Linux distribution, GPU model, NVIDIA driver, CUDA and TensorRT versions, build command, relevant logs, and a minimal reproduction. Do not attach private video, model files, or camera data unless you have permission to share them.

## Pull requests

- Keep changes focused and explain the user-visible behavior or engineering trade-off.
- Build from a fresh directory and include the exact command and result.
- For CUDA, TensorRT, or camera changes, state which hardware-backed checks you ran. Clearly label checks that could not run locally.
- Update the Chinese and English documentation when behavior or setup changes.
- Do not commit build output, model weights/engines, logs, recordings, credentials, or machine-specific configuration.
- Preserve notices for third-party code and add a notice before introducing a dependency.

Use short imperative commit subjects, for example: `Fix V4L2 buffer release`.

## 中文说明

欢迎提交缺陷报告、聚焦明确的修复、文档改进和可复现的性能调查。

- 报告问题时，请附 Linux 发行版、GPU、驱动、CUDA/TensorRT 版本、构建命令、相关日志和最小复现步骤。未经授权不要上传私有视频、模型或相机数据。
- Pull request 请聚焦一个改动，说明行为变化或工程取舍；从干净目录构建，并写明命令、结果及未能执行的硬件检查。
- CUDA、TensorRT 或相机改动请注明验证硬件。行为或安装步骤变化时同步更新中英文 README。
- 不要提交构建输出、模型权重/engine、日志、录制文件、凭据或本机配置；引入依赖前补充来源和许可证说明。
- Commit subject 使用简短祈使句，例如：`Fix V4L2 buffer release`。
