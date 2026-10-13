# Build and Diagnostics / 构建与诊断

## Requirements / 环境

- Linux with an NVIDIA GPU and compatible NVIDIA driver.
- CMake 3.18+, C++17 compiler, and CUDA Toolkit with NVCC.
- TensorRT 10 headers and libraries.
- OpenCV, GLFW, OpenGL, FFmpeg (`libavformat`, `libavcodec`, `libavutil`, `libswscale`) and pkg-config.
- Optional UVC camera support: system libuvc and libusb development packages. Without them, the application still builds with UVC input disabled.

本项目目前针对 Linux 与 NVIDIA GPU。核心依赖包括 CMake 3.18+、C++17、CUDA Toolkit、TensorRT 10、NVIDIA 驱动/NVDEC、OpenCV、FFmpeg 开发库、GLFW、OpenGL 和 pkg-config。安装系统 libuvc/libusb 开发包后才会启用可选 UVC 摄像头；缺少它们时仍可构建，UVC 输入会关闭。

## Configure and build / 配置与构建

```bash
cmake -S . -B build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DTENSORRT_ROOT=/usr/local/TensorRT
cmake --build build -j
```

Pass the actual paths for the local CUDA Toolkit and TensorRT installation. `CMAKE_CUDA_ARCHITECTURES` accepts a semicolon-separated list such as `86;89;120`. The default is `120`, matching the original development target.

将 CUDA Toolkit、GPU 架构和 TensorRT 路径改为本机实际值。`CMAKE_CUDA_ARCHITECTURES` 可传入以分号分隔的列表（例如 `86;89;120`）；默认值 `120` 对应原始开发目标。

If CMake cannot find a system dependency, install its development package or pass the corresponding standard CMake/pkg-config search path. The build does not download CUDA, TensorRT, model weights, or camera firmware.

如果 CMake 找不到系统依赖，请安装相应开发包，或设置标准 CMake/pkg-config 搜索路径。构建过程不会下载 CUDA、TensorRT、模型权重或相机固件。

## Run and diagnose / 运行与诊断

```bash
./build/yolov5_sahi
./build/engine_inspect /path/to/model.engine
./build/camera_bench --device 0 --duration 10
./build/camera_bench --device 0 --preproc --duration 10
./build/camera_bench --device 0 --infer --engine /path/to/model.engine --duration 10
```

The application needs a compatible TensorRT engine selected in the GUI. Engine plans are often specific to the GPU, TensorRT release, and build settings; supply a plan created for the target machine. Camera commands require V4L2 access and suitable hardware. The `engine_inspect` smoke inference skips execution for FP16-I/O plans; its output reports that case.

GUI 中需要选择兼容的 TensorRT engine。Engine plan 通常与 GPU、TensorRT 版本和构建参数相关，请为目标机器准备对应文件。摄像头命令需要 V4L2 权限和实际设备。FP16 输入/输出的 engine 会跳过 `engine_inspect` 的推理 smoke check，工具会在输出中说明。

## Model conversion / 模型转换

Install the optional Python packages `ultralytics` and `onnx`, and make TensorRT `trtexec` available. Provide model weights you are permitted to use:

```bash
python3 tools/convert_model.py \
  --weights /path/to/model.pt \
  --output-dir /path/to/artifacts \
  --trtexec /path/to/TensorRT/bin/trtexec
```

Generated model files are ignored by Git. Review model and dataset licenses independently before redistribution.

权重由使用者自行准备。模型转换依赖 Python 包 `ultralytics`、`onnx` 和 TensorRT 的 `trtexec`：

```bash
python3 tools/convert_model.py \
  --weights /path/to/model.pt \
  --output-dir /path/to/artifacts \
  --trtexec /path/to/TensorRT/bin/trtexec
```

转换生成的模型文件已加入 Git 忽略规则。再次分发模型或数据集前，请单独核实其许可证。
