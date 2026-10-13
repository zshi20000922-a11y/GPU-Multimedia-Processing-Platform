# CUDA/TensorRT Multi-Stream Vision Pipeline

**中文** | [English](README.en.md)

面向 Linux/NVIDIA GPU 的多路实时视觉检测示例。项目把摄像头或视频输入、GPU 预处理、TensorRT 推理、后处理、GUI 展示和录制组织成可并行运行的流水线。

> 当前代码面向 Linux 与 NVIDIA CUDA/TensorRT 环境。TensorRT engine 与 GPU、TensorRT 版本有关；仓库不包含模型权重、ONNX 或 engine 文件。

## 能力与作者贡献

| 能力 | 项目实现 |
| --- | --- |
| 多路采集 | V4L2 摄像头 DMA-BUF/CUDA pinned/fallback 路径；FFmpeg CUDA 视频解码；可选系统 libuvc 支持。 |
| GPU 流水线 | 自定义 NV12、YUYV、BGR CUDA 前处理与 FP16/FP32 转换；多 CUDA stream、CUDA event 与四槽 ring buffer 协调每路工作。 |
| 检测 | TensorRT engine 输入/输出形状解析；DIRECT 与 SAHI 切片推理、结果映射和合并。 |
| 展示与存储 | Dear ImGui 多通道控制、融合/拼接展示、GPU 状态面板，以及显示帧和原始帧录制。 |

以上模块由项目作者实现。下面两张图把 PDF 中的总体框架和多路并发设计整理为与当前代码对应的视图。目前不发布未经当前版本复核的帧率或精度承诺。

## 架构总览

```mermaid
flowchart TB
    V4L2[V4L2 摄像头] --> CAM[摄像头采集后端]
    UVC[可选 UVC 摄像头] --> CAM
    FILE[本地视频] --> DEC[FFmpeg CUDA 解码]

    PM[PipelineManager<br/>最多管理 3 路] -. 启动、停止、配置 .-> P
    CAM --> INGEST
    DEC --> INGEST

    subgraph P[每路独立 Pipeline]
        INGEST[采集或解码输入] --> RB[4 槽 RingBuffer]
        RB --> PRE[CUDA 预处理]
        PRE --> TRT[TensorRT 推理]
        TRT --> POST[CPU 后处理与 NMS]
        POST --> RESULT[PipelineResult]
    end

    RESULT --> GUI[ImGui 预览与多路显示]
    P --> REC[显示帧与原始帧录制]
```

每路管线独立持有推理引擎、执行上下文和帧缓冲；解码/传输、预处理、推理和后处理分别使用 CUDA stream，并通过 event 协调帧槽复用。GUI 使用结果锁读取最新结果。摄像头传输按设备能力选择 DMA-BUF、CUDA pinned 或回退拷贝路径。

## 单帧处理流程

```mermaid
flowchart LR
    IN[选定的相机或视频输入] --> INGEST[采集/解码并填充帧槽]
    INGEST --> SLOT[4 槽 RingBuffer]
    SLOT --> PRE[DIRECT：整帧 CUDA 预处理]
    PRE --> DIRECT[TensorRT 推理]
    SLOT --> TILE[SAHI：生成切片坐标]
    TILE --> PRE_TILE[CUDA 切片预处理]
    PRE_TILE --> BATCH[TensorRT 批量推理]
    DIRECT --> OUT[推理输出]
    BATCH --> OUT
    OUT --> CPU[输出拷贝与 CPU 后处理<br/>坐标映射、置信度过滤、NMS]
    CPU --> RESULT[PipelineResult]
    RESULT --> VIEW[预览、融合/拼接]
    RESULT --> SAVE[可选录制]
```

SAHI 推理路径在代码中实现，目前要求兼容 YOLOv5 的 640×640 FP32 engine；GUI 中的 SAHI 模式入口处于隐藏状态。更多组件映射和实现细节见[架构说明](docs/architecture.md)。

### 项目方案材料

精选的项目背景、方案说明、系统框架和多路并发页：[架构与方案精选页 PDF](docs/portfolio/project-architecture-highlights.pdf)（原 PDF 第 2、3、5、8 页）。其中性能数字按原材料中的目标/测试条件理解，不作为通用性能承诺。

## 构建

依赖：CMake 3.18+、C++17 编译器、CUDA Toolkit、TensorRT 10、NVIDIA 驱动/NVDEC、OpenCV、FFmpeg 开发库、GLFW、OpenGL 和 pkg-config。若系统安装 libuvc 开发库，CMake 会启用可选 UVC 输入。

```bash
cmake -S . -B build \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DTENSORRT_ROOT=/path/to/TensorRT
cmake --build build -j
```

根据本机 Toolkit、GPU 架构和 TensorRT 安装路径调整参数。完整依赖和诊断命令见[构建指南](docs/building.md)。

## 运行

```bash
./build/yolov5_sahi
```

在 GUI 中选择本地 TensorRT engine 和输入源。`engine_inspect` 可检查 engine 元数据并进行推理 smoke check；`camera_bench` 用于有 V4L2 设备时测量采集、预处理和推理路径。模型转换脚本见 `tools/convert_model.py`。

## 开源与贡献

项目自有代码使用 MIT License；第三方组件及其许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。欢迎提交问题与改进，详见 [CONTRIBUTING.md](CONTRIBUTING.md)。
