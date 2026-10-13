# CUDA/TensorRT Multi-Stream Vision Pipeline

[中文](README.md) | **English**

A multi-input real-time vision pipeline for Linux and NVIDIA GPUs. It connects camera or video inputs to GPU preprocessing, TensorRT inference, postprocessing, GUI display, and recording.

> The current code targets Linux with NVIDIA CUDA and TensorRT. TensorRT engine plans are tied to GPU and TensorRT versions. Model weights, ONNX files, and engine plans are not included.

## Features and author contributions

| Area | Implementation |
| --- | --- |
| Multi-input capture | V4L2 DMA-BUF, CUDA-pinned, and fallback paths; FFmpeg CUDA video decode; optional system libuvc support. |
| GPU pipeline | Custom CUDA preprocessing for NV12, YUYV, and BGR plus FP16/FP32 conversion; per-channel CUDA streams, events, and a four-slot ring buffer. |
| Detection | TensorRT engine shape parsing; DIRECT and SAHI tiled inference, coordinate mapping, and result merging. |
| Display and recording | Dear ImGui multi-channel controls, fusion/panorama views, GPU monitoring, and display/raw-frame recording. |

These modules were implemented by the project author. See the [architecture guide](docs/architecture.md) for the component map and data flow. No frame-rate or accuracy guarantee is published until it is revalidated against this source version.

```mermaid
flowchart LR
    IN[Camera / video] --> CAP[Capture and hardware decode]
    CAP --> PIPE[Multi-channel CUDA pipeline]
    PIPE --> TRT[TensorRT / SAHI]
    TRT --> OUT[ImGui display and recording]
```

## Build

Requirements: CMake 3.18+, a C++17 compiler, CUDA Toolkit, TensorRT 10, NVIDIA driver/NVDEC, OpenCV, FFmpeg development libraries, GLFW, OpenGL, and pkg-config. CMake enables optional UVC input when the system libuvc development package is installed.

```bash
cmake -S . -B build \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DTENSORRT_ROOT=/path/to/TensorRT
cmake --build build -j
```

Set paths and the CUDA architecture for your system. See the [build guide](docs/building.md) for dependency details and diagnostics.

## Run

```bash
./build/yolov5_sahi
```

Select a local TensorRT engine and input source in the GUI. `engine_inspect` inspects engine metadata and can run an inference smoke check. `camera_bench` measures capture, preprocessing, and inference paths when a V4L2 device is available. Model conversion instructions are in `tools/convert_model.py`.

## License and contributions

First-party project code is licensed under MIT. Third-party terms are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Issues and focused contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
