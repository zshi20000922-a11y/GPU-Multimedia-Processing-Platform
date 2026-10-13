# Repository Guidelines

## Project Structure

This is a Linux/NVIDIA C++17 and CUDA application. Public headers live in `include/`; application and CUDA implementations live in `src/`; diagnostic tools and model conversion utilities live in `tools/`; architecture and build documentation live in `docs/`. Dear ImGui is vendored under `third_party/imgui/`. Keep third-party code and its notices separate from first-party contributions.

## Build and Run

Configure with local CUDA, GPU architecture, and TensorRT paths, then build:

```bash
cmake -S . -B build \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DTENSORRT_ROOT=/path/to/TensorRT
cmake --build build -j
./build/yolov5_sahi
```

See `docs/building.md` for system dependencies, camera diagnostics, and engine inspection commands. Do not assume a TensorRT plan is portable across GPU models or TensorRT versions.

## Style and Changes

Match nearby code: four-space indentation, opening braces on the declaration line, `snake_case` filenames/functions, `PascalCase` types, and trailing underscores for private members. Keep headers in `include/` aligned with implementation files. CUDA kernels belong in `.cu` files. No formatter or linter is configured; avoid unrelated formatting changes.

## Validation

Build from a fresh directory after changes. Use `engine_inspect` with a compatible external engine for TensorRT changes and `camera_bench` for capture/preprocessing changes when supported hardware is available. Report hardware checks that could not run. Keep model weights, ONNX/engine files, build output, logs, recordings, and local configuration out of commits.

## Commits and Pull Requests

Use concise imperative commit subjects, for example `Fix V4L2 buffer release`. Pull requests should describe behavior and trade-offs, give exact build/check commands and outcomes, identify GPU/CUDA/TensorRT versions for hardware results, and update both README languages when setup or user-visible behavior changes. Follow `CONTRIBUTING.md` and preserve all third-party notices.
