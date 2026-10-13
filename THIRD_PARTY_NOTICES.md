# Third-Party Notices

This project includes the following third-party source materials:

| Component | Included files | License / notice |
| --- | --- | --- |
| Dear ImGui | Core and GLFW/OpenGL3 backend subset in `third_party/imgui/` | MIT License; see `third_party/imgui/LICENSE.txt`. Vendored from upstream commit `d15966f`. |
| NVIDIA Video Codec SDK headers | `include/nvdec/` | NVIDIA permissive license notices are included in each header. |

OpenCV, CUDA, TensorRT, FFmpeg, GLFW, OpenGL, libuvc, libusb, and NVML are system dependencies and are not distributed with this repository. Obtain and use them under their respective terms. Optional UVC support requires a system libuvc development package; no libuvc binaries are bundled here.

The project MIT license applies to first-party project files only. It does not replace the licenses or notices of third-party components.
