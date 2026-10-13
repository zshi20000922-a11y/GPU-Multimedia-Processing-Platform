#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Convert normalized FP32 network input to FP16 when an engine exposes FP16 I/O.
void launch_float_to_half(const float* src, void* dst, size_t elements, cudaStream_t stream);
void launch_half_to_float(const void* src, float* dst, size_t elements, cudaStream_t stream);

// 检测结果结构体
struct CUDADetection {
    float x1, y1, x2, y2;
    float conf;
    int class_id;
    int valid;  // 0=无效, 1=有效
};

// Letterbox kernel - 前处理（GPU）
void launch_letterbox(const uint8_t* src, int src_w, int src_h, int src_stride,
                      float* dst, int dst_w, int dst_h,
                      float scale, float pad_x, float pad_y,
                      cudaStream_t stream);

// NV12 Letterbox kernel - 融合前处理（GPU）
// 直接从 NV12 (YUV) 数据做 letterbox + RGB 转换 + 归一化 + HWC→CHW
// 避免 GPU→CPU→GPU 的数据往返
void launch_nv12_letterbox(const uint8_t* nv12, int src_w, int src_h, int pitch,
                           float* dst, int dst_w, int dst_h,
                           float scale, float pad_x, float pad_y,
                           cudaStream_t stream);

// NV12 Letterbox kernel with separate Y/UV plane pointers (for SAHI tile cropping)
void launch_nv12_tile_letterbox(const uint8_t* y_plane, const uint8_t* uv_plane,
                                 int src_w, int src_h, int y_pitch, int uv_pitch,
                                 float* dst, int dst_w, int dst_h,
                                 float scale, float pad_x, float pad_y,
                                 cudaStream_t stream);

// NMS kernel - 后处理（GPU）
void launch_nms(const float* output, int num_preds,
                float conf_thresh, float nms_thresh,
                CUDADetection* detections, int* num_detections,
                int max_detections, int orig_w, int orig_h,
                int input_size, cudaStream_t stream);

// 初始化检测输出缓冲区
void launch_init_detections(CUDADetection* detections, int max_detections,
                            cudaStream_t stream);

// BGR Letterbox kernel - 摄像头帧前处理（GPU）
// 输入: BGR uint8 (HWC), 输出: RGB float32 NCHW (640x640)
void launch_bgr_letterbox(const uint8_t* src_bgr, int src_w, int src_h, int src_stride,
                           float* dst, int dst_w, int dst_h,
                           float scale, float pad_x, float pad_y,
                           cudaStream_t stream);

// YUYV Letterbox kernel - 摄像头直通前处理（GPU）
// 输入: YUYV uint8 (每 4 字节 = 2 像素 [Y0 U Y1 V])
// 输出: RGB float32 NCHW (640x640)
// 跳过 CPU YUYV→BGR 转换，直接从 YUYV 到 TensorRT 输入
void launch_yuyv_letterbox(const uint8_t* src_yuyv, int src_w, int src_h, int src_stride,
                            float* dst, int dst_w, int dst_h,
                            float scale, float pad_x, float pad_y,
                            cudaStream_t stream, bool gray = false);

// NV12 → BGR + Resize kernel（GPU）
// 直接在 GPU 上完成 NV12→BGR 色彩转换 + 缩放到目标尺寸
// 避免全帧 D2H 拷贝，只拷贝缩小后的结果（1.5MB vs 18MB）
void launch_nv12_resize_bgr(const uint8_t* y_plane, const uint8_t* uv_plane,
                             int src_w, int src_h, int y_pitch, int uv_pitch,
                             uint8_t* dst_bgr, int dst_w, int dst_h,
                             cudaStream_t stream);

// YUYV → BGR + Resize kernel（GPU）
// 直接在 GPU 上完成 YUYV→BGR 色彩转换 + 缩放到目标尺寸
// 用于摄像头路径的显示帧，避免 CPU 端的 YUYV→BGR 转换
void launch_yuyv_resize_bgr(const uint8_t* src_yuyv, int src_w, int src_h, int src_stride,
                             uint8_t* dst_bgr, int dst_w, int dst_h,
                             cudaStream_t stream);

// NV12 → RGB + Resize kernel（GPU）
// 直接输出 RGB 格式，用于 CUDA-GL 互操作纹理（GL_RGB）
void launch_nv12_resize_rgb(const uint8_t* y_plane, const uint8_t* uv_plane,
                             int src_w, int src_h, int y_pitch, int uv_pitch,
                             uint8_t* dst_rgb, int dst_w, int dst_h,
                             cudaStream_t stream);

// YUYV → RGB + Resize kernel（GPU）
// 直接输出 RGB 格式，用于 CUDA-GL 互操作纹理（GL_RGB）
void launch_yuyv_resize_rgb(const uint8_t* src_yuyv, int src_w, int src_h, int src_stride,
                             uint8_t* dst_rgb, int dst_w, int dst_h,
                             cudaStream_t stream, bool gray = false);

// ==================== 10/12-bit 支持 ====================
// Y210/Y212 格式: uint16_t per component, 有效位在高位
//   bit_shift: 10-bit → 6 (16-10), 12-bit → 4 (16-12)
//   stride = width * 4 (每像素 4 字节: Y0[16] U[16] Y1[16] V[16])
//
// P010/P012 格式 (NV12 16-bit): uint16_t per component, 有效位在高位
//   Y plane stride = width * 2, UV plane stride = width * 2

// YUYV 16-bit Letterbox (Y210/Y212 → RGB float32 NCHW)
void launch_yuyv16_letterbox(const uint8_t* src, int src_w, int src_h, int src_stride,
                              float* dst, int dst_w, int dst_h,
                              float scale, float pad_x, float pad_y,
                              int bit_shift, cudaStream_t stream);

// NV12 16-bit Letterbox (P010/P012 → RGB float32 NCHW)
void launch_nv12_16bit_letterbox(const uint8_t* y_plane, const uint8_t* uv_plane,
                                  int src_w, int src_h, int y_pitch, int uv_pitch,
                                  float* dst, int dst_w, int dst_h,
                                  float scale, float pad_x, float pad_y,
                                  int bit_shift, cudaStream_t stream);

// YUYV 16-bit → BGR + Resize (用于显示)
void launch_yuyv16_resize_bgr(const uint8_t* src, int src_w, int src_h, int src_stride,
                               uint8_t* dst_bgr, int dst_w, int dst_h,
                               int bit_shift, cudaStream_t stream);

// YUYV 16-bit → RGB + Resize (用于 CUDA-GL 纹理)
void launch_yuyv16_resize_rgb(const uint8_t* src, int src_w, int src_h, int src_stride,
                               uint8_t* dst_rgb, int dst_w, int dst_h,
                               int bit_shift, cudaStream_t stream);

// NV12 16-bit → BGR + Resize (用于显示)
void launch_nv12_16bit_resize_bgr(const uint8_t* y_plane, const uint8_t* uv_plane,
                                   int src_w, int src_h, int y_pitch, int uv_pitch,
                                   uint8_t* dst_bgr, int dst_w, int dst_h,
                                   int bit_shift, cudaStream_t stream);

// NV12 16-bit → RGB + Resize (用于 CUDA-GL 纹理)
void launch_nv12_16bit_resize_rgb(const uint8_t* y_plane, const uint8_t* uv_plane,
                                   int src_w, int src_h, int y_pitch, int uv_pitch,
                                   uint8_t* dst_rgb, int dst_w, int dst_h,
                                   int bit_shift, cudaStream_t stream);
