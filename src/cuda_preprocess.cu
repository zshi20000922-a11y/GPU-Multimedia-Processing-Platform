/**
 * CUDA Letterbox Kernel
 * Single pass: BGR->RGB + normalize + HWC->CHW + letterbox
 */

#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <cstdint>

__global__ void letterbox_kernel(
    const uint8_t* __restrict__ src, int src_w, int src_h, int src_stride,
    float* __restrict__ dst, int dst_w, int dst_h,
    float scale, float pad_x, float pad_y)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float src_x = (dx - pad_x) / scale + 0.5f;
    float src_y = (dy - pad_y) / scale + 0.5f;

    float c0, c1, c2;
    if (src_x >= 0 && src_x < src_w && src_y >= 0 && src_y < src_h) {
        int x_low = __float2int_rd(src_x);
        int y_low = __float2int_rd(src_y);
        int x_high = min(x_low + 1, src_w - 1);
        int y_high = min(y_low + 1, src_h - 1);

        float lx = src_x - x_low, ly = src_y - y_low;
        float w1 = (1-ly)*(1-lx), w2 = (1-ly)*lx, w3 = ly*(1-lx), w4 = ly*lx;

        const uint8_t* v1 = src + y_low * src_stride + x_low * 3;
        const uint8_t* v2 = src + y_low * src_stride + x_high * 3;
        const uint8_t* v3 = src + y_high * src_stride + x_low * 3;
        const uint8_t* v4 = src + y_high * src_stride + x_high * 3;

        c0 = w1*v1[0] + w2*v2[0] + w3*v3[0] + w4*v4[0];
        c1 = w1*v1[1] + w2*v2[1] + w3*v3[1] + w4*v4[1];
        c2 = w1*v1[2] + w2*v2[2] + w3*v3[2] + w4*v4[2];
    } else {
        c0 = c1 = c2 = 114.0f;
    }

    int area = dst_w * dst_h;
    int idx = dy * dst_w + dx;
    dst[idx] = c2 / 255.0f;         // R
    dst[idx + area] = c1 / 255.0f;  // G
    dst[idx + 2*area] = c0 / 255.0f; // B
}

void launch_letterbox(const uint8_t* src, int src_w, int src_h, int src_stride,
                      float* dst, int dst_w, int dst_h,
                      float scale, float pad_x, float pad_y,
                      cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    letterbox_kernel<<<grid, block, 0, stream>>>(
        src, src_w, src_h, src_stride, dst, dst_w, dst_h, scale, pad_x, pad_y);
}

/**
 * NV12 Letterbox Kernel (融合版)
 * 单个 kernel 完成: NV12→RGB + letterbox + normalize(/255) + HWC→CHW
 * 直接从 GPU 上的 NV12 数据输出 TensorRT 输入，消除 GPU→CPU→GPU 往返
 *
 * NV12 格式:
 *   Y 平面: pitch × height 字节 (每像素 1 字节亮度)
 *   UV 平面: pitch × height/2 字节 (交错 U0 V0 U1 V1..., 2×2 像素共享一组 UV)
 */
__global__ void nv12_letterbox_kernel(
    const uint8_t* __restrict__ nv12, int src_w, int src_h, int pitch,
    float* __restrict__ dst, int dst_w, int dst_h,
    float scale, float pad_x, float pad_y)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    // 反向映射到原图坐标
    float src_x = (dx - pad_x) / scale + 0.5f;
    float src_y = (dy - pad_y) / scale + 0.5f;

    float y_val, u_val, v_val;

    if (src_x >= 0 && src_x < src_w && src_y >= 0 && src_y < src_h) {
        // 双线性插值采样 Y 平面
        int x_low = __float2int_rd(src_x);
        int y_low = __float2int_rd(src_y);
        int x_high = min(x_low + 1, src_w - 1);
        int y_high = min(y_low + 1, src_h - 1);

        float lx = src_x - x_low, ly = src_y - y_low;
        float w1 = (1-ly)*(1-lx), w2 = (1-ly)*lx, w3 = ly*(1-lx), w4 = ly*lx;

        // Y 平面采样
        float y1 = (float)nv12[y_low  * pitch + x_low];
        float y2 = (float)nv12[y_low  * pitch + x_high];
        float y3 = (float)nv12[y_high * pitch + x_low];
        float y4 = (float)nv12[y_high * pitch + x_high];
        y_val = w1*y1 + w2*y2 + w3*y3 + w4*y4;

        // UV 平面采样
        // NV12 UV 是交错的: [U0,V0, U1,V1, ...], 每 2×2 像素共享一个 UV 对
        // UV 平面起始: nv12 + src_h * pitch
        // UV 坐标: (pixel_x / 2, pixel_y / 2), U 在偶数位, V 在奇数位
        int uv_offset = src_h * pitch;
        int half_h = src_h / 2;

        // UV 空间中的浮点坐标
        float uv_src_x = src_x * 0.5f;
        float uv_src_y = src_y * 0.5f;

        int ux_low = __float2int_rd(uv_src_x);
        int uy_low = __float2int_rd(uv_src_y);
        int ux_high = min(ux_low + 1, src_w / 2 - 1);
        int uy_high = min(uy_low + 1, half_h - 1);

        float ulx = uv_src_x - ux_low;
        float uly = uv_src_y - uy_low;
        float uw1 = (1-uly)*(1-ulx), uw2 = (1-uly)*ulx, uw3 = uly*(1-ulx), uw4 = uly*ulx;

        // 4 个 UV 对的位置 (每个 UV 对占 2 字节: U 在 idx, V 在 idx+1)
        int uv_row0 = uv_offset + uy_low  * pitch;
        int uv_row1 = uv_offset + uy_high * pitch;
        int uv_col0 = ux_low * 2;
        int uv_col1 = ux_high * 2;

        u_val = uw1 * (float)nv12[uv_row0 + uv_col0]
              + uw2 * (float)nv12[uv_row0 + uv_col1]
              + uw3 * (float)nv12[uv_row1 + uv_col0]
              + uw4 * (float)nv12[uv_row1 + uv_col1];

        v_val = uw1 * (float)nv12[uv_row0 + uv_col0 + 1]
              + uw2 * (float)nv12[uv_row0 + uv_col1 + 1]
              + uw3 * (float)nv12[uv_row1 + uv_col0 + 1]
              + uw4 * (float)nv12[uv_row1 + uv_col1 + 1];
    } else {
        // 填充区域: 灰色 (114, 114, 114)
        y_val = 114.0f;
        u_val = 128.0f;
        v_val = 128.0f;
    }

    // NV12→RGB 转换 (BT.601)
    float r = 1.164f * (y_val - 16.0f) + 1.596f * (v_val - 128.0f);
    float g = 1.164f * (y_val - 16.0f) - 0.813f * (v_val - 128.0f) - 0.391f * (u_val - 128.0f);
    float b = 1.164f * (y_val - 16.0f) + 2.018f * (u_val - 128.0f);

    // 裁剪到 [0, 255] 并归一化
    r = fminf(255.0f, fmaxf(0.0f, r)) / 255.0f;
    g = fminf(255.0f, fmaxf(0.0f, g)) / 255.0f;
    b = fminf(255.0f, fmaxf(0.0f, b)) / 255.0f;

    // 写入 CHW 布局
    int area = dst_w * dst_h;
    int idx = dy * dst_w + dx;
    dst[idx] = r;              // R channel
    dst[idx + area] = g;       // G channel
    dst[idx + 2 * area] = b;   // B channel
}

void launch_nv12_letterbox(const uint8_t* nv12, int src_w, int src_h, int pitch,
                           float* dst, int dst_w, int dst_h,
                           float scale, float pad_x, float pad_y,
                           cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    nv12_letterbox_kernel<<<grid, block, 0, stream>>>(
        nv12, src_w, src_h, pitch, dst, dst_w, dst_h, scale, pad_x, pad_y);
}

/**
 * NV12 Tile Letterbox Kernel（SAHI 切片用）
 * 与 nv12_letterbox_kernel 相同功能，但接受分离的 Y/UV 平面指针
 * 用于从 NV12 帧中裁剪 tile 区域进行预处理
 * Y 平面: y_plane + tile_y * y_pitch
 * UV 平面: uv_plane + (tile_y / 2) * uv_pitch
 */
__global__ void nv12_tile_letterbox_kernel(
    const uint8_t* __restrict__ y_plane,
    const uint8_t* __restrict__ uv_plane,
    int src_w, int src_h, int y_pitch, int uv_pitch,
    float* __restrict__ dst, int dst_w, int dst_h,
    float scale, float pad_x, float pad_y)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    // 反向映射到原图坐标
    float src_x = (dx - pad_x) / scale + 0.5f;
    float src_y = (dy - pad_y) / scale + 0.5f;

    float y_val, u_val, v_val;

    if (src_x >= 0 && src_x < src_w && src_y >= 0 && src_y < src_h) {
        // 双线性插值采样 Y 平面
        int x_low = __float2int_rd(src_x);
        int y_low = __float2int_rd(src_y);
        int x_high = min(x_low + 1, src_w - 1);
        int y_high = min(y_low + 1, src_h - 1);

        float lx = src_x - x_low, ly = src_y - y_low;
        float w1 = (1-ly)*(1-lx), w2 = (1-ly)*lx, w3 = ly*(1-lx), w4 = ly*lx;

        // Y 平面采样（使用分离的 y_plane 指针和 y_pitch）
        float y1 = (float)y_plane[y_low  * y_pitch + x_low];
        float y2 = (float)y_plane[y_low  * y_pitch + x_high];
        float y3 = (float)y_plane[y_high * y_pitch + x_low];
        float y4 = (float)y_plane[y_high * y_pitch + x_high];
        y_val = w1*y1 + w2*y2 + w3*y3 + w4*y4;

        // UV 平面采样（使用分离的 uv_plane 指针和 uv_pitch）
        int half_h = src_h / 2;
        float uv_src_x = src_x * 0.5f;
        float uv_src_y = src_y * 0.5f;

        int ux_low = __float2int_rd(uv_src_x);
        int uy_low = __float2int_rd(uv_src_y);
        int ux_high = min(ux_low + 1, src_w / 2 - 1);
        int uy_high = min(uy_low + 1, half_h - 1);

        float ulx = uv_src_x - ux_low;
        float uly = uv_src_y - uy_low;
        float uw1 = (1-uly)*(1-ulx), uw2 = (1-uly)*ulx, uw3 = uly*(1-ulx), uw4 = uly*ulx;

        int uv_row0 = uy_low  * uv_pitch;
        int uv_row1 = uy_high * uv_pitch;
        int uv_col0 = ux_low * 2;
        int uv_col1 = ux_high * 2;

        u_val = uw1 * (float)uv_plane[uv_row0 + uv_col0]
              + uw2 * (float)uv_plane[uv_row0 + uv_col1]
              + uw3 * (float)uv_plane[uv_row1 + uv_col0]
              + uw4 * (float)uv_plane[uv_row1 + uv_col1];

        v_val = uw1 * (float)uv_plane[uv_row0 + uv_col0 + 1]
              + uw2 * (float)uv_plane[uv_row0 + uv_col1 + 1]
              + uw3 * (float)uv_plane[uv_row1 + uv_col0 + 1]
              + uw4 * (float)uv_plane[uv_row1 + uv_col1 + 1];
    } else {
        y_val = 114.0f;
        u_val = 128.0f;
        v_val = 128.0f;
    }

    // NV12→RGB 转换 (BT.601)
    float r = 1.164f * (y_val - 16.0f) + 1.596f * (v_val - 128.0f);
    float g = 1.164f * (y_val - 16.0f) - 0.813f * (v_val - 128.0f) - 0.391f * (u_val - 128.0f);
    float b = 1.164f * (y_val - 16.0f) + 2.018f * (u_val - 128.0f);

    r = fminf(255.0f, fmaxf(0.0f, r)) / 255.0f;
    g = fminf(255.0f, fmaxf(0.0f, g)) / 255.0f;
    b = fminf(255.0f, fmaxf(0.0f, b)) / 255.0f;

    int area = dst_w * dst_h;
    int idx = dy * dst_w + dx;
    dst[idx] = r;
    dst[idx + area] = g;
    dst[idx + 2 * area] = b;
}

void launch_nv12_tile_letterbox(const uint8_t* y_plane, const uint8_t* uv_plane,
                                 int src_w, int src_h, int y_pitch, int uv_pitch,
                                 float* dst, int dst_w, int dst_h,
                                 float scale, float pad_x, float pad_y,
                                 cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    nv12_tile_letterbox_kernel<<<grid, block, 0, stream>>>(
        y_plane, uv_plane, src_w, src_h, y_pitch, uv_pitch,
        dst, dst_w, dst_h, scale, pad_x, pad_y);
}

/**
 * BGR Letterbox Kernel（摄像头用）
 * 单个 kernel 完成: BGR→RGB + letterbox + normalize(/255) + HWC→CHW
 * 输入: BGR uint8 HWC 格式（OpenCV 默认格式）
 * 输出: RGB float32 NCHW 格式（TensorRT 输入）
 */
__global__ void bgr_letterbox_kernel(
    const uint8_t* __restrict__ src, int src_w, int src_h, int src_stride,
    float* __restrict__ dst, int dst_w, int dst_h,
    float scale, float pad_x, float pad_y)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float src_x = (dx - pad_x) / scale + 0.5f;
    float src_y = (dy - pad_y) / scale + 0.5f;

    float c0, c1, c2;  // B, G, R

    if (src_x >= 0 && src_x < src_w && src_y >= 0 && src_y < src_h) {
        int x_low = __float2int_rd(src_x);
        int y_low = __float2int_rd(src_y);
        int x_high = min(x_low + 1, src_w - 1);
        int y_high = min(y_low + 1, src_h - 1);

        float lx = src_x - x_low, ly = src_y - y_low;
        float w1 = (1-ly)*(1-lx), w2 = (1-ly)*lx, w3 = ly*(1-lx), w4 = ly*lx;

        const uint8_t* v1 = src + y_low * src_stride + x_low * 3;
        const uint8_t* v2 = src + y_low * src_stride + x_high * 3;
        const uint8_t* v3 = src + y_high * src_stride + x_low * 3;
        const uint8_t* v4 = src + y_high * src_stride + x_high * 3;

        // 双线性插值 BGR
        c0 = w1*v1[0] + w2*v2[0] + w3*v3[0] + w4*v4[0];  // B
        c1 = w1*v1[1] + w2*v2[1] + w3*v3[1] + w4*v4[1];  // G
        c2 = w1*v1[2] + w2*v2[2] + w3*v3[2] + w4*v4[2];  // R
    } else {
        c0 = c1 = c2 = 114.0f;  // 灰色填充
    }

    // 归一化 + 写入 CHW（BGR→RGB: c2=R, c1=G, c0=B）
    int area = dst_w * dst_h;
    int idx = dy * dst_w + dx;
    dst[idx] = c2 / 255.0f;              // R
    dst[idx + area] = c1 / 255.0f;       // G
    dst[idx + 2 * area] = c0 / 255.0f;   // B
}

void launch_bgr_letterbox(const uint8_t* src_bgr, int src_w, int src_h, int src_stride,
                           float* dst, int dst_w, int dst_h,
                           float scale, float pad_x, float pad_y,
                           cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    bgr_letterbox_kernel<<<grid, block, 0, stream>>>(
        src_bgr, src_w, src_h, src_stride, dst, dst_w, dst_h, scale, pad_x, pad_y);
}

// ==================== NV12 → BGR + Resize kernel ====================
// 在 GPU 上直接完成 NV12→BGR 色彩转换 + 缩放到目标尺寸
// 避免全帧 D2H 拷贝，只拷贝缩小后的结果

__global__ void nv12_resize_bgr_kernel(
    const uint8_t* __restrict__ y_plane,
    const uint8_t* __restrict__ uv_plane,
    int src_w, int src_h, int y_pitch, int uv_pitch,
    uint8_t* __restrict__ dst_bgr, int dst_w, int dst_h)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    // 计算源坐标（双线性插值采样）
    float sx = (float)dx * src_w / dst_w;
    float sy = (float)dy * src_h / dst_h;

    // Y 分量（双线性插值）
    int x0 = __float2int_rd(sx), y0 = __float2int_rd(sy);
    int x1 = min(x0 + 1, src_w - 1), y1 = min(y0 + 1, src_h - 1);
    float fx = sx - x0, fy = sy - y0;

    float y_val = (1-fx)*(1-fy) * y_plane[y0 * y_pitch + x0]
                + fx*(1-fy)     * y_plane[y0 * y_pitch + x1]
                + (1-fx)*fy     * y_plane[y1 * y_pitch + x0]
                + fx*fy         * y_plane[y1 * y_pitch + x1];

    // UV 分量（NV12: 交错 U V，每 2×2 像素共享一个 UV）
    int uv_x = (x0 >> 1) * 2;  // 字节偏移（U V 交错）
    int uv_y = y0 >> 1;
    float u_val = uv_plane[uv_y * uv_pitch + uv_x] - 128.0f;
    float v_val = uv_plane[uv_y * uv_pitch + uv_x + 1] - 128.0f;

    // YUV → BGR 转换（BT.601）
    // Video-range BT.601 (FFmpeg NV12 is normally limited-range).
    float c = fmaxf(0.0f, y_val - 16.0f);
    float r = 1.164f * c + 1.596f * v_val;
    float g = 1.164f * c - 0.391f * u_val - 0.813f * v_val;
    float b = 1.164f * c + 2.018f * u_val;

    // 裁剪到 [0, 255]
    uint8_t rb = (uint8_t)max(0.0f, min(255.0f, r));
    uint8_t gb = (uint8_t)max(0.0f, min(255.0f, g));
    uint8_t bb = (uint8_t)max(0.0f, min(255.0f, b));

    // 写入 BGR 格式
    int idx = (dy * dst_w + dx) * 3;
    dst_bgr[idx]     = bb;
    dst_bgr[idx + 1] = gb;
    dst_bgr[idx + 2] = rb;
}

void launch_nv12_resize_bgr(const uint8_t* y_plane, const uint8_t* uv_plane,
                             int src_w, int src_h, int y_pitch, int uv_pitch,
                             uint8_t* dst_bgr, int dst_w, int dst_h,
                             cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    nv12_resize_bgr_kernel<<<grid, block, 0, stream>>>(
        y_plane, uv_plane, src_w, src_h, y_pitch, uv_pitch,
        dst_bgr, dst_w, dst_h);
}

// ==================== NV12 → RGB + Resize kernel ====================
// 直接输出 RGB 格式，用于 CUDA-GL 互操作纹理（GL_RGB）

__global__ void nv12_resize_rgb_kernel(
    const uint8_t* __restrict__ y_plane,
    const uint8_t* __restrict__ uv_plane,
    int src_w, int src_h, int y_pitch, int uv_pitch,
    uint8_t* __restrict__ dst_rgb, int dst_w, int dst_h)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float sx = (float)dx * src_w / dst_w;
    float sy = (float)dy * src_h / dst_h;
    int x0 = __float2int_rd(sx), y0 = __float2int_rd(sy);
    int x1 = min(x0 + 1, src_w - 1), y1 = min(y0 + 1, src_h - 1);
    float fx = sx - x0, fy = sy - y0;

    float y_val = (1-fx)*(1-fy) * y_plane[y0 * y_pitch + x0]
                + fx*(1-fy)     * y_plane[y0 * y_pitch + x1]
                + (1-fx)*fy     * y_plane[y1 * y_pitch + x0]
                + fx*fy         * y_plane[y1 * y_pitch + x1];

    int uv_x = (x0 >> 1) * 2;
    int uv_y = y0 >> 1;
    float u_val = uv_plane[uv_y * uv_pitch + uv_x] - 128.0f;
    float v_val = uv_plane[uv_y * uv_pitch + uv_x + 1] - 128.0f;

    // Video-range BT.601 (FFmpeg NV12 is normally limited-range).
    float c = fmaxf(0.0f, y_val - 16.0f);
    float r = 1.164f * c + 1.596f * v_val;
    float g = 1.164f * c - 0.391f * u_val - 0.813f * v_val;
    float b = 1.164f * c + 2.018f * u_val;

    // 写入 RGB 格式（直接适配 GL_RGB 纹理）
    int idx = (dy * dst_w + dx) * 3;
    dst_rgb[idx]     = (uint8_t)max(0.0f, min(255.0f, r));
    dst_rgb[idx + 1] = (uint8_t)max(0.0f, min(255.0f, g));
    dst_rgb[idx + 2] = (uint8_t)max(0.0f, min(255.0f, b));
}

void launch_nv12_resize_rgb(const uint8_t* y_plane, const uint8_t* uv_plane,
                             int src_w, int src_h, int y_pitch, int uv_pitch,
                             uint8_t* dst_rgb, int dst_w, int dst_h,
                             cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    nv12_resize_rgb_kernel<<<grid, block, 0, stream>>>(
        y_plane, uv_plane, src_w, src_h, y_pitch, uv_pitch,
        dst_rgb, dst_w, dst_h);
}

// ==================== YUYV → BGR + Resize kernel ====================
// 在 GPU 上直接完成 YUYV→BGR 色彩转换 + 缩放到目标尺寸
// 用于摄像头路径的显示帧，避免 CPU 端的 YUYV→BGR 转换

__global__ void yuyv_resize_bgr_kernel(
    const uint8_t* __restrict__ src_yuyv,
    int src_w, int src_h, int src_stride,
    uint8_t* __restrict__ dst_bgr, int dst_w, int dst_h)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    // 计算源坐标（双线性插值采样）
    float sx = (float)dx * src_w / dst_w;
    float sy = (float)dy * src_h / dst_h;
    int x0 = __float2int_rd(sx), y0 = __float2int_rd(sy);
    int x1 = min(x0 + 1, src_w - 1), y1 = min(y0 + 1, src_h - 1);
    float fx = sx - x0, fy = sy - y0;

    // YUYV: 每 4 字节 = 2 像素 [Y0, U, Y1, V]
    auto read_yuyv_bgr = [&](int px, int py) -> float3 {
        int pair_idx = px / 2;
        int byte_offset = py * src_stride + pair_idx * 4;
        int is_odd = px & 1;
        float y_val = (float)src_yuyv[byte_offset + is_odd * 2];
        float u_val = (float)src_yuyv[byte_offset + 1] - 128.0f;
        float v_val = (float)src_yuyv[byte_offset + 3] - 128.0f;
        float r = y_val + 1.402f * v_val;
        float g = y_val - 0.344f * u_val - 0.714f * v_val;
        float b = y_val + 1.772f * u_val;
        return make_float3(r, g, b);
    };

    // 双线性插值
    float3 c00 = read_yuyv_bgr(x0, y0);
    float3 c10 = read_yuyv_bgr(x1, y0);
    float3 c01 = read_yuyv_bgr(x0, y1);
    float3 c11 = read_yuyv_bgr(x1, y1);

    float r = (1-fx)*(1-fy)*c00.x + fx*(1-fy)*c10.x + (1-fx)*fy*c01.x + fx*fy*c11.x;
    float g = (1-fx)*(1-fy)*c00.y + fx*(1-fy)*c10.y + (1-fx)*fy*c01.y + fx*fy*c11.y;
    float b = (1-fx)*(1-fy)*c00.z + fx*(1-fy)*c10.z + (1-fx)*fy*c01.z + fx*fy*c11.z;

    // 写入 BGR 格式
    int idx = (dy * dst_w + dx) * 3;
    dst_bgr[idx]     = (uint8_t)max(0.0f, min(255.0f, b));
    dst_bgr[idx + 1] = (uint8_t)max(0.0f, min(255.0f, g));
    dst_bgr[idx + 2] = (uint8_t)max(0.0f, min(255.0f, r));
}

void launch_yuyv_resize_bgr(const uint8_t* src_yuyv, int src_w, int src_h, int src_stride,
                             uint8_t* dst_bgr, int dst_w, int dst_h,
                             cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    yuyv_resize_bgr_kernel<<<grid, block, 0, stream>>>(
        src_yuyv, src_w, src_h, src_stride, dst_bgr, dst_w, dst_h);
}

// ==================== YUYV → RGB + Resize kernel ====================
// 直接输出 RGB 格式，用于 CUDA-GL 互操作纹理（GL_RGB）

__global__ void yuyv_resize_rgb_kernel(
    const uint8_t* __restrict__ src_yuyv,
    int src_w, int src_h, int src_stride,
    uint8_t* __restrict__ dst_rgb, int dst_w, int dst_h, bool gray)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float sx = (float)dx * src_w / dst_w;
    float sy = (float)dy * src_h / dst_h;
    int x0 = __float2int_rd(sx), y0 = __float2int_rd(sy);
    int x1 = min(x0 + 1, src_w - 1), y1 = min(y0 + 1, src_h - 1);
    float fx = sx - x0, fy = sy - y0;

    auto read_yuyv_rgb = [&](int px, int py) -> float3 {
        // GREY (V4L2_PIX_FMT_GREY): 8-bit 单通道灰度，每像素 1 字节，stride = width
        // 线性寻址 py*stride + px，R=G=B=Y，无色度
        if (gray) {
            float y_val = (float)src_yuyv[py * src_stride + px];
            return make_float3(y_val, y_val, y_val);
        }
        int pair_idx = px / 2;
        int byte_offset = py * src_stride + pair_idx * 4;
        int is_odd = px & 1;
        float y_val = (float)src_yuyv[byte_offset + is_odd * 2];
        float u_val = (float)src_yuyv[byte_offset + 1] - 128.0f;
        float v_val = (float)src_yuyv[byte_offset + 3] - 128.0f;
        float r = y_val + 1.402f * v_val;
        float g = y_val - 0.344f * u_val - 0.714f * v_val;
        float b = y_val + 1.772f * u_val;
        return make_float3(r, g, b);
    };

    float3 c00 = read_yuyv_rgb(x0, y0);
    float3 c10 = read_yuyv_rgb(x1, y0);
    float3 c01 = read_yuyv_rgb(x0, y1);
    float3 c11 = read_yuyv_rgb(x1, y1);

    float r = (1-fx)*(1-fy)*c00.x + fx*(1-fy)*c10.x + (1-fx)*fy*c01.x + fx*fy*c11.x;
    float g = (1-fx)*(1-fy)*c00.y + fx*(1-fy)*c10.y + (1-fx)*fy*c01.y + fx*fy*c11.y;
    float b = (1-fx)*(1-fy)*c00.z + fx*(1-fy)*c10.z + (1-fx)*fy*c01.z + fx*fy*c11.z;

    // 写入 RGB 格式
    int idx = (dy * dst_w + dx) * 3;
    dst_rgb[idx]     = (uint8_t)max(0.0f, min(255.0f, r));
    dst_rgb[idx + 1] = (uint8_t)max(0.0f, min(255.0f, g));
    dst_rgb[idx + 2] = (uint8_t)max(0.0f, min(255.0f, b));
}

void launch_yuyv_resize_rgb(const uint8_t* src_yuyv, int src_w, int src_h, int src_stride,
                             uint8_t* dst_rgb, int dst_w, int dst_h,
                             cudaStream_t stream, bool gray)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    yuyv_resize_rgb_kernel<<<grid, block, 0, stream>>>(
        src_yuyv, src_w, src_h, src_stride, dst_rgb, dst_w, dst_h, gray);
}

/**
 * YUYV Letterbox Kernel（摄像头直通）
 * 单个 kernel 完成: YUYV→RGB + letterbox + normalize(/255) + HWC→CHW
 * 跳过 CPU 的 YUYV→BGR 转换，直接从 V4L2 的 YUYV 数据到 TensorRT 输入
 *
 * YUYV 格式: 每 4 字节 = 2 像素 [Y0, U, Y1, V]
 *   Y0 和 Y1 共享同一组 U, V
 *   像素 x 的 Y = data[(x/2)*4 + (x%2)*2]
 *   像素 x 的 U = data[(x/2)*4 + 1]
 *   像素 x 的 V = data[(x/2)*4 + 3]
 */
__global__ void yuyv_letterbox_kernel(
    const uint8_t* __restrict__ src, int src_w, int src_h, int src_stride,
    float* __restrict__ dst, int dst_w, int dst_h,
    float scale, float pad_x, float pad_y, bool gray)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float src_x = (dx - pad_x) / scale + 0.5f;
    float src_y = (dy - pad_y) / scale + 0.5f;

    float y_val, u_val, v_val;
    bool inside = (src_x >= 0 && src_x < src_w && src_y >= 0 && src_y < src_h);

    if (inside) {
        int x = __float2int_rd(src_x);
        int y = __float2int_rd(src_y);
        x = min(max(x, 0), src_w - 1);
        y = min(max(y, 0), src_h - 1);

        if (gray) {
            // GREY (V4L2_PIX_FMT_GREY): 8-bit 单通道灰度，每像素 1 字节，stride = width
            y_val = (float)src[y * src_stride + x];
            u_val = 128.0f;
            v_val = 128.0f;
        } else {
            // 从 YUYV 提取 Y/U/V（最近邻采样）
            int pair_idx = x / 2;
            int byte_offset = y * src_stride + pair_idx * 4;
            int is_odd = x & 1;

            y_val = (float)src[byte_offset + is_odd * 2];       // Y0(偶数像素) 或 Y1(奇数像素)
            u_val = (float)src[byte_offset + 1];                 // U（共享）
            v_val = (float)src[byte_offset + 3];                 // V（共享）
        }
    } else {
        y_val = 128.0f;
        u_val = 128.0f;
        v_val = 128.0f;
    }

    // YUV → RGB 转换 (BT.601)
    float r, g, b;
    if (gray) {
        // GREY：每个字节即灰度像素，R=G=B=Y，跳过色度运算
        r = g = b = y_val;
    } else {
        float c = y_val - 16.0f;
        float d = u_val - 128.0f;
        float e = v_val - 128.0f;
        r = 1.164f * c + 1.596f * e;
        g = 1.164f * c - 0.392f * d - 0.813f * e;
        b = 1.164f * c + 2.017f * d;
    }

    // 裁剪 + 归一化
    r = fminf(255.0f, fmaxf(0.0f, r)) / 255.0f;
    g = fminf(255.0f, fmaxf(0.0f, g)) / 255.0f;
    b = fminf(255.0f, fmaxf(0.0f, b)) / 255.0f;

    // 写入 CHW 布局
    int area = dst_w * dst_h;
    int idx = dy * dst_w + dx;
    dst[idx] = r;              // R channel
    dst[idx + area] = g;       // G channel
    dst[idx + 2 * area] = b;   // B channel
}

void launch_yuyv_letterbox(const uint8_t* src_yuyv, int src_w, int src_h, int src_stride,
                            float* dst, int dst_w, int dst_h,
                            float scale, float pad_x, float pad_y,
                            cudaStream_t stream, bool gray)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    yuyv_letterbox_kernel<<<grid, block, 0, stream>>>(
        src_yuyv, src_w, src_h, src_stride, dst, dst_w, dst_h, scale, pad_x, pad_y, gray);
}

// ==================== 10/12-bit YUYV Letterbox (Y210/Y212) ====================
// Y210/Y212: 每像素 4 字节, 布局 [Y0_lo Y0_hi U_lo U_hi Y1_lo Y1_hi V_lo V_hi]
//   uint16_t per component, 有效位在高位 (LE: 低字节在前)
//   bit_shift: 10-bit → 6, 12-bit → 4
//   stride = width * 4

__global__ void yuyv16_letterbox_kernel(
    const uint16_t* __restrict__ src, int src_w, int src_h, int src_stride_words,
    float* __restrict__ dst, int dst_w, int dst_h,
    float scale, float pad_x, float pad_y, int bit_shift)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float max_val = (float)((1 << (16 - bit_shift)) - 1);  // 1023 或 4095
    float inv_max = 1.0f / max_val;

    float src_x = (dx - pad_x) / scale + 0.5f;
    float src_y = (dy - pad_y) / scale + 0.5f;

    float y_val, u_val, v_val;

    if (src_x >= 0 && src_x < src_w && src_y >= 0 && src_y < src_h) {
        int x = __float2int_rd(src_x);
        int y = __float2int_rd(src_y);
        x = min(max(x, 0), src_w - 1);
        y = min(max(y, 0), src_h - 1);

        // Y210/Y212: 每 2 像素 = 4 个 uint16_t = 8 字节
        // [Y0, U, Y1, V] 各 16-bit
        int pair_idx = x / 2;
        int word_offset = y * src_stride_words + pair_idx * 4;
        int is_odd = x & 1;

        y_val = (float)(src[word_offset + is_odd * 2] >> bit_shift);
        u_val = (float)(src[word_offset + 1] >> bit_shift);
        v_val = (float)(src[word_offset + 3] >> bit_shift);
    } else {
        y_val = max_val * 0.5f;
        u_val = max_val * 0.5f;
        v_val = max_val * 0.5f;
    }

    // YUV → RGB (BT.601), 归一化到 [0,1]
    float c = y_val - max_val * 0.0625f;     // 16/256 = 0.0625
    float d = u_val - max_val * 0.5f;        // 128/256 = 0.5
    float e = v_val - max_val * 0.5f;

    float r = (1.164f * c + 1.596f * e) * inv_max;
    float g = (1.164f * c - 0.392f * d - 0.813f * e) * inv_max;
    float b = (1.164f * c + 2.017f * d) * inv_max;

    r = fminf(1.0f, fmaxf(0.0f, r));
    g = fminf(1.0f, fmaxf(0.0f, g));
    b = fminf(1.0f, fmaxf(0.0f, b));

    int area = dst_w * dst_h;
    int idx = dy * dst_w + dx;
    dst[idx] = r;
    dst[idx + area] = g;
    dst[idx + 2 * area] = b;
}

void launch_yuyv16_letterbox(const uint8_t* src, int src_w, int src_h, int src_stride,
                              float* dst, int dst_w, int dst_h,
                              float scale, float pad_x, float pad_y,
                              int bit_shift, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    // src_stride 是字节数, kernel 里用 uint16_t 指针, 所以 stride 要除以 2
    int src_stride_words = src_stride / 2;
    yuyv16_letterbox_kernel<<<grid, block, 0, stream>>>(
        (const uint16_t*)src, src_w, src_h, src_stride_words,
        dst, dst_w, dst_h, scale, pad_x, pad_y, bit_shift);
}

// ==================== 10/12-bit NV12 Letterbox (P010/P012) ====================
// P010/P012 NV12: Y 平面和 UV 平面都是 uint16_t per component
//   Y 平面: width * 2 字节/行, UV 平面: width * 2 字节/行
//   有效位在高位, bit_shift: 10-bit → 6, 12-bit → 4

__global__ void nv12_16bit_letterbox_kernel(
    const uint16_t* __restrict__ y_plane,
    const uint16_t* __restrict__ uv_plane,
    int src_w, int src_h, int y_pitch_words, int uv_pitch_words,
    float* __restrict__ dst, int dst_w, int dst_h,
    float scale, float pad_x, float pad_y, int bit_shift)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float max_val = (float)((1 << (16 - bit_shift)) - 1);
    float inv_max = 1.0f / max_val;

    float src_x = (dx - pad_x) / scale + 0.5f;
    float src_y = (dy - pad_y) / scale + 0.5f;

    float y_val, u_val, v_val;

    if (src_x >= 0 && src_x < src_w && src_y >= 0 && src_y < src_h) {
        int x_low = __float2int_rd(src_x);
        int y_low = __float2int_rd(src_y);
        int x_high = min(x_low + 1, src_w - 1);
        int y_high = min(y_low + 1, src_h - 1);

        float lx = src_x - x_low, ly = src_y - y_low;
        float w1 = (1-ly)*(1-lx), w2 = (1-ly)*lx, w3 = ly*(1-lx), w4 = ly*lx;

        // Y 平面双线性插值
        float y1 = (float)(y_plane[y_low  * y_pitch_words + x_low] >> bit_shift);
        float y2 = (float)(y_plane[y_low  * y_pitch_words + x_high] >> bit_shift);
        float y3 = (float)(y_plane[y_high * y_pitch_words + x_low] >> bit_shift);
        float y4 = (float)(y_plane[y_high * y_pitch_words + x_high] >> bit_shift);
        y_val = w1*y1 + w2*y2 + w3*y3 + w4*y4;

        // UV 平面采样 (交错 U V, 每 2×2 共享)
        float uv_src_x = src_x * 0.5f;
        float uv_src_y = src_y * 0.5f;
        int ux_low = __float2int_rd(uv_src_x);
        int uy_low = __float2int_rd(uv_src_y);
        int ux_high = min(ux_low + 1, src_w / 2 - 1);
        int uy_high = min(uy_low + 1, src_h / 2 - 1);

        float ulx = uv_src_x - ux_low;
        float uly = uv_src_y - uy_low;
        float uw1 = (1-uly)*(1-ulx), uw2 = (1-uly)*ulx;
        float uw3 = uly*(1-ulx), uw4 = uly*ulx;

        int uv_row0 = uy_low * uv_pitch_words;
        int uv_row1 = uy_high * uv_pitch_words;
        int uv_col0 = ux_low * 2;
        int uv_col1 = ux_high * 2;

        u_val = uw1 * (float)(uv_plane[uv_row0 + uv_col0] >> bit_shift)
              + uw2 * (float)(uv_plane[uv_row0 + uv_col1] >> bit_shift)
              + uw3 * (float)(uv_plane[uv_row1 + uv_col0] >> bit_shift)
              + uw4 * (float)(uv_plane[uv_row1 + uv_col1] >> bit_shift);

        v_val = uw1 * (float)(uv_plane[uv_row0 + uv_col0 + 1] >> bit_shift)
              + uw2 * (float)(uv_plane[uv_row0 + uv_col1 + 1] >> bit_shift)
              + uw3 * (float)(uv_plane[uv_row1 + uv_col0 + 1] >> bit_shift)
              + uw4 * (float)(uv_plane[uv_row1 + uv_col1 + 1] >> bit_shift);
    } else {
        y_val = max_val * 0.4453f;   // 114/256
        u_val = max_val * 0.5f;
        v_val = max_val * 0.5f;
    }

    // NV12→RGB (BT.601), 归一化到 [0,1]
    float ny = y_val * inv_max;
    float nu = u_val * inv_max - 0.5f;
    float nv = v_val * inv_max - 0.5f;

    float r = ny + 1.402f * nv;
    float g = ny - 0.344f * nu - 0.714f * nv;
    float b = ny + 1.772f * nu;

    r = fminf(1.0f, fmaxf(0.0f, r));
    g = fminf(1.0f, fmaxf(0.0f, g));
    b = fminf(1.0f, fmaxf(0.0f, b));

    int area = dst_w * dst_h;
    int idx = dy * dst_w + dx;
    dst[idx] = r;
    dst[idx + area] = g;
    dst[idx + 2 * area] = b;
}

void launch_nv12_16bit_letterbox(const uint8_t* y_plane, const uint8_t* uv_plane,
                                  int src_w, int src_h, int y_pitch, int uv_pitch,
                                  float* dst, int dst_w, int dst_h,
                                  float scale, float pad_x, float pad_y,
                                  int bit_shift, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    nv12_16bit_letterbox_kernel<<<grid, block, 0, stream>>>(
        (const uint16_t*)y_plane, (const uint16_t*)uv_plane,
        src_w, src_h, y_pitch / 2, uv_pitch / 2,
        dst, dst_w, dst_h, scale, pad_x, pad_y, bit_shift);
}

// ==================== 10/12-bit YUYV → BGR + Resize ====================

__global__ void yuyv16_resize_bgr_kernel(
    const uint16_t* __restrict__ src,
    int src_w, int src_h, int src_stride_words,
    uint8_t* __restrict__ dst_bgr, int dst_w, int dst_h,
    int bit_shift)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float max_val = (float)((1 << (16 - bit_shift)) - 1);
    float inv_max = 1.0f / max_val;

    float sx = (float)dx * src_w / dst_w;
    float sy = (float)dy * src_h / dst_h;
    int x0 = __float2int_rd(sx), y0 = __float2int_rd(sy);
    int x1 = min(x0 + 1, src_w - 1), y1 = min(y0 + 1, src_h - 1);
    float fx = sx - x0, fy = sy - y0;

    auto read_pixel = [&](int px, int py) -> float3 {
        int pair_idx = px / 2;
        int word_offset = py * src_stride_words + pair_idx * 4;
        int is_odd = px & 1;
        float y_val = (float)(src[word_offset + is_odd * 2] >> bit_shift);
        float u_val = (float)(src[word_offset + 1] >> bit_shift) - max_val * 0.5f;
        float v_val = (float)(src[word_offset + 3] >> bit_shift) - max_val * 0.5f;
        float r = y_val + 1.402f * v_val;
        float g = y_val - 0.344f * u_val - 0.714f * v_val;
        float b = y_val + 1.772f * u_val;
        return make_float3(r, g, b);
    };

    float3 c00 = read_pixel(x0, y0);
    float3 c10 = read_pixel(x1, y0);
    float3 c01 = read_pixel(x0, y1);
    float3 c11 = read_pixel(x1, y1);

    float scale_factor = 255.0f * inv_max;
    float r = ((1-fx)*(1-fy)*c00.x + fx*(1-fy)*c10.x + (1-fx)*fy*c01.x + fx*fy*c11.x) * scale_factor;
    float g = ((1-fx)*(1-fy)*c00.y + fx*(1-fy)*c10.y + (1-fx)*fy*c01.y + fx*fy*c11.y) * scale_factor;
    float b = ((1-fx)*(1-fy)*c00.z + fx*(1-fy)*c10.z + (1-fx)*fy*c01.z + fx*fy*c11.z) * scale_factor;

    int idx = (dy * dst_w + dx) * 3;
    dst_bgr[idx]     = (uint8_t)max(0.0f, min(255.0f, b));
    dst_bgr[idx + 1] = (uint8_t)max(0.0f, min(255.0f, g));
    dst_bgr[idx + 2] = (uint8_t)max(0.0f, min(255.0f, r));
}

void launch_yuyv16_resize_bgr(const uint8_t* src, int src_w, int src_h, int src_stride,
                               uint8_t* dst_bgr, int dst_w, int dst_h,
                               int bit_shift, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    yuyv16_resize_bgr_kernel<<<grid, block, 0, stream>>>(
        (const uint16_t*)src, src_w, src_h, src_stride / 2,
        dst_bgr, dst_w, dst_h, bit_shift);
}

// ==================== 10/12-bit YUYV → RGB + Resize ====================

__global__ void yuyv16_resize_rgb_kernel(
    const uint16_t* __restrict__ src,
    int src_w, int src_h, int src_stride_words,
    uint8_t* __restrict__ dst_rgb, int dst_w, int dst_h,
    int bit_shift)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float max_val = (float)((1 << (16 - bit_shift)) - 1);
    float inv_max = 1.0f / max_val;

    float sx = (float)dx * src_w / dst_w;
    float sy = (float)dy * src_h / dst_h;
    int x0 = __float2int_rd(sx), y0 = __float2int_rd(sy);
    int x1 = min(x0 + 1, src_w - 1), y1 = min(y0 + 1, src_h - 1);
    float fx = sx - x0, fy = sy - y0;

    auto read_pixel = [&](int px, int py) -> float3 {
        int pair_idx = px / 2;
        int word_offset = py * src_stride_words + pair_idx * 4;
        int is_odd = px & 1;
        float y_val = (float)(src[word_offset + is_odd * 2] >> bit_shift);
        float u_val = (float)(src[word_offset + 1] >> bit_shift) - max_val * 0.5f;
        float v_val = (float)(src[word_offset + 3] >> bit_shift) - max_val * 0.5f;
        float r = y_val + 1.402f * v_val;
        float g = y_val - 0.344f * u_val - 0.714f * v_val;
        float b = y_val + 1.772f * u_val;
        return make_float3(r, g, b);
    };

    float3 c00 = read_pixel(x0, y0);
    float3 c10 = read_pixel(x1, y0);
    float3 c01 = read_pixel(x0, y1);
    float3 c11 = read_pixel(x1, y1);

    float scale_factor = 255.0f * inv_max;
    float r = ((1-fx)*(1-fy)*c00.x + fx*(1-fy)*c10.x + (1-fx)*fy*c01.x + fx*fy*c11.x) * scale_factor;
    float g = ((1-fx)*(1-fy)*c00.y + fx*(1-fy)*c10.y + (1-fx)*fy*c01.y + fx*fy*c11.y) * scale_factor;
    float b = ((1-fx)*(1-fy)*c00.z + fx*(1-fy)*c10.z + (1-fx)*fy*c01.z + fx*fy*c11.z) * scale_factor;

    int idx = (dy * dst_w + dx) * 3;
    dst_rgb[idx]     = (uint8_t)max(0.0f, min(255.0f, r));
    dst_rgb[idx + 1] = (uint8_t)max(0.0f, min(255.0f, g));
    dst_rgb[idx + 2] = (uint8_t)max(0.0f, min(255.0f, b));
}

void launch_yuyv16_resize_rgb(const uint8_t* src, int src_w, int src_h, int src_stride,
                               uint8_t* dst_rgb, int dst_w, int dst_h,
                               int bit_shift, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    yuyv16_resize_rgb_kernel<<<grid, block, 0, stream>>>(
        (const uint16_t*)src, src_w, src_h, src_stride / 2,
        dst_rgb, dst_w, dst_h, bit_shift);
}

// ==================== 10/12-bit NV12 → BGR + Resize ====================

__global__ void nv12_16bit_resize_bgr_kernel(
    const uint16_t* __restrict__ y_plane,
    const uint16_t* __restrict__ uv_plane,
    int src_w, int src_h, int y_pitch_words, int uv_pitch_words,
    uint8_t* __restrict__ dst_bgr, int dst_w, int dst_h,
    int bit_shift)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float max_val = (float)((1 << (16 - bit_shift)) - 1);
    float inv_max = 1.0f / max_val;
    float scale_factor = 255.0f * inv_max;

    float sx = (float)dx * src_w / dst_w;
    float sy = (float)dy * src_h / dst_h;
    int x0 = __float2int_rd(sx), y0 = __float2int_rd(sy);
    int x1 = min(x0 + 1, src_w - 1), y1 = min(y0 + 1, src_h - 1);
    float fx = sx - x0, fy = sy - y0;

    float y_val = (1-fx)*(1-fy) * (float)(y_plane[y0 * y_pitch_words + x0] >> bit_shift)
                + fx*(1-fy)     * (float)(y_plane[y0 * y_pitch_words + x1] >> bit_shift)
                + (1-fx)*fy     * (float)(y_plane[y1 * y_pitch_words + x0] >> bit_shift)
                + fx*fy         * (float)(y_plane[y1 * y_pitch_words + x1] >> bit_shift);

    int uv_x = (x0 >> 1) * 2;
    int uv_y = y0 >> 1;
    float u_val = (float)(uv_plane[uv_y * uv_pitch_words + uv_x] >> bit_shift) - max_val * 0.5f;
    float v_val = (float)(uv_plane[uv_y * uv_pitch_words + uv_x + 1] >> bit_shift) - max_val * 0.5f;

    float r = (y_val + 1.402f * v_val) * scale_factor;
    float g = (y_val - 0.344f * u_val - 0.714f * v_val) * scale_factor;
    float b = (y_val + 1.772f * u_val) * scale_factor;

    int idx = (dy * dst_w + dx) * 3;
    dst_bgr[idx]     = (uint8_t)max(0.0f, min(255.0f, b));
    dst_bgr[idx + 1] = (uint8_t)max(0.0f, min(255.0f, g));
    dst_bgr[idx + 2] = (uint8_t)max(0.0f, min(255.0f, r));
}

void launch_nv12_16bit_resize_bgr(const uint8_t* y_plane, const uint8_t* uv_plane,
                                   int src_w, int src_h, int y_pitch, int uv_pitch,
                                   uint8_t* dst_bgr, int dst_w, int dst_h,
                                   int bit_shift, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    nv12_16bit_resize_bgr_kernel<<<grid, block, 0, stream>>>(
        (const uint16_t*)y_plane, (const uint16_t*)uv_plane,
        src_w, src_h, y_pitch / 2, uv_pitch / 2,
        dst_bgr, dst_w, dst_h, bit_shift);
}

// ==================== 10/12-bit NV12 → RGB + Resize ====================

__global__ void nv12_16bit_resize_rgb_kernel(
    const uint16_t* __restrict__ y_plane,
    const uint16_t* __restrict__ uv_plane,
    int src_w, int src_h, int y_pitch_words, int uv_pitch_words,
    uint8_t* __restrict__ dst_rgb, int dst_w, int dst_h,
    int bit_shift)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float max_val = (float)((1 << (16 - bit_shift)) - 1);
    float inv_max = 1.0f / max_val;
    float scale_factor = 255.0f * inv_max;

    float sx = (float)dx * src_w / dst_w;
    float sy = (float)dy * src_h / dst_h;
    int x0 = __float2int_rd(sx), y0 = __float2int_rd(sy);
    int x1 = min(x0 + 1, src_w - 1), y1 = min(y0 + 1, src_h - 1);
    float fx = sx - x0, fy = sy - y0;

    float y_val = (1-fx)*(1-fy) * (float)(y_plane[y0 * y_pitch_words + x0] >> bit_shift)
                + fx*(1-fy)     * (float)(y_plane[y0 * y_pitch_words + x1] >> bit_shift)
                + (1-fx)*fy     * (float)(y_plane[y1 * y_pitch_words + x0] >> bit_shift)
                + fx*fy         * (float)(y_plane[y1 * y_pitch_words + x1] >> bit_shift);

    int uv_x = (x0 >> 1) * 2;
    int uv_y = y0 >> 1;
    float u_val = (float)(uv_plane[uv_y * uv_pitch_words + uv_x] >> bit_shift) - max_val * 0.5f;
    float v_val = (float)(uv_plane[uv_y * uv_pitch_words + uv_x + 1] >> bit_shift) - max_val * 0.5f;

    float r = (y_val + 1.402f * v_val) * scale_factor;
    float g = (y_val - 0.344f * u_val - 0.714f * v_val) * scale_factor;
    float b = (y_val + 1.772f * u_val) * scale_factor;

    int idx = (dy * dst_w + dx) * 3;
    dst_rgb[idx]     = (uint8_t)max(0.0f, min(255.0f, r));
    dst_rgb[idx + 1] = (uint8_t)max(0.0f, min(255.0f, g));
    dst_rgb[idx + 2] = (uint8_t)max(0.0f, min(255.0f, b));
}

void launch_nv12_16bit_resize_rgb(const uint8_t* y_plane, const uint8_t* uv_plane,
                                   int src_w, int src_h, int y_pitch, int uv_pitch,
                                   uint8_t* dst_rgb, int dst_w, int dst_h,
                                   int bit_shift, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    nv12_16bit_resize_rgb_kernel<<<grid, block, 0, stream>>>(
        (const uint16_t*)y_plane, (const uint16_t*)uv_plane,
        src_w, src_h, y_pitch / 2, uv_pitch / 2,
        dst_rgb, dst_w, dst_h, bit_shift);
}
