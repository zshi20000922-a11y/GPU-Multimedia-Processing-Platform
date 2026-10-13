/**
 * CUDA Post-processing Kernels
 * GPU-based NMS and coordinate mapping
 */

#include "cuda_kernels.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>

// 常量
#define MAX_DETECTIONS 1000
#define BLOCK_SIZE 256

// 初始化检测输出缓冲区
__global__ void init_detections_kernel(CUDADetection* detections, int max_detections) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < max_detections) {
        detections[idx].valid = 0;
        detections[idx].conf = 0.0f;
    }
}

void launch_init_detections(CUDADetection* detections, int max_detections,
                            cudaStream_t stream) {
    int blocks = (max_detections + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_detections_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(detections, max_detections);
}

// 第一阶段：置信度过滤 + 坐标转换
__global__ void filter_and_transform_kernel(
    const float* __restrict__ output, int num_preds,
    float conf_thresh, int input_size,
    int orig_w, int orig_h,
    CUDADetection* detections, int* valid_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_preds) return;

    const float* p = output + idx * 85;

    // 置信度过滤
    float obj = p[4];
    if (obj < conf_thresh) {
        return;
    }

    // 找到最佳类别
    float best = 0;
    int bc = 0;
    for (int c = 0; c < 80; c++) {
        if (p[5 + c] > best) {
            best = p[5 + c];
            bc = c;
        }
    }

    float fs = obj * best;
    if (fs < conf_thresh) {
        return;
    }

    // 坐标转换：从模型输出坐标到原图坐标
    float cx = p[0], cy = p[1], w = p[2], h = p[3];

    // 计算 letterbox 参数
    float scale = min((float)input_size / orig_w, (float)input_size / orig_h);
    int nw = (int)(orig_w * scale);
    int nh = (int)(orig_h * scale);
    float px = (input_size - nw) / 2.0f;
    float py = (input_size - nh) / 2.0f;

    // 转换到原图坐标
    float x1 = (cx - w * 0.5f - px) / scale;
    float y1 = (cy - h * 0.5f - py) / scale;
    float x2 = (cx + w * 0.5f - px) / scale;
    float y2 = (cy + h * 0.5f - py) / scale;

    // 裁剪到原图范围
    x1 = fmaxf(0.0f, fminf((float)orig_w, x1));
    y1 = fmaxf(0.0f, fminf((float)orig_h, y1));
    x2 = fmaxf(0.0f, fminf((float)orig_w, x2));
    y2 = fmaxf(0.0f, fminf((float)orig_h, y2));

    // 原子操作获取输出索引
    int out_idx = atomicAdd(valid_count, 1);
    if (out_idx < MAX_DETECTIONS) {
        detections[out_idx].x1 = x1;
        detections[out_idx].y1 = y1;
        detections[out_idx].x2 = x2;
        detections[out_idx].y2 = y2;
        detections[out_idx].conf = fs;
        detections[out_idx].class_id = bc;
        detections[out_idx].valid = 1;
    }
}

// NMS kernel（简化版本，按类别分组）
__global__ void nms_kernel(
    CUDADetection* detections, int num_detections,
    float nms_thresh)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_detections) return;
    if (!detections[idx].valid) return;

    for (int j = idx + 1; j < num_detections; j++) {
        if (!detections[j].valid) continue;
        if (detections[idx].class_id != detections[j].class_id) continue;

        // 计算 IoU
        float ix1 = fmaxf(detections[idx].x1, detections[j].x1);
        float iy1 = fmaxf(detections[idx].y1, detections[j].y1);
        float ix2 = fminf(detections[idx].x2, detections[j].x2);
        float iy2 = fminf(detections[idx].y2, detections[j].y2);

        float inter = fmaxf(0.0f, ix2 - ix1) * fmaxf(0.0f, iy2 - iy1);
        float area_i = (detections[idx].x2 - detections[idx].x1) *
                       (detections[idx].y2 - detections[idx].y1);
        float area_j = (detections[j].x2 - detections[j].x1) *
                       (detections[j].y2 - detections[j].y1);

        float iou = inter / (area_i + area_j - inter + 1e-6f);

        // 如果 IoU 超过阈值，抑制置信度较低的检测
        if (iou > nms_thresh) {
            if (detections[idx].conf < detections[j].conf) {
                detections[idx].valid = 0;
                return;
            } else {
                detections[j].valid = 0;
            }
        }
    }
}

__global__ void float_to_half_kernel(const float* src, __half* dst, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

void launch_float_to_half(const float* src, void* dst, size_t elements, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (int)((elements + threads - 1) / threads);
    float_to_half_kernel<<<blocks, threads, 0, stream>>>(src, static_cast<__half*>(dst), elements);
}

__global__ void half_to_float_kernel(const __half* src, float* dst, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

void launch_half_to_float(const void* src, float* dst, size_t elements, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (int)((elements + threads - 1) / threads);
    half_to_float_kernel<<<blocks, threads, 0, stream>>>(static_cast<const __half*>(src), dst, elements);
}

void launch_nms(const float* output, int num_preds,
                float conf_thresh, float nms_thresh,
                CUDADetection* detections, int* num_detections,
                int max_detections, int orig_w, int orig_h,
                int input_size, cudaStream_t stream)
{
    // 初始化输出
    launch_init_detections(detections, max_detections, stream);
    cudaMemsetAsync(num_detections, 0, sizeof(int), stream);

    // 第一阶段：置信度过滤 + 坐标转换
    int blocks = (num_preds + BLOCK_SIZE - 1) / BLOCK_SIZE;
    filter_and_transform_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        output, num_preds, conf_thresh, input_size,
        orig_w, orig_h, detections, num_detections);

    // 第二阶段：NMS（使用少量线程，因为有效检测数量通常不多）
    int nms_blocks = (max_detections + BLOCK_SIZE - 1) / BLOCK_SIZE;
    nms_kernel<<<nms_blocks, BLOCK_SIZE, 0, stream>>>(
        detections, max_detections, nms_thresh);
}
