#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// 流水线配置
namespace pipeline_config {
    constexpr int kRingBufferSize = 4;     // 4 级流水线
    constexpr int kInputH = 640;
    constexpr int kInputW = 640;
    constexpr int kMaxDetections = 1000;
    constexpr int kNumClasses = 80;
    // TensorRT 输出: 25200 × 85
    constexpr int kOutputElements = 25200 * 85;
}

enum class FramePixelFormat { UNKNOWN, NV12, YUYV, GRAY8, YUYV16, BGR };

// 预分配的 GPU 内存槽（每个流水线帧一个）
struct FrameSlot {
    // NV12 解码帧（指向 FFmpeg 的 GPU 帧，无需单独分配）
    uint8_t* nv12_y = nullptr;
    uint8_t* nv12_uv = nullptr;
    int y_pitch = 0;
    int uv_pitch = 0;
    int src_width = 0;
    int src_height = 0;

    // 位深: 0=8-bit, 6=10-bit (>>6), 4=12-bit (>>4)
    // 用于选择正确的 CUDA kernel (yuyv16/nv12_16bit 系列)
    int bit_shift = 0;
    FramePixelFormat pixel_format = FramePixelFormat::UNKNOWN;

    // TensorRT 输入缓冲区（NCHW float32, 640×640）
    float* input_buf = nullptr;
    void* tensor_input_buf = nullptr;  // input_buf or converted FP16 storage
    bool tensor_input_fp16 = false;

    // TensorRT 输出缓冲区
    float* output_buf = nullptr;
    void* tensor_output_buf = nullptr;  // output_buf or temporary FP16 output
    bool tensor_output_fp16 = false;
    float* host_output = nullptr;  // Pinned host memory（异步 D2H）

    // 同步事件
    cudaEvent_t decode_done = nullptr;
    cudaEvent_t preproc_done = nullptr;
    cudaEvent_t infer_done = nullptr;
    cudaEvent_t postproc_done = nullptr;

    // 帧序号
    int64_t frame_idx = -1;
};

class RingBuffer {
public:
    explicit RingBuffer(int num_slots = pipeline_config::kRingBufferSize);
    ~RingBuffer();

    // 分配所有 GPU 缓冲区（初始化时调用一次）
    bool allocate(int input_w = pipeline_config::kInputW,
                  int input_h = pipeline_config::kInputH,
                  int output_elements = pipeline_config::kOutputElements,
                  bool input_fp16 = false,
                  bool output_fp16 = false);

    // 释放所有缓冲区
    void release();

    FrameSlot& slot(int idx) { return slots_[idx]; }
    const FrameSlot& slot(int idx) const { return slots_[idx]; }
    int numSlots() const { return num_slots_; }

private:
    int num_slots_;
    FrameSlot* slots_ = nullptr;
    bool allocated_ = false;
};
