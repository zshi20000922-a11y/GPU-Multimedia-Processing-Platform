#include "ring_buffer.h"
#include "logger.h"

#include <cuda_runtime.h>
#include <cstring>

RingBuffer::RingBuffer(int num_slots) : num_slots_(num_slots) {
    slots_ = new FrameSlot[num_slots_];
}

RingBuffer::~RingBuffer() {
    release();
    delete[] slots_;
}

bool RingBuffer::allocate(int input_w, int input_h, int output_elements,
                          bool input_fp16, bool output_fp16) {
    if (allocated_) return true;

    size_t input_bytes = (size_t)input_w * input_h * 3 * sizeof(float);
    size_t output_bytes = (size_t)output_elements * sizeof(float);

    for (int i = 0; i < num_slots_; i++) {
        FrameSlot& s = slots_[i];

        // TensorRT 输入缓冲区
        if (cudaMalloc(&s.input_buf, input_bytes) != cudaSuccess) {
            LOG_ERROR("RingBuffer: failed to allocate input_buf for slot %d", i);
            release();
            return false;
        }
        s.tensor_input_fp16 = input_fp16;
        if (input_fp16) {
            if (cudaMalloc(&s.tensor_input_buf, input_bytes / 2) != cudaSuccess) {
                LOG_ERROR("RingBuffer: failed to allocate FP16 input for slot %d", i);
                release();
                return false;
            }
        } else {
            s.tensor_input_buf = s.input_buf;
        }

        // TensorRT 输出缓冲区
        if (cudaMalloc(&s.output_buf, output_bytes) != cudaSuccess) {
            LOG_ERROR("RingBuffer: failed to allocate output_buf for slot %d", i);
            release();
            return false;
        }
        s.tensor_output_fp16 = output_fp16;
        if (output_fp16) {
            if (cudaMalloc(&s.tensor_output_buf, output_bytes / 2) != cudaSuccess) {
                LOG_ERROR("RingBuffer: failed to allocate FP16 output for slot %d", i);
                release();
                return false;
            }
        } else {
            s.tensor_output_buf = s.output_buf;
        }

        // Pinned host memory（异步 D2H）
        if (cudaMallocHost(&s.host_output, output_bytes) != cudaSuccess) {
            LOG_ERROR("RingBuffer: failed to allocate host_output for slot %d", i);
            release();
            return false;
        }

        // 创建同步事件
        cudaEventCreate(&s.decode_done);
        cudaEventCreate(&s.preproc_done);
        cudaEventCreate(&s.infer_done);
        cudaEventCreate(&s.postproc_done);

        // 记录初始 postproc_done 事件（让第一帧的 waitEvent 立即通过）
        cudaEventRecord(s.postproc_done);
    }

    allocated_ = true;
    LOG_INFO("RingBuffer allocated: %d slots, %.1f MB GPU + %.1f MB host",
             num_slots_,
             (input_bytes + output_bytes) * num_slots_ / 1024.0 / 1024.0,
             output_bytes * num_slots_ / 1024.0 / 1024.0);
    return true;
}

void RingBuffer::release() {
    if (!allocated_) return;

    for (int i = 0; i < num_slots_; i++) {
        FrameSlot& s = slots_[i];
        if (s.tensor_input_fp16 && s.tensor_input_buf) cudaFree(s.tensor_input_buf);
        s.tensor_input_buf = nullptr;
        s.tensor_input_fp16 = false;
        if (s.input_buf) { cudaFree(s.input_buf); s.input_buf = nullptr; }
        if (s.tensor_output_fp16 && s.tensor_output_buf) cudaFree(s.tensor_output_buf);
        s.tensor_output_buf = nullptr;
        s.tensor_output_fp16 = false;
        if (s.output_buf) { cudaFree(s.output_buf); s.output_buf = nullptr; }
        if (s.host_output) { cudaFreeHost(s.host_output); s.host_output = nullptr; }
        if (s.decode_done) { cudaEventDestroy(s.decode_done); s.decode_done = nullptr; }
        if (s.preproc_done) { cudaEventDestroy(s.preproc_done); s.preproc_done = nullptr; }
        if (s.infer_done) { cudaEventDestroy(s.infer_done); s.infer_done = nullptr; }
        if (s.postproc_done) { cudaEventDestroy(s.postproc_done); s.postproc_done = nullptr; }
    }

    allocated_ = false;
}
