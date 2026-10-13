#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvcuvid.h>
#include <cuviddec.h>

// GPU 上的帧数据（NV12 格式，零拷贝路径用）
struct GpuFrame {
    uint8_t* data = nullptr;  // GPU 上的 NV12 数据指针
    int width = 0;
    int height = 0;
    int pitch = 0;            // NV12 行字节数（可能有对齐 padding）
    int buf_index = -1;       // 缓冲区池索引，用于归还
};

// NVDEC GPU 硬件解码器
class NVDECDecoder {
public:
    NVDECDecoder();
    ~NVDECDecoder();

    // 打开视频文件
    bool open(const std::string& path);

    // 读取一帧到 CPU（兼容旧路径）
    bool read(cv::Mat& frame);

    // 读取一帧 GPU 指针（零拷贝路径，用于直接 GPU 前处理）
    bool read_gpu(GpuFrame& frame);

    // 归还 GPU 缓冲区到池中
    void release_gpu_frame(const GpuFrame& frame);

    // 释放资源
    void release();

    // 获取视频信息
    int width() const { return width_; }
    int height() const { return height_; }
    float fps() const { return fps_; }
    bool is_open() const { return is_open_; }

private:
    // FFmpeg 解封装
    bool init_ffmpeg(const std::string& path);
    void close_ffmpeg();

    // NVDEC 初始化
    bool init_nvdec();
    bool create_parser();
    bool create_decoder(const CUVIDEOFORMAT* pVideoFormat);

    // GPU 缓冲区池
    bool alloc_gpu_buffers();
    void free_gpu_buffers();

    // NVDEC 回调函数
    static int CUDAAPI HandleVideoSequence(void* pUserData, CUVIDEOFORMAT* pVideoFormat);
    static int CUDAAPI HandlePictureDecode(void* pUserData, CUVIDPICPARAMS* pPicParams);
    static int CUDAAPI HandlePictureDisplay(void* pUserData, CUVIDPARSERDISPINFO* pDispInfo);

    // 解码线程
    void decode_thread_func();

    // 成员变量
    std::string path_;
    int width_ = 0;
    int height_ = 0;
    float fps_ = 0;
    bool is_open_ = false;

    // FFmpeg 资源
    AVFormatContext* format_ctx_ = nullptr;
    int video_stream_idx_ = -1;

    // NVDEC 资源
    CUcontext cu_context_ = nullptr;
    CUvideodecoder decoder_ = nullptr;
    CUvideoparser parser_ = nullptr;

    // GPU NV12 缓冲区池（预分配，避免回调中动态分配）
    static const int GPU_BUF_COUNT = 5;
    uint8_t* gpu_nv12_bufs_[GPU_BUF_COUNT] = {};
    size_t gpu_buf_size_ = 0;
    int gpu_pitch_ = 0;         // 实际 pitch（可能与 width 不同）

    // GPU 帧队列（零拷贝路径）
    struct GpuFrameQueueItem {
        uint8_t* data;          // 指向 gpu_nv12_bufs_[buf_index]
        int width;
        int height;
        int pitch;
        int buf_index;
    };
    std::queue<GpuFrameQueueItem> gpu_frame_queue_;
    std::mutex gpu_queue_mutex_;
    std::condition_variable gpu_queue_cv_;

    // CPU 帧队列（兼容旧路径）
    struct DecodedFrame {
        cv::Mat frame;
        bool ready = false;
    };

    std::queue<DecodedFrame> frame_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    static const int MAX_QUEUE_SIZE = 5;

    // 解码线程
    std::thread decode_thread_;
    std::atomic<bool> stop_decode_{false};

    // 帧计数
    int frame_count_ = 0;
};
