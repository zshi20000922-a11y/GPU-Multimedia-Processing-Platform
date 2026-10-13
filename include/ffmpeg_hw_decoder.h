#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
}

#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <chrono>

// GPU 上的 NV12 帧（零拷贝，直接指向 FFmpeg AVFrame）
struct GpuFrame {
    uint8_t* y_plane = nullptr;
    uint8_t* uv_plane = nullptr;
    int y_pitch = 0;
    int uv_pitch = 0;
    int width = 0;
    int height = 0;
    int64_t pts = 0;
};

// FFmpeg 帧包装（持有 AVFrame 引用，消费后释放）
struct DecodedFrame {
    GpuFrame gpu;
    AVFrame* av_frame = nullptr;
};

// FFmpeg + CUDA 硬件加速视频解码器
// 零拷贝：解码后帧直接在 GPU 显存，frame->data[0] 就是 GPU 指针
class FFmpegHWDecoder {
public:
    FFmpegHWDecoder();
    ~FFmpegHWDecoder();

    bool open(const std::string& filepath);

    // 获取下一帧（阻塞直到可用），帧数据在 GPU 显存
    // 消费者必须调用 releaseFrame() 释放
    bool getNextFrame(GpuFrame& frame);

    // 释放当前帧（必须在下次 getNextFrame 前调用）
    void releaseFrame();

    void stop();

    int width() const { return width_; }
    int height() const { return height_; }
    double fps() const { return fps_; }
    bool isOpen() const { return opened_; }
    double durationSeconds() const { return duration_sec_; }
    double positionSeconds() const { return position_sec_.load(std::memory_order_relaxed); }
    void seekSeconds(double seconds);

    // 设置解码目标帧率（线程安全，运行时可调）
    // >0: 锁定到指定帧率
    // 0: 按视频原始帧率解码（默认，避免多路争抢 NVDEC）
    // -1: 全速解码（不限制）
    void setTargetFps(double target_fps) { target_fps_.store(target_fps, std::memory_order_relaxed); }

private:
    bool initHwContext();
    bool openCodec();
    void decodeLoop();

    // FFmpeg 资源
    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    AVBufferRef* hw_device_ctx_ = nullptr;
    AVPacket* packet_ = nullptr;
    int video_stream_idx_ = -1;

    int width_ = 0;
    int height_ = 0;
    double fps_ = 0.0;
    std::atomic<double> target_fps_{0.0};  // 0=不限制（线程安全）
    bool opened_ = false;
    double duration_sec_ = 0.0;
    std::atomic<double> position_sec_{0.0};
    std::atomic<double> seek_request_sec_{-1.0};

    // 解码线程
    std::thread decode_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> eof_{false};

    // 帧队列（预解码帧缓冲）
    static constexpr int MAX_QUEUE = 4;
    std::queue<DecodedFrame> frame_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_not_empty_;
    std::condition_variable queue_not_full_;

    // 当前正在消费的帧
    AVFrame* current_frame_ = nullptr;
};
