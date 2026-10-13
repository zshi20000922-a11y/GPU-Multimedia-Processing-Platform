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
#include <libavutil/imgutils.h>
}

// FFmpeg 解封装 + NVDEC 硬件解码器
class FFmpegDecoder {
public:
    FFmpegDecoder();
    ~FFmpegDecoder();

    // 打开视频文件
    bool open(const std::string& path);

    // 读取一帧
    bool read(cv::Mat& frame);

    // 释放资源
    void release();

    // 获取视频信息
    int width() const { return width_; }
    int height() const { return height_; }
    float fps() const { return fps_; }
    bool is_open() const { return is_open_; }

private:
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
    AVCodecContext* codec_ctx_ = nullptr;
    int video_stream_idx_ = -1;
    AVFrame* frame_ = nullptr;
    AVPacket* packet_ = nullptr;

    // 帧缓冲区
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
