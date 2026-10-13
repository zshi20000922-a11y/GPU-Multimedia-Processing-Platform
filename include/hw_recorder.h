#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <opencv2/opencv.hpp>
#include <string>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>

// GPU 帧数据（NV12 或 BGR）
struct RecordFrame {
    // NV12 数据（视频文件路径）
    const uint8_t* y_plane = nullptr;
    const uint8_t* uv_plane = nullptr;
    int y_pitch = 0;
    int uv_pitch = 0;
    int width = 0;
    int height = 0;
    bool is_nv12 = true;  // true=NV12, false=BGR

    // BGR 数据（摄像头路径或带标注帧）
    const uint8_t* bgr_data = nullptr;
    int bgr_stride = 0;

    // 灰度直通：单通道 8-bit 灰度，直接填 YUV420P 的 Y 平面（UV=128）
    // 跳过 GRAY2BGR 和 sws_scale，用于 packed-gray / grey 相机
    bool is_gray = false;
    const uint8_t* gray_data = nullptr;
    int gray_stride = 0;

    // 持有 gray/BGR 数据所有权（防止 pipeline 线程覆盖）
    cv::Mat owned_frame;
};

// 硬件加速录制器（FFmpeg libavcodec + libx264）
// 支持从 GPU NV12 帧或 CPU BGR 帧直接编码
class HWRecorder {
public:
    HWRecorder() = default;
    ~HWRecorder() { stop(); }

    // 开始录制
    // width/height: 原始帧尺寸（3840x2160）
    // fps: 录制帧率（可低于处理帧率）
    bool start(const std::string& path, int width, int height, float fps = 30.0f);

    // 停止录制
    void stop();

    // 推入一帧（非阻塞，满则丢弃）
    void pushFrame(const RecordFrame& frame);

    // 推入 BGR cv::Mat（便捷接口）
    void pushBGRFrame(const cv::Mat& bgr);

    // 状态
    bool isRecording() const { return recording_; }
    const std::string& filePath() const { return filepath_; }
    float recordedSeconds() const;
    int framesWritten() const { return frames_written_; }

    // 设置时长限制（秒，0=无限制）
    void setDurationLimit(float seconds) { duration_limit_ = seconds; }

private:
    void encodeLoop();

    // FFmpeg 资源
    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    AVStream* stream_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;  // NV12/BGR → YUV420p 转换
    AVFrame* frame_ = nullptr;
    AVPacket* packet_ = nullptr;

    int width_ = 0, height_ = 0;
    float fps_ = 30.0f;
    float duration_limit_ = 0;
    bool use_nvenc_ = false;  // 是否使用 NVENC 硬件编码

    // 帧队列
    std::thread encode_thread_;
    std::queue<RecordFrame> frame_queue_;
    static constexpr int MAX_QUEUE = 120;  // 缓存 120 帧
    std::mutex queue_mtx_;
    std::condition_variable queue_cv_;
    std::atomic<bool> recording_{false};
    std::atomic<int> frames_written_{0};
    std::string filepath_;
    std::chrono::steady_clock::time_point start_time_;
};
