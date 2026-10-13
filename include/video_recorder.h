#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>

// 录制时长选项（秒）
enum class RecordDuration {
    SEC_20 = 20,
    SEC_60 = 60,
    SEC_120 = 120,
};

// 视频录制器（独立编码线程，cv::VideoWriter）
class VideoRecorder {
public:
    VideoRecorder() = default;
    ~VideoRecorder() { stop(); }

    // 开始录制
    bool start(const std::string& path, int width, int height, float fps);

    // 停止录制
    void stop();

    // 推入一帧（非阻塞，满则丢弃）
    void pushFrame(const cv::Mat& bgr_frame);

    // 状态
    bool isRecording() const { return recording_; }
    const std::string& filePath() const { return filepath_; }
    float recordedSeconds() const;
    int framesWritten() const { return frames_written_; }

    // 设置时长限制（0=无限制）
    void setDurationLimit(float seconds) { duration_limit_ = seconds; }

private:
    void encodeLoop();

    cv::VideoWriter writer_;
    int width_ = 0, height_ = 0;
    float fps_ = 30.0f;
    float duration_limit_ = 0;  // 秒，0=无限制

    std::thread encode_thread_;
    std::queue<cv::Mat> frame_queue_;
    static constexpr int MAX_QUEUE = 60;  // 最多缓存 60 帧
    std::mutex queue_mtx_;
    std::condition_variable queue_cv_;
    std::atomic<bool> recording_{false};
    std::atomic<int> frames_written_{0};
    std::string filepath_;
    std::chrono::steady_clock::time_point start_time_;
};
