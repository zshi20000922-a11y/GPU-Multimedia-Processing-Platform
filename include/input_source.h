#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <functional>
#include <queue>
#include <condition_variable>

#ifdef __linux__
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

// 输入源类型
enum class InputType { NONE, CAMERA, VIDEO, UVC };

// 检测模式
enum class DetectMode { OFF, DIRECT, SAHI };

// 输入源配置
struct InputConfig {
    int id = 0;                    // 通道 ID (0/1/2)
    InputType type = InputType::NONE;
    std::string source;            // 摄像头ID 或 视频路径
    DetectMode detect_mode = DetectMode::OFF;
    float conf_thresh = 0.25f;
    float nms_thresh = 0.45f;
    int sahi_stride = 320;         // SAHI 步长
    int sahi_slice = 640;          // SAHI 切片大小
    int sahi_interval = 15;        // SAHI 推理间隔（每隔多少帧推理一次）

    // 摄像头参数
    int cam_width = 1920;
    int cam_height = 1080;
    float cam_fps = 30.0f;
    uint32_t cam_format = 0x56595559;  // V4L2_PIX_FMT_YUYV (default)
    bool cam_packed_gray = false;      // packed-gray：YUYV 管道搬运灰度，按 W×(2H) 重解释

    // 视频文件目标帧率（0=按原视频帧率，负数=不限制/基准测试）
    float target_fps = 0.0f;
};

// 输入源抽象基类（摄像头用，视频走 FFmpegHWDecoder）
class InputSource {
public:
    virtual ~InputSource() = default;
    virtual bool open() = 0;
    virtual bool read(cv::Mat& frame) = 0;
    virtual void release() = 0;
    virtual int width() const = 0;
    virtual int height() const = 0;
    virtual float fps() const = 0;
    virtual bool is_open() const = 0;
    virtual bool is_camera() const = 0;
};

#ifdef __linux__
// V4L2 摄像头 - 零拷贝 DMA 路径
class CameraSource : public InputSource {
public:
    CameraSource(int device_id) : device_id_(device_id) {}
    ~CameraSource() { release(); }

    bool open() override;
    bool read(cv::Mat& frame) override;
    void release() override;
    int width() const override { return width_; }
    int height() const override { return height_; }
    float fps() const override { return fps_; }
    bool is_open() const override { return fd_ >= 0; }
    bool is_camera() const override { return true; }

private:
    int device_id_;
    int fd_ = -1;
    int width_ = 0;
    int height_ = 0;
    float fps_ = 30.0f;

    struct Buffer {
        void* start = nullptr;
        size_t length = 0;
    };
    std::vector<Buffer> buffers_;
    cv::Mat temp_frame_;

    bool init_v4l2();
    bool start_streaming();
    void stop_streaming();
};
#else
// 非 Linux 平台使用 OpenCV 后备
class CameraSource : public InputSource {
public:
    CameraSource(int device_id) : device_id_(device_id) {}
    bool open() override;
    bool read(cv::Mat& frame) override;
    void release() override;
    int width() const override { return cap_.isOpened() ? (int)cap_.get(cv::CAP_PROP_FRAME_WIDTH) : 0; }
    int height() const override { return cap_.isOpened() ? (int)cap_.get(cv::CAP_PROP_FRAME_HEIGHT) : 0; }
    float fps() const override { return cap_.isOpened() ? (float)cap_.get(cv::CAP_PROP_FPS) : 0; }
    bool is_open() const override { return cap_.isOpened(); }
    bool is_camera() const override { return true; }
private:
    int device_id_;
    cv::VideoCapture cap_;
};
#endif

// 创建输入源（仅摄像头，视频走 FFmpegHWDecoder）
inline std::unique_ptr<InputSource> create_input(const InputConfig& cfg) {
    if (cfg.type == InputType::CAMERA) {
        return std::make_unique<CameraSource>(std::stoi(cfg.source));
    }
    return nullptr;
}
