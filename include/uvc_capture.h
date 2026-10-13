#pragma once

// libuvc 摄像头采集接口
// 与 V4L2Capture 平行，通过 InputType::UVC 选择
// 未安装 libuvc 时整个类不可用（HAS_LIBUVC=0）

#if HAS_LIBUVC

#include <libuvc/libuvc.h>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>
#include <atomic>
#include <cstdint>

// UVC 采集帧
struct UvcFrame {
    uint8_t* data = nullptr;    // 帧数据（YUYV 或 BGR）
    int width = 0;
    int height = 0;
    size_t size = 0;            // data 字节数
    bool is_bgr = false;        // true=BGR, false=YUYV
};

// UVC 摄像头采集器
class UvcCapture {
public:
    // 摄像头信息（复用 V4L2Capture::CameraInfo 结构）
    struct CameraInfo {
        int device_id;
        std::string device_path;    // USB bus:dev 路径
        std::string card_name;      // 设备名称
        uint16_t vendor_id;
        uint16_t product_id;
    };

    // 支持的格式能力
    struct FormatCap {
        int width;
        int height;
        float fps;                  // 最大帧率
        enum uvc_frame_format format;  // UVC 格式枚举
        std::string format_name;    // 可读格式名（YUYV/MJPEG/...）
    };

    UvcCapture() = default;
    ~UvcCapture() { stop(); }

    // 枚举 UVC 摄像头
    static std::vector<CameraInfo> enumerateCameras();

    // 查询设备支持的格式列表（需要先打开设备）
    std::vector<FormatCap> queryCapabilities(int device_id);

    // 打开设备（自动检测最佳格式）
    // width/height/fps: 0 表示自动选择最大值
    // 返回后 width_/height_/fps_ 反映实际协商的值
    bool open(int device_id, int width, int height, uint32_t format, float fps);

    // 获取下一帧（阻塞，超时返回 false）
    bool getNextFrame(UvcFrame& frame, int timeout_ms = 100);

    // 释放当前帧（必须在 getNextFrame 后调用）
    void releaseFrame();

    // 停止采集
    void stop();

    bool isOpen() const { return opened_; }
    int width() const { return width_; }
    int height() const { return height_; }
    float fps() const { return fps_; }

private:
    // libuvc 回调（在 libuvc 线程上运行）
    static void streamCallback(uvc_frame_t* frame, void* user_ptr);

    // 初始化 libuvc 上下文
    static bool ensureContext();

    // libuvc 全局上下文（所有 UvcCapture 实例共享）
    static uvc_context_t* ctx_;
    static std::mutex ctx_mtx_;
    static int ctx_refcount_;

    uvc_device_handle_t* dev_handle_ = nullptr;
    uvc_stream_handle_t* stream_handle_ = nullptr;
    uvc_stream_ctrl_t ctrl_;

    int width_ = 0;
    int height_ = 0;
    float fps_ = 0;
    bool opened_ = false;

    // 帧队列（回调线程 → 消费线程）
    static constexpr int MAX_QUEUE = 4;
    std::queue<UvcFrame> frame_queue_;
    std::mutex queue_mtx_;
    std::condition_variable queue_cv_;

    // 当前帧（getNextFrame 返回后持有，releaseFrame 释放）
    UvcFrame current_frame_;
    bool has_current_ = false;
};

#endif // HAS_LIBUVC
