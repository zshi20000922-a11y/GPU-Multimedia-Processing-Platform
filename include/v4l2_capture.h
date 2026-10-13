#pragma once

#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstdint>

#include <cuda_runtime.h>
#include <cuda.h>

// 摄像头支持的格式
struct CameraFormat {
    uint32_t pixelformat;    // V4L2_PIX_FMT_*
    std::string name;        // "YUYV", "MJPEG", etc.
    int width;
    int height;
    std::vector<float> fps_options;  // 支持的帧率列表
};

// 摄像头能力
struct CameraCapabilities {
    std::string device_path;       // "/dev/video0"
    std::string card_name;         // 摄像头名称
    std::vector<CameraFormat> formats;

    bool empty() const { return formats.empty(); }
};

// V4L2 采集帧
struct V4L2Frame {
    uint8_t* data = nullptr;   // mmap 映射的 DMA 数据指针
    int width = 0;
    int height = 0;
    size_t size = 0;           // 数据大小
    int buf_index = -1;        // V4L2 缓冲区索引
    uint64_t timestamp_us = 0; // 时间戳（微秒）
    int bytesperline = 0;      // 行字节跨度（stride）
    int bit_shift = 0;         // 位深: 0=8-bit, 6=10-bit, 4=12-bit
};

// 零拷贝模式
enum class ZeroCopyMode {
    NONE,        // 传统 memcpy 方式
    CUDA_PINNED, // cudaHostRegister: 消除 memcpy，直接从 pinned mmap 异步传输
    DMA_BUF,     // DMA-BUF + CUDA External Memory: 真零拷贝
};

// V4L2 独立采集器
// 独立线程采集，帧队列缓冲，pipeline 非阻塞消费
class V4L2Capture {
public:
    V4L2Capture() = default;
    ~V4L2Capture();

    // 查询摄像头支持的格式、分辨率、帧率
    static CameraCapabilities queryCapabilities(int device_id);

    // 枚举系统中所有可用的摄像头设备（扫描 /dev/video0~31）
    struct CameraInfo {
        int device_id;
        std::string device_path;   // "/dev/video0"
        std::string card_name;     // "USB Camera" 等
    };
    static std::vector<CameraInfo> enumerateCameras();

    // USB 总线级硬复位（USBDEVFS_RESET，等效物理拔插）
    // 由 /dev/videoN 经 sysfs 找到 USB 设备节点并复位
    static bool resetUsbDevice(int device_id);

    bool open(int device_id, int width = 3840, int height = 2160);

    // 使用指定格式打开
    bool open(int device_id, int width, int height, uint32_t pixelformat, float fps);

    void close();

    // 获取下一帧（阻塞直到可用或超时）
    // 消费者必须调用 releaseFrame() 归还
    bool getNextFrame(V4L2Frame& frame, int timeout_ms = 100);

    // 归还帧（释放 V4L2 缓冲区）
    void releaseFrame(int buf_index);

    void stop();

    int width() const { return width_; }
    int height() const { return height_; }
    float fps() const { return fps_; }
    // 实测采集帧率（采集线程统计，每秒更新）
    float measuredFps() const { return measured_fps_.load(); }
    bool isOpen() const { return opened_; }

    // GREY 格式检测：V4L2_PIX_FMT_GREY，8-bit 单通道灰度，每像素 1 字节，stride=width
    // packed-gray 模式同样视为灰度（Y-only 路径）
    bool isGrey() const { return is_grey_ || packed_gray_; }

    // packed-gray：相机以 YUYV 管道搬运纯灰度数据（W×H×2 ≡ W×(2H)×1）。
    // 启用后采集帧按 W×(2H) 单通道灰度重解释。须在 open() 后、采集前调用。
    void setPackedGray(bool enable) { packed_gray_ = enable; }
    bool isPackedGray() const { return packed_gray_; }

    // 零拷贝查询
    ZeroCopyMode zeroCopyMode() const { return zero_copy_mode_; }
    bool isDMABUF() const { return zero_copy_mode_ == ZeroCopyMode::DMA_BUF; }
    bool isCudaPinned() const { return zero_copy_mode_ == ZeroCopyMode::CUDA_PINNED; }

    // 获取指定缓冲区的 GPU 指针（仅 DMA-BUF 模式可用）
    uint8_t* getGpuPtr(int buf_index) const;

private:
    bool initV4L2();
    bool startStreaming();
    void stopStreaming();
    void captureLoop();  // 独立采集线程

    // 零拷贝初始化
    bool initDMABUF();       // 方案A: DMA-BUF → CUDA external memory
    bool initCudaPinned();   // 方案B: cudaHostRegister pinned mmap
    void cleanupDMABUF();
    void cleanupCudaPinned();

    int fd_ = -1;
    int width_ = 0;
    int height_ = 0;
    float fps_ = 30.0f;
    bool opened_ = false;
    bool is_yuyv_ = true;  // true=YUYV, false=MJPEG
    bool is_grey_ = false;  // GREY 格式（V4L2_PIX_FMT_GREY，8-bit 单通道灰度）
    bool packed_gray_ = false;  // packed-gray：YUYV 管道搬运灰度，按 Y-only 重解释
    std::atomic<float> measured_fps_{0.0f};  // 实测采集帧率
    uint32_t actual_pixelformat_ = 0;  // 驱动实际返回的格式
    int bytesperline_ = 0;             // 行字节跨度
    int bit_shift_ = 0;                // 位深: 0=8-bit, 6=10-bit, 4=12-bit

    // 零拷贝模式
    ZeroCopyMode zero_copy_mode_ = ZeroCopyMode::NONE;

    // V4L2 DMA 缓冲区
    struct MmapBuffer {
        void* start = nullptr;
        size_t length = 0;
    };
    std::vector<MmapBuffer> buffers_;

    // DMA-BUF 零拷贝（方案A）
    struct DMABUFMapping {
        int fd = -1;                           // DMA-BUF 文件描述符
        cudaExternalMemory_t ext_mem = nullptr; // CUDA runtime 外部内存句柄
        CUexternalMemory cu_ext_mem = nullptr;  // CUDA driver 外部内存句柄
        uint8_t* dev_ptr = nullptr;             // CUDA 映射的设备指针
        size_t size = 0;
    };
    std::vector<DMABUFMapping> dma_mappings_;

    // 采集线程
    std::thread capture_thread_;
    std::atomic<bool> running_{false};

    // 帧队列（采集线程 → pipeline）
    static constexpr int MAX_QUEUE = 4;
    std::queue<V4L2Frame> frame_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_not_empty_;
    std::condition_variable queue_not_full_;

    // 已释放的缓冲区索引（可重新入队）
    std::queue<int> free_buffers_;
    std::mutex free_mutex_;
};
