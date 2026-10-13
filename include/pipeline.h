#pragma once

#include "input_source.h"
#include "detection_engine.h"
#include "ring_buffer.h"
#include "ffmpeg_hw_decoder.h"
#include "v4l2_capture.h"
#include "uvc_capture.h"
#include "cuda_kernels.h"
#include "sahi_slicer.h"
#include "video_recorder.h"
#include "hw_recorder.h"
#include "logger.h"

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <atomic>
#include <thread>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>
#include <vector>

// 单路处理结果（GUI 消费）
struct PipelineResult {
    cv::Mat frame;                    // 显示帧（BGR，缩小后）
    cv::Mat display;                  // 带标注的显示帧
    std::vector<Detection> dets;      // 检测结果
    float fps = 0;
    float cam_fps = 0;                // 摄像头实测采集帧率（仅摄像头输入有效）
    int num_objects = 0;
    bool has_frame = false;
    bool sahi_cached = false;         // SAHI 使用缓存检测结果（非实时推理帧）

    // GPU 直接显示路径（零拷贝）
    uint8_t* display_gpu = nullptr;   // GPU 上的 RGB 帧数据（CUDA-GL 互操作用）
    int display_w = 0;                // GPU 显示帧宽度
    int display_h = 0;                // GPU 显示帧高度
    int src_w = 0;                    // 原始帧宽度（录制用）
    int src_h = 0;                    // 原始帧高度（录制用）

    // 分时统计（毫秒，每秒平均）
    float ms_decode = 0;
    float ms_preproc = 0;
    float ms_infer = 0;
    float ms_postproc = 0;
    float ms_display = 0;   // 显示帧转换（NV12→BGR + resize）
    float ms_total = 0;
};

// SAHI 批量推理最大 batch 大小
constexpr int kSAHIMaxBatch = 24;

// 4 级流水线 Pipeline
// 解码 → 预处理 → 推理 → 后处理，各阶段在独立 CUDA stream 上并行
class Pipeline {
public:
    Pipeline() = default;
    ~Pipeline() { stop(); }

    // 初始化（使用独立引擎）
    void init_with_engine(int id, const InputConfig& cfg, const std::string& engine_path);

    // 启动/停止
    void start();
    void stop();
    bool is_running() const { return running_; }
    bool is_video() const { return config_.type == InputType::VIDEO; }
    double video_duration() const { return decoder_.durationSeconds(); }
    double video_position() const { return decoder_.positionSeconds(); }
    void seek_video(double seconds) { if (is_video()) decoder_.seekSeconds(seconds); }

    // 更新配置（运行时可调）
    void update_config(const InputConfig& cfg);

    // 获取最新结果
    PipelineResult get_result();

    // 获取当前配置
    InputConfig& config() { return config_; }
    int id() const { return id_; }

    // 引擎信息
    bool has_own_engine() const { return engine_ != nullptr; }

    // 视频录制
    VideoRecorder* recorder() { return &recorder_; }

    // 硬件录制（4K 原始帧）
    HWRecorder* hw_recorder() { return &hw_recorder_; }

    // 共享纹理（零拷贝显示）
    void setSharedDisplayBuffer(uint8_t* cuda_ptr, int width, int height) {
        shared_display_buf_ = cuda_ptr;
        shared_display_w_ = width;
        shared_display_h_ = height;
    }

    // SAHI 推理间隔（每隔多少帧推理一次，1=每帧推理）
    void setSahiInterval(int interval) { sahi_interval_ = std::max(1, interval); }
    int sahiInterval() const { return sahi_interval_; }

    // 快照：返回当前原始帧（YUYV CV_8UC2 或 gray CV_8UC1）
    cv::Mat snapshotRaw();
    bool is_grey_cam() const { return v4l2_.isOpen() && v4l2_.isGrey(); }

    // 显示缩放比例
    static constexpr int kDisplayMaxW = 960;
    static constexpr int kDisplayMaxH = 540;

private:
    void worker_thread();

    // CPU 端后处理：置信度过滤 + NMS
    // tile_offset_x/y: SAHI 模式下 tile 在原图中的偏移，DIRECT 模式下为 0
    std::vector<Detection> cpu_postprocess(const float* output, int num_preds,
                                           float conf_thresh, float nms_thresh,
                                           int orig_w, int orig_h, int input_size,
                                           int tile_offset_x = 0, int tile_offset_y = 0);

    int id_ = 0;
    InputConfig config_;
    std::mutex config_mtx_;  // 保护 config_ 的读写

    // TensorRT（每 Pipeline 独立）
    std::unique_ptr<DetectionEngine> engine_;           // DIRECT 模式引擎（batch=1）
    nvinfer1::IExecutionContext* trt_context_ = nullptr; // DIRECT 模式 context
    std::unique_ptr<DetectionEngine> sahi_engine_;       // SAHI 模式引擎（batch=8）
    nvinfer1::IExecutionContext* sahi_trt_context_ = nullptr; // SAHI 模式 context
    int sahi_batch_size_ = 8;                            // SAHI 引擎 batch 大小

    // Ring Buffer（4 个 slot）
    RingBuffer ring_buffer_;

    // FFmpeg CUDA hwaccel 解码器（视频文件）
    FFmpegHWDecoder decoder_;

    // V4L2 独立采集器（摄像头）
    V4L2Capture v4l2_;

    // libuvc 采集器（UVC 摄像头）
#if HAS_LIBUVC
    UvcCapture uvc_;
#endif

    // CUDA stream（4 级流水线）
    cudaStream_t decode_stream_ = nullptr;
    cudaStream_t preproc_stream_ = nullptr;
    cudaStream_t infer_stream_ = nullptr;
    cudaStream_t postproc_stream_ = nullptr;

    // SAHI 切片器（SAHI 模式用）
    SAHISlicer slicer_;

    // SAHI 批量推理缓冲区（动态 batch）
    float* sahi_batch_input_ = nullptr;     // GPU: [max_batch, 3, 640, 640]
    float* sahi_batch_output_ = nullptr;    // GPU: [max_batch, 25200, 85]
    float* sahi_host_output_ = nullptr;     // Pinned host: [max_batch, 25200, 85]
    size_t sahi_batch_input_bytes_ = 0;
    size_t sahi_batch_output_bytes_ = 0;

    // SAHI 帧跳过 + 检测缓存（保持高帧率）
    int sahi_interval_ = 3;                     // SAHI 每隔多少帧推理一次
    int sahi_counter_ = 0;                      // 冷却计数器
    std::vector<Detection> sahi_cached_dets_;   // 缓存检测框
    cv::Mat sahi_cached_display_;               // 缓存显示帧

    // SAHI 引擎策略（动态 batch 时复用主引擎，否则用独立 SAHI 引擎）
    bool sahi_engine_supports_dynamic_ = false;
    nvinfer1::IExecutionContext* sahi_use_context_ = nullptr;  // SAHI 推理使用的 context
    DetectionEngine* sahi_use_engine_ = nullptr;               // SAHI 推理使用的引擎

    // 摄像头 YUYV 帧（用于显示）
    cv::Mat cam_yuyv_frame_;
    cv::Mat cam_bgr_for_rec_;    // YUYV→BGR 转换缓冲（录制用）
    cv::Mat cam_gray_for_rec_;   // grey 相机 Y 通道提取缓冲（录制用）

    // GPU 显示缓冲区（NV12→BGR+resize 结果，避免全帧 D2H）
    uint8_t* display_gpu_buf_ = nullptr;
    size_t display_gpu_buf_size_ = 0;

    // 共享显示缓冲区（CUDA-OpenGL 互操作，未来零拷贝路径）
    uint8_t* shared_display_buf_ = nullptr;
    int shared_display_w_ = 0;
    int shared_display_h_ = 0;

    // 摄像头 GPU 传输缓冲区（每 Pipeline 独立，不能用 static）
    uint8_t* cam_gpu_buf_ = nullptr;
    size_t cam_gpu_buf_size_ = 0;
    uint8_t* cam_pinned_buf_ = nullptr;
    uint8_t* cam_gpu_fallback_buf_ = nullptr;
    size_t cam_fallback_buf_size_ = 0;

    std::thread worker_;
    std::atomic<bool> running_{false};

    // 结果
    std::mutex result_mtx_;
    PipelineResult result_;

    // 视频录制器
    VideoRecorder recorder_;

    // 硬件录制器（4K 原始帧 H.264 编码）
    HWRecorder hw_recorder_;
    uint8_t* rec_nv12_buf_ = nullptr;   // NV12 D2H 缓冲区（录制用）
    size_t rec_nv12_size_ = 0;

    // Runtime model geometry (queried from TensorRT, supports YOLOv5 and YOLO26).
    int model_input_w_ = 640;
    int model_input_h_ = 640;
    int model_output_elements_ = 25200 * 85;
    int model_output_predictions_ = 25200;
    int model_output_stride_ = 85;
    bool model_end2end_ = false;

    // 统计
    int frame_count_ = 0;
    std::chrono::steady_clock::time_point fps_start_;
    std::chrono::steady_clock::time_point last_display_update_;
};
