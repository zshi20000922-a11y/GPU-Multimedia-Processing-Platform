#pragma once

#include <cuda_runtime.h>
#include <nvcuvid.h>
#include <cuviddec.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <fstream>

// GPU 视频解码器（使用 NVDEC）
class GPUDecoder {
public:
    GPUDecoder();
    ~GPUDecoder();

    // 打开视频文件
    bool open(const std::string& path);

    // 读取一帧（GPU 内存）
    bool read(cv::Mat& frame);

    // 释放资源
    void release();

    // 获取视频信息
    int width() const { return width_; }
    int height() const { return height_; }
    float fps() const { return fps_; }
    bool is_open() const { return is_open_; }

private:
    // NVDEC 回调函数
    static int CUDAAPI HandleVideoSequence(void* pUserData, CUVIDEOFORMAT* pVideoFormat);
    static int CUDAAPI HandlePictureDecode(void* pUserData, CUVIDPICPARAMS* pPicParams);
    static int CUDAAPI HandlePictureDisplay(void* pUserData, CUVIDPARSERDISPINFO* pDispInfo);

    // 初始化 NVDEC
    bool init_nvdec();
    bool create_parser();
    bool create_decoder(const CUVIDEOFORMAT* pVideoFormat);

    // 解码线程
    void decode_thread_func();

    // 成员变量
    std::string path_;
    int width_ = 0;
    int height_ = 0;
    float fps_ = 0;
    bool is_open_ = false;

    // CUDA 资源
    CUcontext cu_context_ = nullptr;
    CUvideodecoder decoder_ = nullptr;
    CUvideoparser parser_ = nullptr;

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

    // 文件流
    std::ifstream file_;
    size_t file_size_ = 0;

    // 帧计数
    int frame_count_ = 0;
};
