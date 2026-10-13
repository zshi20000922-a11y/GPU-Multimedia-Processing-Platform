#include "gpu_decoder.h"
#include "logger.h"
#include <fstream>

// 错误检查宏
#define CUDA_CHECK(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        LOG_ERROR("CUDA error %d at %s:%d - %s", err, __FILE__, __LINE__, errStr); \
        return false; \
    } \
} while(0)

GPUDecoder::GPUDecoder() = default;

GPUDecoder::~GPUDecoder() { release(); }

bool GPUDecoder::open(const std::string& path) {
    path_ = path;

    // 初始化 CUDA
    if (!init_nvdec()) {
        LOG_ERROR("Failed to init NVDEC");
        return false;
    }

    // 创建解析器
    if (!create_parser()) {
        LOG_ERROR("Failed to create video parser");
        release();
        return false;
    }

    // 启动解码线程
    stop_decode_ = false;
    decode_thread_ = std::thread(&GPUDecoder::decode_thread_func, this);

    // 等待第一帧解码完成
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait_for(lock, std::chrono::seconds(10), [this] {
            return !frame_queue_.empty() || stop_decode_;
        });
    }

    if (frame_queue_.empty()) {
        LOG_ERROR("Failed to decode first frame within 10 seconds");
        release();
        return false;
    }

    is_open_ = true;
    LOG_INFO("GPU decoder opened: %s (%dx%d @ %.0ffps)", path.c_str(), width_, height_, fps_);
    return true;
}

bool GPUDecoder::init_nvdec() {
    // 初始化 CUDA 驱动 API
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        LOG_ERROR("cuInit failed: %s", errStr);
        return false;
    }

    // 获取 CUDA 设备
    CUdevice device;
    err = cuDeviceGet(&device, 0);
    if (err != CUDA_SUCCESS) {
        LOG_ERROR("cuDeviceGet failed");
        return false;
    }

    // 创建 CUDA 上下文
    err = cuCtxCreate(&cu_context_, CU_CTX_SCHED_BLOCKING_SYNC, device);
    if (err != CUDA_SUCCESS) {
        LOG_ERROR("cuCtxCreate failed");
        return false;
    }

    LOG_INFO("NVDEC initialized successfully");
    return true;
}

bool GPUDecoder::create_parser() {
    // 设置视频解析器参数
    CUVIDPARSERPARAMS params;
    memset(&params, 0, sizeof(params));
    params.CodecType = cudaVideoCodec_H264;  // 假设 H.264
    params.ulMaxNumDecodeSurfaces = 20;
    params.ulMaxDisplayDelay = 4;
    params.pUserData = this;
    params.pfnSequenceCallback = HandleVideoSequence;
    params.pfnDecodePicture = HandlePictureDecode;
    params.pfnDisplayPicture = HandlePictureDisplay;

    CUresult err = cuvidCreateVideoParser(&parser_, &params);
    if (err != CUDA_SUCCESS) {
        LOG_ERROR("cuvidCreateVideoParser failed");
        return false;
    }

    // 打开视频文件
    file_.open(path_, std::ios::binary);
    if (!file_.is_open()) {
        LOG_ERROR("Cannot open video file: %s", path_.c_str());
        return false;
    }

    // 获取文件大小
    file_.seekg(0, std::ios::end);
    file_size_ = file_.tellg();
    file_.seekg(0, std::ios::beg);

    LOG_INFO("Video file opened: %s (%zu bytes)", path_.c_str(), file_size_);
    return true;
}

bool GPUDecoder::create_decoder(const CUVIDEOFORMAT* pVideoFormat) {
    // 设置解码器参数
    CUVIDDECODECREATEINFO params;
    memset(&params, 0, sizeof(params));
    params.CodecType = pVideoFormat->codec;
    params.ChromaFormat = pVideoFormat->chroma_format;
    params.OutputFormat = cudaVideoSurfaceFormat_NV12;
    params.bitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;
    params.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    params.ulWidth = pVideoFormat->coded_width;
    params.ulHeight = pVideoFormat->coded_height;
    params.ulTargetWidth = pVideoFormat->coded_width;
    params.ulTargetHeight = pVideoFormat->coded_height;
    params.ulNumOutputSurfaces = 2;
    params.ulNumDecodeSurfaces = 20;
    params.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    params.vidLock = nullptr;

    CUresult err = cuvidCreateDecoder(&decoder_, &params);
    if (err != CUDA_SUCCESS) {
        LOG_ERROR("cuvidCreateDecoder failed");
        return false;
    }

    width_ = pVideoFormat->coded_width;
    height_ = pVideoFormat->coded_height;
    fps_ = (float)pVideoFormat->frame_rate.numerator / pVideoFormat->frame_rate.denominator;

    LOG_INFO("Decoder created: %dx%d @ %.0ffps", width_, height_, fps_);
    return true;
}

int CUDAAPI GPUDecoder::HandleVideoSequence(void* pUserData, CUVIDEOFORMAT* pVideoFormat) {
    GPUDecoder* decoder = static_cast<GPUDecoder*>(pUserData);
    LOG_INFO("HandleVideoSequence: %dx%d, codec=%d", pVideoFormat->coded_width, pVideoFormat->coded_height, pVideoFormat->codec);

    if (!decoder->decoder_) {
        if (!decoder->create_decoder(pVideoFormat)) {
            return 0;
        }
    }

    return 1;
}

int CUDAAPI GPUDecoder::HandlePictureDecode(void* pUserData, CUVIDPICPARAMS* pPicParams) {
    GPUDecoder* decoder = static_cast<GPUDecoder*>(pUserData);

    if (!decoder->decoder_) {
        return 0;
    }

    CUresult err = cuvidDecodePicture(decoder->decoder_, pPicParams);
    if (err != CUDA_SUCCESS) {
        return 0;
    }

    return 1;
}

int CUDAAPI GPUDecoder::HandlePictureDisplay(void* pUserData, CUVIDPARSERDISPINFO* pDispInfo) {
    GPUDecoder* decoder = static_cast<GPUDecoder*>(pUserData);

    if (!decoder->decoder_ || !pDispInfo) {
        return 0;
    }

    // 获取解码后的帧
    CUVIDPROCPARAMS procParams;
    memset(&procParams, 0, sizeof(procParams));
    procParams.progressive_frame = pDispInfo->progressive_frame;
    procParams.second_field = pDispInfo->repeat_first_field + 1;
    procParams.top_field_first = pDispInfo->top_field_first;
    procParams.unpaired_field = pDispInfo->repeat_first_field < 0;

    unsigned int pitch;
    CUdeviceptr dptr;
    CUresult err = cuvidMapVideoFrame(decoder->decoder_, pDispInfo->picture_index,
                                       &dptr, &pitch, &procParams);
    if (err != CUDA_SUCCESS) {
        return 0;
    }

    // 创建 NV12 帧
    int nv12_size = pitch * decoder->height_ * 3 / 2;
    std::vector<uint8_t> nv12_buffer(nv12_size);

    // 从 GPU 拷贝到 CPU
    CUDA_MEMCPY2D memcpy2d;
    memset(&memcpy2d, 0, sizeof(memcpy2d));
    memcpy2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    memcpy2d.srcDevice = dptr;
    memcpy2d.srcPitch = pitch;
    memcpy2d.dstMemoryType = CU_MEMORYTYPE_HOST;
    memcpy2d.dstHost = nv12_buffer.data();
    memcpy2d.dstPitch = pitch;
    memcpy2d.WidthInBytes = decoder->width_;
    memcpy2d.Height = decoder->height_ * 3 / 2;

    err = cuMemcpy2D(&memcpy2d);
    if (err != CUDA_SUCCESS) {
        cuvidUnmapVideoFrame(decoder->decoder_, dptr);
        return 0;
    }

    cuvidUnmapVideoFrame(decoder->decoder_, dptr);

    // 转换 NV12 到 BGR
    cv::Mat nv12(decoder->height_ * 3 / 2, decoder->width_, CV_8UC1, nv12_buffer.data());
    cv::Mat bgr;
    cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);

    // 放入队列
    {
        std::unique_lock<std::mutex> lock(decoder->queue_mutex_);
        if (decoder->frame_queue_.size() < MAX_QUEUE_SIZE) {
            DecodedFrame df;
            df.frame = bgr.clone();
            df.ready = true;
            decoder->frame_queue_.push(df);
        }
    }
    decoder->queue_cv_.notify_one();

    decoder->frame_count_++;
    return 1;
}

void GPUDecoder::decode_thread_func() {
    LOG_INFO("GPU decode thread started");

    // 分块读取视频文件并喂给解析器
    const size_t chunk_size = 65536;  // 64KB chunks
    std::vector<uint8_t> buffer(chunk_size);
    size_t total_read = 0;

    while (!stop_decode_ && total_read < file_size_) {
        // 读取一块数据
        size_t to_read = std::min(chunk_size, file_size_ - total_read);
        file_.read(reinterpret_cast<char*>(buffer.data()), to_read);
        size_t bytes_read = file_.gcount();

        if (bytes_read == 0) break;

        total_read += bytes_read;

        // 喂数据给解析器
        CUVIDSOURCEDATAPACKET packet;
        memset(&packet, 0, sizeof(packet));
        packet.payload = buffer.data();
        packet.payload_size = bytes_read;

        // 如果是最后一块，设置结束标志
        if (total_read >= file_size_) {
            packet.flags = CUVID_PKT_ENDOFSTREAM;
        }

        CUresult err = cuvidParseVideoData(parser_, &packet);
        if (err != CUDA_SUCCESS) {
            LOG_ERROR("cuvidParseVideoData failed at offset %zu", total_read);
            break;
        }

        // 稍微等待一下，避免占用太多 CPU
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    // 关闭文件
    file_.close();

    LOG_INFO("GPU decode thread finished, total read: %zu bytes", total_read);
}

bool GPUDecoder::read(cv::Mat& frame) {
    if (!is_open_) return false;

    // 从队列获取帧
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (frame_queue_.empty()) {
            queue_cv_.wait_for(lock, std::chrono::milliseconds(100));
            if (frame_queue_.empty()) return false;
        }

        DecodedFrame df = frame_queue_.front();
        frame_queue_.pop();
        frame = df.frame;
    }
    queue_cv_.notify_one();

    return true;
}

void GPUDecoder::release() {
    stop_decode_ = true;
    queue_cv_.notify_all();

    if (decode_thread_.joinable()) {
        decode_thread_.join();
    }

    if (file_.is_open()) {
        file_.close();
    }

    if (parser_) {
        cuvidDestroyVideoParser(parser_);
        parser_ = nullptr;
    }

    if (decoder_) {
        cuvidDestroyDecoder(decoder_);
        decoder_ = nullptr;
    }

    if (cu_context_) {
        cuCtxDestroy(cu_context_);
        cu_context_ = nullptr;
    }

    is_open_ = false;
}
