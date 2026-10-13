#include "nvdec_decoder.h"
#include "logger.h"

NVDECDecoder::NVDECDecoder() = default;

NVDECDecoder::~NVDECDecoder() { release(); }

bool NVDECDecoder::alloc_gpu_buffers() {
    // 确保 CUDA context 是当前上下文
    cuCtxSetCurrent(cu_context_);

    // 使用 NVDEC 实际报告的 pitch（在 create_decoder 中设置）
    // 如果 pitch 还未知（首次分配），使用 width 的 16 字节对齐值
    if (gpu_pitch_ <= 0) {
        gpu_pitch_ = ((width_ + 15) / 16) * 16;
    }
    // NV12 总大小: Y(height * pitch) + UV(height/2 * pitch)
    gpu_buf_size_ = gpu_pitch_ * height_ * 3 / 2;

    for (int i = 0; i < GPU_BUF_COUNT; i++) {
        CUresult err = cuMemAlloc((CUdeviceptr*)&gpu_nv12_bufs_[i], gpu_buf_size_);
        if (err != CUDA_SUCCESS) {
            LOG_ERROR("Failed to allocate GPU NV12 buffer %d: size=%zu, err=%d", i, gpu_buf_size_, err);
            free_gpu_buffers();
            return false;
        }
    }

    LOG_INFO("Allocated %d GPU NV12 buffers: %zu bytes each, pitch=%d, %dx%d",
             GPU_BUF_COUNT, gpu_buf_size_, gpu_pitch_, width_, height_);
    return true;
}

void NVDECDecoder::free_gpu_buffers() {
    for (int i = 0; i < GPU_BUF_COUNT; i++) {
        if (gpu_nv12_bufs_[i]) {
            cuMemFree((CUdeviceptr)gpu_nv12_bufs_[i]);
            gpu_nv12_bufs_[i] = nullptr;
        }
    }
}

bool NVDECDecoder::open(const std::string& path) {
    path_ = path;

    // 初始化 FFmpeg 解封装
    if (!init_ffmpeg(path)) {
        LOG_ERROR("Failed to init FFmpeg for: %s", path.c_str());
        return false;
    }

    // 初始化 NVDEC
    if (!init_nvdec()) {
        LOG_ERROR("Failed to init NVDEC");
        close_ffmpeg();
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
    decode_thread_ = std::thread(&NVDECDecoder::decode_thread_func, this);

    // 等待第一帧解码完成（确保 decoder 创建成功，width_/height_ 已知）
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

    // GPU NV12 缓冲区在 HandlePictureDisplay 中按需分配（首次帧时）
    // 此时 width_/height_ 已由回调设置，但 pitch 尚未知

    is_open_ = true;
    LOG_INFO("NVDEC decoder opened: %s (%dx%d @ %.0ffps)",
             path.c_str(), width_, height_, fps_);
    return true;
}

bool NVDECDecoder::init_ffmpeg(const std::string& path) {
    // 打开输入文件
    int ret = avformat_open_input(&format_ctx_, path.c_str(), nullptr, nullptr);
    if (ret < 0) {
        char err_buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, err_buf, sizeof(err_buf));
        LOG_ERROR("Could not open input file: %s (%s)", path.c_str(), err_buf);
        return false;
    }

    // 获取流信息
    ret = avformat_find_stream_info(format_ctx_, nullptr);
    if (ret < 0) {
        LOG_ERROR("Could not find stream info");
        close_ffmpeg();
        return false;
    }

    // 查找视频流
    for (unsigned int i = 0; i < format_ctx_->nb_streams; i++) {
        if (format_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx_ = i;
            break;
        }
    }

    if (video_stream_idx_ == -1) {
        LOG_ERROR("Could not find video stream");
        close_ffmpeg();
        return false;
    }

    // 获取视频信息
    AVStream* video_stream = format_ctx_->streams[video_stream_idx_];
    AVCodecParameters* codec_params = video_stream->codecpar;

    width_ = codec_params->width;
    height_ = codec_params->height;

    // 计算帧率
    if (video_stream->avg_frame_rate.den > 0) {
        fps_ = (float)video_stream->avg_frame_rate.num / video_stream->avg_frame_rate.den;
    } else if (video_stream->r_frame_rate.den > 0) {
        fps_ = (float)video_stream->r_frame_rate.num / video_stream->r_frame_rate.den;
    } else {
        fps_ = 30.0f;
    }

    LOG_INFO("FFmpeg demuxer opened: %dx%d @ %.0ffps, codec=%d",
             width_, height_, fps_, codec_params->codec_id);
    return true;
}

void NVDECDecoder::close_ffmpeg() {
    if (format_ctx_) {
        avformat_close_input(&format_ctx_);
        format_ctx_ = nullptr;
    }
    video_stream_idx_ = -1;
}

bool NVDECDecoder::init_nvdec() {
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

    // 设置为当前上下文
    err = cuCtxSetCurrent(cu_context_);
    if (err != CUDA_SUCCESS) {
        LOG_ERROR("cuCtxSetCurrent failed");
        return false;
    }

    LOG_INFO("NVDEC initialized successfully");
    return true;
}

bool NVDECDecoder::create_parser() {
    // 获取视频流的 codec 参数
    AVStream* video_stream = format_ctx_->streams[video_stream_idx_];
    AVCodecParameters* codec_params = video_stream->codecpar;

    // 确定 codec 类型
    cudaVideoCodec codec_type;
    switch (codec_params->codec_id) {
        case AV_CODEC_ID_H264:
            codec_type = cudaVideoCodec_H264;
            break;
        case AV_CODEC_ID_HEVC:
            codec_type = cudaVideoCodec_HEVC;
            break;
        case AV_CODEC_ID_VP9:
            codec_type = cudaVideoCodec_VP9;
            break;
        case AV_CODEC_ID_AV1:
            codec_type = cudaVideoCodec_AV1;
            break;
        default:
            LOG_ERROR("Unsupported codec: %d", codec_params->codec_id);
            return false;
    }

    LOG_INFO("NVDEC codec mapping: AV_CODEC_ID=%d -> cudaVideoCodec=%d", codec_params->codec_id, codec_type);

    // 设置视频解析器参数
    CUVIDPARSERPARAMS params;
    memset(&params, 0, sizeof(params));
    params.CodecType = codec_type;
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

    LOG_INFO("NVDEC parser created for codec %d", codec_type);
    return true;
}

bool NVDECDecoder::create_decoder(const CUVIDEOFORMAT* pVideoFormat) {
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

    LOG_INFO("Creating NVDEC decoder: codec=%d, chroma=%d, %dx%d, bitDepth=%d",
             params.CodecType, params.ChromaFormat,
             params.ulWidth, params.ulHeight, params.bitDepthMinus8);

    CUresult err = cuvidCreateDecoder(&decoder_, &params);
    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        LOG_ERROR("cuvidCreateDecoder failed: %s (err=%d)", errStr, err);
        return false;
    }

    width_ = pVideoFormat->coded_width;
    height_ = pVideoFormat->coded_height;
    fps_ = (float)pVideoFormat->frame_rate.numerator / pVideoFormat->frame_rate.denominator;

    LOG_INFO("NVDEC decoder created: %dx%d @ %.0ffps", width_, height_, fps_);
    return true;
}

int CUDAAPI NVDECDecoder::HandleVideoSequence(void* pUserData, CUVIDEOFORMAT* pVideoFormat) {
    NVDECDecoder* decoder = static_cast<NVDECDecoder*>(pUserData);
    LOG_INFO("HandleVideoSequence called: %dx%d, codec=%d, chroma=%d",
             pVideoFormat->coded_width, pVideoFormat->coded_height,
             pVideoFormat->codec, pVideoFormat->chroma_format);

    // 确保 CUDA 上下文是当前上下文
    CUresult err = cuCtxSetCurrent(decoder->cu_context_);
    if (err != CUDA_SUCCESS) {
        LOG_ERROR("HandleVideoSequence: cuCtxSetCurrent failed, err=%d", err);
        return 0;
    }

    if (!decoder->decoder_) {
        if (!decoder->create_decoder(pVideoFormat)) {
            LOG_ERROR("HandleVideoSequence: create_decoder failed");
            return 0;
        }
    }

    LOG_INFO("HandleVideoSequence: success");
    return 1;
}

int CUDAAPI NVDECDecoder::HandlePictureDecode(void* pUserData, CUVIDPICPARAMS* pPicParams) {
    NVDECDecoder* decoder = static_cast<NVDECDecoder*>(pUserData);

    if (!decoder->decoder_) {
        LOG_ERROR("HandlePictureDecode: decoder is null");
        return 0;
    }

    static int decode_count = 0;
    decode_count++;
    if (decode_count % 100 == 0) {
        LOG_INFO("HandlePictureDecode: called %d times, PicIdx=%d, FrameIdx=%d",
                 decode_count, pPicParams->CurrPicIdx, pPicParams->CurrPicIdx);
    }

    CUresult err = cuvidDecodePicture(decoder->decoder_, pPicParams);
    if (err != CUDA_SUCCESS) {
        LOG_ERROR("HandlePictureDecode: cuvidDecodePicture failed, err=%d", err);
        return 0;
    }

    return 1;
}

int CUDAAPI NVDECDecoder::HandlePictureDisplay(void* pUserData, CUVIDPARSERDISPINFO* pDispInfo) {
    NVDECDecoder* decoder = static_cast<NVDECDecoder*>(pUserData);

    if (!decoder->decoder_ || !pDispInfo) {
        LOG_ERROR("HandlePictureDisplay: decoder=%p, pDispInfo=%p", decoder->decoder_, pDispInfo);
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
        LOG_ERROR("HandlePictureDisplay: cuvidMapVideoFrame failed, err=%d", err);
        return 0;
    }

    int width = decoder->width_;
    int height = decoder->height_;
    size_t nv12_size = (size_t)pitch * height * 3 / 2;

    // ===== GPU 零拷贝路径: NV12 保留在 GPU =====
    bool gpu_queued = false;

    // 首次检测到 pitch 变化时，重新分配 GPU 缓冲区
    if (decoder->gpu_nv12_bufs_[0] && (int)pitch != decoder->gpu_pitch_) {
        LOG_INFO("HandlePictureDisplay: pitch changed %d -> %u, reallocating GPU buffers",
                 decoder->gpu_pitch_, pitch);
        decoder->free_gpu_buffers();
        decoder->gpu_pitch_ = (int)pitch;
        decoder->alloc_gpu_buffers();
    }

    // 如果 GPU 缓冲区未分配（首次帧），尝试分配
    if (!decoder->gpu_nv12_bufs_[0]) {
        decoder->gpu_pitch_ = (int)pitch;
        decoder->alloc_gpu_buffers();
    }

    if (decoder->gpu_nv12_bufs_[0]) {
        std::unique_lock<std::mutex> lock(decoder->gpu_queue_mutex_);
        if (decoder->gpu_frame_queue_.size() < GPU_BUF_COUNT) {
            int buf_idx = decoder->frame_count_ % GPU_BUF_COUNT;
            // GPU 内拷贝: NVDEC 输出 → 预分配缓冲区
            cuMemcpyDtoD((CUdeviceptr)decoder->gpu_nv12_bufs_[buf_idx],
                         dptr, nv12_size);
            GpuFrameQueueItem item;
            item.data = decoder->gpu_nv12_bufs_[buf_idx];
            item.width = width;
            item.height = height;
            item.pitch = (int)pitch;
            item.buf_index = buf_idx;
            decoder->gpu_frame_queue_.push(item);
            gpu_queued = true;
        }
    }

    // ===== CPU 兼容路径: GPU→CPU，NV12→BGR =====
    // 必须在 unmap 之前做 CPU 拷贝
    std::vector<uint8_t> nv12_buffer(nv12_size);
    CUDA_MEMCPY2D memcpy2d;
    memset(&memcpy2d, 0, sizeof(memcpy2d));
    memcpy2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    memcpy2d.srcDevice = dptr;
    memcpy2d.srcPitch = pitch;
    memcpy2d.dstMemoryType = CU_MEMORYTYPE_HOST;
    memcpy2d.dstHost = nv12_buffer.data();
    memcpy2d.dstPitch = pitch;
    memcpy2d.WidthInBytes = width;
    memcpy2d.Height = height * 3 / 2;

    err = cuMemcpy2D(&memcpy2d);
    if (err != CUDA_SUCCESS) {
        LOG_ERROR("HandlePictureDisplay: cuMemcpy2D failed, err=%d", err);
        cuvidUnmapVideoFrame(decoder->decoder_, dptr);
        return 0;
    }

    // 释放 NVDEC 帧映射
    cuvidUnmapVideoFrame(decoder->decoder_, dptr);

    if (gpu_queued) {
        decoder->gpu_queue_cv_.notify_one();
    }

    // 转换 NV12 到 BGR
    cv::Mat bgr;
    if (pitch == (unsigned int)width) {
        cv::Mat nv12(height * 3 / 2, width, CV_8UC1, nv12_buffer.data());
        cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
    } else {
        cv::Mat nv12(height * 3 / 2, width, CV_8UC1);
        for (int y = 0; y < height; y++) {
            memcpy(nv12.ptr<uint8_t>(y), nv12_buffer.data() + y * pitch, width);
        }
        for (int y = 0; y < height / 2; y++) {
            memcpy(nv12.ptr<uint8_t>(height + y),
                   nv12_buffer.data() + pitch * height + y * pitch, width);
        }
        cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
    }

    // 放入 CPU 队列
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

    if (decoder->frame_count_ % 100 == 0) {
        LOG_INFO("HandlePictureDisplay: frame %d, pitch=%u, %dx%d, cpu_q=%zu, gpu_q=%zu",
                 decoder->frame_count_, pitch, width, height,
                 decoder->frame_queue_.size(), decoder->gpu_frame_queue_.size());
    }

    return 1;
}

// 将 H.264 AVCC extradata 转换为 Annex B 格式
// AVCC extradata 格式：
// byte 0: configurationVersion (0x01)
// byte 1: AVCProfileIndication
// byte 2: profile_compatibility
// byte 3: AVCLevelIndication
// byte 4: lengthSizeMinusOne & 0x03 (NALU 长度字段大小 - 1)
// byte 5: numOfSequenceParameterSets & 0x1F (SPS 数量)
// 然后是 SPS 和 PPS 数据
static std::vector<uint8_t> convert_avcc_extradata_to_annex_b(const uint8_t* data, int size) {
    std::vector<uint8_t> result;

    if (size < 6) {
        LOG_ERROR("AVCC extradata too small: %d bytes", size);
        return result;
    }

    int nalu_length_size = (data[4] & 0x03) + 1;  // NALU 长度字段大小（通常为 4）
    int num_sps = data[5] & 0x1F;  // SPS 数量

    LOG_INFO("AVCC extradata: nalu_length_size=%d, num_sps=%d", nalu_length_size, num_sps);

    int offset = 6;

    // 提取 SPS
    for (int i = 0; i < num_sps && offset + 2 <= size; i++) {
        uint16_t sps_length = (data[offset] << 8) | data[offset + 1];
        offset += 2;

        if (offset + sps_length > size) {
            LOG_ERROR("SPS length exceeds data size: %d > %d", offset + sps_length, size);
            break;
        }

        // 添加 Annex B 起始码
        result.push_back(0x00);
        result.push_back(0x00);
        result.push_back(0x00);
        result.push_back(0x01);

        // 添加 SPS 数据
        result.insert(result.end(), data + offset, data + offset + sps_length);
        offset += sps_length;

        LOG_INFO("AVCC extradata: SPS %d, length=%d", i, sps_length);
    }

    // 提取 PPS
    if (offset < size) {
        int num_pps = data[offset];
        offset++;

        LOG_INFO("AVCC extradata: num_pps=%d", num_pps);

        for (int i = 0; i < num_pps && offset + 2 <= size; i++) {
            uint16_t pps_length = (data[offset] << 8) | data[offset + 1];
            offset += 2;

            if (offset + pps_length > size) {
                LOG_ERROR("PPS length exceeds data size: %d > %d", offset + pps_length, size);
                break;
            }

            // 添加 Annex B 起始码
            result.push_back(0x00);
            result.push_back(0x00);
            result.push_back(0x00);
            result.push_back(0x01);

            // 添加 PPS 数据
            result.insert(result.end(), data + offset, data + offset + pps_length);
            offset += pps_length;

            LOG_INFO("AVCC extradata: PPS %d, length=%d", i, pps_length);
        }
    }

    LOG_INFO("AVCC extradata to Annex B: input=%d bytes, output=%zu bytes", size, result.size());
    return result;
}

// 将 AVCC 格式的数据包转换为 Annex B 格式
// AVCC 数据包格式：每个 NALU 前面有 4 字节（或 2 字节）的长度前缀
static std::vector<uint8_t> convert_avcc_packet_to_annex_b(const uint8_t* data, int size) {
    std::vector<uint8_t> result;
    result.reserve(size + 1024);

    int i = 0;
    while (i < size) {
        // 检查是否有足够的数据读取长度
        if (i + 4 > size) break;

        // 读取 4 字节长度（大端序）
        uint32_t length = (data[i] << 24) | (data[i+1] << 16) | (data[i+2] << 8) | data[i+3];
        i += 4;

        // 检查长度是否有效
        if (length == 0 || i + length > (size_t)size) break;

        // 添加起始码
        result.push_back(0x00);
        result.push_back(0x00);
        result.push_back(0x00);
        result.push_back(0x01);

        // 添加 NALU 数据
        result.insert(result.end(), data + i, data + i + length);
        i += length;
    }

    return result;
}

void NVDECDecoder::decode_thread_func() {
    LOG_INFO("NVDEC decode thread started");

    AVPacket* packet = av_packet_alloc();
    if (!packet) {
        LOG_ERROR("Could not allocate packet");
        return;
    }

    // 首先发送 extradata（SPS/PPS）
    AVStream* video_stream = format_ctx_->streams[video_stream_idx_];
    AVCodecParameters* codec_params = video_stream->codecpar;

    if (codec_params->extradata && codec_params->extradata_size > 0) {
        LOG_INFO("NVDEC decode: sending extradata, size=%d", codec_params->extradata_size);

        // 打印 extradata 前 16 字节
        char hex_str[64] = {0};
        for (int i = 0; i < std::min(16, codec_params->extradata_size); i++) {
            sprintf(hex_str + i * 3, "%02x ", codec_params->extradata[i]);
        }
        LOG_INFO("NVDEC decode: extradata hex: %s", hex_str);

        // 检查 extradata 格式
        // AVCC 格式：以 0x01 开头，包含 SPS/PPS
        // Annex B 格式：以 0x00 0x00 0x00 0x01 开头
        bool is_avcc = (codec_params->extradata_size > 0 && codec_params->extradata[0] == 0x01);
        bool is_annex_b = (codec_params->extradata_size >= 4 &&
                           codec_params->extradata[0] == 0x00 &&
                           codec_params->extradata[1] == 0x00 &&
                           codec_params->extradata[2] == 0x00 &&
                           codec_params->extradata[3] == 0x01);

        LOG_INFO("NVDEC decode: extradata format: %s (is_avcc=%d, is_annex_b=%d)",
                 is_avcc ? "AVCC" : (is_annex_b ? "Annex B" : "Unknown"),
                 is_avcc, is_annex_b);

        const uint8_t* payload = nullptr;
        size_t payload_size = 0;
        std::vector<uint8_t> annex_b_extradata;

        if (is_avcc) {
            // AVCC 格式，需要转换为 Annex B
            LOG_INFO("NVDEC decode: converting AVCC extradata to Annex B");
            annex_b_extradata = convert_avcc_extradata_to_annex_b(
                codec_params->extradata, codec_params->extradata_size);

            if (!annex_b_extradata.empty()) {
                payload = annex_b_extradata.data();
                payload_size = annex_b_extradata.size();
                LOG_INFO("NVDEC decode: converted extradata size=%zu", payload_size);
            } else {
                LOG_ERROR("NVDEC decode: failed to convert AVCC extradata");
                // 回退：直接发送原始数据
                payload = codec_params->extradata;
                payload_size = codec_params->extradata_size;
            }
        } else if (is_annex_b) {
            // Annex B 格式，直接使用
            payload = codec_params->extradata;
            payload_size = codec_params->extradata_size;
            LOG_INFO("NVDEC decode: using Annex B extradata directly");
        } else {
            // 未知格式，尝试直接发送
            payload = codec_params->extradata;
            payload_size = codec_params->extradata_size;
            LOG_WARN("NVDEC decode: unknown extradata format, sending as-is");
        }

        CUVIDSOURCEDATAPACKET extradata_packet;
        memset(&extradata_packet, 0, sizeof(extradata_packet));
        extradata_packet.payload = payload;
        extradata_packet.payload_size = payload_size;
        extradata_packet.flags = CUVID_PKT_TIMESTAMP;
        extradata_packet.timestamp = 0;

        CUresult err = cuvidParseVideoData(parser_, &extradata_packet);
        if (err != CUDA_SUCCESS) {
            LOG_ERROR("cuvidParseVideoData failed for extradata, err=%d", err);
        } else {
            LOG_INFO("NVDEC decode: extradata sent successfully");
        }
    } else {
        LOG_WARN("NVDEC decode: no extradata found");
    }

    int packet_count = 0;

    while (!stop_decode_) {
        // 读取一个包
        int ret = av_read_frame(format_ctx_, packet);

        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                // 文件结束，循环播放
                LOG_INFO("NVDEC decode: EOF reached, seeking to start");
                av_seek_frame(format_ctx_, video_stream_idx_, 0, AVSEEK_FLAG_BACKWARD);
                continue;
            }
            LOG_ERROR("Error reading frame: %d", ret);
            break;
        }

        // 只处理视频流
        if (packet->stream_index != video_stream_idx_) {
            av_packet_unref(packet);
            continue;
        }

        packet_count++;

        // 每100个包打印一次日志
        if (packet_count % 100 == 0) {
            LOG_INFO("NVDEC decode: fed %d packets, size=%d, pts=%ld",
                     packet_count, packet->size, packet->pts);
        }

        // 检查数据包格式并转换
        const uint8_t* payload = packet->data;
        size_t payload_size = packet->size;

        // 检查是否是 AVCC 格式（长度前缀）
        // AVCC 格式：每个 NALU 以 4 字节长度前缀开始
        // Annex B 格式：每个 NALU 以 00 00 00 01 开始
        bool is_avcc = false;
        if (packet->size >= 4) {
            // 检查前 4 字节是否是长度值（不是起始码）
            uint32_t first_nalu_length = (packet->data[0] << 24) | (packet->data[1] << 16) |
                                         (packet->data[2] << 8) | packet->data[3];
            // 如果长度值合理（小于包大小），可能是 AVCC 格式
            if (first_nalu_length > 0 && first_nalu_length < packet->size) {
                is_avcc = true;
            }
        }

        std::vector<uint8_t> annex_b_data;
        if (is_avcc) {
            annex_b_data = convert_avcc_packet_to_annex_b(packet->data, packet->size);
            if (!annex_b_data.empty()) {
                payload = annex_b_data.data();
                payload_size = annex_b_data.size();
            } else {
                // 转换失败，使用原始数据
                LOG_WARN("NVDEC decode: failed to convert packet %d to Annex B", packet_count);
            }
        }

        // 将数据喂给 NVDEC 解析器
        CUVIDSOURCEDATAPACKET cuvid_packet;
        memset(&cuvid_packet, 0, sizeof(cuvid_packet));
        cuvid_packet.payload = payload;
        cuvid_packet.payload_size = payload_size;
        cuvid_packet.flags = CUVID_PKT_TIMESTAMP;
        cuvid_packet.timestamp = packet->pts;

        // 每100个包打印一次详细信息
        if (packet_count % 100 == 0) {
            LOG_INFO("NVDEC parse: packet %d, format=%s, original=%d, converted=%zu, pts=%ld",
                     packet_count, is_avcc ? "AVCC" : "Annex B",
                     packet->size, payload_size, packet->pts);
        }

        CUresult err = cuvidParseVideoData(parser_, &cuvid_packet);
        if (err != CUDA_SUCCESS) {
            LOG_ERROR("cuvidParseVideoData failed for packet %d, size=%zu, err=%d",
                      packet_count, annex_b_data.size(), err);
            av_packet_unref(packet);
            continue;
        }

        // 检查是否有帧被解码
        if (packet_count % 100 == 0) {
            LOG_INFO("NVDEC parse: after packet %d, decoded %d frames", packet_count, frame_count_);
        }

        av_packet_unref(packet);

        // 稍微等待一下，避免占用太多 CPU
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    av_packet_free(&packet);

    LOG_INFO("NVDEC decode thread finished, fed %d packets, decoded %d frames",
             packet_count, frame_count_);
}

bool NVDECDecoder::read(cv::Mat& frame) {
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

bool NVDECDecoder::read_gpu(GpuFrame& frame) {
    if (!is_open_ || !gpu_nv12_bufs_[0]) return false;

    // 从 GPU 帧队列获取
    {
        std::unique_lock<std::mutex> lock(gpu_queue_mutex_);
        if (gpu_frame_queue_.empty()) {
            gpu_queue_cv_.wait_for(lock, std::chrono::milliseconds(100));
            if (gpu_frame_queue_.empty()) return false;
        }

        GpuFrameQueueItem item = gpu_frame_queue_.front();
        gpu_frame_queue_.pop();
        frame.data = item.data;
        frame.width = item.width;
        frame.height = item.height;
        frame.pitch = item.pitch;
        frame.buf_index = item.buf_index;
    }
    gpu_queue_cv_.notify_one();

    return true;
}

void NVDECDecoder::release_gpu_frame(const GpuFrame& frame) {
    // 当前使用轮转缓冲区，无需显式归还
    // 如果后续改为引用计数，在此实现
    (void)frame;
}

void NVDECDecoder::release() {
    stop_decode_ = true;
    queue_cv_.notify_all();
    gpu_queue_cv_.notify_all();

    if (decode_thread_.joinable()) {
        decode_thread_.join();
    }

    close_ffmpeg();

    // 释放 GPU 缓冲区池
    free_gpu_buffers();

    // 清空 GPU 帧队列
    {
        std::unique_lock<std::mutex> lock(gpu_queue_mutex_);
        while (!gpu_frame_queue_.empty()) gpu_frame_queue_.pop();
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
