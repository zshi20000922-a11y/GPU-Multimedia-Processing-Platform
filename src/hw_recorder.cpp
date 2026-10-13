#include "hw_recorder.h"
#include "logger.h"

#include <cstring>
#include <algorithm>

bool HWRecorder::start(const std::string& path, int width, int height, float fps) {
    if (recording_) {
        LOG_WARN("HWRecorder: already recording");
        return false;
    }

    width_ = width;
    height_ = height;
    fps_ = fps;
    frames_written_ = 0;
    filepath_ = path;

    // 创建输出格式上下文
    int ret = avformat_alloc_output_context2(&fmt_ctx_, nullptr, nullptr, path.c_str());
    if (ret < 0 || !fmt_ctx_) {
        LOG_ERROR("HWRecorder: cannot create output context for %s", path.c_str());
        return false;
    }

    // 查找 H.264 编码器：优先 NVENC 硬件编码（4K@200fps），回退 libx264
    const AVCodec* codec = avcodec_find_encoder_by_name("h264_nvenc");
    use_nvenc_ = (codec != nullptr);
    if (!codec) codec = avcodec_find_encoder_by_name("libx264");
    if (!codec) codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        LOG_ERROR("HWRecorder: H.264 encoder not found");
        avformat_free_context(fmt_ctx_);
        fmt_ctx_ = nullptr;
        return false;
    }
    LOG_INFO("HWRecorder: using encoder %s", codec->name);

    // 创建编码器上下文
    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
        LOG_ERROR("HWRecorder: cannot alloc codec context");
        avformat_free_context(fmt_ctx_);
        fmt_ctx_ = nullptr;
        return false;
    }

    codec_ctx_->width = width;
    codec_ctx_->height = height;
    codec_ctx_->time_base = {1, (int)fps};
    codec_ctx_->framerate = {(int)fps, 1};
    codec_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
    codec_ctx_->bit_rate = 0;  // 使用 CRF 模式
    codec_ctx_->gop_size = (int)fps;  // 每秒一个关键帧
    codec_ctx_->max_b_frames = 0;

    // 编码器参数
    if (use_nvenc_) {
        // NVENC 硬件编码：最快 preset + 低延迟 tune，4K@200fps 可满帧
        av_opt_set(codec_ctx_->priv_data, "preset", "p1", 0);      // p1=最快 ... p7=最慢
        av_opt_set(codec_ctx_->priv_data, "tune", "ll", 0);        // low latency
        av_opt_set(codec_ctx_->priv_data, "rc", "vbr", 0);         // 码率控制
        av_opt_set(codec_ctx_->priv_data, "cq", "23", 0);          // 质量（类 CRF）
    } else {
        // libx264 ultrafast preset（最低 CPU 占用）
        av_opt_set(codec_ctx_->priv_data, "preset", "ultrafast", 0);
        av_opt_set(codec_ctx_->priv_data, "tune", "zerolatency", 0);
        av_opt_set(codec_ctx_->priv_data, "crf", "23", 0);  // 质量（18=好，23=默认，28=差）
    }

    // MP4 容器需要全局头
    if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
        codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    // 打开编码器
    ret = avcodec_open2(codec_ctx_, codec, nullptr);
    if (ret < 0) {
        LOG_ERROR("HWRecorder: cannot open encoder");
        avcodec_free_context(&codec_ctx_);
        avformat_free_context(fmt_ctx_);
        fmt_ctx_ = nullptr;
        codec_ctx_ = nullptr;
        return false;
    }

    // 创建流
    stream_ = avformat_new_stream(fmt_ctx_, nullptr);
    if (!stream_) {
        LOG_ERROR("HWRecorder: cannot create stream");
        stop();
        return false;
    }
    stream_->id = fmt_ctx_->nb_streams - 1;
    avcodec_parameters_from_context(stream_->codecpar, codec_ctx_);

    // 打开输出文件
    ret = avio_open(&fmt_ctx_->pb, path.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
        LOG_ERROR("HWRecorder: cannot open %s", path.c_str());
        stop();
        return false;
    }

    // 写文件头
    ret = avformat_write_header(fmt_ctx_, nullptr);
    if (ret < 0) {
        LOG_ERROR("HWRecorder: cannot write header");
        stop();
        return false;
    }

    // 分配帧
    frame_ = av_frame_alloc();
    frame_->format = AV_PIX_FMT_YUV420P;
    frame_->width = width;
    frame_->height = height;
    av_frame_get_buffer(frame_, 0);

    // 分配 packet
    packet_ = av_packet_alloc();

    // 创建色彩空间转换器（NV12/BGR → YUV420P）
    // 初始为 NV12，pushFrame 时根据实际格式重新创建
    sws_ctx_ = nullptr;

    recording_ = true;
    start_time_ = std::chrono::steady_clock::now();
    encode_thread_ = std::thread(&HWRecorder::encodeLoop, this);

    LOG_INFO("HWRecorder: started %s (%dx%d @ %.0f fps, codec=%s, limit=%.0fs)",
             path.c_str(), width, height, fps, codec->name, duration_limit_);
    return true;
}

void HWRecorder::stop() {
    if (!recording_ && !fmt_ctx_) return;
    recording_ = false;
    queue_cv_.notify_all();

    if (encode_thread_.joinable()) {
        encode_thread_.join();
    }

    // 刷新编码器
    if (codec_ctx_ && fmt_ctx_ && fmt_ctx_->pb) {
        avcodec_send_frame(codec_ctx_, nullptr);
        while (true) {
            int ret = avcodec_receive_packet(codec_ctx_, packet_);
            if (ret < 0) break;
            av_packet_rescale_ts(packet_, codec_ctx_->time_base, stream_->time_base);
            av_interleaved_write_frame(fmt_ctx_, packet_);
        }
        av_write_trailer(fmt_ctx_);
    }

    if (sws_ctx_) { sws_freeContext(sws_ctx_); sws_ctx_ = nullptr; }
    if (packet_) { av_packet_free(&packet_); }
    if (frame_) { av_frame_free(&frame_); }
    if (codec_ctx_) { avcodec_free_context(&codec_ctx_); }
    if (fmt_ctx_) {
        if (fmt_ctx_->pb) avio_closep(&fmt_ctx_->pb);
        avformat_free_context(fmt_ctx_);
        fmt_ctx_ = nullptr;
    }

    LOG_INFO("HWRecorder: stopped, %d frames written", frames_written_.load());
}

void HWRecorder::pushFrame(const RecordFrame& frame) {
    if (!recording_) return;

    std::lock_guard<std::mutex> lk(queue_mtx_);
    if (frame_queue_.size() >= MAX_QUEUE) {
        frame_queue_.pop();  // 丢弃最旧帧
    }
    frame_queue_.push(frame);
    queue_cv_.notify_one();
}

void HWRecorder::pushBGRFrame(const cv::Mat& bgr) {
    if (!recording_ || bgr.empty()) return;

    RecordFrame rf;
    rf.bgr_data = bgr.data;
    rf.bgr_stride = (int)bgr.step;
    rf.width = bgr.cols;
    rf.height = bgr.rows;
    rf.is_nv12 = false;
    pushFrame(rf);
}

void HWRecorder::encodeLoop() {
    while (recording_) {
        RecordFrame rf;

        // 等待帧
        {
            std::unique_lock<std::mutex> lk(queue_mtx_);
            queue_cv_.wait_for(lk, std::chrono::milliseconds(100),
                               [this]() { return !frame_queue_.empty() || !recording_; });
            if (!recording_ && frame_queue_.empty()) break;
            if (frame_queue_.empty()) continue;
            rf = frame_queue_.front();
            frame_queue_.pop();
        }

        // 时长限制检查
        if (duration_limit_ > 0) {
            float elapsed = std::chrono::duration<float>(
                std::chrono::steady_clock::now() - start_time_).count();
            if (elapsed >= duration_limit_) {
                LOG_INFO("HWRecorder: duration limit reached (%.1fs)", elapsed);
                recording_ = false;
                break;
            }
        }

        // 灰度直通：直接填 YUV420P 的 Y 平面，UV 置 128，跳过色彩转换
        if (rf.is_gray) {
            av_frame_make_writable(frame_);
            // Y 平面：逐行拷贝（处理源 stride 与目标 linesize 差异）
            int copy_w = std::min(width_, rf.gray_stride);
            for (int y = 0; y < height_; y++) {
                memcpy(frame_->data[0] + (size_t)y * frame_->linesize[0],
                       rf.gray_data + (size_t)y * rf.gray_stride, copy_w);
            }
            // U/V 平面（YUV420P 为半分辨率）置 128（无色度）
            int cw = width_ / 2, ch = height_ / 2;
            for (int p = 1; p <= 2; p++) {
                for (int y = 0; y < ch; y++) {
                    memset(frame_->data[p] + (size_t)y * frame_->linesize[p], 128, cw);
                }
            }

            frame_->pts = frames_written_++;
            int ret = avcodec_send_frame(codec_ctx_, frame_);
            if (ret < 0) continue;
            while (ret >= 0) {
                ret = avcodec_receive_packet(codec_ctx_, packet_);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                if (ret < 0) break;
                av_packet_rescale_ts(packet_, codec_ctx_->time_base, stream_->time_base);
                av_interleaved_write_frame(fmt_ctx_, packet_);
            }
            continue;  // 跳过下面的 sws_scale 路径
        }

        // 确保色彩空间转换器正确
        if (rf.is_nv12) {
            if (!sws_ctx_) {
                sws_ctx_ = sws_getContext(
                    width_, height_, AV_PIX_FMT_NV12,
                    width_, height_, AV_PIX_FMT_YUV420P,
                    SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
            }
        } else {
            if (!sws_ctx_) {
                sws_ctx_ = sws_getContext(
                    width_, height_, AV_PIX_FMT_BGR24,
                    width_, height_, AV_PIX_FMT_YUV420P,
                    SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
            }
        }

        if (!sws_ctx_) continue;

        // 准备源数据
        const uint8_t* src_data[4] = {};
        int src_linesize[4] = {};

        if (rf.is_nv12) {
            src_data[0] = rf.y_plane;
            src_data[1] = rf.uv_plane;
            src_linesize[0] = rf.y_pitch;
            src_linesize[1] = rf.uv_pitch;
        } else {
            src_data[0] = rf.bgr_data;
            src_linesize[0] = rf.bgr_stride;
        }

        // 转换到 YUV420P
        av_frame_make_writable(frame_);
        sws_scale(sws_ctx_, src_data, src_linesize, 0, height_,
                   frame_->data, frame_->linesize);

        frame_->pts = frames_written_++;

        // 编码
        int ret = avcodec_send_frame(codec_ctx_, frame_);
        if (ret < 0) continue;

        while (ret >= 0) {
            ret = avcodec_receive_packet(codec_ctx_, packet_);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if (ret < 0) break;

            av_packet_rescale_ts(packet_, codec_ctx_->time_base, stream_->time_base);
            av_interleaved_write_frame(fmt_ctx_, packet_);
        }
    }
}

float HWRecorder::recordedSeconds() const {
    if (!recording_) return 0;
    return std::chrono::duration<float>(
        std::chrono::steady_clock::now() - start_time_).count();
}
