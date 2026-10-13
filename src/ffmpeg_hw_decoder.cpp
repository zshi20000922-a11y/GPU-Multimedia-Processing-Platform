// FFmpeg + CUDA 硬件加速视频解码器
// 零拷贝：解码后帧直接在 GPU 显存，无需 cuMemcpy
// 独立解码线程 + 4 帧队列，吸收 B-frame 尖峰

#include "ffmpeg_hw_decoder.h"
#include "logger.h"
#include <algorithm>

FFmpegHWDecoder::FFmpegHWDecoder() {
    packet_ = av_packet_alloc();
}

FFmpegHWDecoder::~FFmpegHWDecoder() {
    stop();
    if (packet_) av_packet_free(&packet_);
    if (codec_ctx_) avcodec_free_context(&codec_ctx_);
    if (hw_device_ctx_) av_buffer_unref(&hw_device_ctx_);
    if (fmt_ctx_) avformat_close_input(&fmt_ctx_);
}

bool FFmpegHWDecoder::initHwContext() {
    int ret = av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_CUDA,
                                     nullptr, nullptr, 0);
    if (ret < 0) {
        char err[256];
        av_strerror(ret, err, sizeof(err));
        LOG_ERROR("Failed to create CUDA hw device context: %s", err);
        return false;
    }
    LOG_INFO("FFmpeg CUDA hwaccel context created");
    return true;
}

bool FFmpegHWDecoder::openCodec() {
    video_stream_idx_ = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO,
                                            -1, -1, nullptr, 0);
    if (video_stream_idx_ < 0) {
        LOG_ERROR("No video stream found");
        return false;
    }

    AVStream* stream = fmt_ctx_->streams[video_stream_idx_];
    const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!codec) {
        LOG_ERROR("Cannot find decoder for codec %d", stream->codecpar->codec_id);
        return false;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) return false;
    if (avcodec_parameters_to_context(codec_ctx_, stream->codecpar) < 0) return false;

    // 配置 CUDA 硬件加速
    codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
    codec_ctx_->get_format = [](AVCodecContext*, const AVPixelFormat* pix_fmts) -> AVPixelFormat {
        for (const AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; p++)
            if (*p == AV_PIX_FMT_CUDA) return *p;
        return pix_fmts[0];
    };
    codec_ctx_->extra_hw_frames = 8;

    if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
        LOG_ERROR("Cannot open codec %s", codec->name);
        return false;
    }

    width_ = codec_ctx_->width;
    height_ = codec_ctx_->height;

    AVRational fr = stream->avg_frame_rate;
    fps_ = (fr.den > 0) ? (double)fr.num / fr.den : 30.0;
    duration_sec_ = fmt_ctx_->duration > 0 ?
        (double)fmt_ctx_->duration / AV_TIME_BASE : 0.0;

    LOG_INFO("FFmpeg hw decoder opened: %s, %dx%d @ %.1f fps (codec=%s)",
             fmt_ctx_->url, width_, height_, fps_, codec->name);
    return true;
}

bool FFmpegHWDecoder::open(const std::string& filepath) {
    if (opened_) stop();

    if (avformat_open_input(&fmt_ctx_, filepath.c_str(), nullptr, nullptr) < 0) {
        LOG_ERROR("Cannot open file: %s", filepath.c_str());
        return false;
    }
    if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) return false;
    if (!initHwContext()) return false;
    if (!openCodec()) return false;

    opened_ = true;
    running_ = true;
    eof_ = false;

    // 启动独立解码线程
    decode_thread_ = std::thread(&FFmpegHWDecoder::decodeLoop, this);

    return true;
}

void FFmpegHWDecoder::decodeLoop() {
    // 帧率控制（每帧动态检查 target_fps_，支持运行时切换）：
    // target_fps_ > 0: 锁定到指定帧率
    // target_fps_ == 0: 按视频原始帧率解码（默认，避免多路争抢 NVDEC）
    // target_fps_ < 0: 全速解码（不限制）
    auto next_frame_time = std::chrono::steady_clock::now();
    double last_target = -999;  // 强制首帧计算

    LOG_INFO("FFmpeg decode started: %dx%d @ %.1f fps (target=%.1f)",
             width_, height_, fps_, target_fps_.load(std::memory_order_relaxed));

    while (running_) {
        double seek_to = seek_request_sec_.exchange(-1.0, std::memory_order_acq_rel);
        if (seek_to >= 0.0) {
            AVStream* stream = fmt_ctx_->streams[video_stream_idx_];
            int64_t ts = av_rescale_q((int64_t)(seek_to * AV_TIME_BASE),
                                      AVRational{1, AV_TIME_BASE}, stream->time_base);
            if (stream->start_time != AV_NOPTS_VALUE) ts += stream->start_time;
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                while (!frame_queue_.empty()) {
                    av_frame_free(&frame_queue_.front().av_frame);
                    frame_queue_.pop();
                }
            }
            if (av_seek_frame(fmt_ctx_, video_stream_idx_, ts, AVSEEK_FLAG_BACKWARD) >= 0) {
                avcodec_flush_buffers(codec_ctx_);
                eof_ = false;
                position_sec_.store(seek_to, std::memory_order_relaxed);
                next_frame_time = std::chrono::steady_clock::now();
            }
            queue_not_full_.notify_all();
        }
        // 尝试从解码器获取已解码帧
        AVFrame* frame = av_frame_alloc();
        int ret = avcodec_receive_frame(codec_ctx_, frame);

        if (ret == 0) {
            // 成功获取帧
            if (frame->format == AV_PIX_FMT_CUDA) {
                DecodedFrame df;
                df.gpu.y_plane = frame->data[0];
                df.gpu.uv_plane = frame->data[1];
                df.gpu.y_pitch = frame->linesize[0];
                df.gpu.uv_pitch = frame->linesize[1];
                df.gpu.width = width_;
                df.gpu.height = height_;
                df.gpu.pts = frame->pts;
                if (frame->pts != AV_NOPTS_VALUE) {
                    AVStream* stream = fmt_ctx_->streams[video_stream_idx_];
                    double pos = (frame->pts - (stream->start_time == AV_NOPTS_VALUE ? 0 : stream->start_time)) *
                                 av_q2d(stream->time_base);
                    position_sec_.store(std::max(0.0, pos),
                                        std::memory_order_relaxed);
                }
                df.av_frame = frame;

                // 放入队列（满则阻塞等待）
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    queue_not_full_.wait(lock, [this]() {
                        return frame_queue_.size() < MAX_QUEUE || !running_;
                    });
                    if (!running_) {
                        av_frame_free(&frame);
                        break;
                    }
                    frame_queue_.push(df);
                }
                queue_not_empty_.notify_one();

                // 帧率控制（每帧动态检查 target_fps_，支持运行时切换）
                {
                    double cur_target = target_fps_.load(std::memory_order_relaxed);
                    // 0 = source FPS for natural playback; negative = unlimited benchmark mode.
                    bool unlimited = (cur_target < 0);
                    if (!unlimited) {
                        double effective_fps = cur_target > 0 ? cur_target : fps_;
                        if (effective_fps <= 0) effective_fps = 30.0;
                        auto frame_interval = std::chrono::microseconds(
                            (int64_t)(1000000.0 / effective_fps));
                        // target 变化时重置时间基准，避免跳帧
                        if (cur_target != last_target) {
                            next_frame_time = std::chrono::steady_clock::now();
                            last_target = cur_target;
                        }
                        next_frame_time += frame_interval;
                        std::this_thread::sleep_until(next_frame_time);
                        // 如果落后超过 2 帧，重置时间基准
                        auto now = std::chrono::steady_clock::now();
                        if (now - next_frame_time > frame_interval * 2) {
                            next_frame_time = now;
                        }
                    } else if (last_target >= 0) {
                        // 从有限帧率切换到无限，重置时间基准
                        last_target = cur_target;
                    }
                }
            } else {
                av_frame_free(&frame);
            }
            continue;
        }

        if (ret == AVERROR(EAGAIN)) {
            // 需要更多数据
            av_frame_free(&frame);
            ret = av_read_frame(fmt_ctx_, packet_);
            if (ret < 0) {
                if (ret == AVERROR_EOF) {
                    // 刷新解码器
                    avcodec_send_packet(codec_ctx_, nullptr);
                    while (true) {
                        AVFrame* flush_frame = av_frame_alloc();
                        ret = avcodec_receive_frame(codec_ctx_, flush_frame);
                        if (ret != 0) {
                            av_frame_free(&flush_frame);
                            break;
                        }
                        if (flush_frame->format == AV_PIX_FMT_CUDA) {
                            DecodedFrame df;
                            df.gpu.y_plane = flush_frame->data[0];
                            df.gpu.uv_plane = flush_frame->data[1];
                            df.gpu.y_pitch = flush_frame->linesize[0];
                            df.gpu.uv_pitch = flush_frame->linesize[1];
                            df.gpu.width = width_;
                            df.gpu.height = height_;
                            df.gpu.pts = flush_frame->pts;
                            df.av_frame = flush_frame;
                            {
                                std::unique_lock<std::mutex> lock(queue_mutex_);
                                queue_not_full_.wait(lock, [this]() {
                                    return frame_queue_.size() < MAX_QUEUE || !running_;
                                });
                                frame_queue_.push(df);
                            }
                            queue_not_empty_.notify_one();
                        } else {
                            av_frame_free(&flush_frame);
                        }
                    }
                }
                eof_ = true;
                queue_not_empty_.notify_one();
                av_frame_free(&frame);
                break;
            }
            if (packet_->stream_index == video_stream_idx_) {
                avcodec_send_packet(codec_ctx_, packet_);
            }
            av_packet_unref(packet_);
        } else {
            // 真正的错误
            LOG_ERROR("avcodec_receive_frame error: %d", ret);
            av_frame_free(&frame);
            break;
        }
    }

    LOG_INFO("FFmpeg hw decode thread finished");
}

void FFmpegHWDecoder::seekSeconds(double seconds) {
    if (!opened_ || duration_sec_ <= 0.0) return;
    seconds = std::max(0.0, std::min(duration_sec_, seconds));
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!frame_queue_.empty()) {
            av_frame_free(&frame_queue_.front().av_frame);
            frame_queue_.pop();
        }
    }
    seek_request_sec_.store(seconds, std::memory_order_release);
    queue_not_full_.notify_all();
    queue_not_empty_.notify_all();
}

bool FFmpegHWDecoder::getNextFrame(GpuFrame& frame) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_not_empty_.wait(lock, [this]() {
        return !frame_queue_.empty() || eof_ || !running_;
    });

    if (frame_queue_.empty()) return false;

    DecodedFrame df = frame_queue_.front();
    frame_queue_.pop();
    queue_not_full_.notify_one();

    // 释放上一帧
    if (current_frame_) {
        av_frame_free(&current_frame_);
    }

    frame = df.gpu;
    current_frame_ = df.av_frame;
    return true;
}

void FFmpegHWDecoder::releaseFrame() {
    if (current_frame_) {
        av_frame_free(&current_frame_);
        current_frame_ = nullptr;
    }
}

void FFmpegHWDecoder::stop() {
    running_ = false;
    queue_not_full_.notify_all();
    queue_not_empty_.notify_all();

    if (decode_thread_.joinable()) {
        decode_thread_.join();
    }

    // 清空队列
    std::lock_guard<std::mutex> lock(queue_mutex_);
    while (!frame_queue_.empty()) {
        av_frame_free(&frame_queue_.front().av_frame);
        frame_queue_.pop();
    }
    if (current_frame_) {
        av_frame_free(&current_frame_);
        current_frame_ = nullptr;
    }
}
