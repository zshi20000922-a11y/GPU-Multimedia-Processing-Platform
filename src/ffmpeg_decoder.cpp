#include "ffmpeg_decoder.h"
#include "logger.h"

FFmpegDecoder::FFmpegDecoder() = default;

FFmpegDecoder::~FFmpegDecoder() { release(); }

bool FFmpegDecoder::open(const std::string& path) {
    path_ = path;

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
        release();
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
        release();
        return false;
    }

    // 获取视频流参数
    AVStream* video_stream = format_ctx_->streams[video_stream_idx_];
    AVCodecParameters* codec_params = video_stream->codecpar;

    // 查找解码器
    const AVCodec* codec = avcodec_find_decoder(codec_params->codec_id);
    if (!codec) {
        LOG_ERROR("Could not find decoder for codec %d", codec_params->codec_id);
        release();
        return false;
    }

    // 创建解码器上下文
    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
        LOG_ERROR("Could not allocate codec context");
        release();
        return false;
    }

    // 复制解码器参数
    ret = avcodec_parameters_to_context(codec_ctx_, codec_params);
    if (ret < 0) {
        LOG_ERROR("Could not copy codec parameters");
        release();
        return false;
    }

    // 打开解码器
    ret = avcodec_open2(codec_ctx_, codec, nullptr);
    if (ret < 0) {
        LOG_ERROR("Could not open codec");
        release();
        return false;
    }

    // 获取视频信息
    width_ = codec_ctx_->width;
    height_ = codec_ctx_->height;

    // 计算帧率
    if (video_stream->avg_frame_rate.den > 0) {
        fps_ = (float)video_stream->avg_frame_rate.num / video_stream->avg_frame_rate.den;
    } else if (video_stream->r_frame_rate.den > 0) {
        fps_ = (float)video_stream->r_frame_rate.num / video_stream->r_frame_rate.den;
    } else {
        fps_ = 30.0f;  // 默认帧率
    }

    // 分配帧和包
    frame_ = av_frame_alloc();
    packet_ = av_packet_alloc();
    if (!frame_ || !packet_) {
        LOG_ERROR("Could not allocate frame or packet");
        release();
        return false;
    }

    is_open_ = true;

    // 启动解码线程
    stop_decode_ = false;
    decode_thread_ = std::thread(&FFmpegDecoder::decode_thread_func, this);

    LOG_INFO("FFmpeg decoder opened: %s (%dx%d @ %.0ffps, codec=%s)",
             path.c_str(), width_, height_, fps_, codec->name);
    return true;
}

void FFmpegDecoder::decode_thread_func() {
    LOG_INFO("FFmpeg decode thread started");

    while (!stop_decode_) {
        // 读取一个包
        int ret = av_read_frame(format_ctx_, packet_);

        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                // 文件结束，循环播放
                av_seek_frame(format_ctx_, video_stream_idx_, 0, AVSEEK_FLAG_BACKWARD);
                avcodec_flush_buffers(codec_ctx_);
                continue;
            }
            LOG_ERROR("Error reading frame: %d", ret);
            break;
        }

        // 只处理视频流
        if (packet_->stream_index != video_stream_idx_) {
            av_packet_unref(packet_);
            continue;
        }

        // 发送包到解码器
        ret = avcodec_send_packet(codec_ctx_, packet_);
        av_packet_unref(packet_);

        if (ret < 0) {
            LOG_ERROR("Error sending packet to decoder: %d", ret);
            continue;
        }

        // 接收解码后的帧
        while (ret >= 0) {
            ret = avcodec_receive_frame(codec_ctx_, frame_);

            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            }

            if (ret < 0) {
                LOG_ERROR("Error receiving frame from decoder: %d", ret);
                break;
            }

            // 转换 YUV 到 BGR
            cv::Mat yuv(height_ * 3 / 2, width_, CV_8UC1);
            memcpy(yuv.data, frame_->data[0], width_ * height_);  // Y
            memcpy(yuv.data + width_ * height_, frame_->data[1], width_ * height_ / 4);  // U
            memcpy(yuv.data + width_ * height_ * 5 / 4, frame_->data[2], width_ * height_ / 4);  // V

            cv::Mat bgr;
            cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_I420);

            // 放入队列
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                if (frame_queue_.size() < MAX_QUEUE_SIZE) {
                    DecodedFrame df;
                    df.frame = bgr.clone();
                    df.ready = true;
                    frame_queue_.push(df);
                }
            }
            queue_cv_.notify_one();

            frame_count_++;

            // 释放帧引用
            av_frame_unref(frame_);
        }
    }

    LOG_INFO("FFmpeg decode thread finished, decoded %d frames", frame_count_);
}

bool FFmpegDecoder::read(cv::Mat& frame) {
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

void FFmpegDecoder::release() {
    stop_decode_ = true;
    queue_cv_.notify_all();

    if (decode_thread_.joinable()) {
        decode_thread_.join();
    }

    if (frame_) {
        av_frame_free(&frame_);
        frame_ = nullptr;
    }

    if (packet_) {
        av_packet_free(&packet_);
        packet_ = nullptr;
    }

    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
        codec_ctx_ = nullptr;
    }

    if (format_ctx_) {
        avformat_close_input(&format_ctx_);
        format_ctx_ = nullptr;
    }

    is_open_ = false;
}
