#include "video_recorder.h"
#include "logger.h"

bool VideoRecorder::start(const std::string& path, int width, int height, float fps) {
    if (recording_) {
        LOG_WARN("VideoRecorder: already recording");
        return false;
    }

    width_ = width;
    height_ = height;
    fps_ = fps;
    frames_written_ = 0;
    filepath_ = path;

    // 使用 MJPG 编码器，AVI 容器（兼容性最好）
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    writer_.open(path, fourcc, fps, cv::Size(width, height));
    if (!writer_.isOpened()) {
        LOG_ERROR("VideoRecorder: cannot open %s", path.c_str());
        return false;
    }

    recording_ = true;
    start_time_ = std::chrono::steady_clock::now();
    encode_thread_ = std::thread(&VideoRecorder::encodeLoop, this);

    LOG_INFO("VideoRecorder: started %s (%dx%d @ %.0f fps, limit=%.0fs)",
             path.c_str(), width, height, fps, duration_limit_);
    return true;
}

void VideoRecorder::stop() {
    if (!recording_) return;
    recording_ = false;
    queue_cv_.notify_all();

    if (encode_thread_.joinable()) {
        encode_thread_.join();
    }

    // Flush: 写入队列中剩余的帧
    std::lock_guard<std::mutex> lk(queue_mtx_);
    while (!frame_queue_.empty()) {
        writer_.write(frame_queue_.front());
        frame_queue_.pop();
        frames_written_++;
    }
    writer_.release();

    float elapsed = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - start_time_).count();
    LOG_INFO("VideoRecorder: stopped, %d frames written in %.1fs",
             frames_written_.load(), elapsed);
}

void VideoRecorder::pushFrame(const cv::Mat& bgr_frame) {
    if (!recording_) return;

    // 非阻塞入队
    std::lock_guard<std::mutex> lk(queue_mtx_);
    if ((int)frame_queue_.size() < MAX_QUEUE) {
        frame_queue_.push(bgr_frame.clone());
    }
    // 队列满则丢弃
    queue_cv_.notify_one();
}

float VideoRecorder::recordedSeconds() const {
    return std::chrono::duration<float>(
        std::chrono::steady_clock::now() - start_time_).count();
}

void VideoRecorder::encodeLoop() {
    while (recording_) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lk(queue_mtx_);
            queue_cv_.wait_for(lk, std::chrono::milliseconds(33), [this] {
                return !frame_queue_.empty() || !recording_;
            });
            if (!frame_queue_.empty()) {
                frame = frame_queue_.front();
                frame_queue_.pop();
            }
        }

        if (!frame.empty()) {
            writer_.write(frame);
            frames_written_++;
        }

        // 检查时长限制
        if (duration_limit_ > 0 && recordedSeconds() >= duration_limit_) {
            LOG_INFO("VideoRecorder: duration limit reached (%.0fs)", duration_limit_);
            recording_ = false;
            break;
        }
    }
}
