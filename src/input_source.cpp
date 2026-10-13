#include "input_source.h"
#include "logger.h"

#ifdef __linux__
// ==================== V4L2 摄像头实现 ====================
bool CameraSource::open() {
    std::string device = "/dev/video" + std::to_string(device_id_);
    fd_ = ::open(device.c_str(), O_RDWR | O_NONBLOCK);
    if (fd_ < 0) {
        LOG_ERROR("Cannot open camera device: %s", device.c_str());
        return false;
    }

    if (!init_v4l2()) {
        release();
        return false;
    }

    if (!start_streaming()) {
        release();
        return false;
    }

    LOG_INFO("Camera %d opened via V4L2: %dx%d @ %.0ffps (zero-copy DMA)",
             device_id_, width_, height_, fps_);
    return true;
}

bool CameraSource::init_v4l2() {
    // 查询设备能力
    struct v4l2_capability cap;
    if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
        LOG_ERROR("VIDIOC_QUERYCAP failed");
        return false;
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        LOG_ERROR("Device does not support video capture");
        return false;
    }

    // 设置格式 (YUYV 或 MJPEG)
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = 3840;
    fmt.fmt.pix.height = 2160;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        // 尝试 MJPEG
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
            LOG_ERROR("VIDIOC_S_FMT failed");
            return false;
        }
    }

    width_ = fmt.fmt.pix.width;
    height_ = fmt.fmt.pix.height;

    // 设置帧率
    struct v4l2_streamparm parm;
    memset(&parm, 0, sizeof(parm));
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = 30;
    if (ioctl(fd_, VIDIOC_S_PARM, &parm) >= 0) {
        fps_ = (float)parm.parm.capture.timeperframe.denominator /
               parm.parm.capture.timeperframe.numerator;
    }

    // 请求缓冲区 (DMA 内存)
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
        LOG_ERROR("VIDIOC_REQBUFS failed");
        return false;
    }

    // 映射缓冲区
    buffers_.resize(req.count);
    for (size_t i = 0; i < req.count; i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
            LOG_ERROR("VIDIOC_QUERYBUF failed");
            return false;
        }

        buffers_[i].length = buf.length;
        buffers_[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE,
                                 MAP_SHARED, fd_, buf.m.offset);

        if (buffers_[i].start == MAP_FAILED) {
            LOG_ERROR("mmap failed");
            return false;
        }
    }

    return true;
}

bool CameraSource::start_streaming() {
    // 入队所有缓冲区
    for (size_t i = 0; i < buffers_.size(); i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
            LOG_ERROR("VIDIOC_QBUF failed");
            return false;
        }
    }

    // 开始流
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
        LOG_ERROR("VIDIOC_STREAMON failed");
        return false;
    }

    return true;
}

void CameraSource::stop_streaming() {
    if (fd_ < 0) return;
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(fd_, VIDIOC_STREAMOFF, &type);
}

bool CameraSource::read(cv::Mat& frame) {
    if (fd_ < 0) return false;

    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    // 从队列获取帧 (非阻塞)
    if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
        if (errno == EAGAIN) return false;
        LOG_ERROR("VIDIOC_DQBUF failed");
        return false;
    }

    // 转换 YUYV 到 BGR
    if (temp_frame_.empty()) {
        temp_frame_ = cv::Mat(height_, width_, CV_8UC2, buffers_[buf.index].start);
    } else {
        temp_frame_.data = (uint8_t*)buffers_[buf.index].start;
    }

    cv::cvtColor(temp_frame_, frame, cv::COLOR_YUV2BGR_YUYV);

    // 重新入队缓冲区
    if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
        LOG_ERROR("VIDIOC_QBUF re-queue failed");
        return false;
    }

    return true;
}

void CameraSource::release() {
    if (fd_ < 0) return;
    stop_streaming();
    for (auto& buf : buffers_) {
        if (buf.start && buf.start != MAP_FAILED) {
            munmap(buf.start, buf.length);
        }
    }
    buffers_.clear();
    ::close(fd_);
    fd_ = -1;
}

#else
// 非 Linux 平台后备实现
bool CameraSource::open() {
    cap_.open(device_id_);
    if (!cap_.isOpened()) {
        LOG_ERROR("Cannot open camera: %d", device_id_);
        return false;
    }
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, 3840);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 2160);
    cap_.set(cv::CAP_PROP_FPS, 30);
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    LOG_INFO("Camera %d opened: %dx%d @ %.0ffps",
             device_id_, width(), height(), fps());
    return true;
}

bool CameraSource::read(cv::Mat& frame) {
    return cap_.read(frame);
}

void CameraSource::release() {
    if (cap_.isOpened()) cap_.release();
}
#endif
