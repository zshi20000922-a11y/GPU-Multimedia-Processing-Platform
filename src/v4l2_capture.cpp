// V4L2 独立采集器
// 独立线程持续采集，帧队列缓冲，pipeline 非阻塞消费
// V4L2 DMA 缓冲区通过 mmap 映射，后续可扩展 DMA-BUF 导出

#include "v4l2_capture.h"
#include "logger.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <linux/videodev2.h>
#include <linux/usbdevice_fs.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <climits>
#include <cmath>
#include <chrono>

#include <cuda.h>

V4L2Capture::~V4L2Capture() {
    close();
}

CameraCapabilities V4L2Capture::queryCapabilities(int device_id) {
    CameraCapabilities caps;
    caps.device_path = "/dev/video" + std::to_string(device_id);

    int fd = ::open(caps.device_path.c_str(), O_RDWR | O_NONBLOCK);
    if (fd < 0) {
        LOG_ERROR("Cannot open camera: %s", caps.device_path.c_str());
        return caps;
    }

    // 查询设备信息
    struct v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) >= 0) {
        caps.card_name = (const char*)cap.card;
    }

    // 枚举像素格式
    struct v4l2_fmtdesc fmtdesc;
    memset(&fmtdesc, 0, sizeof(fmtdesc));
    fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    while (ioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc) >= 0) {
        uint32_t pixfmt = fmtdesc.pixelformat;
        std::string fmt_name = (const char*)fmtdesc.description;

        // 简化格式名
        char fourcc[5] = {
            (char)(pixfmt & 0xFF),
            (char)((pixfmt >> 8) & 0xFF),
            (char)((pixfmt >> 16) & 0xFF),
            (char)((pixfmt >> 24) & 0xFF), 0
        };

        // 枚举该格式支持的分辨率
        struct v4l2_frmsizeenum frmsize;
        memset(&frmsize, 0, sizeof(frmsize));
        frmsize.pixel_format = pixfmt;

        while (ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &frmsize) >= 0) {
            if (frmsize.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
                int w = frmsize.discrete.width;
                int h = frmsize.discrete.height;

                // 枚举该分辨率支持的帧率
                std::vector<float> fps_list;
                struct v4l2_frmivalenum frmival;
                memset(&frmival, 0, sizeof(frmival));
                frmival.pixel_format = pixfmt;
                frmival.width = w;
                frmival.height = h;

                while (ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &frmival) >= 0) {
                    if (frmival.type == V4L2_FRMIVAL_TYPE_DISCRETE) {
                        float fps = (float)frmival.discrete.denominator /
                                    frmival.discrete.numerator;
                        fps_list.push_back(fps);
                    }
                    frmival.index++;
                }

                if (fps_list.empty()) {
                    fps_list.push_back(30.0f);  // 默认
                }

                CameraFormat cfmt;
                cfmt.pixelformat = pixfmt;
                cfmt.name = std::string(fourcc);
                cfmt.width = w;
                cfmt.height = h;
                cfmt.fps_options = fps_list;
                caps.formats.push_back(cfmt);
            }
            frmsize.index++;
        }

        fmtdesc.index++;
    }

    ::close(fd);

    LOG_INFO("Camera %d: %s, %zu formats found",
             device_id, caps.card_name.c_str(), caps.formats.size());
    return caps;
}

std::vector<V4L2Capture::CameraInfo> V4L2Capture::enumerateCameras() {
    std::vector<CameraInfo> cameras;
    // 扫描 /dev/video0 ~ /dev/video31
    for (int i = 0; i < 32; i++) {
        std::string path = "/dev/video" + std::to_string(i);
        int fd = ::open(path.c_str(), O_RDWR | O_NONBLOCK);
        if (fd < 0) continue;

        struct v4l2_capability cap;
        if (ioctl(fd, VIDIOC_QUERYCAP, &cap) >= 0) {
            // 只保留视频采集设备（排除 metadata 等）
            if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) {
                CameraInfo info;
                info.device_id = i;
                info.device_path = path;
                info.card_name = (const char*)cap.card;
                cameras.push_back(info);
                LOG_INFO("Found camera: %s (%s)", path.c_str(), info.card_name.c_str());
            }
        }
        ::close(fd);
    }
    return cameras;
}

bool V4L2Capture::open(int device_id, int width, int height) {
    return open(device_id, width, height, V4L2_PIX_FMT_YUYV, 30.0f);
}

bool V4L2Capture::open(int device_id, int width, int height, uint32_t pixelformat, float fps) {
    std::string device = "/dev/video" + std::to_string(device_id);
    fd_ = ::open(device.c_str(), O_RDWR | O_NONBLOCK);
    if (fd_ < 0) {
        LOG_ERROR("Cannot open camera: %s", device.c_str());
        return false;
    }

    width_ = width;
    height_ = height;
    fps_ = fps;

    // 直接设置指定格式
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width_;
    fmt.fmt.pix.height = height_;
    fmt.fmt.pix.pixelformat = pixelformat;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        char fourcc[5] = {
            (char)(pixelformat & 0xFF), (char)((pixelformat >> 8) & 0xFF),
            (char)((pixelformat >> 16) & 0xFF), (char)((pixelformat >> 24) & 0xFF), 0
        };
        LOG_ERROR("VIDIOC_S_FMT failed for format %s", fourcc);
        close();
        return false;
    }

    width_ = fmt.fmt.pix.width;
    height_ = fmt.fmt.pix.height;

    // 读取驱动实际返回的格式（驱动可能静默替换）
    actual_pixelformat_ = fmt.fmt.pix.pixelformat;
    bytesperline_ = fmt.fmt.pix.bytesperline;
    is_yuyv_ = (actual_pixelformat_ == V4L2_PIX_FMT_YUYV);
    is_grey_ = (actual_pixelformat_ == V4L2_PIX_FMT_GREY);

    // 检测位深
    bit_shift_ = 0;  // 默认 8-bit
    switch (actual_pixelformat_) {
        case 0x30313259:  // Y210 (YUYV 10-bit)
        case 0x30313050:  // P010 (NV12 10-bit)
            bit_shift_ = 6;
            break;
        case 0x32313259:  // Y212 (YUYV 12-bit)
        case 0x32313050:  // P012 (NV12 12-bit)
            bit_shift_ = 4;
            break;
        default:
            bit_shift_ = 0;
            break;
    }

    // 如果 bytesperline 为 0，根据格式计算
    if (bytesperline_ == 0) {
        if (bit_shift_ > 0) {
            // 10/12-bit YUYV: 每像素 4 字节 (2x uint16_t)
            bytesperline_ = width_ * 4;
        } else if (is_yuyv_) {
            bytesperline_ = width_ * 2;
        } else if (is_grey_) {
            // GREY: 8-bit 单通道灰度，每像素 1 字节
            bytesperline_ = width_;
        }
    }

    char actual_fourcc[5] = {
        (char)(actual_pixelformat_ & 0xFF), (char)((actual_pixelformat_ >> 8) & 0xFF),
        (char)((actual_pixelformat_ >> 16) & 0xFF), (char)((actual_pixelformat_ >> 24) & 0xFF), 0
    };
    LOG_INFO("V4L2: actual format=%s, bytesperline=%d, bit_shift=%d",
             actual_fourcc, bytesperline_, bit_shift_);

    // 请求 DMA 缓冲区
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
        LOG_ERROR("VIDIOC_REQBUFS failed");
        close();
        return false;
    }

    // mmap 映射缓冲区
    buffers_.resize(req.count);
    for (size_t i = 0; i < req.count; i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
            LOG_ERROR("VIDIOC_QUERYBUF failed");
            close();
            return false;
        }
        buffers_[i].length = buf.length;
        buffers_[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE,
                                 MAP_SHARED, fd_, buf.m.offset);
        if (buffers_[i].start == MAP_FAILED) {
            LOG_ERROR("mmap failed");
            close();
            return false;
        }
    }

    // 零拷贝优化
    if (initDMABUF()) {
        LOG_INFO("Zero-copy: DMA-BUF mode enabled");
    } else if (initCudaPinned()) {
        LOG_INFO("Zero-copy: cudaHostRegister mode enabled");
    } else {
        LOG_INFO("Zero-copy: fallback to memcpy mode");
    }

    // 设置帧率
    struct v4l2_streamparm parm;
    memset(&parm, 0, sizeof(parm));
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = (uint32_t)fps;
    if (ioctl(fd_, VIDIOC_S_PARM, &parm) >= 0) {
        fps_ = (float)parm.parm.capture.timeperframe.denominator /
               parm.parm.capture.timeperframe.numerator;
    }

    if (!startStreaming()) {
        close();
        return false;
    }

    // 初始化空闲缓冲区队列
    for (size_t i = 0; i < buffers_.size(); i++) {
        free_buffers_.push(i);
    }

    opened_ = true;
    running_ = true;
    capture_thread_ = std::thread(&V4L2Capture::captureLoop, this);

    char fourcc[5] = {
        (char)(pixelformat & 0xFF), (char)((pixelformat >> 8) & 0xFF),
        (char)((pixelformat >> 16) & 0xFF), (char)((pixelformat >> 24) & 0xFF), 0
    };
    LOG_INFO("V4L2Capture opened: %s, %dx%d @ %.0ffps, format=%s, %zu buffers",
             device.c_str(), width_, height_, fps_, fourcc, buffers_.size());
    return true;
}

bool V4L2Capture::initV4L2() {
    // 查询设备能力
    struct v4l2_capability cap;
    if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
        LOG_ERROR("VIDIOC_QUERYCAP failed");
        return false;
    }

    // 设置格式
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width_;
    fmt.fmt.pix.height = height_;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
            LOG_ERROR("VIDIOC_S_FMT failed");
            return false;
        }
        is_yuyv_ = false;
    } else {
        is_yuyv_ = true;
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

    // 请求 DMA 缓冲区
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
        LOG_ERROR("VIDIOC_REQBUFS failed");
        return false;
    }

    // mmap 映射缓冲区
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

    // 尝试零拷贝优化：先尝试 DMA-BUF，失败则尝试 cudaHostRegister
    if (initDMABUF()) {
        LOG_INFO("Zero-copy: DMA-BUF mode enabled");
    } else if (initCudaPinned()) {
        LOG_INFO("Zero-copy: cudaHostRegister mode enabled");
    } else {
        LOG_INFO("Zero-copy: fallback to memcpy mode");
    }

    return true;
}

bool V4L2Capture::startStreaming() {
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

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
        LOG_ERROR("VIDIOC_STREAMON failed");
        return false;
    }
    return true;
}

void V4L2Capture::stopStreaming() {
    if (fd_ < 0) return;
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(fd_, VIDIOC_STREAMOFF, &type);
}

// 独立采集线程
void V4L2Capture::captureLoop() {
    LOG_INFO("V4L2 capture thread started");
    auto last_frame_time = std::chrono::steady_clock::now();
    auto fps_window_start = last_frame_time;
    int fps_window_count = 0;
    int capture_count = 0;

    while (running_) {
        // 先归还已释放的缓冲区给驱动（确保 V4L2 有可用缓冲区）
        {
            std::unique_lock<std::mutex> lock(free_mutex_);
            while (!free_buffers_.empty()) {
                int idx = free_buffers_.front();
                free_buffers_.pop();
                struct v4l2_buffer ret_buf;
                memset(&ret_buf, 0, sizeof(ret_buf));
                ret_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                ret_buf.memory = V4L2_MEMORY_MMAP;
                ret_buf.index = idx;
                ioctl(fd_, VIDIOC_QBUF, &ret_buf);
            }
        }

        // 等待帧就绪（select 阻塞，不浪费 CPU）
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd_, &fds);
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 10000;  // 10ms 超时（低延迟）

        int r = select(fd_ + 1, &fds, NULL, NULL, &tv);
        if (r <= 0) continue;  // 超时或错误

        // DQBUF：从驱动获取已填充的缓冲区
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
            if (errno == EAGAIN) continue;
            LOG_ERROR("VIDIOC_DQBUF failed");
            break;
        }

        // 帧间隔统计
        auto now = std::chrono::steady_clock::now();
        double interval_ms = std::chrono::duration<double, std::milli>(now - last_frame_time).count();
        last_frame_time = now;
        capture_count++;

        // 实测采集帧率：每秒用累计帧数更新一次（比单帧间隔稳定）
        {
            double since_fps = std::chrono::duration<double>(now - fps_window_start).count();
            fps_window_count++;
            if (since_fps >= 1.0) {
                measured_fps_.store((float)(fps_window_count / since_fps));
                fps_window_count = 0;
                fps_window_start = now;
            }
        }

        // 构造帧信息
        V4L2Frame frame;
        frame.data = (uint8_t*)buffers_[buf.index].start;
        frame.width = width_;
        frame.height = height_;
        frame.size = buf.bytesused;
        frame.buf_index = buf.index;
        frame.timestamp_us = buf.timestamp.tv_sec * 1000000ULL + buf.timestamp.tv_usec;
        frame.bytesperline = bytesperline_;
        frame.bit_shift = bit_shift_;

        // packed-gray 重解释：某些 4K 灰度相机（如 GMAX3412/FX20）无法协商
        // GREY 4K 模式，于是把 3840×2160 灰度图逐字节塞进 3840×1080 的 YUYV
        // 管道传输（字节恒等式 3840×1080×2 ≡ 3840×2160×1）。这里按灰度重新
        // 解释整块缓冲：宽不变、高×2、stride=宽，走 Y-only 路径。
        if (packed_gray_) {
            frame.width = width_;
            frame.height = height_ * 2;
            frame.bytesperline = width_;
        }

        // 入队列（满则丢弃最旧帧）
        size_t qsize = 0;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            qsize = frame_queue_.size();
            if (qsize >= MAX_QUEUE) {
                // 丢弃最旧帧，归还其缓冲区
                V4L2Frame old = frame_queue_.front();
                frame_queue_.pop();
                // 归还旧帧的缓冲区
                struct v4l2_buffer ret_buf;
                memset(&ret_buf, 0, sizeof(ret_buf));
                ret_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                ret_buf.memory = V4L2_MEMORY_MMAP;
                ret_buf.index = old.buf_index;
                ioctl(fd_, VIDIOC_QBUF, &ret_buf);
            }
            frame_queue_.push(frame);
        }
        queue_not_empty_.notify_one();

        if (capture_count % 30 == 0) {
            LOG_DEBUG("V4L2 capture: %.1f fps, queue=%zu", 1000.0 / interval_ms, qsize);
        }
    }

    LOG_INFO("V4L2 capture thread finished");
}

bool V4L2Capture::getNextFrame(V4L2Frame& frame, int timeout_ms) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    if (frame_queue_.empty()) {
        queue_not_empty_.wait_for(lock, std::chrono::milliseconds(timeout_ms));
        if (frame_queue_.empty()) return false;
    }
    frame = frame_queue_.front();
    frame_queue_.pop();
    queue_not_full_.notify_one();
    return true;
}

void V4L2Capture::releaseFrame(int buf_index) {
    std::unique_lock<std::mutex> lock(free_mutex_);
    free_buffers_.push(buf_index);
}

void V4L2Capture::stop() {
    running_ = false;
    queue_not_empty_.notify_all();
    queue_not_full_.notify_all();

    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }

    // 清空队列，归还所有缓冲区
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        while (!frame_queue_.empty()) frame_queue_.pop();
    }

    stopStreaming();

    // 清理零拷贝资源（必须在 munmap 之前）
    cleanupDMABUF();
    cleanupCudaPinned();

    // unmap 缓冲区
    for (auto& buf : buffers_) {
        if (buf.start && buf.start != MAP_FAILED) {
            munmap(buf.start, buf.length);
        }
    }
    buffers_.clear();

    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    opened_ = false;
}

void V4L2Capture::close() {
    stop();
}

// ==================== USB 总线级硬复位 ====================

// 从某 sysfs 路径向上查找含 busnum/devnum 的 USB 设备目录，读取总线号/设备号
static bool findUsbBusDev(const std::string& start_path, int& busnum, int& devnum) {
    char real[PATH_MAX];
    if (!realpath(start_path.c_str(), real)) return false;
    std::string dir = real;
    while (!dir.empty() && dir != "/") {
        std::string bus_f = dir + "/busnum";
        std::string dev_f = dir + "/devnum";
        FILE* fb = fopen(bus_f.c_str(), "r");
        FILE* fd = fopen(dev_f.c_str(), "r");
        if (fb && fd) {
            int b = -1, d = -1;
            if (fscanf(fb, "%d", &b) == 1 && fscanf(fd, "%d", &d) == 1) {
                fclose(fb); fclose(fd);
                busnum = b; devnum = d;
                return true;
            }
        }
        if (fb) fclose(fb);
        if (fd) fclose(fd);
        // 上一级目录
        size_t slash = dir.find_last_of('/');
        if (slash == std::string::npos) break;
        dir = dir.substr(0, slash);
    }
    return false;
}

bool V4L2Capture::resetUsbDevice(int device_id) {
    // 1. /dev/videoN → sysfs device 链接
    std::string sys_dev = "/sys/class/video4linux/video" +
                          std::to_string(device_id) + "/device";

    int busnum = -1, devnum = -1;
    if (!findUsbBusDev(sys_dev, busnum, devnum)) {
        LOG_WARN("USB reset: cannot locate USB node for /dev/video%d", device_id);
        return false;
    }

    // 2. 拼出 USB 设备节点路径
    char usb_path[64];
    snprintf(usb_path, sizeof(usb_path), "/dev/bus/usb/%03d/%03d", busnum, devnum);

    int fd = ::open(usb_path, O_WRONLY);
    if (fd < 0) {
        LOG_WARN("USB reset: open %s failed (%s) — 需要对 /dev/bus/usb 的写权限",
                 usb_path, strerror(errno));
        return false;
    }

    // 3. USBDEVFS_RESET ioctl（等效物理拔插）
    int rc = ioctl(fd, USBDEVFS_RESET, 0);
    ::close(fd);

    if (rc < 0) {
        LOG_WARN("USB reset: ioctl on %s failed (%s)", usb_path, strerror(errno));
        return false;
    }

    LOG_INFO("USB reset OK: /dev/video%d → %s (bus %d dev %d)",
             device_id, usb_path, busnum, devnum);
    return true;
}

// ==================== 零拷贝: DMA-BUF + CUDA External Memory ====================

bool V4L2Capture::initDMABUF() {
    dma_mappings_.resize(buffers_.size());

    for (size_t i = 0; i < buffers_.size(); i++) {
        dma_mappings_[i].fd = -1;
        dma_mappings_[i].ext_mem = nullptr;
        dma_mappings_[i].dev_ptr = nullptr;
        dma_mappings_[i].size = 0;
    }

    // 初始化 CUDA driver API
    CUresult cures = cuInit(0);
    if (cures != CUDA_SUCCESS) {
        LOG_WARN("cuInit failed: %d", cures);
        return false;
    }

    // 检查设备是否支持 DMA-BUF
    CUdevice dev;
    cuCtxGetDevice(&dev);
    int dmabuf_supported = 0;
    cuDeviceGetAttribute(&dmabuf_supported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev);
    if (!dmabuf_supported) {
        LOG_WARN("CUDA device does not support DMA-BUF");
        return false;
    }
    LOG_INFO("DMA-BUF: device supports DMA-BUF sharing");

    for (size_t i = 0; i < buffers_.size(); i++) {
        // 1. 将 V4L2 缓冲区导出为 DMA-BUF 文件描述符
        struct v4l2_exportbuffer expbuf;
        memset(&expbuf, 0, sizeof(expbuf));
        expbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        expbuf.index = i;
        expbuf.flags = O_CLOEXEC;

        if (ioctl(fd_, VIDIOC_EXPBUF, &expbuf) < 0) {
            LOG_WARN("VIDIOC_EXPBUF failed for buffer %zu (errno=%d: %s), DMA-BUF not available",
                     i, errno, strerror(errno));
            cleanupDMABUF();
            return false;
        }
        dma_mappings_[i].fd = expbuf.fd;
        LOG_INFO("DMA-BUF: buffer %zu exported, fd=%d, mmap_length=%zu",
                 i, expbuf.fd, buffers_[i].length);

        // 2. 使用 CUDA driver API 导入 DMA-BUF 为外部内存
        CUDA_EXTERNAL_MEMORY_HANDLE_DESC cu_desc;
        memset(&cu_desc, 0, sizeof(cu_desc));
        cu_desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        cu_desc.handle.fd = expbuf.fd;
        cu_desc.size = buffers_[i].length;

        cures = cuImportExternalMemory(&dma_mappings_[i].cu_ext_mem, &cu_desc);
        if (cures != CUDA_SUCCESS) {
            LOG_WARN("cuImportExternalMemory failed for buffer %zu (fd=%d, size=%zu): %d",
                     i, expbuf.fd, buffers_[i].length, (int)cures);
            cleanupDMABUF();
            return false;
        }
        LOG_INFO("DMA-BUF: buffer %zu imported via driver API", i);

        // 3. 映射为 CUDA 设备指针
        CUDA_EXTERNAL_MEMORY_BUFFER_DESC buf_desc;
        memset(&buf_desc, 0, sizeof(buf_desc));
        buf_desc.offset = 0;
        buf_desc.size = buffers_[i].length;

        cures = cuExternalMemoryGetMappedBuffer((CUdeviceptr*)&dma_mappings_[i].dev_ptr,
                                                 dma_mappings_[i].cu_ext_mem, &buf_desc);
        if (cures != CUDA_SUCCESS) {
            LOG_WARN("cuExternalMemoryGetMappedBuffer failed for buffer %zu: %d",
                     i, (int)cures);
            cleanupDMABUF();
            return false;
        }

        dma_mappings_[i].size = buffers_[i].length;
        LOG_INFO("DMA-BUF: buffer %zu mapped to GPU ptr=%p", i, dma_mappings_[i].dev_ptr);
    }

    zero_copy_mode_ = ZeroCopyMode::DMA_BUF;
    LOG_INFO("DMA-BUF zero-copy: %zu buffers exported and mapped to GPU", buffers_.size());
    return true;
}

void V4L2Capture::cleanupDMABUF() {
    for (auto& m : dma_mappings_) {
        if (m.dev_ptr) {
            cuMemFree((CUdeviceptr)m.dev_ptr);
            m.dev_ptr = nullptr;
        }
        if (m.cu_ext_mem) {
            cuDestroyExternalMemory(m.cu_ext_mem);
            m.cu_ext_mem = nullptr;
        }
        if (m.ext_mem) {
            cudaDestroyExternalMemory(m.ext_mem);
            m.ext_mem = nullptr;
        }
        if (m.fd >= 0) {
            ::close(m.fd);
            m.fd = -1;
        }
    }
    dma_mappings_.clear();
    if (zero_copy_mode_ == ZeroCopyMode::DMA_BUF) {
        zero_copy_mode_ = ZeroCopyMode::NONE;
    }
}

// ==================== 零拷贝: cudaHostRegister ====================

bool V4L2Capture::initCudaPinned() {
    for (size_t i = 0; i < buffers_.size(); i++) {
        cudaError_t err = cudaHostRegister(buffers_[i].start, buffers_[i].length,
                                           cudaHostRegisterDefault);
        if (err != cudaSuccess) {
            LOG_WARN("cudaHostRegister failed for buffer %zu: %s", i, cudaGetErrorString(err));
            // 回滚已注册的缓冲区
            for (size_t j = 0; j < i; j++) {
                cudaHostUnregister(buffers_[j].start);
            }
            return false;
        }
    }

    zero_copy_mode_ = ZeroCopyMode::CUDA_PINNED;
    LOG_INFO("cudaHostRegister: %zu buffers pinned for direct async DMA transfer", buffers_.size());
    return true;
}

void V4L2Capture::cleanupCudaPinned() {
    for (auto& buf : buffers_) {
        if (buf.start && buf.start != MAP_FAILED) {
            cudaHostUnregister(buf.start);
        }
    }
    if (zero_copy_mode_ == ZeroCopyMode::CUDA_PINNED) {
        zero_copy_mode_ = ZeroCopyMode::NONE;
    }
}

// ==================== DMA-BUF GPU 指针查询 ====================

uint8_t* V4L2Capture::getGpuPtr(int buf_index) const {
    if (zero_copy_mode_ != ZeroCopyMode::DMA_BUF ||
        buf_index < 0 || (size_t)buf_index >= dma_mappings_.size()) {
        return nullptr;
    }
    return dma_mappings_[buf_index].dev_ptr;
}
