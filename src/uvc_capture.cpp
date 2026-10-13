// libuvc 摄像头采集实现
// 仅在 HAS_LIBUVC=1 时编译

#if HAS_LIBUVC

#include "uvc_capture.h"
#include "logger.h"

#include <libuvc/libuvc.h>
#include <cstring>
#include <chrono>
#include <algorithm>

// 静态成员初始化
uvc_context_t* UvcCapture::ctx_ = nullptr;
std::mutex UvcCapture::ctx_mtx_;
int UvcCapture::ctx_refcount_ = 0;

bool UvcCapture::ensureContext() {
    std::lock_guard<std::mutex> lock(ctx_mtx_);
    if (ctx_) {
        ctx_refcount_++;
        return true;
    }
    uvc_error_t res = uvc_init(&ctx_, nullptr);
    if (res < 0) {
        LOG_ERROR("UVC: uvc_init failed: %s", uvc_strerror(res));
        ctx_ = nullptr;
        return false;
    }
    ctx_refcount_ = 1;
    LOG_INFO("UVC: libuvc context initialized");
    return true;
}

std::vector<UvcCapture::CameraInfo> UvcCapture::enumerateCameras() {
    std::vector<CameraInfo> cameras;

    if (!ensureContext()) return cameras;

    uvc_device_t** dev_list = nullptr;
    uvc_error_t res = uvc_get_device_list(ctx_, &dev_list);
    if (res < 0) {
        LOG_ERROR("UVC: uvc_get_device_list failed: %s", uvc_strerror(res));
        return cameras;
    }

    int idx = 0;
    for (uvc_device_t** dev = dev_list; *dev != nullptr; dev++) {
        uvc_device_descriptor_t* desc = nullptr;
        res = uvc_get_device_descriptor(*dev, &desc);
        if (res < 0) continue;

        CameraInfo info;
        info.device_id = idx;
        info.vendor_id = desc->idVendor;
        info.product_id = desc->idProduct;

        // 构建设备路径
        char path[64];
        snprintf(path, sizeof(path), "USB %04x:%04x", desc->idVendor, desc->idProduct);
        info.device_path = path;

        // 设备名称
        if (desc->product) {
            info.card_name = desc->product;
        } else {
            char name[64];
            snprintf(name, sizeof(name), "UVC Device %04x:%04x", desc->idVendor, desc->idProduct);
            info.card_name = name;
        }

        cameras.push_back(info);
        uvc_free_device_descriptor(desc);
        idx++;
    }

    uvc_free_device_list(dev_list, 1);
    LOG_INFO("UVC: found %zu devices", cameras.size());
    return cameras;
}

bool UvcCapture::open(int device_id, int width, int height, uint32_t format, float fps) {
    if (opened_) {
        LOG_WARN("UVC: already opened");
        return false;
    }

    if (!ensureContext()) return false;

    // 枚举设备找到目标
    uvc_device_t** dev_list = nullptr;
    uvc_error_t res = uvc_get_device_list(ctx_, &dev_list);
    if (res < 0) {
        LOG_ERROR("UVC: uvc_get_device_list failed: %s", uvc_strerror(res));
        return false;
    }

    uvc_device_t* target_dev = nullptr;
    int idx = 0;
    for (uvc_device_t** dev = dev_list; *dev != nullptr; dev++) {
        if (idx == device_id) {
            target_dev = *dev;
            break;
        }
        idx++;
    }

    if (!target_dev) {
        LOG_ERROR("UVC: device %d not found", device_id);
        uvc_free_device_list(dev_list, 1);
        return false;
    }

    // 打开设备
    res = uvc_open(target_dev, &dev_handle_);
    if (res < 0) {
        LOG_ERROR("UVC: uvc_open failed: %s", uvc_strerror(res));
        uvc_free_device_list(dev_list, 1);
        return false;
    }

    // ===== 自动检测最佳格式 =====
    // 枚举设备支持的所有格式/分辨率/帧率
    const uvc_format_desc_t* fmt_descs = uvc_get_format_descs(dev_handle_);

    // 收集所有可用的 (format, width, height, fps) 组合
    struct Candidate {
        enum uvc_frame_format format;
        int width, height;
        float fps;
        int priority;  // YUYV=3, MJPEG=2, 其他=1
    };
    std::vector<Candidate> candidates;

    for (const uvc_format_desc_t* fmt = fmt_descs; fmt != nullptr; fmt = fmt->next) {
        // 确定格式类型和优先级
        enum uvc_frame_format uvc_fmt = UVC_FRAME_FORMAT_ANY;
        int priority = 1;
        const char* fmt_name = "Unknown";

        if (fmt->bDescriptorSubtype == UVC_VS_FORMAT_UNCOMPRESSED) {
            // 非压缩格式，检查 FourCC
            if (fmt->fourccFormat[0] == 'Y' && fmt->fourccFormat[1] == 'U' &&
                fmt->fourccFormat[2] == 'Y' && fmt->fourccFormat[3] == 'V') {
                uvc_fmt = UVC_FRAME_FORMAT_YUYV;
                priority = 3;  // YUYV 优先（GPU 零拷贝兼容）
                fmt_name = "YUYV";
            } else if (fmt->fourccFormat[0] == 'N' && fmt->fourccFormat[1] == 'V' &&
                       fmt->fourccFormat[2] == '1' && fmt->fourccFormat[3] == '2') {
                uvc_fmt = UVC_FRAME_FORMAT_NV12;
                priority = 2;
                fmt_name = "NV12";
            } else {
                uvc_fmt = UVC_FRAME_FORMAT_UNCOMPRESSED;
                priority = 1;
                fmt_name = "Uncompressed";
            }
        } else if (fmt->bDescriptorSubtype == UVC_VS_FORMAT_MJPEG) {
            uvc_fmt = UVC_FRAME_FORMAT_MJPEG;
            priority = 2;
            fmt_name = "MJPEG";
        } else {
            continue;  // 跳过不支持的格式类型
        }

        // 遍历该格式下的所有分辨率
        for (const uvc_frame_desc_t* frame = fmt->frame_descs; frame != nullptr; frame = frame->next) {
            int w = frame->wWidth;
            int h = frame->wHeight;

            // 解析可用帧率
            if (frame->bFrameIntervalType == 0) {
                // 连续范围：使用最大帧率
                // dwMinFrameInterval 和 dwMaxFrameInterval 是 100ns 单位
                if (frame->dwMaxFrameInterval > 0) {
                    float max_fps = 10000000.0f / frame->dwMaxFrameInterval;
                    candidates.push_back({uvc_fmt, w, h, max_fps, priority});
                }
            } else {
                // 离散帧率列表（100ns 单位）
                for (int i = 0; i < frame->bFrameIntervalType; i++) {
                    if (frame->intervals[i] == 0) break;
                    float f = 10000000.0f / frame->intervals[i];
                    candidates.push_back({uvc_fmt, w, h, f, priority});
                }
            }
        }

        if (!candidates.empty()) {
            LOG_INFO("UVC: format %s - found %zu candidates so far", fmt_name, candidates.size());
        }
    }

    if (candidates.empty()) {
        LOG_ERROR("UVC: device %d has no supported formats", device_id);
        uvc_close(dev_handle_);
        dev_handle_ = nullptr;
        uvc_free_device_list(dev_list, 1);
        return false;
    }

    // 选择最佳候选
    // 策略：优先匹配请求的分辨率，然后按格式优先级和帧率排序
    // 如果请求的分辨率不可用，选择最大分辨率
    auto best = candidates.begin();

    if (width > 0 && height > 0) {
        // 用户指定了分辨率：找精确匹配
        auto exact = std::find_if(candidates.begin(), candidates.end(),
            [&](const Candidate& c) { return c.width == width && c.height == height; });

        if (exact != candidates.end()) {
            // 精确匹配找到，在同分辨率中按优先级+帧率排序
            best = exact;
            for (auto it = candidates.begin(); it != candidates.end(); ++it) {
                if (it->width == width && it->height == height) {
                    if (it->priority > best->priority ||
                        (it->priority == best->priority && it->fps > best->fps)) {
                        best = it;
                    }
                }
            }
        } else {
            // 精确匹配失败，选择最接近的较大分辨率
            best = std::min_element(candidates.begin(), candidates.end(),
                [&](const Candidate& a, const Candidate& b) {
                    int da = std::abs(a.width - width) + std::abs(a.height - height);
                    int db = std::abs(b.width - width) + std::abs(b.height - height);
                    if (da != db) return da < db;
                    return a.priority > b.priority;
                });
            LOG_WARN("UVC: requested %dx%d not found, using %dx%d",
                     width, height, best->width, best->height);
        }
    } else {
        // 用户未指定分辨率：选择最大分辨率（按像素数），同分辨率按优先级
        best = std::max_element(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) {
                int pa = a.width * a.height;
                int pb = b.width * b.height;
                if (pa != pb) return pa < pb;
                if (a.priority != b.priority) return a.priority < b.priority;
                return a.fps < b.fps;
            });
    }

    // 用选中的格式协商流控
    int sel_w = best->width;
    int sel_h = best->height;
    int sel_fps = (fps > 0 && best->fps >= fps) ? (int)fps : (int)best->fps;

    res = uvc_get_stream_ctrl_format_size(dev_handle_, &ctrl_,
                                           best->format, sel_w, sel_h, sel_fps);
    if (res < 0) {
        // 最后回退：用 ANY 格式不限分辨率
        LOG_WARN("UVC: negotiated %dx%d@%d failed (%s), trying ANY format",
                 sel_w, sel_h, sel_fps, uvc_strerror(res));
        res = uvc_get_stream_ctrl_format_size(dev_handle_, &ctrl_,
                                               UVC_FRAME_FORMAT_ANY, 0, 0, 0);
        if (res < 0) {
            LOG_ERROR("UVC: cannot open any stream: %s", uvc_strerror(res));
            uvc_close(dev_handle_);
            dev_handle_ = nullptr;
            uvc_free_device_list(dev_list, 1);
            return false;
        }
    }

    // 开始采集
    res = uvc_start_streaming(dev_handle_, &ctrl_, streamCallback, this, 0);
    if (res < 0) {
        LOG_ERROR("UVC: uvc_start_streaming failed: %s", uvc_strerror(res));
        uvc_close(dev_handle_);
        dev_handle_ = nullptr;
        uvc_free_device_list(dev_list, 1);
        return false;
    }

    // 读取实际协商的分辨率（流控可能调整了值）
    width_ = sel_w;
    height_ = sel_h;
    fps_ = (float)sel_fps;
    opened_ = true;

    uvc_free_device_list(dev_list, 1);

    LOG_INFO("UVC: device %d opened (%dx%d @ %.0f fps, format=%d, priority=%d)",
             device_id, width_, height_, fps_, (int)best->format, best->priority);
    return true;
}

std::vector<UvcCapture::FormatCap> UvcCapture::queryCapabilities(int device_id) {
    std::vector<FormatCap> caps;

    if (!ensureContext()) return caps;

    uvc_device_t** dev_list = nullptr;
    uvc_error_t res = uvc_get_device_list(ctx_, &dev_list);
    if (res < 0) return caps;

    uvc_device_t* target_dev = nullptr;
    int idx = 0;
    for (uvc_device_t** dev = dev_list; *dev != nullptr; dev++) {
        if (idx == device_id) { target_dev = *dev; break; }
        idx++;
    }

    if (!target_dev) {
        uvc_free_device_list(dev_list, 1);
        return caps;
    }

    uvc_device_handle_t* handle = nullptr;
    res = uvc_open(target_dev, &handle);
    if (res < 0) {
        uvc_free_device_list(dev_list, 1);
        return caps;
    }

    const uvc_format_desc_t* fmt_descs = uvc_get_format_descs(handle);
    for (const uvc_format_desc_t* fmt = fmt_descs; fmt != nullptr; fmt = fmt->next) {
        const char* fmt_name = "Unknown";
        enum uvc_frame_format uvc_fmt = UVC_FRAME_FORMAT_ANY;

        if (fmt->bDescriptorSubtype == UVC_VS_FORMAT_UNCOMPRESSED) {
            if (fmt->fourccFormat[0] == 'Y' && fmt->fourccFormat[1] == 'U' &&
                fmt->fourccFormat[2] == 'Y' && fmt->fourccFormat[3] == 'V') {
                uvc_fmt = UVC_FRAME_FORMAT_YUYV; fmt_name = "YUYV";
            } else {
                uvc_fmt = UVC_FRAME_FORMAT_UNCOMPRESSED; fmt_name = "Uncompressed";
            }
        } else if (fmt->bDescriptorSubtype == UVC_VS_FORMAT_MJPEG) {
            uvc_fmt = UVC_FRAME_FORMAT_MJPEG; fmt_name = "MJPEG";
        } else {
            continue;
        }

        for (const uvc_frame_desc_t* frame = fmt->frame_descs; frame != nullptr; frame = frame->next) {
            float max_fps = 0;
            if (frame->bFrameIntervalType == 0) {
                if (frame->dwMaxFrameInterval > 0)
                    max_fps = 10000000.0f / frame->dwMaxFrameInterval;
            } else {
                for (int i = 0; i < frame->bFrameIntervalType; i++) {
                    if (frame->intervals[i] == 0) break;
                    float f = 10000000.0f / frame->intervals[i];
                    if (f > max_fps) max_fps = f;
                }
            }

            FormatCap cap;
            cap.width = frame->wWidth;
            cap.height = frame->wHeight;
            cap.fps = max_fps;
            cap.format = uvc_fmt;
            cap.format_name = fmt_name;
            caps.push_back(cap);
        }
    }

    uvc_close(handle);
    uvc_free_device_list(dev_list, 1);
    return caps;
}

void UvcCapture::streamCallback(uvc_frame_t* frame, void* user_ptr) {
    auto* self = static_cast<UvcCapture*>(user_ptr);
    if (!self || !frame || !frame->data) return;

    // 转换为 YUYV 或 BGR
    UvcFrame uvc_frame;
    uvc_frame.width = frame->width;
    uvc_frame.height = frame->height;

    if (frame->frame_format == UVC_FRAME_FORMAT_YUYV) {
        // YUYV 直通
        uvc_frame.size = frame->data_bytes;
        uvc_frame.data = new uint8_t[uvc_frame.size];
        memcpy(uvc_frame.data, frame->data, uvc_frame.size);
        uvc_frame.is_bgr = false;
    } else if (frame->frame_format == UVC_FRAME_FORMAT_MJPEG) {
        // MJPEG → BGR（通过 libuvc 转换）
        uvc_frame_t* bgr_frame = uvc_allocate_frame(frame->width * frame->height * 3);
        if (bgr_frame) {
            uvc_error_t res = uvc_mjpeg2rgb(frame, bgr_frame);
            if (res == UVC_SUCCESS) {
                uvc_frame.size = bgr_frame->data_bytes;
                uvc_frame.data = new uint8_t[uvc_frame.size];
                memcpy(uvc_frame.data, bgr_frame->data, uvc_frame.size);
                uvc_frame.is_bgr = true;
            } else {
                LOG_WARN("UVC: mjpeg2rgb failed: %s", uvc_strerror(res));
                uvc_free_frame(bgr_frame);
                return;
            }
            uvc_free_frame(bgr_frame);
        } else {
            return;
        }
    } else {
        // 其他格式 → RGB（通过 libuvc 转换）
        uvc_frame_t* rgb_frame = uvc_allocate_frame(frame->width * frame->height * 3);
        if (rgb_frame) {
            uvc_error_t res = uvc_any2rgb(frame, rgb_frame);
            if (res == UVC_SUCCESS) {
                // RGB → BGR
                uvc_frame.size = rgb_frame->data_bytes;
                uvc_frame.data = new uint8_t[uvc_frame.size];
                memcpy(uvc_frame.data, rgb_frame->data, uvc_frame.size);
                // RGB → BGR 转换
                for (size_t i = 0; i < uvc_frame.size; i += 3) {
                    std::swap(uvc_frame.data[i], uvc_frame.data[i + 2]);
                }
                uvc_frame.is_bgr = true;
            } else {
                LOG_WARN("UVC: any2rgb failed: %s", uvc_strerror(res));
                uvc_free_frame(rgb_frame);
                return;
            }
            uvc_free_frame(rgb_frame);
        } else {
            return;
        }
    }

    // 入队（丢弃最旧帧如果满）
    {
        std::lock_guard<std::mutex> lock(self->queue_mtx_);
        if ((int)self->frame_queue_.size() >= MAX_QUEUE) {
            auto& old = self->frame_queue_.front();
            delete[] old.data;
            self->frame_queue_.pop();
        }
        self->frame_queue_.push(uvc_frame);
    }
    self->queue_cv_.notify_one();
}

bool UvcCapture::getNextFrame(UvcFrame& frame, int timeout_ms) {
    if (!opened_) return false;

    std::unique_lock<std::mutex> lock(queue_mtx_);
    if (!queue_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                            [this] { return !frame_queue_.empty(); })) {
        return false;
    }

    // 释放上一帧
    if (has_current_) {
        delete[] current_frame_.data;
        current_frame_.data = nullptr;
        has_current_ = false;
    }

    current_frame_ = frame_queue_.front();
    frame_queue_.pop();
    has_current_ = true;

    frame = current_frame_;
    return true;
}

void UvcCapture::releaseFrame() {
    if (has_current_) {
        delete[] current_frame_.data;
        current_frame_.data = nullptr;
        has_current_ = false;
    }
}

void UvcCapture::stop() {
    if (!opened_) return;
    opened_ = false;

    if (dev_handle_) {
        uvc_stop_streaming(dev_handle_);
    }
    if (stream_handle_) {
        stream_handle_ = nullptr;
    }
    if (dev_handle_) {
        uvc_close(dev_handle_);
        dev_handle_ = nullptr;
    }

    // 清空队列
    {
        std::lock_guard<std::mutex> lock(queue_mtx_);
        while (!frame_queue_.empty()) {
            delete[] frame_queue_.front().data;
            frame_queue_.pop();
        }
    }

    // 释放当前帧
    releaseFrame();

    // 释放全局上下文
    {
        std::lock_guard<std::mutex> lock(ctx_mtx_);
        ctx_refcount_--;
        if (ctx_refcount_ <= 0) {
            if (ctx_) {
                uvc_exit(ctx_);
                ctx_ = nullptr;
            }
            ctx_refcount_ = 0;
        }
    }

    LOG_INFO("UVC: stopped");
}

#endif // HAS_LIBUVC
