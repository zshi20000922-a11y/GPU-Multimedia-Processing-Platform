#include "pipeline.h"
#include "logger.h"

#include <opencv2/opencv.hpp>
#include <chrono>
#include <algorithm>
#include <cstring>

// COCO 类名（与 detection_engine.cpp 一致）
static const char* COCO_NAMES[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
};

// 绘制检测框
static void draw_detections_inplace(cv::Mat& frame, const std::vector<Detection>& dets) {
    static const cv::Scalar colors[] = {
        {255,56,56}, {255,157,151}, {255,112,31}, {255,178,29}, {207,210,49},
        {72,249,10}, {146,204,23}, {61,219,134}, {26,147,52}, {0,212,187},
        {44,153,168}, {0,194,255}, {52,69,147}, {100,115,255}, {0,24,236},
    };

    for (const auto& d : dets) {
        cv::Scalar color = colors[d.class_id % 15];
        cv::rectangle(frame, cv::Point((int)d.x1, (int)d.y1),
                      cv::Point((int)d.x2, (int)d.y2), color, 2);

        char label[64];
        snprintf(label, sizeof(label), "%s %.0f%%", d.class_name.c_str(), d.conf * 100);

        int baseline = 0;
        auto tsz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
        cv::rectangle(frame,
                      cv::Point((int)d.x1, (int)d.y1 - tsz.height - 4),
                      cv::Point((int)d.x1 + tsz.width + 4, (int)d.y1),
                      color, -1);
        cv::putText(frame, label,
                    cv::Point((int)d.x1 + 2, (int)d.y1 - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 0, 0}, 1, cv::LINE_AA);
    }
}

// GPU NV12 → CPU BGR 转换（用于 GUI 显示）
// 先从 GPU 拷贝 NV12 到 CPU，再做色彩转换
static cv::Mat gpu_nv12_to_bgr(const uint8_t* y_plane_gpu, const uint8_t* uv_plane_gpu,
                                int y_pitch, int uv_pitch, int width, int height) {
    // 分配 CPU 缓冲区
    size_t y_size = (size_t)y_pitch * height;
    size_t uv_size = (size_t)uv_pitch * (height / 2);
    std::vector<uint8_t> nv12_cpu(y_size + uv_size);

    // 从 GPU 拷贝到 CPU
    cudaMemcpy(nv12_cpu.data(), y_plane_gpu, y_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(nv12_cpu.data() + y_size, uv_plane_gpu, uv_size, cudaMemcpyDeviceToHost);

    // 转换 NV12 → BGR
    cv::Mat nv12(height * 3 / 2, width, CV_8UC1);
    if (y_pitch == width) {
        // 无 padding，直接转换
        cv::Mat nv12_wrap(height * 3 / 2, width, CV_8UC1, nv12_cpu.data());
        cv::cvtColor(nv12_wrap, nv12, cv::COLOR_YUV2BGR_NV12);
        return nv12;
    } else {
        // 有 padding，去除
        for (int y = 0; y < height; y++) {
            memcpy(nv12.ptr<uint8_t>(y), nv12_cpu.data() + y * y_pitch, width);
        }
        for (int y = 0; y < height / 2; y++) {
            memcpy(nv12.ptr<uint8_t>(height + y),
                   nv12_cpu.data() + y_size + y * uv_pitch, width);
        }
        cv::Mat bgr;
        cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
        return bgr;
    }
}

// ==================== Pipeline 初始化 ====================

void Pipeline::init_with_engine(int id, const InputConfig& cfg, const std::string& engine_path) {
    id_ = id;
    config_ = cfg;

    // 加载主引擎
    engine_ = std::make_unique<DetectionEngine>();
    if (!engine_->load(engine_path)) {
        LOG_ERROR("Pipeline %d: failed to load engine %s", id, engine_path.c_str());
        engine_.reset();
        return;
    }
    trt_context_ = engine_->create_context();
    if (!trt_context_) {
        LOG_ERROR("Pipeline %d: failed to create TRT context", id);
        return;
    }
    model_input_w_ = engine_->input_width();
    model_input_h_ = engine_->input_height();
    model_output_elements_ = engine_->output_elements();
    model_output_predictions_ = engine_->output_predictions();
    model_output_stride_ = engine_->output_stride();
    model_end2end_ = engine_->is_end2end();

    // SAHI 引擎策略：动态 batch 时复用主引擎，否则尝试加载独立 SAHI 引擎
    if (model_end2end_) {
        // YOLO26 export already contains end-to-end NMS and this application
        // exposes DIRECT mode for it; do not probe a nonexistent _batch8 file.
        sahi_engine_supports_dynamic_ = false;
        sahi_use_engine_ = nullptr;
        sahi_use_context_ = nullptr;
        sahi_batch_size_ = 1;
    } else if (engine_->supports_dynamic_batch() &&
        engine_->input_type() == nvinfer1::DataType::kFLOAT &&
        engine_->output_type() == nvinfer1::DataType::kFLOAT &&
        model_input_w_ == 640 && model_input_h_ == 640) {
        // 动态 batch：同一引擎同时用于 DIRECT (N=1) 和 SAHI (N=tile_count)
        sahi_engine_supports_dynamic_ = true;
        sahi_use_engine_ = engine_.get();
        sahi_use_context_ = trt_context_;
        sahi_batch_size_ = engine_->max_batch_size();
        LOG_INFO("Pipeline %d: dynamic batch engine, max_batch=%d, will reuse for SAHI",
                 id, sahi_batch_size_);
    } else {
        // 静态 batch：尝试加载独立 SAHI 引擎
        std::string sahi_engine_path = engine_path;
        auto pos = sahi_engine_path.find("batch1");
        if (pos != std::string::npos) {
            sahi_engine_path.replace(pos, 6, "batch8");
        } else {
            auto dot = sahi_engine_path.rfind('.');
            if (dot != std::string::npos) sahi_engine_path.insert(dot, "_batch8");
        }
        sahi_engine_ = std::make_unique<DetectionEngine>();
        if (sahi_engine_->load(sahi_engine_path)) {
            sahi_trt_context_ = sahi_engine_->create_context();
            if (sahi_trt_context_) {
                sahi_use_engine_ = sahi_engine_.get();
                sahi_use_context_ = sahi_trt_context_;
                sahi_batch_size_ = sahi_engine_->max_batch_size();
                LOG_INFO("Pipeline %d: SAHI engine loaded: %s (batch=%d)",
                         id, sahi_engine_path.c_str(), sahi_batch_size_);
            }
        } else {
            LOG_WARN("Pipeline %d: no dynamic batch, no SAHI engine found. SAHI will use single-tile fallback.",
                     id);
            sahi_engine_.reset();
            sahi_use_engine_ = nullptr;
            sahi_use_context_ = nullptr;
        }
    }

    // 分配 Ring Buffer
    if (!ring_buffer_.allocate(model_input_w_, model_input_h_, model_output_elements_,
                               engine_->input_type() == nvinfer1::DataType::kHALF,
                               engine_->output_type() == nvinfer1::DataType::kHALF)) {
        LOG_ERROR("Pipeline %d: failed to allocate ring buffer", id);
        return;
    }

    // 创建 4 个 CUDA stream
    cudaError_t e1 = cudaStreamCreate(&decode_stream_);
    cudaError_t e2 = cudaStreamCreate(&preproc_stream_);
    cudaError_t e3 = cudaStreamCreate(&infer_stream_);
    cudaError_t e4 = cudaStreamCreate(&postproc_stream_);
    LOG_INFO("Pipeline %d: streams created: %d/%d/%d/%d", id_, e1, e2, e3, e4);

    // SAHI 缓冲区延迟分配：首次使用 SAHI 模式时再分配
    // 避免不使用 SAHI 的 Pipeline 浪费 GPU 显存

    fps_start_ = std::chrono::steady_clock::now();
    LOG_INFO("Pipeline %d init: engine=%s, model=%dx%d output=%dx%d %s, dynamic_batch=%s, max_batch=%d, ring_buffer=%d slots",
             id, engine_path.c_str(),
             model_input_w_, model_input_h_, model_output_predictions_, model_output_stride_,
             model_end2end_ ? "YOLO26" : "YOLOv5",
             engine_->supports_dynamic_batch() ? "yes" : "no",
             sahi_batch_size_,
             ring_buffer_.numSlots());
}

void Pipeline::update_config(const InputConfig& cfg) {
    std::lock_guard<std::mutex> lk(config_mtx_);
    config_ = cfg;
    SAHIConfig sahi;
    sahi.slice_size = cfg.sahi_slice;
    sahi.stride = cfg.sahi_stride;
    slicer_.configure(sahi);
    // 更新解码帧率限制
    decoder_.setTargetFps(cfg.target_fps);
}

PipelineResult Pipeline::get_result() {
    std::lock_guard<std::mutex> lk(result_mtx_);
    return result_;
}

// ==================== Pipeline 启动/停止 ====================

void Pipeline::start() {
    if (running_) return;

    // 检查引擎和上下文是否就绪
    if (!engine_ || !engine_->is_loaded() || !trt_context_) {
        LOG_ERROR("Pipeline %d: engine/context not ready, cannot start", id_);
        return;
    }

    // 视频源：使用 FFmpeg CUDA hwaccel 解码器
    if (config_.type == InputType::VIDEO) {
        if (!decoder_.open(config_.source)) {
            LOG_ERROR("Pipeline %d: cannot open video %s", id_, config_.source.c_str());
            return;
        }
        // 设置目标帧率
        if (config_.target_fps > 0) {
            decoder_.setTargetFps(config_.target_fps);
        }
        LOG_INFO("Pipeline %d: video opened (%dx%d @ %.1f fps, target=%.1f fps)",
                 id_, decoder_.width(), decoder_.height(), decoder_.fps(),
                 config_.target_fps > 0 ? config_.target_fps : decoder_.fps());
    } else if (config_.type == InputType::CAMERA) {
        int cam_id = 0;
        try {
            cam_id = config_.source.empty() ? 0 : std::stoi(config_.source);
        } catch (...) {
            LOG_ERROR("Pipeline %d: invalid camera ID '%s'", id_, config_.source.c_str());
            return;
        }
        // packed-gray：相机以 YUYV 管道搬运灰度，传输分辨率为 W×(H/2)，
        // 协商后按 W×H 单通道灰度重解释
        int open_w = config_.cam_width;
        int open_h = config_.cam_packed_gray ? config_.cam_height / 2 : config_.cam_height;
        if (!v4l2_.open(cam_id, open_w, open_h,
                        config_.cam_format, config_.cam_fps)) {
            LOG_ERROR("Pipeline %d: cannot open camera %d", id_, cam_id);
            return;
        }
        if (config_.cam_packed_gray) {
            v4l2_.setPackedGray(true);
            LOG_INFO("Pipeline %d: packed-gray enabled (YUYV %dx%d -> grey %dx%d)",
                     id_, open_w, open_h, v4l2_.width(), v4l2_.height());
        }
        LOG_INFO("Pipeline %d: camera %d opened via V4L2 (%dx%d @ %.0f fps)",
                 id_, cam_id, v4l2_.width(), v4l2_.height(), v4l2_.fps());
    }
#if HAS_LIBUVC
    else if (config_.type == InputType::UVC) {
        int cam_id = 0;
        try {
            cam_id = config_.source.empty() ? 0 : std::stoi(config_.source);
        } catch (...) {
            LOG_ERROR("Pipeline %d: invalid UVC device ID '%s'", id_, config_.source.c_str());
            return;
        }
        if (!uvc_.open(cam_id, config_.cam_width, config_.cam_height,
                        config_.cam_format, config_.cam_fps)) {
            LOG_ERROR("Pipeline %d: cannot open UVC device %d", id_, cam_id);
            return;
        }
        LOG_INFO("Pipeline %d: UVC device %d opened (%dx%d @ %.0f fps)",
                 id_, cam_id, uvc_.width(), uvc_.height(), uvc_.fps());
    }
#endif

    running_ = true;
    frame_count_ = 0;
    fps_start_ = std::chrono::steady_clock::now();
    worker_ = std::thread(&Pipeline::worker_thread, this);

    LOG_INFO("Pipeline %d started", id_);
}

void Pipeline::stop() {
    if (!running_) return;
    running_ = false;

    decoder_.stop();
    v4l2_.stop();
#if HAS_LIBUVC
    uvc_.stop();
#endif

    if (worker_.joinable()) {
        worker_.join();
    }

    // 清空结果
    {
        std::lock_guard<std::mutex> lk(result_mtx_);
        result_ = PipelineResult();
    }

    // 释放 CUDA stream
    if (decode_stream_) { cudaStreamDestroy(decode_stream_); decode_stream_ = nullptr; }
    if (preproc_stream_) { cudaStreamDestroy(preproc_stream_); preproc_stream_ = nullptr; }
    if (infer_stream_) { cudaStreamDestroy(infer_stream_); infer_stream_ = nullptr; }
    if (postproc_stream_) { cudaStreamDestroy(postproc_stream_); postproc_stream_ = nullptr; }

    // 释放 TRT context
    if (trt_context_) { delete trt_context_; trt_context_ = nullptr; }
    if (sahi_trt_context_) { delete sahi_trt_context_; sahi_trt_context_ = nullptr; }
    sahi_engine_.reset();

    ring_buffer_.release();

    // 释放摄像头传输缓冲区
    if (cam_gpu_buf_) { cudaFree(cam_gpu_buf_); cam_gpu_buf_ = nullptr; }
    cam_gpu_buf_size_ = 0;
    if (cam_pinned_buf_) { cudaFreeHost(cam_pinned_buf_); cam_pinned_buf_ = nullptr; }
    if (cam_gpu_fallback_buf_) { cudaFree(cam_gpu_fallback_buf_); cam_gpu_fallback_buf_ = nullptr; }
    cam_fallback_buf_size_ = 0;

    // 释放 SAHI 批量推理缓冲区
    if (sahi_batch_input_) { cudaFree(sahi_batch_input_); sahi_batch_input_ = nullptr; }
    if (sahi_batch_output_) { cudaFree(sahi_batch_output_); sahi_batch_output_ = nullptr; }
    if (sahi_host_output_) { cudaFreeHost(sahi_host_output_); sahi_host_output_ = nullptr; }
    sahi_batch_input_bytes_ = 0;
    sahi_batch_output_bytes_ = 0;

    // 释放 GPU 显示缓冲区
    if (display_gpu_buf_) { cudaFree(display_gpu_buf_); display_gpu_buf_ = nullptr; }
    display_gpu_buf_size_ = 0;

    // 停止硬件录制
    hw_recorder_.stop();
    if (rec_nv12_buf_) { cudaFreeHost(rec_nv12_buf_); rec_nv12_buf_ = nullptr; }
    rec_nv12_size_ = 0;

    LOG_INFO("Pipeline %d stopped", id_);
}

// ==================== CPU 后处理 ====================

std::vector<Detection> Pipeline::cpu_postprocess(const float* output, int num_preds,
                                                  float conf_thresh, float nms_thresh,
                                                  int orig_w, int orig_h, int input_size,
                                                  int tile_offset_x, int tile_offset_y) {
    // YOLO26 export=True returns TensorRT end-to-end NMS rows:
    // [x1, y1, x2, y2, confidence, class_id].
    if (model_end2end_) {
        float scale_x = std::min((float)model_input_w_ / orig_w,
                                 (float)model_input_h_ / orig_h);
        int nw = (int)(orig_w * scale_x), nh = (int)(orig_h * scale_x);
        float px = (model_input_w_ - nw) / 2.0f;
        float py = (model_input_h_ - nh) / 2.0f;
        std::vector<Detection> candidates;
        candidates.reserve(num_preds);
        for (int i = 0; i < num_preds; ++i) {
            const float* p = output + (size_t)i * model_output_stride_;
            if (p[4] < conf_thresh) continue;
            const int class_id = std::max(0, (int)p[5]);
            if (class_id >= engine_->num_classes()) continue;
            // The presentation camera uses the official COCO model but only
            // publishes people. The one-class ARD100 video model is unaffected.
            if (config_.type == InputType::CAMERA && engine_->num_classes() > 1 &&
                class_id != 0) continue;
            Detection d;
            d.x1 = (p[0] - px) / scale_x + tile_offset_x;
            d.y1 = (p[1] - py) / scale_x + tile_offset_y;
            d.x2 = (p[2] - px) / scale_x + tile_offset_x;
            d.y2 = (p[3] - py) / scale_x + tile_offset_y;
            d.x1 = std::clamp(d.x1, (float)tile_offset_x, (float)(orig_w + tile_offset_x));
            d.y1 = std::clamp(d.y1, (float)tile_offset_y, (float)(orig_h + tile_offset_y));
            d.x2 = std::clamp(d.x2, (float)tile_offset_x, (float)(orig_w + tile_offset_x));
            d.y2 = std::clamp(d.y2, (float)tile_offset_y, (float)(orig_h + tile_offset_y));
            d.conf = p[4];
            d.class_id = class_id;
            d.class_name = engine_->model_class_name(d.class_id);
            if (d.x2 > d.x1 && d.y2 > d.y1) candidates.push_back(std::move(d));
        }

        // Ultralytics end-to-end engines expose [N,300,6], but an engine may
        // still be exported with metadata args.nms=false. Apply a deterministic
        // class-aware NMS here for both the one-class Drone model and COCO
        // person model. This also makes the GUI NMS slider effective.
        std::sort(candidates.begin(), candidates.end(),
                  [](const Detection& a, const Detection& b) {
                      return a.conf > b.conf;
                  });
        std::vector<Detection> kept;
        kept.reserve(candidates.size());
        for (const auto& candidate : candidates) {
            bool suppressed = false;
            const float candidate_area =
                (candidate.x2 - candidate.x1) * (candidate.y2 - candidate.y1);
            for (const auto& selected : kept) {
                if (selected.class_id != candidate.class_id) continue;
                const float ix1 = std::max(candidate.x1, selected.x1);
                const float iy1 = std::max(candidate.y1, selected.y1);
                const float ix2 = std::min(candidate.x2, selected.x2);
                const float iy2 = std::min(candidate.y2, selected.y2);
                const float iw = std::max(0.0f, ix2 - ix1);
                const float ih = std::max(0.0f, iy2 - iy1);
                const float intersection = iw * ih;
                const float selected_area =
                    (selected.x2 - selected.x1) * (selected.y2 - selected.y1);
                const float iou = intersection /
                    std::max(1e-6f, candidate_area + selected_area - intersection);
                if (iou > nms_thresh) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) kept.push_back(candidate);
        }
        return kept;
    }

    // YOLOv5 raw output: confidence filtering + coordinate conversion + CPU NMS.
    float scale = std::min((float)input_size / orig_w, (float)input_size / orig_h);
    int nw = (int)(orig_w * scale), nh = (int)(orig_h * scale);
    float px = (input_size - nw) / 2.0f, py = (input_size - nh) / 2.0f;

    struct RawDet { float x1, y1, x2, y2, conf; int class_id; };
    std::vector<RawDet> raw_dets;
    const int classes = std::max(1, model_output_stride_ - 5);

    // 诊断：打印前3个通过置信度过滤的预测的原始值
    static int diag_count = 0;
    bool do_diag = (diag_count++ % 300 == 0);
    int diag_printed = 0;

    for (int i = 0; i < num_preds; i++) {
        const float* p = output + (size_t)i * model_output_stride_;
        float obj = p[4];
        if (obj < conf_thresh) continue;

        if (do_diag && diag_printed < 3) {
            LOG_INFO("  raw[%d]: cx=%.3f cy=%.3f w=%.3f h=%.3f obj=%.3f best_cls=%.3f",
                     i, p[0], p[1], p[2], p[3], p[4], p[5]);
            diag_printed++;
        }

        float best = 0;
        int bc = 0;
        for (int c = 0; c < classes; c++) {
            if (p[5 + c] > best) { best = p[5 + c]; bc = c; }
        }
        float score = obj * best;
        if (score < conf_thresh) continue;

        float cx = p[0], cy = p[1], w = p[2], h = p[3];
        float x1 = (cx - w * 0.5f - px) / scale + tile_offset_x;
        float y1 = (cy - h * 0.5f - py) / scale + tile_offset_y;
        float x2 = (cx + w * 0.5f - px) / scale + tile_offset_x;
        float y2 = (cy + h * 0.5f - py) / scale + tile_offset_y;

        x1 = std::max(0.0f, std::min((float)(orig_w + tile_offset_x), x1));
        y1 = std::max(0.0f, std::min((float)(orig_h + tile_offset_y), y1));
        x2 = std::max(0.0f, std::min((float)(orig_w + tile_offset_x), x2));
        y2 = std::max(0.0f, std::min((float)(orig_h + tile_offset_y), y2));

        raw_dets.push_back({x1, y1, x2, y2, score, bc});
    }

    // Step 2: NMS（按置信度排序，贪心抑制）
    std::sort(raw_dets.begin(), raw_dets.end(),
              [](const RawDet& a, const RawDet& b) { return a.conf > b.conf; });

    if (raw_dets.size() > 1000) {
        LOG_WARN("cpu_postprocess: %zu raw dets (conf_thresh too low?), truncating to top-1000",
                 raw_dets.size());
        raw_dets.resize(1000);
    }

    std::vector<bool> suppressed(raw_dets.size(), false);
    std::vector<Detection> result;

    for (size_t i = 0; i < raw_dets.size(); i++) {
        if (suppressed[i]) continue;
        Detection d;
        d.x1 = raw_dets[i].x1; d.y1 = raw_dets[i].y1;
        d.x2 = raw_dets[i].x2; d.y2 = raw_dets[i].y2;
        d.conf = raw_dets[i].conf;
        d.class_id = raw_dets[i].class_id;
            d.class_name = classes == 1 ? "Drone" : COCO_NAMES[d.class_id < 80 ? d.class_id : 0];
        result.push_back(d);

        for (size_t j = i + 1; j < raw_dets.size(); j++) {
            if (suppressed[j] || raw_dets[j].class_id != raw_dets[i].class_id) continue;
            float ix1 = std::max(raw_dets[i].x1, raw_dets[j].x1);
            float iy1 = std::max(raw_dets[i].y1, raw_dets[j].y1);
            float ix2 = std::min(raw_dets[i].x2, raw_dets[j].x2);
            float iy2 = std::min(raw_dets[i].y2, raw_dets[j].y2);
            float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
            float area_i = (raw_dets[i].x2 - raw_dets[i].x1) * (raw_dets[i].y2 - raw_dets[i].y1);
            float area_j = (raw_dets[j].x2 - raw_dets[j].x1) * (raw_dets[j].y2 - raw_dets[j].y1);
            if (inter / (area_i + area_j - inter + 1e-6f) > nms_thresh) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

// ==================== 4 级流水线工作线程 ====================

void Pipeline::worker_thread() {
    SAHIConfig sahi;
    sahi.slice_size = config_.sahi_slice;
    sahi.stride = config_.sahi_stride;
    slicer_.configure(sahi);

    if (!engine_ || !engine_->is_loaded() || !trt_context_) {
        LOG_ERROR("Pipeline %d: engine/context not ready", id_);
        return;
    }

    // 绑定 tensor 名称
    const char* input_name = engine_->engine()->getIOTensorName(0);
    const char* output_name = engine_->engine()->getIOTensorName(1);

    int slot_idx = 0;
    int64_t frame_num = 0;
    size_t output_bytes = (size_t)model_output_elements_ * sizeof(float);

    // 性能统计
    double sum_decode = 0, sum_preproc = 0, sum_infer = 0, sum_postproc = 0;
    auto total_start = std::chrono::high_resolution_clock::now();

    // 上一帧的检测结果（异步获取）
    std::vector<Detection> prev_dets;
    cv::Mat prev_frame;
    bool has_prev = false;

    LOG_INFO("Pipeline %d: 4-stream pipeline started, input_name=%s, output_name=%s",
             id_, input_name ? input_name : "null", output_name ? output_name : "null");

    while (running_) {
        FrameSlot& slot = ring_buffer_.slot(slot_idx);
        int acquired_v4l2_buffer = -1;

        LOG_DEBUG("Pipeline %d: frame %ld, slot %d", id_, frame_num, slot_idx);

        // 等待该 slot 的上一轮后处理完成（释放缓冲区）
        cudaError_t err = cudaStreamWaitEvent(decode_stream_, slot.postproc_done, 0);
        if (err != cudaSuccess) {
            LOG_ERROR("Pipeline %d: cudaStreamWaitEvent failed: %s", id_, cudaGetErrorString(err));
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        // ===== Stage 1: 解码 =====
        GpuFrame gpu_frame;
        bool got_frame = false;
        // The dequeued V4L2 buffer remains owned by this iteration until the
        // end of the loop.  Keep a non-owning CPU view so the GUI preview can
        // fall back to the camera's native bytes independently of CUDA/TRT.
        cv::Mat camera_cpu_view;
        bool camera_cpu_gray = false;

        if (config_.type == InputType::VIDEO && decoder_.isOpen()) {
            got_frame = decoder_.getNextFrame(gpu_frame);
            LOG_DEBUG("Pipeline %d: got_frame=%d, %dx%d", id_, got_frame, gpu_frame.width, gpu_frame.height);
        } else if (v4l2_.isOpen()) {
            // 摄像头：V4L2 独立线程采集 YUYV → GPU → YUYV letterbox → TRT
            V4L2Frame cam_frame;
            auto t_before_getframe = std::chrono::high_resolution_clock::now();
            if (v4l2_.getNextFrame(cam_frame, 100)) {
                auto t_after_getframe = std::chrono::high_resolution_clock::now();
                size_t data_size = cam_frame.size;

                if (v4l2_.isDMABUF()) {
                    // 方案A: 真零拷贝 — DMA-BUF 映射到 GPU 地址空间
                    slot.nv12_y = v4l2_.getGpuPtr(cam_frame.buf_index);
                } else if (v4l2_.isCudaPinned()) {
                    // 方案B: cudaHostRegister — 直接从 pinned mmap 异步传输
                    if (cam_gpu_buf_size_ < data_size) {
                        if (cam_gpu_buf_) cudaFree(cam_gpu_buf_);
                        cudaMalloc(&cam_gpu_buf_, data_size);
                        cam_gpu_buf_size_ = data_size;
                    }
                    cudaMemcpyAsync(cam_gpu_buf_, cam_frame.data, data_size,
                                    cudaMemcpyHostToDevice, decode_stream_);
                    slot.nv12_y = cam_gpu_buf_;
                } else {
                    // Fallback: 传统方式（memcpy + cudaMemcpyAsync）
                    if (cam_fallback_buf_size_ < data_size) {
                        if (cam_pinned_buf_) cudaFreeHost(cam_pinned_buf_);
                        if (cam_gpu_fallback_buf_) cudaFree(cam_gpu_fallback_buf_);
                        cudaMallocHost(&cam_pinned_buf_, data_size);
                        cudaMalloc(&cam_gpu_fallback_buf_, data_size);
                        cam_fallback_buf_size_ = data_size;
                    }
                    memcpy(cam_pinned_buf_, cam_frame.data, data_size);
                    cudaMemcpyAsync(cam_gpu_fallback_buf_, cam_pinned_buf_, data_size,
                                    cudaMemcpyHostToDevice, decode_stream_);
                    slot.nv12_y = cam_gpu_fallback_buf_;
                }

                auto t_memcpy_done = std::chrono::high_resolution_clock::now();
                double dt_getframe = std::chrono::duration<double, std::milli>(t_after_getframe - t_before_getframe).count();
                double dt_transfer = std::chrono::duration<double, std::milli>(t_memcpy_done - t_after_getframe).count();
                if (frame_num % 30 == 0) {
                    LOG_INFO("Pipeline %d: getNextFrame=%.2f ms, transfer=%.2f ms (mode=%s, size=%zu)",
                             id_, dt_getframe, dt_transfer,
                             v4l2_.isDMABUF() ? "DMA-BUF" :
                             v4l2_.isCudaPinned() ? "cudaPinned" : "fallback",
                             data_size);
                }

                // 填充 slot（所有模式共用）
                slot.nv12_uv = nullptr;        // 标记为非 NV12（YUYV 路径）
                slot.y_pitch = cam_frame.bytesperline > 0 ? cam_frame.bytesperline : cam_frame.width * 2;
                slot.uv_pitch = 0;
                slot.src_width = cam_frame.width;
                slot.src_height = cam_frame.height;
                slot.frame_idx = frame_num;
                slot.bit_shift = cam_frame.bit_shift;
                slot.pixel_format = v4l2_.isGrey() ? FramePixelFormat::GRAY8 :
                                    (cam_frame.bit_shift > 0 ? FramePixelFormat::YUYV16
                                                             : FramePixelFormat::YUYV);
                if (slot.pixel_format == FramePixelFormat::GRAY8) {
                    camera_cpu_view = cv::Mat(cam_frame.height, cam_frame.width,
                                              CV_8UC1, cam_frame.data,
                                              (size_t)slot.y_pitch);
                    camera_cpu_gray = true;
                } else if (slot.pixel_format == FramePixelFormat::YUYV) {
                    camera_cpu_view = cv::Mat(cam_frame.height, cam_frame.width,
                                              CV_8UC2, cam_frame.data,
                                              (size_t)slot.y_pitch);
                }
                got_frame = true;
                acquired_v4l2_buffer = cam_frame.buf_index;

                // CPU copy is only needed by recording/snapshot-related paths.
                // DIRECT/OFF display consumes the mapped GPU buffer directly.
                const bool need_cpu_camera_frame = hw_recorder_.isRecording() || recorder_.isRecording();
                if (need_cpu_camera_frame && v4l2_.isPackedGray()) {
                    // packed-gray：W×H 单通道灰度，stride=W
                    cam_yuyv_frame_ = cv::Mat(cam_frame.height, cam_frame.width,
                                              CV_8UC1, cam_frame.data,
                                              slot.y_pitch).clone();
                } else if (need_cpu_camera_frame && cam_frame.bit_shift > 0) {
                    // 10/12-bit: uint16_t per component, 4 bytes per pixel pair
                    cam_yuyv_frame_ = cv::Mat(cam_frame.height, cam_frame.width,
                                              CV_16UC2, cam_frame.data).clone();
                } else if (need_cpu_camera_frame) {
                    cam_yuyv_frame_ = cv::Mat(cam_frame.height, cam_frame.width,
                                              CV_8UC2, cam_frame.data).clone();
                }

                // Do not return this mmap/DMA-BUF to the driver yet. CUDA kernels
                // still consume it asynchronously; it is released at frame end.
            }
        }
#if HAS_LIBUVC
        else if (uvc_.isOpen()) {
            // libuvc 采集（无零拷贝，使用 fallback 路径）
            UvcFrame uvc_frame;
            if (uvc_.getNextFrame(uvc_frame, 100)) {
                size_t data_size = uvc_frame.size;

                if (cam_fallback_buf_size_ < data_size) {
                    if (cam_pinned_buf_) cudaFreeHost(cam_pinned_buf_);
                    if (cam_gpu_fallback_buf_) cudaFree(cam_gpu_fallback_buf_);
                    cudaMallocHost(&cam_pinned_buf_, data_size);
                    cudaMalloc(&cam_gpu_fallback_buf_, data_size);
                    cam_fallback_buf_size_ = data_size;
                }
                memcpy(cam_pinned_buf_, uvc_frame.data, data_size);
                cudaMemcpyAsync(cam_gpu_fallback_buf_, cam_pinned_buf_, data_size,
                                cudaMemcpyHostToDevice, decode_stream_);
                slot.nv12_y = cam_gpu_fallback_buf_;
                slot.nv12_uv = nullptr;        // YUYV 或 BGR 路径
                slot.y_pitch = uvc_frame.width * 2;  // YUYV stride
                slot.uv_pitch = 0;
                slot.src_width = uvc_frame.width;
                slot.src_height = uvc_frame.height;
                slot.frame_idx = frame_num;
                slot.pixel_format = uvc_frame.is_bgr ? FramePixelFormat::BGR : FramePixelFormat::YUYV;
                got_frame = true;

                // 保存帧用于显示
                if (uvc_frame.is_bgr) {
                    cam_yuyv_frame_ = cv::Mat(uvc_frame.height, uvc_frame.width,
                                              CV_8UC3, uvc_frame.data).clone();
                } else {
                    cam_yuyv_frame_ = cv::Mat(uvc_frame.height, uvc_frame.width,
                                              CV_8UC2, uvc_frame.data).clone();
                }

                uvc_.releaseFrame();
            }
        }
#endif

        if (!got_frame) {
            if (config_.type == InputType::VIDEO) {
                // 视频结束
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        // 视频路径：用 FFmpeg GPU 帧指针填充 slot
        // （摄像头路径已在上面填充完毕，不要覆盖）
        if (config_.type == InputType::VIDEO && decoder_.isOpen()) {
            slot.nv12_y = gpu_frame.y_plane;
            slot.nv12_uv = gpu_frame.uv_plane;
            slot.y_pitch = gpu_frame.y_pitch;
            slot.uv_pitch = gpu_frame.uv_pitch;
            slot.src_width = gpu_frame.width;
            slot.src_height = gpu_frame.height;
            slot.frame_idx = frame_num;
            slot.pixel_format = FramePixelFormat::NV12;
        }

        LOG_DEBUG("Pipeline %d: slot filled, %dx%d", id_, slot.src_width, slot.src_height);

        // Camera-only display fallback. Detection continues to consume the
        // original GPU upload; this function only builds the GUI preview from
        // the exact dequeued V4L2 bytes, preserving GREY versus YUYV layout.
        auto make_camera_preview = [&](int out_w, int out_h) -> cv::Mat {
            if (camera_cpu_view.empty()) return {};
            if (camera_cpu_gray) {
                // Resize the single channel first (far cheaper than converting
                // the full 4K frame to BGR). FX20 can legitimately deliver a
                // very dark 8-bit stream; apply display-only gain so a valid
                // low-exposure frame is not mistaken for a blank GUI texture.
                cv::Mat gray_small;
                cv::resize(camera_cpu_view, gray_small, cv::Size(out_w, out_h),
                           0, 0, cv::INTER_LINEAR);
                const double mean_luma = cv::mean(gray_small)[0];
                const double preview_gain = std::max(1.0, std::min(16.0,
                    90.0 / std::max(1.0, mean_luma)));
                if (preview_gain > 1.01) {
                    gray_small.convertTo(gray_small, CV_8UC1, preview_gain);
                }
                cv::Mat bgr;
                cv::cvtColor(gray_small, bgr, cv::COLOR_GRAY2BGR);
                if (frame_num == 0 || frame_num % 240 == 0) {
                    LOG_INFO("Pipeline %d: FX20 preview published (%dx%d, raw mean=%.2f, display gain=%.2fx)",
                             id_, out_w, out_h, mean_luma, preview_gain);
                }
                return bgr;
            }

            cv::Mat bgr_full;
            cv::cvtColor(camera_cpu_view, bgr_full, cv::COLOR_YUV2BGR_YUYV);
            cv::Mat resized;
            cv::resize(bgr_full, resized, cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);
            return resized;
        };

        cudaEventRecord(slot.decode_done, decode_stream_);
        auto t1 = std::chrono::high_resolution_clock::now();

        // 硬件录制：推送原始 4K 帧
        if (hw_recorder_.isRecording()) {
            RecordFrame rf;
            if (slot.pixel_format == FramePixelFormat::NV12) {
                // NV12 视频文件：GPU 帧在 releaseFrame 后失效，需要 D2H 拷贝
                cudaStreamSynchronize(decode_stream_);
                size_t y_size = (size_t)slot.y_pitch * slot.src_height;
                size_t uv_size = (size_t)slot.uv_pitch * (slot.src_height / 2);
                // 使用 pinned host buffer 缓存（复用 cam_pinned_buf_ 或动态分配）
                if (!rec_nv12_buf_ || rec_nv12_size_ < y_size + uv_size) {
                    if (rec_nv12_buf_) cudaFreeHost(rec_nv12_buf_);
                    cudaMallocHost(&rec_nv12_buf_, y_size + uv_size);
                    rec_nv12_size_ = y_size + uv_size;
                }
                cudaMemcpy(rec_nv12_buf_, slot.nv12_y, y_size, cudaMemcpyDeviceToHost);
                cudaMemcpy(rec_nv12_buf_ + y_size, slot.nv12_uv, uv_size, cudaMemcpyDeviceToHost);
                rf.y_plane = rec_nv12_buf_;
                rf.uv_plane = rec_nv12_buf_ + y_size;
                rf.y_pitch = slot.y_pitch;
                rf.uv_pitch = slot.uv_pitch;
                rf.width = slot.src_width;
                rf.height = slot.src_height;
                rf.is_nv12 = true;
            } else if (v4l2_.isPackedGray() || v4l2_.isGrey()) {
                // 灰度相机：克隆到独立缓冲区后传给编码器（避免编码线程与下一帧竞态）
                cv::Mat gray_copy;
                if (v4l2_.isPackedGray()) {
                    gray_copy = cam_yuyv_frame_.clone();
                } else {
                    cv::extractChannel(cam_yuyv_frame_, gray_copy, 0);
                }
                rf.is_gray = true;
                rf.gray_data = gray_copy.data;
                rf.gray_stride = (int)gray_copy.step;
                rf.width = gray_copy.cols;
                rf.height = gray_copy.rows;
                rf.is_nv12 = false;
                rf.owned_frame = std::move(gray_copy);
            } else {
                // 彩色 YUYV 摄像头：转 BGR 并持有所有权
                cv::Mat bgr_copy;
                cv::cvtColor(cam_yuyv_frame_, bgr_copy, cv::COLOR_YUV2BGR_YUYV);
                rf.bgr_data = bgr_copy.data;
                rf.bgr_stride = (int)bgr_copy.step;
                rf.width = bgr_copy.cols;
                rf.height = bgr_copy.rows;
                rf.is_nv12 = false;
                rf.owned_frame = std::move(bgr_copy);
            }
            hw_recorder_.pushFrame(rf);
        }
        auto t2 = t1, t3 = t1, t4 = t1, t5 = t1;  // 各阶段时间戳

        // ===== Stage 2-4: 检测（DIRECT / SAHI / OFF） =====
        DetectMode mode;
        float conf, nms;
        int sahi_slice, sahi_stride;
        {
            std::lock_guard<std::mutex> lk(config_mtx_);
            mode = config_.detect_mode;
            conf = config_.conf_thresh;
            nms = config_.nms_thresh;
            sahi_slice = config_.sahi_slice;
            sahi_stride = config_.sahi_stride;
            sahi_interval_ = std::max(1, config_.sahi_interval);
        }

        if (mode == DetectMode::SAHI) {
            // ==================== SAHI 模式（动态 batch 切片推理） ====================

            // 检查 SAHI 引擎是否可用
            if (model_end2end_ || model_input_w_ != 640 || model_input_h_ != 640 ||
                engine_->input_type() != nvinfer1::DataType::kFLOAT ||
                engine_->output_type() != nvinfer1::DataType::kFLOAT) {
                static bool warned_yolo26_sahi = false;
                if (!warned_yolo26_sahi) {
                    LOG_WARN("Pipeline %d: SAHI currently requires a YOLOv5-compatible 640 dynamic-batch engine; using DIRECT for this model", id_);
                    warned_yolo26_sahi = true;
                }
                mode = DetectMode::DIRECT;
            } else if (!sahi_use_context_ || !sahi_use_engine_) {
                LOG_WARN("Pipeline %d: SAHI engine not available, falling back to OFF", id_);
                std::lock_guard<std::mutex> lock(result_mtx_);
                result_.dets.clear();
                result_.num_objects = 0;
                result_.sahi_cached = false;
                mode = DetectMode::OFF;
            } else {

            // 帧跳过：非推理帧复用缓存检测结果
            sahi_counter_++;
            bool do_infer = (sahi_counter_ >= sahi_interval_);
            if (do_infer) sahi_counter_ = 0;

            std::vector<Detection> dets;

            if (do_infer) {
                // 延迟分配 SAHI 缓冲区
                if (!sahi_batch_input_) {
                    int alloc_batch = std::min(kSAHIMaxBatch, sahi_batch_size_);
                    sahi_batch_input_bytes_ = (size_t)alloc_batch * 3 * 640 * 640 * sizeof(float);
                    sahi_batch_output_bytes_ = (size_t)alloc_batch * 25200 * 85 * sizeof(float);
                    cudaMalloc(&sahi_batch_input_, sahi_batch_input_bytes_);
                    cudaMalloc(&sahi_batch_output_, sahi_batch_output_bytes_);
                    cudaMallocHost(&sahi_host_output_, sahi_batch_output_bytes_);
                    LOG_INFO("Pipeline %d: SAHI buffers allocated (batch=%d, input=%.1fMB, output=%.1fMB)",
                             id_, alloc_batch,
                             sahi_batch_input_bytes_ / 1048576.0f,
                             sahi_batch_output_bytes_ / 1048576.0f);
                }

                // 配置切片器
                SAHIConfig sahi_cfg;
                sahi_cfg.slice_size = sahi_slice;
                sahi_cfg.stride = sahi_stride;
                slicer_.configure(sahi_cfg);

                // 生成 tile 坐标
                cv::Mat dummy_frame(slot.src_height, slot.src_width, CV_8UC1);
                auto tiles = slicer_.generate_tiles(dummy_frame);
                int num_tiles = (int)tiles.size();

                if (num_tiles > 0) {
                    int max_batch = std::min(kSAHIMaxBatch, sahi_batch_size_);
                    int num_batches = (num_tiles + max_batch - 1) / max_batch;

                    // 获取 tensor 名称
                    const char* sahi_in_name = sahi_use_engine_->engine()->getIOTensorName(0);
                    const char* sahi_out_name = sahi_use_engine_->engine()->getIOTensorName(1);

                    // Stage 2: 逐 tile GPU 预处理
                    cudaStreamWaitEvent(preproc_stream_, slot.decode_done, 0);

                    for (int t = 0; t < num_tiles; t++) {
                        const auto& tile = tiles[t];
                        float* tile_dst = sahi_batch_input_ + (size_t)t * 3 * 640 * 640;

                        // SAHI tile 是 640×640，scale=1.0，pad=0（无需 letterbox 缩放）
                        if (slot.pixel_format == FramePixelFormat::NV12) {
                            // NV12 路径：使用分离 Y/UV 指针的 tile kernel
                            const uint8_t* tile_y = slot.nv12_y + tile.y * slot.y_pitch;
                            const uint8_t* tile_uv = slot.nv12_uv + (tile.y / 2) * slot.uv_pitch;
                            if (slot.bit_shift > 0) {
                                launch_nv12_16bit_letterbox(tile_y, tile_uv,
                                    tile.w, tile.h, slot.y_pitch, slot.uv_pitch,
                                    tile_dst, 640, 640, 1.0f, 0.0f, 0.0f, slot.bit_shift, preproc_stream_);
                            } else {
                                launch_nv12_tile_letterbox(tile_y, tile_uv,
                                    tile.w, tile.h, slot.y_pitch, slot.uv_pitch,
                                    tile_dst, 640, 640, 1.0f, 0.0f, 0.0f, preproc_stream_);
                            }
                        } else if (slot.pixel_format != FramePixelFormat::BGR) {
                            // YUYV 路径：偏移指针
                            const uint8_t* tile_ptr = slot.nv12_y + tile.y * slot.y_pitch + tile.x * 2;
                            if (slot.bit_shift > 0) {
                                launch_yuyv16_letterbox(tile_ptr, tile.w, tile.h, slot.y_pitch,
                                    tile_dst, 640, 640, 1.0f, 0.0f, 0.0f, slot.bit_shift, preproc_stream_);
                            } else {
                                launch_yuyv_letterbox(tile_ptr, tile.w, tile.h, slot.y_pitch,
                                    tile_dst, 640, 640, 1.0f, 0.0f, 0.0f, preproc_stream_,
                                    v4l2_.isGrey());
                            }
                        } else {
                            // BGR 路径（UVC）：偏移指针
                            const uint8_t* tile_ptr = slot.nv12_y + tile.y * slot.y_pitch + tile.x * 3;
                            launch_bgr_letterbox(tile_ptr, tile.w, tile.h, slot.y_pitch,
                                tile_dst, 640, 640, 1.0f, 0.0f, 0.0f, preproc_stream_);
                        }
                    }
                    cudaEventRecord(slot.preproc_done, preproc_stream_);
                    t2 = std::chrono::high_resolution_clock::now();

                    // Stage 3: 多 batch 推理
                    cudaStreamWaitEvent(infer_stream_, slot.preproc_done, 0);

                    for (int batch = 0; batch < num_batches; batch++) {
                        int start = batch * max_batch;
                        int count = std::min(max_batch, num_tiles - start);

                        float* batch_input = sahi_batch_input_ + (size_t)start * 3 * 640 * 640;
                        float* batch_output = sahi_batch_output_ + (size_t)start * 25200 * 85;

                        sahi_use_context_->setInputShape(sahi_in_name,
                            nvinfer1::Dims4{count, 3, 640, 640});
                        sahi_use_context_->setTensorAddress(sahi_in_name, batch_input);
                        sahi_use_context_->setTensorAddress(sahi_out_name, batch_output);
                        sahi_use_context_->enqueueV3(infer_stream_);
                    }

                    cudaEventRecord(slot.infer_done, infer_stream_);
                    t3 = std::chrono::high_resolution_clock::now();

                    // Stage 4: D2H + CPU 后处理
                    cudaStreamWaitEvent(postproc_stream_, slot.infer_done, 0);
                    size_t total_output_bytes = (size_t)num_tiles * 25200 * 85 * sizeof(float);
                    cudaMemcpyAsync(sahi_host_output_, sahi_batch_output_, total_output_bytes,
                                    cudaMemcpyDeviceToHost, postproc_stream_);
                    cudaEventRecord(slot.postproc_done, postproc_stream_);
                    cudaEventSynchronize(slot.postproc_done);
                    t4 = std::chrono::high_resolution_clock::now();

                    // 逐 tile CPU 后处理（tile 是 640×640，scale=1.0，pad=0）
                    std::vector<std::vector<Detection>> tile_dets(num_tiles);
                    for (int t = 0; t < num_tiles; t++) {
                        const float* tile_output = sahi_host_output_ + (size_t)t * 25200 * 85;
                        tile_dets[t] = cpu_postprocess(tile_output, 25200, conf, nms,
                                                       640, 640, 640,
                                                       tiles[t].x, tiles[t].y);
                    }

                    // 全局 NMS 合并
                    dets = slicer_.merge_detections(tile_dets, tiles, nms);
                }

                // 缓存检测结果
                sahi_cached_dets_ = dets;
            } else {
                // 非推理帧：复用缓存
                dets = sahi_cached_dets_;
            }

            // 显示帧生成（与 DIRECT/OFF 模式相同的 GPU 路径）
            float disp_scale = std::min((float)kDisplayMaxW / slot.src_width,
                                        (float)kDisplayMaxH / slot.src_height);
            int disp_w = (int)(slot.src_width * disp_scale);
            int disp_h = (int)(slot.src_height * disp_scale);
            disp_w &= ~1; disp_h &= ~1;
            cv::Mat stable_display;

            {
                size_t buf_size = (size_t)disp_w * disp_h * 3;
                if (display_gpu_buf_ == nullptr || display_gpu_buf_size_ < buf_size) {
                    if (display_gpu_buf_) cudaFree(display_gpu_buf_);
                    cudaMalloc(&display_gpu_buf_, buf_size);
                    display_gpu_buf_size_ = buf_size;
                }

                if (slot.pixel_format == FramePixelFormat::NV12) {
                    if (slot.bit_shift > 0) {
                        launch_nv12_16bit_resize_rgb(slot.nv12_y, slot.nv12_uv,
                                                     slot.src_width, slot.src_height,
                                                     slot.y_pitch, slot.uv_pitch,
                                                     display_gpu_buf_, disp_w, disp_h,
                                                     slot.bit_shift, postproc_stream_);
                    } else {
                        launch_nv12_resize_rgb(slot.nv12_y, slot.nv12_uv,
                                               slot.src_width, slot.src_height,
                                               slot.y_pitch, slot.uv_pitch,
                                               display_gpu_buf_, disp_w, disp_h,
                                               postproc_stream_);
                    }
                } else {
                    if (slot.bit_shift > 0) {
                        launch_yuyv16_resize_rgb(slot.nv12_y, slot.src_width, slot.src_height,
                                                 slot.y_pitch, display_gpu_buf_, disp_w, disp_h,
                                                 slot.bit_shift, postproc_stream_);
                    } else {
                        launch_yuyv_resize_rgb(slot.nv12_y, slot.src_width, slot.src_height,
                                               slot.y_pitch, display_gpu_buf_, disp_w, disp_h,
                                               postproc_stream_, v4l2_.isGrey());
                    }
                }
                cudaStreamSynchronize(postproc_stream_);
                cv::Mat rgb(disp_h, disp_w, CV_8UC3);
                cudaMemcpy(rgb.data, display_gpu_buf_, buf_size, cudaMemcpyDeviceToHost);
                cv::cvtColor(rgb, stable_display, cv::COLOR_RGB2BGR);
                if (!camera_cpu_view.empty()) {
                    stable_display = make_camera_preview(disp_w, disp_h);
                }
            }

            // 缩放检测坐标到显示尺寸
            if (disp_scale != 1.0f) {
                for (auto& d : dets) {
                    d.x1 *= disp_scale; d.y1 *= disp_scale;
                    d.x2 *= disp_scale; d.y2 *= disp_scale;
                }
            }

            // 录制
            if (recorder_.isRecording()) {
                cv::Mat record_frame = stable_display.clone();
                if (!dets.empty()) draw_detections_inplace(record_frame, dets);
                recorder_.pushFrame(record_frame);
            }

            {
                std::lock_guard<std::mutex> lk(result_mtx_);
                result_.display = stable_display;
                result_.display_gpu = nullptr;
                result_.display_w = disp_w;
                result_.display_h = disp_h;
                result_.src_w = slot.src_width;
                result_.src_h = slot.src_height;
                result_.dets = dets;
                result_.num_objects = (int)dets.size();
                result_.has_frame = true;
                result_.sahi_cached = !do_infer;
            }
            t5 = std::chrono::high_resolution_clock::now();

            } // end SAHI engine available check
        }

        if (mode == DetectMode::DIRECT) {
            // ==================== DIRECT 模式（4 级流水线） ====================
            cudaStreamWaitEvent(preproc_stream_, slot.decode_done, 0);

            float scale = std::min((float)model_input_w_ / slot.src_width,
                                   (float)model_input_h_ / slot.src_height);
            int new_w = (int)(slot.src_width * scale);
            int new_h = (int)(slot.src_height * scale);
            float pad_x = (model_input_w_ - new_w) / 2.0f;
            float pad_y = (model_input_h_ - new_h) / 2.0f;

            if (slot.pixel_format == FramePixelFormat::NV12) {
                // NV12 路径
                if (slot.bit_shift > 0) {
                    launch_nv12_16bit_letterbox(slot.nv12_y, slot.nv12_uv,
                                                slot.src_width, slot.src_height,
                                                slot.y_pitch, slot.uv_pitch,
                                                slot.input_buf, model_input_w_, model_input_h_,
                                                scale, pad_x, pad_y, slot.bit_shift, preproc_stream_);
                } else {
                    launch_nv12_tile_letterbox(slot.nv12_y, slot.nv12_uv,
                                               slot.src_width, slot.src_height,
                                               slot.y_pitch, slot.uv_pitch,
                                               slot.input_buf, model_input_w_, model_input_h_,
                                               scale, pad_x, pad_y, preproc_stream_);
                }
            } else if (slot.pixel_format != FramePixelFormat::BGR) {
                // YUYV 路径
                if (slot.bit_shift > 0) {
                    launch_yuyv16_letterbox(slot.nv12_y, slot.src_width, slot.src_height, slot.y_pitch,
                                            slot.input_buf, model_input_w_, model_input_h_,
                                            scale, pad_x, pad_y, slot.bit_shift, preproc_stream_);
                } else {
                    launch_yuyv_letterbox(slot.nv12_y, slot.src_width, slot.src_height, slot.y_pitch,
                                          slot.input_buf, model_input_w_, model_input_h_,
                                          scale, pad_x, pad_y, preproc_stream_, v4l2_.isGrey());
                }
            } else {
                // BGR 路径（UVC）
                launch_bgr_letterbox(slot.nv12_y, slot.src_width, slot.src_height, slot.y_pitch,
                                     slot.input_buf, model_input_w_, model_input_h_,
                                     scale, pad_x, pad_y, preproc_stream_);
            }
            if (slot.tensor_input_fp16) {
                launch_float_to_half(slot.input_buf, slot.tensor_input_buf,
                                     (size_t)3 * model_input_w_ * model_input_h_, preproc_stream_);
            }
            cudaEventRecord(slot.preproc_done, preproc_stream_);
            auto t2_local = std::chrono::high_resolution_clock::now();
            t2 = t2_local;

            // Stage 3: 推理
            cudaStreamWaitEvent(infer_stream_, slot.preproc_done, 0);
            // 动态 batch 引擎需要设置 shape
            bool trt_ok = trt_context_->setInputShape(input_name,
                nvinfer1::Dims4{1, 3, model_input_h_, model_input_w_});
            trt_ok = trt_context_->setTensorAddress(input_name, slot.tensor_input_buf) && trt_ok;
            trt_ok = trt_context_->setTensorAddress(output_name, slot.tensor_output_buf) && trt_ok;
            trt_ok = trt_context_->enqueueV3(infer_stream_) && trt_ok;
            if (!trt_ok) {
                LOG_ERROR("Pipeline %d: TensorRT shape/binding/enqueue failed", id_);
                running_ = false;
                break;
            }
            cudaEventRecord(slot.infer_done, infer_stream_);
            auto t3_local = std::chrono::high_resolution_clock::now();
            t3 = t3_local;

            // Stage 4: D2H
            cudaStreamWaitEvent(postproc_stream_, slot.infer_done, 0);
            if (slot.tensor_output_fp16) {
                launch_half_to_float(slot.tensor_output_buf, slot.output_buf,
                                     model_output_elements_, postproc_stream_);
            }
            cudaMemcpyAsync(slot.host_output, slot.output_buf, output_bytes,
                            cudaMemcpyDeviceToHost, postproc_stream_);
            cudaEventRecord(slot.postproc_done, postproc_stream_);
            auto t4_local = std::chrono::high_resolution_clock::now();
            t4 = t4_local;

            // 同步 + CPU NMS
            cudaEventSynchronize(slot.postproc_done);
            std::vector<Detection> dets = cpu_postprocess(slot.host_output, model_output_predictions_, conf, nms,
                                   slot.src_width, slot.src_height, model_input_w_);

            // 显示帧转换（GPU 直接输出 RGB 到 GPU 缓冲区，GUI 零拷贝渲染）
            cv::Mat display_frame;
            float disp_scale = std::min((float)kDisplayMaxW / slot.src_width,
                                        (float)kDisplayMaxH / slot.src_height);
            int disp_w = (int)(slot.src_width * disp_scale);
            int disp_h = (int)(slot.src_height * disp_scale);
            disp_w &= ~1; disp_h &= ~1;

            auto now_disp_d = std::chrono::steady_clock::now();
            bool do_display_d = recorder_.isRecording() ||
                std::chrono::duration<float, std::milli>(
                    now_disp_d - last_display_update_).count() > 8.0f;

            if (do_display_d) {
                size_t buf_size = (size_t)disp_w * disp_h * 3;
                if (display_gpu_buf_ == nullptr || display_gpu_buf_size_ < buf_size) {
                    if (display_gpu_buf_) cudaFree(display_gpu_buf_);
                    cudaMalloc(&display_gpu_buf_, buf_size);
                    display_gpu_buf_size_ = buf_size;
                }

                if (slot.pixel_format == FramePixelFormat::NV12) {
                    if (slot.bit_shift > 0) {
                        launch_nv12_16bit_resize_rgb(slot.nv12_y, slot.nv12_uv,
                                                     slot.src_width, slot.src_height,
                                                     slot.y_pitch, slot.uv_pitch,
                                                     display_gpu_buf_, disp_w, disp_h,
                                                     slot.bit_shift, postproc_stream_);
                    } else {
                        launch_nv12_resize_rgb(slot.nv12_y, slot.nv12_uv,
                                               slot.src_width, slot.src_height,
                                               slot.y_pitch, slot.uv_pitch,
                                               display_gpu_buf_, disp_w, disp_h,
                                               postproc_stream_);
                    }
                } else {
                    if (slot.bit_shift > 0) {
                        launch_yuyv16_resize_rgb(slot.nv12_y, slot.src_width, slot.src_height,
                                                 slot.y_pitch, display_gpu_buf_, disp_w, disp_h,
                                                 slot.bit_shift, postproc_stream_);
                    } else {
                        launch_yuyv_resize_rgb(slot.nv12_y, slot.src_width, slot.src_height,
                                               slot.y_pitch, display_gpu_buf_, disp_w, disp_h,
                                               postproc_stream_, v4l2_.isGrey());
                    }
                }
                cudaStreamSynchronize(postproc_stream_);
                display_frame = cv::Mat(disp_h, disp_w, CV_8UC3);
                cudaMemcpy(display_frame.data, display_gpu_buf_, buf_size, cudaMemcpyDeviceToHost);
                cv::cvtColor(display_frame, display_frame, cv::COLOR_RGB2BGR);
                if (!camera_cpu_view.empty()) {
                    display_frame = make_camera_preview(disp_w, disp_h);
                }
                last_display_update_ = now_disp_d;
            }

            // 缩放检测坐标到显示尺寸
            if (disp_scale != 1.0f) {
                for (auto& d : dets) {
                    d.x1 *= disp_scale; d.y1 *= disp_scale;
                    d.x2 *= disp_scale; d.y2 *= disp_scale;
                }
            }

            // 录制需要 CPU BGR 帧（带标注）
            if (recorder_.isRecording()) {
                cv::Mat record_frame = display_frame.clone();
                if (!dets.empty()) draw_detections_inplace(record_frame, dets);
                recorder_.pushFrame(record_frame);
            }

            {
                std::lock_guard<std::mutex> lk(result_mtx_);
                if (do_display_d) {
                    result_.display = display_frame;
                    result_.display_gpu = nullptr;
                    result_.display_w = disp_w;
                    result_.display_h = disp_h;
                }
                result_.src_w = slot.src_width;
                result_.src_h = slot.src_height;
                result_.dets = dets;
                result_.num_objects = (int)dets.size();
                result_.has_frame = true;
                result_.sahi_cached = false;
            }
            t5 = std::chrono::high_resolution_clock::now();

        } else {
            // ==================== OFF 模式（GPU 直接显示） ====================
            float disp_scale = std::min((float)kDisplayMaxW / slot.src_width,
                                        (float)kDisplayMaxH / slot.src_height);
            int disp_w = (int)(slot.src_width * disp_scale);
            int disp_h = (int)(slot.src_height * disp_scale);
            disp_w &= ~1; disp_h &= ~1;

            auto now_disp = std::chrono::steady_clock::now();
            bool do_display = std::chrono::duration<float, std::milli>(
                now_disp - last_display_update_).count() > 8.0f;  // ~120fps 上限

            if (do_display) {
                // Camera cudaPinned/fallback upload runs asynchronously on
                // decode_stream_. OFF mode has no preprocess stage to create
                // this dependency, so the display stream must wait explicitly.
                cudaStreamWaitEvent(postproc_stream_, slot.decode_done, 0);
                size_t buf_size = (size_t)disp_w * disp_h * 3;
                if (display_gpu_buf_ == nullptr || display_gpu_buf_size_ < buf_size) {
                    if (display_gpu_buf_) cudaFree(display_gpu_buf_);
                    cudaMalloc(&display_gpu_buf_, buf_size);
                    display_gpu_buf_size_ = buf_size;
                }

                if (slot.pixel_format == FramePixelFormat::NV12) {
                    if (slot.bit_shift > 0) {
                        launch_nv12_16bit_resize_rgb(slot.nv12_y, slot.nv12_uv,
                                                     slot.src_width, slot.src_height,
                                                     slot.y_pitch, slot.uv_pitch,
                                                     display_gpu_buf_, disp_w, disp_h,
                                                     slot.bit_shift, postproc_stream_);
                    } else {
                        launch_nv12_resize_rgb(slot.nv12_y, slot.nv12_uv,
                                               slot.src_width, slot.src_height,
                                               slot.y_pitch, slot.uv_pitch,
                                               display_gpu_buf_, disp_w, disp_h,
                                               postproc_stream_);
                    }
                } else {
                    if (slot.bit_shift > 0) {
                        launch_yuyv16_resize_rgb(slot.nv12_y, slot.src_width, slot.src_height,
                                                 slot.y_pitch, display_gpu_buf_, disp_w, disp_h,
                                                 slot.bit_shift, postproc_stream_);
                    } else {
                        launch_yuyv_resize_rgb(slot.nv12_y, slot.src_width, slot.src_height,
                                               slot.y_pitch, display_gpu_buf_, disp_w, disp_h,
                                               postproc_stream_, v4l2_.isGrey());
                    }
                }
                cudaError_t display_err = cudaStreamSynchronize(postproc_stream_);
                if (display_err != cudaSuccess) {
                    LOG_ERROR("Pipeline %d: camera/video display kernel failed: %s (format=%d, pitch=%d, %dx%d)",
                              id_, cudaGetErrorString(display_err), (int)slot.pixel_format,
                              slot.y_pitch, slot.src_width, slot.src_height);
                    if (acquired_v4l2_buffer >= 0) v4l2_.releaseFrame(acquired_v4l2_buffer);
                    continue;
                }
                cv::Mat rgb(disp_h, disp_w, CV_8UC3);
                cudaError_t copy_err = cudaMemcpy(rgb.data, display_gpu_buf_, buf_size, cudaMemcpyDeviceToHost);
                if (copy_err != cudaSuccess) {
                    LOG_ERROR("Pipeline %d: display D2H failed: %s", id_, cudaGetErrorString(copy_err));
                    if (acquired_v4l2_buffer >= 0) v4l2_.releaseFrame(acquired_v4l2_buffer);
                    continue;
                }
                cv::Mat bgr;
                cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
                if (!camera_cpu_view.empty()) {
                    bgr = make_camera_preview(disp_w, disp_h);
                }
                last_display_update_ = now_disp;

                std::lock_guard<std::mutex> lk(result_mtx_);
                result_.display = bgr;
                result_.display_gpu = nullptr;
                result_.display_w = disp_w;
                result_.display_h = disp_h;
                result_.src_w = slot.src_width;
                result_.src_h = slot.src_height;
                result_.dets.clear();
                result_.num_objects = 0;
                result_.has_frame = true;
                result_.sahi_cached = false;
            }
        }

        // CUDA has completed all reads of the camera mmap/DMA-BUF at this point.
        if (acquired_v4l2_buffer >= 0) {
            v4l2_.releaseFrame(acquired_v4l2_buffer);
        }

        // 释放 FFmpeg 帧引用（在 NV12→RGB 转换完成后）
        if (config_.type == InputType::VIDEO) {
            decoder_.releaseFrame();
        }

        // 性能统计
        double dt_decode = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double dt_preproc = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double dt_infer = std::chrono::duration<double, std::milli>(t3 - t2).count();
        double dt_postproc = std::chrono::duration<double, std::milli>(t4 - t3).count();
        double dt_display = std::chrono::duration<double, std::milli>(t5 - t4).count();
        sum_decode += dt_decode;
        sum_preproc += dt_preproc;
        sum_infer += dt_infer;
        sum_postproc += dt_postproc;
        frame_num++;

        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - fps_start_).count();
        if (elapsed >= 1.0f) {
            float fps = frame_num / elapsed;
            float avg_decode = sum_decode / frame_num;
            float avg_preproc = sum_preproc / frame_num;
            float avg_infer = sum_infer / frame_num;
            float avg_postproc = sum_postproc / frame_num;
            float avg_display = dt_display;  // 最新帧的显示耗时
            {
                std::lock_guard<std::mutex> lk(result_mtx_);
                result_.fps = fps;
                result_.cam_fps = v4l2_.isOpen() ? v4l2_.measuredFps() : 0.0f;
                result_.ms_decode = avg_decode;
                result_.ms_preproc = avg_preproc;
                result_.ms_infer = avg_infer;
                result_.ms_postproc = avg_postproc;
                result_.ms_display = avg_display;
                result_.ms_total = avg_decode + avg_preproc + avg_infer + avg_postproc + avg_display;
            }

            double total_ms = (sum_decode + sum_preproc + sum_infer + sum_postproc) / frame_num;
            LOG_INFO("Pipeline %d: %.1f fps | decode=%.2f preproc=%.2f infer=%.2f postproc=%.2f total=%.2f ms",
                     id_, fps,
                     sum_decode / frame_num, sum_preproc / frame_num,
                     sum_infer / frame_num, sum_postproc / frame_num,
                     total_ms);

            frame_num = 0;
            fps_start_ = now;
            sum_decode = sum_preproc = sum_infer = sum_postproc = 0;
        }

        slot_idx = (slot_idx + 1) % ring_buffer_.numSlots();
    }

    LOG_INFO("Pipeline %d: worker thread finished", id_);
}

cv::Mat Pipeline::snapshotRaw() {
    return cam_yuyv_frame_.clone();
}
