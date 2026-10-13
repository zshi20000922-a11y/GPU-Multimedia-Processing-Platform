#include "detection_engine.h"
#include "logger.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <regex>

using namespace nvinfer1;

class EngineLogger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR)
            LOG_ERROR("[TRT] %s", msg);
        else if (severity == Severity::kWARNING)
            LOG_WARN("[TRT] %s", msg);
        else
            LOG_DEBUG("[TRT] %s", msg);
    }
} gLogger;

// COCO 类名
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

const char* DetectionEngine::get_class_name(int class_id) {
    if (class_id >= 0 && class_id < 80) return COCO_NAMES[class_id];
    return "?";
}

const char* DetectionEngine::model_class_name(int class_id) const {
    if (class_id >= 0 && class_id < (int)class_names_.size() &&
        !class_names_[class_id].empty()) return class_names_[class_id].c_str();
    return get_class_name(class_id);
}

bool DetectionEngine::load(const std::string& engine_path, int input_size) {
    input_size_ = input_size;

    LOG_INFO("Loading engine: %s", engine_path.c_str());

    std::ifstream f(engine_path, std::ios::binary);
    if (!f.good()) {
        LOG_ERROR("Cannot open engine file: %s", engine_path.c_str());
        return false;
    }
    f.seekg(0, f.end);
    size_t sz = f.tellg();
    f.seekg(0, f.beg);
    std::vector<char> data(sz);
    f.read(data.data(), sz);
    f.close();
    LOG_INFO("Engine file read: %zu bytes", sz);

    auto runtime = createInferRuntime(gLogger);
    if (!runtime) {
        LOG_ERROR("Failed to create TensorRT runtime");
        return false;
    }

    // Ultralytics .engine files prepend a uint32 JSON metadata block.
    // Detect it before deserialization to avoid a noisy TensorRT magic-tag error.
    size_t plan_offset = 0;
    if (sz > sizeof(uint32_t) + 1 && data[sizeof(uint32_t)] == '{') {
        uint32_t metadata_len = 0;
        std::memcpy(&metadata_len, data.data(), sizeof(metadata_len));
        const size_t candidate = sizeof(metadata_len) + metadata_len;
        if (metadata_len > 0 && candidate < sz) {
            LOG_INFO("Detected Ultralytics engine metadata (%u bytes)", metadata_len);
            plan_offset = candidate;

            // Read the per-engine label map. This is essential when different
            // channels use a one-class Drone engine and an 80-class COCO
            // engine at the same time.
            std::string metadata(data.data() + sizeof(metadata_len), metadata_len);
            const size_t names_pos = metadata.find("\"names\"");
            if (names_pos != std::string::npos) {
                const size_t begin = metadata.find('{', names_pos);
                const size_t end = begin == std::string::npos
                    ? std::string::npos : metadata.find('}', begin);
                if (end != std::string::npos) {
                    const std::string names = metadata.substr(begin, end - begin + 1);
                    const std::regex item("\\\"([0-9]+)\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
                    for (std::sregex_iterator it(names.begin(), names.end(), item), last;
                         it != last; ++it) {
                        const int id = std::stoi((*it)[1].str());
                        if (id >= (int)class_names_.size()) class_names_.resize(id + 1);
                        class_names_[id] = (*it)[2].str();
                    }
                    if (!class_names_.empty()) num_classes_ = (int)class_names_.size();
                }
            }
        }
    }
    engine_ = runtime->deserializeCudaEngine(data.data() + plan_offset, sz - plan_offset);
    delete runtime;
    if (!engine_) {
        LOG_ERROR("Engine deserialize failed");
        return false;
    }

    // 检测是否支持动态 batch
    // 检查输入 tensor 的第一个维度是否为 -1（动态）
    const char* input_tensor_name = nullptr;
    const char* output_tensor_name = nullptr;
    int num_tensors = engine_->getNbIOTensors();
    for (int i = 0; i < num_tensors; i++) {
        const char* name = engine_->getIOTensorName(i);
        TensorIOMode mode = engine_->getTensorIOMode(name);
        if (mode == TensorIOMode::kINPUT) {
            Dims dims = engine_->getTensorShape(name);
            input_tensor_name = name;
            input_type_ = engine_->getTensorDataType(name);
            if (dims.nbDims == 4) {
                if (dims.d[2] > 0) input_height_ = dims.d[2];
                if (dims.d[3] > 0) input_width_ = dims.d[3];
            }
            if (dims.d[0] == -1) {
                dynamic_batch_ = true;
                LOG_INFO("Engine supports dynamic batch (input: %s)", name);
            }
        } else if (!output_tensor_name) {
            output_tensor_name = name;
        }
    }
    input_size_ = input_width_;

    if (input_type_ != DataType::kFLOAT && input_type_ != DataType::kHALF) {
        LOG_ERROR("Unsupported engine input datatype %d: CUDA preprocess currently emits FP32", (int)input_type_);
        delete engine_;
        engine_ = nullptr;
        return false;
    }

    if (output_tensor_name) {
        Dims dims = engine_->getTensorShape(output_tensor_name);
        output_type_ = engine_->getTensorDataType(output_tensor_name);
        long elements = 1;
        for (int d = 1; d < dims.nbDims; ++d) {
            if (dims.d[d] <= 0) { elements = 0; break; }
            elements *= dims.d[d];
        }
        if (elements > 0) output_elements_ = (int)elements;
        if (dims.nbDims == 3) {
            output_predictions_ = dims.d[1];
            output_stride_ = dims.d[2];
            end2end_ = output_stride_ == 6;
        }
        if (output_type_ != DataType::kFLOAT && output_type_ != DataType::kHALF) {
            LOG_ERROR("Unsupported engine output datatype %d: host postprocess expects FP32", (int)output_type_);
            delete engine_;
            engine_ = nullptr;
            return false;
        }
    }

    // 查询最大 batch 大小
    if (dynamic_batch_ && input_tensor_name) {
        // TensorRT 10.0 不直接暴露 profile 维度查询接口
        // 使用合理默认值，实际限制由 TensorRT 在 enqueueV3 时强制执行
        max_batch_size_ = 24;  // 合理默认值
        LOG_INFO("Dynamic batch engine detected, max_batch set to %d (actual limit enforced by TRT)", max_batch_size_);
    } else {
        // 静态 batch: 从输入 shape 读取
        const char* in_name = engine_->getIOTensorName(0);
        Dims dims = engine_->getTensorShape(in_name);
        max_batch_size_ = dims.d[0];
    }

    // 打印所有 IO tensor 的 shape，用于诊断输出格式是否匹配
    for (int i = 0; i < num_tensors; i++) {
        const char* name = engine_->getIOTensorName(i);
        TensorIOMode mode = engine_->getTensorIOMode(name);
        Dims dims = engine_->getTensorShape(name);
        std::string shape = "[";
        for (int d = 0; d < dims.nbDims; d++) {
            if (d) shape += ",";
            shape += std::to_string(dims.d[d]);
        }
        shape += "]";
        LOG_INFO("  tensor[%d] %s: %s shape=%s",
                 i, mode == TensorIOMode::kINPUT ? "INPUT " : "OUTPUT",
                 name, shape.c_str());
    }
    LOG_INFO("Engine loaded: %s (input=%dx%d %s, output=%dx%d %s, format=%s, dynamic_batch=%s, max_batch=%d)",
             engine_path.c_str(), input_width_, input_height_,
             input_type_ == DataType::kHALF ? "FP16" : "FP32",
             output_predictions_, output_stride_, output_type_ == DataType::kHALF ? "FP16" : "FP32",
             end2end_ ? "YOLO26 end-to-end" : "YOLOv5 raw",
             dynamic_batch_ ? "yes" : "no", max_batch_size_);
    if (!class_names_.empty()) {
        LOG_INFO("Engine labels: %d classes (class 0=%s)", num_classes_,
                 class_names_[0].c_str());
    }
    return true;
}

IExecutionContext* DetectionEngine::create_context() {
    if (!engine_) {
        LOG_ERROR("Engine not loaded");
        return nullptr;
    }
    return engine_->createExecutionContext();
}

void DetectionEngine::destroy() {
    if (engine_) { delete engine_; engine_ = nullptr; }
    class_names_.clear();
}
