#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

// 检测结果
struct Detection {
    float x1, y1, x2, y2;
    float conf;
    int class_id;
    std::string class_name;
};

// TensorRT 引擎封装（共享，只读）
// Pipeline 直接使用引擎创建自己的 ExecutionContext
class DetectionEngine {
public:
    DetectionEngine() = default;
    ~DetectionEngine() { destroy(); }

    // 加载引擎
    bool load(const std::string& engine_path, int input_size = 0);

    // 创建独立的 execution context（每个 Pipeline 调用一次）
    nvinfer1::IExecutionContext* create_context();

    // 获取引擎指针（用于创建 context）
    nvinfer1::ICudaEngine* engine() const { return engine_; }

    // 获取模型信息
    int input_size() const { return input_size_; }
    int num_classes() const { return num_classes_; }
    const char* model_class_name(int class_id) const;
    bool is_loaded() const { return engine_ != nullptr; }
    bool supports_dynamic_batch() const { return dynamic_batch_; }
    int max_batch_size() const { return max_batch_size_; }
    int input_width() const { return input_width_; }
    int input_height() const { return input_height_; }
    int output_elements() const { return output_elements_; }
    int output_predictions() const { return output_predictions_; }
    int output_stride() const { return output_stride_; }
    bool is_end2end() const { return end2end_; }
    nvinfer1::DataType input_type() const { return input_type_; }
    nvinfer1::DataType output_type() const { return output_type_; }

    // COCO 类名
    static const char* get_class_name(int class_id);

    void destroy();

private:
    nvinfer1::ICudaEngine* engine_ = nullptr;
    int input_size_ = 640;
    int num_classes_ = 80;
    bool dynamic_batch_ = false;
    int max_batch_size_ = 1;
    int input_width_ = 640;
    int input_height_ = 640;
    int output_elements_ = 25200 * 85;
    int output_predictions_ = 25200;
    int output_stride_ = 85;
    bool end2end_ = false;
    nvinfer1::DataType input_type_ = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType output_type_ = nvinfer1::DataType::kFLOAT;
    std::vector<std::string> class_names_;
};
