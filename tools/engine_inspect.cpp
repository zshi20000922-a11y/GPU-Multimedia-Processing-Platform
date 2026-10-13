#include "detection_engine.h"
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s model.engine\n", argv[0]);
        return 2;
    }
    DetectionEngine engine;
    if (!engine.load(argv[1])) return 1;
    std::printf("OK input=%dx%d output=%dx%d elements=%d format=%s\n",
                engine.input_width(), engine.input_height(),
                engine.output_predictions(), engine.output_stride(),
                engine.output_elements(), engine.is_end2end() ? "YOLO26" : "YOLOv5");

    if (engine.input_type() != nvinfer1::DataType::kFLOAT ||
        engine.output_type() != nvinfer1::DataType::kFLOAT) {
        std::printf("SMOKE skipped for FP16-I/O engine (Pipeline performs explicit FP32/FP16 conversion)\n");
        return 0;
    }

    auto* context = engine.create_context();
    if (!context) return 1;
    const char* input_name = engine.engine()->getIOTensorName(0);
    const char* output_name = engine.engine()->getIOTensorName(1);
    float *input = nullptr, *output = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    size_t input_bytes = (size_t)3 * engine.input_width() * engine.input_height() * sizeof(float);
    size_t output_bytes = (size_t)engine.output_elements() * sizeof(float);
    cudaMalloc(&input, input_bytes);
    cudaMalloc(&output, output_bytes);
    cudaMemset(input, 0, input_bytes);
    cudaStreamCreate(&stream);
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    context->setInputShape(input_name, nvinfer1::Dims4{1, 3, engine.input_height(), engine.input_width()});
    if (!context->setTensorAddress(input_name, input) ||
        !context->setTensorAddress(output_name, output)) return 1;
    for (int i = 0; i < 20; ++i) context->enqueueV3(stream);
    cudaStreamSynchronize(stream);
    cudaEventRecord(begin, stream);
    for (int i = 0; i < 200; ++i) context->enqueueV3(stream);
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, begin, end);
    std::printf("SMOKE inference=%.3f ms %.1f FPS (batch=1, 200 iterations)\n",
                elapsed_ms / 200.0f, 200000.0f / elapsed_ms);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);
    cudaFree(input);
    cudaFree(output);
    delete context;
    return 0;
}
