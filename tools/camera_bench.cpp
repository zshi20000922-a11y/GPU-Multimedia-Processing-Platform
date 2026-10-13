/**
 * 摄像头帧率基准测试
 * 纯采集 / 采集+预处理 / 采集+预处理+推理 三种模式
 * 无 GUI 显示，纯后台处理，测量最大帧率
 *
 * 用法:
 *   ./camera_bench              # 默认: 纯采集
 *   ./camera_bench --preproc    # 采集 + GPU 预处理
 *   ./camera_bench --infer      # 采集 + 预处理 + TensorRT 推理
 *   ./camera_bench --infer --engine /path/to/engine.trt
 *   ./camera_bench --device 0 --width 3840 --height 2160 --duration 10
 */

#include "v4l2_capture.h"
#include "cuda_kernels.h"
#include "detection_engine.h"
#include "ring_buffer.h"
#include "logger.h"

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <csignal>
#include <atomic>
#include <thread>

static std::atomic<bool> g_running{true};

void signal_handler(int) { g_running = false; }

struct BenchConfig {
    int device_id = 0;
    int width = 3840;
    int height = 2160;
    int duration_sec = 10;
    bool do_preproc = false;
    bool do_infer = false;
    std::string engine_path;
};

BenchConfig parse_args(int argc, char** argv) {
    BenchConfig cfg;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--preproc") == 0) cfg.do_preproc = true;
        else if (strcmp(argv[i], "--infer") == 0) { cfg.do_preproc = true; cfg.do_infer = true; }
        else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) cfg.device_id = atoi(argv[++i]);
        else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) cfg.width = atoi(argv[++i]);
        else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) cfg.height = atoi(argv[++i]);
        else if (strcmp(argv[i], "--duration") == 0 && i + 1 < argc) cfg.duration_sec = atoi(argv[++i]);
        else if (strcmp(argv[i], "--engine") == 0 && i + 1 < argc) cfg.engine_path = argv[++i];
    }
    return cfg;
}

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    BenchConfig cfg = parse_args(argc, argv);

    if (cfg.do_infer && cfg.engine_path.empty()) {
        fprintf(stderr, "ERROR: --infer requires --engine <path>\n");
        return 2;
    }

    printf("========================================\n");
    printf("  摄像头帧率基准测试\n");
    printf("========================================\n");
    printf("  设备:     /dev/video%d\n", cfg.device_id);
    printf("  分辨率:   %dx%d\n", cfg.width, cfg.height);
    printf("  测试时长: %d 秒\n", cfg.duration_sec);
    printf("  模式:     %s\n", cfg.do_infer ? "采集+预处理+推理" :
                                    cfg.do_preproc ? "采集+GPU预处理" : "纯采集");
    if (cfg.do_infer) printf("  引擎:     %s\n", cfg.engine_path.c_str());
    printf("========================================\n\n");

    // ===== 1. 打开摄像头 =====
    V4L2Capture cap;
    if (!cap.open(cfg.device_id, cfg.width, cfg.height)) {
        fprintf(stderr, "ERROR: 无法打开摄像头 /dev/video%d\n", cfg.device_id);
        return 1;
    }
    printf("摄像头已打开: %dx%d @ %.0f fps, 格式=%s\n",
           cap.width(), cap.height(), cap.fps(),
           cap.isDMABUF() ? "DMA-BUF" : cap.isCudaPinned() ? "cudaPinned" : "mmap");

    // ===== 2. GPU 预处理缓冲区 =====
    float* gpu_input = nullptr;
    uint8_t* gpu_staging = nullptr;
    size_t staging_size = 0;
    cudaStream_t stream = nullptr;

    if (cfg.do_preproc) {
        // TensorRT 输入: 640x640x3 float32
        cudaMalloc(&gpu_input, 640 * 640 * 3 * sizeof(float));
        cudaStreamCreate(&stream);
        staging_size = (size_t)cfg.width * cfg.height * 2;  // YUYV
        cudaMalloc(&gpu_staging, staging_size);
        printf("GPU 预处理缓冲区已分配\n");
    }

    // ===== 3. TensorRT 推理 =====
    std::unique_ptr<DetectionEngine> engine;
    nvinfer1::IExecutionContext* trt_ctx = nullptr;
    float* gpu_output = nullptr;
    float* host_output = nullptr;

    if (cfg.do_infer) {
        engine = std::make_unique<DetectionEngine>();
        if (!engine->load(cfg.engine_path)) {
            fprintf(stderr, "ERROR: 无法加载引擎 %s\n", cfg.engine_path.c_str());
            return 1;
        }
        trt_ctx = engine->create_context();
        if (!trt_ctx) {
            fprintf(stderr, "ERROR: 无法创建 TRT context\n");
            return 1;
        }
        cudaMalloc(&gpu_output, 25200 * 85 * sizeof(float));
        cudaMallocHost(&host_output, 25200 * 85 * sizeof(float));
        printf("TensorRT 引擎已加载: %s\n", cfg.engine_path.c_str());
    }

    // ===== 4. 主循环 =====
    printf("\n开始测试...\n");

    auto start_time = std::chrono::steady_clock::now();
    int frame_count = 0;
    double sum_capture = 0, sum_preproc = 0, sum_infer = 0;

    while (g_running) {
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - start_time).count();
        if (elapsed >= cfg.duration_sec) break;

        auto t0 = std::chrono::high_resolution_clock::now();

        // --- 采集 ---
        V4L2Frame frame;
        if (!cap.getNextFrame(frame, 200)) continue;

        auto t1 = std::chrono::high_resolution_clock::now();

        // --- GPU 预处理 (可选) ---
        if (cfg.do_preproc) {
            // 拷贝 YUYV 到 GPU
            cudaMemcpyAsync(gpu_staging, frame.data, frame.size,
                            cudaMemcpyHostToDevice, stream);

            // YUYV letterbox kernel
            float scale = std::min(640.0f / frame.width, 640.0f / frame.height);
            int new_w = (int)(frame.width * scale);
            int new_h = (int)(frame.height * scale);
            float pad_x = (640.0f - new_w) / 2.0f;
            float pad_y = (640.0f - new_h) / 2.0f;

            launch_yuyv_letterbox(gpu_staging, frame.width, frame.height,
                                  frame.width * 2, gpu_input, 640, 640,
                                  scale, pad_x, pad_y, stream);
        }

        auto t2 = std::chrono::high_resolution_clock::now();

        // --- TensorRT 推理 (可选) ---
        if (cfg.do_infer) {
            cudaStreamSynchronize(stream);
            const char* in_name = engine->engine()->getIOTensorName(0);
            const char* out_name = engine->engine()->getIOTensorName(1);
            trt_ctx->setTensorAddress(in_name, gpu_input);
            trt_ctx->setTensorAddress(out_name, gpu_output);
            trt_ctx->enqueueV3(stream);

            // D2H
            cudaMemcpyAsync(host_output, gpu_output,
                            25200 * 85 * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }

        auto t3 = std::chrono::high_resolution_clock::now();

        cap.releaseFrame(frame.buf_index);

        sum_capture += std::chrono::duration<double, std::milli>(t1 - t0).count();
        sum_preproc += std::chrono::duration<double, std::milli>(t2 - t1).count();
        sum_infer += std::chrono::duration<double, std::milli>(t3 - t2).count();
        frame_count++;

        // 每秒打印一次
        if (frame_count % 30 == 0) {
            float fps_so_far = frame_count / elapsed;
            printf("  [%5.1fs] %d 帧 | %.1f fps | capture=%.2f preproc=%.2f infer=%.2f ms\n",
                   elapsed, frame_count, fps_so_far,
                   sum_capture / frame_count, sum_preproc / frame_count, sum_infer / frame_count);
        }
    }

    // ===== 5. 结果 =====
    auto end_time = std::chrono::steady_clock::now();
    float total_sec = std::chrono::duration<float>(end_time - start_time).count();
    float avg_fps = frame_count / total_sec;

    printf("\n========================================\n");
    printf("  测试结果\n");
    printf("========================================\n");
    printf("  总帧数:   %d\n", frame_count);
    printf("  总耗时:   %.2f 秒\n", total_sec);
    printf("  平均帧率: %.1f fps\n", avg_fps);
    printf("  帧间隔:   %.1f ms\n", total_sec / frame_count * 1000);
    printf("----------------------------------------\n");
    printf("  平均耗时分解:\n");
    printf("    采集:     %.2f ms\n", sum_capture / frame_count);
    if (cfg.do_preproc)
        printf("    预处理:   %.2f ms\n", sum_preproc / frame_count);
    if (cfg.do_infer)
        printf("    推理:     %.2f ms\n", sum_infer / frame_count);
    printf("    总计:     %.2f ms\n",
           (sum_capture + sum_preproc + sum_infer) / frame_count);
    printf("========================================\n");

    // ===== 6. 清理 =====
    cap.stop();
    if (gpu_input) cudaFree(gpu_input);
    if (gpu_staging) cudaFree(gpu_staging);
    if (stream) cudaStreamDestroy(stream);
    if (gpu_output) cudaFree(gpu_output);
    if (host_output) cudaFreeHost(host_output);
    if (trt_ctx) delete trt_ctx;

    return 0;
}
