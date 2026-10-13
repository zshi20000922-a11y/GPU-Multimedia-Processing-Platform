#pragma once

#include "pipeline_manager.h"
#include "v4l2_capture.h"
#include "uvc_capture.h"
#include "gpu_monitor.h"
#include "hw_recorder.h"
#include "logger.h"

#include <opencv2/opencv.hpp>
#include <chrono>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <string>
#include <vector>

// 每路输入的独立配置
struct PipelineGUIConfig {
    // 引擎配置
    bool use_own_engine = false;           // 是否使用独立引擎
    std::string own_engine_path = "";      // 独立引擎路径

    // 后处理参数
    float conf_thresh = 0.25f;
    float nms_thresh = 0.45f;

    // SAHI 参数
    int sahi_stride = 320;
    int sahi_slice = 640;
    int sahi_interval = 15;        // SAHI 推理间隔（每隔多少帧推理一次）

    // 录制参数
    float record_duration = 60.0f;  // 默认 1 分钟

    // 摄像头能力（查询后缓存）
    CameraCapabilities cam_caps;
    int cam_format_idx = 0;    // 当前选中的格式索引
    int cam_resolution_idx = 0;
    int cam_fps_idx = 0;
    bool cam_caps_queried = false;
};

// GUI 配置
struct GUIState {
    // 专家演示默认使用精简页面；可在顶栏切换到高级设置。
    bool demo_mode = true;
    // 全局模型
    std::string engine_path;

    // 输入源配置 (最多3路)
    InputConfig inputs[3];

    // 每路独立配置
    PipelineGUIConfig pipeline_configs[3];

    // 拼接模式
    bool panorama_mode = false;      // 是否启用拼接显示
    int stitch_direction = 1;        // 0=横向(H), 1=竖向(V)

    // 融合模式（与 panorama 互斥）
    bool fusion_mode = false;
    enum class FusionOp { VIEW, DETECT, SHELL };
    FusionOp fusion_op = FusionOp::VIEW;
    float fusion_conf_thresh = 0.25f;
    float fusion_nms_thresh = 0.45f;

    // 显示帧率限制（GUI 渲染刷新率上限，0=不限制）
    float display_fps_limit = 60.0f;

    // 通道可见性（默认只显示通道0）
    bool ch_visible[3] = {true, false, false};

    // 运行状态
    bool engine_loaded = false;
};

class AppGUI {
public:
    AppGUI() = default;
    ~AppGUI();

    bool init(const char* title, int width, int height);
    void shutdown();

    void begin_frame();
    void end_frame();
    bool should_close() const;
    void get_window_size(int& w, int& h) const;

    // 渲染各面板
    void render_toolbar(PipelineManager& mgr);
    void render_input_panel(int id, PipelineManager& mgr);
    void render_input_overlay(int id, PipelineManager& mgr);  // 视频左上角叠加的精简配置浮层
    void render_log_panel();
    void render_video(int id, const PipelineResult& result);
    void render_panorama(PipelineManager& mgr);
    void render_gpu_panel();

    // 融合模式
    void render_fusion_page(PipelineManager& mgr);

    // 初始化共享显示纹理（在 pipeline 启动后调用）
    void setupSharedDisplay(int id, int width, int height);

    // 状态
    GUIState& state() { return state_; }

    // 放大显示的通道（-1=全部平铺, 0/1/2=放大某路）
    int zoomed_channel() const { return zoomed_channel_; }
    void set_zoomed_channel(int c) { zoomed_channel_ = c; }

    // 文件浏览器
    static std::string browse_file(const std::string& title,
                                   const std::vector<std::string>& exts = {});

private:
    void setup_style();
    void update_texture(int id, const cv::Mat& frame);
    void delete_texture(int id);

    GLFWwindow* window_ = nullptr;
    GUIState state_;
    GPUMonitor gpu_monitor_;

    // 纹理 (每路视频一个)
    unsigned int textures_[3] = {0, 0, 0};
    int tex_width_[3] = {0, 0, 0};
    int tex_height_[3] = {0, 0, 0};

    // 放大显示的通道（-1=全部平铺, 0/1/2=放大某路）
    int zoomed_channel_ = -1;

    // 摄像头设备列表（枚举缓存）
    std::vector<V4L2Capture::CameraInfo> cameras_;
    bool cameras_enumerated_ = false;

    // Reset USB 后的重新枚举宽限期：USB 硬复位后内核重新枚举 /dev/videoN 需数百ms，
    // 在该时间点之前即使枚举为空也显示 "Resetting…" 并每帧重试，而非误报 "No camera"
    std::chrono::steady_clock::time_point reset_pending_until_[3];

    // UVC 摄像头设备列表
#if HAS_LIBUVC
    std::vector<UvcCapture::CameraInfo> uvc_cameras_;
    bool uvc_cameras_enumerated_ = false;
#endif

    // 融合模式
    HWRecorder fusion_recorder_;       // Fusion 拼接录制（H.264）
    HWRecorder panorama_recorder_;    // Panorama 拼接录制（H.264）
    unsigned int fusion_tex_ = 0;
    int fusion_tex_w_ = 0, fusion_tex_h_ = 0;
    char shell_cmd_[512] = "";
    std::string shell_output_;
    std::vector<Detection> fusion_dets_;  // Detect 模式的检测结果

    // CUDA-OpenGL 互操作纹理（零拷贝显示）
    struct CudaGLTexture {
        unsigned int gl_tex = 0;
        cudaGraphicsResource_t cuda_res = nullptr;
        uint8_t* cuda_ptr = nullptr;       // CUDA 映射的设备指针
        int width = 0, height = 0;
        size_t size = 0;
        bool mapped = false;
    };
    CudaGLTexture shared_tex_[3];  // 每路一个共享纹理

    // 创建/销毁共享纹理
    bool createSharedTexture(int id, int width, int height);
    void destroySharedTexture(int id);
    bool shared_tex_failed_[3] = {false, false, false};  // 创建失败后不再重试

    // 获取共享纹理的 CUDA 设备指针（pipeline 写入用）
    // 返回 nullptr 表示不可用（回退到普通路径）
    uint8_t* mapSharedTexture(int id);
    void unmapSharedTexture(int id);
    int sharedTexWidth(int id) const { return shared_tex_[id].width; }
    int sharedTexHeight(int id) const { return shared_tex_[id].height; }

    bool initialized_ = false;
};
