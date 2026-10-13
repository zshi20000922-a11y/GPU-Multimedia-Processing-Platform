/**
 * YOLOv5 SAHI Pipeline - Multi-input with ImGui
 */

#include "pipeline_manager.h"
#include "app_gui.h"
#include "logger.h"

#include "imgui.h"

#include <chrono>
#include <cstdlib>
#include <thread>

int main(int argc, char** argv) {
    LOG_INFO("Starting Drone Detection Demonstrator...");

    AppGUI gui;
    if (!gui.init("AI Drone Detection — Expert Demonstration", 1600, 900)) {
        LOG_ERROR("GUI init failed");
        return 1;
    }
    ImGui::GetIO().FontGlobalScale = 1.2f;

    PipelineManager mgr;

    // 默认配置：3 路摄像头
    for (int i = 0; i < 3; i++) {
        gui.state().inputs[i].id = i;
        gui.state().inputs[i].type = InputType::CAMERA;
        gui.state().inputs[i].source = std::to_string(i);
        gui.state().inputs[i].detect_mode = DetectMode::OFF;
        gui.state().inputs[i].cam_width = 1920;
        gui.state().inputs[i].cam_height = 1080;
        gui.state().inputs[i].cam_fps = 30.0f;
        // 保持原系统的 GMAX3412/FX20 packed-gray 采集约定：驱动以
        // 3840x1080 YUYV 搬运，程序按 3840x2160 单通道图像解释。
        gui.state().inputs[i].cam_packed_gray = false;
        // Models are user supplied and selected in the GUI. No engine is
        // bundled with the repository because TensorRT plans are platform-specific.
        gui.state().pipeline_configs[i].use_own_engine = false;
        gui.state().pipeline_configs[i].own_engine_path.clear();
    }

    // Non-interactive visual smoke test used by development/CI. It is opt-in
    // and leaves the normal expert-demo startup behaviour unchanged.
    if (std::getenv("CPP_PIPELINE_AUTOTEST")) {
        gui.state().inputs[1].type = InputType::NONE;
        gui.state().inputs[2].type = InputType::NONE;
        if (const char* video = std::getenv("CPP_PIPELINE_AUTOTEST_VIDEO")) {
            gui.state().inputs[0].type = InputType::VIDEO;
            gui.state().inputs[0].source = video;
            gui.state().inputs[0].cam_packed_gray = false;
            gui.state().inputs[0].detect_mode = DetectMode::DIRECT;
            gui.state().pipeline_configs[0].use_own_engine = false;
            gui.state().pipeline_configs[0].own_engine_path.clear();
            gui.state().pipeline_configs[0].conf_thresh = 0.25f;
            gui.state().inputs[0].conf_thresh = 0.25f;
        } else if (std::getenv("CPP_PIPELINE_AUTOTEST_DETECT")) {
            gui.state().inputs[0].detect_mode = DetectMode::DIRECT;
        }
        if (const char* engine = std::getenv("CPP_PIPELINE_AUTOTEST_ENGINE"))
            gui.state().engine_path = engine;
        gui.state().engine_loaded = mgr.load_engine(gui.state().engine_path);
        if (gui.state().engine_loaded) {
            if (gui.state().pipeline_configs[0].use_own_engine)
                mgr.start_pipeline_with_engine(
                    0, gui.state().inputs[0],
                    gui.state().pipeline_configs[0].own_engine_path);
            else
                mgr.start_pipeline(0, gui.state().inputs[0]);
            LOG_INFO("AUTOTEST: model loaded and input 0 started");
        }
    }

    while (!gui.should_close()) {
        auto frame_start = std::chrono::steady_clock::now();
        gui.begin_frame();

        int win_w, win_h;
        gui.get_window_size(win_w, win_h);

        float toolbar_h = 50.0f;
        float video_h = win_h - toolbar_h;  // 视频区铺满工具栏以下全部空间
        float single_w = (float)win_w / 3.0f;

        // === 工具栏 ===
        gui.render_toolbar(mgr);

        int zoom = gui.zoomed_channel();

        // === 视频窗口（无缝：去边框、去内边距、纯黑背景）===
        // 视频绘制完成后，在左上角叠加半透明配置浮层
        auto render_video_window = [&](int i, float x, float w) {
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 1));
            ImGui::SetNextWindowPos(ImVec2(x, toolbar_h));
            ImGui::SetNextWindowSize(ImVec2(w, video_h));
            char vname[32];
            snprintf(vname, sizeof(vname), "##vid%d", i);
            ImGui::Begin(vname, nullptr,
                         ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                         ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                         ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoBringToFrontOnFocus);
            gui.render_video(i, mgr.get_result(i));
            ImGui::End();
            ImGui::PopStyleColor();
            ImGui::PopStyleVar(2);

            // 配置浮层（叠加在该视频列左上角）
            ImGui::SetNextWindowPos(ImVec2(x + 8, toolbar_h + 8));
            ImGui::SetNextWindowBgAlpha(0.55f);
            char cname[32];
            snprintf(cname, sizeof(cname), "##cfgov%d", i);
            ImGui::Begin(cname, nullptr,
                         ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                         ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                         ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoScrollbar);
            gui.render_input_overlay(i, mgr);
            ImGui::End();
        };

        // === 视频区 ===
        if (gui.state().panorama_mode) {
            // 拼接显示：单窗口铺满视频区
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4, 4));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 1));
            ImGui::SetNextWindowPos(ImVec2(0, toolbar_h));
            ImGui::SetNextWindowSize(ImVec2((float)win_w, video_h));
            ImGui::Begin("##panorama", nullptr,
                         ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                         ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                         ImGuiWindowFlags_NoBringToFrontOnFocus);
            gui.render_panorama(mgr);
            ImGui::End();
            ImGui::PopStyleColor();
            ImGui::PopStyleVar();
        } else if (zoom >= 0 && zoom < 3) {
            render_video_window(zoom, 0, (float)win_w);  // 放大：铺满宽度
        } else {
            // 按可见通道动态分配宽度
            int vis[3], nvis = 0;
            for (int i = 0; i < 3; i++)
                if (gui.state().ch_visible[i]) vis[nvis++] = i;
            if (nvis == 0) nvis = 1, vis[0] = 0;  // 至少显示一路
            float cw = (float)win_w / nvis;
            for (int j = 0; j < nvis; j++)
                render_video_window(vis[j], j * cw, cw);
        }

        // === GPU 状态栏：右下角半透明小浮窗 ===
        float gpu_w = 300.0f, gpu_panel_h = 92.0f;
        ImGui::SetNextWindowPos(ImVec2(win_w - gpu_w - 8, win_h - gpu_panel_h - 8));
        ImGui::SetNextWindowSize(ImVec2(gpu_w, gpu_panel_h));
        ImGui::SetNextWindowBgAlpha(0.75f);
        ImGui::Begin("GPU", nullptr,
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar);
        gui.render_gpu_panel();
        ImGui::End();

        gui.end_frame();

        // 显示帧率限制：限制 GUI 渲染刷新率（不影响采集/推理线程）
        float disp_limit = gui.state().display_fps_limit;
        if (disp_limit > 0) {
            auto target = std::chrono::duration<double>(1.0 / disp_limit);
            auto elapsed = std::chrono::steady_clock::now() - frame_start;
            auto remain = std::chrono::duration_cast<std::chrono::duration<double>>(target - elapsed);
            if (remain.count() > 0)
                std::this_thread::sleep_for(remain);
        }
    }

    mgr.stop_all();
    gui.shutdown();
    LOG_INFO("Exited");
    return 0;
}
