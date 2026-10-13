#include "app_gui.h"
#include "logger.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <cstdio>
#include <ctime>
#include <algorithm>

static void glfw_error_callback(int error, const char* description) {
    LOG_ERROR("GLFW Error %d: %s", error, description);
}

AppGUI::~AppGUI() { shutdown(); }

void AppGUI::setup_style() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 3.0f;
    style.ScrollbarRounding = 6.0f;
    style.TabRounding = 4.0f;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.WindowPadding = ImVec2(10, 8);
    style.FramePadding = ImVec2(8, 4);
    style.ItemSpacing = ImVec2(8, 5);
    style.ItemInnerSpacing = ImVec2(6, 4);
    style.ScrollbarSize = 12.0f;
    style.GrabMinSize = 8.0f;

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg] = ImVec4(0.09f, 0.09f, 0.11f, 1.00f);
    colors[ImGuiCol_ChildBg] = ImVec4(0.11f, 0.11f, 0.13f, 1.00f);
    colors[ImGuiCol_PopupBg] = ImVec4(0.12f, 0.12f, 0.15f, 0.95f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.16f, 0.16f, 0.20f, 1.00f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.22f, 0.22f, 0.28f, 1.00f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.26f, 0.26f, 0.32f, 1.00f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.12f, 0.12f, 0.15f, 1.00f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.15f, 0.15f, 0.20f, 1.00f);
    colors[ImGuiCol_Button] = ImVec4(0.20f, 0.20f, 0.25f, 1.00f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.28f, 0.28f, 0.35f, 1.00f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.35f, 0.35f, 0.42f, 1.00f);
    colors[ImGuiCol_Header] = ImVec4(0.20f, 0.20f, 0.25f, 1.00f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.28f, 0.28f, 0.35f, 1.00f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.35f, 0.35f, 0.42f, 1.00f);
    colors[ImGuiCol_SliderGrab] = ImVec4(0.32f, 0.44f, 0.68f, 1.00f);
    colors[ImGuiCol_SliderGrabActive] = ImVec4(0.42f, 0.54f, 0.78f, 1.00f);
    colors[ImGuiCol_Separator] = ImVec4(0.20f, 0.20f, 0.25f, 1.00f);
    colors[ImGuiCol_Text] = ImVec4(0.88f, 0.88f, 0.92f, 1.00f);
    colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.55f, 1.00f);
    colors[ImGuiCol_ScrollbarBg] = ImVec4(0.10f, 0.10f, 0.12f, 1.00f);
    colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.25f, 0.25f, 0.30f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.32f, 0.32f, 0.38f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.38f, 0.38f, 0.44f, 1.00f);
    colors[ImGuiCol_CheckMark] = ImVec4(0.32f, 0.56f, 0.88f, 1.00f);
    colors[ImGuiCol_Tab] = ImVec4(0.16f, 0.16f, 0.20f, 1.00f);
    colors[ImGuiCol_TabHovered] = ImVec4(0.28f, 0.28f, 0.35f, 1.00f);
}

bool AppGUI::init(const char* title, int width, int height) {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) return false;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window_) { glfwTerminate(); return false; }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(0);  // 关闭 VSync，不限制渲染帧率到显示器刷新率

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    setup_style();

    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    for (int i = 0; i < 3; i++) {
        state_.inputs[i].id = i;
        state_.inputs[i].type = InputType::NONE;
        state_.inputs[i].detect_mode = DetectMode::OFF;
        // Prefer the deterministic D2H + OpenGL upload path for presentation.
        // CUDA-GL registration can silently map a different display GPU and
        // produce blank/gray textures while ImGui overlays remain visible.
        shared_tex_failed_[i] = true;
    }

    initialized_ = true;
    // 清除任何残留的 GPU 频率锁定（-lgc 在驱动层持久，会跨进程拖慢整条 pipeline，
    // 进而把摄像头采集帧率从 ~200 拖到 ~150）。启动时强制恢复默认时钟。
    gpu_monitor_.lockFrequency(0);
    LOG_INFO("GUI initialized");
    return true;
}

void AppGUI::shutdown() {
    if (!initialized_) return;
    // 退出时解锁 GPU 频率，避免把锁定状态遗留给下次运行
    gpu_monitor_.lockFrequency(0);
    for (int i = 0; i < 3; i++) delete_texture(i);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    if (window_) { glfwDestroyWindow(window_); window_ = nullptr; }
    glfwTerminate();
    initialized_ = false;
}

bool AppGUI::should_close() const {
    return window_ ? glfwWindowShouldClose(window_) : true;
}

void AppGUI::get_window_size(int& w, int& h) const {
    if (window_) glfwGetFramebufferSize(window_, &w, &h);
}

void AppGUI::begin_frame() {
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void AppGUI::end_frame() {
    ImGui::Render();
    int dw, dh;
    glfwGetFramebufferSize(window_, &dw, &dh);
    glViewport(0, 0, dw, dh);
    glClearColor(0.08f, 0.08f, 0.10f, 1.00f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window_);
}

void AppGUI::delete_texture(int id) {
    if (id >= 0 && id < 3 && textures_[id]) {
        glDeleteTextures(1, &textures_[id]);
        textures_[id] = 0;
    }
}

void AppGUI::update_texture(int id, const cv::Mat& frame) {
    if (frame.empty() || id < 0 || id >= 3) return;

    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

    if (textures_[id] == 0 || tex_width_[id] != rgb.cols || tex_height_[id] != rgb.rows) {
        delete_texture(id);
        glGenTextures(1, &textures_[id]);
        glBindTexture(GL_TEXTURE_2D, textures_[id]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.cols, rgb.rows, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
        tex_width_[id] = rgb.cols;
        tex_height_[id] = rgb.rows;
    } else {
        glBindTexture(GL_TEXTURE_2D, textures_[id]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, rgb.cols, rgb.rows,
                        GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
    }
}

// ==================== CUDA-OpenGL 互操作纹理 ====================

bool AppGUI::createSharedTexture(int id, int width, int height) {
    if (id < 0 || id >= 3) return false;
    destroySharedTexture(id);

    // 创建 OpenGL 纹理
    glGenTextures(1, &shared_tex_[id].gl_tex);
    glBindTexture(GL_TEXTURE_2D, shared_tex_[id].gl_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    // 分配 BGR 纹理（3 通道）
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, nullptr);

    // 注册到 CUDA
    cudaError_t err = cudaGraphicsGLRegisterImage(
        &shared_tex_[id].cuda_res, shared_tex_[id].gl_tex,
        GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    if (err != cudaSuccess) {
        LOG_ERROR("cudaGraphicsGLRegisterImage failed: %s", cudaGetErrorString(err));
        glDeleteTextures(1, &shared_tex_[id].gl_tex);
        shared_tex_[id].gl_tex = 0;
        return false;
    }

    shared_tex_[id].width = width;
    shared_tex_[id].height = height;
    shared_tex_[id].size = (size_t)width * height * 3;
    shared_tex_[id].mapped = false;

    LOG_INFO("Shared texture %d created: %dx%d", id, width, height);
    return true;
}

void AppGUI::destroySharedTexture(int id) {
    if (id < 0 || id >= 3) return;
    if (shared_tex_[id].cuda_res) {
        if (shared_tex_[id].mapped) unmapSharedTexture(id);
        cudaGraphicsUnregisterResource(shared_tex_[id].cuda_res);
        shared_tex_[id].cuda_res = nullptr;
    }
    if (shared_tex_[id].gl_tex) {
        glDeleteTextures(1, &shared_tex_[id].gl_tex);
        shared_tex_[id].gl_tex = 0;
    }
    shared_tex_[id].width = 0;
    shared_tex_[id].height = 0;
}

uint8_t* AppGUI::mapSharedTexture(int id) {
    if (id < 0 || id >= 3 || !shared_tex_[id].cuda_res) return nullptr;
    if (shared_tex_[id].mapped) return shared_tex_[id].cuda_ptr;

    // 映射 CUDA 资源
    cudaGraphicsMapResources(1, &shared_tex_[id].cuda_res);

    // 获取 CUDA 数组（用于 surface 写入）
    cudaArray_t array;
    cudaGraphicsSubResourceGetMappedArray(&array, shared_tex_[id].cuda_res, 0, 0);

    // 分配临时 GPU 缓冲区（kernel 写入这里，然后拷贝到纹理）
    // 注意：CUDA surface 写入需要特殊的 kernel，这里用临时缓冲区更简单
    if (!shared_tex_[id].cuda_ptr) {
        cudaMalloc(&shared_tex_[id].cuda_ptr, shared_tex_[id].size);
    }

    shared_tex_[id].mapped = true;
    return shared_tex_[id].cuda_ptr;
}

void AppGUI::unmapSharedTexture(int id) {
    if (id < 0 || id >= 3 || !shared_tex_[id].mapped) return;

    // 将 CUDA 缓冲区拷贝到 OpenGL 纹理
    cudaArray_t array;
    cudaGraphicsSubResourceGetMappedArray(&array, shared_tex_[id].cuda_res, 0, 0);
    cudaMemcpy2DToArray(array, 0, 0,
                        shared_tex_[id].cuda_ptr, shared_tex_[id].width * 3,
                        shared_tex_[id].width * 3, shared_tex_[id].height,
                        cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &shared_tex_[id].cuda_res);
    shared_tex_[id].mapped = false;
}

std::string AppGUI::browse_file(const std::string& title,
                                const std::vector<std::string>& exts) {
    std::string cmd = "zenity --file-selection --title=\"" + title + "\"";
    if (!exts.empty()) {
        cmd += " --file-filter='";
        for (size_t i = 0; i < exts.size(); i++) {
            if (i > 0) cmd += " ";
            cmd += "*." + exts[i];
        }
        cmd += "'";
    }
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    char buf[4096]; std::string result;
    while (fgets(buf, sizeof(buf), pipe)) result += buf;
    pclose(pipe);
    while (!result.empty() && (result.back()=='\n'||result.back()=='\r')) result.pop_back();
    return result;
}

void AppGUI::render_toolbar(PipelineManager& mgr) {
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    int win_w, win_h;
    get_window_size(win_w, win_h);
    ImGui::SetNextWindowSize(ImVec2((float)win_w, 55), ImGuiCond_Always);

    ImGui::Begin("##Toolbar", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoScrollbar);

    if (state_.demo_mode) {
        ImGui::TextColored(ImVec4(0.30f, 0.78f, 1.00f, 1.00f), "AI DRONE DETECTION");
        ImGui::SameLine();
        ImGui::TextDisabled(" | YOLO26s · TensorRT FP16 · 1280");
        ImGui::SameLine(500);

        if (!state_.engine_loaded) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.12f, 0.48f, 0.72f, 1));
            if (ImGui::Button("Load model")) {
                state_.engine_loaded = mgr.load_engine(state_.engine_path);
            }
            ImGui::PopStyleColor();
        } else {
            ImGui::TextColored(ImVec4(0.25f, 0.90f, 0.48f, 1), "● MODEL READY");
        }

        ImGui::SameLine();
        bool detecting = false;
        for (int i = 0; i < 3; ++i)
            detecting |= state_.inputs[i].type != InputType::NONE &&
                         state_.inputs[i].detect_mode == DetectMode::DIRECT;
        ImGui::PushStyleColor(ImGuiCol_Button, detecting ? ImVec4(0.08f,0.62f,0.38f,1)
                                                         : ImVec4(0.20f,0.22f,0.27f,1));
        if (ImGui::Button(detecting ? "Detection ON" : "Detection OFF")) {
            for (int i = 0; i < 3; ++i) {
                if (state_.inputs[i].type == InputType::NONE) continue;
                state_.inputs[i].detect_mode = detecting ? DetectMode::OFF : DetectMode::DIRECT;
                mgr.update_pipeline(i, state_.inputs[i]);
            }
        }
        ImGui::PopStyleColor();

        ImGui::SameLine();
        if (ImGui::Button("Start")) {
            if (!state_.engine_loaded) state_.engine_loaded = mgr.load_engine(state_.engine_path);
            if (state_.engine_loaded) {
                for (int i = 0; i < 3; ++i) {
                    if (state_.inputs[i].type == InputType::NONE || mgr.get_result(i).has_frame)
                        continue;
                    auto& pcfg = state_.pipeline_configs[i];
                    if (pcfg.use_own_engine && !pcfg.own_engine_path.empty())
                        mgr.start_pipeline_with_engine(i, state_.inputs[i], pcfg.own_engine_path);
                    else
                        mgr.start_pipeline(i, state_.inputs[i]);
                }
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Stop")) {
            for (int i = 0; i < 3; ++i) mgr.stop_pipeline(i);
        }

        ImGui::SameLine();
        ImGui::TextDisabled("Channels");
        for (int i = 0; i < 3; ++i) {
            ImGui::SameLine();
            char label[16];
            std::snprintf(label, sizeof(label), "%d##demo", i + 1);
            if (state_.ch_visible[i]) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.12f,0.48f,0.72f,1));
            if (ImGui::SmallButton(label)) state_.ch_visible[i] = !state_.ch_visible[i];
            if (state_.ch_visible[i]) ImGui::PopStyleColor();
        }

        ImGui::SameLine();
        if (ImGui::Button("Advanced")) state_.demo_mode = false;
        ImGui::End();
        return;
    }

    if (ImGui::Button("← Demo view")) state_.demo_mode = true;
    ImGui::SameLine();

    // 模型
    ImGui::Text("Model:");
    ImGui::SameLine();
    ImGui::PushItemWidth(300);
    char ebuf[512];
    strncpy(ebuf, state_.engine_path.c_str(), sizeof(ebuf));
    if (ImGui::InputText("##engine", ebuf, sizeof(ebuf)))
        state_.engine_path = ebuf;
    ImGui::PopItemWidth();
    ImGui::SameLine();
    if (ImGui::Button("Browse##eng")) {
        auto p = AppGUI::browse_file("Select Engine", {"engine", "trt"});
        if (!p.empty()) state_.engine_path = p;
    }
    ImGui::SameLine();
    if (ImGui::Button("Load Engine")) {
        if (mgr.load_engine(state_.engine_path)) {
            state_.engine_loaded = true;
            LOG_INFO("Engine loaded: %s", state_.engine_path.c_str());
        }
    }
    ImGui::SameLine();
    ImGui::TextColored(state_.engine_loaded ? ImVec4(0,1,0,1) : ImVec4(1,0,0,1),
                       state_.engine_loaded ? "[OK]" : "[Not Loaded]");

    // 全局检测模式开关
    ImGui::SameLine();
    ImGui::Text("  |  ");
    ImGui::SameLine();
    ImGui::TextDisabled("All:");
    ImGui::SameLine();

    // 获取当前三路的模式状态
    bool all_off = true, all_direct = true, all_sahi = true;
    for (int i = 0; i < 3; i++) {
        if (state_.inputs[i].type == InputType::NONE) continue;
        auto m = state_.inputs[i].detect_mode;
        if (m != DetectMode::OFF) all_off = false;
        if (m != DetectMode::DIRECT) all_direct = false;
        if (m != DetectMode::SAHI) all_sahi = false;
    }

    auto mode_btn = [&](DetectMode mode, const char* label, bool highlight) {
        if (highlight) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1));
        if (ImGui::SmallButton(label)) {
            for (int i = 0; i < 3; i++) {
                if (state_.inputs[i].type != InputType::NONE) {
                    state_.inputs[i].detect_mode = mode;
                    mgr.update_pipeline(i, state_.inputs[i]);
                }
            }
            LOG_INFO("Global mode: %s", label);
        }
        if (highlight) ImGui::PopStyleColor();
    };

    mode_btn(DetectMode::OFF, "View", all_off);
    ImGui::SameLine();
    mode_btn(DetectMode::DIRECT, "Detect", all_direct);
#if 0  // [HIDDEN] SAHI mode
    ImGui::SameLine();
    mode_btn(DetectMode::SAHI, "SAHI", all_sahi);
#endif

    // === 总开始 / 停止（一键启停所有非 NONE 通道）===
    ImGui::SameLine();
    ImGui::Text("  |  ");
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.55f, 0.15f, 1));
    if (ImGui::Button("Start All")) {
        for (int i = 0; i < 3; i++) {
            if (state_.inputs[i].type == InputType::NONE) continue;
            if (mgr.get_result(i).has_frame) continue;  // 已在运行
            auto& pcfg = state_.pipeline_configs[i];
            if (pcfg.use_own_engine && !pcfg.own_engine_path.empty())
                mgr.start_pipeline_with_engine(i, state_.inputs[i], pcfg.own_engine_path);
            else
                mgr.start_pipeline(i, state_.inputs[i]);
        }
        LOG_INFO("Start All channels");
    }
    ImGui::PopStyleColor();
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.65f, 0.15f, 0.15f, 1));
    if (ImGui::Button("Stop All")) {
        for (int i = 0; i < 3; i++) {
            if (mgr.get_result(i).has_frame) mgr.stop_pipeline(i);
        }
        LOG_INFO("Stop All channels");
    }
    ImGui::PopStyleColor();

    // === 显示帧率限制（仅限制 GUI 渲染刷新率，不影响采集/推理）===
    ImGui::SameLine();
    ImGui::Text("  |  ");
    ImGui::SameLine();
    ImGui::TextDisabled("Disp:");
    ImGui::SameLine();
    {
        const char* disp_labels[] = {"Unlimited", "30", "60", "120"};
        float disp_values[] = {0, 30, 60, 120};
        int disp_idx = state_.display_fps_limit <= 0 ? 0 :
                       state_.display_fps_limit <= 30 ? 1 :
                       state_.display_fps_limit <= 60 ? 2 : 3;
        ImGui::PushItemWidth(90);
        if (ImGui::Combo("##dispfps", &disp_idx, disp_labels, 4))
            state_.display_fps_limit = disp_values[disp_idx];
        ImGui::PopItemWidth();
    }

    // === 拼接显示开关 + 方向 ===
    ImGui::SameLine();
    ImGui::Text("  |  ");
    ImGui::SameLine();
    bool pan = state_.panorama_mode;
    if (ImGui::Checkbox("Stitch", &pan)) {
        state_.panorama_mode = pan;
        if (pan) state_.fusion_mode = false;
    }
    if (state_.panorama_mode) {
        ImGui::SameLine();
        const char* dir_names[] = {"Horizontal", "Vertical"};
        ImGui::PushItemWidth(110);
        ImGui::Combo("##stitchdir", &state_.stitch_direction, dir_names, 2);
        ImGui::PopItemWidth();
    }

#if 0  // [HIDDEN] Fusion + Panorama
    // 拼接模式
    ImGui::SameLine();
    ImGui::Text("  |  ");
    ImGui::SameLine();

    // Fusion 按钮（与 Panorama 互斥）
    if (state_.fusion_mode) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.55f, 0.85f, 1));
        if (ImGui::Button("Fusion [ON]")) {
            state_.fusion_mode = false;
        }
        ImGui::PopStyleColor();
    } else {
        if (ImGui::Button("Fusion")) {
            state_.fusion_mode = true;
            state_.panorama_mode = false;
        }
    }

    ImGui::SameLine();
    ImGui::Text("  |  ");
    ImGui::SameLine();

    // Panorama checkbox（与 Fusion 互斥）
    bool pan = state_.panorama_mode;
    if (ImGui::Checkbox("Panorama", &pan)) {
        state_.panorama_mode = pan;
        if (pan) state_.fusion_mode = false;
    }
    if (state_.panorama_mode) {
        ImGui::SameLine();
        const char* dir_names[] = {"H", "V"};
        char dir_id[] = "##dir";
        ImGui::PushItemWidth(40);
        ImGui::Combo(dir_id, &state_.stitch_direction, dir_names, 2);
        ImGui::PopItemWidth();
    }
#endif

    // 各通道可见性切换
    ImGui::SameLine();
    ImGui::Text("  |  ");
    for (int i = 0; i < 3; i++) {
        ImGui::SameLine();
        bool& vis = state_.ch_visible[i];
        auto r = mgr.get_result(i);
        bool running = r.has_frame;
        if (vis)
            ImGui::PushStyleColor(ImGuiCol_Button, running ? ImVec4(0.15f,0.55f,0.15f,1) : ImVec4(0.25f,0.25f,0.35f,1));
        else
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.12f,0.12f,0.15f,1));
        char lbl[16];
        snprintf(lbl, sizeof(lbl), "CH%d##chv", i + 1);
        if (ImGui::SmallButton(lbl)) vis = !vis;
        ImGui::PopStyleColor();
    }

    ImGui::End();
}

void AppGUI::render_input_panel(int id, PipelineManager& mgr) {
    if (id < 0 || id >= 3) return;
    InputConfig& cfg = state_.inputs[id];
    PipelineGUIConfig& pcfg = state_.pipeline_configs[id];
    bool config_changed = false;

    // === 行1: Type + Source + Detect Mode ===
#if HAS_LIBUVC
    const char* type_names[] = {"Off", "Camera", "Video", "UVC"};
    int type_count = 4;
#else
    const char* type_names[] = {"Off", "Camera", "Video"};
    int type_count = 3;
#endif
    int type_idx = (int)cfg.type;
    // 防止越界（例如旧配置中 type=UVC 但 libuvc 未编译）
    if (type_idx >= type_count) type_idx = 0;
    char tid[16];
    snprintf(tid, sizeof(tid), "##t%d", id);
    // 通道标签（三行布局下用于区分各路）
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1), "CH%d", id + 1);
    ImGui::SameLine();
    ImGui::PushItemWidth(90);
    if (ImGui::Combo(tid, &type_idx, type_names, type_count)) {
        // 类型切换时自动停止当前 pipeline
        if (mgr.get_result(id).has_frame) {
            mgr.stop_pipeline(id);
            LOG_INFO("CH %d: auto-stopped on type change", id + 1);
        }
        cfg.type = (InputType)type_idx;
        cfg.source.clear();  // 切换类型时清空源，避免旧路径导致崩溃
        if (cfg.type == InputType::VIDEO) {
            pcfg.use_own_engine = false;  // trained ARD100 1280 global engine
            pcfg.own_engine_path.clear();
            pcfg.conf_thresh = cfg.conf_thresh = 0.25f;
        } else if (cfg.type == InputType::CAMERA) {
            pcfg.use_own_engine = false;
            pcfg.own_engine_path.clear();
            pcfg.conf_thresh = cfg.conf_thresh = 0.40f;
        }
        config_changed = true;
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();

    // 源输入
    if (cfg.type == InputType::CAMERA) {
        // 首次选择摄像头时自动枚举设备
        if (!cameras_enumerated_) {
            cameras_ = V4L2Capture::enumerateCameras();
            cameras_enumerated_ = true;
        }

        if (cameras_.empty()) {
            // Reset USB 宽限期内：设备节点尚未回归，显示 Resetting… 并每帧重试枚举
            bool resetting = std::chrono::steady_clock::now() < reset_pending_until_[id];
            if (resetting) {
                cameras_ = V4L2Capture::enumerateCameras();  // 重试
                if (cameras_.empty())
                    ImGui::TextColored(ImVec4(1, 0.8f, 0.2f, 1), "Resetting...");
            }
            if (cameras_.empty()) {
                if (!resetting)
                    ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "No camera found");
                ImGui::SameLine();
                char rid[16];
                snprintf(rid, sizeof(rid), "Rescan##%d", id);
                if (ImGui::SmallButton(rid)) {
                    cameras_ = V4L2Capture::enumerateCameras();
                }
            }
        }
        if (!cameras_.empty()) {
            // 构建设备列表字符串
            std::string cam_items;
            int cur_idx = 0;
            int cam_id = 0;
            try { if (!cfg.source.empty()) cam_id = std::stoi(cfg.source); } catch (...) {}
            for (size_t ci = 0; ci < cameras_.size(); ci++) {
                char label[128];
                snprintf(label, sizeof(label), "%s (%s)",
                         cameras_[ci].device_path.c_str(),
                         cameras_[ci].card_name.c_str());
                cam_items += label;
                cam_items += '\0';
                if (cameras_[ci].device_id == cam_id) cur_idx = (int)ci;
            }
            cam_items += '\0';

            char cid[16];
            snprintf(cid, sizeof(cid), "##c%d", id);
            ImGui::PushItemWidth(180);
            if (ImGui::Combo(cid, &cur_idx, cam_items.c_str())) {
                cfg.source = std::to_string(cameras_[cur_idx].device_id);
                pcfg.cam_caps_queried = false;
                config_changed = true;
            }
            ImGui::PopItemWidth();
            ImGui::SameLine();

            // 查询选中摄像头的能力
            int sel_id = cameras_[cur_idx].device_id;
            char qid[16];
            snprintf(qid, sizeof(qid), "Query##%d", id);
            if (ImGui::SmallButton(qid)) {
                pcfg.cam_caps = V4L2Capture::queryCapabilities(sel_id);
                pcfg.cam_caps_queried = true;
                pcfg.cam_format_idx = 0;
                pcfg.cam_resolution_idx = 0;
                pcfg.cam_fps_idx = 0;
                LOG_INFO("Camera %d: %s, %zu formats",
                         sel_id, pcfg.cam_caps.card_name.c_str(),
                         pcfg.cam_caps.formats.size());
            }
        }

        // 显示摄像头能力选择
        if (pcfg.cam_caps_queried && !pcfg.cam_caps.empty()) {
            auto& fmts = pcfg.cam_caps.formats;

            // 构建格式+分辨率组合字符串（null 分隔）
            std::string format_items;
            for (size_t fi = 0; fi < fmts.size(); fi++) {
                char label[128];
                snprintf(label, sizeof(label), "%s %dx%d",
                         fmts[fi].name.c_str(), fmts[fi].width, fmts[fi].height);
                format_items += label;
                format_items += '\0';
            }
            format_items += '\0';

            // 格式+分辨率选择
            char frid[16];
            snprintf(frid, sizeof(frid), "##fr%d", id);
            ImGui::PushItemWidth(160);
            if (ImGui::Combo(frid, &pcfg.cam_resolution_idx, format_items.c_str())) {
                pcfg.cam_fps_idx = 0;
                config_changed = true;
            }
            ImGui::PopItemWidth();
            ImGui::SameLine();

            // 帧率选择
            if (pcfg.cam_resolution_idx < (int)fmts.size()) {
                auto& fps_opts = fmts[pcfg.cam_resolution_idx].fps_options;
                std::string fps_items;
                for (auto f : fps_opts) {
                    char label[32];
                    snprintf(label, sizeof(label), "%.0f fps", f);
                    fps_items += label;
                    fps_items += '\0';
                }
                fps_items += '\0';

                char fsid[16];
                snprintf(fsid, sizeof(fsid), "##fs%d", id);
                ImGui::PushItemWidth(80);
                if (ImGui::Combo(fsid, &pcfg.cam_fps_idx, fps_items.c_str())) {
                    config_changed = true;
                }
                ImGui::PopItemWidth();

                // 同步到 InputConfig
                if (config_changed && pcfg.cam_resolution_idx < (int)fmts.size()) {
                    auto& sel = fmts[pcfg.cam_resolution_idx];
                    cfg.cam_width = sel.width;
                    cfg.cam_height = sel.height;
                    cfg.cam_format = sel.pixelformat;
                    if (pcfg.cam_fps_idx < (int)sel.fps_options.size()) {
                        cfg.cam_fps = sel.fps_options[pcfg.cam_fps_idx];
                    }
                }
            }
        }
    }
    else if (cfg.type == InputType::VIDEO) {
        char sbuf[256];
        strncpy(sbuf, cfg.source.c_str(), sizeof(sbuf));
        char sid[16];
        snprintf(sid, sizeof(sid), "##s%d", id);
        ImGui::PushItemWidth(120);
        if (ImGui::InputText(sid, sbuf, sizeof(sbuf))) {
            cfg.source = sbuf;
            config_changed = true;
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        char bid[16];
        snprintf(bid, sizeof(bid), "Browse##b%d", id);
        if (ImGui::Button(bid)) {
            auto p = AppGUI::browse_file("Select Video", {"mp4","MP4","avi","AVI","mkv","MKV","mov","MOV"});
            if (!p.empty()) { cfg.source = p; config_changed = true; }
        }
    }
#if HAS_LIBUVC
    else if (cfg.type == InputType::UVC) {
        // 首次选择 UVC 时自动枚举设备
        if (!uvc_cameras_enumerated_) {
            uvc_cameras_ = UvcCapture::enumerateCameras();
            uvc_cameras_enumerated_ = true;
        }

        if (uvc_cameras_.empty()) {
            ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "No UVC device found");
            ImGui::SameLine();
            char rid[16];
            snprintf(rid, sizeof(rid), "Rescan##uvc%d", id);
            if (ImGui::SmallButton(rid)) {
                uvc_cameras_ = UvcCapture::enumerateCameras();
            }
        } else {
            // 构建设备列表字符串
            std::string uvc_items;
            int cur_idx = 0;
            int cam_id = 0;
            try { if (!cfg.source.empty()) cam_id = std::stoi(cfg.source); } catch (...) {}
            for (size_t ci = 0; ci < uvc_cameras_.size(); ci++) {
                char label[128];
                snprintf(label, sizeof(label), "%s (%s)",
                         uvc_cameras_[ci].device_path.c_str(),
                         uvc_cameras_[ci].card_name.c_str());
                uvc_items += label;
                uvc_items += '\0';
                if (uvc_cameras_[ci].device_id == cam_id) cur_idx = (int)ci;
            }
            uvc_items += '\0';

            char cid[16];
            snprintf(cid, sizeof(cid), "##uc%d", id);
            ImGui::PushItemWidth(180);
            if (ImGui::Combo(cid, &cur_idx, uvc_items.c_str())) {
                cfg.source = std::to_string(uvc_cameras_[cur_idx].device_id);
                config_changed = true;
            }
            ImGui::PopItemWidth();
        }
    }
#endif

#if 0  // [HIDDEN] Per-channel detect mode combo — 使用全局 toolbar 切换
    // 检测模式
    const char* mode_names[] = {"Off", "Direct", "SAHI"};
    int mode_idx = (int)cfg.detect_mode;
    if (mode_idx > 2) mode_idx = 0;
    char mid[16];
    snprintf(mid, sizeof(mid), "##m%d", id);
    ImGui::PushItemWidth(80);
    if (ImGui::Combo(mid, &mode_idx, mode_names, 3)) {
        cfg.detect_mode = (DetectMode)mode_idx;
        config_changed = true;
    }
    ImGui::PopItemWidth();
#endif  // [HIDDEN] Per-channel detect mode

    // === Reset USB：停止并重启该路 pipeline（重新打开 V4L2 设备）===
    if (cfg.type == InputType::CAMERA) {
        ImGui::SameLine();
        char rst_id[24];
        snprintf(rst_id, sizeof(rst_id), "Reset USB##%d", id);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.75f, 0.5f, 0.1f, 1));
        if (ImGui::SmallButton(rst_id)) {
            bool was_running = mgr.get_result(id).has_frame;
            int dev_id = 0;
            try { if (!cfg.source.empty()) dev_id = std::stoi(cfg.source); } catch (...) {}
            mgr.stop_pipeline(id);                          // 1. 释放设备占用
            bool ok = V4L2Capture::resetUsbDevice(dev_id);  // 2. USB 总线级硬复位
            pcfg.cam_caps_queried = false;
            cameras_enumerated_ = false;                    // 3. 强制重新枚举
            // 复位后内核重新枚举需数百ms，设 1.5s 宽限期，期间显示 Resetting… 并重试
            reset_pending_until_[id] = std::chrono::steady_clock::now() +
                                       std::chrono::milliseconds(1500);
            LOG_INFO("CH %d: USB hard reset %s", id + 1, ok ? "OK" : "FAILED");
            if (was_running && cfg.type != InputType::NONE) {
                // 4. 重新启动（复位后内核重新枚举需数百ms，若失败可再点 START）
                if (pcfg.use_own_engine && !pcfg.own_engine_path.empty())
                    mgr.start_pipeline_with_engine(id, cfg, pcfg.own_engine_path);
                else
                    mgr.start_pipeline(id, cfg);
            }
        }
        ImGui::PopStyleColor();
    }

    // === Conf + NMS（接同一行）===
    ImGui::SameLine();
    ImGui::PushItemWidth(70);
    char confid[16], nmsid[16];
    snprintf(confid, sizeof(confid), "##conf%d", id);
    snprintf(nmsid, sizeof(nmsid), "##nms%d", id);
    ImGui::Text("Conf");
    ImGui::SameLine();
    if (ImGui::SliderFloat(confid, &pcfg.conf_thresh, 0.01f, 0.99f, "%.2f"))
        config_changed = true;
    ImGui::SameLine();
    ImGui::Text("NMS");
    ImGui::SameLine();
    if (ImGui::SliderFloat(nmsid, &pcfg.nms_thresh, 0.01f, 0.99f, "%.2f"))
        config_changed = true;
    ImGui::PopItemWidth();

#if 0  // [HIDDEN] SAHI params
    // === SAHI 参数（仅 SAHI 模式显示） ===
    if (cfg.detect_mode == DetectMode::SAHI) {
        ImGui::PushItemWidth(80);
        char ssid[16], sstid[16], sintid[16];
        snprintf(ssid, sizeof(ssid), "##ssid%d", id);
        snprintf(sstid, sizeof(sstid), "##sstid%d", id);
        snprintf(sintid, sizeof(sintid), "##sintid%d", id);

        ImGui::Text("Slice");
        ImGui::SameLine();
        if (ImGui::SliderInt(ssid, &pcfg.sahi_slice, 320, 1280))
            config_changed = true;
        ImGui::SameLine();
        ImGui::Text("Stride");
        ImGui::SameLine();
        if (ImGui::SliderInt(sstid, &pcfg.sahi_stride, 160, pcfg.sahi_slice))
            config_changed = true;
        ImGui::SameLine();
        ImGui::Text("Interval");
        ImGui::SameLine();
        if (ImGui::SliderInt(sintid, &pcfg.sahi_interval, 1, 30))
            config_changed = true;
        ImGui::PopItemWidth();
    }
#endif
#if 0  // [HIDDEN] Video target FPS
    // === 视频文件帧率限制 ===
    if (cfg.type == InputType::VIDEO) {
        ImGui::PushItemWidth(90);
        char tfps_id[16];
        snprintf(tfps_id, sizeof(tfps_id), "##tfps%d", id);
        // 用滑块：0=不限制, 30/60/120/200/300
        int fps_idx = cfg.target_fps < 0 ? 0 :
                      cfg.target_fps <= 0 ? 1 :
                      cfg.target_fps <= 30 ? 2 :
                      cfg.target_fps <= 60 ? 3 :
                      cfg.target_fps <= 120 ? 4 :
                      cfg.target_fps <= 200 ? 5 : 6;
        const char* fps_labels[] = {"Max", "Native", "30", "60", "120", "200", "300"};
        float fps_values[] = {-1, 0, 30, 60, 120, 200, 300};
        if (ImGui::Combo(tfps_id, &fps_idx, fps_labels, 7)) {
            cfg.target_fps = fps_values[fps_idx];
            config_changed = true;
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        ImGui::TextDisabled("Target FPS");
    }
#endif
#if 0  // [HIDDEN] Own Engine
    // === 行3: Own Engine (可选) ===
    ImGui::Checkbox("Own Engine", &pcfg.use_own_engine);
    if (pcfg.use_own_engine) {
        ImGui::SameLine();
        char eid[16];
        snprintf(eid, sizeof(eid), "##e%d", id);
        char ebuf[256];
        strncpy(ebuf, pcfg.own_engine_path.c_str(), sizeof(ebuf));
        ImGui::PushItemWidth(120);
        if (ImGui::InputText(eid, ebuf, sizeof(ebuf)))
            pcfg.own_engine_path = ebuf;
        ImGui::PopItemWidth();
        ImGui::SameLine();
        char ebid[16];
        snprintf(ebid, sizeof(ebid), "##eb%d", id);
        if (ImGui::Button(ebid)) {
            auto p = AppGUI::browse_file("Select Engine", {"trt","engine"});
            if (!p.empty()) pcfg.own_engine_path = p;
        }
    }
#endif

    // 仅在参数变化时同步到 pipeline
    if (config_changed) {
        cfg.conf_thresh = pcfg.conf_thresh;
        cfg.nms_thresh = pcfg.nms_thresh;
        cfg.sahi_slice = pcfg.sahi_slice;
        cfg.sahi_stride = pcfg.sahi_stride;
        cfg.sahi_interval = pcfg.sahi_interval;
        mgr.update_pipeline(id, cfg);
    }

    // === START/STOP + 状态（接同一行）===
    auto result = mgr.get_result(id);
    bool is_running = result.has_frame;

    char btn_label[32];
    snprintf(btn_label, sizeof(btn_label), "%s##%d", is_running ? "STOP" : "START", id);

    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Button,
                          is_running ? ImVec4(0.65f,0.15f,0.15f,1) : ImVec4(0.15f,0.55f,0.15f,1));
    if (ImGui::Button(btn_label, ImVec2(70, 24))) {
        if (is_running) {
            mgr.stop_pipeline(id);
            LOG_INFO("Pipeline %d stopped", id);
        } else if (cfg.type != InputType::NONE) {
            if (pcfg.use_own_engine && !pcfg.own_engine_path.empty()) {
                mgr.start_pipeline_with_engine(id, cfg, pcfg.own_engine_path);
            } else {
                mgr.start_pipeline(id, cfg);
            }
        }
    }
    ImGui::PopStyleColor();

    // 状态信息
    if (is_running) {
        ImGui::SameLine();
        if (cfg.detect_mode == DetectMode::OFF) {
            // View 模式：只有显示/采集帧率
            if (result.cam_fps > 0)
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1), "%.0f FPS", result.cam_fps);
            else
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1), "%.0f FPS", result.fps);
        } else if (cfg.detect_mode == DetectMode::SAHI) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1), "%.0f FPS | %d obj | SAHI%s",
                               result.fps, result.num_objects,
                               result.sahi_cached ? " (cached)" : "");
        } else {
            // Detect 模式：分别显示相机帧率 + 推理帧率
            if (result.cam_fps > 0)
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1), "Cam %.0f | Infer %.0f FPS | %d obj",
                                   result.cam_fps, result.fps, result.num_objects);
            else
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1), "Infer %.0f FPS | %d obj",
                                   result.fps, result.num_objects);
        }
#if 0  // [HIDDEN] Timing bar + tooltip
        // 分时统计条
        if (result.ms_total > 0) {
            float bar_w = 180.0f;
            float max_ms = result.ms_total;
            // Decode (蓝)
            float dec_w = bar_w * (result.ms_decode / max_ms);
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.3f,0.5f,1.0f,1));
            ImGui::SameLine();
            ImGui::ProgressBar(result.ms_decode / max_ms, ImVec2(dec_w, 10), "");
            ImGui::PopStyleColor();
            // Preproc (黄)
            if (result.ms_preproc > 0) {
                ImGui::SameLine(0, 0);
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(1.0f,0.8f,0.2f,1));
                ImGui::ProgressBar(result.ms_preproc / max_ms, ImVec2(bar_w * (result.ms_preproc / max_ms), 10), "");
                ImGui::PopStyleColor();
            }
            // 分时数字
            ImGui::SameLine();
            ImGui::TextDisabled("%.1fms", result.ms_total);
            // 详细数字（鼠标悬停提示）
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::TextColored(ImVec4(0.3f,0.5f,1,1),   "Decode:    %.2f ms", result.ms_decode);
                ImGui::TextColored(ImVec4(1,0.8f,0.2f,1),   "Preproc:   %.2f ms", result.ms_preproc);
                ImGui::TextColored(ImVec4(0.3f,1,0.3f,1),   "Infer:     %.2f ms", result.ms_infer);
                ImGui::TextColored(ImVec4(0.8f,0.4f,1,1),   "Postproc:  %.2f ms", result.ms_postproc);
                ImGui::TextColored(ImVec4(1,0.5f,0.3f,1),   "Display:   %.2f ms", result.ms_display);
                ImGui::Separator();
                ImGui::Text("Total:     %.2f ms → %.0f fps", result.ms_total,
                            result.ms_total > 0 ? 1000.0f / result.ms_total : 0);
                ImGui::EndTooltip();
            }
        }
#endif
    } else {
        ImGui::SameLine();
        ImGui::TextDisabled("Stopped");
    }

    // === 录制控制（全分辨率 H.264，接同一行）===
    if (is_running) {
        ImGui::SameLine();
        auto* hw_rec = mgr.hw_recorder(id);
        if (hw_rec && hw_rec->isRecording()) {
            // 录制中：显示红点 + 时长 + 停止按钮
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.2f, 0.2f, 1.0f));
            ImGui::Text("\xe2\x97\x89 REC");
            ImGui::PopStyleColor();
            ImGui::SameLine();
            float elapsed = hw_rec->recordedSeconds();
            int mins = (int)elapsed / 60;
            int secs = (int)elapsed % 60;
            ImGui::Text("%02d:%02d", mins, secs);
            ImGui::SameLine();
            char stop_id[16];
            snprintf(stop_id, sizeof(stop_id), "Stop##rec%d", id);
            if (ImGui::SmallButton(stop_id)) {
                hw_rec->stop();
            }
            ImGui::SameLine();
            float rec_secs = hw_rec->recordedSeconds();
            float wfps = rec_secs > 0.5f ? hw_rec->framesWritten() / rec_secs : 0.0f;
            ImGui::TextDisabled("%d frames (%.1f fps)", hw_rec->framesWritten(), wfps);
            ImGui::SameLine();
            ImGui::TextDisabled("%s", hw_rec->filePath().c_str());
        } else {
            // 未录制：显示录制按钮 + 时长选择
            char rec_id[16];
            snprintf(rec_id, sizeof(rec_id), "Record##%d", id);
            if (ImGui::SmallButton(rec_id)) {
                auto now = std::chrono::system_clock::now();
                auto t = std::chrono::system_clock::to_time_t(now);
                struct tm tm_buf;
                localtime_r(&t, &tm_buf);
                char path[256];
                snprintf(path, sizeof(path), "%s/rec_ch%d_%04d%02d%02d_%02d%02d%02d.mp4",
                         std::getenv("HOME") ? std::getenv("HOME") : "/tmp",
                         id + 1,
                         tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
                         tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
                if (hw_rec) {
                    hw_rec->setDurationLimit(pcfg.record_duration);
                    int rw = result.src_w > 0 ? result.src_w : result.display_w;
                    int rh = result.src_h > 0 ? result.src_h : result.display_h;
                    // 录制帧率：优先相机实测帧率，其次处理帧率（不再限 60，保证回放速度正确）
                    float rec_fps = result.cam_fps > 0 ? result.cam_fps :
                                    (result.fps > 0 ? result.fps : 30.0f);
                    if (hw_rec->start(path, rw, rh, rec_fps)) {
                        LOG_INFO("Recording started: %s (%dx%d @ %.0f fps)", path, rw, rh, rec_fps);
                    }
                }
            }
            ImGui::SameLine();
            // 时长选择
            char dur_id[16];
            snprintf(dur_id, sizeof(dur_id), "##dur%d", id);
            const char* dur_names[] = {"30s", "1min", "2min"};
            float dur_values[] = {30.0f, 60.0f, 120.0f};
            int dur_idx = 0;
            for (int d = 0; d < 3; d++) {
                if (pcfg.record_duration == dur_values[d]) dur_idx = d;
            }
            ImGui::PushItemWidth(60);
            if (ImGui::Combo(dur_id, &dur_idx, dur_names, 3)) {
                pcfg.record_duration = dur_values[dur_idx];
            }
            ImGui::PopItemWidth();
        }
    }

    // === 快照（Snapshot）===
    if (is_running) {
        ImGui::SameLine();
        static int snap_fmt[3] = {0, 0, 0};  // 0=JPEG, 1=PNG, 2=RAW
        char sfmt_id[16];
        snprintf(sfmt_id, sizeof(sfmt_id), "##sfmt%d", id);
        ImGui::SetNextItemWidth(72);
        ImGui::Combo(sfmt_id, &snap_fmt[id], "JPEG\0PNG\0RAW\0");
        ImGui::SameLine();
        char snap_id[16];
        snprintf(snap_id, sizeof(snap_id), "Snapshot##%d", id);
        if (ImGui::SmallButton(snap_id)) {
            auto* pl = mgr.pipeline(id);
            if (pl) {
                cv::Mat raw = pl->snapshotRaw();
                if (!raw.empty()) {
                    auto now = std::chrono::system_clock::now();
                    auto t = std::chrono::system_clock::to_time_t(now);
                    struct tm tm_buf;
                    localtime_r(&t, &tm_buf);
                    const char* exts[] = {"jpg", "png", "raw"};
                    char path[256];
                    snprintf(path, sizeof(path), "%s/snap_ch%d_%04d%02d%02d_%02d%02d%02d.%s",
                             std::getenv("HOME") ? std::getenv("HOME") : "/tmp",
                             id + 1,
                             tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
                             tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec,
                             exts[snap_fmt[id]]);
                    if (snap_fmt[id] == 2) {
                        FILE* f = fopen(path, "wb");
                        if (f) { fwrite(raw.data, 1, raw.total() * raw.elemSize(), f); fclose(f); }
                    } else {
                        cv::Mat bgr;
                        if (pl->is_grey_cam()) cv::cvtColor(raw, bgr, cv::COLOR_GRAY2BGR);
                        else cv::cvtColor(raw, bgr, cv::COLOR_YUV2BGR_YUYV);
                        cv::imwrite(path, bgr);
                    }
                    LOG_INFO("Snapshot saved: %s", path);
                }
            }
        }
    }
}

// 视频左上角的半透明精简配置浮层（纵向布局，仅保留当前可见控件）
void AppGUI::render_input_overlay(int id, PipelineManager& mgr) {
    if (id < 0 || id >= 3) return;
    InputConfig& cfg = state_.inputs[id];
    PipelineGUIConfig& pcfg = state_.pipeline_configs[id];
    bool config_changed = false;

    if (state_.demo_mode) {
        auto result = mgr.get_result(id);
        const bool running = result.has_frame;
        const char* source_name = cfg.type == InputType::CAMERA ? "CAMERA" :
                                  cfg.type == InputType::VIDEO ? "VIDEO" :
                                  cfg.type == InputType::UVC ? "UVC" : "NO SOURCE";
        ImGui::TextColored(ImVec4(0.35f, 0.82f, 1.0f, 1), "CHANNEL %d", id + 1);
        ImGui::SameLine();
        ImGui::TextDisabled("%s", source_name);
        if (running) {
            ImGui::TextColored(ImVec4(0.30f,0.92f,0.50f,1), "● LIVE");
            ImGui::SameLine();
            ImGui::Text("%.0f FPS", result.cam_fps > 0 ? result.cam_fps : result.fps);
            ImGui::SameLine();
            ImGui::Text("  Objects  %d", result.num_objects);
        } else {
            ImGui::TextDisabled("○ STANDBY");
        }

        // Draggable video timeline. Seeking is executed by the decoder thread,
        // so dragging never calls FFmpeg concurrently from the GUI thread.
        if (cfg.type == InputType::VIDEO && running) {
            Pipeline* pipeline = mgr.pipeline(id);
            if (pipeline && pipeline->video_duration() > 0.0) {
                static float seek_value[3] = {0, 0, 0};
                static bool dragging[3] = {false, false, false};
                float duration = (float)pipeline->video_duration();
                if (!dragging[id]) seek_value[id] = (float)pipeline->video_position();
                char seek_id[24];
                std::snprintf(seek_id, sizeof(seek_id), "##timeline%d", id);
                ImGui::PushItemWidth(250);
                ImGui::SliderFloat(seek_id, &seek_value[id], 0.0f, duration, "");
                ImGui::PopItemWidth();
                if (ImGui::IsItemActivated()) dragging[id] = true;
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                    pipeline->seek_video(seek_value[id]);
                    dragging[id] = false;
                }
                ImGui::SameLine();
                int cur = (int)seek_value[id], total = (int)duration;
                ImGui::Text("%02d:%02d / %02d:%02d",
                            cur / 60, cur % 60, total / 60, total % 60);
            }
        }

        if (cfg.type == InputType::VIDEO && !running) {
            if (ImGui::Button("Select video")) {
                auto p = AppGUI::browse_file("Select Video", {"mp4","MP4","avi","AVI","mkv","MKV","mov","MOV"});
                if (!p.empty()) cfg.source = p;
            }
        }

        // Each channel may use its own TensorRT engine. Changing the model on
        // a live channel restarts only that channel; other inputs keep running.
        const std::string& active_model =
            (pcfg.use_own_engine && !pcfg.own_engine_path.empty())
                ? pcfg.own_engine_path : state_.engine_path;
        size_t model_sep = active_model.find_last_of("/\\");
        const char* model_name = active_model.c_str() +
            (model_sep == std::string::npos ? 0 : model_sep + 1);
        ImGui::TextDisabled("Model: %.26s", model_name);
        ImGui::SameLine();
        char model_btn[32];
        std::snprintf(model_btn, sizeof(model_btn), "Choose##model%d", id);
        if (ImGui::SmallButton(model_btn)) {
            auto path = AppGUI::browse_file("Select channel TensorRT engine", {"engine", "trt"});
            if (!path.empty()) {
                const bool restart = mgr.get_result(id).has_frame;
                if (restart) mgr.stop_pipeline(id);
                pcfg.use_own_engine = true;
                pcfg.own_engine_path = path;
                if (restart) mgr.start_pipeline_with_engine(id, cfg, path);
                LOG_INFO("CH %d: selected independent model: %s", id + 1, path.c_str());
            }
        }
        if (pcfg.use_own_engine) {
            ImGui::SameLine();
            char global_btn[32];
            std::snprintf(global_btn, sizeof(global_btn), "Global##model%d", id);
            if (ImGui::SmallButton(global_btn)) {
                const bool restart = mgr.get_result(id).has_frame;
                if (restart) mgr.stop_pipeline(id);
                pcfg.use_own_engine = false;
                pcfg.own_engine_path.clear();
                if (restart) mgr.start_pipeline(id, cfg);
                LOG_INFO("CH %d: switched to global model: %s", id + 1,
                         state_.engine_path.c_str());
            }
        }
        ImGui::PushItemWidth(120);
        if (ImGui::SliderFloat("Confidence", &pcfg.conf_thresh, 0.05f, 0.90f, "%.2f")) {
            cfg.conf_thresh = pcfg.conf_thresh;
            mgr.update_pipeline(id, cfg);
        }
        ImGui::PopItemWidth();
        return;
    }

    // === 标题：CH + Type ===
#if HAS_LIBUVC
    const char* type_names[] = {"Off", "Camera", "Video", "UVC"};
    int type_count = 4;
#else
    const char* type_names[] = {"Off", "Camera", "Video"};
    int type_count = 3;
#endif
    int type_idx = (int)cfg.type;
    if (type_idx >= type_count) type_idx = 0;

    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1), "CH%d", id + 1);
    ImGui::SameLine();
    char tid[16];
    snprintf(tid, sizeof(tid), "##ovt%d", id);
    ImGui::PushItemWidth(80);
    if (ImGui::Combo(tid, &type_idx, type_names, type_count)) {
        if (mgr.get_result(id).has_frame) {
            mgr.stop_pipeline(id);
            LOG_INFO("CH %d: auto-stopped on type change", id + 1);
        }
        cfg.type = (InputType)type_idx;
        cfg.source.clear();
        if (cfg.type == InputType::VIDEO) {
            pcfg.use_own_engine = false;
            pcfg.own_engine_path.clear();
            pcfg.conf_thresh = cfg.conf_thresh = 0.25f;
        } else if (cfg.type == InputType::CAMERA) {
            pcfg.use_own_engine = false;
            pcfg.own_engine_path.clear();
            pcfg.conf_thresh = cfg.conf_thresh = 0.40f;
        }
        config_changed = true;
    }
    ImGui::PopItemWidth();

    // === 源选择 ===
    if (cfg.type == InputType::CAMERA) {
        if (!cameras_enumerated_) {
            cameras_ = V4L2Capture::enumerateCameras();
            cameras_enumerated_ = true;
        }
        if (cameras_.empty()) {
            bool resetting = std::chrono::steady_clock::now() < reset_pending_until_[id];
            if (resetting) {
                cameras_ = V4L2Capture::enumerateCameras();  // 重试
                if (cameras_.empty())
                    ImGui::TextColored(ImVec4(1, 0.8f, 0.2f, 1), "Resetting...");
            }
            if (cameras_.empty()) {
                if (!resetting)
                    ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "No camera");
                ImGui::SameLine();
                char rid[16];
                snprintf(rid, sizeof(rid), "Rescan##ov%d", id);
                if (ImGui::SmallButton(rid)) cameras_ = V4L2Capture::enumerateCameras();
            }
        }
        if (!cameras_.empty()) {
            std::string cam_items;
            int cur_idx = 0, cam_id = 0;
            try { if (!cfg.source.empty()) cam_id = std::stoi(cfg.source); } catch (...) {}
            for (size_t ci = 0; ci < cameras_.size(); ci++) {
                char label[128];
                snprintf(label, sizeof(label), "%s (%s)",
                         cameras_[ci].device_path.c_str(), cameras_[ci].card_name.c_str());
                cam_items += label; cam_items += '\0';
                if (cameras_[ci].device_id == cam_id) cur_idx = (int)ci;
            }
            cam_items += '\0';
            char cid[16];
            snprintf(cid, sizeof(cid), "##ovc%d", id);
            ImGui::PushItemWidth(180);
            if (ImGui::Combo(cid, &cur_idx, cam_items.c_str())) {
                cfg.source = std::to_string(cameras_[cur_idx].device_id);
                pcfg.cam_caps_queried = false;
                config_changed = true;
            }
            ImGui::PopItemWidth();
            ImGui::SameLine();
            int sel_id = cameras_[cur_idx].device_id;
            char qid[16];
            snprintf(qid, sizeof(qid), "Query##ov%d", id);
            if (ImGui::SmallButton(qid)) {
                pcfg.cam_caps = V4L2Capture::queryCapabilities(sel_id);
                pcfg.cam_caps_queried = true;
                pcfg.cam_format_idx = 0;
                pcfg.cam_resolution_idx = 0;
                pcfg.cam_fps_idx = 0;
            }
            // 格式/分辨率 + 帧率
            if (pcfg.cam_caps_queried && !pcfg.cam_caps.empty()) {
                auto& fmts = pcfg.cam_caps.formats;
                std::string format_items;
                for (size_t fi = 0; fi < fmts.size(); fi++) {
                    char label[128];
                    snprintf(label, sizeof(label), "%s %dx%d",
                             fmts[fi].name.c_str(), fmts[fi].width, fmts[fi].height);
                    format_items += label; format_items += '\0';
                }
                format_items += '\0';
                char frid[16];
                snprintf(frid, sizeof(frid), "##ovfr%d", id);
                ImGui::PushItemWidth(160);
                if (ImGui::Combo(frid, &pcfg.cam_resolution_idx, format_items.c_str())) {
                    pcfg.cam_fps_idx = 0;
                    config_changed = true;
                }
                ImGui::PopItemWidth();
                if (pcfg.cam_resolution_idx < (int)fmts.size()) {
                    auto& fps_opts = fmts[pcfg.cam_resolution_idx].fps_options;
                    std::string fps_items;
                    for (auto f : fps_opts) {
                        char label[32];
                        snprintf(label, sizeof(label), "%.0f fps", f);
                        fps_items += label; fps_items += '\0';
                    }
                    fps_items += '\0';
                    char fsid[16];
                    snprintf(fsid, sizeof(fsid), "##ovfs%d", id);
                    ImGui::SameLine();
                    ImGui::PushItemWidth(80);
                    if (ImGui::Combo(fsid, &pcfg.cam_fps_idx, fps_items.c_str()))
                        config_changed = true;
                    ImGui::PopItemWidth();
                    if (config_changed && pcfg.cam_resolution_idx < (int)fmts.size()) {
                        auto& sel = fmts[pcfg.cam_resolution_idx];
                        cfg.cam_width = sel.width;
                        cfg.cam_height = sel.height;
                        cfg.cam_format = sel.pixelformat;
                        if (pcfg.cam_fps_idx < (int)sel.fps_options.size())
                            cfg.cam_fps = sel.fps_options[pcfg.cam_fps_idx];
                    }
                }
            }
            // Reset USB
            ImGui::SameLine();
            char rst_id[24];
            snprintf(rst_id, sizeof(rst_id), "Reset USB##ov%d", id);
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.75f, 0.5f, 0.1f, 1));
            if (ImGui::SmallButton(rst_id)) {
                bool was_running = mgr.get_result(id).has_frame;
                int dev_id = 0;
                try { if (!cfg.source.empty()) dev_id = std::stoi(cfg.source); } catch (...) {}
                mgr.stop_pipeline(id);
                V4L2Capture::resetUsbDevice(dev_id);
                pcfg.cam_caps_queried = false;
                cameras_enumerated_ = false;
                reset_pending_until_[id] = std::chrono::steady_clock::now() +
                                           std::chrono::milliseconds(1500);
                if (was_running && cfg.type != InputType::NONE) {
                    if (pcfg.use_own_engine && !pcfg.own_engine_path.empty())
                        mgr.start_pipeline_with_engine(id, cfg, pcfg.own_engine_path);
                    else
                        mgr.start_pipeline(id, cfg);
                }
            }
            ImGui::PopStyleColor();
        }
    }
    else if (cfg.type == InputType::VIDEO) {
        char sbuf[256];
        strncpy(sbuf, cfg.source.c_str(), sizeof(sbuf));
        char sid[16];
        snprintf(sid, sizeof(sid), "##ovs%d", id);
        ImGui::PushItemWidth(160);
        if (ImGui::InputText(sid, sbuf, sizeof(sbuf))) {
            cfg.source = sbuf;
            config_changed = true;
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        char bid[16];
        snprintf(bid, sizeof(bid), "Browse##ovb%d", id);
        if (ImGui::SmallButton(bid)) {
            auto p = AppGUI::browse_file("Select Video", {"mp4","MP4","avi","AVI","mkv","MKV","mov","MOV"});
            if (!p.empty()) { cfg.source = p; config_changed = true; }
        }
    }
#if HAS_LIBUVC
    else if (cfg.type == InputType::UVC) {
        if (!uvc_cameras_enumerated_) {
            uvc_cameras_ = UvcCapture::enumerateCameras();
            uvc_cameras_enumerated_ = true;
        }
        if (uvc_cameras_.empty()) {
            ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "No UVC device");
            ImGui::SameLine();
            char rid[16];
            snprintf(rid, sizeof(rid), "Rescan##ovuvc%d", id);
            if (ImGui::SmallButton(rid)) uvc_cameras_ = UvcCapture::enumerateCameras();
        } else {
            std::string uvc_items;
            int cur_idx = 0, cam_id = 0;
            try { if (!cfg.source.empty()) cam_id = std::stoi(cfg.source); } catch (...) {}
            for (size_t ci = 0; ci < uvc_cameras_.size(); ci++) {
                char label[128];
                snprintf(label, sizeof(label), "%s (%s)",
                         uvc_cameras_[ci].device_path.c_str(), uvc_cameras_[ci].card_name.c_str());
                uvc_items += label; uvc_items += '\0';
                if (uvc_cameras_[ci].device_id == cam_id) cur_idx = (int)ci;
            }
            uvc_items += '\0';
            char cid[16];
            snprintf(cid, sizeof(cid), "##ovuvccfg%d", id);
            ImGui::PushItemWidth(180);
            if (ImGui::Combo(cid, &cur_idx, uvc_items.c_str())) {
                cfg.source = std::to_string(uvc_cameras_[cur_idx].device_id);
                config_changed = true;
            }
            ImGui::PopItemWidth();
        }
    }
#endif

    // === Conf + NMS ===
    ImGui::PushItemWidth(70);
    char confid[16], nmsid[16];
    snprintf(confid, sizeof(confid), "##ovconf%d", id);
    snprintf(nmsid, sizeof(nmsid), "##ovnms%d", id);
    ImGui::Text("Conf");
    ImGui::SameLine();
    if (ImGui::SliderFloat(confid, &pcfg.conf_thresh, 0.01f, 0.99f, "%.2f"))
        config_changed = true;
    ImGui::SameLine();
    ImGui::Text("NMS");
    ImGui::SameLine();
    if (ImGui::SliderFloat(nmsid, &pcfg.nms_thresh, 0.01f, 0.99f, "%.2f"))
        config_changed = true;
    ImGui::PopItemWidth();

    if (config_changed) {
        cfg.conf_thresh = pcfg.conf_thresh;
        cfg.nms_thresh = pcfg.nms_thresh;
        cfg.sahi_slice = pcfg.sahi_slice;
        cfg.sahi_stride = pcfg.sahi_stride;
        cfg.sahi_interval = pcfg.sahi_interval;
        mgr.update_pipeline(id, cfg);
    }

    // === START/STOP + 状态 ===
    auto result = mgr.get_result(id);
    bool is_running = result.has_frame;
    char btn_label[32];
    snprintf(btn_label, sizeof(btn_label), "%s##ov%d", is_running ? "STOP" : "START", id);
    ImGui::PushStyleColor(ImGuiCol_Button,
                          is_running ? ImVec4(0.65f,0.15f,0.15f,1) : ImVec4(0.15f,0.55f,0.15f,1));
    if (ImGui::Button(btn_label, ImVec2(70, 24))) {
        if (is_running) {
            mgr.stop_pipeline(id);
        } else if (cfg.type != InputType::NONE) {
            if (pcfg.use_own_engine && !pcfg.own_engine_path.empty())
                mgr.start_pipeline_with_engine(id, cfg, pcfg.own_engine_path);
            else
                mgr.start_pipeline(id, cfg);
        }
    }
    ImGui::PopStyleColor();

    if (is_running) {
        ImGui::SameLine();
        if (cfg.detect_mode == DetectMode::OFF) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1), "%.0f FPS",
                               result.cam_fps > 0 ? result.cam_fps : result.fps);
        } else if (cfg.detect_mode == DetectMode::SAHI) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1), "%.0f FPS | %d obj | SAHI%s",
                               result.fps, result.num_objects,
                               result.sahi_cached ? " (cached)" : "");
        } else {
            if (result.cam_fps > 0)
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1), "Cam %.0f | Infer %.0f | %d obj",
                                   result.cam_fps, result.fps, result.num_objects);
            else
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1), "Infer %.0f | %d obj",
                                   result.fps, result.num_objects);
        }
    } else {
        ImGui::SameLine();
        ImGui::TextDisabled("Stopped");
    }

    // === 录制 ===
    if (is_running) {
        auto* hw_rec = mgr.hw_recorder(id);
        if (hw_rec && hw_rec->isRecording()) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.2f, 0.2f, 1.0f));
            ImGui::Text("\xe2\x97\x89 REC");
            ImGui::PopStyleColor();
            ImGui::SameLine();
            float elapsed = hw_rec->recordedSeconds();
            ImGui::Text("%02d:%02d", (int)elapsed / 60, (int)elapsed % 60);
            ImGui::SameLine();
            char stop_id[16];
            snprintf(stop_id, sizeof(stop_id), "Stop##ovrec%d", id);
            if (ImGui::SmallButton(stop_id)) hw_rec->stop();
            ImGui::SameLine();
            float wfps = elapsed > 0.5f ? hw_rec->framesWritten() / elapsed : 0.0f;
            ImGui::TextColored(ImVec4(1, 1, 1, 0.85f), "%d frames (%.1f fps)",
                               hw_rec->framesWritten(), wfps);
        } else {
            char rec_id[16];
            snprintf(rec_id, sizeof(rec_id), "Record##ov%d", id);
            if (ImGui::SmallButton(rec_id)) {
                auto now = std::chrono::system_clock::now();
                auto t = std::chrono::system_clock::to_time_t(now);
                struct tm tm_buf;
                localtime_r(&t, &tm_buf);
                char path[256];
                snprintf(path, sizeof(path), "%s/rec_ch%d_%04d%02d%02d_%02d%02d%02d.mp4",
                         std::getenv("HOME") ? std::getenv("HOME") : "/tmp", id + 1,
                         tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
                         tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
                if (hw_rec) {
                    hw_rec->setDurationLimit(pcfg.record_duration);
                    int rw = result.src_w > 0 ? result.src_w : result.display_w;
                    int rh = result.src_h > 0 ? result.src_h : result.display_h;
                    float rec_fps = result.cam_fps > 0 ? result.cam_fps :
                                    (result.fps > 0 ? result.fps : 30.0f);
                    if (hw_rec->start(path, rw, rh, rec_fps))
                        LOG_INFO("Recording started: %s (%dx%d @ %.0f fps)", path, rw, rh, rec_fps);
                }
            }
            ImGui::SameLine();
            // 时长选择：30s / 1min / 2min
            char dur_id[16];
            snprintf(dur_id, sizeof(dur_id), "##ovdur%d", id);
            const char* dur_names[] = {"30s", "1min", "2min"};
            float dur_values[] = {30.0f, 60.0f, 120.0f};
            int dur_idx = 0;
            for (int d = 0; d < 3; d++) {
                if (pcfg.record_duration == dur_values[d]) dur_idx = d;
            }
            ImGui::PushItemWidth(60);
            if (ImGui::Combo(dur_id, &dur_idx, dur_names, 3)) {
                pcfg.record_duration = dur_values[dur_idx];
            }
            ImGui::PopItemWidth();
        }
    }
}

void AppGUI::setupSharedDisplay(int id, int width, int height) {
    if (id < 0 || id >= 3) return;
    if (shared_tex_failed_[id]) return;  // 之前创建失败，不再重试
    if (shared_tex_[id].gl_tex && shared_tex_[id].width == width && shared_tex_[id].height == height)
        return;  // 已创建，尺寸相同
    if (!createSharedTexture(id, width, height)) {
        shared_tex_failed_[id] = true;
        LOG_WARN("Shared texture %d creation failed, falling back to CPU path", id);
        return;
    }
    LOG_INFO("Shared display texture %d: %dx%d", id, width, height);
}

void AppGUI::render_video(int id, const PipelineResult& result) {
    if (id < 0 || id >= 3) return;

    if (!result.has_frame) {
        ImVec2 center = ImGui::GetContentRegionAvail();
        ImVec2 text_size = ImGui::CalcTextSize("No input");
        ImGui::SetCursorPos(ImVec2((center.x - text_size.x) * 0.5f,
                                   (center.y - text_size.y) * 0.5f));
        ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.45f, 1), "No input");
        return;
    }

    unsigned int display_tex = 0;
    int disp_w = 0, disp_h = 0;

    // GPU 直接路径：从 GPU 缓冲区拷贝到 GL 纹理（GPU→GPU，无 CPU 参与）
    if (result.display_gpu && result.display_w > 0 && result.display_h > 0) {
        setupSharedDisplay(id, result.display_w, result.display_h);
        if (shared_tex_[id].gl_tex) {
            // mapSharedTexture 映射 GL 资源，返回 staging 缓冲区指针
            uint8_t* staging = mapSharedTexture(id);
            if (staging) {
                size_t needed = (size_t)result.display_w * result.display_h * 3;
                // GPU→GPU 拷贝：pipeline 的 display_gpu_buf → staging buffer
                cudaMemcpy(staging, result.display_gpu, needed, cudaMemcpyDeviceToDevice);
                // staging buffer → GL 纹理（cudaMemcpy2DToArray + unmap）
                unmapSharedTexture(id);
            }
            display_tex = shared_tex_[id].gl_tex;
            disp_w = result.display_w;
            disp_h = result.display_h;
        } else {
            // 共享纹理不可用，回退到 CPU 路径：D2H + update_texture
            size_t buf_size = (size_t)result.display_w * result.display_h * 3;
            cv::Mat bgr_frame(result.display_h, result.display_w, CV_8UC3);
            cudaMemcpy(bgr_frame.data, result.display_gpu, buf_size, cudaMemcpyDeviceToHost);
            cv::cvtColor(bgr_frame, bgr_frame, cv::COLOR_RGB2BGR);
            // 检测框由 ImGui 叠加层统一绘制
            update_texture(id, bgr_frame);
            display_tex = textures_[id];
            disp_w = tex_width_[id];
            disp_h = tex_height_[id];
        }
    }

    // CPU 回退路径：从 cv::Mat 上传到 GL 纹理
    if (display_tex == 0 && !result.display.empty()) {
        update_texture(id, result.display);
        display_tex = textures_[id];
        disp_w = tex_width_[id];
        disp_h = tex_height_[id];
    }

    if (display_tex == 0) return;

    // 强制 16:9 显示矩形，居中填充可用区域
    ImVec2 avail = ImGui::GetContentRegionAvail();
    constexpr float kAR = 16.0f / 9.0f;
    float dw, dh;
    if (avail.x / avail.y >= kAR) {
        dh = avail.y;
        dw = dh * kAR;
    } else {
        dw = avail.x;
        dh = dw / kAR;
    }

    float ox = (avail.x - dw) * 0.5f;
    float oy = (avail.y - dh) * 0.5f;
    ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPos().x + ox,
                               ImGui::GetCursorPos().y + oy));

    ImVec2 img_min = ImGui::GetCursorScreenPos();
    ImVec2 img_max = ImVec2(img_min.x + dw, img_min.y + dh);
    ImGui::Image((ImTextureID)(intptr_t)display_tex, ImVec2(dw, dh));

    // 双击切换放大/还原
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        zoomed_channel_ = (zoomed_channel_ == id) ? -1 : id;
    }

    // ImGui 检测框叠加（GPU 路径和 CPU 路径统一由 GUI 绘制）
    if (!result.dets.empty()) {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        float scale_x = dw / disp_w;
        float scale_y = dh / disp_h;

        for (const auto& d : result.dets) {
            float x1 = img_min.x + d.x1 * scale_x;
            float y1 = img_min.y + d.y1 * scale_y;
            float x2 = img_min.x + d.x2 * scale_x;
            float y2 = img_min.y + d.y2 * scale_y;

            // 颜色映射（按类别循环）
            static const ImU32 colors[] = {
                IM_COL32(255, 56, 56, 255),   // 红
                IM_COL32(56, 255, 56, 255),   // 绿
                IM_COL32(56, 56, 255, 255),   // 蓝
                IM_COL32(255, 255, 56, 255),  // 黄
                IM_COL32(255, 56, 255, 255),  // 品红
                IM_COL32(56, 255, 255, 255),  // 青
                IM_COL32(255, 128, 0, 255),   // 橙
                IM_COL32(128, 0, 255, 255),   // 紫
            };
            ImU32 col = colors[d.class_id % 8];

            // 绘制边框
            draw_list->AddRect(ImVec2(x1, y1), ImVec2(x2, y2), col, 0.0f, 0, 2.0f);

            // 绘制标签背景 + 文字
            char label[64];
            snprintf(label, sizeof(label), "%s %.0f%%",
                     d.class_name.empty() ? "Drone" : d.class_name.c_str(), d.conf * 100);
            ImVec2 tsize = ImGui::CalcTextSize(label);
            draw_list->AddRectFilled(ImVec2(x1, y1 - tsize.y - 4),
                                     ImVec2(x1 + tsize.x + 4, y1),
                                     col, 2.0f);
            draw_list->AddText(ImVec2(x1 + 2, y1 - tsize.y - 2),
                               IM_COL32(255, 255, 255, 255), label);
        }
    }

    // 叠加状态信息（左上角）
    ImVec2 img_pos = ImVec2(ImGui::GetCursorPos().x + ox - dw,
                            ImGui::GetCursorPos().y + oy);
    ImGui::SetCursorPos(ImVec2(img_pos.x + 6, img_pos.y + 4));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.9f));
    char overlay[64];
    snprintf(overlay, sizeof(overlay), "%.0f FPS | %d obj", result.fps, result.num_objects);
    ImVec2 tsize = ImGui::CalcTextSize(overlay);
    ImVec2 tl = ImGui::GetCursorScreenPos();
    ImGui::GetWindowDrawList()->AddRectFilled(
        tl, ImVec2(tl.x + tsize.x + 8, tl.y + tsize.y + 4),
        IM_COL32(0, 0, 0, 140), 3.0f);
    ImGui::SetCursorPos(ImVec2(img_pos.x + 10, img_pos.y + 6));
    ImGui::Text("%s", overlay);
    ImGui::PopStyleColor();
}

void AppGUI::render_log_panel() {
#if 1  // [HIDDEN] Log panel — 渲染为空窗口占位
    ImGui::Begin("Logs", nullptr,
                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    if (ImGui::SmallButton("Clear")) Logger::instance().clear();

    ImGui::BeginChild("##logscroll", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);
    {
        std::lock_guard<std::mutex> lk(Logger::instance().mutex());
        auto& entries = Logger::instance().get_entries_ref();
        ImGui::SameLine();
        ImGui::TextDisabled("(%d entries)", (int)entries.size());
        ImGui::Separator();

        // 只渲染最近 200 条（避免大日志卡顿）
        size_t start = entries.size() > 200 ? entries.size() - 200 : 0;
        for (size_t i = start; i < entries.size(); i++) {
            auto& e = entries[i];
            ImVec4 color;
            switch (e.level) {
                case LogLevel::DEBUG: color = ImVec4(0.45f,0.45f,0.50f,1); break;
                case LogLevel::INFO:  color = ImVec4(0.75f,0.75f,0.80f,1); break;
                case LogLevel::WARN:  color = ImVec4(1.0f,0.8f,0.2f,1); break;
                case LogLevel::ERROR: color = ImVec4(1.0f,0.3f,0.3f,1); break;
            }
            ImGui::TextColored(color, "[%s] %s", e.timestamp.c_str(), e.message.c_str());
        }
    }
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
        ImGui::SetScrollHereY(1.0f);
    ImGui::EndChild();

    ImGui::End();
#endif
}

void AppGUI::render_panorama(PipelineManager& mgr) {
#if 0  // [HIDDEN] Panorama mode
    // 获取 3 路结果（支持 GPU 路径和 CPU 路径）
    cv::Mat frames[3];
    std::vector<Detection> all_dets[3];
    int max_w = 0, max_h = 0;
    for (int i = 0; i < 3; i++) {
        auto r = mgr.get_result(i);
        if (!r.has_frame) continue;

        all_dets[i] = r.dets;

        if (!r.display.empty()) {
            frames[i] = r.display;
            max_w = std::max(max_w, r.display.cols);
            max_h = std::max(max_h, r.display.rows);
        } else if (r.display_gpu && r.display_w > 0 && r.display_h > 0) {
            cv::Mat rgb(r.display_h, r.display_w, CV_8UC3);
            cudaMemcpy(rgb.data, r.display_gpu,
                       (size_t)r.display_w * r.display_h * 3, cudaMemcpyDeviceToHost);
            cv::cvtColor(rgb, frames[i], cv::COLOR_RGB2BGR);
            max_w = std::max(max_w, r.display_w);
            max_h = std::max(max_h, r.display_h);
        }
    }

    if (max_w == 0 || max_h == 0) {
        ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.45f, 1), "No input for panorama");
        return;
    }

    for (int i = 0; i < 3; i++) {
        if (frames[i].empty()) {
            frames[i] = cv::Mat(max_h, max_w, CV_8UC3, cv::Scalar(30, 30, 30));
        } else if (frames[i].cols != max_w || frames[i].rows != max_h) {
            cv::resize(frames[i], frames[i], cv::Size(max_w, max_h));
        }
    }

    cv::Mat stitched;
    if (state_.stitch_direction == 0) {
        cv::hconcat(frames, 3, stitched);
    } else {
        cv::vconcat(frames, 3, stitched);
    }

    // Panorama 录制控制
    if (panorama_recorder_.isRecording()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.2f, 0.2f, 1.0f));
        ImGui::Text("\xe2\x97\x89 REC");
        ImGui::PopStyleColor();
        ImGui::SameLine();
        float elapsed = panorama_recorder_.recordedSeconds();
        int mins = (int)elapsed / 60;
        int secs = (int)elapsed % 60;
        ImGui::Text("%02d:%02d", mins, secs);
        ImGui::SameLine();
        if (ImGui::SmallButton("Stop##pano_rec")) {
            panorama_recorder_.stop();
        }
        ImGui::SameLine();
        ImGui::TextDisabled("%s", panorama_recorder_.filePath().c_str());
        // 推送拼接帧（全分辨率）
        panorama_recorder_.pushBGRFrame(stitched);
    } else {
        if (ImGui::SmallButton("Record##pano")) {
            auto now = std::chrono::system_clock::now();
            auto t = std::chrono::system_clock::to_time_t(now);
            struct tm tm_buf;
            localtime_r(&t, &tm_buf);
            char path[256];
            snprintf(path, sizeof(path), "%s/panorama_%04d%02d%02d_%02d%02d%02d.mp4",
                     std::getenv("HOME") ? std::getenv("HOME") : "/tmp",
                     tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
                     tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
            panorama_recorder_.start(path, stitched.cols, stitched.rows, 30.0f);
            LOG_INFO("Panorama recording started: %s (%dx%d)", path, stitched.cols, stitched.rows);
        }
    }
    ImGui::Separator();

    ImVec2 avail = ImGui::GetContentRegionAvail();
    float sx = avail.x / stitched.cols;
    float sy = avail.y / stitched.rows;
    float scale = std::min(sx, sy);
    int dw = (int)(stitched.cols * scale);
    int dh = (int)(stitched.rows * scale);
    if (dw > 0 && dh > 0) {
        cv::Mat display;
        cv::resize(stitched, display, cv::Size(dw, dh));

        update_texture(0, display);
        float ox = (avail.x - dw) * 0.5f;
        float oy = (avail.y - dh) * 0.5f;
        ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPos().x + ox,
                                   ImGui::GetCursorPos().y + oy));
        ImVec2 img_min = ImGui::GetCursorScreenPos();
        ImVec2 img_max = ImVec2(img_min.x + dw, img_min.y + dh);
        ImGui::Image((ImTextureID)(intptr_t)textures_[0], ImVec2((float)dw, (float)dh));

        // ImGui 检测框叠加
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        float display_scale = (float)dw / stitched.cols;

        for (int i = 0; i < 3; i++) {
            if (all_dets[i].empty()) continue;

            // 计算该路在拼接图中的偏移
            float ch_offset_x = 0, ch_offset_y = 0;
            if (state_.stitch_direction == 0) {
                ch_offset_x = (float)i * max_w;
            } else {
                ch_offset_y = (float)i * max_h;
            }

            for (auto& d : all_dets[i]) {
                float x1 = img_min.x + (d.x1 + ch_offset_x) * display_scale;
                float y1 = img_min.y + (d.y1 + ch_offset_y) * display_scale;
                float x2 = img_min.x + (d.x2 + ch_offset_x) * display_scale;
                float y2 = img_min.y + (d.y2 + ch_offset_y) * display_scale;

                // 裁剪到图像区域
                x1 = std::max(x1, img_min.x);
                y1 = std::max(y1, img_min.y);
                x2 = std::min(x2, img_max.x);
                y2 = std::min(y2, img_max.y);
                if (x2 <= x1 || y2 <= y1) continue;

                static const ImU32 colors[] = {
                    IM_COL32(255,56,56,255), IM_COL32(56,255,56,255),
                    IM_COL32(56,56,255,255), IM_COL32(255,255,56,255),
                    IM_COL32(255,56,255,255), IM_COL32(56,255,255,255),
                    IM_COL32(255,128,0,255), IM_COL32(128,0,255,255),
                };
                ImU32 col = colors[d.class_id % 8];

                draw_list->AddRect(ImVec2(x1, y1), ImVec2(x2, y2), col, 0.0f, 0, 2.0f);

                char label[64];
                snprintf(label, sizeof(label), "%s %.0f%%",
                         d.class_name.empty() ? "Drone" : d.class_name.c_str(), d.conf * 100);
                ImVec2 tsize = ImGui::CalcTextSize(label);
                draw_list->AddRectFilled(ImVec2(x1, y1 - tsize.y - 4),
                                         ImVec2(x1 + tsize.x + 4, y1), col, 2.0f);
                draw_list->AddText(ImVec2(x1 + 2, y1 - tsize.y - 2),
                                   IM_COL32(255,255,255,255), label);
            }
        }
    }
#endif  // [HIDDEN] Panorama mode
}

void AppGUI::render_gpu_panel() {
    // 每秒刷新一次 GPU 状态（避免频繁 NVML 调用）
    static GPUStatus cached_status;
    static auto last_update = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<float>(now - last_update).count() >= 1.0f) {
        cached_status = gpu_monitor_.getStatus();
        last_update = now;
    }

    if (!cached_status.valid) {
        ImGui::TextDisabled("GPU: N/A");
        return;
    }

    // GPU 名称（小字，避免过长抢镜）
    ImGui::TextDisabled("%s", cached_status.gpu_name.c_str());
    ImGui::Separator();

    const float kBarH = 12.0f;
    const float kLabelX = 48.0f;
    float bar_w = ImGui::GetContentRegionAvail().x - 110;
    if (bar_w < 60) bar_w = 60;

    // GPU 利用率条
    ImGui::AlignTextToFramePadding();
    ImGui::Text("GPU");
    ImGui::SameLine(kLabelX);
    float util_f = cached_status.gpu_util / 100.0f;
    ImVec4 util_color = util_f > 0.8f ? ImVec4(1,0.3f,0.3f,1) :
                        util_f > 0.5f ? ImVec4(1,0.8f,0.2f,1) :
                        ImVec4(0.3f,1,0.3f,1);
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, util_color);
    ImGui::ProgressBar(util_f, ImVec2(bar_w, kBarH), "");
    ImGui::PopStyleColor();
    ImGui::SameLine();
    ImGui::Text("%d%%", cached_status.gpu_util);

    // 显存
    ImGui::AlignTextToFramePadding();
    ImGui::Text("VRAM");
    ImGui::SameLine(kLabelX);
    float mem_f = cached_status.mem_total_mb > 0 ?
                  (float)cached_status.mem_used_mb / cached_status.mem_total_mb : 0;
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.3f,0.6f,1,1));
    ImGui::ProgressBar(mem_f, ImVec2(bar_w, kBarH), "");
    ImGui::PopStyleColor();
    ImGui::SameLine();
    ImGui::Text("%d/%dM", cached_status.mem_used_mb, cached_status.mem_total_mb);

    // 温度 + 频率（同一行）
    ImVec4 temp_color = cached_status.temperature > 80 ? ImVec4(1,0.3f,0.3f,1) :
                        cached_status.temperature > 60 ? ImVec4(1,0.8f,0.2f,1) :
                        ImVec4(0.3f,1,0.3f,1);
    ImGui::Text("Temp");
    ImGui::SameLine(kLabelX);
    ImGui::TextColored(temp_color, "%d°C", cached_status.temperature);
    float panel_w = ImGui::GetContentRegionAvail().x;
    float clock_x = panel_w * 0.5f + 10;
    ImGui::SameLine(clock_x);
    ImGui::TextDisabled("Clock");
    ImGui::SameLine(clock_x + 52);
    ImGui::Text("%d MHz", cached_status.clock_current);
    if (gpu_monitor_.isFrequencyLocked()) {
        ImGui::TextColored(ImVec4(1,0.8f,0.2f,1), "[Locked %d MHz]", gpu_monitor_.lockedFrequency());
    }

#if 0  // [HIDDEN] Frequency lock controls
    // 定频控制
    ImGui::Separator();
    if (gpu_monitor_.isFrequencyLocked()) {
        ImGui::TextColored(ImVec4(1,0.8f,0.2f,1), "Locked: %d MHz", gpu_monitor_.lockedFrequency());
        char unlock_id[] = "Unlock##gpu";
        if (ImGui::Button(unlock_id, ImVec2(-1, 20))) {
            gpu_monitor_.lockFrequency(0);
        }
    } else {
        ImGui::Text("Lock Freq:");
        int max_freq = cached_status.clock_max;
        // 第一行：低频
        int row1[] = {500, 800, 1000, 1500};
        const char* labels1[] = {"500", "800", "1G", "1.5G"};
        float btn_w = (ImGui::GetContentRegionAvail().x - 6 * 2) / 4;
        for (int i = 0; i < 4; i++) {
            if (row1[i] > max_freq) continue;
            if (i > 0) ImGui::SameLine(0, 6);
            char id[16];
            snprintf(id, sizeof(id), "%s##g1%d", labels1[i], i);
            if (ImGui::Button(id, ImVec2(btn_w, 20))) {
                gpu_monitor_.lockFrequency(row1[i]);
            }
        }
        // 第二行：高频
        int row2[] = {2000, 2500, 3000};
        const char* labels2[] = {"2G", "2.5G", "3G"};
        btn_w = (ImGui::GetContentRegionAvail().x - 6 * 2) / 3;
        for (int i = 0; i < 3; i++) {
            if (row2[i] > max_freq) continue;
            if (i > 0) ImGui::SameLine(0, 6);
            char id[16];
            snprintf(id, sizeof(id), "%s##g2%d", labels2[i], i);
            if (ImGui::Button(id, ImVec2(btn_w, 20))) {
                gpu_monitor_.lockFrequency(row2[i]);
            }
        }
        // Max 按钮
        char max_id[] = "Max Freq##gpu";
        if (ImGui::Button(max_id, ImVec2(-1, 20))) {
            gpu_monitor_.lockFrequency(max_freq);
        }
    }
#endif
}

// ==================== 融合模式 ====================

void AppGUI::render_fusion_page(PipelineManager& mgr) {
#if 0  // [HIDDEN] Fusion mode
    // ---- 工具栏 ----
    // View / Detect / Shell 模式切换
    auto op_label = [&](GUIState::FusionOp op, const char* label) {
        bool active = state_.fusion_op == op;
        if (active) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1));
        if (ImGui::Button(label)) state_.fusion_op = op;
        if (active) ImGui::PopStyleColor();
    };

    op_label(GUIState::FusionOp::VIEW, "View");
    ImGui::SameLine();
    op_label(GUIState::FusionOp::DETECT, "Detect");
    ImGui::SameLine();
    op_label(GUIState::FusionOp::SHELL, "Shell");
    ImGui::SameLine();
    ImGui::Separator();
    ImGui::SameLine();

    // 拼接方向
    const char* dir_names[] = {"H-Stitch", "V-Stitch"};
    char dir_id[] = "##fusion_dir";
    ImGui::PushItemWidth(80);
    ImGui::Combo(dir_id, &state_.stitch_direction, dir_names, 2);
    ImGui::PopItemWidth();
    ImGui::SameLine();

    // Detect 模式参数
    if (state_.fusion_op == GUIState::FusionOp::DETECT) {
        ImGui::Separator();
        ImGui::SameLine();
        ImGui::PushItemWidth(60);
        ImGui::SliderFloat("##fconf", &state_.fusion_conf_thresh, 0.05f, 0.9f, "%.2f");
        ImGui::SameLine();
        ImGui::TextDisabled("Conf");
        ImGui::SameLine();
        ImGui::SliderFloat("##fnms", &state_.fusion_nms_thresh, 0.1f, 0.9f, "%.2f");
        ImGui::SameLine();
        ImGui::TextDisabled("NMS");
        ImGui::PopItemWidth();
        ImGui::SameLine();
    }

    ImGui::Separator();
    ImGui::SameLine();

    // 录制控制
    if (!fusion_recorder_.isRecording()) {
        if (ImGui::Button("Record")) {
            // 获取合并画面尺寸来启动录制
            int max_w = 0, max_h = 0;
            for (int i = 0; i < 3; i++) {
                auto r = mgr.get_result(i);
                if (!r.has_frame) continue;
                if (!r.display.empty()) {
                    max_w = std::max(max_w, r.display.cols);
                    max_h = std::max(max_h, r.display.rows);
                } else if (r.display_gpu && r.display_w > 0) {
                    max_w = std::max(max_w, r.display_w);
                    max_h = std::max(max_h, r.display_h);
                }
            }
            if (max_w > 0 && max_h > 0) {
                int sw = state_.stitch_direction == 0 ? max_w * 3 : max_w;
                int sh = state_.stitch_direction == 0 ? max_h : max_h * 3;
                auto now = std::chrono::system_clock::now();
                auto tt = std::chrono::system_clock::to_time_t(now);
                struct tm tm_buf;
                localtime_r(&tt, &tm_buf);
                char path[256];
                snprintf(path, sizeof(path), "%s/fusion_%04d%02d%02d_%02d%02d%02d.mp4",
                         std::getenv("HOME") ? std::getenv("HOME") : "/tmp",
                         tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
                         tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
                fusion_recorder_.start(path, sw, sh, 30.0f);
                LOG_INFO("Fusion recording started: %s (%dx%d)", path, sw, sh);
            }
        }
    } else {
        if (ImGui::Button("Stop Rec")) {
            fusion_recorder_.stop();
            LOG_INFO("Fusion recording stopped");
        }
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "REC %.0fs", fusion_recorder_.recordedSeconds());
    }

    ImGui::Separator();

    // ---- 获取 3 路结果（支持 GPU 路径和 CPU 路径） ----
    cv::Mat frames[3];
    std::vector<Detection> channel_dets[3];
    int max_w = 0, max_h = 0;
    for (int i = 0; i < 3; i++) {
        auto r = mgr.get_result(i);
        if (!r.has_frame) continue;

        channel_dets[i] = r.dets;

        if (!r.display.empty()) {
            frames[i] = r.display;
            max_w = std::max(max_w, r.display.cols);
            max_h = std::max(max_h, r.display.rows);
        } else if (r.display_gpu && r.display_w > 0 && r.display_h > 0) {
            cv::Mat rgb(r.display_h, r.display_w, CV_8UC3);
            cudaMemcpy(rgb.data, r.display_gpu,
                       (size_t)r.display_w * r.display_h * 3, cudaMemcpyDeviceToHost);
            cv::cvtColor(rgb, frames[i], cv::COLOR_RGB2BGR);
            max_w = std::max(max_w, r.display_w);
            max_h = std::max(max_h, r.display_h);
        }
    }

    if (max_w == 0 || max_h == 0) {
        ImVec2 center = ImGui::GetContentRegionAvail();
        ImGui::SetCursorPos(ImVec2(center.x * 0.5f - 60, center.y * 0.5f));
        ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.45f, 1), "No input for fusion");
        return;
    }

    for (int i = 0; i < 3; i++) {
        if (frames[i].empty()) {
            frames[i] = cv::Mat(max_h, max_w, CV_8UC3, cv::Scalar(30, 30, 30));
        } else if (frames[i].cols != max_w || frames[i].rows != max_h) {
            cv::resize(frames[i], frames[i], cv::Size(max_w, max_h));
        }
    }

    // ---- 拼接 ----
    cv::Mat stitched;
    if (state_.stitch_direction == 0) {
        cv::hconcat(frames, 3, stitched);
    } else {
        cv::vconcat(frames, 3, stitched);
    }

    // ---- Detect 模式：对合并画面做 DIRECT 检测（单次推理） ----
    if (state_.fusion_op != GUIState::FusionOp::DETECT) {
        fusion_dets_.clear();
    }
    if (state_.fusion_op == GUIState::FusionOp::DETECT && mgr.engine_loaded()) {
        auto& engine = mgr.engine();
        auto* ctx = engine.create_context();
        if (ctx) {
            const char* in_name = engine.engine()->getIOTensorName(0);
            const char* out_name = engine.engine()->getIOTensorName(1);
            constexpr int INPUT_SIZE = 640;
            constexpr int NUM_PREDS = 25200;
            constexpr int NUM_CLASSES = 80;
            size_t output_bytes = NUM_PREDS * (4 + 1 + NUM_CLASSES) * sizeof(float);

            // 静态 GPU 缓冲区（跨帧复用）
            static float* d_input = nullptr;
            static float* d_output = nullptr;
            static float* h_output = nullptr;
            static uint8_t* d_stitched = nullptr;
            static size_t d_stitched_cap = 0;
            if (!d_input) {
                cudaMalloc(&d_input, INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float));
                cudaMalloc(&d_output, output_bytes);
                cudaMallocHost(&h_output, output_bytes);
            }

            int src_w = stitched.cols;
            int src_h = stitched.rows;
            size_t img_bytes = (size_t)src_w * src_h * 3;

            // 动态分配拼接图 GPU 缓冲区
            if (d_stitched_cap < img_bytes) {
                if (d_stitched) cudaFree(d_stitched);
                cudaMalloc(&d_stitched, img_bytes);
                d_stitched_cap = img_bytes;
            }

            // 上传拼接图到 GPU
            cudaMemcpy(d_stitched, stitched.data, img_bytes, cudaMemcpyHostToDevice);

            // Letterbox 缩放到 640x640
            float scale = std::min((float)INPUT_SIZE / src_w, (float)INPUT_SIZE / src_h);
            int nw = (int)(src_w * scale), nh = (int)(src_h * scale);
            float px = (INPUT_SIZE - nw) / 2.0f, py = (INPUT_SIZE - nh) / 2.0f;
            launch_bgr_letterbox(d_stitched, src_w, src_h, src_w * 3,
                                 d_input, INPUT_SIZE, INPUT_SIZE, scale, px, py, 0);
            cudaStreamSynchronize(0);

            // 单次推理
            ctx->setInputShape(in_name, nvinfer1::Dims4{1, 3, INPUT_SIZE, INPUT_SIZE});
            ctx->setTensorAddress(in_name, d_input);
            ctx->setTensorAddress(out_name, d_output);
            ctx->enqueueV3(0);
            cudaStreamSynchronize(0);

            // D2H
            cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

            // CPU 后处理：置信度过滤 + 坐标映射回拼接图
            struct RawDet { float x1, y1, x2, y2, conf; int class_id; };
            std::vector<RawDet> raw_dets;

            for (int i = 0; i < NUM_PREDS; i++) {
                const float* p = h_output + i * (4 + 1 + NUM_CLASSES);
                float obj = p[4];
                if (obj < state_.fusion_conf_thresh) continue;

                float best = 0;
                int bc = 0;
                for (int c = 0; c < NUM_CLASSES; c++) {
                    if (p[5 + c] > best) { best = p[5 + c]; bc = c; }
                }
                float score = obj * best;
                if (score < state_.fusion_conf_thresh) continue;

                // 从 letterbox 坐标映射回拼接图坐标
                float cx = p[0], cy = p[1], w = p[2], h = p[3];
                float x1 = (cx - w * 0.5f - px) / scale;
                float y1 = (cy - h * 0.5f - py) / scale;
                float x2 = (cx + w * 0.5f - px) / scale;
                float y2 = (cy + h * 0.5f - py) / scale;

                x1 = std::max(0.0f, std::min((float)src_w, x1));
                y1 = std::max(0.0f, std::min((float)src_h, y1));
                x2 = std::max(0.0f, std::min((float)src_w, x2));
                y2 = std::max(0.0f, std::min((float)src_h, y2));

                raw_dets.push_back({x1, y1, x2, y2, score, bc});
            }

            // NMS（贪心抑制，按置信度排序）
            std::sort(raw_dets.begin(), raw_dets.end(),
                      [](const RawDet& a, const RawDet& b) { return a.conf > b.conf; });

            std::vector<bool> suppressed(raw_dets.size(), false);
            fusion_dets_.clear();

            for (size_t i = 0; i < raw_dets.size(); i++) {
                if (suppressed[i]) continue;
                Detection d;
                d.x1 = raw_dets[i].x1; d.y1 = raw_dets[i].y1;
                d.x2 = raw_dets[i].x2; d.y2 = raw_dets[i].y2;
                d.conf = raw_dets[i].conf;
                d.class_id = raw_dets[i].class_id;
                d.class_name = DetectionEngine::get_class_name(d.class_id);
                fusion_dets_.push_back(d);

                for (size_t j = i + 1; j < raw_dets.size(); j++) {
                    if (suppressed[j] || raw_dets[j].class_id != raw_dets[i].class_id) continue;
                    float ix1 = std::max(raw_dets[i].x1, raw_dets[j].x1);
                    float iy1 = std::max(raw_dets[i].y1, raw_dets[j].y1);
                    float ix2 = std::min(raw_dets[i].x2, raw_dets[j].x2);
                    float iy2 = std::min(raw_dets[i].y2, raw_dets[j].y2);
                    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
                    float area_i = (raw_dets[i].x2 - raw_dets[i].x1) * (raw_dets[i].y2 - raw_dets[i].y1);
                    float area_j = (raw_dets[j].x2 - raw_dets[j].x1) * (raw_dets[j].y2 - raw_dets[j].y1);
                    if (inter / (area_i + area_j - inter + 1e-6f) > state_.fusion_nms_thresh) {
                        suppressed[j] = true;
                    }
                }
            }

            delete ctx;
        }
    }

    // ---- Shell 模式：底部终端面板 ----
    float shell_h = 0;
    if (state_.fusion_op == GUIState::FusionOp::SHELL) {
        shell_h = 150.0f;
    }

    // ---- 显示合并画面 ----
    float avail_w = ImGui::GetContentRegionAvail().x;
    float avail_h = ImGui::GetContentRegionAvail().y - shell_h;
    if (avail_h < 100) avail_h = 100;

    float sx = avail_w / stitched.cols;
    float sy = avail_h / stitched.rows;
    float scale = std::min(sx, sy);
    int dw = (int)(stitched.cols * scale);
    int dh = (int)(stitched.rows * scale);

    if (dw > 0 && dh > 0) {
        cv::Mat display;
        cv::resize(stitched, display, cv::Size(dw, dh));

        update_texture(0, display);
        float ox = (avail_w - dw) * 0.5f;
        float oy = (avail_h - dh) * 0.5f;
        ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPos().x + ox,
                                   ImGui::GetCursorPos().y + oy));
        ImVec2 img_min = ImGui::GetCursorScreenPos();
        ImVec2 img_max = ImVec2(img_min.x + dw, img_min.y + dh);
        ImGui::Image((ImTextureID)(intptr_t)textures_[0], ImVec2((float)dw, (float)dh));

        // ImGui 检测框叠加
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        float display_scale = (float)dw / stitched.cols;

        static const ImU32 colors[] = {
            IM_COL32(255,56,56,255), IM_COL32(56,255,56,255),
            IM_COL32(56,56,255,255), IM_COL32(255,255,56,255),
            IM_COL32(255,56,255,255), IM_COL32(56,255,255,255),
            IM_COL32(255,128,0,255), IM_COL32(128,0,255,255),
        };

        auto draw_det = [&](const Detection& d) {
            float x1 = img_min.x + d.x1 * display_scale;
            float y1 = img_min.y + d.y1 * display_scale;
            float x2 = img_min.x + d.x2 * display_scale;
            float y2 = img_min.y + d.y2 * display_scale;
            x1 = std::max(x1, img_min.x); y1 = std::max(y1, img_min.y);
            x2 = std::min(x2, img_max.x); y2 = std::min(y2, img_max.y);
            if (x2 <= x1 || y2 <= y1) return;
            ImU32 col = colors[d.class_id % 8];
            draw_list->AddRect(ImVec2(x1, y1), ImVec2(x2, y2), col, 0.0f, 0, 2.0f);
            char label[64];
            snprintf(label, sizeof(label), "%s %.0f%%",
                     d.class_name.empty() ? "Drone" : d.class_name.c_str(), d.conf * 100);
            ImVec2 tsize = ImGui::CalcTextSize(label);
            draw_list->AddRectFilled(ImVec2(x1, y1 - tsize.y - 4),
                                     ImVec2(x1 + tsize.x + 4, y1), col, 2.0f);
            draw_list->AddText(ImVec2(x1 + 2, y1 - tsize.y - 2),
                               IM_COL32(255,255,255,255), label);
        };

        // View 模式：显示各路独立检测结果
        if (state_.fusion_op == GUIState::FusionOp::VIEW) {
            for (int i = 0; i < 3; i++) {
                float ch_off_x = state_.stitch_direction == 0 ? (float)i * max_w : 0;
                float ch_off_y = state_.stitch_direction == 1 ? (float)i * max_h : 0;
                for (auto& d : channel_dets[i]) {
                    Detection shifted = d;
                    shifted.x1 += ch_off_x; shifted.x2 += ch_off_x;
                    shifted.y1 += ch_off_y; shifted.y2 += ch_off_y;
                    draw_det(shifted);
                }
            }
        }

        // Detect 模式：fusion_dets_ 中的检测结果（拼接图坐标）
        if (state_.fusion_op == GUIState::FusionOp::DETECT) {
            for (auto& d : fusion_dets_) {
                draw_det(d);
            }
        }

        // 录制（全分辨率拼接帧，H.264）
        if (fusion_recorder_.isRecording()) {
            fusion_recorder_.pushBGRFrame(stitched);
        }
    }

    // ---- Shell 终端面板 ----
    if (state_.fusion_op == GUIState::FusionOp::SHELL) {
        ImGui::Separator();
        ImGui::BeginChild("##shell_output", ImVec2(0, shell_h - 35), true,
                          ImGuiWindowFlags_HorizontalScrollbar);
        ImGui::TextUnformatted(shell_output_.c_str());
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
            ImGui::SetScrollHereY(1.0f);
        ImGui::EndChild();

        // 输入框
        ImGui::PushItemWidth(-1);
        if (ImGui::InputText("##shell", shell_cmd_, sizeof(shell_cmd_),
                             ImGuiInputTextFlags_EnterReturnsTrue)) {
            // 执行命令
            std::string cmd = shell_cmd_;
            shell_output_ += "$ " + cmd + "\n";

            FILE* pipe = popen(cmd.c_str(), "r");
            if (pipe) {
                char buf[256];
                while (fgets(buf, sizeof(buf), pipe)) {
                    shell_output_ += buf;
                }
                int ret = pclose(pipe);
                if (ret != 0) {
                    shell_output_ += "[exit code: " + std::to_string(ret) + "]\n";
                }
            } else {
                shell_output_ += "[error: popen failed]\n";
            }
            shell_cmd_[0] = '\0';
        }
        ImGui::PopItemWidth();
    }
#endif  // [HIDDEN] Fusion mode
}
