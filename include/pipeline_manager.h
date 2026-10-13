#pragma once

#include "pipeline.h"
#include "detection_engine.h"
#include "logger.h"

#include <vector>
#include <memory>
#include <string>
#include <map>

// 多路管理器
class PipelineManager {
public:
    PipelineManager() = default;
    ~PipelineManager() { stop_all(); }

    // 加载全局模型（默认）
    bool load_engine(const std::string& engine_path);

    // 管理管线（使用全局引擎）
    void start_pipeline(int id, const InputConfig& cfg);

    // 管理管线（使用独立引擎）
    void start_pipeline_with_engine(int id, const InputConfig& cfg, const std::string& engine_path);

    void stop_pipeline(int id);
    void update_pipeline(int id, const InputConfig& cfg);

    // 获取结果
    PipelineResult get_result(int id);

    // 全部停止
    void stop_all();

    // 获取录制器
    VideoRecorder* recorder(int id);

    // 获取硬件录制器（4K 原始帧）
    HWRecorder* hw_recorder(int id);

    // 获取 Pipeline 指针（快照等直接访问）
    Pipeline* pipeline(int id);

    // 获取全局引擎信息
    DetectionEngine& engine() { return engine_; }
    bool engine_loaded() const { return engine_.is_loaded(); }
    std::string engine_path() const { return engine_path_; }

    // 获取 Pipeline 引擎信息
    bool pipeline_has_own_engine(int id) const;
    std::string pipeline_engine_path(int id) const;

private:
    DetectionEngine engine_;
    std::string engine_path_;
    std::vector<std::unique_ptr<Pipeline>> pipelines_;  // 最多3路
    std::map<int, std::string> pipeline_engine_paths_;  // 每个 Pipeline 的引擎路径
};
