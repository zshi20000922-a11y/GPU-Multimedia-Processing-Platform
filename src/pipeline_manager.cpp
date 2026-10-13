#include "pipeline_manager.h"
#include "hw_recorder.h"
#include "logger.h"

bool PipelineManager::load_engine(const std::string& engine_path) {
    if (engine_.is_loaded()) {
        engine_.destroy();
    }
    engine_path_ = engine_path;
    bool ok = engine_.load(engine_path);
    if (ok) {
        LOG_INFO("Global engine loaded: %s", engine_path.c_str());
    } else {
        LOG_ERROR("Global engine load failed: %s", engine_path.c_str());
    }
    return ok;
}

void PipelineManager::start_pipeline(int id, const InputConfig& cfg) {
    if (id < 0 || id >= 3) return;

    // 检查引擎路径是否有效
    if (engine_path_.empty()) {
        LOG_ERROR("Pipeline %d: engine path not set, cannot start", id);
        return;
    }

    // 确保 pipelines_ 有足够空间
    while ((int)pipelines_.size() <= id) {
        pipelines_.push_back(nullptr);
    }

    // 停止已有的
    if (pipelines_[id]) {
        pipelines_[id]->stop();
    }

    // 创建新的（使用独立引擎，避免多路冲突）
    pipelines_[id] = std::make_unique<Pipeline>();
    pipelines_[id]->init_with_engine(id, cfg, engine_path_);
    pipelines_[id]->start();

    LOG_INFO("Pipeline %d started with own engine: %s", id, engine_path_.c_str());
}

void PipelineManager::start_pipeline_with_engine(int id, const InputConfig& cfg, const std::string& engine_path) {
    if (id < 0 || id >= 3) return;

    // 确保 pipelines_ 有足够空间
    while ((int)pipelines_.size() <= id) {
        pipelines_.push_back(nullptr);
    }

    // 停止已有的
    if (pipelines_[id]) {
        pipelines_[id]->stop();
    }

    // 创建新的（使用独立引擎）
    pipelines_[id] = std::make_unique<Pipeline>();
    pipelines_[id]->init_with_engine(id, cfg, engine_path);
    pipelines_[id]->start();

    // 记录引擎路径
    pipeline_engine_paths_[id] = engine_path;

    LOG_INFO("Pipeline %d started with own engine: %s", id, engine_path.c_str());
}

void PipelineManager::stop_pipeline(int id) {
    if (id < 0 || id >= (int)pipelines_.size()) return;
    if (pipelines_[id]) {
        pipelines_[id]->stop();
        pipelines_[id].reset();
        pipeline_engine_paths_.erase(id);
    }
}

void PipelineManager::update_pipeline(int id, const InputConfig& cfg) {
    if (id < 0 || id >= (int)pipelines_.size()) return;
    if (pipelines_[id]) {
        pipelines_[id]->update_config(cfg);
    }
}

PipelineResult PipelineManager::get_result(int id) {
    if (id < 0 || id >= (int)pipelines_.size()) return {};
    if (pipelines_[id]) {
        return pipelines_[id]->get_result();
    }
    return {};
}

bool PipelineManager::pipeline_has_own_engine(int id) const {
    if (id < 0 || id >= (int)pipelines_.size()) return false;
    if (pipelines_[id]) {
        return pipelines_[id]->has_own_engine();
    }
    return false;
}

std::string PipelineManager::pipeline_engine_path(int id) const {
    auto it = pipeline_engine_paths_.find(id);
    if (it != pipeline_engine_paths_.end()) {
        return it->second;
    }
    return "";
}

VideoRecorder* PipelineManager::recorder(int id) {
    if (id < 0 || id >= (int)pipelines_.size()) return nullptr;
    if (pipelines_[id]) return pipelines_[id]->recorder();
    return nullptr;
}

HWRecorder* PipelineManager::hw_recorder(int id) {
    if (id < 0 || id >= (int)pipelines_.size()) return nullptr;
    if (pipelines_[id]) return pipelines_[id]->hw_recorder();
    return nullptr;
}

Pipeline* PipelineManager::pipeline(int id) {
    if (id < 0 || id >= (int)pipelines_.size()) return nullptr;
    return pipelines_[id].get();
}

void PipelineManager::stop_all() {
    for (auto& p : pipelines_) {
        if (p) {
            p->stop();
            p.reset();
        }
    }
    pipeline_engine_paths_.clear();
    engine_.destroy();
}
