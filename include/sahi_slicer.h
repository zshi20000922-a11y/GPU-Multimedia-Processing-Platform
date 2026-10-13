#pragma once

#include "detection_engine.h"
#include <opencv2/opencv.hpp>
#include <vector>

// SAHI 切片配置
struct SAHIConfig {
    int slice_size = 640;   // 切片大小
    int stride = 320;       // 步长 (stride <= slice_size 有重叠)
};

// SAHI 切片推理
class SAHISlicer {
public:
    SAHISlicer() = default;

    // 设置配置
    void configure(const SAHIConfig& cfg) { config_ = cfg; }
    SAHIConfig& config() { return config_; }

    // 获取切片信息
    int get_num_tiles(int img_w, int img_h) const;

    // Tile 结构体（轻量级，不存储像素数据）
    struct Tile {
        int x, y, w, h;       // 在原图中的位置
    };

    // 生成切片
    std::vector<Tile> generate_tiles(const cv::Mat& frame);

    // 合并检测结果（不修改输入数据）
    std::vector<Detection> merge_detections(const std::vector<std::vector<Detection>>& tile_dets,
                                             const std::vector<Tile>& tiles,
                                             float nms_thresh);

private:
    SAHIConfig config_;
};
