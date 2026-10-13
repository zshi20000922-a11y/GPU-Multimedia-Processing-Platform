#include "sahi_slicer.h"
#include "logger.h"
#include <algorithm>

int SAHISlicer::get_num_tiles(int img_w, int img_h) const {
    int stride = config_.stride;
    int slice = config_.slice_size;
    if (img_w <= slice && img_h <= slice) return 1;

    int nx = std::max(1, (img_w - slice) / stride + 1);
    int ny = std::max(1, (img_h - slice) / stride + 1);
    int count = nx * ny;

    // 右边缘：仅当滑动窗口未覆盖到 w-slice 时才需要
    int last_x = ((img_w - slice) / stride) * stride;
    bool need_right = (img_w > slice && last_x != img_w - slice);
    // 下边缘：仅当滑动窗口未覆盖到 h-slice 时才需要
    int last_y = ((img_h - slice) / stride) * stride;
    bool need_bottom = (img_h > slice && last_y != img_h - slice);

    if (need_right) count += ny;           // 每行多一个右边缘 tile
    if (need_bottom) count += nx;          // 每列多一个下边缘 tile
    if (need_right && need_bottom) count += 1;  // 右下角

    return count;
}

std::vector<SAHISlicer::Tile> SAHISlicer::generate_tiles(const cv::Mat& frame) {
    std::vector<Tile> tiles;
    int w = frame.cols, h = frame.rows;
    int stride = config_.stride;
    int slice = config_.slice_size;

    if (w <= slice && h <= slice) {
        tiles.push_back({0, 0, w, h});
        return tiles;
    }

    // 右边缘/下边缘是否需要额外 tile（滑动窗口未覆盖到边界）
    int last_x = ((w - slice) / stride) * stride;
    int last_y = ((h - slice) / stride) * stride;
    bool need_right = (w > slice && last_x != w - slice);
    bool need_bottom = (h > slice && last_y != h - slice);

    // 滑动窗口切片
    for (int y = 0; y + slice <= h; y += stride) {
        for (int x = 0; x + slice <= w; x += stride) {
            tiles.push_back({x, y, slice, slice});
        }
        // 右边缘
        if (need_right) {
            tiles.push_back({w - slice, y, slice, slice});
        }
    }
    // 下边缘
    if (need_bottom) {
        for (int x = 0; x + slice <= w; x += stride) {
            tiles.push_back({x, h - slice, slice, slice});
        }
        // 右下角
        if (need_right) {
            tiles.push_back({w - slice, h - slice, slice, slice});
        }
    }

    return tiles;
}

std::vector<Detection> SAHISlicer::merge_detections(
    const std::vector<std::vector<Detection>>& tile_dets,
    const std::vector<Tile>& tiles,
    float nms_thresh)
{
    // 将所有 tile 的检测结果映射回原图坐标（不修改输入数据）
    std::vector<Detection> all_dets;
    for (size_t i = 0; i < tile_dets.size(); i++) {
        for (const auto& d : tile_dets[i]) {
            Detection shifted = d;
            shifted.x1 += tiles[i].x;
            shifted.y1 += tiles[i].y;
            shifted.x2 += tiles[i].x;
            shifted.y2 += tiles[i].y;
            all_dets.push_back(shifted);
        }
    }

    if (all_dets.empty()) return {};

    // 全局 NMS
    std::vector<int> order(all_dets.size());
    for (size_t i = 0; i < all_dets.size(); i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return all_dets[a].conf > all_dets[b].conf;
    });

    std::vector<bool> sup(all_dets.size(), false);
    std::vector<int> keep;
    for (size_t i = 0; i < order.size(); i++) {
        int idx = order[i];
        if (sup[idx]) continue;
        keep.push_back(idx);
        float area_a = (all_dets[idx].x2 - all_dets[idx].x1) * (all_dets[idx].y2 - all_dets[idx].y1);
        for (size_t k = i + 1; k < order.size(); k++) {
            int j = order[k];
            if (sup[j] || all_dets[j].class_id != all_dets[idx].class_id) continue;
            float ix1 = std::max(all_dets[idx].x1, all_dets[j].x1);
            float iy1 = std::max(all_dets[idx].y1, all_dets[j].y1);
            float ix2 = std::min(all_dets[idx].x2, all_dets[j].x2);
            float iy2 = std::min(all_dets[idx].y2, all_dets[j].y2);
            float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
            float area_b = (all_dets[j].x2 - all_dets[j].x1) * (all_dets[j].y2 - all_dets[j].y1);
            if (inter / (area_a + area_b - inter + 1e-6f) > nms_thresh) sup[j] = true;
        }
    }

    std::vector<Detection> result;
    for (int i : keep) result.push_back(all_dets[i]);
    return result;
}
