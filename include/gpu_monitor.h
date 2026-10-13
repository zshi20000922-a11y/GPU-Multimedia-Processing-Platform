#pragma once

#include <nvml.h>
#include <string>
#include <mutex>

// GPU 状态信息
struct GPUStatus {
    int gpu_util = 0;          // GPU 利用率 %
    int mem_util = 0;          // 显存利用率 %
    int temperature = 0;       // 温度 °C
    int clock_current = 0;     // 当前频率 MHz
    int clock_max = 0;         // 最大频率 MHz
    int mem_used_mb = 0;       // 已用显存 MB
    int mem_total_mb = 0;      // 总显存 MB
    std::string gpu_name;      // GPU 名称
    bool valid = false;
};

// GPU 监控和管理
class GPUMonitor {
public:
    GPUMonitor();
    ~GPUMonitor();

    // 查询 GPU 状态
    GPUStatus getStatus();

    // GPU 定频（需要 root 权限）
    // lock_freq: 锁定频率（MHz），0=解锁恢复默认
    bool lockFrequency(int lock_freq);

    // 获取当前锁定状态
    bool isFrequencyLocked() const { return freq_locked_; }
    int lockedFrequency() const { return locked_freq_; }

private:
    bool init();
    nvmlDevice_t device_ = nullptr;
    bool initialized_ = false;
    bool freq_locked_ = false;
    int locked_freq_ = 0;
    std::mutex mtx_;
};
