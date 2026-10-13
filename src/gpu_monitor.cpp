#include "gpu_monitor.h"
#include "logger.h"
#include <cstdlib>

GPUMonitor::GPUMonitor() {
    init();
}

GPUMonitor::~GPUMonitor() {
    if (initialized_) {
        nvmlShutdown();
    }
}

bool GPUMonitor::init() {
    nvmlReturn_t ret = nvmlInit_v2();
    if (ret != NVML_SUCCESS) {
        LOG_WARN("NVML init failed: %s", nvmlErrorString(ret));
        return false;
    }
    ret = nvmlDeviceGetHandleByIndex_v2(0, &device_);
    if (ret != NVML_SUCCESS) {
        LOG_WARN("NVML get device failed: %s", nvmlErrorString(ret));
        return false;
    }
    initialized_ = true;
    return true;
}

GPUStatus GPUMonitor::getStatus() {
    std::lock_guard<std::mutex> lk(mtx_);
    GPUStatus s;
    if (!initialized_) return s;

    nvmlReturn_t ret;

    // GPU 利用率
    nvmlUtilization_t util;
    ret = nvmlDeviceGetUtilizationRates(device_, &util);
    if (ret == NVML_SUCCESS) {
        s.gpu_util = util.gpu;
        s.mem_util = util.memory;
    }

    // 温度
    unsigned int temp;
    ret = nvmlDeviceGetTemperature(device_, NVML_TEMPERATURE_GPU, &temp);
    if (ret == NVML_SUCCESS) s.temperature = (int)temp;

    // 当前频率
    unsigned int clock;
    ret = nvmlDeviceGetClockInfo(device_, NVML_CLOCK_GRAPHICS, &clock);
    if (ret == NVML_SUCCESS) s.clock_current = (int)clock;

    // 最大频率
    unsigned int max_clock;
    ret = nvmlDeviceGetMaxClockInfo(device_, NVML_CLOCK_GRAPHICS, &max_clock);
    if (ret == NVML_SUCCESS) s.clock_max = (int)max_clock;

    // 显存
    nvmlMemory_t mem;
    ret = nvmlDeviceGetMemoryInfo(device_, &mem);
    if (ret == NVML_SUCCESS) {
        s.mem_used_mb = (int)(mem.used / 1048576);
        s.mem_total_mb = (int)(mem.total / 1048576);
    }

    // GPU 名称
    char name[NVML_DEVICE_NAME_V2_BUFFER_SIZE];
    ret = nvmlDeviceGetName(device_, name, NVML_DEVICE_NAME_V2_BUFFER_SIZE);
    if (ret == NVML_SUCCESS) s.gpu_name = name;

    s.valid = true;
    return s;
}

bool GPUMonitor::lockFrequency(int lock_freq) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!initialized_) return false;

    char cmd[256];
    if (lock_freq <= 0) {
        // 解锁：恢复默认
        snprintf(cmd, sizeof(cmd), "nvidia-smi -rgc 2>/dev/null");
        freq_locked_ = false;
        locked_freq_ = 0;
        LOG_INFO("GPU: frequency unlocked");
    } else {
        // 锁定频率
        snprintf(cmd, sizeof(cmd), "nvidia-smi -lgc %d,%d 2>/dev/null", lock_freq, lock_freq);
        locked_freq_ = lock_freq;
        freq_locked_ = true;
        LOG_INFO("GPU: frequency locked to %d MHz", lock_freq);
    }

    int ret = system(cmd);
    return ret == 0;
}
