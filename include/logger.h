#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <cstdio>
#include <cstdarg>

// 日志级别
enum class LogLevel { DEBUG, INFO, WARN, ERROR };

// 日志条目
struct LogEntry {
    std::string timestamp;
    std::string message;
    LogLevel level;
};

// 线程安全的日志系统
class Logger {
public:
    static Logger& instance() {
        static Logger inst;
        return inst;
    }

    void log(LogLevel level, const char* fmt, ...) {
        char buf[1024];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);

        // 时间戳（线程安全）
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        struct tm tm_buf;
        localtime_r(&t, &tm_buf);

        char ts[32];
        snprintf(ts, sizeof(ts), "%02d:%02d:%02d", tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);

        const char* level_str = "";
        switch (level) {
            case LogLevel::DEBUG: level_str = "DEBUG"; break;
            case LogLevel::INFO:  level_str = "INFO"; break;
            case LogLevel::WARN:  level_str = "WARN"; break;
            case LogLevel::ERROR: level_str = "ERROR"; break;
        }

        // 锁内完成终端输出 + 内存存储
        std::lock_guard<std::mutex> lk(mtx_);
        fprintf(stderr, "[%s] %s: %s\n", ts, level_str, buf);
        entries_.push_back({ts, buf, level});
        if (entries_.size() > max_entries_) entries_.erase(entries_.begin());
    }

    void debug(const char* fmt, ...) {
        char buf[1024]; va_list args; va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args); va_end(args);
        log(LogLevel::DEBUG, "%s", buf);
    }
    void info(const char* fmt, ...) {
        char buf[1024]; va_list args; va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args); va_end(args);
        log(LogLevel::INFO, "%s", buf);
    }
    void warn(const char* fmt, ...) {
        char buf[1024]; va_list args; va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args); va_end(args);
        log(LogLevel::WARN, "%s", buf);
    }
    void error(const char* fmt, ...) {
        char buf[1024]; va_list args; va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args); va_end(args);
        log(LogLevel::ERROR, "%s", buf);
    }

    std::vector<LogEntry> get_entries() {
        std::lock_guard<std::mutex> lk(mtx_);
        return entries_;
    }

    // 返回互斥锁引用，配合 get_entries_ref() 使用
    std::mutex& mutex() { return mtx_; }

    // 返回引用（调用者必须持有 mutex() 锁）
    const std::vector<LogEntry>& get_entries_ref() const {
        return entries_;
    }

    void clear() {
        std::lock_guard<std::mutex> lk(mtx_);
        entries_.clear();
    }

private:
    Logger() = default;
    std::mutex mtx_;
    std::vector<LogEntry> entries_;
    size_t max_entries_ = 500;
};

#define LOG_DEBUG(...) Logger::instance().debug(__VA_ARGS__)
#define LOG_INFO(...)  Logger::instance().info(__VA_ARGS__)
#define LOG_WARN(...)  Logger::instance().warn(__VA_ARGS__)
#define LOG_ERROR(...) Logger::instance().error(__VA_ARGS__)
