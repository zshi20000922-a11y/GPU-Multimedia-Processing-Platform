#!/usr/bin/env bash
set -u

APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_BIN="$APP_DIR/build/yolov5_sahi"
APP_LOG_DIR="$APP_DIR/logs"
APP_LOG="$APP_LOG_DIR/drone_detection_latest.log"
APP_LOCK="/tmp/drone-detection-demo.lock"

mkdir -p "$APP_LOG_DIR"

if [[ ! -x "$APP_BIN" ]]; then
    notify-send "无人机检测演示" "未找到可执行程序，请先编译项目。" 2>/dev/null || true
    exit 1
fi

# Prevent accidental double-clicks from opening two processes that compete
# for the same V4L2 camera and TensorRT resources.
exec 9>"$APP_LOCK"
if ! flock -n 9; then
    notify-send "无人机检测演示" "程序已经在运行。" 2>/dev/null || true
    exit 0
fi

cd "$APP_DIR" || exit 1
{
    printf '\n===== Launch %s =====\n' "$(date '+%F %T')"
    exec "$APP_BIN"
} >>"$APP_LOG" 2>&1
