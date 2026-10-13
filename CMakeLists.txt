cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES "120" CACHE STRING
    "CUDA GPU architectures to compile for (for example: 86;89;120)")

project(cuda_multistream_vision_pipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil libswscale)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation prefix")
if(NOT TENSORRT_ROOT AND DEFINED ENV{TENSORRT_ROOT})
    set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT installation prefix" FORCE)
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS "${TENSORRT_ROOT}/include"
    REQUIRED)
find_library(TENSORRT_NVINFER_LIBRARY NAMES nvinfer
    HINTS "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64"
    REQUIRED)
get_filename_component(TENSORRT_LIBRARY_DIR "${TENSORRT_NVINFER_LIBRARY}" DIRECTORY)
find_library(NVCUVID_LIBRARY NAMES nvcuvid REQUIRED)
find_library(NVML_LIBRARY NAMES nvidia-ml REQUIRED)

option(ENABLE_LIBUVC "Enable UVC cameras when a system libuvc is available" ON)
if(ENABLE_LIBUVC)
    find_path(LIBUVC_INCLUDE_DIR NAMES libuvc/libuvc.h)
    find_library(LIBUVC_LIBRARY NAMES uvc libuvc)
    find_library(LIBUSB_LIBRARY NAMES usb-1.0)
endif()
if(ENABLE_LIBUVC AND LIBUVC_INCLUDE_DIR AND LIBUVC_LIBRARY AND LIBUSB_LIBRARY)
    set(HAS_LIBUVC 1)
    message(STATUS "libuvc enabled: ${LIBUVC_LIBRARY}")
else()
    set(HAS_LIBUVC 0)
    message(STATUS "libuvc not found; UVC camera support disabled")
endif()

set(IMGUI_DIR "${PROJECT_SOURCE_DIR}/third_party/imgui")
set(IMGUI_SRCS
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp)

set(CUDA_SRCS
    src/cuda_preprocess.cu
    src/cuda_postprocess.cu)

set(CPP_SRCS
    src/input_source.cpp
    src/detection_engine.cpp
    src/sahi_slicer.cpp
    src/pipeline.cpp
    src/pipeline_manager.cpp
    src/app_gui.cpp
    src/ffmpeg_hw_decoder.cpp
    src/v4l2_capture.cpp
    src/uvc_capture.cpp
    src/ring_buffer.cpp
    src/video_recorder.cpp
    src/hw_recorder.cpp
    src/gpu_monitor.cpp
    src/main.cpp)

function(configure_pipeline_target target)
    set_target_properties(${target} PROPERTIES BUILD_RPATH "${TENSORRT_LIBRARY_DIR}")
    target_include_directories(${target} PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/include/nvdec"
        "${TENSORRT_INCLUDE_DIR}"
        ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(${target} PRIVATE HAS_LIBUVC=${HAS_LIBUVC})
    target_link_libraries(${target} PRIVATE
        "${TENSORRT_NVINFER_LIBRARY}"
        CUDA::cudart
        CUDA::cuda_driver
        "${NVCUVID_LIBRARY}"
        ${OpenCV_LIBS}
        pthread dl)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endfunction()

add_executable(yolov5_sahi ${CPP_SRCS} ${CUDA_SRCS} ${IMGUI_SRCS})
configure_pipeline_target(yolov5_sahi)
target_include_directories(yolov5_sahi PRIVATE
    "${IMGUI_DIR}" "${IMGUI_DIR}/backends")
target_link_libraries(yolov5_sahi PRIVATE
    glfw OpenGL::GL "${NVML_LIBRARY}" PkgConfig::FFMPEG)

if(HAS_LIBUVC)
    target_include_directories(yolov5_sahi PRIVATE "${LIBUVC_INCLUDE_DIR}")
    target_link_libraries(yolov5_sahi PRIVATE "${LIBUVC_LIBRARY}" "${LIBUSB_LIBRARY}")
endif()

add_executable(camera_bench
    tools/camera_bench.cpp
    src/v4l2_capture.cpp
    src/detection_engine.cpp
    src/ring_buffer.cpp
    ${CUDA_SRCS})
configure_pipeline_target(camera_bench)

add_executable(engine_inspect tools/engine_inspect.cpp src/detection_engine.cpp)
set_target_properties(engine_inspect PROPERTIES BUILD_RPATH "${TENSORRT_LIBRARY_DIR}")
target_include_directories(engine_inspect PRIVATE
    "${PROJECT_SOURCE_DIR}/include" "${TENSORRT_INCLUDE_DIR}")
target_link_libraries(engine_inspect PRIVATE
    "${TENSORRT_NVINFER_LIBRARY}" CUDA::cudart pthread dl)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
