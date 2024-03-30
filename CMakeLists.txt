# cmake
cmake_minimum_required(VERSION 3.10)
project(yolov8seg)

set(CMAKE_CXX_STANDARD 11)


# include ncnn library
set(ncnn_DIR "/home/renshujia/AiStudy/deploy/ncnn_ubuntu/ncnn-20240102-ubuntu-2204-shared/lib/cmake/ncnn/")
find_package(ncnn REQUIRED)
# include opencv library
find_package(OpenCV REQUIRED)

# add source files
set(SRC_LISTS
    src/main.cpp
    src/yolov8_seg.cpp
)
include_directories(/home/renshujia/AiStudy/deploy/ncnn_ubuntu/ncnn-20240102-ubuntu-2204-shared/include/ncnn)

# add executable target
add_executable(${PROJECT_NAME} ${SRC_LISTS})
target_link_libraries(${PROJECT_NAME} ncnn ${OpenCV_LIBS})