cmake_minimum_required(VERSION 3.5)
project(yolov3Detection LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_BUILD_TYPE "Release")

#### specify the compiler flag
SET(CMAKE_CXX_FLAGS  "-std=c++11")

set(YOLOV3_DETECTION_SRC
        yolov3Detection.cpp)

#set(MNN_ROOT_PATH /mnt/d/Projects/MNN)

include_directories("${MNN_ROOT_PATH}/include/" "${MNN_ROOT_PATH}/3rd_party/imageHelper/")
link_directories("${MNN_ROOT_PATH}/build/")
add_executable(yolov3Detection ${YOLOV3_DETECTION_SRC})
target_link_libraries(yolov3Detection -lMNN -lstdc++ -lpthread)
