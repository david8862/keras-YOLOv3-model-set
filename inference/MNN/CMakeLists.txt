cmake_minimum_required(VERSION 3.5)
project(yoloDetection LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_BUILD_TYPE "Release")

#### specify the compiler flag
SET(CMAKE_CXX_FLAGS  "-std=c++11 -O2")

set(YOLO_DETECTION_SRC
        yoloDetection.cpp)

#set(MNN_ROOT_PATH /mnt/d/Projects/MNN)

include_directories("${MNN_ROOT_PATH}/include/" "${MNN_ROOT_PATH}/3rd_party/imageHelper/")
link_directories("${MNN_ROOT_PATH}/build/")
add_executable(yoloDetection ${YOLO_DETECTION_SRC})
target_link_libraries(yoloDetection -lMNN -lstdc++ -lpthread)
#target_link_libraries(yoloDetection libMNN.a -Wl,--whole-archive -Wl,--no-whole-archive -lstdc++ -lpthread)
