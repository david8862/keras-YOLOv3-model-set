cmake_minimum_required(VERSION 3.5)
project(yoloDetection LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_BUILD_TYPE "Release")

#### specify the build flag
SET(CMAKE_CXX_FLAGS  "-std=c++11 -O2")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")
SET(TARGET_PLAT "linux_x86_64" CACHE STRING INTERNAL)

set(YOLO_DETECTION_SRC
        yoloDetection.cpp)

#set(TF_ROOT_PATH /mnt/d/Downloads/tensorflow)

include_directories("${TF_ROOT_PATH}" "${TF_ROOT_PATH}/tensorflow/lite/tools/make/downloads/flatbuffers/include")
link_directories("${TF_ROOT_PATH}/tensorflow/lite/tools/make/gen/${TARGET_PLAT}/lib/")
add_executable(yoloDetection ${YOLO_DETECTION_SRC})
target_link_libraries(yoloDetection libtensorflow-lite.a -lstdc++ -lpthread -lm -ldl -lrt)
#target_link_libraries(yoloDetection -ltensorflow-lite -lstdc++ -lpthread -lm -ldl -lrt)
