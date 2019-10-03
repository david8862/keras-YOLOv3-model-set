cmake_minimum_required(VERSION 3.5)
project(yolov3Detection LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_BUILD_TYPE "Release")

#### specify the build flag
SET(CMAKE_CXX_FLAGS  "-std=c++11 -O2")
SET(TARGET_PLAT "linux_x86_64" CACHE STRING INTERNAL)

set(YOLOV3_DETECTION_SRC
        yolov3Detection.cpp)

#set(TF_ROOT_PATH /mnt/d/Downloads/tensorflow)

include_directories("${TF_ROOT_PATH}" "${TF_ROOT_PATH}/tensorflow/lite/tools/make/downloads/flatbuffers/include")
link_directories("${TF_ROOT_PATH}/tensorflow/lite/tools/make/gen/${TARGET_PLAT}/lib/")
add_executable(yolov3Detection ${YOLOV3_DETECTION_SRC})
target_link_libraries(yolov3Detection libtensorflow-lite.a -lstdc++ -lpthread -lm -ldl -lrt)
#target_link_libraries(label_image -ltensorflow-lite -lstdc++ -lpthread -lm -ldl -lrt)
