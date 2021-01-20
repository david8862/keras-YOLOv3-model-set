cmake_minimum_required(VERSION 3.5)
project(yoloSort LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_BUILD_TYPE "Release")

#### specify the compiler flag
SET(CMAKE_CXX_FLAGS  "-std=c++11 -O2")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")

set(YOLO_SORT_SRC
        yoloSort.cpp
        Sort.cpp
        KalmanTracker.cpp
        Hungarian.cpp)

#set(OpenCV_INSTALL_PATH /mnt/d/Downloads/opencv/build/install)
find_package(OpenCV REQUIRED COMPONENTS video highgui
             PATHS ${OpenCV_INSTALL_PATH}
             NO_DEFAULT_PATH)

#set(MNN_ROOT_PATH /mnt/d/Projects/MNN)
link_directories("${MNN_ROOT_PATH}/build/")

add_executable(yoloSort ${YOLO_SORT_SRC})

target_include_directories(yoloSort PRIVATE ${OpenCV_INCLUDE_DIRS} ${MNN_ROOT_PATH}/include/ ${MNN_ROOT_PATH}/3rd_party/imageHelper/)
target_link_libraries(yoloSort PRIVATE ${OpenCV_LIBRARIES} -lMNN -lstdc++ -lpthread -ldl)
