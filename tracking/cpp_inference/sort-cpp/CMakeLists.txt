# This is a C++ reimplementation of the open source tracker from
# https://github.com/mcximing/sort-cpp
#
# Based on the work of Alex Bewley (https://github.com/abewley/sort)
# alex@dynamicdetection.com, 2016
cmake_minimum_required(VERSION 3.5)
project(sort_cpp LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_BUILD_TYPE "Release")

#### specify the compiler flag
SET(CMAKE_CXX_FLAGS  "-std=c++11 -O2")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")

set(SORT_CPP_SRC
        #main.cpp
        mot_tracker.cpp
        Sort.cpp
        KalmanTracker.cpp
        Hungarian.cpp)

#set(OpenCV_INSTALL_PATH /mnt/d/Downloads/opencv/build/install)
find_package(OpenCV REQUIRED COMPONENTS video highgui
             PATHS ${OpenCV_INSTALL_PATH}
             NO_DEFAULT_PATH)

#set(MNN_ROOT_PATH /mnt/d/Projects/MNN)
link_directories("${MNN_ROOT_PATH}/build/")

add_executable(sort_cpp ${SORT_CPP_SRC})

target_include_directories(sort_cpp PRIVATE ${OpenCV_INCLUDE_DIRS} ${MNN_ROOT_PATH}/include/ ${MNN_ROOT_PATH}/3rd_party/imageHelper/)
target_link_libraries(sort_cpp PRIVATE ${OpenCV_LIBRARIES} -lMNN -lstdc++ -lpthread -ldl)
