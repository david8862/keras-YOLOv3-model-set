cmake_minimum_required(VERSION 3.5)
project(kalman_demo LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_BUILD_TYPE "Release")

#### specify the compiler flag
SET(CMAKE_CXX_FLAGS  "-std=c++11 -O2")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")

set(KALMAN_DEMO_SRC
        kalman_demo.cpp)

#set(OpenCV_INSTALL_PATH /mnt/d/Downloads/opencv/build/install)
find_package(OpenCV REQUIRED COMPONENTS video highgui
             PATHS ${OpenCV_INSTALL_PATH}
             NO_DEFAULT_PATH)

add_executable(kalman_demo ${KALMAN_DEMO_SRC})

target_include_directories(kalman_demo PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(kalman_demo PRIVATE ${OpenCV_LIBRARIES})
