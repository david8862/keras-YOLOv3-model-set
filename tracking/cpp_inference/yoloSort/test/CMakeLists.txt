cmake_minimum_required(VERSION 3.5)
project(kalman_test LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_BUILD_TYPE "Release")

#### specify the compiler flag
SET(CMAKE_CXX_FLAGS  "-std=c++11 -O2")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")

set(KALMAN_TEST_SRC
        kalman_test.cpp)

#set(OpenCV_INSTALL_PATH /mnt/d/Downloads/opencv/build/install)
find_package(OpenCV REQUIRED COMPONENTS video highgui
             PATHS ${OpenCV_INSTALL_PATH}
             NO_DEFAULT_PATH)

add_executable(kalman_test ${KALMAN_TEST_SRC})

target_include_directories(kalman_test PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(kalman_test PRIVATE ${OpenCV_LIBRARIES})
