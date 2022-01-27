# Kalman Filter 2-D mouse tracking demo ball

## Introduction

Simple Python & C++ demo application for standard Kalman Filter on tracking mouse position, based on Ubuntu 18.04/20.04 and OpenCV


## Guide

1. Run Python app
```
# python kalman_demo.py
```

2. Install packages and build OpenCV-3.4.2 C++ library

```
# apt install libgtk2.0-dev libjpeg62-dev libpng-dev libopenexr-dev
# git clone https://github.com/opencv/opencv.git
# git checkout 3.4.2
# mkdir build && cd build
# cmake [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>]
        -DWITH_CUDA=OFF -DCMAKE_INSTALL_PREFIX=<opencv install path> -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_EXAMPLES=OFF  -DBUILD_DOCS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF ..
# make -j4 && make install
```

3. Build demo application

```
# cd keras-YOLOv3-model-set/tracking/cpp_inference/yoloSort/kalman_demo
# mkdir build && cd build
# cmake -DOpenCV_INSTALL_PATH=<opencv install path> [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] ..
# make
# ./kalman_demo
```

