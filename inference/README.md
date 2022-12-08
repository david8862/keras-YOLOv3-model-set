## C++ on-device (X86/ARM) inference app for YOLOv4/v3/v2 detection modelset

Here are some C++ implementation of the on-device inference for trained YOLOv4/v3/v2 inference model, including forward propagation of the model, YOLO postprocess and bounding box NMS. Generally it should support all YOLOv4/v3/v2 related archs and all kinds of backbones & head. Now we have 2 approaches with different inference engine for that:

* Tensorflow-Lite (verified on tag: v2.6.0)
* [MNN](https://github.com/alibaba/MNN) from Alibaba (verified on release: [1.0.0](https://github.com/alibaba/MNN/releases/tag/1.0.0))


### MNN

1. Install Python runtime and Build libMNN

Refer to [MNN build guide](https://www.yuque.com/mnn/cn/build_linux), we need to prepare cmake & protobuf first for MNN build. And since MNN support both X86 & ARM platform, we can do either native compile or ARM cross-compile
```
# apt install cmake autoconf automake libtool ocl-icd-opencl-dev
# wget https://github.com/google/protobuf/releases/download/v3.4.1/protobuf-cpp-3.4.1.tar.gz
# tar xzvf protobuf-cpp-3.4.1.tar.gz
# cd protobuf-3.4.1
# ./autogen.sh
# ./configure && make && make check && make install && ldconfig
# pip install --upgrade pip && pip install --upgrade mnn

# git clone https://github.com/alibaba/MNN.git <Path_to_MNN>
# cd <Path_to_MNN>
# ./schema/generate.sh
# ./tools/script/get_model.sh  # optional
# mkdir build && cd build
# cmake [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>]
        [-DMNN_BUILD_QUANTOOLS=ON -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_BENCHMARK=ON -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_TRAIN_MINI=ON -DMNN_USE_OPENCV=OFF] ..
        && make -j4

### MNN OpenCL backend build
# apt install ocl-icd-opencl-dev
# cmake [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>]
        [-DMNN_OPENCL=ON -DMNN_SEP_BUILD=OFF -DMNN_USE_SYSTEM_LIB=ON] ..
        && make -j4
```
If you want to do cross compile for ARM platform, "CMAKE_TOOLCHAIN_FILE" should be specified

"MNN_BUILD_QUANTOOLS" is for enabling MNN Quantization tool

"MNN_BUILD_CONVERTER" is for enabling MNN model converter

"MNN_BUILD_BENCHMARK" is for enabling on-device inference benchmark tool

"MNN_BUILD_TRAIN" related are for enabling MNN training tools


2. Build demo inference application
```
# cd keras-YOLOv3-model-set/inference/MNN
# mkdir build && cd build
# cmake -DMNN_ROOT_PATH=<Path_to_MNN> [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] ..
# make
```

3. Convert trained YOLOv3/v2 model to MNN model

Refer to [Model dump](https://github.com/david8862/keras-YOLOv3-model-set#model-dump), [Tensorflow model convert](https://github.com/david8862/keras-YOLOv3-model-set#tensorflow-model-convert) and [MNN model convert](https://www.yuque.com/mnn/cn/model_convert), we need to:

* dump out inference model from training checkpoint:

    ```
    # python yolo.py --model_type=mobilenet_lite --model_path=logs/000/<checkpoint>.h5 --anchors_path=configs/tiny_yolo3_anchors.txt --classes_path=configs/voc_classes.txt --model_input_shape=320x320 --dump_model --output_model_file=model.h5
    ```

* convert keras .h5 model to tensorflow frozen pb model:

    ```
    # python keras_to_tensorflow.py
        --input_model="path/to/keras/model.h5"
        --output_model="path/to/save/model.pb"
    ```

* convert TF pb model to MNN model:

    ```
    # cd <Path_to_MNN>/build/
    # ./MNNConvert -f TF --modelFile model.pb --MNNModel model.pb.mnn --bizCode MNN
    ```
    or

    ```
    # mnnconvert -f TF --modelFile model.pb --MNNModel model.pb.mnn
    ```

MNN support Post Training Integer quantization, so we can use its python CLI interface to do quantization on the generated .mnn model to get quantized .mnn model for ARM acceleration . A json config file [quantizeConfig.json](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/inference/MNN/configs/quantizeConfig.json) is needed to describe the feeding data:

* Quantized MNN model:

    ```
    # cd <Path_to_MNN>/build/
    # ./quantized.out model.pb.mnn model_quant.pb.mnn quantizeConfig.json
    ```
    or

    ```
    # mnnquant model.pb.mnn model_quant.pb.mnn quantizeConfig.json
    ```

4. Run validate script to check MNN model
```
# cd keras-YOLOv3-model-set/tools/evaluation/
# python validate_yolo.py --model_path=model_quant.pb.mnn --anchors_path=../../configs/tiny_yolo3_anchors.txt --classes_path=../../configs/voc_classes.txt --image_file=../../example/dog.jpg --loop_count=5
```

Visualized detection result:

<p align="center">
  <img src="../assets/dog_inference.jpg">
</p>

#### You can also use [eval.py](https://github.com/david8862/keras-YOLOv3-model-set#evaluation) to do evaluation on the MNN model


5. Run application to do inference with model, or put all the assets to your ARM board and run if you use cross-compile
```
# cd keras-YOLOv3-model-set/inference/MNN/build
# ./yoloDetection -h
Usage: yoloDetection
--mnn_model, -m: model_name.mnn
--image, -i: image_name.jpg
--classes, -l: classes labels for the model
--anchors, -a: anchor values for the model
--conf_thrd, -n: confidence threshold for detection filter
--input_mean, -b: input mean
--input_std, -s: input standard deviation
--threads, -t: number of threads
--count, -c: loop model run for certain times
--warmup_runs, -w: number of warmup runs
--result, -r: result txt file to save detection output


# ./yoloDetection -m model.pb.mnn -i ../../../example/dog.jpg -l ../../../configs/voc_classes.txt -a ../../../configs/tiny_yolo3_anchors.txt -n 0.1 -t 8 -c 10 -w 3
Can't Find type=4 backend, use 0 instead
image_input: w:320 , h:320, bpp: 3
num_classes: 20
origin image size: 768, 576
model invoke time: 43.015000 ms
output tensor name: conv2d_1/Conv2D
output tensor name: conv2d_3/Conv2D
Caffe format: NCHW
batch 0:
Caffe format: NCHW
batch 0:
yolo_postprocess time: 1.635000 ms
prediction_list size before NMS: 7
NMS time: 0.044000 ms
Detection result:
bicycle 0.779654 (145, 147) (541, 497)
car 0.929868 (471, 80) (676, 173)
dog 0.519254 (111, 213) (324, 520)
```
Here the [classes](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/configs/voc_classes.txt) & [anchors](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/configs/tiny_yolo3_anchors.txt) file format are the same as used in training part




### Tensorflow-Lite

1. Build TF-Lite lib

We can do either native compile for X86 or cross-compile for ARM

```
# git clone https://github.com/tensorflow/tensorflow <Path_to_TF>
# cd <Path_to_TF>
# git checkout v2.6.0
# ./tensorflow/lite/tools/make/download_dependencies.sh
# make -f tensorflow/lite/tools/make/Makefile   #for X86 native compile
# ./tensorflow/lite/tools/make/build_rpi_lib.sh #for ARM cross compile, e.g Rasperberry Pi
```

you can also create your own build script for new ARM platform, like:

```shell
# vim ./tensorflow/lite/tools/make/build_my_arm_lib.sh

#!/bin/bash -x
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../../.."
make CC_PREFIX=/root/toolchain/aarch64-linux-gnu/bin/aarch64-linux-gnu- -j 3 -f tensorflow/lite/tools/make/Makefile TARGET=myarm TARGET_ARCH=aarch64 $@
```

**NOTE:**
* Using Makefile to build TensorFlow Lite is deprecated since Aug 2021. So v2.6.0 should be the last major version to support Makefile build (cmake is enabled on new version)
* by default TF-Lite build only generate static lib (.a), but we can do minor change in Makefile to generate .so shared lib together, as follow:

```diff
diff --git a/tensorflow/lite/tools/make/Makefile b/tensorflow/lite/tools/make/Makefile
index 662c6bb5129..83219a42845 100644
--- a/tensorflow/lite/tools/make/Makefile
+++ b/tensorflow/lite/tools/make/Makefile
@@ -99,6 +99,7 @@ endif
 # This library is the main target for this makefile. It will contain a minimal
 # runtime that can be linked in to other programs.
 LIB_NAME := libtensorflow-lite.a
+SHARED_LIB_NAME := libtensorflow-lite.so

 # Benchmark static library and binary
 BENCHMARK_LIB_NAME := benchmark-lib.a
@@ -301,6 +302,7 @@ BINDIR := $(GENDIR)bin/
 LIBDIR := $(GENDIR)lib/

 LIB_PATH := $(LIBDIR)$(LIB_NAME)
+SHARED_LIB_PATH := $(LIBDIR)$(SHARED_LIB_NAME)
 BENCHMARK_LIB := $(LIBDIR)$(BENCHMARK_LIB_NAME)
 BENCHMARK_BINARY := $(BINDIR)$(BENCHMARK_BINARY_NAME)
 BENCHMARK_PERF_OPTIONS_BINARY := $(BINDIR)$(BENCHMARK_PERF_OPTIONS_BINARY_NAME)
@@ -344,7 +346,7 @@ $(OBJDIR)%.o: %.c
        $(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

 # The target that's compiled if there's no command-line arguments.
-all: $(LIB_PATH)  $(MINIMAL_BINARY) $(BENCHMARK_BINARY) $(BENCHMARK_PERF_OPTIONS_BINARY)
+all: $(LIB_PATH) $(SHARED_LIB_PATH) $(MINIMAL_BINARY) $(BENCHMARK_BINARY) $(BENCHMARK_PERF_OPTIONS_BINARY)

 # The target that's compiled for micro-controllers
 micro: $(LIB_PATH)
@@ -361,7 +363,14 @@ $(LIB_PATH): tensorflow/lite/experimental/acceleration/configuration/configurati
        @mkdir -p $(dir $@)
        $(AR) $(ARFLAGS) $(LIB_PATH) $(LIB_OBJS)

-lib: $(LIB_PATH)
+$(SHARED_LIB_PATH): tensorflow/lite/schema/schema_generated.h $(LIB_OBJS)
+       @mkdir -p $(dir $@)
+       $(CXX) $(CXXFLAGS) -shared -o $(SHARED_LIB_PATH) $(LIB_OBJS)
+$(SHARED_LIB_PATH): tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h $(LIB_OBJS)
+       @mkdir -p $(dir $@)
+       $(CXX) $(CXXFLAGS) -shared -o $(SHARED_LIB_PATH) $(LIB_OBJS)
+
+lib: $(LIB_PATH) $(SHARED_LIB_PATH)

 $(MINIMAL_BINARY): $(MINIMAL_OBJS) $(LIB_PATH)
        @mkdir -p $(dir $@)
```

2. Build demo inference application
```
# cd keras-YOLOv3-model-set/inference/tflite
# mkdir build && cd build
# cmake -DTF_ROOT_PATH=<Path_to_TF> [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] [-DTARGET_PLAT=<target>] ..
# make
```
If you want to do cross compile for ARM platform, "CMAKE_TOOLCHAIN_FILE" and "TARGET_PLAT" should be specified. Refer [CMakeLists.txt](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/inference/tflite/CMakeLists.txt) for details.

3. Convert trained YOLOv3/v2 model to tflite model

Tensorflow-lite support both Float32 and UInt8 type model. We can dump out the keras .h5 model to Float32 .tflite model or use [post_train_quant_convert.py](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/tools/model_converter/post_train_quant_convert.py) script to convert to UInt8 model with TF 2.0 Post-training integer quantization tech, which could be smaller and faster on ARM:

* dump out inference model from training checkpoint:

    ```
    # python yolo.py --model_type=mobilenet_lite --model_path=logs/000/<checkpoint>.h5 --anchors_path=configs/tiny_yolo3_anchors.txt --classes_path=configs/voc_classes.txt --model_input_shape=320x320 --dump_model --output_model_file=model.h5
    ```

* convert keras .h5 model to Float32 tflite model:

    ```
    # cd keras-YOLOv3-model-set/tools/model_converter/
    # python custom_tflite_convert.py --keras_model_file=model.h5 --output_file=model.tflite
    ```

* convert keras .h5 model to UInt8 tflite model with TF 2.0 Post-training integer quantization:

    ```
    # cd keras-YOLOv3-model-set/tools/model_converter/
    # python post_train_quant_convert.py --keras_model_file=model.h5 --annotation_file=<train/test annotation file to feed converter> --model_input_shape=320x320 --sample_num=30 --output_file=model_quant.tflite
    ```


4. Run validate script to check TFLite model
```
# cd keras-YOLOv3-model-set/tools/evaluation/
# python validate_yolo.py --model_path=model.tflite --anchors_path=../../configs/tiny_yolo3_anchors.txt --classes_path=../../configs/voc_classes.txt --image_file=../../example/dog.jpg --loop_count=5
```
#### You can also use [eval.py](https://github.com/david8862/keras-YOLOv3-model-set#evaluation) to do evaluation on the TFLite model



5. Run application to do inference with model, or put assets to ARM board and run if cross-compile
```
# cd keras-YOLOv3-model-set/inference/tflite/build
# ./yoloDetection -h
Usage: yoloDetection
--tflite_model, -m: model_name.tflite
--image, -i: image_name.jpg
--classes, -l: classes labels for the model
--anchors, -a: anchor values for the model
--conf_thrd, -n: confidence threshold for detection filter
--input_mean, -b: input mean
--input_std, -s: input standard deviation
--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not
--threads, -t: number of threads
--count, -c: loop interpreter->Invoke() for certain times
--warmup_runs, -w: number of warmup runs
--result, -r: result txt file to save detection output
--verbose, -v: [0|1] print more information

# ./yoloDetection -m model.tflite -i ../../../example/dog.jpg -l ../../../configs/voc_classes.txt -a ../../../configs/tiny_yolo3_anchors.txt -n 0.1 -t 8 -c 10 -w 3 -v 1
Loaded model model.tflite
resolved reporter
num_classes: 20
origin image size: width:768, height:576, channel:3
input tensor info: type 1, batch 1, height 320, width 320, channels 3
invoked average time:107.479 ms
output tensor info: name conv2d_1/BiasAdd, type 1, batch 1, height 10, width 10, channels 75
batch 0
output tensor info: name conv2d_3/BiasAdd, type 1, batch 1, height 20, width 20, channels 75
batch 0
yolo_postprocess time: 3.618 ms
prediction_list size before NMS: 7
NMS time: 0.358 ms
Detection result:
bicycle 0.838566 (144, 141) (549, 506)
car 0.945672 (466, 79) (678, 173)
dog 0.597517 (109, 215) (326, 519)
```

### On-device evaluation

1. Build your MNN/TFLite version "yoloDetection" application and put it on device together with [eval_inference.sh](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/inference/eval_inference.sh). Then run the script to generate on-device inference result txt file for test images:

```
# ./eval_inference.sh
Usage: ./eval_inference.sh <model_file> <image_path> <anchor_file> <class_file> <result_file> [conf_thrd=0.1]
```

2. Use independent evaluation tool [object_detection_eval.py](https://github.com/david8862/Object-Detection-Evaluation/blob/master/object_detection_eval.py) to get mAP from result txt.


### TODO
- [ ] further latency optimize on yolo3 postprocess C++ implementation
- [ ] refactor demo app to get common interface
