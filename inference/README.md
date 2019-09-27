## C++ on-device (X86/ARM) inference app for YOLOv3 detection modelset

Here are some C++ implementation of the on-device inference for trained YOLOv3 inference model, including forward propagation of the model, YOLO postprocess and bounding box NMS. It support YOLOv3/Tiny YOLOv3 arch and all kinds of backbones & head. Currently the forward propagation part is implement with [MNN](https://github.com/alibaba/MNN) inference engine

### MNN

1. Build libMNN and model convert tool

Refer to [MNN build guide](https://www.yuque.com/mnn/cn/build_linux). Since MNN support both X86 & ARM platform, we can do either native compile or ARM cross-compile
```
# apt install ocl-icd-opencl-dev
# git clone https://github.com/alibaba/MNN.git <Path_to_MNN>
# cd <Path_to_MNN>
# ./schema/generate.sh
# ./tools/script/get_model.sh  # optional
# mkdir build && cd build
# cmake [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] .. && make -j4

# cd ../tools/converter
# ./generate_schema.sh
# mkdir build && cd build && cmake .. && make -j4
```

2. Build demo inference application
```
# cd keras-YOLOv3-model-set/inference/MNN
# mkdir build && cd build
# cmake -DMNN_ROOT_PATH=<Path_to_MNN> [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] ..
# make
```

3. Convert trained YOLOv3 model to MNN model

Refer to [Model dump](https://github.com/david8862/keras-YOLOv3-model-set#model-dump), [Tensorflow model convert](https://github.com/david8862/keras-YOLOv3-model-set#tensorflow-model-convert) and [MNN model convert](https://www.yuque.com/mnn/cn/model_convert), we need to

    1. dump out inference model from training checkpoint

    ```
    # python yolo.py --model_type=mobilenet_lite --model_path=logs/000/<checkpoint>.h5 --anchors_path=configs/yolo_anchors.txt --classes_path=configs/voc_classes.txt --dump_model --output_model_file=model.h5
    ```

    2. convert keras .h5 model to tensorflow model (frozen pb)

    ```
    # python keras_to_tensorflow.py
        --input_model="path/to/keras/model.h5"
        --output_model="path/to/save/model.pb"
    ```

    3. convert TF pb model to MNN model

    ```
    # cd <Path_to_MNN>/tools/converter/build
    # ./MNNConvert -f TF --modelFile model.pb --MNNModel model.pb.mnn --bizCode biz
    ```

4. Run application to do inference with model, or put all the assets to your ARM board and run if you use cross-compile
```
# cd keras-YOLOv3-model-set/inference/MNN/build
# ./yolov3Detection
Usage: ./yolov3Detection model.mnn input.jpg classes.txt anchors.txt

# ./yolov3Detection model.pb.mnn ../../../example/dog.jpg ../../../configs/voc_classes.txt ../../../configs/tiny_yolo_anchors.txt
Can't Find type=4 backend, use 0 instead
image_input: w:320 , h:320, bpp: 3
num_classes: 20
origin image size: 768, 576
model invoke time: 71.015000 ms
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
Here the [classes](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/configs/voc_classes.txt) & [anchors](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/configs/tiny_yolo_anchors.txt) file format are the same as used in training part
