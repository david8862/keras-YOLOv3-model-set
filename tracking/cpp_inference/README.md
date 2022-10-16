## C++ on-device (X86/ARM) inference app for Multi Object Tracking

Here are C++ implementation of SORT/DeepSORT Multi Object Tracking algorithm with YOLOv4/v3/v2 detection model.


### SORT

1. Build libMNN and prepare YOLO model

Refer to [YOLO on-device inference](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/inference/README.md)


2. Build OpenCV
```
# git clone https://github.com/opencv/opencv.git
# cd opencv
# git checkout 3.4.2
# mkdir build && cd build
# cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<opencv path>/build/install/ [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] ..
# make -j4
# make install
```

3. Build demo inference application
```
# cd keras-YOLOv3-model-set/tracking/cpp_inference/yoloSort
# mkdir build && cd build
# cmake -DOpenCV_INSTALL_PATH=<Path_to_OpenCV> -DMNN_ROOT_PATH=<Path_to_MNN> [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] ..
# make
```

4. Run application to do MOT inference, or put all the assets to your ARM board and run if you use cross-compile. Now the application only support sequential images folder as input:
```
# cd keras-YOLOv3-model-set/tracking/cpp_inference/yoloSort/build
# ./yoloSort -h
Usage: yoloSort
--mnn_model, -m: model_name.mnn
--image_path, -i: ./images/
--classes, -l: classes labels for detection model
--track_classes, -k: classes labels for tracker, will track all detect classes if not provided
--anchors, -a: anchor values for the model
--conf_thrd, -n: confidence threshold for detection filter
--input_mean, -b: input mean
--input_std, -s: input standard deviation
--elim_grid_sense, -e: [0|1] eliminate grid sensitivity
--threads, -t: number of threads
--count, -c: loop model run for certain times
--warmup_runs, -w: number of warmup runs
--result, -r: result txt file to save detection output
```
Here the `--classes` & `--anchors` file format are the same as used for detection model. `--track_classes` is for filtering object classes in tracking and use the same format as `--classes`.


### TODO
- [ ] support DeepSORT C++ inference
- [ ] support video input

