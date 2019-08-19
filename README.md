# TF Keras YOLOv3 Modelset

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A tf.keras implementation of a common YOLOv3 object detection architecture with different Backbone/Head/Loss/Postprocess.
#### Backbone
- [x] Darknet53/Tiny Darknet
- [x] MobilenetV1
- [x] VGG16
- [x] Xception

#### Head
- [x] YOLOv3
- [x] YOLOv3 Lite
- [x] Tiny YOLOv3
- [x] Tiny YOLOv3 Lite

#### Loss
- [x] Standard YOLOv3 loss
- [x] Binary focal classification loss
- [x] Softmax focal classification loss
- [x] GIoU localization loss
- [x] Binary focal loss for objectness (experimental)

#### Postprocess
- [x] TF YOLOv3 postprocess model
- [x] Numpy YOLOv3 postprocess implementation


## Quick Start

1. Download Darknet/YOLOv3/Tiny YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection on your image or video.

```
# wget -O model_data/darknet53.conv.74.weights https://pjreddie.com/media/files/darknet53.conv.74
# wget -O model_data/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
# wget -O model_data/yolov3-tiny.weights https://pjreddie.com/media/files/yolov3-tiny.weights

# cd tools && python convert.py yolov3.cfg ../model_data/yolov3.weights ../model_data/yolov3.h5
# python convert.py yolov3-tiny.cfg ../model_data/yolov3-tiny.weights ../model_data/tiny_yolo_weights.h5
# python convert.py darknet53.cfg ../model_data/darknet53.conv.74.weights ../model_data/darknet53_weights.h5
# cd ..

# python yolo.py --model_type=darknet --model_path=model_data/yolov3.h5 --anchors_path=model_data/yolo_anchors.txt --classes_path=model_data/coco_classes.txt --image
# python yolo.py --model_type=darknet --model_path=model_data/yolov3.h5 --anchors_path=model_data/yolo_anchors.txt --classes_path=model_data/coco_classes.txt --input=<your video file>
```
For Tiny YOLOv3, just do in a similar way, but specify different model path and anchor path with `--model_path` and `--anchors_path`.


## Guide of train & evaluate & demo process

### Train
1. Generate your own train/val/test annotation file and class names file.
    One row for one image in annotation file;
    Row format: `image_file_path box1 box2 ... boxN`;
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).
    For VOC style dataset, try `python tools/voc_annotation.py`
    For COCO style dataset, try `python tools/coco_annotation.py`
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```
   Merge train & val annotation file as train script needed

   If you want to download PascalVOC or COCO dataset, refer to Dockerfile for cmd

   For class names file format, refer to model_data/coco_classes.txt

2. If you're training Darknet YOLOv3/Tiny YOLOv3, make sure you have converted pretrain model weights as in "Quick Start" part

3. train.py
```
# python train.py -h
usage: train.py [-h] [--model_type MODEL_TYPE] [--tiny_version]
                [--annotation_file ANNOTATION_FILE]
                [--classes_path CLASSES_PATH] [--weights_path WEIGHTS_PATH]
                [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
                [--freeze_level FREEZE_LEVEL] [--val_split VAL_SPLIT]
                [--model_image_size MODEL_IMAGE_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        YOLO model type: mobilenet_lite/mobilenet/darknet/vgg1
                        6/xception/xception_lite, default=mobilenet_lite
  --tiny_version        Whether to use a tiny YOLO version
  --annotation_file ANNOTATION_FILE
                        train&val annotation txt file, default=trainval.txt
  --classes_path CLASSES_PATH
                        path to class definitions,
                        default=model_data/voc_classes.txt
  --weights_path WEIGHTS_PATH
                        Pretrained model/weights file for fine tune
  --learning_rate LEARNING_RATE
                        Initial learning rate, default=0.001
  --batch_size BATCH_SIZE
                        Initial batch size for train, default=16
  --freeze_level FREEZE_LEVEL
                        Freeze level of the training model. 0:NA/1:backbone
  --val_split VAL_SPLIT
                        validation data persentage in dataset, default=0.1
  --model_image_size MODEL_IMAGE_SIZE
                        model image input size as <num>x<num>, default 416x416
```
Loss type couldn't be changed from CLI options. You can try them by changing params in yolo3/model.py

4. train_multiscale.py
> * Multiscale training script for the supported models

Checkpoints during training could be found at logs/000/. Choose a best one as result

### Model dump
We need to dump out inference model from training checkpoint for eval or demo. Following script cmd work for that.

```
python yolo.py --model_type=mobilenet_lite --model_path=logs/000/<checkpoint>.h5 --anchors_path=model_data/yolo_anchors.txt --classes_path=model_data/voc_classes.txt --dump_model --output_model_file=test.h5
```

Change model_type, anchors file & class file for different training mode

### Evaluation
Use "eval.py" to do evaluation on the inference model with your test data. It support following metrics:

1. Pascal VOC mAP: draw rec/pre curve for each class and AP/mAP result chart in "result" dir with default 0.5 IOU or specified IOU, and optionally save all the detection result on evaluation dataset as images

2. MS COCO AP evaluation. Will draw AP chart and optionally save all the detection result

```
python eval.py --model_path=test.h5 --anchors_path=model_data/yolo_anchors.txt --classes_path=model_data/voc_classes.txt --model_image_size=416x416 --eval_type=VOC --iou_threshold=0.5 --annotation_file=2007_test.txt --save_result
```

### Demo
1. yolo.py
> * Demo script for trained model
image detection mode
```
python yolo.py --model_type=mobilenet_lite --model_path=test.h5 --anchors_path=model_data/yolo_anchors.txt --classes_path=model_data/voc_classes.txt --image
```
video detection mode
```
python yolo.py --model_type=mobilenet_lite --model_path=test.h5 --anchors_path=model_data/yolo_anchors.txt --classes_path=model_data/voc_classes.txt --input=test.mp4
```
For video detection mode, you can use "input=0" to capture live video from web camera and "output=<video name>" to dump out detection result to another video

2. MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).

### TFLite convert & validate
1. Use tflite_convert to generate TFLite inference model. We need to specify input node name and input shape since our inference model doesn't have input image shape. Only valid under tensorflow 1.13
```
tflite_convert [--post_training_quantize] --input_arrays=input_1 --input_shapes=1,416,416,3 --output_file=test[_quant].tflite --keras_model_file=test.h5
```
2. Run TFLite validate script
```
python validate_yolo_tflite.py --model_path=test.tflite --anchors_path=model_data/yolo_anchors.txt --classes_path=model_data/voc_classes.txt --image_file=test.jpg --loop_count=1
```
#### You can also use "eval.py" to do evaluate on the TFLite model

3. TODO item
- [] TFLite C++ inplementation of yolo head


## Some issues to know

1. The test environment is
    - Python 3.6.7
    - tensorflow 1.14.0
    - tf.keras 2.2.4-tf

2. Default YOLOv3 anchors are used. If you want to use your own anchors, probably some changes are needed. tools/kmeans.py could be used to do K-Means anchor clustering on your dataset

3. Always load pretrained weights and freeze layers in the first stage of training.

4. The training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.


# Citation
Please cite keras-YOLOv3-model-set in your publications if it helps your research:
```
@article{MobileNet-Yolov3,
     Author = {Adam Yang},
     Year = {2018}
}
@article{keras-yolo3,
     Author = {qqwweee},
     Year = {2018}
}
@article{yolov3,
     title={YOLOv3: An Incremental Improvement},
     author={Redmon, Joseph and Farhadi, Ali},
     journal = {arXiv},
     year={2018}
}
@article{Focal Loss,
     title={Focal Loss for Dense Object Detection},
     author={Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár},
     journal = {arXiv},
     year={2017}
}
@article{GIoU,
     title={Generalized Intersection over Union: A Metric and A Loss for Bounding Box
Regression},
     author={Hamid Rezatofighi, Nathan Tsoi1, JunYoung Gwak1, Amir Sadeghian, Ian Reid, Silvio Savarese},
     journal = {arXiv},
     year={2019}
}
```
