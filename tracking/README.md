# DeepSORT Multi Object Tracking with YOLO Detection Model

## Introduction

A simple demo of DeepSORT MOT(Multi Object Tracking) implementation using YOLO detection model family, which is ported from [deep_sort_yolov3](https://github.com/Qidian213/deep_sort_yolov3).


## Quick Start

1. Train a YOLOv4/v3/v2 detection model and dump out inference model file or convert from Darknet weights, as [README](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/README.md) guided.

2. Run demo script [deepsort.py](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/tracking/deepsort.py). Input could be a video file or a folder containing sequential images as video frames (image file name should be ordered):

```
# python deepsort.py -h
usage: deepsort.py [-h] [--model_type MODEL_TYPE]
                   [--weights_path WEIGHTS_PATH] [--pruning_model]
                   [--anchors_path ANCHORS_PATH] [--classes_path CLASSES_PATH]
                   [--model_image_size MODEL_IMAGE_SIZE] [--score SCORE]
                   [--iou IOU] [--elim_grid_sense]
                   [--deepsort_model_path DEEPSORT_MODEL_PATH]
                   [--tracking_classes_path TRACKING_CLASSES_PATH]
                   [--input [INPUT]] [--output [OUTPUT]]

Demo of deepsort multi object tracking with YOLO detection model

optional arguments:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        YOLO model type: yolo3_mobilenet_lite/tiny_yolo3_mobil
                        enet/yolo3_darknet/..., default tiny_yolo3_darknet
  --weights_path WEIGHTS_PATH
                        path to YOLO model weight file, default
                        weights/yolov3-tiny.h5
  --pruning_model       Whether using a pruning YOLO model/weights file,
                        default False
  --anchors_path ANCHORS_PATH
                        path to YOLO anchor definitions, default
                        configs/tiny_yolo3_anchors.txt
  --classes_path CLASSES_PATH
                        path to YOLO detection class definitions, default
                        configs/coco_classes.txt
  --model_image_size MODEL_IMAGE_SIZE
                        YOLO detection model input size as <height>x<width>,
                        default 416x416
  --score SCORE         score threshold for YOLO detection model, default 0.1
  --iou IOU             iou threshold for YOLO detection NMS, default 0.4
  --elim_grid_sense     Whether to apply eliminate grid sensitivity in YOLO,
                        default False
  --deepsort_model_path DEEPSORT_MODEL_PATH
                        DeepSORT encoder model path, default=model/mars-
                        small128.pb
  --tracking_classes_path TRACKING_CLASSES_PATH
                        [Optional] Path to DeepSORT tracking class
                        definitions, default=None
  --input [INPUT]       Input video file or images folder path
  --output [OUTPUT]     [Optional] output video file path
```

## Train deep association metric model

  To train the deep association metric model on your datasets, you can reference to [cosine_metric_learning](https://github.com/nwojke/cosine_metric_learning) from original author.


# Citation
```
@article{deep_sort_yolov3,
     Author = {Qidian213},
     Year = {2020}
}

@inproceedings{Wojke2017simple,
  title={Simple Online and Realtime Tracking with a Deep Association Metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
  year={2017},
  pages={3645--3649},
  organization={IEEE},
  doi={10.1109/ICIP.2017.8296962}
}

@inproceedings{Wojke2018deep,
  title={Deep Cosine Metric Learning for Person Re-identification},
  author={Wojke, Nicolai and Bewley, Alex},
  booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2018},
  pages={748--756},
  organization={IEEE},
  doi={10.1109/WACV.2018.00087}
}
```
