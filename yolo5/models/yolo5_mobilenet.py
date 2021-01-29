#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v5 MobileNet Model Defined in Keras."""

#from tensorflow.keras.layers import ZeroPadding2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet

from yolo5.models.layers import yolo5_predictions, yolo5lite_predictions, yolo5_spp_neck


def yolo5_mobilenet_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V5 MobileNet model CNN body in Keras."""
    mobilenet = MobileNet(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenet.layers)))

    # input: 416 x 416 x 3
    # conv_pw_13_relu :13 x 13 x (1024*alpha)
    # conv_pw_11_relu :26 x 26 x (512*alpha)
    # conv_pw_5_relu : 52 x 52 x (256*alpha)

    # f1: 13 x 13 x (1024*alpha) for 416 input
    f1 = mobilenet.get_layer('conv_pw_13_relu').output
    # f2: 26 x 26 x (512*alpha) for 416 input
    f2 = mobilenet.get_layer('conv_pw_11_relu').output
    # f3: 52 x 52 x (256*alpha) for 416 input
    f3 = mobilenet.get_layer('conv_pw_5_relu').output

    # add SPP neck with original channel number
    f1 = yolo5_spp_neck(f1, int(1024*alpha))

    # use yolo5_small depth_multiple and width_multiple for head
    depth_multiple = 0.33
    width_multiple = 0.5

    f1_channel_num = int(1024*width_multiple)
    f2_channel_num = int(512*width_multiple)
    f3_channel_num = int(256*width_multiple)

    y1, y2, y3 = yolo5_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes, depth_multiple, width_multiple, with_spp=False)

    return Model(inputs, [y1, y2, y3])


def yolo5lite_mobilenet_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V5 Lite MobileNet model CNN body in Keras."""
    mobilenet = MobileNet(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenet.layers)))

    # input: 416 x 416 x 3
    # conv_pw_13_relu :13 x 13 x (1024*alpha)
    # conv_pw_11_relu :26 x 26 x (512*alpha)
    # conv_pw_5_relu : 52 x 52 x (256*alpha)

    # f1: 13 x 13 x (1024*alpha) for 416 input
    f1 = mobilenet.get_layer('conv_pw_13_relu').output
    # f2: 26 x 26 x (512*alpha) for 416 input
    f2 = mobilenet.get_layer('conv_pw_11_relu').output
    # f3: 52 x 52 x (256*alpha) for 416 input
    f3 = mobilenet.get_layer('conv_pw_5_relu').output

    # add SPP neck with original channel number
    f1 = yolo5_spp_neck(f1, int(1024*alpha))

    # use yolo5_small depth_multiple and width_multiple for head
    depth_multiple = 0.33
    width_multiple = 0.5

    f1_channel_num = int(1024*width_multiple)
    f2_channel_num = int(512*width_multiple)
    f3_channel_num = int(256*width_multiple)

    y1, y2, y3 = yolo5lite_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes, depth_multiple, width_multiple, with_spp=False)

    return Model(inputs, [y1, y2, y3])

