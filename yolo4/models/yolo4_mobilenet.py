#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v4 MobileNet Model Defined in Keras."""

from tensorflow.keras.layers import ZeroPadding2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet

from yolo4.models.layers import yolo4_predictions, yolo4lite_predictions, tiny_yolo4_predictions, tiny_yolo4lite_predictions


def yolo4_mobilenet_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V4 MobileNet model CNN body in Keras."""
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

    f1_channel_num = int(1024*alpha)
    f2_channel_num = int(512*alpha)
    f3_channel_num = int(256*alpha)

    y1, y2, y3 = yolo4_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1, y2, y3])


def yolo4lite_mobilenet_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create YOLO_v4 Lite MobileNet model CNN body in keras.'''
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

    f1_channel_num = int(1024*alpha)
    f2_channel_num = int(512*alpha)
    f3_channel_num = int(256*alpha)

    y1, y2, y3 = yolo4lite_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1, y2, y3])


def tiny_yolo4_mobilenet_body(inputs, num_anchors, num_classes, alpha=1.0, use_spp=True):
    '''Create Tiny YOLO_v4 MobileNet model CNN body in keras.'''
    mobilenet = MobileNet(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenet.layers)))

    # input: 416 x 416 x 3
    # conv_pw_13_relu :13 x 13 x (1024*alpha)
    # conv_pw_11_relu :26 x 26 x (512*alpha)
    # conv_pw_5_relu : 52 x 52 x (256*alpha)

    # f1 :13 x 13 x (1024*alpha) for 416 input
    f1 = mobilenet.get_layer('conv_pw_13_relu').output
    # f2: 26 x 26 x (512*alpha) for 416 input
    f2 = mobilenet.get_layer('conv_pw_11_relu').output

    f1_channel_num = int(1024*alpha)
    f2_channel_num = int(512*alpha)

    y1, y2 = tiny_yolo4_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes, use_spp)

    return Model(inputs, [y1,y2])


def tiny_yolo4lite_mobilenet_body(inputs, num_anchors, num_classes, alpha=1.0, use_spp=True):
    '''Create Tiny YOLO_v3 Lite MobileNet model CNN body in keras.'''
    mobilenet = MobileNet(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenet.layers)))

    # input: 416 x 416 x 3
    # conv_pw_13_relu :13 x 13 x (1024*alpha)
    # conv_pw_11_relu :26 x 26 x (512*alpha)
    # conv_pw_5_relu : 52 x 52 x (256*alpha)

    # f1 :13 x 13 x (1024*alpha) for 416 input
    f1 = mobilenet.get_layer('conv_pw_13_relu').output
    # f2: 26 x 26 x (512*alpha) for 416 input
    f2 = mobilenet.get_layer('conv_pw_11_relu').output

    f1_channel_num = int(1024*alpha)
    f2_channel_num = int(512*alpha)

    y1, y2 = tiny_yolo4lite_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes, use_spp)

    return Model(inputs, [y1,y2])

