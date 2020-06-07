#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v2 Xception Model Defined in Keras."""

from tensorflow.keras.layers import MaxPooling2D, Lambda, Concatenate, GlobalAveragePooling2D, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import Xception

from yolo2.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, yolo2_predictions, yolo2lite_predictions


def yolo2_xception_body(inputs, num_anchors, num_classes):
    """Create YOLO_V2 Xception model CNN body in Keras."""
    xception = Xception(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(xception.layers)))

    # input: 416 x 416 x 3
    # xception.output: 13 x 13 x 2048
    # block13_sepconv2_bn(middle in block13, layers[121]): 26 x 26 x 1024
    # add_46(end of block12, layers[115]): 26 x 26 x 728

    # f1: 13 x 13 x 2048
    f1 = xception.output
    # f2: 26 x 26 x 1024
    f2 = xception.layers[121].output

    f1_channel_num = 2048
    f2_channel_num = 1024
    #f1_channel_num = 1024
    #f2_channel_num = 512

    y = yolo2_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)
    return Model(inputs, y)


def yolo2lite_xception_body(inputs, num_anchors, num_classes):
    """Create YOLO_V2 Lite Xception model CNN body in Keras."""
    xception = Xception(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(xception.layers)))

    # input: 416 x 416 x 3
    # xception.output: 13 x 13 x 2048
    # block13_sepconv2_bn(middle in block13, layers[121]): 26 x 26 x 1024
    # add_46(end of block12, layers[115]): 26 x 26 x 728

    # f1: 13 x 13 x 2048
    f1 = xception.output
    # f2: 26 x 26 x 1024
    f2 = xception.layers[121].output

    f1_channel_num = 2048
    f2_channel_num = 1024
    #f1_channel_num = 1024
    #f2_channel_num = 512

    y = yolo2lite_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)
    return Model(inputs, y)

