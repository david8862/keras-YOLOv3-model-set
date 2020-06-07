#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v2 MobileNet Model Defined in Keras."""

from tensorflow.keras.layers import MaxPooling2D, Lambda, Concatenate, GlobalAveragePooling2D, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet

from yolo2.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, yolo2_predictions, yolo2lite_predictions


def yolo2_mobilenet_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V2 MobileNet model CNN body in Keras."""
    mobilenet = MobileNet(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenet.layers)))

    # input: 416 x 416 x 3
    # mobilenet.output            : 13 x 13 x (1024*alpha)
    # conv_pw_11_relu(layers[73]) : 26 x 26 x (512*alpha)

    # f1: 13 x 13 x (1024*alpha)
    f1 = mobilenet.output
    # f2: 26 x 26 x (512*alpha)
    f2 = mobilenet.get_layer('conv_pw_11_relu').output

    f1_channel_num = int(1024*alpha)
    f2_channel_num = int(512*alpha)

    y = yolo2_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)
    return Model(inputs, y)


def yolo2lite_mobilenet_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V2 Lite MobileNet model CNN body in Keras."""
    mobilenet = MobileNet(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenet.layers)))

    # input: 416 x 416 x 3
    # mobilenet.output            : 13 x 13 x (1024*alpha)
    # conv_pw_11_relu(layers[73]) : 26 x 26 x (512*alpha)

    # f1: 13 x 13 x (1024*alpha)
    f1 = mobilenet.output
    # f2: 26 x 26 x (512*alpha)
    f2 = mobilenet.get_layer('conv_pw_11_relu').output

    f1_channel_num = int(1024*alpha)
    f2_channel_num = int(512*alpha)

    y = yolo2lite_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)
    return Model(inputs, y)


def tiny_yolo2_mobilenet_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create Tiny YOLO_V2 MobileNet model CNN body in Keras."""
    mobilenet = MobileNet(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenet.layers)))

    # input: 416 x 416 x 3
    # mobilenet.output : 13 x 13 x (1024*alpha)

    # f1: 13 x 13 x (1024*alpha)
    f1 = mobilenet.output
    f1_channel_num = int(1024*alpha)

    y = compose(
            DarknetConv2D_BN_Leaky(f1_channel_num, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv'))(f1)

    return Model(inputs, y)


def tiny_yolo2lite_mobilenet_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create Tiny YOLO_V2 Lite MobileNet model CNN body in Keras."""
    mobilenet = MobileNet(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenet.layers)))

    # input: 416 x 416 x 3
    # mobilenet.output : 13 x 13 x (1024*alpha)

    # f1: 13 x 13 x (1024*alpha)
    f1 = mobilenet.output
    f1_channel_num = int(1024*alpha)

    y = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(f1_channel_num, (3,3), block_id_str='pred_1'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv'))(f1)

    return Model(inputs, y)
