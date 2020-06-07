#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v2 MobileNetV2 Model Defined in Keras."""

from tensorflow.keras.layers import MaxPooling2D, Lambda, Concatenate, GlobalAveragePooling2D, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from yolo2.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, yolo2_predictions, yolo2lite_predictions


def yolo2_mobilenetv2_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V2 MobileNetV2 model CNN body in Keras."""
    mobilenetv2 = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv2.layers)))

    # input: 416 x 416 x 3
    # mobilenetv2.output   : 13 x 13 x 1280
    # block_13_expand_relu(layers[119]) : 26 x 26 x (576*alpha)

    # f1: 13 x 13 x 1280
    f1 = mobilenetv2.output
    # f2: 26 x 26 x (576*alpha)
    f2 = mobilenetv2.get_layer('block_13_expand_relu').output

    f1_channel_num = 1280
    f2_channel_num = int(576*alpha)

    y = yolo2_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)
    return Model(inputs, y)


def yolo2lite_mobilenetv2_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V2 Lite MobileNetV2 model CNN body in Keras."""
    mobilenetv2 = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv2.layers)))

    # input: 416 x 416 x 3
    # mobilenetv2.output   : 13 x 13 x 1280
    # block_13_expand_relu(layers[119]) : 26 x 26 x (576*alpha)

    # f1: 13 x 13 x 1280
    f1 = mobilenetv2.output
    # f2: 26 x 26 x (576*alpha)
    f2 = mobilenetv2.get_layer('block_13_expand_relu').output

    f1_channel_num = 1280
    f2_channel_num = int(576*alpha)

    y = yolo2lite_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)
    return Model(inputs, y)


def tiny_yolo2_mobilenetv2_body(inputs, num_anchors, num_classes):
    """Create Tiny YOLO_V2 MobileNetV2 model CNN body in Keras."""
    mobilenetv2 = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False, alpha=1.0)
    print('backbone layers number: {}'.format(len(mobilenetv2.layers)))

    # input: 416 x 416 x 3
    # mobilenetv2.output : 13 x 13 x 1280

    # f1: 13 x 13 x 1280
    f1 = mobilenetv2.output
    f1_channel_num = 1280

    y = compose(
            DarknetConv2D_BN_Leaky(f1_channel_num, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv'))(f1)

    return Model(inputs, y)


def tiny_yolo2lite_mobilenetv2_body(inputs, num_anchors, num_classes):
    """Create Tiny YOLO_V2 Lite MobileNetV2 model CNN body in Keras."""
    mobilenetv2 = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False, alpha=1.0)
    print('backbone layers number: {}'.format(len(mobilenetv2.layers)))

    # input: 416 x 416 x 3
    # mobilenetv2.output : 13 x 13 x 1280

    # f1: 13 x 13 x 1280
    f1 = mobilenetv2.output
    f1_channel_num = 1280

    y = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(f1_channel_num, (3,3), block_id_str='pred_1'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv'))(f1)

    return Model(inputs, y)
