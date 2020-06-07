#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v2 MobileNetV3Small Model Defined in Keras."""

from tensorflow.keras.layers import MaxPooling2D, Lambda, Concatenate, GlobalAveragePooling2D, Softmax
from tensorflow.keras.models import Model
from common.backbones.mobilenet_v3 import MobileNetV3Small

from yolo2.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, yolo2_predictions, yolo2lite_predictions


def yolo2_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V2 MobileNetV3Small model CNN body in Keras."""
    mobilenetv3small = MobileNetV3Small(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv3small.layers)))

    # input: 416 x 416 x 3
    # mobilenetv3small.output(layer 165, final feature map): 13 x 13 x (576*alpha)
    # expanded_conv_10/Add(layer 162, end of block10): 13 x 13 x (96*alpha)

    # activation_22(layer 117, middle in block8) : 26 x 26 x (288*alpha)
    # expanded_conv_7/Add(layer 114, end of block7) : 26 x 26 x (48*alpha)

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1: 13 x 13 x (960*alpha)
    f1 = mobilenetv3small.output
    # f2: 26 x 26 x (672*alpha)
    f2 = mobilenetv3small.layers[117].output

    f1_channel_num = int(576*alpha)
    f2_channel_num = int(288*alpha)

    y = yolo2_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)
    return Model(inputs, y)


def yolo2lite_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V2 Lite MobileNetV3Small model CNN body in Keras."""
    mobilenetv3small = MobileNetV3Small(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv3small.layers)))

    # input: 416 x 416 x 3
    # mobilenetv3small.output(layer 165, final feature map): 13 x 13 x (576*alpha)
    # expanded_conv_10/Add(layer 162, end of block10): 13 x 13 x (96*alpha)

    # activation_22(layer 117, middle in block8) : 26 x 26 x (288*alpha)
    # expanded_conv_7/Add(layer 114, end of block7) : 26 x 26 x (48*alpha)

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1: 13 x 13 x (960*alpha)
    f1 = mobilenetv3small.output
    # f2: 26 x 26 x (672*alpha)
    f2 = mobilenetv3small.layers[117].output

    f1_channel_num = int(576*alpha)
    f2_channel_num = int(288*alpha)

    y = yolo2lite_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)
    return Model(inputs, y)


def tiny_yolo2_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create Tiny YOLO_V2 MobileNetV3Small model CNN body in Keras."""
    mobilenetv3small = MobileNetV3Small(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv3small.layers)))

    # input: 416 x 416 x 3
    # mobilenetv3small.output(layer 165, final feature map): 13 x 13 x (576*alpha)

    # f1: 13 x 13 x (576*alpha)
    f1 = mobilenetv3small.output
    f1_channel_num = int(576*alpha)

    y = compose(
            DarknetConv2D_BN_Leaky(f1_channel_num, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv'))(f1)

    return Model(inputs, y)


def tiny_yolo2lite_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create Tiny YOLO_V2 Lite MobileNetV3Small model CNN body in Keras."""
    mobilenetv3small = MobileNetV3Small(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv3small.layers)))

    # input: 416 x 416 x 3
    # mobilenetv3small.output(layer 165, final feature map): 13 x 13 x (576*alpha)

    # f1: 13 x 13 x (576*alpha)
    f1 = mobilenetv3small.output
    f1_channel_num = int(576*alpha)

    y = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(f1_channel_num, (3,3), block_id_str='pred_1'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv'))(f1)

    return Model(inputs, y)
