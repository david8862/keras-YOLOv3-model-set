#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v3 MobileNetV3Small Model Defined in Keras."""

from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from common.backbones.mobilenet_v3 import MobileNetV3Small

from yolo3.models.layers import yolo3_predictions, yolo3lite_predictions, tiny_yolo3_predictions, tiny_yolo3lite_predictions
from yolo3.models.ultralite_layers import yolo3_ultralite_predictions, tiny_yolo3_ultralite_predictions


def yolo3_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V3 MobileNetV3Small model CNN body in Keras."""
    mobilenetv3small = MobileNetV3Small(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv3small.layers)))

    # input: 416 x 416 x 3
    # activation_31(layer 165, final feature map): 13 x 13 x (576*alpha)
    # expanded_conv_10/Add(layer 162, end of block10): 13 x 13 x (96*alpha)

    # activation_22(layer 117, middle in block8) : 26 x 26 x (288*alpha)
    # expanded_conv_7/Add(layer 114, end of block7) : 26 x 26 x (48*alpha)

    # activation_7(layer 38, middle in block3) : 52 x 52 x (96*alpha)
    # expanded_conv_2/Add(layer 35, end of block2): 52 x 52 x (24*alpha)

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1 :13 x 13 x (576*alpha)
    f1 = mobilenetv3small.layers[165].output
    # f2: 26 x 26 x (288*alpha)
    f2 = mobilenetv3small.layers[117].output
    # f3 : 52 x 52 x (96*alpha)
    f3 = mobilenetv3small.layers[38].output

    f1_channel_num = int(576*alpha)
    f2_channel_num = int(288*alpha)
    f3_channel_num = int(96*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512
    #f3_channel_num = 256

    y1, y2, y3 = yolo3_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yolo3lite_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create YOLO_v3 Lite MobileNetV3Small model CNN body in keras.'''
    mobilenetv3small = MobileNetV3Small(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv3small.layers)))

    # input: 416 x 416 x 3
    # activation_31(layer 165, final feature map): 13 x 13 x (576*alpha)
    # expanded_conv_10/Add(layer 162, end of block10): 13 x 13 x (96*alpha)

    # activation_22(layer 117, middle in block8) : 26 x 26 x (288*alpha)
    # expanded_conv_7/Add(layer 114, end of block7) : 26 x 26 x (48*alpha)

    # activation_7(layer 38, middle in block3) : 52 x 52 x (96*alpha)
    # expanded_conv_2/Add(layer 35, end of block2): 52 x 52 x (24*alpha)

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1 :13 x 13 x (576*alpha)
    f1 = mobilenetv3small.layers[165].output
    # f2: 26 x 26 x (288*alpha)
    f2 = mobilenetv3small.layers[117].output
    # f3 : 52 x 52 x (96*alpha)
    f3 = mobilenetv3small.layers[38].output

    f1_channel_num = int(576*alpha)
    f2_channel_num = int(288*alpha)
    f3_channel_num = int(96*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512
    #f3_channel_num = 256

    y1, y2, y3 = yolo3lite_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def tiny_yolo3_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create Tiny YOLO_v3 MobileNetV3Small model CNN body in keras.'''
    mobilenetv3small = MobileNetV3Small(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv3small.layers)))

    # input: 416 x 416 x 3
    # activation_31(layer 165, final feature map): 13 x 13 x (576*alpha)
    # expanded_conv_10/Add(layer 162, end of block10): 13 x 13 x (96*alpha)

    # activation_22(layer 117, middle in block8) : 26 x 26 x (288*alpha)
    # expanded_conv_7/Add(layer 114, end of block7) : 26 x 26 x (48*alpha)

    # activation_7(layer 38, middle in block3) : 52 x 52 x (96*alpha)
    # expanded_conv_2/Add(layer 35, end of block2): 52 x 52 x (24*alpha)

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1 :13 x 13 x (576*alpha)
    f1 = mobilenetv3small.layers[165].output
    # f2: 26 x 26 x (288*alpha)
    f2 = mobilenetv3small.layers[117].output

    f1_channel_num = int(576*alpha)
    f2_channel_num = int(288*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512

    y1, y2 = tiny_yolo3_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2])


def tiny_yolo3lite_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create Tiny YOLO_v3 Lite MobileNetV3Small model CNN body in keras.'''
    mobilenetv3small = MobileNetV3Small(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv3small.layers)))

    # input: 416 x 416 x 3
    # activation_31(layer 165, final feature map): 13 x 13 x (576*alpha)
    # expanded_conv_10/Add(layer 162, end of block10): 13 x 13 x (96*alpha)

    # activation_22(layer 117, middle in block8) : 26 x 26 x (288*alpha)
    # expanded_conv_7/Add(layer 114, end of block7) : 26 x 26 x (48*alpha)

    # activation_7(layer 38, middle in block3) : 52 x 52 x (96*alpha)
    # expanded_conv_2/Add(layer 35, end of block2): 52 x 52 x (24*alpha)

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1 :13 x 13 x (576*alpha)
    f1 = mobilenetv3small.layers[165].output
    # f2: 26 x 26 x (288*alpha)
    f2 = mobilenetv3small.layers[117].output

    f1_channel_num = int(576*alpha)
    f2_channel_num = int(288*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512

    y1, y2 = tiny_yolo3lite_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2])


def yolo3_ultralite_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create YOLO_v3 Ultra-Lite MobileNetV3Small model CNN body in keras.'''
    mobilenetv3small = MobileNetV3Small(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv3small.layers)))

    # input: 416 x 416 x 3
    # activation_31(layer 165, final feature map): 13 x 13 x (576*alpha)
    # expanded_conv_10/Add(layer 162, end of block10): 13 x 13 x (96*alpha)

    # activation_22(layer 117, middle in block8) : 26 x 26 x (288*alpha)
    # expanded_conv_7/Add(layer 114, end of block7) : 26 x 26 x (48*alpha)

    # activation_7(layer 38, middle in block3) : 52 x 52 x (96*alpha)
    # expanded_conv_2/Add(layer 35, end of block2): 52 x 52 x (24*alpha)

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1 :13 x 13 x (576*alpha)
    f1 = mobilenetv3small.layers[165].output
    # f2: 26 x 26 x (288*alpha)
    f2 = mobilenetv3small.layers[117].output
    # f3 : 52 x 52 x (96*alpha)
    f3 = mobilenetv3small.layers[38].output

    f1_channel_num = int(576*alpha)
    f2_channel_num = int(288*alpha)
    f3_channel_num = int(96*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512
    #f3_channel_num = 256

    y1, y2, y3 = yolo3_ultralite_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def tiny_yolo3_ultralite_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create Tiny YOLO_v3 Ultra-Lite MobileNetV3Small model CNN body in keras.'''
    mobilenetv3small = MobileNetV3Small(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv3small.layers)))

    # input: 416 x 416 x 3
    # activation_31(layer 165, final feature map): 13 x 13 x (576*alpha)
    # expanded_conv_10/Add(layer 162, end of block10): 13 x 13 x (96*alpha)

    # activation_22(layer 117, middle in block8) : 26 x 26 x (288*alpha)
    # expanded_conv_7/Add(layer 114, end of block7) : 26 x 26 x (48*alpha)

    # activation_7(layer 38, middle in block3) : 52 x 52 x (96*alpha)
    # expanded_conv_2/Add(layer 35, end of block2): 52 x 52 x (24*alpha)

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1 :13 x 13 x (576*alpha)
    f1 = mobilenetv3small.layers[165].output
    # f2: 26 x 26 x (288*alpha)
    f2 = mobilenetv3small.layers[117].output

    f1_channel_num = int(576*alpha)
    f2_channel_num = int(288*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512

    y1, y2 = tiny_yolo3_ultralite_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2])

