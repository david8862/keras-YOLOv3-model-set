#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v3 Xception Model Defined in Keras."""

import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import Xception
#from yolo3.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, make_last_layers, make_depthwise_separable_last_layers, make_spp_last_layers
from yolo3.models.layers import yolo3_predictions, yolo3lite_predictions, tiny_yolo3_predictions, tiny_yolo3lite_predictions


def yolo3_xception_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 Xception model CNN body in Keras."""
    xception = Xception(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(xception.layers)))

    # input: 416 x 416 x 3
    # block14_sepconv2_act: 13 x 13 x 2048
    # block13_sepconv2_bn(middle in block13): 26 x 26 x 1024
    # add_46(end of block12): 26 x 26 x 728
    # block4_sepconv2_bn(middle in block4) : 52 x 52 x 728
    # add_37(end of block3) : 52 x 52 x 256

    # f1: 13 x 13 x 2048
    f1 = xception.get_layer('block14_sepconv2_act').output
    # f2: 26 x 26 x 1024
    f2 = xception.get_layer('block13_sepconv2_bn').output
    # f3: 52 x 52 x 728
    f3 = xception.get_layer('block4_sepconv2_bn').output

    #f1_channel_num = 2048
    #f2_channel_num = 1024
    #f3_channel_num = 728
    f1_channel_num = 1024
    f2_channel_num = 512
    f3_channel_num = 256

    y1, y2, y3 = yolo3_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yolo3_spp_xception_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 SPP Xception model CNN body in Keras."""
    xception = Xception(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(xception.layers)))

    # input: 416 x 416 x 3
    # block14_sepconv2_act: 13 x 13 x 2048
    # block13_sepconv2_bn(middle in block13): 26 x 26 x 1024
    # add_46(end of block12): 26 x 26 x 728
    # block4_sepconv2_bn(middle in block4) : 52 x 52 x 728
    # add_37(end of block3) : 52 x 52 x 256

    # f1: 13 x 13 x 2048
    f1 = xception.get_layer('block14_sepconv2_act').output
    # f2: 26 x 26 x 1024
    f2 = xception.get_layer('block13_sepconv2_bn').output
    # f3: 52 x 52 x 728
    f3 = xception.get_layer('block4_sepconv2_bn').output

    #f1_channel_num = 2048
    #f2_channel_num = 1024
    #f3_channel_num = 728
    f1_channel_num = 1024
    f2_channel_num = 512
    f3_channel_num = 256

    y1, y2, y3 = yolo3_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes, use_spp=True)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yolo3lite_xception_body(inputs, num_anchors, num_classes):
    '''Create YOLO_v3 Lite Xception model CNN body in keras.'''
    xception = Xception(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(xception.layers)))

    # input: 416 x 416 x 3
    # block14_sepconv2_act: 13 x 13 x 2048
    # block13_sepconv2_bn(middle in block13): 26 x 26 x 1024
    # add_46(end of block12): 26 x 26 x 728
    # block4_sepconv2_bn(middle in block4) : 52 x 52 x 728
    # add_37(end of block3) : 52 x 52 x 256

    # f1: 13 x 13 x 2048
    f1 = xception.get_layer('block14_sepconv2_act').output
    # f2: 26 x 26 x 1024
    f2 = xception.get_layer('block13_sepconv2_bn').output
    # f3: 52 x 52 x 728
    f3 = xception.get_layer('block4_sepconv2_bn').output

    #f1_channel_num = 2048
    #f2_channel_num = 1024
    #f3_channel_num = 728
    f1_channel_num = 1024
    f2_channel_num = 512
    f3_channel_num = 256

    y1, y2, y3 = yolo3lite_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def tiny_yolo3_xception_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 Xception model CNN body in keras.'''
    xception = Xception(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(xception.layers)))

    # input: 416 x 416 x 3
    # block14_sepconv2_act: 13 x 13 x 2048
    # block13_sepconv2_bn(middle in block13): 26 x 26 x 1024
    # add_46(end of block12): 26 x 26 x 728
    # block4_sepconv2_bn(middle in block4) : 52 x 52 x 728
    # add_37(end of block3) : 52 x 52 x 256

    # f1 :13 x 13 x 2048
    f1 = xception.get_layer('block14_sepconv2_act').output
    # f2 :26 x 26 x 1024
    f2 = xception.get_layer('block13_sepconv2_bn').output

    f1_channel_num = 2048
    f2_channel_num = 1024
    #f1_channel_num = 1024
    #f2_channel_num = 512

    y1, y2 = tiny_yolo3_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2])


def tiny_yolo3lite_xception_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 Lite Xception model CNN body in keras.'''
    xception = Xception(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(xception.layers)))

    # input: 416 x 416 x 3
    # block14_sepconv2_act: 13 x 13 x 2048
    # block13_sepconv2_bn(middle in block13): 26 x 26 x 1024
    # add_46(end of block12): 26 x 26 x 728
    # block4_sepconv2_bn(middle in block4) : 52 x 52 x 728
    # add_37(end of block3) : 52 x 52 x 256

    # f1 :13 x 13 x 2048
    f1 = xception.get_layer('block14_sepconv2_act').output
    # f2 :26 x 26 x 1024
    f2 = xception.get_layer('block13_sepconv2_bn').output

    f1_channel_num = 2048
    f2_channel_num = 1024
    #f1_channel_num = 1024
    #f2_channel_num = 512

    y1, y2 = tiny_yolo3lite_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2])

