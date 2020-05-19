#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v3 ShuffleNetV2 Model Defined in Keras."""

from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from common.backbones.shufflenet_v2 import ShuffleNetV2

#from yolo3.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, make_last_layers, make_depthwise_separable_last_layers, make_spp_depthwise_separable_last_layers
from yolo3.models.layers import yolo3_predictions, yolo3lite_predictions, tiny_yolo3_predictions, tiny_yolo3lite_predictions


def yolo3_shufflenetv2_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 ShuffleNetV2 model CNN body in Keras."""
    shufflenetv2 = ShuffleNetV2(input_tensor=inputs, weights=None, include_top=False)
    print('backbone layers number: {}'.format(len(shufflenetv2.layers)))

    # input: 416 x 416 x 3
    # 1x1conv5_out: 13 x 13 x 1024
    # stage4/block1/relu_1x1conv_1: 26 x 26 x 464
    # stage3/block1/relu_1x1conv_1: 52 x 52 x 232

    # f1: 13 x 13 x 1024
    f1 = shufflenetv2.get_layer('1x1conv5_out').output
    # f2: 26 x 26 x 464
    f2 = shufflenetv2.get_layer('stage4/block1/relu_1x1conv_1').output
    # f3: 52 x 52 x 232
    f3 = shufflenetv2.get_layer('stage3/block1/relu_1x1conv_1').output

    f1_channel_num = 1024
    f2_channel_num = 464
    f3_channel_num = 232
    #f1_channel_num = 1024
    #f2_channel_num = 512
    #f3_channel_num = 256

    y1, y2, y3 = yolo3_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yolo3lite_shufflenetv2_body(inputs, num_anchors, num_classes):
    '''Create YOLO_v3 Lite ShuffleNetV2 model CNN body in keras.'''
    shufflenetv2 = ShuffleNetV2(input_tensor=inputs, weights=None, include_top=False)
    print('backbone layers number: {}'.format(len(shufflenetv2.layers)))

    # input: 416 x 416 x 3
    # 1x1conv5_out: 13 x 13 x 1024
    # stage4/block1/relu_1x1conv_1: 26 x 26 x 464
    # stage3/block1/relu_1x1conv_1: 52 x 52 x 232

    # f1: 13 x 13 x 1024
    f1 = shufflenetv2.get_layer('1x1conv5_out').output
    # f2: 26 x 26 x 464
    f2 = shufflenetv2.get_layer('stage4/block1/relu_1x1conv_1').output
    # f3: 52 x 52 x 232
    f3 = shufflenetv2.get_layer('stage3/block1/relu_1x1conv_1').output

    f1_channel_num = 1024
    f2_channel_num = 464
    f3_channel_num = 232
    #f1_channel_num = 1024
    #f2_channel_num = 512
    #f3_channel_num = 256

    y1, y2, y3 = yolo3lite_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yolo3lite_spp_shufflenetv2_body(inputs, num_anchors, num_classes):
    '''Create YOLO_v3 Lite SPP ShuffleNetV2 model CNN body in keras.'''
    shufflenetv2 = ShuffleNetV2(input_tensor=inputs, weights=None, include_top=False)
    print('backbone layers number: {}'.format(len(shufflenetv2.layers)))

    # input: 416 x 416 x 3
    # 1x1conv5_out: 13 x 13 x 1024
    # stage4/block1/relu_1x1conv_1: 26 x 26 x 464
    # stage3/block1/relu_1x1conv_1: 52 x 52 x 232

    # f1: 13 x 13 x 1024
    f1 = shufflenetv2.get_layer('1x1conv5_out').output
    # f2: 26 x 26 x 464
    f2 = shufflenetv2.get_layer('stage4/block1/relu_1x1conv_1').output
    # f3: 52 x 52 x 232
    f3 = shufflenetv2.get_layer('stage3/block1/relu_1x1conv_1').output

    f1_channel_num = 1024
    f2_channel_num = 464
    f3_channel_num = 232
    #f1_channel_num = 1024
    #f2_channel_num = 512
    #f3_channel_num = 256

    y1, y2, y3 = yolo3lite_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes, use_spp=True)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def tiny_yolo3_shufflenetv2_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 ShuffleNetV2 model CNN body in keras.'''
    shufflenetv2 = ShuffleNetV2(input_tensor=inputs, weights=None, include_top=False)
    print('backbone layers number: {}'.format(len(shufflenetv2.layers)))

    # input: 416 x 416 x 3
    # 1x1conv5_out: 13 x 13 x 1024
    # stage4/block1/relu_1x1conv_1: 26 x 26 x 464
    # stage3/block1/relu_1x1conv_1: 52 x 52 x 232

    # f1: 13 x 13 x 1024
    f1 = shufflenetv2.get_layer('1x1conv5_out').output
    # f2: 26 x 26 x 464
    f2 = shufflenetv2.get_layer('stage4/block1/relu_1x1conv_1').output

    f1_channel_num = 1024
    f2_channel_num = 464
    #f1_channel_num = 1024
    #f2_channel_num = 512

    y1, y2 = tiny_yolo3_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2])


def tiny_yolo3lite_shufflenetv2_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 Lite ShuffleNetV2 model CNN body in keras.'''
    shufflenetv2 = ShuffleNetV2(input_tensor=inputs, weights=None, include_top=False)
    print('backbone layers number: {}'.format(len(shufflenetv2.layers)))

    # input: 416 x 416 x 3
    # 1x1conv5_out: 13 x 13 x 1024
    # stage4/block1/relu_1x1conv_1: 26 x 26 x 464
    # stage3/block1/relu_1x1conv_1: 52 x 52 x 232

    # f1: 13 x 13 x 1024
    f1 = shufflenetv2.get_layer('1x1conv5_out').output
    # f2: 26 x 26 x 464
    f2 = shufflenetv2.get_layer('stage4/block1/relu_1x1conv_1').output

    f1_channel_num = 1024
    f2_channel_num = 464
    #f1_channel_num = 1024
    #f2_channel_num = 512

    y1, y2 = tiny_yolo3lite_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2])

