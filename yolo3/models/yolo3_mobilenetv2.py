#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v3 MobileNetV2 Model Defined in Keras."""

from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.models import Model
#from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from common.backbones.mobilenet_v2 import MobileNetV2

from yolo3.models.layers import yolo3_predictions, yolo3lite_predictions, tiny_yolo3_predictions, tiny_yolo3lite_predictions
from yolo3.models.ultralite_layers import yolo3_ultralite_predictions, tiny_yolo3_ultralite_predictions


def yolo3_mobilenetv2_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V3 MobileNetV2 model CNN body in Keras."""
    mobilenetv2 = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv2.layers)))

    # input: 416 x 416 x 3
    # out_relu: 13 x 13 x 1280
    # block_13_expand_relu: 26 x 26 x (576*alpha)
    # block_6_expand_relu: 52 x 52 x (192*alpha)

    # f1 :13 x 13 x 1280
    f1 = mobilenetv2.get_layer('out_relu').output
    # f2: 26 x 26 x (576*alpha)
    f2 = mobilenetv2.get_layer('block_13_expand_relu').output
    # f3 : 52 x 52 x (192*alpha)
    f3 = mobilenetv2.get_layer('block_6_expand_relu').output

    f1_channel_num = int(1280*alpha)
    f2_channel_num = int(576*alpha)
    f3_channel_num = int(192*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512
    #f3_channel_num = 256

    y1, y2, y3 = yolo3_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yolo3lite_mobilenetv2_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create YOLO_v3 Lite MobileNetV2 model CNN body in keras.'''
    mobilenetv2 = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv2.layers)))

    # input: 416 x 416 x 3
    # out_relu: 13 x 13 x 1280
    # block_13_expand_relu: 26 x 26 x (576*alpha)
    # block_6_expand_relu: 52 x 52 x (192*alpha)

    # f1 :13 x 13 x 1280
    f1 = mobilenetv2.get_layer('out_relu').output
    # f2: 26 x 26 x (576*alpha)
    f2 = mobilenetv2.get_layer('block_13_expand_relu').output
    # f3 : 52 x 52 x (192*alpha)
    f3 = mobilenetv2.get_layer('block_6_expand_relu').output

    f1_channel_num = int(1280*alpha)
    f2_channel_num = int(576*alpha)
    f3_channel_num = int(192*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512
    #f3_channel_num = 256

    y1, y2, y3 = yolo3lite_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yolo3lite_spp_mobilenetv2_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create YOLO_v3 Lite SPP MobileNetV2 model CNN body in keras.'''
    mobilenetv2 = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv2.layers)))

    # input: 416 x 416 x 3
    # out_relu: 13 x 13 x 1280
    # block_13_expand_relu: 26 x 26 x (576*alpha)
    # block_6_expand_relu: 52 x 52 x (192*alpha)

    # f1 :13 x 13 x 1280
    f1 = mobilenetv2.get_layer('out_relu').output
    # f2: 26 x 26 x (576*alpha)
    f2 = mobilenetv2.get_layer('block_13_expand_relu').output
    # f3 : 52 x 52 x (192*alpha)
    f3 = mobilenetv2.get_layer('block_6_expand_relu').output

    f1_channel_num = int(1280*alpha)
    f2_channel_num = int(576*alpha)
    f3_channel_num = int(192*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512
    #f3_channel_num = 256

    y1, y2, y3 = yolo3lite_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes, use_spp=True)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def tiny_yolo3_mobilenetv2_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create Tiny YOLO_v3 MobileNetV2 model CNN body in keras.'''
    mobilenetv2 = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv2.layers)))

    # input: 416 x 416 x 3
    # out_relu: 13 x 13 x 1280
    # block_13_expand_relu: 26 x 26 x (576*alpha)
    # block_6_expand_relu: 52 x 52 x (192*alpha)

    # f1 :13 x 13 x 1280
    f1 = mobilenetv2.get_layer('out_relu').output
    # f2: 26 x 26 x (576*alpha)
    f2 = mobilenetv2.get_layer('block_13_expand_relu').output

    f1_channel_num = int(1280*alpha)
    f2_channel_num = int(576*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512

    y1, y2 = tiny_yolo3_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2])


def tiny_yolo3lite_mobilenetv2_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create Tiny YOLO_v3 Lite MobileNetV2 model CNN body in keras.'''
    mobilenetv2 = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv2.layers)))

    # input: 416 x 416 x 3
    # out_relu: 13 x 13 x 1280
    # block_13_expand_relu: 26 x 26 x (576*alpha)
    # block_6_expand_relu: 52 x 52 x (192*alpha)

    # f1 :13 x 13 x 1280
    f1 = mobilenetv2.get_layer('out_relu').output
    # f2: 26 x 26 x (576*alpha)
    f2 = mobilenetv2.get_layer('block_13_expand_relu').output

    f1_channel_num = int(1280*alpha)
    f2_channel_num = int(576*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512

    y1, y2 = tiny_yolo3lite_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2])


def yolo3_ultralite_mobilenetv2_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create YOLO_v3 Ultra-Lite MobileNetV2 model CNN body in keras.'''
    mobilenetv2 = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv2.layers)))

    # input: 416 x 416 x 3
    # out_relu: 13 x 13 x 1280
    # block_13_expand_relu: 26 x 26 x (576*alpha)
    # block_6_expand_relu: 52 x 52 x (192*alpha)

    # f1 :13 x 13 x 1280
    f1 = mobilenetv2.get_layer('out_relu').output
    # f2: 26 x 26 x (576*alpha)
    f2 = mobilenetv2.get_layer('block_13_expand_relu').output
    # f3 : 52 x 52 x (192*alpha)
    f3 = mobilenetv2.get_layer('block_6_expand_relu').output

    f1_channel_num = int(1280*alpha)
    f2_channel_num = int(576*alpha)
    f3_channel_num = int(192*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512
    #f3_channel_num = 256

    y1, y2, y3 = yolo3_ultralite_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def tiny_yolo3_ultralite_mobilenetv2_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create Tiny YOLO_v3 Ultra-Lite MobileNetV2 model CNN body in keras.'''
    mobilenetv2 = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv2.layers)))

    # input: 416 x 416 x 3
    # out_relu: 13 x 13 x 1280
    # block_13_expand_relu: 26 x 26 x (576*alpha)
    # block_6_expand_relu: 52 x 52 x (192*alpha)

    # f1 :13 x 13 x 1280
    f1 = mobilenetv2.get_layer('out_relu').output
    # f2: 26 x 26 x (576*alpha)
    f2 = mobilenetv2.get_layer('block_13_expand_relu').output

    f1_channel_num = int(1280*alpha)
    f2_channel_num = int(576*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512

    y1, y2 = tiny_yolo3_ultralite_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2])

