#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v5 MobileNetV2 Model Defined in Keras."""

#from tensorflow.keras.layers import ZeroPadding2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from yolo5.models.layers import yolo5_predictions, yolo5lite_predictions, yolo5_spp_neck


def yolo5_mobilenetv2_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V5 MobileNetV2 model CNN body in Keras."""
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


    # add SPP neck with original channel number
    f1 = yolo5_spp_neck(f1, int(1280*alpha))

    # use yolo5_small depth_multiple and width_multiple for head
    depth_multiple = 0.33
    width_multiple = 0.5

    f1_channel_num = int(1280*width_multiple)
    f2_channel_num = int(576*width_multiple)
    f3_channel_num = int(192*width_multiple)

    y1, y2, y3 = yolo5_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes, depth_multiple, width_multiple, with_spp=False)

    return Model(inputs, [y1, y2, y3])



def yolo5lite_mobilenetv2_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V5 Lite MobileNetV2 model CNN body in Keras."""
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


    # add SPP neck with original channel number
    f1 = yolo5_spp_neck(f1, int(1280*alpha))

    # use yolo5_small depth_multiple and width_multiple for head
    depth_multiple = 0.33
    width_multiple = 0.5

    f1_channel_num = int(1280*width_multiple)
    f2_channel_num = int(576*width_multiple)
    f3_channel_num = int(192*width_multiple)

    y1, y2, y3 = yolo5lite_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes, depth_multiple, width_multiple, with_spp=False)

    return Model(inputs, [y1, y2, y3])

