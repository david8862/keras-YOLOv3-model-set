#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v4 MobileNetV3Large Model Defined in Keras."""

from tensorflow.keras.layers import ZeroPadding2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from common.backbones.mobilenet_v3 import MobileNetV3Large

#from yolo4.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Spp_Conv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, Darknet_Depthwise_Separable_Conv2D_BN_Leaky, make_yolo_head, make_yolo_spp_head, make_yolo_depthwise_separable_head, make_yolo_spp_depthwise_separable_head
from yolo4.models.layers import yolo4_predictions, yolo4lite_predictions, tiny_yolo4_predictions, tiny_yolo4lite_predictions


def yolo4_mobilenetv3large_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V4 MobileNetV3Large model CNN body in Keras."""
    mobilenetv3large = MobileNetV3Large(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv3large.layers)))

    # input: 416 x 416 x 3
    # activation_38(layer 194, final feature map): 13 x 13 x (960*alpha)
    # expanded_conv_14/Add(layer 191, end of block14): 13 x 13 x (160*alpha)

    # activation_29(layer 146, middle in block12) : 26 x 26 x (672*alpha)
    # expanded_conv_11/Add(layer 143, end of block11) : 26 x 26 x (112*alpha)

    # activation_15(layer 79, middle in block6) : 52 x 52 x (240*alpha)
    # expanded_conv_5/Add(layer 76, end of block5): 52 x 52 x (40*alpha)

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1: 13 x 13 x (960*alpha)
    f1 = mobilenetv3large.layers[194].output
    # f2: 26 x 26 x (672*alpha) for 416 input
    f2 = mobilenetv3large.layers[146].output
    # f3: 52 x 52 x (240*alpha) for 416 input
    f3 = mobilenetv3large.layers[79].output

    f1_channel_num = int(960*alpha)
    f2_channel_num = int(672*alpha)
    f3_channel_num = int(240*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512
    #f3_channel_num = 256

    y1, y2, y3 = yolo4_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1, y2, y3])



def yolo4lite_mobilenetv3large_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create YOLO_v4 Lite MobileNetV3Large model CNN body in keras.'''
    mobilenetv3large = MobileNetV3Large(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv3large.layers)))

    # input: 416 x 416 x 3
    # activation_38(layer 194, final feature map): 13 x 13 x (960*alpha)
    # expanded_conv_14/Add(layer 191, end of block14): 13 x 13 x (160*alpha)

    # activation_29(layer 146, middle in block12) : 26 x 26 x (672*alpha)
    # expanded_conv_11/Add(layer 143, end of block11) : 26 x 26 x (112*alpha)

    # activation_15(layer 79, middle in block6) : 52 x 52 x (240*alpha)
    # expanded_conv_5/Add(layer 76, end of block5): 52 x 52 x (40*alpha)

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1: 13 x 13 x (960*alpha)
    f1 = mobilenetv3large.layers[194].output
    # f2: 26 x 26 x (672*alpha) for 416 input
    f2 = mobilenetv3large.layers[146].output
    # f3: 52 x 52 x (240*alpha) for 416 input
    f3 = mobilenetv3large.layers[79].output

    f1_channel_num = int(960*alpha)
    f2_channel_num = int(672*alpha)
    f3_channel_num = int(240*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512
    #f3_channel_num = 256

    y1, y2, y3 = yolo4lite_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1, y2, y3])


def tiny_yolo4_mobilenetv3large_body(inputs, num_anchors, num_classes, alpha=1.0, use_spp=True):
    '''Create Tiny YOLO_v4 MobileNetV3Large model CNN body in keras.'''
    mobilenetv3large = MobileNetV3Large(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv3large.layers)))

    # input: 416 x 416 x 3
    # activation_38(layer 194, final feature map): 13 x 13 x (960*alpha)
    # expanded_conv_14/Add(layer 191, end of block14): 13 x 13 x (160*alpha)

    # activation_29(layer 146, middle in block12) : 26 x 26 x (672*alpha)
    # expanded_conv_11/Add(layer 143, end of block11) : 26 x 26 x (112*alpha)

    # activation_15(layer 79, middle in block6) : 52 x 52 x (240*alpha)
    # expanded_conv_5/Add(layer 76, end of block5): 52 x 52 x (40*alpha)

    # f1 :13 x 13 x (960*alpha)
    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    f1 = mobilenetv3large.layers[194].output
    # f2: 26 x 26 x (672*alpha) for 416 input
    f2 = mobilenetv3large.layers[146].output

    f1_channel_num = int(960*alpha)
    f2_channel_num = int(672*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512

    y1, y2 = tiny_yolo4_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes, use_spp)

    return Model(inputs, [y1,y2])


def tiny_yolo4lite_mobilenetv3large_body(inputs, num_anchors, num_classes, alpha=1.0, use_spp=True):
    '''Create Tiny YOLO_v4 Lite MobileNetV3Large model CNN body in keras.'''
    mobilenetv3large = MobileNetV3Large(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv3large.layers)))

    # input: 416 x 416 x 3
    # activation_38(layer 194, final feature map): 13 x 13 x (960*alpha)
    # expanded_conv_14/Add(layer 191, end of block14): 13 x 13 x (160*alpha)

    # activation_29(layer 146, middle in block12) : 26 x 26 x (672*alpha)
    # expanded_conv_11/Add(layer 143, end of block11) : 26 x 26 x (112*alpha)

    # activation_15(layer 79, middle in block6) : 52 x 52 x (240*alpha)
    # expanded_conv_5/Add(layer 76, end of block5): 52 x 52 x (40*alpha)

    # f1 :13 x 13 x (960*alpha)
    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    f1 = mobilenetv3large.layers[194].output
    # f2: 26 x 26 x (672*alpha) for 416 input
    f2 = mobilenetv3large.layers[146].output

    f1_channel_num = int(960*alpha)
    f2_channel_num = int(672*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512

    y1, y2 = tiny_yolo4lite_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes, use_spp)

    return Model(inputs, [y1,y2])

