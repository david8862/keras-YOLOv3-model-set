#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v3 Xception Model Defined in Keras."""

import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import Xception
from yolo3.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, make_last_layers, make_depthwise_separable_last_layers, make_spp_last_layers


def yolo3_xception_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 Xception model CNN body in Keras."""
    xception = Xception(input_tensor=inputs, weights='imagenet', include_top=False)

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

    #feature map 1 head & output (13x13 for 416 input)
    x, y1 = make_last_layers(f1, f1_channel_num//2, num_anchors * (num_classes + 5))

    #upsample fpn merge for feature map 1 & 2
    x = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,f2])

    #feature map 2 head & output (26x26 for 416 input)
    x, y2 = make_last_layers(x, f2_channel_num//2, num_anchors*(num_classes+5))

    #upsample fpn merge for feature map 2 & 3
    x = compose(
            DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, f3])

    #feature map 3 head & output (52x52 for 416 input)
    x, y3 = make_last_layers(x, f3_channel_num//2, num_anchors*(num_classes+5))

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yolo3_spp_xception_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 SPP Xception model CNN body in Keras."""
    xception = Xception(input_tensor=inputs, weights='imagenet', include_top=False)

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

    #feature map 1 head & output (13x13 for 416 input)
    x, y1 = make_spp_last_layers(f1, f1_channel_num//2, num_anchors * (num_classes + 5))

    #upsample fpn merge for feature map 1 & 2
    x = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,f2])

    #feature map 2 head & output (26x26 for 416 input)
    x, y2 = make_last_layers(x, f2_channel_num//2, num_anchors*(num_classes+5))

    #upsample fpn merge for feature map 2 & 3
    x = compose(
            DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, f3])

    #feature map 3 head & output (52x52 for 416 input)
    x, y3 = make_last_layers(x, f3_channel_num//2, num_anchors*(num_classes+5))

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yolo3lite_xception_body(inputs, num_anchors, num_classes):
    '''Create YOLO_v3 Lite Xception model CNN body in keras.'''
    xception = Xception(input_tensor=inputs, weights='imagenet', include_top=False)

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

    #feature map 1 head & output (13x13 for 416 input)
    x, y1 = make_depthwise_separable_last_layers(f1, f1_channel_num//2, num_anchors * (num_classes + 5), block_id_str='14')

    #upsample fpn merge for feature map 1 & 2
    x = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,f2])

    #feature map 2 head & output (26x26 for 416 input)
    x, y2 = make_depthwise_separable_last_layers(x, f2_channel_num//2, num_anchors * (num_classes + 5), block_id_str='15')

    #upsample fpn merge for feature map 2 & 3
    x = compose(
            DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, f3])

    #feature map 3 head & output (52x52 for 416 input)
    x, y3 = make_depthwise_separable_last_layers(x, f3_channel_num//2, num_anchors * (num_classes + 5), block_id_str='16')

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def tiny_yolo3_xception_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 Xception model CNN body in keras.'''
    xception = Xception(input_tensor=inputs, weights='imagenet', include_top=False)

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

    #feature map 1 transform
    x1 = DarknetConv2D_BN_Leaky(f1_channel_num//2, (1,1))(f1)

    #feature map 1 output (13x13 for 416 input)
    y1 = compose(
            DarknetConv2D_BN_Leaky(f1_channel_num, (3,3)),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=f1_channel_num, kernel_size=(3, 3), block_id_str='14'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x1)

    #upsample fpn merge for feature map 1 & 2
    x2 = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x1)

    #feature map 2 output (26x26 for 416 input)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(f2_channel_num, (3,3)),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=f2_channel_num, kernel_size=(3, 3), block_id_str='15'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2, f2])

    return Model(inputs, [y1,y2])


def tiny_yolo3lite_xception_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 Lite Xception model CNN body in keras.'''
    xception = Xception(input_tensor=inputs, weights='imagenet', include_top=False)

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

    #feature map 1 transform
    x1 = DarknetConv2D_BN_Leaky(f1_channel_num//2, (1,1))(f1)

    #feature map 1 output (13x13 for 416 input)
    y1 = compose(
            #DarknetConv2D_BN_Leaky(f1_channel_num, (3,3)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=f1_channel_num, kernel_size=(3, 3), block_id_str='14'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x1)

    #upsample fpn merge for feature map 1 & 2
    x2 = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x1)

    #feature map 2 output (26x26 for 416 input)
    y2 = compose(
            Concatenate(),
            #DarknetConv2D_BN_Leaky(f2_channel_num, (3,3)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=f2_channel_num, kernel_size=(3, 3), block_id_str='15'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2, f2])

    return Model(inputs, [y1,y2])

