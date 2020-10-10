#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some Ultra-lite structure related layer definitions for YOLOv3 models building
"""
import tensorflow.keras.backend as K
from tensorflow.keras.layers import DepthwiseConv2D, Concatenate
from tensorflow.keras.layers import LeakyReLU, UpSampling2D
from tensorflow.keras.layers import BatchNormalization

from common.backbones.layers import YoloDepthwiseConv2D, CustomBatchNormalization
from yolo3.models.layers import *


def Depthwise_Conv2D_BN_Leaky(kernel_size=(3, 3), block_id_str=None):
    """Depthwise Convolution2D."""
    if not block_id_str:
        block_id_str = str(K.get_uid())
    return compose(
        YoloDepthwiseConv2D(kernel_size, padding='same', name='conv_dw_' + block_id_str),
        CustomBatchNormalization(name='conv_dw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1, name='conv_dw_%s_leaky_relu' % block_id_str)
        )


def tiny_yolo3_ultralite_predictions(feature_maps, feature_channel_nums, num_anchors, num_classes):
    f1, f2 = feature_maps
    f1_channel_num, f2_channel_num = feature_channel_nums

    #feature map 1 transform
    #x1 = DarknetConv2D_BN_Leaky(f1_channel_num//2, (1,1))(f1)
    x1 = f1

    #feature map 1 output (13x13 for 416 input)
    y1 = compose(
            #DarknetConv2D_BN_Leaky(f1_channel_num, (3,3)),
            Depthwise_Conv2D_BN_Leaky(kernel_size=(3, 3), block_id_str='pred_1'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_1'))(x1)

    #upsample fpn merge for feature map 1 & 2
    x2 = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x1)

    #feature map 2 output (26x26 for 416 input)
    y2 = compose(
            Concatenate(),
            #DarknetConv2D_BN_Leaky(f2_channel_num, (3,3)),
            Depthwise_Conv2D_BN_Leaky(kernel_size=(3, 3), block_id_str='pred_2'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_2'))([x2, f2])

    return y1, y2


def make_ultralite_last_layers(x, num_filters, out_filters, block_id_str=None, predict_filters=None, predict_id='1'):
    '''3 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    if not block_id_str:
        block_id_str = str(K.get_uid())
    x = compose(
            #DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=num_filters*2, kernel_size=(3, 3), block_id_str=block_id_str+'_1'),
            #DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=num_filters*2, kernel_size=(3, 3), block_id_str=block_id_str+'_2'),
            #DarknetConv2D_BN_Leaky(num_filters, (1,1))
            Depthwise_Conv2D_BN_Leaky(kernel_size=(3, 3), block_id_str=block_id_str+'_1'),
            )(x)

    if predict_filters is None:
        predict_filters = num_filters*2
    y = compose(
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=predict_filters, kernel_size=(3, 3), block_id_str=block_id_str+'_3'),
            Depthwise_Conv2D_BN_Leaky(kernel_size=(3, 3), block_id_str=block_id_str+'_3'),
            DarknetConv2D(out_filters, (1,1), name='predict_conv_' + predict_id))(x)
    return x, y


def yolo3_ultralite_predictions(feature_maps, feature_channel_nums, num_anchors, num_classes, use_spp=False):
    f1, f2, f3 = feature_maps
    f1_channel_num, f2_channel_num, f3_channel_num = feature_channel_nums

    #feature map 1 head & output (13x13 for 416 input)
    if use_spp:
        x, y1 = make_spp_depthwise_separable_last_layers(f1, f1_channel_num//2, num_anchors * (num_classes + 5), block_id_str='pred_1', predict_id='1')
    else:
        x, y1 = make_ultralite_last_layers(f1, f1_channel_num//2, num_anchors * (num_classes + 5), block_id_str='pred_1', predict_id='1')

    #upsample fpn merge for feature map 1 & 2
    x = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,f2])

    #feature map 2 head & output (26x26 for 416 input)
    x, y2 = make_ultralite_last_layers(x, f2_channel_num//2, num_anchors * (num_classes + 5), block_id_str='pred_2', predict_id='2')

    #upsample fpn merge for feature map 2 & 3
    x = compose(
            DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, f3])

    #feature map 3 head & output (52x52 for 416 input)
    x, y3 = make_ultralite_last_layers(x, f3_channel_num//2, num_anchors * (num_classes + 5), block_id_str='pred_3', predict_id='3')

    return y1, y2, y3
