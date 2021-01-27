#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common layer definition for Scaled-YOLOv4 models building
"""
from functools import wraps, reduce

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Concatenate, MaxPooling2D, BatchNormalization, Activation, UpSampling2D, ZeroPadding2D
from tensorflow.keras.regularizers import l2

from common.backbones.layers import YoloConv2D, YoloDepthwiseConv2D, CustomBatchNormalization
from yolo4.models.layers import compose, DarknetConv2D, DarknetDepthwiseConv2D, DarknetConv2D_BN_Mish, mish



def Darknet_Depthwise_Separable_Conv2D_BN_Mish(filters, kernel_size=(3, 3), block_id_str=None, **kwargs):
    """Depthwise Separable Convolution2D."""
    if not block_id_str:
        block_id_str = str(K.get_uid())
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetDepthwiseConv2D(kernel_size, name='conv_dw_' + block_id_str, **no_bias_kwargs),
        CustomBatchNormalization(name='conv_dw_%s_bn' % block_id_str),
        Activation(mish, name='conv_dw_%s_mish' % block_id_str),
        YoloConv2D(filters, (1,1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%s' % block_id_str),
        CustomBatchNormalization(name='conv_pw_%s_bn' % block_id_str),
        Activation(mish, name='conv_pw_%s_mish' % block_id_str))


def Depthwise_Separable_Conv2D_BN_Mish(filters, kernel_size=(3, 3), block_id_str=None):
    """Depthwise Separable Convolution2D."""
    if not block_id_str:
        block_id_str = str(K.get_uid())
    return compose(
        YoloDepthwiseConv2D(kernel_size, padding='same', name='conv_dw_' + block_id_str),
        CustomBatchNormalization(name='conv_dw_%s_bn' % block_id_str),
        Activation(mish, name='conv_dw_%s_mish' % block_id_str),
        YoloConv2D(filters, (1,1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%s' % block_id_str),
        CustomBatchNormalization(name='conv_pw_%s_bn' % block_id_str),
        Activation(mish, name='conv_pw_%s_mish' % block_id_str))


def Spp_Conv2D_BN_Mish(x, num_filters):
    y1 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(x)
    y2 = MaxPooling2D(pool_size=(9,9), strides=(1,1), padding='same')(x)
    y3 = MaxPooling2D(pool_size=(13,13), strides=(1,1), padding='same')(x)

    y = compose(
            Concatenate(),
            DarknetConv2D_BN_Mish(num_filters, (1,1)))([y3, y2, y1, x])
    return y


def make_csp_yolo_head(x, num_filters):
    '''6 Conv2D_BN_Mish layers followed by a Conv2D_linear layer'''
    x = DarknetConv2D_BN_Mish(num_filters, (1,1))(x)
    res_connection = DarknetConv2D_BN_Mish(num_filters, (1,1))(x)

    x = compose(
            DarknetConv2D_BN_Mish(num_filters, (1,1)),
            DarknetConv2D_BN_Mish(num_filters, (3,3)),
            DarknetConv2D_BN_Mish(num_filters, (1,1)),
            DarknetConv2D_BN_Mish(num_filters, (3,3)))(x)

    x = Concatenate()([x , res_connection])
    x = DarknetConv2D_BN_Mish(num_filters, (1,1))(x)

    return x


def make_csp_yolo_spp_head(x, num_filters):
    '''6 Conv2D_BN_Mish layers followed by a Conv2D_linear layer'''
    res_connection = DarknetConv2D_BN_Mish(num_filters, (1,1))(x)

    x = compose(
            DarknetConv2D_BN_Mish(num_filters, (1,1)),
            DarknetConv2D_BN_Mish(num_filters, (3,3)),
            DarknetConv2D_BN_Mish(num_filters, (1,1)))(x)

    x = Spp_Conv2D_BN_Mish(x, num_filters)

    x = DarknetConv2D_BN_Mish(num_filters, (3,3))(x)
    x = Concatenate()([x , res_connection])
    x = DarknetConv2D_BN_Mish(num_filters, (1,1))(x)

    return x


def make_csp_yolo_depthwise_separable_head(x, num_filters, block_id_str=None):
    '''6 Conv2D_BN_Mish layers followed by a Conv2D_linear layer'''
    if not block_id_str:
        block_id_str = str(K.get_uid())

    x = DarknetConv2D_BN_Mish(num_filters, (1,1))(x)
    res_connection = DarknetConv2D_BN_Mish(num_filters, (1,1))(x)

    x = compose(
            DarknetConv2D_BN_Mish(num_filters, (1,1)),
            Depthwise_Separable_Conv2D_BN_Mish(filters=num_filters, kernel_size=(3, 3), block_id_str=block_id_str+'_1'),
            DarknetConv2D_BN_Mish(num_filters, (1,1)),
            Depthwise_Separable_Conv2D_BN_Mish(filters=num_filters, kernel_size=(3, 3), block_id_str=block_id_str+'_2'))(x)

    x = Concatenate()([x , res_connection])
    x = DarknetConv2D_BN_Mish(num_filters, (1,1))(x)

    return x


def make_csp_yolo_spp_depthwise_separable_head(x, num_filters, block_id_str=None):
    '''6 Conv2D_BN_Mish layers followed by a Conv2D_linear layer'''
    if not block_id_str:
        block_id_str = str(K.get_uid())
    res_connection = DarknetConv2D_BN_Mish(num_filters, (1,1))(x)

    x = compose(
            DarknetConv2D_BN_Mish(num_filters, (1,1)),
            Depthwise_Separable_Conv2D_BN_Mish(filters=num_filters, kernel_size=(3, 3), block_id_str=block_id_str+'_1'),
            DarknetConv2D_BN_Mish(num_filters, (1,1)))(x)

    x = Spp_Conv2D_BN_Mish(x, num_filters)

    x = Depthwise_Separable_Conv2D_BN_Mish(filters=num_filters, kernel_size=(3, 3), block_id_str=block_id_str+'_2')(x)
    x = Concatenate()([x , res_connection])
    x = DarknetConv2D_BN_Mish(num_filters, (1,1))(x)

    return x


def scaled_yolo4_csp_predictions(feature_maps, feature_channel_nums, num_anchors, num_classes):
    f1, f2, f3 = feature_maps
    f1_channel_num, f2_channel_num, f3_channel_num = feature_channel_nums

    #feature map 1 head (19x19 for 608 input)
    x1 = make_csp_yolo_spp_head(f1, f1_channel_num//2)

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Mish(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x1)

    x2 = DarknetConv2D_BN_Mish(f2_channel_num//2, (1,1))(f2)
    x2 = Concatenate()([x2, x1_upsample])

    #feature map 2 head (38x38 for 608 input)
    x2 = make_csp_yolo_head(x2, f2_channel_num//2)

    #upsample fpn merge for feature map 2 & 3
    x2_upsample = compose(
            DarknetConv2D_BN_Mish(f3_channel_num//2, (1,1)),
            UpSampling2D(2))(x2)

    x3 = DarknetConv2D_BN_Mish(f3_channel_num//2, (1,1))(f3)
    x3 = Concatenate()([x3, x2_upsample])

    #feature map 3 head & output (76x76 for 608 input)
    #x3, y3 = make_last_layers(x3, f3_channel_num//2, num_anchors*(num_classes+5))
    x3 = make_csp_yolo_head(x3, f3_channel_num//2)
    y3 = compose(
            DarknetConv2D_BN_Mish(f3_channel_num, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_3'))(x3)

    #downsample fpn merge for feature map 3 & 2
    x3_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            DarknetConv2D_BN_Mish(f2_channel_num//2, (3,3), strides=(2,2)))(x3)

    x2 = Concatenate()([x3_downsample, x2])

    #feature map 2 output (38x38 for 608 input)
    #x2, y2 = make_last_layers(x2, 256, num_anchors*(num_classes+5))
    x2 = make_csp_yolo_head(x2, f2_channel_num//2)
    y2 = compose(
            DarknetConv2D_BN_Mish(f2_channel_num, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_2'))(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            DarknetConv2D_BN_Mish(f1_channel_num//2, (3,3), strides=(2,2)))(x2)

    x1 = Concatenate()([x2_downsample, x1])

    #feature map 1 output (19x19 for 608 input)
    #x1, y1 = make_last_layers(x1, f1_channel_num//2, num_anchors*(num_classes+5))
    x1 = make_csp_yolo_head(x1, f1_channel_num//2)
    y1 = compose(
            DarknetConv2D_BN_Mish(f1_channel_num, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_1'))(x1)

    return y1, y2, y3


def scaled_yolo4_csp_lite_predictions(feature_maps, feature_channel_nums, num_anchors, num_classes):
    f1, f2, f3 = feature_maps
    f1_channel_num, f2_channel_num, f3_channel_num = feature_channel_nums

    #feature map 1 head (13 x 13 x f1_channel_num//2 for 416 input)
    x1 = make_csp_yolo_spp_depthwise_separable_head(f1, f1_channel_num//2, block_id_str='pred_1')

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Mish(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x1)

    x2 = DarknetConv2D_BN_Mish(f2_channel_num//2, (1,1))(f2)
    x2 = Concatenate()([x2, x1_upsample])

    #feature map 2 head (26 x 26 x f2_channel_num//2 for 416 input)
    x2 = make_csp_yolo_depthwise_separable_head(x2, f2_channel_num//2, block_id_str='pred_2')

    #upsample fpn merge for feature map 2 & 3
    x2_upsample = compose(
            DarknetConv2D_BN_Mish(f3_channel_num//2, (1,1)),
            UpSampling2D(2))(x2)

    x3 = DarknetConv2D_BN_Mish(f3_channel_num//2, (1,1))(f3)
    x3 = Concatenate()([x3, x2_upsample])

    #feature map 3 head & output (52 x 52 x f3_channel_num for 416 input)
    x3 = make_csp_yolo_depthwise_separable_head(x3, f3_channel_num//2, block_id_str='pred_3')
    y3 = compose(
            Depthwise_Separable_Conv2D_BN_Mish(f3_channel_num, (3,3), block_id_str='pred_3_3'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_3'))(x3)

    #downsample fpn merge for feature map 3 & 2
    x3_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            Darknet_Depthwise_Separable_Conv2D_BN_Mish(f2_channel_num//2, (3,3), strides=(2,2), block_id_str='pred_3_4'))(x3)

    x2 = Concatenate()([x3_downsample, x2])

    #feature map 2 output (26 x 26 x f2_channel_num for 416 input)
    x2 = make_csp_yolo_depthwise_separable_head(x2, f2_channel_num//2, block_id_str='pred_4')
    y2 = compose(
            Depthwise_Separable_Conv2D_BN_Mish(f2_channel_num, (3,3), block_id_str='pred_4_3'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_2'))(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            Darknet_Depthwise_Separable_Conv2D_BN_Mish(f1_channel_num//2, (3,3), strides=(2,2), block_id_str='pred_4_4'))(x2)

    x1 = Concatenate()([x2_downsample, x1])

    #feature map 1 output (13 x 13 x f1_channel_num for 416 input)
    x1 = make_csp_yolo_depthwise_separable_head(x1, f1_channel_num//2, block_id_str='pred_5')
    y1 = compose(
            Depthwise_Separable_Conv2D_BN_Mish(f1_channel_num, (3,3), block_id_str='pred_5_3'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_1'))(x1)

    return y1, y2, y3


def tiny_scaled_yolo4_csp_predictions(feature_maps, feature_channel_nums, num_anchors, num_classes, use_spp):
    f1, f2 = feature_maps
    f1_channel_num, f2_channel_num = feature_channel_nums

    #feature map 1 head (13 x 13 x f1_channel_num//2 for 416 input)
    x1 = DarknetConv2D_BN_Mish(f1_channel_num//2, (1,1))(f1)
    if use_spp:
        x1 = Spp_Conv2D_BN_Mish(x1, f1_channel_num//2)

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Mish(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x1)
    x2 = compose(
            Concatenate(),
            #Depthwise_Separable_Conv2D_BN_Mish(filters=f2_channel_num, kernel_size=(3, 3), block_id_str='15'),
            DarknetConv2D_BN_Mish(f2_channel_num, (3,3)))([x1_upsample, f2])

    #feature map 2 output (26 x 26 x f2_channel_num for 416 input)
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_2')(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            #Darknet_Depthwise_Separable_Conv2D_BN_Mish(f1_channel_num//2, (3,3), strides=(2,2), block_id_str='16'),
            DarknetConv2D_BN_Mish(f1_channel_num//2, (3,3), strides=(2,2)))(x2)
    x1 = compose(
            Concatenate(),
            #Depthwise_Separable_Conv2D_BN_Mish(filters=f1_channel_num, kernel_size=(3, 3), block_id_str='17'),
            DarknetConv2D_BN_Mish(f1_channel_num, (3,3)))([x2_downsample, x1])

    #feature map 1 output (13 x 13 x f1_channel_num for 416 input)
    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_1')(x1)

    return y1, y2


def tiny_scaled_yolo4_csp_lite_predictions(feature_maps, feature_channel_nums, num_anchors, num_classes, use_spp):
    f1, f2 = feature_maps
    f1_channel_num, f2_channel_num = feature_channel_nums

    #feature map 1 head (13 x 13 x f1_channel_num//2 for 416 input)
    x1 = DarknetConv2D_BN_Mish(f1_channel_num//2, (1,1))(f1)
    if use_spp:
        x1 = Spp_Conv2D_BN_Mish(x1, f1_channel_num//2)

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Mish(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x1)
    x2 = compose(
            Concatenate(),
            #DarknetConv2D_BN_Mish(f2_channel_num, (3,3)),
            Depthwise_Separable_Conv2D_BN_Mish(filters=f2_channel_num, kernel_size=(3, 3), block_id_str='pred_1'))([x1_upsample, f2])

    #feature map 2 output (26 x 26 x f2_channel_num for 416 input)
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_2')(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            #DarknetConv2D_BN_Mish(f1_channel_num//2, (3,3), strides=(2,2)),
            Darknet_Depthwise_Separable_Conv2D_BN_Mish(f1_channel_num//2, (3,3), strides=(2,2), block_id_str='pred_2'))(x2)
    x1 = compose(
            Concatenate(),
            #DarknetConv2D_BN_Mish(f1_channel_num, (3,3)),
            Depthwise_Separable_Conv2D_BN_Mish(filters=f1_channel_num, kernel_size=(3, 3), block_id_str='pred_3'))([x2_downsample, x1])

    #feature map 1 output (13 x 13 x f1_channel_num for 416 input)
    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_1')(x1)

    return y1, y2

