#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common layer definition for Scaled-YOLOv4 & YOLOv5 models building
"""
import math
from functools import wraps, reduce

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, Concatenate, MaxPooling2D, BatchNormalization, Activation, UpSampling2D, ZeroPadding2D, Lambda
from tensorflow.keras.regularizers import l2

from common.backbones.layers import YoloConv2D, YoloDepthwiseConv2D, CustomBatchNormalization
from common.backbones.efficientnet import swish
from yolo4.models.layers import compose, DarknetConv2D, DarknetDepthwiseConv2D


def make_divisible(x, divisor):
    # Returns x evenly divisble by divisor
    return math.ceil(x / divisor) * divisor


def DarknetConv2D_BN_Swish(*args, **kwargs):
    """Darknet Convolution2D followed by CustomBatchNormalization and Swish."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        CustomBatchNormalization(),
        Activation(swish))


def Darknet_Depthwise_Separable_Conv2D_BN_Swish(filters, kernel_size=(3, 3), block_id_str=None, **kwargs):
    """Depthwise Separable Convolution2D."""
    if not block_id_str:
        block_id_str = str(K.get_uid())
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetDepthwiseConv2D(kernel_size, name='conv_dw_' + block_id_str, **no_bias_kwargs),
        CustomBatchNormalization(name='conv_dw_%s_bn' % block_id_str),
        Activation(swish, name='conv_dw_%s_swish' % block_id_str),
        YoloConv2D(filters, (1,1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%s' % block_id_str),
        CustomBatchNormalization(name='conv_pw_%s_bn' % block_id_str),
        Activation(swish, name='conv_pw_%s_swish' % block_id_str))


def Depthwise_Separable_Conv2D_BN_Swish(filters, kernel_size=(3, 3), block_id_str=None):
    """Depthwise Separable Convolution2D."""
    if not block_id_str:
        block_id_str = str(K.get_uid())
    return compose(
        YoloDepthwiseConv2D(kernel_size, padding='same', name='conv_dw_' + block_id_str),
        CustomBatchNormalization(name='conv_dw_%s_bn' % block_id_str),
        Activation(swish, name='conv_dw_%s_swish' % block_id_str),
        YoloConv2D(filters, (1,1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%s' % block_id_str),
        CustomBatchNormalization(name='conv_pw_%s_bn' % block_id_str),
        Activation(swish, name='conv_pw_%s_swish' % block_id_str))


def Spp_Conv2D_BN_Swish(x, num_filters):
    y1 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(x)
    y2 = MaxPooling2D(pool_size=(9,9), strides=(1,1), padding='same')(x)
    y3 = MaxPooling2D(pool_size=(13,13), strides=(1,1), padding='same')(x)

    y = compose(
            Concatenate(),
            DarknetConv2D_BN_Swish(num_filters, (1,1)))([y3, y2, y1, x])
    return y


def Spp_Conv2D_BN_Swish_Fast(x, num_filters):
    """
    An optimized SPP block using smaller size pooling layer,
    which would be more friendly to some edge inference device (NPU).
    """
    y1 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(x)
    y2 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(y1)
    y3 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(y2)

    y = compose(
            Concatenate(),
            DarknetConv2D_BN_Swish(num_filters, (1,1)))([y3, y2, y1, x])
    return y


def focus_block(x, num_filters, width_multiple, kernel):
    num_filters = make_divisible(num_filters*width_multiple, 8)
    x1 = Lambda(lambda z: z[:, ::2, ::2, :], name='focus_slice1')(x)
    x2 = Lambda(lambda z: z[:, 1::2, ::2, :], name='focus_slice2')(x)
    x3 = Lambda(lambda z: z[:, ::2, 1::2, :], name='focus_slice3')(x)
    x4 = Lambda(lambda z: z[:, 1::2, 1::2, :], name='focus_slice4')(x)
    x = Concatenate()([x1, x2, x3, x4])
    x = DarknetConv2D_BN_Swish(num_filters, (kernel, kernel))(x)

    return x


def bottleneck_csp_block(x, num_filters, num_blocks, depth_multiple, width_multiple, shortcut=False):
    '''CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks'''
    num_filters = make_divisible(num_filters*width_multiple, 8)
    num_blocks = max(round(num_blocks * depth_multiple), 1) if num_blocks > 1 else num_blocks  # depth gain

    res_connection = DarknetConv2D(num_filters//2, (1,1))(x)
    x = DarknetConv2D_BN_Swish(num_filters//2, (1,1))(x)

    # Bottleneck block stack
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Swish(num_filters//2, (1,1)),
                DarknetConv2D_BN_Swish(num_filters//2, (3,3)))(x)
        x = Add()([x,y]) if shortcut else y

    x = DarknetConv2D(num_filters//2, (1,1))(x)
    x = Concatenate()([x, res_connection])

    x = CustomBatchNormalization()(x)
    x = Activation(swish)(x)
    return DarknetConv2D_BN_Swish(num_filters, (1,1))(x)


def bottleneck_csp_c3_block(x, num_filters, num_blocks, depth_multiple, width_multiple, shortcut=False):
    '''CSP Bottleneck with 3 convolutions'''
    num_filters = make_divisible(num_filters*width_multiple, 8)
    num_blocks = max(round(num_blocks * depth_multiple), 1) if num_blocks > 1 else num_blocks  # depth gain

    res_connection = DarknetConv2D_BN_Swish(num_filters//2, (1,1))(x)
    x = DarknetConv2D_BN_Swish(num_filters//2, (1,1))(x)

    # Bottleneck block stack
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Swish(num_filters//2, (1,1)),
                DarknetConv2D_BN_Swish(num_filters//2, (3,3)))(x)
        x = Add()([x,y]) if shortcut else y

    #x = DarknetConv2D(num_filters//2, (1,1))(x)
    x = Concatenate()([x, res_connection])

    return DarknetConv2D_BN_Swish(num_filters, (1,1))(x)


def bottleneck_csp_lite_block(x, num_filters, num_blocks, depth_multiple, width_multiple, shortcut=False, block_id_str=None):
    '''CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks'''
    num_filters = make_divisible(num_filters*width_multiple, 8)
    num_blocks = max(round(num_blocks * depth_multiple), 1) if num_blocks > 1 else num_blocks  # depth gain

    res_connection = DarknetConv2D(num_filters//2, (1,1))(x)
    x = DarknetConv2D_BN_Swish(num_filters//2, (1,1))(x)

    # Bottleneck block stack
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Swish(num_filters//2, (1,1)),
                #DarknetConv2D_BN_Swish(num_filters//2, (3,3)))(x)
                Depthwise_Separable_Conv2D_BN_Swish(filters=num_filters//2, kernel_size=(3, 3), block_id_str=block_id_str+'_1'))(x)
        x = Add()([x,y]) if shortcut else y

    x = DarknetConv2D(num_filters//2, (1,1))(x)
    x = Concatenate()([x, res_connection])

    x = CustomBatchNormalization()(x)
    x = Activation(swish)(x)
    return DarknetConv2D_BN_Swish(num_filters, (1,1))(x)


def bottleneck_csp_c3_lite_block(x, num_filters, num_blocks, depth_multiple, width_multiple, shortcut=False, block_id_str=None):
    '''CSP Bottleneck with 3 convolutions'''
    num_filters = make_divisible(num_filters*width_multiple, 8)
    num_blocks = max(round(num_blocks * depth_multiple), 1) if num_blocks > 1 else num_blocks  # depth gain

    res_connection = DarknetConv2D_BN_Swish(num_filters//2, (1,1))(x)
    x = DarknetConv2D_BN_Swish(num_filters//2, (1,1))(x)

    # Bottleneck block stack
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Swish(num_filters//2, (1,1)),
                #DarknetConv2D_BN_Swish(num_filters//2, (3,3)))(x)
                Depthwise_Separable_Conv2D_BN_Swish(filters=num_filters//2, kernel_size=(3, 3), block_id_str=block_id_str+'_1'))(x)
        x = Add()([x,y]) if shortcut else y

    #x = DarknetConv2D(num_filters//2, (1,1))(x)
    x = Concatenate()([x, res_connection])

    return DarknetConv2D_BN_Swish(num_filters, (1,1))(x)


def yolo5_spp_neck(x, num_filters):
    '''Conv2D_BN_Swish layer followed by a SPP_Conv block'''
    x = DarknetConv2D_BN_Swish(num_filters//2, (1,1))(x)
    x = Spp_Conv2D_BN_Swish(x, num_filters)

    return x


def yolo5_predictions(feature_maps, feature_channel_nums, num_anchors, num_classes, depth_multiple, width_multiple, with_spp=True):
    f1, f2, f3 = feature_maps
    f1_channel_num, f2_channel_num, f3_channel_num = feature_channel_nums

    # SPP & BottleneckCSP block, in ultralytics PyTorch version
    # they're defined in backbone
    if with_spp:
        f1 = yolo5_spp_neck(f1, f1_channel_num)

    x1 = bottleneck_csp_block(f1, f1_channel_num, 3, depth_multiple, width_multiple, shortcut=False)

    #feature map 1 head (19x19 for 608 input)
    x1 = DarknetConv2D_BN_Swish(f2_channel_num, (1,1))(x1)

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = UpSampling2D(2)(x1)
    x2 = Concatenate()([f2, x1_upsample])

    x2 = bottleneck_csp_block(x2, f2_channel_num, 3, depth_multiple, width_multiple, shortcut=False)
    #feature map 2 head (38x38 for 608 input)
    x2 = DarknetConv2D_BN_Swish(f3_channel_num, (1,1))(x2)

    #upsample fpn merge for feature map 2 & 3
    x2_upsample = UpSampling2D(2)(x2)
    x3 = Concatenate()([f3, x2_upsample])

    #feature map 3 head & output (76x76 for 608 input)
    x3 = bottleneck_csp_block(x3, f3_channel_num, 3, depth_multiple, width_multiple, shortcut=False)
    y3 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_3')(x3)

    #downsample fpn merge for feature map 3 & 2
    x3_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            DarknetConv2D_BN_Swish(f3_channel_num, (3,3), strides=(2,2)))(x3)

    x2 = Concatenate()([x3_downsample, x2])

    #feature map 2 output (38x38 for 608 input)
    x2 = bottleneck_csp_block(x2, f2_channel_num, 3, depth_multiple, width_multiple, shortcut=False)

    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_2')(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            DarknetConv2D_BN_Swish(f2_channel_num, (3,3), strides=(2,2)))(x2)

    x1 = Concatenate()([x2_downsample, x1])

    #feature map 1 output (19x19 for 608 input)
    x1 = bottleneck_csp_block(x1, f1_channel_num, 3, depth_multiple, width_multiple, shortcut=False)

    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_1')(x1)

    return y1, y2, y3


def yolo5lite_predictions(feature_maps, feature_channel_nums, num_anchors, num_classes, depth_multiple, width_multiple, with_spp=True):
    f1, f2, f3 = feature_maps
    f1_channel_num, f2_channel_num, f3_channel_num = feature_channel_nums

    # SPP & BottleneckCSP block, in ultralytics PyTorch version
    # they're defined in backbone
    if with_spp:
        f1 = yolo5_spp_neck(f1, f1_channel_num)

    x1 = bottleneck_csp_lite_block(f1, f1_channel_num, 3, depth_multiple, width_multiple, shortcut=False, block_id_str='pred_1')

    #feature map 1 head (19x19 for 608 input)
    x1 = DarknetConv2D_BN_Swish(f2_channel_num, (1,1))(x1)

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = UpSampling2D(2)(x1)
    x2 = Concatenate()([f2, x1_upsample])

    x2 = bottleneck_csp_lite_block(x2, f2_channel_num, 3, depth_multiple, width_multiple, shortcut=False, block_id_str='pred_2')
    #feature map 2 head (38x38 for 608 input)
    x2 = DarknetConv2D_BN_Swish(f3_channel_num, (1,1))(x2)

    #upsample fpn merge for feature map 2 & 3
    x2_upsample = UpSampling2D(2)(x2)
    x3 = Concatenate()([f3, x2_upsample])

    #feature map 3 head & output (76x76 for 608 input)
    x3 = bottleneck_csp_lite_block(x3, f3_channel_num, 3, depth_multiple, width_multiple, shortcut=False, block_id_str='pred_3')
    y3 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_3')(x3)

    #downsample fpn merge for feature map 3 & 2
    x3_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            #DarknetConv2D_BN_Swish(f3_channel_num, (3,3), strides=(2,2)))(x3)
            Darknet_Depthwise_Separable_Conv2D_BN_Swish(f3_channel_num, (3,3), strides=(2,2), block_id_str='pred_3_2'))(x3)

    x2 = Concatenate()([x3_downsample, x2])

    #feature map 2 output (38x38 for 608 input)
    x2 = bottleneck_csp_lite_block(x2, f2_channel_num, 3, depth_multiple, width_multiple, shortcut=False, block_id_str='pred_4')

    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_2')(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            #DarknetConv2D_BN_Swish(f2_channel_num, (3,3), strides=(2,2)))(x2)
            Darknet_Depthwise_Separable_Conv2D_BN_Swish(f2_channel_num, (3,3), strides=(2,2), block_id_str='pred_4_2'))(x2)

    x1 = Concatenate()([x2_downsample, x1])

    #feature map 1 output (19x19 for 608 input)
    x1 = bottleneck_csp_lite_block(x1, f1_channel_num, 3, depth_multiple, width_multiple, shortcut=False, block_id_str='pred_5')

    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_1')(x1)

    return y1, y2, y3

