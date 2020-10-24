#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common layer definition for YOLOv2 models building
"""
from functools import wraps, reduce, partial

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, MaxPooling2D, Concatenate, Lambda
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

from common.backbones.layers import YoloConv2D, YoloDepthwiseConv2D, CustomBatchNormalization

# Partial wrapper for Convolution2D with static default argument.
_DarknetConv2D = partial(YoloConv2D, padding='same')


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


@wraps(YoloConv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet weight regularizer for YoloConv2D."""
    #darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    #darknet_conv_kwargs.update(kwargs)
    darknet_conv_kwargs = kwargs
    return _DarknetConv2D(*args, **darknet_conv_kwargs)

@wraps(YoloDepthwiseConv2D)
def DarknetDepthwiseConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    #darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    #darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs = {'padding': 'valid' if kwargs.get('strides')==(2,2) else 'same'}
    darknet_conv_kwargs.update(kwargs)
    return YoloDepthwiseConv2D(*args, **darknet_conv_kwargs)


def Darknet_Depthwise_Separable_Conv2D_BN_Leaky(filters, kernel_size=(3, 3), block_id_str=None, **kwargs):
    """Depthwise Separable Convolution2D."""
    if not block_id_str:
        block_id_str = str(K.get_uid())
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetDepthwiseConv2D(kernel_size, name='conv_dw_' + block_id_str, **no_bias_kwargs),
        CustomBatchNormalization(name='conv_dw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1, name='conv_dw_%s_leaky_relu' % block_id_str),
        YoloConv2D(filters, (1,1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%s' % block_id_str),
        CustomBatchNormalization(name='conv_pw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1, name='conv_pw_%s_leaky_relu' % block_id_str))


def Depthwise_Separable_Conv2D_BN_Leaky(filters, kernel_size=(3, 3), block_id_str=None):
    """Depthwise Separable Convolution2D."""
    if not block_id_str:
        block_id_str = str(K.get_uid())
    return compose(
        YoloDepthwiseConv2D(kernel_size, padding='same', name='conv_dw_' + block_id_str),
        CustomBatchNormalization(name='conv_dw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1, name='conv_dw_%s_leaky_relu' % block_id_str),
        YoloConv2D(filters, (1,1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%s' % block_id_str),
        CustomBatchNormalization(name='conv_pw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1, name='conv_pw_%s_leaky_relu' % block_id_str))


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by CustomBatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        CustomBatchNormalization(),
        LeakyReLU(alpha=0.1))


def bottleneck_block(outer_filters, bottleneck_filters):
    """Bottleneck block of 3x3, 1x1, 3x3 convolutions."""
    return compose(
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)),
        DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)))


def bottleneck_x2_block(outer_filters, bottleneck_filters):
    """Bottleneck block of 3x3, 1x1, 3x3, 1x1, 3x3 convolutions."""
    return compose(
        bottleneck_block(outer_filters, bottleneck_filters),
        DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)))


def space_to_depth_x2(x):
    """Thin wrapper for Tensorflow space_to_depth with block_size=2."""
    # Import currently required to make Lambda work.
    # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
    import tensorflow as tf
    return tf.nn.space_to_depth(x, block_size=2)


def space_to_depth_x2_output_shape(input_shape):
    """Determine space_to_depth output shape for block_size=2.

    Note: For Lambda with TensorFlow backend, output shape may not be needed.
    """
    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 *
            input_shape[3]) if input_shape[1] else (input_shape[0], None, None,
                                                    4 * input_shape[3])

def yolo2_predictions(feature_maps, feature_channel_nums, num_anchors, num_classes):
    f1, f2 = feature_maps
    f1_channel_num, f2_channel_num = feature_channel_nums

    x1 = compose(
        DarknetConv2D_BN_Leaky(f1_channel_num, (3, 3)),
        DarknetConv2D_BN_Leaky(f1_channel_num, (3, 3)))(f1)

    # Here change the f2 channel number to f2_channel_num//8 first,
    # then expand back to f2_channel_num//2 with "space_to_depth_x2"
    x2 = DarknetConv2D_BN_Leaky(f2_channel_num//8, (1, 1))(f2)
    # TODO: Allow Keras Lambda to use func arguments for output_shape?
    x2_reshaped = Lambda(
        space_to_depth_x2,
        output_shape=space_to_depth_x2_output_shape,
        name='space_to_depth')(x2)

    x = Concatenate()([x2_reshaped, x1])
    x = DarknetConv2D_BN_Leaky(f1_channel_num, (3, 3))(x)
    y = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name='predict_conv')(x)

    return y


def yolo2lite_predictions(feature_maps, feature_channel_nums, num_anchors, num_classes):
    f1, f2 = feature_maps
    f1_channel_num, f2_channel_num = feature_channel_nums

    x1 = compose(
        Depthwise_Separable_Conv2D_BN_Leaky(filters=f1_channel_num, kernel_size=(3, 3), block_id_str='pred_1'),
        Depthwise_Separable_Conv2D_BN_Leaky(filters=f1_channel_num, kernel_size=(3, 3), block_id_str='pred_2'))(f1)

    # Here change the f2 channel number to f2_channel_num//8 first,
    # then expand back to f2_channel_num//2 with "space_to_depth_x2"
    x2 = DarknetConv2D_BN_Leaky(f2_channel_num//8, (1, 1))(f2)
    # TODO: Allow Keras Lambda to use func arguments for output_shape?
    x2_reshaped = Lambda(
        space_to_depth_x2,
        output_shape=space_to_depth_x2_output_shape,
        name='space_to_depth')(x2)

    x = Concatenate()([x2_reshaped, x1])
    x = Depthwise_Separable_Conv2D_BN_Leaky(filters=f1_channel_num, kernel_size=(3, 3), block_id_str='pred_3')(x)
    y = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name='predict_conv')(x)

    return y

