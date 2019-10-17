#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common layer definition for YOLOv3 models building
"""
from functools import wraps, reduce

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Concatenate, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

@wraps(DepthwiseConv2D)
def DarknetDepthwiseConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return DepthwiseConv2D(*args, **darknet_conv_kwargs)

def Darknet_Depthwise_Separable_Conv2D_BN_Leaky(filters, kernel_size=(3, 3), block_id_str=None, **kwargs):
    """Depthwise Separable Convolution2D."""
    if not block_id_str:
        block_id_str = str(K.get_uid())
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetDepthwiseConv2D(kernel_size, name='conv_dw_' + block_id_str, **no_bias_kwargs),
        BatchNormalization(name='conv_dw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1, name='conv_dw_%s_leaky_relu' % block_id_str),
        Conv2D(filters, (1,1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%s' % block_id_str),
        BatchNormalization(name='conv_pw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1, name='conv_pw_%s_leaky_relu' % block_id_str))


def Depthwise_Separable_Conv2D_BN_Leaky(filters, kernel_size=(3, 3), block_id_str=None):
    """Depthwise Separable Convolution2D."""
    if not block_id_str:
        block_id_str = str(K.get_uid())
    return compose(
        DepthwiseConv2D(kernel_size, padding='same', name='conv_dw_' + block_id_str),
        BatchNormalization(name='conv_dw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1, name='conv_dw_%s_leaky_relu' % block_id_str),
        Conv2D(filters, (1,1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%s' % block_id_str),
        BatchNormalization(name='conv_pw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1, name='conv_pw_%s_leaky_relu' % block_id_str))


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def Spp_Conv2D_BN_Leaky(x, num_filters):
    y1 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(x)
    y2 = MaxPooling2D(pool_size=(9,9), strides=(1,1), padding='same')(x)
    y3 = MaxPooling2D(pool_size=(13,13), strides=(1,1), padding='same')(x)

    y = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))([y1, y2, y3, x])
    return y


def make_last_layers(x, num_filters, out_filters, predict_filters=None):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)

    if predict_filters is None:
        predict_filters = num_filters*2
    y = compose(
            DarknetConv2D_BN_Leaky(predict_filters, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y

def make_spp_last_layers(x, num_filters, out_filters, predict_filters=None):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)

    x = Spp_Conv2D_BN_Leaky(x, num_filters)

    x = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)

    if predict_filters is None:
        predict_filters = num_filters*2
    y = compose(
            DarknetConv2D_BN_Leaky(predict_filters, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y

def make_depthwise_separable_last_layers(x, num_filters, out_filters, block_id_str=None, predict_filters=None):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    if not block_id_str:
        block_id_str = str(K.get_uid())
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=num_filters*2, kernel_size=(3, 3), block_id_str=block_id_str+'_1'),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=num_filters*2, kernel_size=(3, 3), block_id_str=block_id_str+'_2'),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)

    if predict_filters is None:
        predict_filters = num_filters*2
    y = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(filters=predict_filters, kernel_size=(3, 3), block_id_str=block_id_str+'_3'),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y

def make_spp_depthwise_separable_last_layers(x, num_filters, out_filters, block_id_str=None, predict_filters=None):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    if not block_id_str:
        block_id_str = str(K.get_uid())
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=num_filters*2, kernel_size=(3, 3), block_id_str=block_id_str+'_1'),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)

    x = Spp_Conv2D_BN_Leaky(x, num_filters)

    x = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(filters=num_filters*2, kernel_size=(3, 3), block_id_str=block_id_str+'_2'),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)

    if predict_filters is None:
        predict_filters = num_filters*2
    y = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(filters=predict_filters, kernel_size=(3, 3), block_id_str=block_id_str+'_3'),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y
