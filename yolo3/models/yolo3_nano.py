#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v3 Nano Model Defined in Keras."""

import os
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras.utils import get_source_inputs, get_file
from tensorflow.keras.layers import UpSampling2D, Concatenate, Dense, Multiply, Add, Lambda, Input, Reshape
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Concatenate, BatchNormalization, ReLU, ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from common.backbones.layers import YoloConv2D, YoloDepthwiseConv2D, CustomBatchNormalization
from yolo3.models.layers import compose, DarknetConv2D


def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.

    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def NanoConv2D_BN_Relu6(*args, **kwargs):
    """Darknet Convolution2D followed by CustomBatchNormalization and ReLU6."""
    nano_name = kwargs.get('name')
    if nano_name:
        name_kwargs = {'name': nano_name + '_conv2d'}
        name_kwargs.update(kwargs)
        bn_name = nano_name + '_BN'
        relu_name = nano_name + '_relu'
    else:
        name_kwargs = {}
        name_kwargs.update(kwargs)
        bn_name = None
        relu_name = None

    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(name_kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        CustomBatchNormalization(name=bn_name),
        ReLU(6., name=relu_name))


def _ep_block(inputs, filters, stride, expansion, block_id):
    #in_channels = backend.int_shape(inputs)[-1]
    in_channels = inputs.shape.as_list()[-1]

    pointwise_conv_filters = int(filters)
    x = inputs
    prefix = 'ep_block_{}_'.format(block_id)

    # Expand
    x = YoloConv2D(int(expansion * in_channels), kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'expand')(x)
    x = CustomBatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
    x = ReLU(6., name=prefix + 'expand_relu')(x)

    # Depthwise
    if stride == 2:
        x = ZeroPadding2D(padding=correct_pad(K, x, 3), name=prefix + 'pad')(x)

    x = YoloDepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same' if stride == 1 else 'valid', name=prefix + 'depthwise')(x)
    x = CustomBatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)
    x = ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = YoloConv2D(pointwise_conv_filters, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'project')(x)
    x = CustomBatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if in_channels == pointwise_conv_filters and stride == 1:
        return Add(name=prefix + 'add')([inputs, x])
    return x


def _pep_block(inputs, proj_filters, filters, stride, expansion, block_id):
    #in_channels = backend.int_shape(inputs)[-1]
    in_channels = inputs.shape.as_list()[-1]

    pointwise_conv_filters = int(filters)
    x = inputs
    prefix = 'pep_block_{}_'.format(block_id)


    # Pre-project
    x = YoloConv2D(proj_filters, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'preproject')(x)
    x = CustomBatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'preproject_BN')(x)
    x = ReLU(6., name=prefix + 'preproject_relu')(x)

    # Expand
    #x = YoloConv2D(int(expansion * in_channels), kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'expand')(x)
    x = YoloConv2D(int(expansion * proj_filters), kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'expand')(x)
    x = CustomBatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
    x = ReLU(6., name=prefix + 'expand_relu')(x)

    # Depthwise
    if stride == 2:
        x = ZeroPadding2D(padding=correct_pad(K, x, 3), name=prefix + 'pad')(x)

    x = YoloDepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same' if stride == 1 else 'valid', name=prefix + 'depthwise')(x)
    x = CustomBatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)
    x = ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = YoloConv2D(pointwise_conv_filters, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'project')(x)
    x = CustomBatchNormalization( epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if in_channels == pointwise_conv_filters and stride == 1:
        return Add(name=prefix + 'add')([inputs, x])
    return x


#def expand_dims2d(x):
    #x = K.expand_dims(x, axis=1)
    #x = K.expand_dims(x, axis=2)
    #return x


#def expand_upsampling2d(args):
    #import tensorflow as tf
    #x = args[0]
    #inputs = args[1]
    #in_shapes = K.shape(inputs)[1:3]
    #x = K.expand_dims(x, axis=1)
    #x = K.expand_dims(x, axis=2)
    #x = tf.image.resize(x, in_shapes)
    #return x


def _fca_block(inputs, reduct_ratio, block_id):
    in_channels = inputs.shape.as_list()[-1]
    #in_shapes = inputs.shape.as_list()[1:3]
    reduct_channels = int(in_channels // reduct_ratio)
    prefix = 'fca_block_{}_'.format(block_id)
    x = GlobalAveragePooling2D(name=prefix + 'average_pooling')(inputs)
    x = Dense(reduct_channels, activation='relu', name=prefix + 'fc1')(x)
    x = Dense(in_channels, activation='sigmoid', name=prefix + 'fc2')(x)

    x = Reshape((1,1,in_channels),name='reshape')(x)
    x = Multiply(name=prefix + 'multiply')([x, inputs])
    return x


EP_EXPANSION = 2
PEP_EXPANSION = 2

def nano_net_body(x):
    '''YOLO Nano backbone network body'''
    x = NanoConv2D_BN_Relu6(12, (3,3), name='Conv_1')(x)
    x = NanoConv2D_BN_Relu6(24, (3,3), strides=2, name='Conv_2')(x)
    x = _pep_block(x, proj_filters=7, filters=24, stride=1, expansion=PEP_EXPANSION, block_id=1)
    x = _ep_block(x, filters=70, stride=2, expansion=EP_EXPANSION, block_id=1)
    x = _pep_block(x, proj_filters=25, filters=70, stride=1, expansion=PEP_EXPANSION, block_id=2)
    x = _pep_block(x, proj_filters=24, filters=70, stride=1, expansion=PEP_EXPANSION, block_id=3)
    x = _ep_block(x, filters=150, stride=2, expansion=EP_EXPANSION, block_id=2)
    x = _pep_block(x, proj_filters=56, filters=150, stride=1, expansion=PEP_EXPANSION, block_id=4)
    x = NanoConv2D_BN_Relu6(150, (1,1), name='Conv_pw_1')(x)
    x = _fca_block(x, reduct_ratio=8, block_id=1)
    x = _pep_block(x, proj_filters=73, filters=150, stride=1, expansion=PEP_EXPANSION, block_id=5)
    x = _pep_block(x, proj_filters=71, filters=150, stride=1, expansion=PEP_EXPANSION, block_id=6)
    x = _pep_block(x, proj_filters=75, filters=150, stride=1, expansion=PEP_EXPANSION, block_id=7)
    x = _ep_block(x, filters=325, stride=2, expansion=EP_EXPANSION, block_id=3)
    x = _pep_block(x, proj_filters=132, filters=325, stride=1, expansion=PEP_EXPANSION, block_id=8)
    x = _pep_block(x, proj_filters=124, filters=325, stride=1, expansion=PEP_EXPANSION, block_id=9)
    x = _pep_block(x, proj_filters=141, filters=325, stride=1, expansion=PEP_EXPANSION, block_id=10)
    x = _pep_block(x, proj_filters=140, filters=325, stride=1, expansion=PEP_EXPANSION, block_id=11)
    x = _pep_block(x, proj_filters=137, filters=325, stride=1, expansion=PEP_EXPANSION, block_id=12)
    x = _pep_block(x, proj_filters=135, filters=325, stride=1, expansion=PEP_EXPANSION, block_id=13)
    x = _pep_block(x, proj_filters=133, filters=325, stride=1, expansion=PEP_EXPANSION, block_id=14)
    x = _pep_block(x, proj_filters=140, filters=325, stride=1, expansion=PEP_EXPANSION, block_id=15)
    x = _ep_block(x, filters=545, stride=2, expansion=EP_EXPANSION, block_id=4)
    x = _pep_block(x, proj_filters=276, filters=545, stride=1, expansion=PEP_EXPANSION, block_id=16)
    x = NanoConv2D_BN_Relu6(230, (1,1), name='Conv_pw_2')(x)
    x = _ep_block(x, filters=489, stride=1, expansion=EP_EXPANSION, block_id=5)
    x = _pep_block(x, proj_filters=213, filters=469, stride=1, expansion=PEP_EXPANSION, block_id=17)
    x = NanoConv2D_BN_Relu6(189, (1,1), name='Conv_pw_3')(x)

    return x


def yolo3_nano_body(inputs, num_anchors, num_classes, weights_path=None):
    """
    Create YOLO_V3 Nano model CNN body in Keras.

    Reference Paper:
        "YOLO Nano: a Highly Compact You Only Look Once Convolutional Neural Network for Object Detection"
        https://arxiv.org/abs/1910.01271
    """
    nano_net = NanoNet(input_tensor=inputs, weights='imagenet', include_top=False)
    if weights_path is not None:
        nano_net.load_weights(weights_path, by_name=True)
        print('Load weights {}.'.format(weights_path))

    # input: 416 x 416 x 3
    # Conv_pw_3_relu: 13 x 13 x 189
    # pep_block_15_add: 26 x 26 x 325
    # pep_block_7_add: 52 x 52 x 150

    # f1 :13 x 13 x 189
    f1 = nano_net.get_layer('Conv_pw_3').output
    # f2: 26 x 26 x 325
    f2 = nano_net.get_layer('pep_block_15_add').output
    # f3 : 52 x 52 x 150
    f3 = nano_net.get_layer('pep_block_7_add').output

    #feature map 1 head & output (13x13 for 416 input)
    y1 = _ep_block(f1, filters=462, stride=1, expansion=EP_EXPANSION, block_id=6)
    y1 = DarknetConv2D(num_anchors * (num_classes + 5), (1,1))(y1)

    #upsample fpn merge for feature map 1 & 2
    x = compose(
            NanoConv2D_BN_Relu6(105, (1,1)),
            UpSampling2D(2))(f1)
    x = Concatenate()([x,f2])

    #feature map 2 head & output (26x26 for 416 input)
    x = _pep_block(x, proj_filters=113, filters=325, stride=1, expansion=PEP_EXPANSION, block_id=18)
    x = _pep_block(x, proj_filters=99, filters=207, stride=1, expansion=PEP_EXPANSION, block_id=19)
    x = DarknetConv2D(98, (1,1))(x)
    y2 = _ep_block(x, filters=183, stride=1, expansion=EP_EXPANSION, block_id=7)
    y2 = DarknetConv2D(num_anchors * (num_classes + 5), (1,1))(y2)

    #upsample fpn merge for feature map 2 & 3
    x = compose(
            NanoConv2D_BN_Relu6(47, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, f3])

    #feature map 3 head & output (52x52 for 416 input)
    x = _pep_block(x, proj_filters=58, filters=122, stride=1, expansion=PEP_EXPANSION, block_id=20)
    x = _pep_block(x, proj_filters=52, filters=87, stride=1, expansion=PEP_EXPANSION, block_id=21)
    x = _pep_block(x, proj_filters=47, filters=93, stride=1, expansion=PEP_EXPANSION, block_id=22)
    y3 = DarknetConv2D(num_anchors * (num_classes + 5), (1,1))(x)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


BASE_WEIGHT_PATH = (
    'https://github.com/david8862/keras-YOLOv3-model-set/'
    'releases/download/v1.0.1/')

def NanoNet(input_shape=None,
            input_tensor=None,
            include_top=True,
            weights='imagenet',
            pooling=None,
            classes=1000,
            **kwargs):
    """Generate nano net model for Imagenet classification."""

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=28,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    x = nano_net_body(img_input)

    if include_top:
        model_name='nano_net'
        x = DarknetConv2D(classes, (1, 1))(x)
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Softmax()(x)
    else:
        model_name='nano_net_headless'
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name=model_name)

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            file_name = 'nanonet_weights_tf_dim_ordering_tf_kernels_224.h5'
            weight_path = BASE_WEIGHT_PATH + file_name
        else:
            file_name = 'nanonet_weights_tf_dim_ordering_tf_kernels_224_no_top.h5'
            weight_path = BASE_WEIGHT_PATH + file_name

        weights_path = get_file(file_name, weight_path, cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model

