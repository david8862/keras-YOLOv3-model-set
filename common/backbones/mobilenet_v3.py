#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# A tf.keras implementation of mobilenet_v3,
# which is ported from https://github.com/xiaochus/MobileNetV3
#
# Reference
#   [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244?context=cs)
#

import os
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras.utils import get_source_inputs, get_file
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Add, Multiply, Reshape
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import warnings

BASE_WEIGHT_PATH = ('https://github.com/JonathanCMitchell/mobilenet_v2_keras/'
                    'releases/download/v1.1/')

def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)

def hard_swish(x):
    """Hard swish
    """
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0


def return_activation(x, nl):
    """Convolution Block
    This function defines a activation choice.

    # Arguments
        x: Tensor, input tensor of conv layer.
        nl: String, nonlinearity activation type.

    # Returns
        Output tensor.
    """
    if nl == 'HS':
        x = Activation(hard_swish)(x)
    if nl == 'RE':
        x = Activation(relu6)(x)

    return x

def conv_block(inputs, filters, kernel, strides, nl):
    """Convolution Block
    This function defines a 2D convolution operation with BN and activation.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        nl: String, nonlinearity activation type.

    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)

    return return_activation(x, nl)


def SE(inputs):
    """Squeeze and Excitation.
    This function defines a squeeze structure.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
    """
    input_channels = int(inputs.shape[-1])

    x = GlobalAveragePooling2D()(inputs)
    x = Dense(input_channels, activation='relu')(x)
    x = Dense(input_channels, activation='hard_sigmoid')(x)
    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])

    return x


def bottleneck(inputs, filters, kernel, e, s, squeeze, nl,alpha=1.0):
    """Bottleneck
    This function defines a basic bottleneck structure.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        e: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        squeeze: Boolean, Whether to use the squeeze.
        nl: String, nonlinearity activation type.
        alpha: Integer, width multiplier.

    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    input_shape = K.int_shape(inputs)

    tchannel = int(e)
    cchannel = int(alpha * filters)

    r = s == 1 and input_shape[3] == filters

    x = conv_block(inputs, tchannel, (1, 1), (1, 1), nl)

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = return_activation(x, nl)

    if squeeze:
        x = SE(x)

    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = Add()([x, inputs])

    return x


def MobileNetV3_Large(input_shape=None,
                alpha=1.0,
                include_top=True,
                weights='imagenet',
                input_tensor=None,
                pooling=None,
                classes=1000,
                **kwargs):

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
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        #if not K.is_keras_tensor(input_tensor):
            #img_input = Input(tensor=input_tensor, shape=input_shape)
        #else:
            #img_input = input_tensor
        img_input = input_tensor

    x = conv_block(img_input, 16, (3, 3), strides=(2, 2), nl='HS')

    x = bottleneck(x, 16, (3, 3), e=16, s=1, squeeze=False, nl='RE', alpha=alpha)
    x = bottleneck(x, 24, (3, 3), e=64, s=2, squeeze=False, nl='RE', alpha=alpha)
    x = bottleneck(x, 24, (3, 3), e=72, s=1, squeeze=False, nl='RE', alpha=alpha)
    x = bottleneck(x, 40, (5, 5), e=72, s=2, squeeze=True, nl='RE', alpha=alpha)
    x = bottleneck(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE', alpha=alpha)
    x = bottleneck(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE', alpha=alpha)
    x = bottleneck(x, 80, (3, 3), e=240, s=2, squeeze=False, nl='HS', alpha=alpha)
    x = bottleneck(x, 80, (3, 3), e=200, s=1, squeeze=False, nl='HS', alpha=alpha)
    x = bottleneck(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS', alpha=alpha)
    x = bottleneck(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS', alpha=alpha)
    x = bottleneck(x, 112, (3, 3), e=480, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 112, (3, 3), e=672, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 160, (5, 5), e=672, s=2, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS', alpha=alpha)

    x = conv_block(x, 960, (1, 1), strides=(1, 1), nl='HS')
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 960))(x)

    x = Conv2D(1280, (1, 1), padding='same')(x)
    x = return_activation(x, 'HS')

    if include_top:
        x = Conv2D(classes, (1, 1), padding='same', activation='softmax')(x)
        x = Reshape((classes,))(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)

    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x,
                  name='mobilenetv3_large_%0.2f' % (alpha))

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            model_name = ('mobilenet_v3_large_weights_tf_dim_ordering_tf_kernels_' +
                          str(alpha) + '.h5')
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = get_file(
                model_name, weight_path, cache_subdir='models')
        else:
            model_name = ('mobilenet_v3_large_weights_tf_dim_ordering_tf_kernels_' +
                          str(alpha) + '_no_top' + '.h5')
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = get_file(
                model_name, weight_path, cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def MobileNetV3_Small(input_shape=None,
                alpha=1.0,
                include_top=True,
                weights='imagenet',
                input_tensor=None,
                pooling=None,
                classes=1000,
                **kwargs):

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
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        #if not K.is_keras_tensor(input_tensor):
            #img_input = Input(tensor=input_tensor, shape=input_shape)
        #else:
            #img_input = input_tensor
        img_input = input_tensor

    x = conv_block(img_input, 16, (3, 3), strides=(2, 2), nl='HS')

    x = bottleneck(x, 16, (3, 3), e=16, s=2, squeeze=True, nl='RE', alpha=alpha)
    x = bottleneck(x, 24, (3, 3), e=72, s=2, squeeze=False, nl='RE', alpha=alpha)
    x = bottleneck(x, 24, (3, 3), e=88, s=1, squeeze=False, nl='RE', alpha=alpha)
    x = bottleneck(x, 40, (5, 5), e=96, s=2, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 48, (5, 5), e=120, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 48, (5, 5), e=144, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 96, (5, 5), e=288, s=2, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS', alpha=alpha)

    x = conv_block(x, 576, (1, 1), strides=(1, 1), nl='HS')
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 576))(x)

    x = Conv2D(1280, (1, 1), padding='same')(x)
    x = return_activation(x, 'HS')

    if include_top:
        x = Conv2D(classes, (1, 1), padding='same', activation='softmax')(x)
        x = Reshape((classes,))(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)

    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x,
                  name='mobilenetv3_small_%0.2f' % (alpha))

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            model_name = ('mobilenet_v3_small_weights_tf_dim_ordering_tf_kernels_' +
                          str(alpha) + '.h5')
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = get_file(
                model_name, weight_path, cache_subdir='models')
        else:
            model_name = ('mobilenet_v3_small_weights_tf_dim_ordering_tf_kernels_' +
                          str(alpha) + '_no_top' + '.h5')
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = get_file(
                model_name, weight_path, cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


if __name__ == '__main__':
    input_tensor = Input(shape=(None, None, 3), name='image_input')
    #model = MobileNetV3_Large(include_top=False, input_shape=(224, 224, 3), weights=None, alpha=1.0)
    model = MobileNetV3_Large(include_top=True, input_tensor=input_tensor, weights=None, alpha=1.0)
    model.summary()

    import numpy as np
    #from keras_applications.imagenet_utils import preprocess_input, decode_predictions
    from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
    from keras_preprocessing import image

    img = image.load_img('../../example/eagle.jpg', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
