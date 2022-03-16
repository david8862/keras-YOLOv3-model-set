#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# A tf.keras implementation of ghostnet
#
#
import os, sys
import warnings
import math

from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import preprocess_input as _preprocess_input
from tensorflow.keras.utils import get_source_inputs, get_file
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, Dense, Flatten, ReLU, Reshape, Activation
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Dropout, Add, Multiply
from tensorflow.keras import backend as K

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.backbones.layers import YoloConv2D, YoloDepthwiseConv2D, CustomBatchNormalization


BASE_WEIGHT_PATH = (
    'https://github.com/david8862/tf-keras-image-classifier/'
    'releases/download/v1.0.0/')


def preprocess_input(x):
    """
    "mode" option description in preprocess_input
    mode: One of "caffe", "tf" or "torch".
        - caffe: will convert the images from RGB to BGR,
            then will zero-center each color channel with
            respect to the ImageNet dataset,
            without scaling.
        - tf: will scale pixels between -1 and 1,
            sample-wise.
        - torch: will scale pixels between 0 and 1 and then
            will normalize each channel with respect to the
            ImageNet dataset.
    """
    # here we use pytorch mode preprocess to align with origin
    #x = _preprocess_input(x, mode='torch', backend=K)

    x /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    if std is not None:
        x[..., 0] /= std[0]
        x[..., 1] /= std[1]
        x[..., 2] /= std[2]

    return x


# This function is taken from the original tf repo.
# It ensures that all layers have a channel number that is divisible by 8
# It can be seen here:
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x):
    return ReLU(6.0)(x + 3.0) / 6.0


def primary_conv(x, output_filters, kernel_size, strides=(1,1), padding='same', act=True, use_bias=False, name=None):
    x = YoloConv2D(filters=output_filters,
               kernel_size=kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name + '_0')(x)
    x = CustomBatchNormalization(name=name+'_1')(x)
    x = ReLU(name=name+'_relu')(x) if act else x
    return x


def cheap_operations(x, output_filters, kernel_size, strides=(1,1), padding='same', act=True, use_bias=False, name=None):
    x = YoloDepthwiseConv2D(kernel_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        use_bias=use_bias,
                        name=name+'_0')(x)
    x = CustomBatchNormalization(name=name+'_1')(x)
    x = ReLU(name=name+'_relu')(x) if act else x
    return x


def SqueezeExcite(input_x, se_ratio=0.25, reduced_base_chs=None, divisor=4, name=None):
    reduce_chs =_make_divisible((reduced_base_chs or int(input_x.shape[-1]))*se_ratio, divisor)

    x = GlobalAveragePooling2D(name=name+'_avg_pool2d')(input_x)
    if K.image_data_format() == 'channels_first':
        x = Reshape((int(input_x.shape[-1]), 1, 1))(x)
    else:
        x = Reshape((1, 1, int(input_x.shape[-1])))(x)

    x = YoloConv2D(filters=reduce_chs, kernel_size=1, use_bias=True, name=name+'_conv_reduce')(x)
    x = ReLU(name=name+'_act')(x)
    x = YoloConv2D(filters=int(input_x.shape[-1]), kernel_size=1, use_bias=True, name=name+'_conv_expand')(x)

    x = Activation(hard_sigmoid, name=name+'_hard_sigmoid')(x)
    x = Multiply()([input_x, x])

    return x


def ConvBnAct(input_x, out_chs, kernel_size, stride=(1,1), name=None):
    x = YoloConv2D(filters=out_chs,
               kernel_size=kernel_size,
               strides=stride,
               padding='valid',
               use_bias=False,
               name=name+'_conv')(input_x)
    x = CustomBatchNormalization(name=name+'_bn1')(x)
    x = ReLU(name=name+'_relu')(x)
    return x


def GhostModule(input_x, output_chs, kernel_size=1, ratio=2, dw_size=3, stride=(1,1), act=True, name=None):
    init_channels = int(math.ceil(output_chs / ratio))
    new_channels = int(init_channels * (ratio - 1))
    x1 = primary_conv(input_x,
                      init_channels,
                      kernel_size=kernel_size,
                      strides=stride,
                      padding='valid',
                      act=act,
                      name = name + '_primary_conv')
    x2 = cheap_operations(x1,
                          new_channels,
                          kernel_size=dw_size,
                          strides=(1,1),
                          padding= 'same',
                          act=act,
                          name = name + '_cheap_operation')
    x = Concatenate(axis=3, name=name+'_concat')([x1,x2])
    return x


def GhostBottleneck(input_x, mid_chs, out_chs, dw_kernel_size=3, stride=(1,1), se_ratio=0., name=None):
    '''ghostnet bottleneck w/optional se'''
    has_se = se_ratio is not None and se_ratio > 0.

    #1st ghost bottleneck
    x = GhostModule(input_x, mid_chs, act=True, name=name+'_ghost1')

    #depth_with convolution
    if stride[0] > 1:
        x = YoloDepthwiseConv2D(kernel_size=dw_kernel_size,
                            strides=stride,
                            padding='same',
                            use_bias=False,
                            name=name+'_conv_dw')(x)
        x = CustomBatchNormalization(name=name+'_bn_dw')(x)

    #Squeeze_and_excitation
    if has_se:
        x = SqueezeExcite(x, se_ratio=se_ratio, name=name+'_se')

    #2nd ghost bottleneck
    x = GhostModule(x, out_chs, act=False, name=name+'_ghost2')

    #short cut
    if (input_x.shape[-1] == out_chs and stride[0] == 1):
        sc = input_x
    else:
        name1 = name + '_shortcut'
        sc = YoloDepthwiseConv2D(kernel_size=dw_kernel_size,
                             strides=stride,
                             padding='same',
                             use_bias=False,
                             name=name1+'_0')(input_x)
        sc = CustomBatchNormalization(name=name1+'_1')(sc)
        sc = YoloConv2D(filters=out_chs,
                    kernel_size=1,
                    strides=(1,1),
                    padding='valid',
                    use_bias=False,
                    name=name1+'_2')(sc)
        sc = CustomBatchNormalization(name=name1+'_3')(sc)

    x = Add(name=name+'_add')([x, sc])
    return x


DEFAULT_CFGS = [
        # k, t, c, SE, s
        # stage1
        [[3,  16,  16, 0, 1]],
        # stage2
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        # stage3
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        # stage4
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]


def GhostNet(input_shape=None,
             include_top=True,
             weights='imagenet',
             input_tensor=None,
             cfgs=DEFAULT_CFGS,
             width=1.0,
             dropout_rate=0.2,
             pooling=None,
             classes=1000,
             **kwargs):
    """Instantiates the GhostNet architecture.

    # Arguments
        input_shape: optional shape tuple, to be specified if you would
            like to use a model with an input img resolution that is not
            (224, 224, 3).
            It should have exactly 3 inputs channels (224, 224, 3).
            You can also omit this option if you would like
            to infer input_shape from an input_tensor.
            If you choose to include both input_tensor and input_shape then
            input_shape will be used if they match, if the shapes
            do not match then we will throw an error.
            E.g. `(160, 160, 3)` would be one valid value.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        cfgs: model structure config list
        width: controls the width of the network
        dropout_rate: fraction of the input units to drop on the last layer
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape or invalid alpha, rows when
            weights='imagenet'
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
                         'as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)


    # If input_shape is None and input_tensor is None using standard shape
    if input_shape is None and input_tensor is None:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        #if not K.is_keras_tensor(input_tensor):
            #img_input = Input(tensor=input_tensor, shape=input_shape)
        #else:
            #img_input = input_tensor
        img_input = input_tensor

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # building first layer
    output_channel = int(_make_divisible(16 * width, 4))
    x = YoloConv2D(filters=output_channel,
               kernel_size=3,
               strides=(2, 2),
               padding='same',
               use_bias=False,
               name='conv_stem')(img_input)
    x = CustomBatchNormalization(name='bn1')(x)
    x = ReLU(name='Conv2D_1_act')(x)

    # building inverted residual blocks
    for index, cfg in enumerate(cfgs):
        sub_index = 0
        for k,exp_size,c,se_ratio,s in cfg:
            output_channel = int(_make_divisible(c * width, 4))
            hidden_channel = int(_make_divisible(exp_size * width, 4))
            x = GhostBottleneck(x, hidden_channel, output_channel, k, (s,s),
                                se_ratio=se_ratio,
                                name='blocks_'+str(index)+'_'+str(sub_index))
            sub_index += 1

    output_channel = _make_divisible(exp_size * width, 4)
    x = ConvBnAct(x, output_channel, kernel_size=1, name='blocks_9_0')

    if include_top:
        x = GlobalAveragePooling2D(name='global_avg_pooling2D')(x)
        if K.image_data_format() == 'channels_first':
            x = Reshape((output_channel, 1, 1))(x)
        else:
            x = Reshape((1, 1, output_channel))(x)

        # building last several layers
        output_channel = 1280
        x = YoloConv2D(filters=output_channel,
                   kernel_size=1,
                   strides=(1,1),
                   padding='valid',
                   use_bias=True,
                   name='conv_head')(x)
        x = ReLU(name='relu_head')(x)

        if dropout_rate > 0.:
            x = Dropout(dropout_rate, name='dropout_1')(x)
        x = Flatten()(x)
        x = Dense(units=classes, activation='softmax',
                         use_bias=True, name='classifier')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='ghostnet_%0.2f' % (width))

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            file_name = 'ghostnet_weights_tf_dim_ordering_tf_kernels_224.h5'
            weight_path = BASE_WEIGHT_PATH + file_name
        else:
            file_name = 'ghostnet_weights_tf_dim_ordering_tf_kernels_224_no_top.h5'
            weight_path = BASE_WEIGHT_PATH + file_name

        weights_path = get_file(file_name, weight_path, cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


if __name__ == '__main__':
    input_tensor = Input(shape=(None, None, 3), name='image_input')
    #model = GhostNet(include_top=False, input_tensor=input_tensor, weights='imagenet')
    model = GhostNet(include_top=True, input_shape=(224, 224, 3), weights='imagenet')
    model.summary()
    K.set_learning_phase(0)

    import numpy as np
    from tensorflow.keras.applications.resnet50 import decode_predictions
    from keras_preprocessing import image

    img = image.load_img('../../example/eagle.jpg', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))

