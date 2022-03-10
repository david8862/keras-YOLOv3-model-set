#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# A tf.keras implementation of peleenet,
# ported from https://github.com/markshih91/peleenet_pytorch2keras_convertor/blob/master/keras_peleenet.py
#
# Reference Paper
#   [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
#   [Pelee: A Real-Time Object Detection System on Mobile Devices](https://arxiv.org/abs/1804.06882)
#
import os, sys
import warnings

from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import preprocess_input as _preprocess_input
from tensorflow.keras.utils import get_source_inputs, get_file
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, \
    MaxPooling2D, Concatenate, AveragePooling2D, Flatten, Dropout, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras import backend as K

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.backbones.layers import YoloConv2D, CustomBatchNormalization


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


def dense_graph(x, growth_rate, bottleneck_width, name=''):
    growth_rate = int(growth_rate / 2)
    inter_channel = int(growth_rate * bottleneck_width / 4) * 4

    num_input_features = K.int_shape(x)[-1]

    if inter_channel > num_input_features / 2:
        inter_channel = int(num_input_features / 8) * 4
        print('adjust inter_channel to ', inter_channel)

    branch1 = basic_conv2d_graph(
        x, inter_channel, kernel_size=1, strides=1, padding='valid', name=name + '_branch1a')
    branch1 = basic_conv2d_graph(
        branch1, growth_rate, kernel_size=3, strides=1, padding='same', name=name + '_branch1b')

    branch2 = basic_conv2d_graph(
        x, inter_channel, kernel_size=1, strides=1, padding='valid', name=name + '_branch2a')
    branch2 = basic_conv2d_graph(
        branch2, growth_rate, kernel_size=3, strides=1, padding='same', name=name + '_branch2b')
    branch2 = basic_conv2d_graph(
        branch2, growth_rate, kernel_size=3, strides=1, padding='same', name=name + '_branch2c')

    out = Concatenate(axis=-1)([x, branch1, branch2])

    return out


def dense_block_graph(x, num_layers, bn_size, growth_rate, name=''):
    for i in range(num_layers):
        x = dense_graph(x, growth_rate, bn_size, name=name + '_denselayer{}'.format(i + 1))

    return x


def stem_block_graph(x, num_init_features, name=''):
    num_stem_features = int(num_init_features / 2)

    out = basic_conv2d_graph(x, num_init_features, kernel_size=3, strides=2, padding='same', name=name + '_stem1')

    branch2 = basic_conv2d_graph(
        out, num_stem_features, kernel_size=1, strides=1, padding='valid', name=name + '_stem2a')
    branch2 = basic_conv2d_graph(
        branch2, num_init_features, kernel_size=3, strides=2, padding='same', name=name + '_stem2b')

    branch1 = MaxPooling2D(pool_size=2, strides=2)(out)

    out = Concatenate(axis=-1)([branch1, branch2])

    out = basic_conv2d_graph(out, num_init_features, kernel_size=1, strides=1, padding='valid', name=name + '_stem3')

    return out


def basic_conv2d_graph(x, out_channels, kernel_size, strides, padding, activation=True, name=''):
    x = YoloConv2D(
        out_channels, kernel_size=kernel_size, strides=strides,
        padding=padding, use_bias=False, name=name + '_conv')(x)
    x = CustomBatchNormalization(name=name + '_norm')(x)
    if activation:
        x = ReLU()(x)

    return x


def PeleeNet(input_shape=None,
             growth_rate=32,
             block_config=[3, 4, 8, 6],
             num_init_features=32,
             bottleneck_width=[1, 2, 4, 4],
             include_top=True,
             weights='imagenet',
             input_tensor=None,
             pooling=None,
             dropout_rate=0.05,
             classes=1000,
             **kwargs):
    """Instantiates the PeleeNet architecture.

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

    if type(growth_rate) is list:
        growth_rates = growth_rate
        assert len(growth_rates) == 4, 'The growth rate must be the list and the size must be 4'
    else:
        growth_rates = [growth_rate] * 4

    if type(bottleneck_width) is list:
        bottleneck_widths = bottleneck_width
        assert len(bottleneck_widths) == 4, 'The bottleneck width must be the list and the size must be 4'
    else:
        bottleneck_widths = [bottleneck_width] * 4

    features = stem_block_graph(img_input, num_init_features, name='bbn_features_stemblock')
    num_features = num_init_features
    for i, num_layers in enumerate(block_config):
        features = dense_block_graph(
            features, num_layers=num_layers, bn_size=bottleneck_widths[i],
            growth_rate=growth_rates[i], name='bbn_features_denseblock{}'.format(i + 1))

        num_features = num_features + num_layers * growth_rates[i]
        features = basic_conv2d_graph(
            features, num_features, kernel_size=1, strides=1,
            padding='valid', name='bbn_features_transition{}'.format(i + 1))

        if i != len(block_config) - 1:
            features = AveragePooling2D(pool_size=2, strides=2)(features)

    features_shape = K.int_shape(features)

    if include_top:
        x = GlobalAveragePooling2D()(features)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        x = Dense(classes, activation='softmax',
                         use_bias=True, name='Logits')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(features)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(features)
        else:
            x = features

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='peleenet')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            file_name = 'peleenet_weights_tf_dim_ordering_tf_kernels_224.h5'
            weight_path = BASE_WEIGHT_PATH + file_name
        else:
            file_name = 'peleenet_weights_tf_dim_ordering_tf_kernels_224_no_top.h5'
            weight_path = BASE_WEIGHT_PATH + file_name

        weights_path = get_file(file_name, weight_path, cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


setattr(PeleeNet, '__doc__', PeleeNet.__doc__)


if __name__ == '__main__':
    input_tensor = Input(shape=(None, None, 3), name='image_input')
    #model = PeleeNet(include_top=False, input_tensor=input_tensor, weights='imagenet')
    model = PeleeNet(include_top=True, input_shape=(224, 224, 3), weights='imagenet')
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

