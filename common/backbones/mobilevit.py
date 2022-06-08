#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# A tf.keras implementation of MobileViT,
# ported from https://keras.io/examples/vision/mobilevit/
#
# Reference:
#   [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178)
#   https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py
#
import os, sys
import warnings
import math

from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import preprocess_input as _preprocess_input
from tensorflow.keras.utils import get_source_inputs, get_file
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, ZeroPadding2D, Lambda
from tensorflow.keras.layers import Input, BatchNormalization, Add, Reshape, LayerNormalization, MultiHeadAttention, Concatenate, Activation
from tensorflow.keras.activations import swish
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf

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
    x = _preprocess_input(x, mode='tf', backend=K)

    return x


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


def conv_block(x, filters=16, kernel_size=3, strides=2, name=''):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = YoloConv2D(filters,
               kernel_size,
               strides=strides,
               padding='same',
               use_bias=False,
               name=name)(x)
    x = CustomBatchNormalization(axis=channel_axis,
                           momentum=0.1,
                           name=name+'_BN')(x)
    x = Activation(swish, name=name+'_swish')(x)
    return x


# Reference: https://git.io/JKgtC
def inverted_residual_block(inputs, expanded_channels, output_channels, strides, block_id):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    prefix = 'mv2_block_{}_'.format(block_id)

    # Expand
    x = YoloConv2D(expanded_channels, 1, padding='same', use_bias=False, name=prefix+'_expand')(inputs)
    x = CustomBatchNormalization(axis=channel_axis,
                           momentum=0.1,
                           name=prefix+'expand_BN')(x)
    x = Activation(swish, name=prefix+'expand_swish')(x)

    # Depthwise
    if strides == 2:
        x = ZeroPadding2D(padding=correct_pad(K, x, 3), name=prefix+'pad')(x)

    x = YoloDepthwiseConv2D(kernel_size=3,
                        strides=strides,
                        activation=None,
                        use_bias=False,
                        padding='same' if strides == 1 else 'valid',
                        name=prefix+'depthwise')(x)
    x = CustomBatchNormalization(axis=channel_axis,
                           momentum=0.1,
                           name=prefix+'depthwise_BN')(x)
    x = Activation(swish, name=prefix+'depthwise_swish')(x)

    # Project
    x = YoloConv2D(output_channels,
               kernel_size=1,
               padding='same',
               use_bias=False,
               activation=None,
               name=prefix+'project')(x)
    x = CustomBatchNormalization(axis=channel_axis,
                           momentum=0.1,
                           name=prefix+'project_BN')(x)

    if inputs.shape[-1] == output_channels and strides == 1:
        return Add(name=prefix+'add')([inputs, x])
    return x


# Reference:
# https://keras.io/examples/vision/image_classification_with_vision_transformer/
def feedforward(x, hidden_units, dropout_rate, name):
    for i, units in enumerate(hidden_units):
        prefix = name + '_' + str(i)
        x = Dense(units, activation=swish, name=prefix+'_dense')(x)
        x = Dropout(dropout_rate, name=prefix+'_dropout')(x)
    return x


def transformer_block(x, projection_dim, num_heads, dropout, prefix):
    """
    Transformer encoder block for MobileViT. See official pytorch implementation:
    https://github.com/apple/ml-cvnets/blob/main/cvnets/modules/transformer.py
    """
    # Layer normalization 1.
    x1 = LayerNormalization(epsilon=1e-6, name=prefix+'_LN1')(x)
    # Create a multi-head attention layer.
    attention_output = MultiHeadAttention(num_heads=num_heads,
                                          key_dim=projection_dim,
                                          dropout=dropout,
                                          name=prefix+'_attention')(x1, x1)
    # Skip connection 1.
    x2 = Add(name=prefix+'_add1')([attention_output, x])
    # Layer normalization 2.
    x3 = LayerNormalization(epsilon=1e-6, name=prefix+'_LN2')(x2)
    # FeedForward network.
    x3 = feedforward(x3, hidden_units=[x.shape[-1] * 2, x.shape[-1]],
                     dropout_rate=dropout,
                     name=prefix+'_ff')
    # Skip connection 2.
    x = Add(name=prefix+'_add2')([x3, x2])

    return x


def img_resize(x, size, mode='bilinear'):
    if mode == 'bilinear':
        return tf.image.resize(x, size=size, method='bilinear')
    elif mode == 'nearest':
        return tf.image.resize(x, size=size, method='nearest')
    elif mode == 'bicubic':
        return tf.image.resize(x, size=size, method='bicubic')
    elif mode == 'area':
        return tf.image.resize(x, size=size, method='area')
    elif mode == 'gaussian':
        return tf.image.resize(x, size=size, method='gaussian')
    else:
        raise ValueError('invalid resize type {}'.format(mode))


def unfolding(x, patch_h, patch_w, prefix):
    batch_size, orig_h, orig_w, in_channels = x.shape

    # get tensor width & height aligned with patch size
    new_h = int(math.ceil(orig_h / patch_h) * patch_h)
    new_w = int(math.ceil(orig_w / patch_w) * patch_w)

    if new_h != orig_h or new_w != orig_w:
        # resize feature tensor for unfolding
        x = Lambda(img_resize,
                   arguments={'size': (new_h, new_w), 'mode': 'bilinear'},
                   name=prefix+'unfold_resize')(x)

    # number of patches along new width and height
    num_patch_w = new_w // patch_w # n_w
    num_patch_h = new_h // patch_h # n_h
    num_patches = num_patch_h * num_patch_w # N
    patch_size = patch_h * patch_w # P

    # [new_h, new_w, C] --> [P, N, C]
    x = Reshape((patch_size, num_patches, -1),
                name=prefix+'unfold')(x)

    return x, new_h, new_w


def mobilevit_block(x, num_blocks, num_heads, projection_dim, strides, dropout, block_id):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    in_channels = x.shape[channel_axis]
    prefix = 'mvit_block_{}_'.format(block_id)

    # Local projection with convolutions.
    local_features = conv_block(x, filters=projection_dim,
                                strides=strides,
                                name=prefix+'conv1')
    local_features = conv_block(local_features,
                                filters=projection_dim,
                                kernel_size=1,
                                strides=strides,
                                name=prefix+'conv2')

    # Unfold into patches and then pass through Transformers.
    patch_h, patch_w = 2, 2 # 2x2, for the Transformer blocks.
    non_overlapping_patches, new_h, new_w = unfolding(local_features, patch_h, patch_w, prefix)

    # Transformer blocks
    global_features = non_overlapping_patches
    for i in range(num_blocks):
        name = prefix + 'transformer_' + str(i)
        global_features = transformer_block(global_features,
                                            projection_dim,
                                            num_heads=num_heads,
                                            dropout=dropout,
                                            prefix=name)

    # Fold into conv-like feature-maps.
    folded_feature_map = Reshape((new_h, new_w, projection_dim),
                                 name=prefix+'fold')(global_features)

    # resize back to local feature shape
    orig_h, orig_w = local_features.shape[1:-1]
    if new_h != orig_h or new_w != orig_w:
        folded_feature_map = Lambda(img_resize,
                                    arguments={'size': (orig_h, orig_w), 'mode': 'bilinear'},
                                    name=prefix+'fold_resize')(folded_feature_map)

    # Apply point-wise conv -> concatenate with the input features.
    folded_feature_map = conv_block(folded_feature_map,
                                    filters=in_channels,
                                    kernel_size=1,
                                    strides=strides,
                                    name=prefix+'conv3')

    local_global_features = Concatenate(axis=-1,
                                        name=prefix+'concat')([x, folded_feature_map])

    # Fuse the local and global features using a convoluion layer.
    local_global_features = conv_block(local_global_features,
                                       #filters=projection_dim,
                                       filters=in_channels,
                                       strides=strides,
                                       name=prefix+'conv4')
    return local_global_features



def MobileViT(channels,
              dims,
              expansion=4,
              model_type='xxs',
              input_shape=None,
              include_top=True,
              weights='imagenet',
              input_tensor=None,
              classes=1000,
              pooling=None,
              dropout_rate=0.1,
              **kwargs):
    """Instantiates the MobileViT architecture.
    # Arguments
        channels: a list that defines channel number for each stage.
        dims: a list that defines project dimention of each MobileViT block.
        expansion: integer number for expansion ratio in MV2 blocks
        model_type: MobileViT is defined as three models: s, xs and xxs. These
        models are targeted at high and low resource use cases respectively.
        input_shape: optional shape tuple, to be specified if you would
            like to use a model with an input img resolution that is not
            (256, 256, 3).
            It should have exactly 3 inputs channels (256, 256, 3).
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
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        dropout_rate: fraction of the input units to drop on the last layer
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid model type, argument for `weights`,
            or invalid input shape when weights='imagenet'
    """
    # Check TF version for compatibility
    import tensorflow as tf
    if float(tf.__version__[:3]) < 2.4:
        raise ValueError('Only valid for tensorflow >= 2.4 with MultiHeadAttention Layer support.')

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
                                      default_size=256,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if K.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if rows and cols and (rows < 32 or cols < 32):
        raise ValueError('Input size must be at least 32x32; got `input_shape=' +
                         str(input_shape) + '`')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        rows = input_tensor.shape[row_axis+1]
        cols = input_tensor.shape[col_axis+1]
        if rows is None or cols is None:
            raise ValueError('Does not support dynamic input shape; got input tensor with shape `' +
                             str(input_tensor.shape) + '`')
        img_input = input_tensor

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # Transformer block_number for each MobileViT block
    mvit_blocks = [2, 4, 3]

    # TODO: In official model configs (see following link) the num of heads in transformer is 4,
    #       but seems when using num_heads=4 here, the params statistics doesn't match with paper
    #       (not sure why, but num_heads only impact tf.keras.layers.MultiHeadAttention), so we just
    #       change to "num_heads = 1"
    #
    # Official model config:
    #       https://github.com/apple/ml-cvnets/blob/main/config/classification/mobilevit_small.yaml#L64
    #
    num_heads = 1

    # Initial stem-conv -> MV2 block.
    x = conv_block(img_input, filters=channels[0], name='stem_conv')
    x = inverted_residual_block(
        x, expanded_channels=channels[0]*expansion, output_channels=channels[1], strides=1, block_id=0)

    # Downsampling with MV2 block.
    x = inverted_residual_block(
        x, expanded_channels=channels[1] * expansion, output_channels=channels[2], strides=2, block_id=1)
    x = inverted_residual_block(
        x, expanded_channels=channels[2] * expansion, output_channels=channels[3], strides=1, block_id=2)
    # Repeat block
    x = inverted_residual_block(
        x, expanded_channels=channels[2] * expansion, output_channels=channels[3], strides=1, block_id=3)

    # First MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=channels[3] * expansion, output_channels=channels[4], strides=2, block_id=4)
    x = mobilevit_block(x, num_blocks=mvit_blocks[0], num_heads=num_heads, projection_dim=dims[0], strides=1, dropout=0.1, block_id=0)

    # Second MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=channels[5] * expansion, output_channels=channels[5], strides=2, block_id=5)
    x = mobilevit_block(x, num_blocks=mvit_blocks[1], num_heads=num_heads, projection_dim=dims[1], strides=1, dropout=0.1, block_id=1)

    # Third MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=channels[6] * expansion, output_channels=channels[6], strides=2, block_id=6)
    x = mobilevit_block(x, num_blocks=mvit_blocks[2], num_heads=num_heads, projection_dim=dims[2], strides=1, dropout=0.1, block_id=2)

    x = conv_block(x, filters=channels[7], kernel_size=1, strides=1, name='1x1_conv')

    if include_top:
        # Classification head.
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate, name='predict_dropout')(x)
        x = Dense(classes, activation='softmax',
                  use_bias=True, name='logits')(x)
    else:
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
    model = Model(inputs, x, name='mobilevit_%s' % (model_type))

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            model_name = ('mobilevit_' + model_type + '_weights_tf_dim_ordering_tf_kernels_256.h5')
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = get_file(
                model_name, weight_path, cache_subdir='models')
        else:
            model_name = ('mobilevit_' + model_type + '_weights_tf_dim_ordering_tf_kernels_256_no_top.h5')
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = get_file(
                model_name, weight_path, cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model



def MobileViT_S(input_shape=None,
                include_top=True,
                weights='imagenet',
                input_tensor=None,
                classes=1000,
                pooling=None,
                **kwargs):
    channels = [16, 32, 64, 64, 96, 128, 160, 640]
    dims = [144, 192, 240]
    expansion = 4

    return MobileViT(channels, dims, expansion, 's', input_shape, include_top, weights, input_tensor, classes, pooling, **kwargs)



def MobileViT_XS(input_shape=None,
                 include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 classes=1000,
                 pooling=None,
                 **kwargs):
    channels = [16, 32, 48, 48, 64, 80, 96, 384]
    dims = [96, 120, 144]
    expansion = 4

    return MobileViT(channels, dims, expansion, 'xs', input_shape, include_top, weights, input_tensor, classes, pooling, **kwargs)


def MobileViT_XXS(input_shape=None,
                  include_top=True,
                  weights='imagenet',
                  input_tensor=None,
                  classes=1000,
                  pooling=None,
                  **kwargs):
    channels = [16, 16, 24, 24, 48, 64, 80, 320]
    dims = [64, 80, 96]
    expansion = 2

    return MobileViT(channels, dims, expansion, 'xxs', input_shape, include_top, weights, input_tensor, classes, pooling, **kwargs)



if __name__ == '__main__':
    input_tensor = Input(shape=(None, None, 3), name='image_input')
    #model = MobileViT_XXS(include_top=False, input_tensor=input_tensor, weights='imagenet')
    model = MobileViT_XXS(include_top=True, input_shape=(256, 256, 3), weights='imagenet')
    model.summary()
    K.set_learning_phase(0)

    import numpy as np
    from tensorflow.keras.applications.resnet50 import decode_predictions
    from keras_preprocessing import image

    img = image.load_img('../../example/eagle.jpg', target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
