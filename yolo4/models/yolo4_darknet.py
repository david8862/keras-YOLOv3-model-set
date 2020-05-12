#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v4 Model Defined in Keras."""

from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras.utils import get_source_inputs, get_file
from tensorflow.keras.layers import Add, ZeroPadding2D, UpSampling2D, Concatenate
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Flatten, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from yolo4.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, DarknetConv2D_BN_Mish
from yolo4.models.layers import make_yolo_head, make_yolo_spp_head
#from yolo3.models.layers import make_last_layers, make_spp_last_layers


def resblock_body(x, num_filters, num_blocks, all_narrow=True):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Mish(num_filters, (3,3), strides=(2,2))(x)

    res_connection = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(x)
    x = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(x)

    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Mish(num_filters//2, (1,1)),
                DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (3,3)))(x)
        x = Add()([x,y])

    x = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(x)
    x = Concatenate()([x , res_connection])

    return DarknetConv2D_BN_Mish(num_filters, (1,1))(x)


def csp_darknet53_body(x):
    '''CSPDarknet53 body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Mish(32, (3,3))(x)
    x = resblock_body(x, 64, 1, False)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def yolo4_body(inputs, num_anchors, num_classes, weights_path=None):
    """Create YOLO_V4 model CNN body in Keras."""
    darknet = Model(inputs, csp_darknet53_body(inputs))
    if weights_path is not None:
        darknet.load_weights(weights_path, by_name=True)
        print('Load weights {}.'.format(weights_path))

    # f1: 13 x 13 x 1024
    f1 = darknet.output
    # f2: 26 x 26 x 512
    f2 = darknet.layers[204].output
    # f3: 52 x 52 x 256
    f3 = darknet.layers[131].output

    f1_channel_num = 1024
    f2_channel_num = 512
    f3_channel_num = 256

    #feature map 1 head (19x19 for 608 input)
    x1 = make_yolo_spp_head(f1, f1_channel_num//2)

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x1)

    x2 = DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1))(f2)
    x2 = Concatenate()([x2, x1_upsample])

    #feature map 2 head (38x38 for 608 input)
    x2 = make_yolo_head(x2, f2_channel_num//2)

    #upsample fpn merge for feature map 2 & 3
    x2_upsample = compose(
            DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1)),
            UpSampling2D(2))(x2)

    x3 = DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1))(f3)
    x3 = Concatenate()([x3, x2_upsample])

    #feature map 3 head & output (76x76 for 608 input)
    #x3, y3 = make_last_layers(x3, f3_channel_num//2, num_anchors*(num_classes+5))
    x3 = make_yolo_head(x3, f3_channel_num//2)
    y3 = compose(
            DarknetConv2D_BN_Leaky(f3_channel_num, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x3)

    #downsample fpn merge for feature map 3 & 2
    x3_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (3,3), strides=(2,2)))(x3)

    x2 = Concatenate()([x3_downsample, x2])

    #feature map 2 output (38x38 for 608 input)
    #x2, y2 = make_last_layers(x2, 256, num_anchors*(num_classes+5))
    x2 = make_yolo_head(x2, f2_channel_num//2)
    y2 = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            DarknetConv2D_BN_Leaky(f1_channel_num//2, (3,3), strides=(2,2)))(x2)

    x1 = Concatenate()([x2_downsample, x1])

    #feature map 1 output (19x19 for 608 input)
    #x1, y1 = make_last_layers(x1, f1_channel_num//2, num_anchors*(num_classes+5))
    x1 = make_yolo_head(x1, f1_channel_num//2)
    y1 = compose(
            DarknetConv2D_BN_Leaky(f1_channel_num, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x1)

    return Model(inputs, [y1, y2, y3])


BASE_WEIGHT_PATH = (
    'https://github.com/david8862/keras-YOLOv3-model-set/'
    'releases/download/v1.0.1/')

def CSPDarkNet53(input_shape=None,
            input_tensor=None,
            include_top=True,
            weights='imagenet',
            pooling=None,
            classes=1000,
            **kwargs):
    """Generate cspdarknet53 model for Imagenet classification."""

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

    x = csp_darknet53_body(img_input)

    if include_top:
        model_name='cspdarknet53'
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Reshape((1, 1, 1024))(x)
        x = DarknetConv2D(classes, (1, 1))(x)
        x = Flatten()(x)
        x = Softmax(name='Predictions/Softmax')(x)
    else:
        model_name='cspdarknet53_headless'
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
            file_name = 'cspdarknet53_weights_tf_dim_ordering_tf_kernels_224.h5'
            weight_path = BASE_WEIGHT_PATH + file_name
        else:
            file_name = 'cspdarknet53_weights_tf_dim_ordering_tf_kernels_224_no_top.h5'
            weight_path = BASE_WEIGHT_PATH + file_name

        weights_path = get_file(file_name, weight_path, cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model

