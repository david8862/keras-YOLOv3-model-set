#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLOv5 Darknet Model Defined in Keras."""

import os
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras.utils import get_source_inputs, get_file
from tensorflow.keras.layers import Add, ZeroPadding2D, UpSampling2D, Concatenate
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, AveragePooling2D, GlobalMaxPooling2D, Reshape, Flatten, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from yolo4.models.layers import compose, DarknetConv2D
from yolo5.models.layers import make_divisible, DarknetConv2D_BN_Swish, yolo5_predictions, bottleneck_csp_block, focus_block


def csp_resblock_body(x, num_filters, num_blocks, depth_multiple, width_multiple):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Swish(make_divisible(num_filters*width_multiple, 8), (3,3), strides=(2,2))(x)

    x = bottleneck_csp_block(x, num_filters, num_blocks, depth_multiple, width_multiple, shortcut=True)
    return x


def yolo5_darknet_body(x, depth_multiple, width_multiple):
    '''A modified darknet body for YOLOv5'''
    #x = ZeroPadding2D(((3,0),(3,0)))(x)
    #x = DarknetConv2D_BN_Swish(make_divisible(64*width_multiple, 8), (5,5), strides=(2,2))(x)
    x = focus_block(x, 64, width_multiple, kernel=3)

    x = csp_resblock_body(x, 128, 3, depth_multiple, width_multiple)
    x = csp_resblock_body(x, 256, 9, depth_multiple, width_multiple)
    # f3: 52 x 52 x (256*width_multiple)
    f3 = x

    x = csp_resblock_body(x, 512, 9, depth_multiple, width_multiple)
    # f2: 26 x 26 x (512*width_multiple)
    f2 = x

    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Swish(make_divisible(1024*width_multiple, 8), (3,3), strides=(2,2))(x)
    # different with ultralytics PyTorch version, we will try to leave
    # the SPP & BottleneckCSP block to head part

    # f1 = x: 13 x 13 x (1024*width_multiple)
    return x, f2, f3


def yolo5_body(inputs, num_anchors, num_classes, depth_multiple=1.0, width_multiple=1.0, weights_path=None):
    """Create YOLOv5 model CNN body in Keras."""
    # due to depth_multiple, we need to get feature tensors from darknet
    # body function:
    # f1: 13 x 13 x (1024*width_multiple)
    # f2: 26 x 26 x (512*width_multiple)
    # f3: 52 x 52 x (256*width_multiple)
    f1, f2, f3 = yolo5_darknet_body(inputs, depth_multiple, width_multiple)
    darknet = Model(inputs, f1)

    print('backbone layers number: {}'.format(len(darknet.layers)))
    if weights_path is not None:
        darknet.load_weights(weights_path, by_name=True)
        print('Load weights {}.'.format(weights_path))

    f1_channel_num = int(1024*width_multiple)
    f2_channel_num = int(512*width_multiple)
    f3_channel_num = int(256*width_multiple)

    y1, y2, y3 = yolo5_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes, depth_multiple, width_multiple)

    return Model(inputs, [y1, y2, y3])


#BASE_WEIGHT_PATH = (
    #'https://github.com/david8862/keras-YOLOv3-model-set/'
    #'releases/download/v1.0.1/')

#def CSPDarkNet53(input_shape=None,
            #input_tensor=None,
            #include_top=True,
            #weights='imagenet',
            #pooling=None,
            #classes=1000,
            #**kwargs):
    #"""Generate cspdarknet53 model for Imagenet classification."""

    #if not (weights in {'imagenet', None} or os.path.exists(weights)):
        #raise ValueError('The `weights` argument should be either '
                         #'`None` (random initialization), `imagenet` '
                         #'(pre-training on ImageNet), '
                         #'or the path to the weights file to be loaded.')

    #if weights == 'imagenet' and include_top and classes != 1000:
        #raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         #' as true, `classes` should be 1000')

    ## Determine proper input shape
    #input_shape = _obtain_input_shape(input_shape,
                                      #default_size=224,
                                      #min_size=28,
                                      #data_format=K.image_data_format(),
                                      #require_flatten=include_top,
                                      #weights=weights)

    #if input_tensor is None:
        #img_input = Input(shape=input_shape)
    #else:
        #img_input = input_tensor

    #x = csp_darknet53_body(img_input)

    #if include_top:
        #model_name='cspdarknet53'
        ##x = AveragePooling2D(pool_size=2, strides=2, padding='valid', name='avg_pool')(x)
        #x = GlobalAveragePooling2D(name='avg_pool')(x)
        #x = Reshape((1, 1, 1024))(x)
        #x = DarknetConv2D(classes, (1, 1))(x)
        #x = Flatten()(x)
        #x = Softmax(name='Predictions/Softmax')(x)
    #else:
        #model_name='cspdarknet53_headless'
        #if pooling == 'avg':
            #x = GlobalAveragePooling2D(name='avg_pool')(x)
        #elif pooling == 'max':
            #x = GlobalMaxPooling2D(name='max_pool')(x)

    ## Ensure that the model takes into account
    ## any potential predecessors of `input_tensor`.
    #if input_tensor is not None:
        #inputs = get_source_inputs(input_tensor)
    #else:
        #inputs = img_input

    ## Create model.
    #model = Model(inputs, x, name=model_name)

    ## Load weights.
    #if weights == 'imagenet':
        #if include_top:
            #file_name = 'cspdarknet53_weights_tf_dim_ordering_tf_kernels_224.h5'
            #weight_path = BASE_WEIGHT_PATH + file_name
        #else:
            #file_name = 'cspdarknet53_weights_tf_dim_ordering_tf_kernels_224_no_top.h5'
            #weight_path = BASE_WEIGHT_PATH + file_name

        #weights_path = get_file(file_name, weight_path, cache_subdir='models')
        #model.load_weights(weights_path)
    #elif weights is not None:
        #model.load_weights(weights)

    #return model

