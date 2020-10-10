#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v3 Darknet Model Defined in Keras."""

from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, GlobalAveragePooling2D, Flatten, Softmax, Reshape, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from keras_applications.imagenet_utils import _obtain_input_shape

from yolo3.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, Darknet_Depthwise_Separable_Conv2D_BN_Leaky
#from yolo3.models.layers import make_last_layers, make_depthwise_separable_last_layers, make_spp_last_layers
from yolo3.models.layers import yolo3_predictions, yolo3lite_predictions, tiny_yolo3_predictions, tiny_yolo3lite_predictions


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x

def darknet53_body(x):
    '''Darknet53 body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def depthwise_separable_resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = Darknet_Depthwise_Separable_Conv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                Darknet_Depthwise_Separable_Conv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x

def darknet53lite_body(x):
    '''Darknet body having 52 Convolution2D layers'''
    x = Darknet_Depthwise_Separable_Conv2D_BN_Leaky(32, (3,3))(x)
    x = depthwise_separable_resblock_body(x, 64, 1)
    x = depthwise_separable_resblock_body(x, 128, 2)
    x = depthwise_separable_resblock_body(x, 256, 8)
    x = depthwise_separable_resblock_body(x, 512, 8)
    x = depthwise_separable_resblock_body(x, 1024, 4)
    return x


def yolo3_body(inputs, num_anchors, num_classes, weights_path=None):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet53_body(inputs))
    if weights_path is not None:
        darknet.load_weights(weights_path, by_name=True)
        print('Load weights {}.'.format(weights_path))

    # f1: 13 x 13 x 1024
    f1 = darknet.output
    # f2: 26 x 26 x 512
    f2 = darknet.layers[152].output
    # f3: 52 x 52 x 256
    f3 = darknet.layers[92].output

    f1_channel_num = 1024
    f2_channel_num = 512
    f3_channel_num = 256

    y1, y2, y3 = yolo3_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2,y3])


#def custom_yolo3_body(inputs, num_anchors, num_classes, weights_path):
    #'''Create a custom YOLO_v3 model, use
       #pre-trained weights from darknet and fit
       #for our target classes.'''
    ##TODO: get darknet class number from class file
    #num_classes_coco = 80
    #base_model = yolo3_body(inputs, num_anchors, num_classes_coco)
    #base_model.load_weights(weights_path, by_name=True)
    #print('Load weights {}.'.format(weights_path))

    ##base_model.summary()
    ##from tensorflow.keras.utils import plot_model as plot
    ##plot(base_model, to_file='model.png', show_shapes=True)

    ##get conv output in original network
    #y1 = base_model.get_layer('leaky_re_lu_57').output
    #y2 = base_model.get_layer('leaky_re_lu_64').output
    #y3 = base_model.get_layer('leaky_re_lu_71').output
    #y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_1')(y1)
    #y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_2')(y2)
    #y3 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_3')(y3)
    #return Model(inputs, [y1,y2,y3])


def yolo3_spp_body(inputs, num_anchors, num_classes, weights_path=None):
    """Create YOLO_V3 SPP model CNN body in Keras."""
    darknet = Model(inputs, darknet53_body(inputs))
    if weights_path is not None:
        darknet.load_weights(weights_path, by_name=True)
        print('Load weights {}.'.format(weights_path))

    # f1: 13 x 13 x 1024
    f1 = darknet.output
    # f2: 26 x 26 x 512
    f2 = darknet.layers[152].output
    # f3: 52 x 52 x 256
    f3 = darknet.layers[92].output

    f1_channel_num = 1024
    f2_channel_num = 512
    f3_channel_num = 256

    y1, y2, y3 = yolo3_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes, use_spp=True)

    return Model(inputs, [y1,y2,y3])


def custom_yolo3_spp_body(inputs, num_anchors, num_classes, weights_path):
    '''Create a custom YOLO_v3 SPP model, use
       pre-trained weights from darknet and fit
       for our target classes.'''
    #TODO: get darknet class number from class file
    num_classes_coco = 80
    base_model = yolo3_spp_body(inputs, num_anchors, num_classes_coco)
    base_model.load_weights(weights_path, by_name=False)
    print('Load weights {}.'.format(weights_path))

    # reform the predict conv layer for custom dataset classes
    #y1 = base_model.get_layer('leaky_re_lu_58').output
    #y2 = base_model.get_layer('leaky_re_lu_65').output
    #y3 = base_model.get_layer('leaky_re_lu_72').output
    y1 = base_model.layers[-6].output
    y2 = base_model.layers[-5].output
    y3 = base_model.layers[-4].output
    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_1')(y1)
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_2')(y2)
    y3 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_3')(y3)
    return Model(inputs, [y1,y2,y3])


def yolo3lite_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 Lite model CNN body in Keras."""
    darknetlite = Model(inputs, darknet53lite_body(inputs))

    # f1: 13 x 13 x 1024
    f1 = darknetlite.output
    # f2: 26 x 26 x 512
    f2 = darknetlite.layers[152].output
    # f3: 52 x 52 x 256
    f3 = darknetlite.layers[92].output

    f1_channel_num = 1024
    f2_channel_num = 512
    f3_channel_num = 256

    y1, y2, y3 = yolo3lite_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2,y3])


def tiny_yolo3_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    #feature map 2 (26x26x256 for 416 input)
    f2 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)

    #feature map 1 (13x13x1024 for 416 input)
    f1 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)))(f2)

    #feature map 1 transform
    x1 = DarknetConv2D_BN_Leaky(256, (1,1))(f1)

    #feature map 1 output (13x13 for 416 input)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_1'))(x1)

    #upsample fpn merge for feature map 1 & 2
    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x1)

    #feature map 2 output (26x26 for 416 input)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_2'))([x2, f2])

    return Model(inputs, [y1,y2])


def custom_tiny_yolo3_body(inputs, num_anchors, num_classes, weights_path):
    '''Create a custom Tiny YOLO_v3 model, use
       pre-trained weights from darknet and fit
       for our target classes.'''
    #TODO: get darknet class number from class file
    num_classes_coco = 80
    base_model = tiny_yolo3_body(inputs, num_anchors, num_classes_coco)
    base_model.load_weights(weights_path, by_name=False)
    print('Load weights {}.'.format(weights_path))

    #get conv output in original network
    #y1 = base_model.get_layer('leaky_re_lu_8').output
    #y2 = base_model.get_layer('leaky_re_lu_10').output
    y1 = base_model.layers[40].output
    y2 = base_model.layers[41].output
    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_1')(y1)
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_2')(y2)
    return Model(inputs, [y1,y2])


def tiny_yolo3lite_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 Lite model CNN body in keras.'''
    #feature map 2 (26x26x256 for 416 input)
    f2 = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            Depthwise_Separable_Conv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            Depthwise_Separable_Conv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            Depthwise_Separable_Conv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            Depthwise_Separable_Conv2D_BN_Leaky(256, (3,3)))(inputs)

    #feature map 1 (13x13x1024 for 416 input)
    f1 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            Depthwise_Separable_Conv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            Depthwise_Separable_Conv2D_BN_Leaky(1024, (3,3)))(f2)

    #feature map 1 transform
    x1 = DarknetConv2D_BN_Leaky(256, (1,1))(f1)

    #feature map 1 output (13x13 for 416 input)
    y1 = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_1'))(x2)

    #upsample fpn merge for feature map 1 & 2
    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x1)

    #feature map 2 output (26x26 for 416 input)
    y2 = compose(
            Concatenate(),
            Depthwise_Separable_Conv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_2'))([x2, f2])

    return Model(inputs, [y1,y2])


BASE_WEIGHT_PATH = (
    'https://github.com/david8862/keras-YOLOv3-model-set/'
    'releases/download/v1.0.1/')

def DarkNet53(input_shape=None,
            input_tensor=None,
            include_top=True,
            weights='imagenet',
            pooling=None,
            classes=1000,
            **kwargs):
    """Generate darknet53 model for Imagenet classification."""

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

    x = darknet53_body(img_input)

    if include_top:
        model_name='darknet53'
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Reshape((1, 1, 1024))(x)
        x = DarknetConv2D(classes, (1, 1))(x)
        x = Flatten()(x)
        x = Softmax(name='Predictions/Softmax')(x)
    else:
        model_name='darknet53_headless'
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
            file_name = 'darknet53_weights_tf_dim_ordering_tf_kernels_224.h5'
            weight_path = BASE_WEIGHT_PATH + file_name
        else:
            file_name = 'darknet53_weights_tf_dim_ordering_tf_kernels_224_no_top.h5'
            weight_path = BASE_WEIGHT_PATH + file_name

        weights_path = get_file(file_name, weight_path, cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model

