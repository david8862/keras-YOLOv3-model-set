#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v3 Darknet Model Defined in Keras."""

from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from tensorflow.keras.models import Model

from yolo3.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, Darknet_Depthwise_Separable_Conv2D_BN_Leaky
from yolo3.models.layers import make_last_layers, make_depthwise_separable_last_layers, make_spp_last_layers


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
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

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
    #y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='prediction_13')(y1)
    #y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='prediction_26')(y2)
    #y3 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='prediction_52')(y3)
    #return Model(inputs, [y1,y2,y3])


def yolo3_spp_body(inputs, num_anchors, num_classes, weights_path=None):
    """Create YOLO_V3 SPP model CNN body in Keras."""
    darknet = Model(inputs, darknet53_body(inputs))
    if weights_path is not None:
        darknet.load_weights(weights_path, by_name=True)
        print('Load weights {}.'.format(weights_path))
    #x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))
    x, y1 = make_spp_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1,y2,y3])


def custom_yolo3_spp_body(inputs, num_anchors, num_classes, weights_path):
    '''Create a custom YOLO_v3 SPP model, use
       pre-trained weights from darknet and fit
       for our target classes.'''
    #TODO: get darknet class number from class file
    num_classes_coco = 80
    base_model = yolo3_spp_body(inputs, num_anchors, num_classes_coco)
    base_model.load_weights(weights_path, by_name=True)
    print('Load weights {}.'.format(weights_path))

    #get conv output in original network
    y1 = base_model.get_layer('leaky_re_lu_58').output
    y2 = base_model.get_layer('leaky_re_lu_65').output
    y3 = base_model.get_layer('leaky_re_lu_72').output
    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='prediction_13')(y1)
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='prediction_26')(y2)
    y3 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='prediction_52')(y3)
    return Model(inputs, [y1,y2,y3])


def yolo3lite_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 Lite model CNN body in Keras."""
    darknetlite = Model(inputs, darknet53lite_body(inputs))
    x, y1 = make_depthwise_separable_last_layers(darknetlite.output, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknetlite.layers[152].output])
    x, y2 = make_depthwise_separable_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknetlite.layers[92].output])
    x, y3 = make_depthwise_separable_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1,y2,y3])


def tiny_yolo3_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])

def custom_tiny_yolo3_body(inputs, num_anchors, num_classes, weights_path):
    '''Create a custom Tiny YOLO_v3 model, use
       pre-trained weights from darknet and fit
       for our target classes.'''
    #TODO: get darknet class number from class file
    num_classes_coco = 80
    base_model = tiny_yolo3_body(inputs, num_anchors, num_classes_coco)
    base_model.load_weights(weights_path, by_name=True)
    print('Load weights {}.'.format(weights_path))

    #get conv output in original network
    y1 = base_model.get_layer('leaky_re_lu_8').output
    y2 = base_model.get_layer('leaky_re_lu_10').output
    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='prediction_13')(y1)
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='prediction_26')(y2)
    return Model(inputs, [y1,y2])


def tiny_yolo3lite_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 Lite model CNN body in keras.'''
    x1 = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            Depthwise_Separable_Conv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            Depthwise_Separable_Conv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            Depthwise_Separable_Conv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            Depthwise_Separable_Conv2D_BN_Leaky(256, (3,3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            Depthwise_Separable_Conv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            Depthwise_Separable_Conv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            Depthwise_Separable_Conv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])
