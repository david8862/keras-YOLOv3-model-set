#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v4 MobileNet Model Defined in Keras."""

from tensorflow.keras.layers import ZeroPadding2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet

from yolo4.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, Darknet_Depthwise_Separable_Conv2D_BN_Leaky, make_yolo_head, make_yolo_spp_head, make_yolo_depthwise_separable_head, make_yolo_spp_depthwise_separable_head


def yolo4_mobilenet_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V4 MobileNet model CNN body in Keras."""
    mobilenet = MobileNet(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # conv_pw_13_relu :13 x 13 x (1024*alpha)
    # conv_pw_11_relu :26 x 26 x (512*alpha)
    # conv_pw_5_relu : 52 x 52 x (256*alpha)

    f1 = mobilenet.get_layer('conv_pw_13_relu').output
    # f1 :13 x 13 x (1024*alpha) for 416 input
    #feature map 1 head (13 x 13 x (512*alpha) for 416 input)
    x1 = make_yolo_spp_head(f1, int(512*alpha))

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Leaky(int(256*alpha), (1,1)),
            UpSampling2D(2))(x1)

    f2 = mobilenet.get_layer('conv_pw_11_relu').output
    # f2: 26 x 26 x (512*alpha) for 416 input
    x2 = DarknetConv2D_BN_Leaky(int(256*alpha), (1,1))(f2)
    x2 = Concatenate()([x2, x1_upsample])

    #feature map 2 head (26 x 26 x (256*alpha) for 416 input)
    x2 = make_yolo_head(x2, int(256*alpha))

    #upsample fpn merge for feature map 2 & 3
    x2_upsample = compose(
            DarknetConv2D_BN_Leaky(int(128*alpha), (1,1)),
            UpSampling2D(2))(x2)

    f3 = mobilenet.get_layer('conv_pw_5_relu').output
    # f3 : 52 x 52 x  (256*alpha) for 416 input
    x3 = DarknetConv2D_BN_Leaky(int(128*alpha), (1,1))(f3)
    x3 = Concatenate()([x3, x2_upsample])

    #feature map 3 head & output (52 x 52 x (256*alpha) for 416 input)
    #x3, y3 = make_last_layers(x3, int(128*alpha), num_anchors*(num_classes+5))
    x3 = make_yolo_head(x3, int(128*alpha))
    y3 = compose(
            DarknetConv2D_BN_Leaky(int(256*alpha), (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x3)

    #downsample fpn merge for feature map 3 & 2
    x3_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            DarknetConv2D_BN_Leaky(int(256*alpha), (3,3), strides=(2,2)))(x3)

    x2 = Concatenate()([x3_downsample, x2])

    #feature map 2 output (26 x 26 x (512*alpha) for 416 input)
    #x2, y2 = make_last_layers(x2, int(256*alpha), num_anchors*(num_classes+5))
    x2 = make_yolo_head(x2, int(256*alpha))
    y2 = compose(
            DarknetConv2D_BN_Leaky(int(512*alpha), (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            DarknetConv2D_BN_Leaky(int(512*alpha), (3,3), strides=(2,2)))(x2)

    x1 = Concatenate()([x2_downsample, x1])

    #feature map 1 output (13 x 13 x (1024*alpha) for 416 input)
    #x1, y1 = make_last_layers(x1, int(512*alpha), num_anchors*(num_classes+5))
    x1 = make_yolo_head(x1, int(512*alpha))
    y1 = compose(
            DarknetConv2D_BN_Leaky(int(1024*alpha), (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x1)

    return Model(inputs, [y1, y2, y3])


def yolo4lite_mobilenet_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create YOLO_v4 Lite MobileNet model CNN body in keras.'''
    mobilenet = MobileNet(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # conv_pw_13_relu :13 x 13 x (1024*alpha)
    # conv_pw_11_relu :26 x 26 x (512*alpha)
    # conv_pw_5_relu : 52 x 52 x (256*alpha)

    f1 = mobilenet.get_layer('conv_pw_13_relu').output
    # f1 :13 x 13 x (1024*alpha) for 416 input
    #feature map 1 head (13 x 13 x (512*alpha) for 416 input)
    x1 = make_yolo_spp_depthwise_separable_head(f1, int(512*alpha), block_id_str='14')

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Leaky(int(256*alpha), (1,1)),
            UpSampling2D(2))(x1)

    f2 = mobilenet.get_layer('conv_pw_11_relu').output
    # f2: 26 x 26 x (512*alpha) for 416 input
    x2 = DarknetConv2D_BN_Leaky(int(256*alpha), (1,1))(f2)
    x2 = Concatenate()([x2, x1_upsample])

    #feature map 2 head (26 x 26 x (256*alpha) for 416 input)
    x2 = make_yolo_depthwise_separable_head(x2, int(256*alpha), block_id_str='15')

    #upsample fpn merge for feature map 2 & 3
    x2_upsample = compose(
            DarknetConv2D_BN_Leaky(int(128*alpha), (1,1)),
            UpSampling2D(2))(x2)

    f3 = mobilenet.get_layer('conv_pw_5_relu').output
    # f3 : 52 x 52 x  (256*alpha) for 416 input
    x3 = DarknetConv2D_BN_Leaky(int(128*alpha), (1,1))(f3)
    x3 = Concatenate()([x3, x2_upsample])

    #feature map 3 head & output (52 x 52 x (256*alpha) for 416 input)
    #x3, y3 = make_depthwise_separable_last_layers(x3, int(128*alpha), num_anchors*(num_classes+5), block_id_str='16')
    x3 = make_yolo_depthwise_separable_head(x3, int(128*alpha), block_id_str='16')
    y3 = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(int(256*alpha), (3,3), block_id_str='16_3'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x3)

    #downsample fpn merge for feature map 3 & 2
    x3_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            Darknet_Depthwise_Separable_Conv2D_BN_Leaky(int(256*alpha), (3,3), strides=(2,2), block_id_str='16_4'))(x3)

    x2 = Concatenate()([x3_downsample, x2])

    #feature map 2 output (26 x 26 x (512*alpha) for 416 input)
    #x2, y2 = make_depthwise_separable_last_layers(x2, int(256*alpha), num_anchors*(num_classes+5), block_id_str='17')
    x2 = make_yolo_depthwise_separable_head(x2, int(256*alpha), block_id_str='17')
    y2 = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(int(512*alpha), (3,3), block_id_str='17_3'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            Darknet_Depthwise_Separable_Conv2D_BN_Leaky(int(512*alpha), (3,3), strides=(2,2), block_id_str='17_4'))(x2)

    x1 = Concatenate()([x2_downsample, x1])

    #feature map 1 output (13 x 13 x (1024*alpha) for 416 input)
    #x1, y1 = make_depthwise_separable_last_layers(x1, int(512*alpha), num_anchors*(num_classes+5), block_id_str='18')
    x1 = make_yolo_depthwise_separable_head(x1, int(512*alpha), block_id_str='18')
    y1 = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(int(1024*alpha), (3,3), block_id_str='18_3'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x1)

    return Model(inputs, [y1, y2, y3])


def tiny_yolo4_mobilenet_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create Tiny YOLO_v4 MobileNet model CNN body in keras.'''
    mobilenet = MobileNet(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # conv_pw_13_relu :13 x 13 x (1024*alpha)
    # conv_pw_11_relu :26 x 26 x (512*alpha)
    # conv_pw_5_relu : 52 x 52 x (256*alpha)

    # f1 :13 x 13 x (1024*alpha) for 416 input
    f1 = mobilenet.get_layer('conv_pw_13_relu').output
    # f2: 26 x 26 x (512*alpha) for 416 input
    f2 = mobilenet.get_layer('conv_pw_11_relu').output

    #feature map 1 head (13 x 13 x (512*alpha) for 416 input)
    x1 = DarknetConv2D_BN_Leaky(int(512*alpha), (1,1))(f1)

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Leaky(int(256*alpha), (1,1)),
            UpSampling2D(2))(x1)
    x2 = compose(
            Concatenate(),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=int(512*alpha), kernel_size=(3, 3), block_id_str='15'),
            DarknetConv2D_BN_Leaky(int(512*alpha), (3,3)))([x1_upsample, f2])

    #feature map 2 output (26 x 26 x (512*alpha) for 416 input)
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            #Darknet_Depthwise_Separable_Conv2D_BN_Leaky(int(512*alpha), (3,3), strides=(2,2), block_id_str='16'),
            DarknetConv2D_BN_Leaky(int(512*alpha), (3,3), strides=(2,2)))(x2)
    x1 = compose(
            Concatenate(),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=int(1024*alpha), kernel_size=(3, 3), block_id_str='17'),
            DarknetConv2D_BN_Leaky(int(1024*alpha), (3,3)))([x2_downsample, x1])

    #feature map 1 output (13 x 13 x (1024*alpha) for 416 input)
    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(x1)

    return Model(inputs, [y1,y2])


def tiny_yolo4lite_mobilenet_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create Tiny YOLO_v3 Lite MobileNet model CNN body in keras.'''
    mobilenet = MobileNet(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # conv_pw_13_relu :13 x 13 x (1024*alpha)
    # conv_pw_11_relu :26 x 26 x (512*alpha)
    # conv_pw_5_relu : 52 x 52 x (256*alpha)

    # f1 :13 x 13 x (1024*alpha) for 416 input
    f1 = mobilenet.get_layer('conv_pw_13_relu').output
    # f2: 26 x 26 x (512*alpha) for 416 input
    f2 = mobilenet.get_layer('conv_pw_11_relu').output

    #feature map 1 head (13 x 13 x (512*alpha) for 416 input)
    x1 = DarknetConv2D_BN_Leaky(int(512*alpha), (1,1))(f1)

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Leaky(int(256*alpha), (1,1)),
            UpSampling2D(2))(x1)
    x2 = compose(
            Concatenate(),
            #DarknetConv2D_BN_Leaky(int(512*alpha), (3,3)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=int(512*alpha), kernel_size=(3, 3), block_id_str='15'))([x1_upsample, f2])

    #feature map 2 output (26 x 26 x (512*alpha) for 416 input)
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            #DarknetConv2D_BN_Leaky(int(512*alpha), (3,3), strides=(2,2)),
            Darknet_Depthwise_Separable_Conv2D_BN_Leaky(int(512*alpha), (3,3), strides=(2,2), block_id_str='16'))(x2)
    x1 = compose(
            Concatenate(),
            #DarknetConv2D_BN_Leaky(int(1024*alpha), (3,3)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=int(1024*alpha), kernel_size=(3, 3), block_id_str='17'))([x2_downsample, x1])

    #feature map 1 output (13 x 13 x (1024*alpha) for 416 input)
    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(x1)

    return Model(inputs, [y1,y2])

