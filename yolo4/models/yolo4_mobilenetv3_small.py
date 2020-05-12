#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v4 MobileNetV3Small Model Defined in Keras."""

from tensorflow.keras.layers import ZeroPadding2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from common.backbones.mobilenet_v3 import MobileNetV3Small

from yolo4.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Spp_Conv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, Darknet_Depthwise_Separable_Conv2D_BN_Leaky, make_yolo_head, make_yolo_spp_head, make_yolo_depthwise_separable_head, make_yolo_spp_depthwise_separable_head


def yolo4_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V4 MobileNetV3Small model CNN body in Keras."""
    mobilenetv3small = MobileNetV3Small(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # activation_31(layer 165, final feature map): 13 x 13 x (576*alpha)
    # expanded_conv_10/Add(layer 162, end of block10): 13 x 13 x (96*alpha)

    # activation_22(layer 117, middle in block8) : 26 x 26 x (288*alpha)
    # expanded_conv_7/Add(layer 114, end of block7) : 26 x 26 x (48*alpha)

    # activation_7(layer 38, middle in block3) : 52 x 52 x (96*alpha)
    # expanded_conv_2/Add(layer 35, end of block2): 52 x 52 x (24*alpha)

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1: 13 x 13 x (576*alpha)
    f1 = mobilenetv3small.layers[165].output
    # f2: 26 x 26 x (288*alpha) for 416 input
    f2 = mobilenetv3small.layers[117].output
    # f3: 52 x 52 x (96*alpha)
    f3 = mobilenetv3small.layers[38].output

    f1_channel_num = int(576*alpha)
    f2_channel_num = int(288*alpha)
    f3_channel_num = int(96*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512
    #f3_channel_num = 256

    #feature map 1 head (13 x 13 x f1_channel_num//2 for 416 input)
    x1 = make_yolo_spp_head(f1, f1_channel_num//2)

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x1)

    x2 = DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1))(f2)
    x2 = Concatenate()([x2, x1_upsample])

    #feature map 2 head (26 x 26 x f2_channel_num//2 for 416 input)
    x2 = make_yolo_head(x2, f2_channel_num//2)

    #upsample fpn merge for feature map 2 & 3
    x2_upsample = compose(
            DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1)),
            UpSampling2D(2))(x2)

    x3 = DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1))(f3)
    x3 = Concatenate()([x3, x2_upsample])

    #feature map 3 head & output (52 x 52 x f3_channel_num for 416 input)
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

    #feature map 2 output (26 x 26 x f2_channel_num for 416 input)
    #x2, y2 = make_last_layers(x2, f2_channel_num//2, num_anchors*(num_classes+5))
    x2 = make_yolo_head(x2, f2_channel_num//2)
    y2 = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            DarknetConv2D_BN_Leaky(f1_channel_num//2, (3,3), strides=(2,2)))(x2)

    x1 = Concatenate()([x2_downsample, x1])

    #feature map 1 output (13 x 13 x f1_channel_num for 416 input)
    #x1, y1 = make_last_layers(x1, f1_channel_num//2, num_anchors*(num_classes+5))
    x1 = make_yolo_head(x1, f1_channel_num//2)
    y1 = compose(
            DarknetConv2D_BN_Leaky(f1_channel_num, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x1)

    return Model(inputs, [y1, y2, y3])


def yolo4lite_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create YOLO_v4 Lite MobileNetV3Small model CNN body in keras.'''
    mobilenetv3small = MobileNetV3Small(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # activation_31(layer 165, final feature map): 13 x 13 x (576*alpha)
    # expanded_conv_10/Add(layer 162, end of block10): 13 x 13 x (96*alpha)

    # activation_22(layer 117, middle in block8) : 26 x 26 x (288*alpha)
    # expanded_conv_7/Add(layer 114, end of block7) : 26 x 26 x (48*alpha)

    # activation_7(layer 38, middle in block3) : 52 x 52 x (96*alpha)
    # expanded_conv_2/Add(layer 35, end of block2): 52 x 52 x (24*alpha)

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1: 13 x 13 x (576*alpha)
    f1 = mobilenetv3small.layers[165].output
    # f2: 26 x 26 x (288*alpha) for 416 input
    f2 = mobilenetv3small.layers[117].output
    # f3: 52 x 52 x (96*alpha)
    f3 = mobilenetv3small.layers[38].output

    f1_channel_num = int(576*alpha)
    f2_channel_num = int(288*alpha)
    f3_channel_num = int(96*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512
    #f3_channel_num = 256

    #feature map 1 head (13 x 13 x f1_channel_num//2 for 416 input)
    x1 = make_yolo_spp_depthwise_separable_head(f1, f1_channel_num//2, block_id_str='11')

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x1)

    x2 = DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1))(f2)
    x2 = Concatenate()([x2, x1_upsample])

    #feature map 2 head (26 x 26 x f2_channel_num//2 for 416 input)
    x2 = make_yolo_depthwise_separable_head(x2, f2_channel_num//2, block_id_str='12')

    #upsample fpn merge for feature map 2 & 3
    x2_upsample = compose(
            DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1)),
            UpSampling2D(2))(x2)

    x3 = DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1))(f3)
    x3 = Concatenate()([x3, x2_upsample])

    #feature map 3 head & output (52 x 52 x f3_channel_num for 416 input)
    #x3, y3 = make_depthwise_separable_last_layers(x3, f3_channel_num//2, num_anchors*(num_classes+5), block_id_str='13')
    x3 = make_yolo_depthwise_separable_head(x3, f3_channel_num//2, block_id_str='13')
    y3 = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(f3_channel_num, (3,3), block_id_str='13_3'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x3)

    #downsample fpn merge for feature map 3 & 2
    x3_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            Darknet_Depthwise_Separable_Conv2D_BN_Leaky(f2_channel_num//2, (3,3), strides=(2,2), block_id_str='13_4'))(x3)

    x2 = Concatenate()([x3_downsample, x2])

    #feature map 2 output (26 x 26 x f2_channel_num for 416 input)
    #x2, y2 = make_depthwise_separable_last_layers(x2, f2_channel_num//2, num_anchors*(num_classes+5), block_id_str='14')
    x2 = make_yolo_depthwise_separable_head(x2, f2_channel_num//2, block_id_str='14')
    y2 = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(f2_channel_num, (3,3), block_id_str='14_3'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            Darknet_Depthwise_Separable_Conv2D_BN_Leaky(f1_channel_num//2, (3,3), strides=(2,2), block_id_str='14_4'))(x2)

    x1 = Concatenate()([x2_downsample, x1])

    #feature map 1 output (13 x 13 x f1_channel_num for 416 input)
    #x1, y1 = make_depthwise_separable_last_layers(x1, f1_channel_num//2, num_anchors*(num_classes+5), block_id_str='15')
    x1 = make_yolo_depthwise_separable_head(x1, f1_channel_num//2, block_id_str='15')
    y1 = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(f1_channel_num, (3,3), block_id_str='15_3'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x1)

    return Model(inputs, [y1, y2, y3])


def tiny_yolo4_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0, use_spp=True):
    '''Create Tiny YOLO_v4 MobileNetV3Small model CNN body in keras.'''
    mobilenetv3small = MobileNetV3Small(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # activation_31(layer 165, final feature map): 13 x 13 x (576*alpha)
    # expanded_conv_10/Add(layer 162, end of block10): 13 x 13 x (96*alpha)

    # activation_22(layer 117, middle in block8) : 26 x 26 x (288*alpha)
    # expanded_conv_7/Add(layer 114, end of block7) : 26 x 26 x (48*alpha)

    # activation_7(layer 38, middle in block3) : 52 x 52 x (96*alpha)
    # expanded_conv_2/Add(layer 35, end of block2): 52 x 52 x (24*alpha)

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1 :13 x 13 x (576*alpha)
    f1 = mobilenetv3small.layers[165].output
    # f2: 26 x 26 x (288*alpha) for 416 input
    f2 = mobilenetv3small.layers[117].output

    f1_channel_num = int(576*alpha)
    f2_channel_num = int(288*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512

    #feature map 1 head (13 x 13 x f1_channel_num//2 for 416 input)
    x1 = DarknetConv2D_BN_Leaky(f1_channel_num//2, (1,1))(f1)
    if use_spp:
        x1 = Spp_Conv2D_BN_Leaky(x1, f1_channel_num//2)

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x1)
    x2 = compose(
            Concatenate(),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=f2_channel_num, kernel_size=(3, 3), block_id_str='11'),
            DarknetConv2D_BN_Leaky(f2_channel_num, (3,3)))([x1_upsample, f2])

    #feature map 2 output (26 x 26 x f2_channel_num for 416 input)
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            #Darknet_Depthwise_Separable_Conv2D_BN_Leaky(f1_channel_num//2, (3,3), strides=(2,2), block_id_str='12'),
            DarknetConv2D_BN_Leaky(f1_channel_num//2, (3,3), strides=(2,2)))(x2)
    x1 = compose(
            Concatenate(),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=f1_channel_num, kernel_size=(3, 3), block_id_str='13'),
            DarknetConv2D_BN_Leaky(f1_channel_num, (3,3)))([x2_downsample, x1])

    #feature map 1 output (13 x 13 x f1_channel_num for 416 input)
    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(x1)

    return Model(inputs, [y1,y2])


def tiny_yolo4lite_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0, use_spp=True):
    '''Create Tiny YOLO_v4 Lite MobileNetV3Small model CNN body in keras.'''
    mobilenetv3small = MobileNetV3Small(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # activation_31(layer 165, final feature map): 13 x 13 x (576*alpha)
    # expanded_conv_10/Add(layer 162, end of block10): 13 x 13 x (96*alpha)

    # activation_22(layer 117, middle in block8) : 26 x 26 x (288*alpha)
    # expanded_conv_7/Add(layer 114, end of block7) : 26 x 26 x (48*alpha)

    # activation_7(layer 38, middle in block3) : 52 x 52 x (96*alpha)
    # expanded_conv_2/Add(layer 35, end of block2): 52 x 52 x (24*alpha)

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1 :13 x 13 x (576*alpha)
    f1 = mobilenetv3small.layers[165].output
    # f2: 26 x 26 x (288*alpha) for 416 input
    f2 = mobilenetv3small.layers[117].output

    f1_channel_num = int(576*alpha)
    f2_channel_num = int(288*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512

    #feature map 1 head (13 x 13 x f1_channel_num//2 for 416 input)
    x1 = DarknetConv2D_BN_Leaky(f1_channel_num//2, (1,1))(f1)
    if use_spp:
        x1 = Spp_Conv2D_BN_Leaky(x1, f1_channel_num//2)

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x1)
    x2 = compose(
            Concatenate(),
            #DarknetConv2D_BN_Leaky(f2_channel_num, (3,3)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=f2_channel_num, kernel_size=(3, 3), block_id_str='11'))([x1_upsample, f2])

    #feature map 2 output (26 x 26 x f2_channel_num for 416 input)
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            #DarknetConv2D_BN_Leaky(f1_channel_num//2, (3,3), strides=(2,2)),
            Darknet_Depthwise_Separable_Conv2D_BN_Leaky(f1_channel_num//2, (3,3), strides=(2,2), block_id_str='12'))(x2)
    x1 = compose(
            Concatenate(),
            #DarknetConv2D_BN_Leaky(f1_channel_num, (3,3)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=f1_channel_num, kernel_size=(3, 3), block_id_str='13'))([x2_downsample, x1])

    #feature map 1 output (13 x 13 x f1_channel_num for 416 input)
    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(x1)

    return Model(inputs, [y1,y2])

