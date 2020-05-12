#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v3 MobileNetV3Small Model Defined in Keras."""

from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from common.backbones.mobilenet_v3 import MobileNetV3Small

from yolo3.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, make_last_layers, make_depthwise_separable_last_layers, make_spp_depthwise_separable_last_layers


def yolo3_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V3 MobileNetV3Small model CNN body in Keras."""
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
    # f2: 26 x 26 x (288*alpha)
    f2 = mobilenetv3small.layers[117].output
    # f3 : 52 x 52 x (96*alpha)
    f3 = mobilenetv3small.layers[38].output

    f1_channel_num = int(576*alpha)
    f2_channel_num = int(288*alpha)
    f3_channel_num = int(96*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512
    #f3_channel_num = 256

    #feature map 1 head & output (13x13 for 416 input)
    x, y1 = make_last_layers(f1, f1_channel_num//2, num_anchors * (num_classes + 5))
    #x, y1 = make_last_layers(f1, f1_channel_num//2, num_anchors * (num_classes + 5), predict_filters=int(1024*alpha))

    #upsample fpn merge for feature map 1 & 2
    x = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,f2])

    #feature map 2 head & output (26x26 for 416 input)
    x, y2 = make_last_layers(x, f2_channel_num//2, num_anchors*(num_classes+5))
    #x, y2 = make_last_layers(x, f2_channel_num//2, num_anchors*(num_classes+5), predict_filters=int(512*alpha))

    #upsample fpn merge for feature map 2 & 3
    x = compose(
            DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, f3])

    #feature map 3 head & output (52x52 for 416 input)
    x, y3 = make_last_layers(x, f3_channel_num//2, num_anchors*(num_classes+5))
    #x, y3 = make_last_layers(x, f3_channel_num//2, num_anchors*(num_classes+5), predict_filters=int(256*alpha))

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yolo3lite_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create YOLO_v3 Lite MobileNetV3Small model CNN body in keras.'''
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
    # f2: 26 x 26 x (288*alpha)
    f2 = mobilenetv3small.layers[117].output
    # f3 : 52 x 52 x (96*alpha)
    f3 = mobilenetv3small.layers[38].output

    f1_channel_num = int(576*alpha)
    f2_channel_num = int(288*alpha)
    f3_channel_num = int(96*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512
    #f3_channel_num = 256

    #feature map 1 head & output (13x13 for 416 input)
    x, y1 = make_depthwise_separable_last_layers(f1, f1_channel_num//2, num_anchors * (num_classes + 5))
    #x, y1 = make_depthwise_separable_last_layers(f1, f1_channel_num//2, num_anchors * (num_classes + 5), predict_filters=int(1024*alpha))

    #upsample fpn merge for feature map 1 & 2
    x = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,f2])

    #feature map 2 head & output (26x26 for 416 input)
    x, y2 = make_depthwise_separable_last_layers(x, f2_channel_num//2, num_anchors*(num_classes+5))
    #x, y2 = make_depthwise_separable_last_layers(x, f2_channel_num//2, num_anchors*(num_classes+5), predict_filters=int(512*alpha))

    #upsample fpn merge for feature map 2 & 3
    x = compose(
            DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, f3])

    #feature map 3 head & output (52x52 for 416 input)
    x, y3 = make_depthwise_separable_last_layers(x, f3_channel_num//2, num_anchors*(num_classes+5))
    #x, y3 = make_depthwise_separable_last_layers(x, f3_channel_num//2, num_anchors*(num_classes+5), predict_filters=int(256*alpha))

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def tiny_yolo3_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create Tiny YOLO_v3 MobileNetV3Small model CNN body in keras.'''
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
    # f2: 26 x 26 x (288*alpha)
    f2 = mobilenetv3small.layers[117].output

    f1_channel_num = int(576*alpha)
    f2_channel_num = int(288*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512

    #feature map 1 transform
    x1 = DarknetConv2D_BN_Leaky(f1_channel_num//2, (1,1))(f1)

    #feature map 1 output (13x13 for 416 input)
    y1 = compose(
            DarknetConv2D_BN_Leaky(f1_channel_num, (3,3)),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=f1_channel_num, kernel_size=(3, 3), block_id_str='15'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x1)

    #upsample fpn merge for feature map 1 & 2
    x2 = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x1)

    #feature map 2 output (26x26 for 416 input)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(f2_channel_num, (3,3)),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=f2_channel_num, kernel_size=(3, 3), block_id_str='16'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2, f2])

    return Model(inputs, [y1,y2])


def tiny_yolo3lite_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create Tiny YOLO_v3 Lite MobileNetV3Small model CNN body in keras.'''
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
    # f2: 26 x 26 x (288*alpha)
    f2 = mobilenetv3small.layers[117].output

    f1_channel_num = int(576*alpha)
    f2_channel_num = int(288*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512

    #feature map 1 transform
    x1 = DarknetConv2D_BN_Leaky(f1_channel_num//2, (1,1))(f1)

    #feature map 1 output (13x13 for 416 input)
    y1 = compose(
            #DarknetConv2D_BN_Leaky(f1_channel_num, (3,3)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=f1_channel_num, kernel_size=(3, 3), block_id_str='15'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x1)

    #upsample fpn merge for feature map 1 & 2
    x2 = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x1)

    #feature map 2 output (26x26 for 416 input)
    y2 = compose(
            Concatenate(),
            #DarknetConv2D_BN_Leaky(f2_channel_num, (3,3)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=f2_channel_num, kernel_size=(3, 3), block_id_str='16'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2, f2])

    return Model(inputs, [y1,y2])

