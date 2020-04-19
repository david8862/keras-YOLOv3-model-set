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
    # activation_31(final feature map): 13 x 13 x (576*alpha)
    # expanded_conv_10/Add(end of block10): 13 x 13 x (96*alpha)

    # activation_22(middle in block8) : 26 x 26 x (288*alpha)
    # expanded_conv_7/Add(end of block7) : 26 x 26 x (48*alpha)

    # activation_7(middle in block3) : 52 x 52 x (96*alpha)
    # expanded_conv_2/Add(end of block2): 52 x 52 x (24*alpha)

    f1 = mobilenetv3small.get_layer('activation_31').output
    # f1 :13 x 13 x (576*alpha)
    x, y1 = make_last_layers(f1, int(288*alpha), num_anchors * (num_classes + 5))
    #x, y1 = make_last_layers(f1, int(288*alpha), num_anchors * (num_classes + 5), predict_filters=int(1024*alpha))

    x = compose(
            DarknetConv2D_BN_Leaky(int(144*alpha), (1,1)),
            UpSampling2D(2))(x)

    f2 = mobilenetv3small.get_layer('activation_22').output
    # f2: 26 x 26 x (288*alpha)
    x = Concatenate()([x,f2])

    x, y2 = make_last_layers(x, int(96*alpha), num_anchors*(num_classes+5))
    #x, y2 = make_last_layers(x, int(96*alpha), num_anchors*(num_classes+5), predict_filters=int(512*alpha))

    x = compose(
            DarknetConv2D_BN_Leaky(int(48*alpha), (1,1)),
            UpSampling2D(2))(x)

    f3 = mobilenetv3small.get_layer('activation_7').output
    # f3 : 52 x 52 x (96*alpha)
    x = Concatenate()([x, f3])
    x, y3 = make_last_layers(x, int(48*alpha), num_anchors*(num_classes+5))
    #x, y3 = make_last_layers(x, int(48*alpha), num_anchors*(num_classes+5), predict_filters=int(256*alpha))

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yolo3lite_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create YOLO_v3 Lite MobileNetV3Small model CNN body in keras.'''
    mobilenetv3small = MobileNetV3Small(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # activation_31(final feature map): 13 x 13 x (576*alpha)
    # expanded_conv_10/Add(end of block10): 13 x 13 x (96*alpha)

    # activation_22(middle in block8) : 26 x 26 x (288*alpha)
    # expanded_conv_7/Add(end of block7) : 26 x 26 x (48*alpha)

    # activation_7(middle in block3) : 52 x 52 x (96*alpha)
    # expanded_conv_2/Add(end of block2): 52 x 52 x (24*alpha)

    f1 = mobilenetv3small.get_layer('activation_31').output
    # f1 :13 x 13 x (576*alpha)
    x, y1 = make_depthwise_separable_last_layers(f1, int(288*alpha), num_anchors * (num_classes + 5))
    #x, y1 = make_depthwise_separable_last_layers(f1, int(288*alpha), num_anchors * (num_classes + 5), predict_filters=int(1024*alpha))

    x = compose(
            DarknetConv2D_BN_Leaky(int(144*alpha), (1,1)),
            UpSampling2D(2))(x)

    f2 = mobilenetv3small.get_layer('activation_22').output
    # f2: 26 x 26 x (288*alpha)
    x = Concatenate()([x,f2])

    x, y2 = make_depthwise_separable_last_layers(x, int(96*alpha), num_anchors*(num_classes+5))
    #x, y2 = make_depthwise_separable_last_layers(x, int(96*alpha), num_anchors*(num_classes+5), predict_filters=int(512*alpha))

    x = compose(
            DarknetConv2D_BN_Leaky(int(48*alpha), (1,1)),
            UpSampling2D(2))(x)

    f3 = mobilenetv3small.get_layer('activation_7').output
    # f3 : 52 x 52 x (96*alpha)
    x = Concatenate()([x, f3])
    x, y3 = make_depthwise_separable_last_layers(x, int(48*alpha), num_anchors*(num_classes+5))
    #x, y3 = make_depthwise_separable_last_layers(x, int(48*alpha), num_anchors*(num_classes+5), predict_filters=int(256*alpha))

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def tiny_yolo3_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create Tiny YOLO_v3 MobileNetV3Small model CNN body in keras.'''
    mobilenetv3small = MobileNetV3Small(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # activation_31(final feature map): 13 x 13 x (576*alpha)
    # expanded_conv_10/Add(end of block10): 13 x 13 x (96*alpha)

    # activation_22(middle in block8) : 26 x 26 x (288*alpha)
    # expanded_conv_7/Add(end of block7) : 26 x 26 x (48*alpha)

    # activation_7(middle in block3) : 52 x 52 x (96*alpha)
    # expanded_conv_2/Add(end of block2): 52 x 52 x (24*alpha)

    x1 = mobilenetv3small.get_layer('activation_22').output

    x2 = mobilenetv3small.get_layer('activation_31').output
    x2 = DarknetConv2D_BN_Leaky(int(288*alpha), (1,1))(x2)

    y1 = compose(
            DarknetConv2D_BN_Leaky(int(576*alpha), (3,3)),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=int(576*alpha), kernel_size=(3, 3), block_id_str='15'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(int(144*alpha), (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(int(288*alpha), (3,3)),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=int(288*alpha), kernel_size=(3, 3), block_id_str='16'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])


def tiny_yolo3lite_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create Tiny YOLO_v3 Lite MobileNetV3Small model CNN body in keras.'''
    mobilenetv3small = MobileNetV3Small(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # activation_31(final feature map): 13 x 13 x (576*alpha)
    # expanded_conv_10/Add(end of block10): 13 x 13 x (96*alpha)

    # activation_22(middle in block8) : 26 x 26 x (288*alpha)
    # expanded_conv_7/Add(end of block7) : 26 x 26 x (48*alpha)

    # activation_7(middle in block3) : 52 x 52 x (96*alpha)
    # expanded_conv_2/Add(end of block2): 52 x 52 x (24*alpha)

    x1 = mobilenetv3small.get_layer('activation_22').output

    x2 = mobilenetv3small.get_layer('activation_31').output
    x2 = DarknetConv2D_BN_Leaky(int(288*alpha), (1,1))(x2)

    y1 = compose(
            #DarknetConv2D_BN_Leaky(int(576*alpha), (3,3)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=int(576*alpha), kernel_size=(3, 3), block_id_str='15'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(int(144*alpha), (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            #DarknetConv2D_BN_Leaky(int(288*alpha), (3,3)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=int(288*alpha), kernel_size=(3, 3), block_id_str='16'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])

