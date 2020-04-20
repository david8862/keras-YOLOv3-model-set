#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v3 MobileNetV3Large Model Defined in Keras."""

from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from common.backbones.mobilenet_v3 import MobileNetV3Large

from yolo3.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, make_last_layers, make_depthwise_separable_last_layers, make_spp_depthwise_separable_last_layers


def yolo3_mobilenetv3large_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V3 MobileNetV3Large model CNN body in Keras."""
    mobilenetv3large = MobileNetV3Large(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # activation_38(layer 194, final feature map): 13 x 13 x (960*alpha)
    # expanded_conv_14/Add(layer 191, end of block14): 13 x 13 x (160*alpha)

    # activation_29(layer 146, middle in block12) : 26 x 26 x (672*alpha)
    # expanded_conv_11/Add(layer 143, end of block11) : 26 x 26 x (112*alpha)

    # activation_15(layer 79, middle in block6) : 52 x 52 x (240*alpha)
    # expanded_conv_5/Add(layer 76, end of block5): 52 x 52 x (40*alpha)


    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    f1 = mobilenetv3large.layers[194].output
    # f1 :13 x 13 x (960*alpha)
    x, y1 = make_last_layers(f1, int(672*alpha), num_anchors * (num_classes + 5))
    #x, y1 = make_last_layers(f1, int(672*alpha), num_anchors * (num_classes + 5), predict_filters=int(1024*alpha))

    x = compose(
            DarknetConv2D_BN_Leaky(int(336*alpha), (1,1)),
            UpSampling2D(2))(x)

    f2 = mobilenetv3large.layers[146].output
    # f2: 26 x 26 x (672*alpha)
    x = Concatenate()([x,f2])

    x, y2 = make_last_layers(x, int(240*alpha), num_anchors*(num_classes+5))
    #x, y2 = make_last_layers(x, int(240*alpha), num_anchors*(num_classes+5), predict_filters=int(512*alpha))

    x = compose(
            DarknetConv2D_BN_Leaky(int(120*alpha), (1,1)),
            UpSampling2D(2))(x)

    f3 = mobilenetv3large.layers[79].output
    # f3 : 52 x 52 x (240*alpha)
    x = Concatenate()([x, f3])
    x, y3 = make_last_layers(x, int(120*alpha), num_anchors*(num_classes+5))
    #x, y3 = make_last_layers(x, int(120*alpha), num_anchors*(num_classes+5), predict_filters=int(256*alpha))

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yolo3lite_mobilenetv3large_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create YOLO_v3 Lite MobileNetV3Large model CNN body in keras.'''
    mobilenetv3large = MobileNetV3Large(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # activation_38(layer 194, final feature map): 13 x 13 x (960*alpha)
    # expanded_conv_14/Add(layer 191, end of block14): 13 x 13 x (160*alpha)

    # activation_29(layer 146, middle in block12) : 26 x 26 x (672*alpha)
    # expanded_conv_11/Add(layer 143, end of block11) : 26 x 26 x (112*alpha)

    # activation_15(layer 79, middle in block6) : 52 x 52 x (240*alpha)
    # expanded_conv_5/Add(layer 76, end of block5): 52 x 52 x (40*alpha)

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    f1 = mobilenetv3large.layers[194].output
    # f1 :13 x 13 x (960*alpha)
    x, y1 = make_depthwise_separable_last_layers(f1, int(672*alpha), num_anchors * (num_classes + 5), block_id_str='15')
    #x, y1 = make_depthwise_separable_last_layers(f1, int(672*alpha), num_anchors * (num_classes + 5), block_id_str='15', predict_filters=int(1024*alpha))

    x = compose(
            DarknetConv2D_BN_Leaky(int(336*alpha), (1,1)),
            UpSampling2D(2))(x)

    f2 = mobilenetv3large.layers[146].output
    # f2: 26 x 26 x (672*alpha)
    x = Concatenate()([x,f2])

    x, y2 = make_depthwise_separable_last_layers(x, int(240*alpha), num_anchors*(num_classes+5), block_id_str='16')
    #x, y2 = make_depthwise_separable_last_layers(x, int(240*alpha), num_anchors*(num_classes+5), block_id_str='16', predict_filters=int(512*alpha))

    x = compose(
            DarknetConv2D_BN_Leaky(int(120*alpha), (1,1)),
            UpSampling2D(2))(x)

    f3 = mobilenetv3large.layers[79].output
    # f3 : 52 x 52 x (240*alpha)
    x = Concatenate()([x, f3])
    x, y3 = make_depthwise_separable_last_layers(x, int(120*alpha), num_anchors*(num_classes+5), block_id_str='17')
    #x, y3 = make_depthwise_separable_last_layers(x, int(120*alpha), num_anchors*(num_classes+5), block_id_str='17', predict_filters=int(256*alpha))

    return Model(inputs = inputs, outputs=[y1,y2,y3])



def tiny_yolo3_mobilenetv3large_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create Tiny YOLO_v3 MobileNetV3Large model CNN body in keras.'''
    mobilenetv3large = MobileNetV3Large(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # activation_38(layer 194, final feature map): 13 x 13 x (960*alpha)
    # expanded_conv_14/Add(layer 191, end of block14): 13 x 13 x (160*alpha)

    # activation_29(layer 146, middle in block12) : 26 x 26 x (672*alpha)
    # expanded_conv_11/Add(layer 143, end of block11) : 26 x 26 x (112*alpha)

    # activation_15(layer 79, middle in block6) : 52 x 52 x (240*alpha)
    # expanded_conv_5/Add(layer 76, end of block5): 52 x 52 x (40*alpha)

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    x1 = mobilenetv3large.layers[146].output

    x2 = mobilenetv3large.layers[194].output
    x2 = DarknetConv2D_BN_Leaky(int(672*alpha), (1,1))(x2)

    y1 = compose(
            DarknetConv2D_BN_Leaky(int(960*alpha), (3,3)),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=int(960*alpha), kernel_size=(3, 3), block_id_str='15'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(int(336*alpha), (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(int(672*alpha), (3,3)),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=int(672*alpha), kernel_size=(3, 3), block_id_str='16'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])


def tiny_yolo3lite_mobilenetv3large_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create Tiny YOLO_v3 Lite MobileNetV3Large model CNN body in keras.'''
    mobilenetv3large = MobileNetV3Large(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # activation_38(layer 194, final feature map): 13 x 13 x (960*alpha)
    # expanded_conv_14/Add(layer 191, end of block14): 13 x 13 x (160*alpha)

    # activation_29(layer 146, middle in block12) : 26 x 26 x (672*alpha)
    # expanded_conv_11/Add(layer 143, end of block11) : 26 x 26 x (112*alpha)

    # activation_15(layer 79, middle in block6) : 52 x 52 x (240*alpha)
    # expanded_conv_5/Add(layer 76, end of block5): 52 x 52 x (40*alpha)

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    x1 = mobilenetv3large.layers[146].output

    x2 = mobilenetv3large.layers[194].output
    x2 = DarknetConv2D_BN_Leaky(int(672*alpha), (1,1))(x2)

    y1 = compose(
            #DarknetConv2D_BN_Leaky(int(960*alpha), (3,3)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=int(960*alpha), kernel_size=(3, 3), block_id_str='15'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(int(336*alpha), (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            #DarknetConv2D_BN_Leaky(int(672*alpha), (3,3)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=int(672*alpha), kernel_size=(3, 3), block_id_str='16'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])

