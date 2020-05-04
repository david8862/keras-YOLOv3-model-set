#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v4 MobileNetV3Large Model Defined in Keras."""

from tensorflow.keras.layers import ZeroPadding2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from common.backbones.mobilenet_v3 import MobileNetV3Large

from yolo4.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Spp_Conv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, Darknet_Depthwise_Separable_Conv2D_BN_Leaky, make_yolo_head, make_yolo_spp_head, make_yolo_depthwise_separable_head, make_yolo_spp_depthwise_separable_head


def yolo4_mobilenetv3large_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V4 MobileNetV3Large model CNN body in Keras."""
    mobilenetv3large = MobileNetV3Large(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # activation_38(layer 194, final feature map): 13 x 13 x (960*alpha)
    # expanded_conv_14/Add(layer 191, end of block14): 13 x 13 x (160*alpha)

    # activation_29(layer 146, middle in block12) : 26 x 26 x (672*alpha)
    # expanded_conv_11/Add(layer 143, end of block11) : 26 x 26 x (112*alpha)

    # activation_15(layer 79, middle in block6) : 52 x 52 x (240*alpha)
    # expanded_conv_5/Add(layer 76, end of block5): 52 x 52 x (40*alpha)


    # f1 :13 x 13 x (960*alpha)
    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    f1 = mobilenetv3large.layers[194].output
    #feature map 1 head (13 x 13 x (480*alpha) for 416 input)
    x1 = make_yolo_spp_head(f1, int(480*alpha))

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Leaky(int(336*alpha), (1,1)),
            UpSampling2D(2))(x1)

    f2 = mobilenetv3large.layers[146].output
    # f2: 26 x 26 x (672*alpha) for 416 input
    x2 = DarknetConv2D_BN_Leaky(int(336*alpha), (1,1))(f2)
    x2 = Concatenate()([x2, x1_upsample])

    #feature map 2 head (26 x 26 x (336*alpha) for 416 input)
    x2 = make_yolo_head(x2, int(336*alpha))

    #upsample fpn merge for feature map 2 & 3
    x2_upsample = compose(
            DarknetConv2D_BN_Leaky(int(120*alpha), (1,1)),
            UpSampling2D(2))(x2)

    f3 = mobilenetv3large.layers[79].output
    # f3 : 52 x 52 x (240*alpha) for 416 input
    x3 = DarknetConv2D_BN_Leaky(int(120*alpha), (1,1))(f3)
    x3 = Concatenate()([x3, x2_upsample])

    #feature map 3 head & output (52 x 52 x (240*alpha) for 416 input)
    #x3, y3 = make_last_layers(x3, int(120*alpha), num_anchors*(num_classes+5))
    x3 = make_yolo_head(x3, int(120*alpha))
    y3 = compose(
            DarknetConv2D_BN_Leaky(int(240*alpha), (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x3)

    #downsample fpn merge for feature map 3 & 2
    x3_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            DarknetConv2D_BN_Leaky(int(336*alpha), (3,3), strides=(2,2)))(x3)

    x2 = Concatenate()([x3_downsample, x2])

    #feature map 2 output (26 x 26 x (672*alpha) for 416 input)
    #x2, y2 = make_last_layers(x2, int(336*alpha), num_anchors*(num_classes+5))
    x2 = make_yolo_head(x2, int(336*alpha))
    y2 = compose(
            DarknetConv2D_BN_Leaky(int(672*alpha), (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            DarknetConv2D_BN_Leaky(int(480*alpha), (3,3), strides=(2,2)))(x2)

    x1 = Concatenate()([x2_downsample, x1])

    #feature map 1 output (13 x 13 x (960*alpha) for 416 input)
    #x1, y1 = make_last_layers(x1, int(480*alpha), num_anchors*(num_classes+5))
    x1 = make_yolo_head(x1, int(480*alpha))
    y1 = compose(
            DarknetConv2D_BN_Leaky(int(960*alpha), (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x1)

    return Model(inputs, [y1, y2, y3])



def yolo4lite_mobilenetv3large_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create YOLO_v4 Lite MobileNetV3Large model CNN body in keras.'''
    mobilenetv3large = MobileNetV3Large(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # activation_38(layer 194, final feature map): 13 x 13 x (960*alpha)
    # expanded_conv_14/Add(layer 191, end of block14): 13 x 13 x (160*alpha)

    # activation_29(layer 146, middle in block12) : 26 x 26 x (672*alpha)
    # expanded_conv_11/Add(layer 143, end of block11) : 26 x 26 x (112*alpha)

    # activation_15(layer 79, middle in block6) : 52 x 52 x (240*alpha)
    # expanded_conv_5/Add(layer 76, end of block5): 52 x 52 x (40*alpha)

    # f1 :13 x 13 x (960*alpha)
    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    f1 = mobilenetv3large.layers[194].output
    #feature map 1 head (13 x 13 x (480*alpha) for 416 input)
    x1 = make_yolo_spp_depthwise_separable_head(f1, int(480*alpha), block_id_str='15')

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Leaky(int(336*alpha), (1,1)),
            UpSampling2D(2))(x1)

    f2 = mobilenetv3large.layers[146].output
    # f2: 26 x 26 x (672*alpha) for 416 input
    x2 = DarknetConv2D_BN_Leaky(int(336*alpha), (1,1))(f2)
    x2 = Concatenate()([x2, x1_upsample])

    #feature map 2 head (26 x 26 x (336*alpha) for 416 input)
    x2 = make_yolo_depthwise_separable_head(x2, int(336*alpha), block_id_str='16')

    #upsample fpn merge for feature map 2 & 3
    x2_upsample = compose(
            DarknetConv2D_BN_Leaky(int(120*alpha), (1,1)),
            UpSampling2D(2))(x2)

    f3 = mobilenetv3large.layers[79].output
    # f3 : 52 x 52 x (240*alpha) for 416 input
    x3 = DarknetConv2D_BN_Leaky(int(120*alpha), (1,1))(f3)
    x3 = Concatenate()([x3, x2_upsample])

    #feature map 3 head & output (52 x 52 x (240*alpha) for 416 input)
    #x3, y3 = make_depthwise_separable_last_layers(x3, int(120*alpha), num_anchors*(num_classes+5), block_id_str='17')
    x3 = make_yolo_depthwise_separable_head(x3, int(120*alpha), block_id_str='17')
    y3 = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(int(240*alpha), (3,3), block_id_str='17_3'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x3)

    #downsample fpn merge for feature map 3 & 2
    x3_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            Darknet_Depthwise_Separable_Conv2D_BN_Leaky(int(336*alpha), (3,3), strides=(2,2), block_id_str='17_4'))(x3)

    x2 = Concatenate()([x3_downsample, x2])

    #feature map 2 output (26 x 26 x (672*alpha) for 416 input)
    #x2, y2 = make_depthwise_separable_last_layers(x2, int(336*alpha), num_anchors*(num_classes+5), block_id_str='18')
    x2 = make_yolo_depthwise_separable_head(x2, int(336*alpha), block_id_str='18')
    y2 = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(int(672*alpha), (3,3), block_id_str='18_3'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            Darknet_Depthwise_Separable_Conv2D_BN_Leaky(int(480*alpha), (3,3), strides=(2,2), block_id_str='18_4'))(x2)

    x1 = Concatenate()([x2_downsample, x1])

    #feature map 1 output (13 x 13 x (960*alpha) for 416 input)
    #x1, y1 = make_depthwise_separable_last_layers(x1, int(480*alpha), num_anchors*(num_classes+5))
    x1 = make_yolo_depthwise_separable_head(x1, int(480*alpha), block_id_str='19')
    y1 = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(int(960*alpha), (3,3), block_id_str='19_3'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x1)

    return Model(inputs, [y1, y2, y3])


def tiny_yolo4_mobilenetv3large_body(inputs, num_anchors, num_classes, alpha=1.0, use_spp=True):
    '''Create Tiny YOLO_v4 MobileNetV3Large model CNN body in keras.'''
    mobilenetv3large = MobileNetV3Large(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # activation_38(layer 194, final feature map): 13 x 13 x (960*alpha)
    # expanded_conv_14/Add(layer 191, end of block14): 13 x 13 x (160*alpha)

    # activation_29(layer 146, middle in block12) : 26 x 26 x (672*alpha)
    # expanded_conv_11/Add(layer 143, end of block11) : 26 x 26 x (112*alpha)

    # activation_15(layer 79, middle in block6) : 52 x 52 x (240*alpha)
    # expanded_conv_5/Add(layer 76, end of block5): 52 x 52 x (40*alpha)

    # f1 :13 x 13 x (960*alpha)
    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    f1 = mobilenetv3large.layers[194].output
    # f2: 26 x 26 x (672*alpha) for 416 input
    f2 = mobilenetv3large.layers[146].output

    #feature map 1 head (13 x 13 x (480*alpha) for 416 input)
    x1 = DarknetConv2D_BN_Leaky(int(480*alpha), (1,1))(f1)
    if use_spp:
        x1 = Spp_Conv2D_BN_Leaky(x1, int(480*alpha))

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Leaky(int(336*alpha), (1,1)),
            UpSampling2D(2))(x1)
    x2 = compose(
            Concatenate(),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=int(672*alpha), kernel_size=(3, 3), block_id_str='15'),
            DarknetConv2D_BN_Leaky(int(672*alpha), (3,3)))([x1_upsample, f2])

    #feature map 2 output (26 x 26 x (672*alpha) for 416 input)
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            #Darknet_Depthwise_Separable_Conv2D_BN_Leaky(int(480*alpha), (3,3), strides=(2,2), block_id_str='16'),
            DarknetConv2D_BN_Leaky(int(480*alpha), (3,3), strides=(2,2)))(x2)
    x1 = compose(
            Concatenate(),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=int(960*alpha), kernel_size=(3, 3), block_id_str='17'),
            DarknetConv2D_BN_Leaky(int(960*alpha), (3,3)))([x2_downsample, x1])

    #feature map 1 output (13 x 13 x (960*alpha) for 416 input)
    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(x1)

    return Model(inputs, [y1,y2])


def tiny_yolo4lite_mobilenetv3large_body(inputs, num_anchors, num_classes, alpha=1.0, use_spp=True):
    '''Create Tiny YOLO_v4 Lite MobileNetV3Large model CNN body in keras.'''
    mobilenetv3large = MobileNetV3Large(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # activation_38(layer 194, final feature map): 13 x 13 x (960*alpha)
    # expanded_conv_14/Add(layer 191, end of block14): 13 x 13 x (160*alpha)

    # activation_29(layer 146, middle in block12) : 26 x 26 x (672*alpha)
    # expanded_conv_11/Add(layer 143, end of block11) : 26 x 26 x (112*alpha)

    # activation_15(layer 79, middle in block6) : 52 x 52 x (240*alpha)
    # expanded_conv_5/Add(layer 76, end of block5): 52 x 52 x (40*alpha)

    # f1 :13 x 13 x (960*alpha)
    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    f1 = mobilenetv3large.layers[194].output
    # f2: 26 x 26 x (672*alpha) for 416 input
    f2 = mobilenetv3large.layers[146].output

    #feature map 1 head (13 x 13 x (480*alpha) for 416 input)
    x1 = DarknetConv2D_BN_Leaky(int(480*alpha), (1,1))(f1)
    if use_spp:
        x1 = Spp_Conv2D_BN_Leaky(x1, int(480*alpha))

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Leaky(int(336*alpha), (1,1)),
            UpSampling2D(2))(x1)
    x2 = compose(
            Concatenate(),
            #DarknetConv2D_BN_Leaky(int(672*alpha), (3,3)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=int(672*alpha), kernel_size=(3, 3), block_id_str='15'))([x1_upsample, f2])

    #feature map 2 output (26 x 26 x (672*alpha) for 416 input)
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            #DarknetConv2D_BN_Leaky(int(480*alpha), (3,3), strides=(2,2)),
            Darknet_Depthwise_Separable_Conv2D_BN_Leaky(int(480*alpha), (3,3), strides=(2,2), block_id_str='16'))(x2)
    x1 = compose(
            Concatenate(),
            #DarknetConv2D_BN_Leaky(int(960*alpha), (3,3)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=int(960*alpha), kernel_size=(3, 3), block_id_str='17'))([x2_downsample, x1])

    #feature map 1 output (13 x 13 x (960*alpha) for 416 input)
    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(x1)

    return Model(inputs, [y1,y2])

