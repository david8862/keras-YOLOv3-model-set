#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v3 EfficientNet Model Defined in Keras."""

from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from common.backbones.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7

from yolo3.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, make_last_layers, make_depthwise_separable_last_layers, make_spp_depthwise_separable_last_layers


def yolo3_efficientnetb0_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 EfficientNetB0 model CNN body in Keras."""
    efficientnetb0 = EfficientNetB0(input_tensor=inputs, weights='imagenet', include_top=False)

    # input: 416 x 416 x 3
    # top_activation: 13 x 13 x 1280
    # block6a_expand_activation(middle in block6a): 26 x 26 x 672
    # block5c_add(end of block5c): 26 x 26 x 112
    # block4a_expand_activation(middle in block4a): 52 x 52 x 240
    # block3b_add(end of block3b): 52 x 52 x 40

    f1 = efficientnetb0.get_layer('top_activation').output
    # f1 :13 x 13 x 1280
    x, y1 = make_last_layers(f1, 672, num_anchors * (num_classes + 5))

    x = compose(
            DarknetConv2D_BN_Leaky(336, (1,1)),
            UpSampling2D(2))(x)

    f2 = efficientnetb0.get_layer('block6a_expand_activation').output
    # f2: 26 x 26 x 672
    x = Concatenate()([x,f2])

    x, y2 = make_last_layers(x, 240, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(120, (1,1)),
            UpSampling2D(2))(x)

    f3 = efficientnetb0.get_layer('block4a_expand_activation').output
    # f3 : 52 x 52 x 240
    x = Concatenate()([x, f3])
    x, y3 = make_last_layers(x, 120, num_anchors*(num_classes+5))

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yolo3lite_efficientnetb0_body(inputs, num_anchors, num_classes):
    '''Create YOLO_v3 Lite EfficientNetB0 model CNN body in keras.'''
    efficientnetb0 = EfficientNetB0(input_tensor=inputs, weights='imagenet', include_top=False)

    # input: 416 x 416 x 3
    # top_activation: 13 x 13 x 1280
    # block6a_expand_activation(middle in block6a): 26 x 26 x 672
    # block5c_add(end of block5c): 26 x 26 x 112
    # block4a_expand_activation(middle in block4a): 52 x 52 x 240
    # block3b_add(end of block3b): 52 x 52 x 40

    f1 = efficientnetb0.get_layer('top_activation').output
    # f1 :13 x 13 x 1280
    x, y1 = make_depthwise_separable_last_layers(f1, 672, num_anchors * (num_classes + 5), block_id_str='8')

    x = compose(
            DarknetConv2D_BN_Leaky(336, (1,1)),
            UpSampling2D(2))(x)

    f2 = efficientnetb0.get_layer('block6a_expand_activation').output
    # f2: 26 x 26 x 672
    x = Concatenate()([x,f2])

    x, y2 = make_depthwise_separable_last_layers(x, 240, num_anchors*(num_classes+5), block_id_str='9')

    x = compose(
            DarknetConv2D_BN_Leaky(120, (1,1)),
            UpSampling2D(2))(x)

    f3 = efficientnetb0.get_layer('block4a_expand_activation').output
    # f3 : 52 x 52 x 240
    x = Concatenate()([x, f3])
    x, y3 = make_depthwise_separable_last_layers(x, 120, num_anchors*(num_classes+5), block_id_str='10')

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yolo3lite_spp_efficientnetb0_body(inputs, num_anchors, num_classes):
    '''Create YOLO_v3 Lite SPP EfficientNetB0 model CNN body in keras.'''
    efficientnetb0 = EfficientNetB0(input_tensor=inputs, weights='imagenet', include_top=False)

    # input: 416 x 416 x 3
    # top_activation: 13 x 13 x 1280
    # block6a_expand_activation(middle in block6a): 26 x 26 x 672
    # block5c_add(end of block5c): 26 x 26 x 112
    # block4a_expand_activation(middle in block4a): 52 x 52 x 240
    # block3b_add(end of block3b): 52 x 52 x 40

    f1 = efficientnetb0.get_layer('top_activation').output
    # f1 :13 x 13 x 1280
    #x, y1 = make_depthwise_separable_last_layers(f1, 672, num_anchors * (num_classes + 5), block_id_str='8')
    x, y1 = make_spp_depthwise_separable_last_layers(f1, 672, num_anchors * (num_classes + 5), block_id_str='8')

    x = compose(
            DarknetConv2D_BN_Leaky(336, (1,1)),
            UpSampling2D(2))(x)

    f2 = efficientnetb0.get_layer('block6a_expand_activation').output
    # f2: 26 x 26 x 672
    x = Concatenate()([x,f2])

    x, y2 = make_depthwise_separable_last_layers(x, 240, num_anchors*(num_classes+5), block_id_str='9')

    x = compose(
            DarknetConv2D_BN_Leaky(120, (1,1)),
            UpSampling2D(2))(x)

    f3 = efficientnetb0.get_layer('block4a_expand_activation').output
    # f3 : 52 x 52 x 240
    x = Concatenate()([x, f3])
    x, y3 = make_depthwise_separable_last_layers(x, 120, num_anchors*(num_classes+5), block_id_str='10')

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def tiny_yolo3_efficientnetb0_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 EfficientNetB0 model CNN body in keras.'''
    efficientnetb0 = EfficientNetB0(input_tensor=inputs, weights='imagenet', include_top=False)

    # input: 416 x 416 x 3
    # top_activation: 13 x 13 x 1280
    # block6a_expand_activation(middle in block6a): 26 x 26 x 672
    # block5c_add(end of block5c): 26 x 26 x 112
    # block4a_expand_activation(middle in block4a): 52 x 52 x 240
    # block3b_add(end of block3b): 52 x 52 x 40

    x1 = efficientnetb0.get_layer('block6a_expand_activation').output

    x2 = efficientnetb0.get_layer('top_activation').output
    x2 = DarknetConv2D_BN_Leaky(672, (1,1))(x2)

    y1 = compose(
            DarknetConv2D_BN_Leaky(1280, (3,3)),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=1280, kernel_size=(3, 3), block_id_str='17'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(336, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(672, (3,3)),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=672, kernel_size=(3, 3), block_id_str='18'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])


def tiny_yolo3lite_efficientnetb0_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 Lite EfficientNetB0 model CNN body in keras.'''
    efficientnetb0 = EfficientNetB0(input_tensor=inputs, weights='imagenet', include_top=False)

    # input: 416 x 416 x 3
    # top_activation: 13 x 13 x 1280
    # block6a_expand_activation(middle in block6a): 26 x 26 x 672
    # block5c_add(end of block5c): 26 x 26 x 112
    # block4a_expand_activation(middle in block4a): 52 x 52 x 240
    # block3b_add(end of block3b): 52 x 52 x 40

    x1 = efficientnetb0.get_layer('block6a_expand_activation').output

    x2 = efficientnetb0.get_layer('top_activation').output
    x2 = DarknetConv2D_BN_Leaky(672, (1,1))(x2)

    y1 = compose(
            #DarknetConv2D_BN_Leaky(1280, (3,3)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=1280, kernel_size=(3, 3), block_id_str='17'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(336, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            #DarknetConv2D_BN_Leaky(672, (3,3)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=672, kernel_size=(3, 3), block_id_str='18'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])

