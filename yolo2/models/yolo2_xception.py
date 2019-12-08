#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v2 Xception Model Defined in Keras."""

from tensorflow.keras.layers import MaxPooling2D, Lambda, Concatenate, GlobalAveragePooling2D, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import Xception

from yolo2.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, bottleneck_block, bottleneck_x2_block, space_to_depth_x2, space_to_depth_x2_output_shape


def yolo2_xception_body(inputs, num_anchors, num_classes):
    """Create YOLO_V2 Xception model CNN body in Keras."""
    xception = Xception(input_tensor=inputs, weights='imagenet', include_top=False)

    # input: 416 x 416 x 3
    # xception.output: 13 x 13 x 2048
    # block13_sepconv2_bn(middle in block13, layers[121]): 26 x 26 x 1024
    # add_46(end of block12, layers[115]): 26 x 26 x 728

    conv_head1 = compose(
        DarknetConv2D_BN_Leaky(2048, (3, 3)),
        DarknetConv2D_BN_Leaky(2048, (3, 3)))(xception.output)

    # block13_sepconv2_bn output shape: 26 x 26 x 1024
    block13_sepconv2_bn = xception.layers[121].output
    conv_head2 = DarknetConv2D_BN_Leaky(128, (1, 1))(block13_sepconv2_bn)
    # TODO: Allow Keras Lambda to use func arguments for output_shape?
    conv_head2_reshaped = Lambda(
        space_to_depth_x2,
        output_shape=space_to_depth_x2_output_shape,
        name='space_to_depth')(conv_head2)

    x = Concatenate()([conv_head2_reshaped, conv_head1])
    x = DarknetConv2D_BN_Leaky(2048, (3, 3))(x)
    x = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name='predict_conv')(x)
    return Model(inputs, x)


def yolo2lite_xception_body(inputs, num_anchors, num_classes):
    """Create YOLO_V2 Lite Xception model CNN body in Keras."""
    xception = Xception(input_tensor=inputs, weights='imagenet', include_top=False)

    # input: 416 x 416 x 3
    # xception.output: 13 x 13 x 2048
    # block13_sepconv2_bn(middle in block13, layers[121]): 26 x 26 x 1024
    # add_46(end of block12, layers[115]): 26 x 26 x 728

    conv_head1 = compose(
        Depthwise_Separable_Conv2D_BN_Leaky(2048, (3, 3)),
        Depthwise_Separable_Conv2D_BN_Leaky(2048, (3, 3)))(xception.output)

    # block13_sepconv2_bn output shape: 26 x 26 x 1024
    block13_sepconv2_bn = xception.layers[121].output
    conv_head2 = DarknetConv2D_BN_Leaky(128, (1, 1))(block13_sepconv2_bn)
    # TODO: Allow Keras Lambda to use func arguments for output_shape?
    conv_head2_reshaped = Lambda(
        space_to_depth_x2,
        output_shape=space_to_depth_x2_output_shape,
        name='space_to_depth')(conv_head2)

    x = Concatenate()([conv_head2_reshaped, conv_head1])
    x = Depthwise_Separable_Conv2D_BN_Leaky(2048, (3, 3))(x)
    x = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name='predict_conv')(x)
    return Model(inputs, x)

