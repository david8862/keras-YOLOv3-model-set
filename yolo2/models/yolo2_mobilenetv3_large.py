#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v2 MobileNetV2 Model Defined in Keras."""

from tensorflow.keras.layers import MaxPooling2D, Lambda, Concatenate, GlobalAveragePooling2D, Softmax
from tensorflow.keras.models import Model
from common.backbones.mobilenet_v3 import MobileNetV3Large

from yolo2.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, bottleneck_block, bottleneck_x2_block, space_to_depth_x2, space_to_depth_x2_output_shape



def yolo2_mobilenetv3large_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V2 MobileNetV3Large model CNN body in Keras."""

    mobilenetv3large = MobileNetV3Large(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # mobilenetv3large.output(layer 194, final feature map): 13 x 13 x (960*alpha)
    # expanded_conv_14/Add(layer 191, end of block14): 13 x 13 x (160*alpha)

    # activation_29(layer 146, middle in block12) : 26 x 26 x (672*alpha)
    # expanded_conv_11/Add(layer 143, end of block11) : 26 x 26 x (112*alpha)

    conv_head1 = compose(
        DarknetConv2D_BN_Leaky(int(960*alpha), (3, 3)),
        DarknetConv2D_BN_Leaky(int(960*alpha), (3, 3)))(mobilenetv3large.output)

    # activation_29(layer 146) output shape: 26 x 26 x (672*alpha)
    activation_29 = mobilenetv3large.layers[146].output
    conv_head2 = DarknetConv2D_BN_Leaky(int(64*alpha), (1, 1))(activation_29)
    # TODO: Allow Keras Lambda to use func arguments for output_shape?
    conv_head2_reshaped = Lambda(
        space_to_depth_x2,
        output_shape=space_to_depth_x2_output_shape,
        name='space_to_depth')(conv_head2)

    x = Concatenate()([conv_head2_reshaped, conv_head1])
    x = DarknetConv2D_BN_Leaky(int(960*alpha), (3, 3))(x)
    x = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name='predict_conv')(x)
    return Model(inputs, x)


def yolo2lite_mobilenetv3large_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V2 Lite MobileNetV3Large model CNN body in Keras."""

    mobilenetv3large = MobileNetV3Large(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # mobilenetv3large.output(layer 194, final feature map): 13 x 13 x (960*alpha)
    # expanded_conv_14/Add(layer 191, end of block14): 13 x 13 x (160*alpha)

    # activation_29(layer 146, middle in block12) : 26 x 26 x (672*alpha)
    # expanded_conv_11/Add(layer 143, end of block11) : 26 x 26 x (112*alpha)

    conv_head1 = compose(
        Depthwise_Separable_Conv2D_BN_Leaky(int(960*alpha), (3, 3), block_id_str='15'),
        Depthwise_Separable_Conv2D_BN_Leaky(int(960*alpha), (3, 3), block_id_str='16'))(mobilenetv3large.output)

    # activation_29(layer 146) output shape: 26 x 26 x (672*alpha)
    activation_29 = mobilenetv3large.layers[146].output
    conv_head2 = DarknetConv2D_BN_Leaky(int(64*alpha), (1, 1))(activation_29)
    # TODO: Allow Keras Lambda to use func arguments for output_shape?
    conv_head2_reshaped = Lambda(
        space_to_depth_x2,
        output_shape=space_to_depth_x2_output_shape,
        name='space_to_depth')(conv_head2)

    x = Concatenate()([conv_head2_reshaped, conv_head1])
    x = Depthwise_Separable_Conv2D_BN_Leaky(int(960*alpha), (3, 3), block_id_str='17')(x)
    x = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name='predict_conv')(x)
    return Model(inputs, x)


def tiny_yolo2_mobilenetv3large_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create Tiny YOLO_V2 MobileNetV3Large model CNN body in Keras."""
    mobilenetv3large = MobileNetV3Large(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # mobilenetv3large.output(layer 194, final feature map): 13 x 13 x (960*alpha)
    y = compose(
            DarknetConv2D_BN_Leaky(int(960*alpha), (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv'))(mobilenetv3large.output)

    return Model(inputs, y)


def tiny_yolo2lite_mobilenetv3large_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create Tiny YOLO_V2 Lite MobileNetV3Large model CNN body in Keras."""
    mobilenetv3large = MobileNetV3Large(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # mobilenetv3large.output(layer 194, final feature map): 13 x 13 x (960*alpha)
    y = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(int(960*alpha), (3,3), block_id_str='15'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv'))(mobilenetv3large.output)

    return Model(inputs, y)
