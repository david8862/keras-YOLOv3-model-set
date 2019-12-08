#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v2 MobileNetV2 Model Defined in Keras."""

from tensorflow.keras.layers import MaxPooling2D, Lambda, Concatenate, GlobalAveragePooling2D, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from yolo2.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, bottleneck_block, bottleneck_x2_block, space_to_depth_x2, space_to_depth_x2_output_shape



def yolo2_mobilenetv2_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V2 MobileNetV2 model CNN body in Keras."""

    mobilenetv2 = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # mobilenetv2.output   : 13 x 13 x 1280
    # block_13_expand_relu(layers[119]) : 26 x 26 x (576*alpha)

    conv_head1 = compose(
        DarknetConv2D_BN_Leaky(1280, (3, 3)),
        DarknetConv2D_BN_Leaky(1280, (3, 3)))(mobilenetv2.output)

    # block_13_expand_relu output shape: 26 x 26 x (576*alpha)
    block_13_expand_relu = mobilenetv2.layers[119].output
    conv_head2 = DarknetConv2D_BN_Leaky(int(64*alpha), (1, 1))(block_13_expand_relu)
    # TODO: Allow Keras Lambda to use func arguments for output_shape?
    conv_head2_reshaped = Lambda(
        space_to_depth_x2,
        output_shape=space_to_depth_x2_output_shape,
        name='space_to_depth')(conv_head2)

    x = Concatenate()([conv_head2_reshaped, conv_head1])
    x = DarknetConv2D_BN_Leaky(1280, (3, 3))(x)
    x = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name='predict_conv')(x)
    return Model(inputs, x)


def yolo2lite_mobilenetv2_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V2 Lite MobileNetV2 model CNN body in Keras."""

    mobilenetv2 = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # mobilenetv2.output   : 13 x 13 x 1280
    # block_13_expand_relu(layers[119]) : 26 x 26 x (576*alpha)

    conv_head1 = compose(
        Depthwise_Separable_Conv2D_BN_Leaky(1280, (3, 3)),
        Depthwise_Separable_Conv2D_BN_Leaky(1280, (3, 3)))(mobilenetv2.output)

    # block_13_expand_relu output shape: 26 x 26 x (576*alpha)
    block_13_expand_relu = mobilenetv2.layers[119].output
    conv_head2 = DarknetConv2D_BN_Leaky(int(64*alpha), (1, 1))(block_13_expand_relu)
    # TODO: Allow Keras Lambda to use func arguments for output_shape?
    conv_head2_reshaped = Lambda(
        space_to_depth_x2,
        output_shape=space_to_depth_x2_output_shape,
        name='space_to_depth')(conv_head2)

    x = Concatenate()([conv_head2_reshaped, conv_head1])
    x = Depthwise_Separable_Conv2D_BN_Leaky(1280, (3, 3))(x)
    x = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name='predict_conv')(x)
    return Model(inputs, x)


def tiny_yolo2_mobilenetv2_body(inputs, num_anchors, num_classes):
    """Create Tiny YOLO_V2 MobileNetV2 model CNN body in Keras."""
    mobilenetv2 = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False, alpha=1.0)

    # input: 416 x 416 x 3
    # mobilenetv2.output : 13 x 13 x 1280
    y = compose(
            DarknetConv2D_BN_Leaky(1280, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv'))(mobilenetv2.output)

    return Model(inputs, y)


def tiny_yolo2lite_mobilenetv2_body(inputs, num_anchors, num_classes):
    """Create Tiny YOLO_V2 Lite MobileNetV2 model CNN body in Keras."""
    mobilenetv2 = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False, alpha=1.0)

    # input: 416 x 416 x 3
    # mobilenetv2.output : 13 x 13 x 1280
    y = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(1280, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv'))(mobilenetv2.output)

    return Model(inputs, y)
