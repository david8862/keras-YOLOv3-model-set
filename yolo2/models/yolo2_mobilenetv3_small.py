#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v2 MobileNetV2 Model Defined in Keras."""

from tensorflow.keras.layers import MaxPooling2D, Lambda, Concatenate, GlobalAveragePooling2D, Softmax
from tensorflow.keras.models import Model
from common.backbones.mobilenet_v3 import MobileNetV3Small

from yolo2.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, bottleneck_block, bottleneck_x2_block, space_to_depth_x2, space_to_depth_x2_output_shape



def yolo2_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V2 MobileNetV3Small model CNN body in Keras."""

    mobilenetv3small = MobileNetV3Small(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # mobilenetv3small.output(layer 165, final feature map): 13 x 13 x (576*alpha)
    # expanded_conv_10/Add(layer 162, end of block10): 13 x 13 x (96*alpha)

    # activation_22(layer 117, middle in block8) : 26 x 26 x (288*alpha)
    # expanded_conv_7/Add(layer 114, end of block7) : 26 x 26 x (48*alpha)

    conv_head1 = compose(
        DarknetConv2D_BN_Leaky(int(576*alpha), (3, 3)),
        DarknetConv2D_BN_Leaky(int(576*alpha), (3, 3)))(mobilenetv3small.output)

    # activation_22(layer 117) output shape: 26 x 26 x (288*alpha)
    activation_22 = mobilenetv3small.layers[117].output
    conv_head2 = DarknetConv2D_BN_Leaky(int(64*alpha), (1, 1))(activation_22)
    # TODO: Allow Keras Lambda to use func arguments for output_shape?
    conv_head2_reshaped = Lambda(
        space_to_depth_x2,
        output_shape=space_to_depth_x2_output_shape,
        name='space_to_depth')(conv_head2)

    x = Concatenate()([conv_head2_reshaped, conv_head1])
    x = DarknetConv2D_BN_Leaky(int(576*alpha), (3, 3))(x)
    x = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name='predict_conv')(x)
    return Model(inputs, x)


def yolo2lite_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V2 Lite MobileNetV3Small model CNN body in Keras."""

    mobilenetv3small = MobileNetV3Small(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # mobilenetv3small.output(layer 165, final feature map): 13 x 13 x (576*alpha)
    # expanded_conv_10/Add(layer 162, end of block10): 13 x 13 x (96*alpha)

    # activation_22(layer 117, middle in block8) : 26 x 26 x (288*alpha)
    # expanded_conv_7/Add(layer 114, end of block7) : 26 x 26 x (48*alpha)

    conv_head1 = compose(
        Depthwise_Separable_Conv2D_BN_Leaky(int(576*alpha), (3, 3), block_id_str='11'),
        Depthwise_Separable_Conv2D_BN_Leaky(int(576*alpha), (3, 3), block_id_str='12'))(mobilenetv3small.output)

    # activation_22(layer 117) output shape: 26 x 26 x (288*alpha)
    activation_22 = mobilenetv3small.layers[117].output
    conv_head2 = DarknetConv2D_BN_Leaky(int(64*alpha), (1, 1))(activation_22)
    # TODO: Allow Keras Lambda to use func arguments for output_shape?
    conv_head2_reshaped = Lambda(
        space_to_depth_x2,
        output_shape=space_to_depth_x2_output_shape,
        name='space_to_depth')(conv_head2)

    x = Concatenate()([conv_head2_reshaped, conv_head1])
    x = Depthwise_Separable_Conv2D_BN_Leaky(int(576*alpha), (3, 3), block_id_str='13')(x)
    x = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name='predict_conv')(x)
    return Model(inputs, x)


def tiny_yolo2_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create Tiny YOLO_V2 MobileNetV3Small model CNN body in Keras."""
    mobilenetv3small = MobileNetV3Small(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # mobilenetv3small.output(layer 165, final feature map): 13 x 13 x (576*alpha)
    y = compose(
            DarknetConv2D_BN_Leaky(int(576*alpha), (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv'))(mobilenetv3small.output)

    return Model(inputs, y)


def tiny_yolo2lite_mobilenetv3small_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create Tiny YOLO_V2 Lite MobileNetV3Small model CNN body in Keras."""
    mobilenetv3small = MobileNetV3Small(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # mobilenetv3small.output(layer 165, final feature map): 13 x 13 x (576*alpha)
    y = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(int(576*alpha), (3,3), block_id_str='11'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv'))(mobilenetv3small.output)

    return Model(inputs, y)
