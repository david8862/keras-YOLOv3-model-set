#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v2 MobileNet Model Defined in Keras."""

from tensorflow.keras.layers import MaxPooling2D, Lambda, Concatenate, GlobalAveragePooling2D, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet

from yolo2.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, bottleneck_block, bottleneck_x2_block, space_to_depth_x2, space_to_depth_x2_output_shape



def yolo2_mobilenet_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V2 MobileNet model CNN body in Keras."""

    mobilenet = MobileNet(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # mobilenet.output            : 13 x 13 x (1024*alpha)
    # conv_pw_11_relu(layers[73]) : 26 x 26 x (512*alpha)

    conv_head1 = compose(
        DarknetConv2D_BN_Leaky(int(1024*alpha), (3, 3)),
        DarknetConv2D_BN_Leaky(int(1024*alpha), (3, 3)))(mobilenet.output)

    # conv_pw_11_relu output shape: 26 x 26 x (512*alpha)
    conv_pw_11_relu = mobilenet.layers[73].output
    conv_head2 = DarknetConv2D_BN_Leaky(int(64*alpha), (1, 1))(conv_pw_11_relu)
    # TODO: Allow Keras Lambda to use func arguments for output_shape?
    conv_head2_reshaped = Lambda(
        space_to_depth_x2,
        output_shape=space_to_depth_x2_output_shape,
        name='space_to_depth')(conv_head2)

    x = Concatenate()([conv_head2_reshaped, conv_head1])
    x = DarknetConv2D_BN_Leaky(int(1024*alpha), (3, 3))(x)
    x = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name='predict_conv')(x)
    return Model(inputs, x)


def yolo2lite_mobilenet_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create YOLO_V2 Lite MobileNet model CNN body in Keras."""

    mobilenet = MobileNet(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # mobilenet.output            : 13 x 13 x (1024*alpha)
    # conv_pw_11_relu(layers[73]) : 26 x 26 x (512*alpha)

    conv_head1 = compose(
        Depthwise_Separable_Conv2D_BN_Leaky(int(1024*alpha), (3, 3), block_id_str='14'),
        Depthwise_Separable_Conv2D_BN_Leaky(int(1024*alpha), (3, 3), block_id_str='15'))(mobilenet.output)

    # conv_pw_11_relu output shape: 26 x 26 x (512*alpha)
    conv_pw_11_relu = mobilenet.layers[73].output
    conv_head2 = DarknetConv2D_BN_Leaky(int(64*alpha), (1, 1))(conv_pw_11_relu)
    # TODO: Allow Keras Lambda to use func arguments for output_shape?
    conv_head2_reshaped = Lambda(
        space_to_depth_x2,
        output_shape=space_to_depth_x2_output_shape,
        name='space_to_depth')(conv_head2)

    x = Concatenate()([conv_head2_reshaped, conv_head1])
    x = Depthwise_Separable_Conv2D_BN_Leaky(int(1024*alpha), (3, 3), block_id_str='16')(x)
    x = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name='predict_conv')(x)
    return Model(inputs, x)


def tiny_yolo2_mobilenet_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create Tiny YOLO_V2 MobileNet model CNN body in Keras."""
    mobilenet = MobileNet(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # mobilenet.output : 13 x 13 x (1024*alpha)
    y = compose(
            DarknetConv2D_BN_Leaky(int(1024*alpha), (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv'))(mobilenet.output)

    return Model(inputs, y)


def tiny_yolo2lite_mobilenet_body(inputs, num_anchors, num_classes, alpha=1.0):
    """Create Tiny YOLO_V2 Lite MobileNet model CNN body in Keras."""
    mobilenet = MobileNet(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)

    # input: 416 x 416 x 3
    # mobilenet.output : 13 x 13 x (1024*alpha)
    y = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(int(1024*alpha), (3,3), block_id_str='14'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv'))(mobilenet.output)

    return Model(inputs, y)
