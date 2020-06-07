#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v2 Darknet Model Defined in Keras."""

from tensorflow.keras.layers import MaxPooling2D, Lambda, Concatenate, GlobalAveragePooling2D, Softmax
from tensorflow.keras.models import Model

from yolo2.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, bottleneck_block, bottleneck_x2_block, yolo2_predictions


def darknet19_body():
    """Generate first 18 conv layers of Darknet-19."""
    return compose(
        DarknetConv2D_BN_Leaky(32, (3, 3)),
        MaxPooling2D(),
        DarknetConv2D_BN_Leaky(64, (3, 3)),
        MaxPooling2D(),
        bottleneck_block(128, 64),
        MaxPooling2D(),
        bottleneck_block(256, 128),
        MaxPooling2D(),
        bottleneck_x2_block(512, 256),
        MaxPooling2D(),
        bottleneck_x2_block(1024, 512))


def darknet19(inputs):
    """Generate Darknet-19 model for Imagenet classification."""
    body = darknet19_body()(inputs)
    x = DarknetConv2D(1000, (1, 1))(body)
    x = GlobalAveragePooling2D()(x)
    logits = Softmax()(x)
    return Model(inputs, logits)


def yolo2_body(inputs, num_anchors, num_classes, weights_path=None):
    """Create YOLO_V2 model CNN body in Keras."""
    darknet19 = Model(inputs, darknet19_body()(inputs))
    if weights_path is not None:
        darknet19.load_weights(weights_path, by_name=True)
        print('Load weights {}.'.format(weights_path))

    # input: 416 x 416 x 3
    # darknet19.output : 13 x 13 x 1024
    # conv13(layers[43]) : 26 x 26 x 512
    print('backbone layers number: {}'.format(len(darknet19.layers)))

    # f1: 13 x 13 x 1024
    f1 = darknet19.output
    # f2: 26 x 26 x 512
    f2 = darknet19.layers[43].output

    f1_channel_num = 1024
    f2_channel_num = 512

    y = yolo2_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)
    return Model(inputs, y)


def tiny_yolo2_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v2 model CNN body in keras.'''
    # backbone feature output
    f1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)))(inputs)

    # TODO: darknet tiny YOLOv2 use different filter number for COCO and VOC
    if num_classes == 80:
        f1_channel_num = 512
    else:
        f1_channel_num = 1024

    y = compose(
            DarknetConv2D_BN_Leaky(f1_channel_num, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv'))(f1)

    return Model(inputs, y)

