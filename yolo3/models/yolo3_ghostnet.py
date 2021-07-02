#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v3 GhostNet Model Defined in Keras."""

from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from common.backbones.ghostnet import GhostNet

from yolo3.models.layers import yolo3_predictions, yolo3lite_predictions, tiny_yolo3_predictions, tiny_yolo3lite_predictions
from yolo3.models.ultralite_layers import yolo3_ultralite_predictions, tiny_yolo3_ultralite_predictions


def yolo3_ghostnet_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 GhostNet model CNN body in Keras."""
    ghostnet = GhostNet(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(ghostnet.layers)))

    # input: 416 x 416 x 3
    # blocks_9_0_relu(layer 291, final feature map): 13 x 13 x 960
    # blocks_8_3_add(layer 288, end of block8): 13 x 13 x 160

    # blocks_7_0_ghost1_concat(layer 203, middle in block7) : 26 x 26 x 672
    # blocks_6_4_add(layer 196, end of block6) : 26 x 26 x 112

    # blocks_5_0_ghost1_concat(layer 101, middle in block5) : 52 x 52 x 240
    # blocks_4_0_add(layer 94, end of block4): 52 x 52 x 40

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1: 13 x 13 x 960
    f1 = ghostnet.layers[291].output
    # f2: 26 x 26 x 672
    f2 = ghostnet.layers[203].output
    # f3: 52 x 52 x 240
    f3 = ghostnet.layers[101].output

    f1_channel_num = 960
    f2_channel_num = 672
    f3_channel_num = 240

    y1, y2, y3 = yolo3_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yolo3lite_ghostnet_body(inputs, num_anchors, num_classes):
    '''Create YOLO_v3 Lite GhostNet model CNN body in keras.'''
    ghostnet = GhostNet(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(ghostnet.layers)))

    # input: 416 x 416 x 3
    # blocks_9_0_relu(layer 291, final feature map): 13 x 13 x 960
    # blocks_8_3_add(layer 288, end of block8): 13 x 13 x 160

    # blocks_7_0_ghost1_concat(layer 203, middle in block7) : 26 x 26 x 672
    # blocks_6_4_add(layer 196, end of block6) : 26 x 26 x 112

    # blocks_5_0_ghost1_concat(layer 101, middle in block5) : 52 x 52 x 240
    # blocks_4_0_add(layer 94, end of block4): 52 x 52 x 40

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1: 13 x 13 x 960
    f1 = ghostnet.layers[291].output
    # f2: 26 x 26 x 672
    f2 = ghostnet.layers[203].output
    # f3: 52 x 52 x 240
    f3 = ghostnet.layers[101].output

    f1_channel_num = 960
    f2_channel_num = 672
    f3_channel_num = 240

    y1, y2, y3 = yolo3lite_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs = inputs, outputs=[y1,y2,y3])



def tiny_yolo3_ghostnet_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 GhostNet model CNN body in keras.'''
    ghostnet = GhostNet(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(ghostnet.layers)))

    # input: 416 x 416 x 3
    # blocks_9_0_relu(layer 291, final feature map): 13 x 13 x 960
    # blocks_8_3_add(layer 288, end of block8): 13 x 13 x 160

    # blocks_7_0_ghost1_concat(layer 203, middle in block7) : 26 x 26 x 672
    # blocks_6_4_add(layer 196, end of block6) : 26 x 26 x 112

    # blocks_5_0_ghost1_concat(layer 101, middle in block5) : 52 x 52 x 240
    # blocks_4_0_add(layer 94, end of block4): 52 x 52 x 40

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1: 13 x 13 x 960
    f1 = ghostnet.layers[291].output
    # f2: 26 x 26 x 672
    f2 = ghostnet.layers[203].output
    # f3: 52 x 52 x 240
    #f3 = ghostnet.layers[101].output

    f1_channel_num = 960
    f2_channel_num = 672
    #f3_channel_num = 240

    y1, y2 = tiny_yolo3_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2])


def tiny_yolo3lite_ghostnet_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 Lite GhostNet model CNN body in keras.'''
    ghostnet = GhostNet(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(ghostnet.layers)))

    # input: 416 x 416 x 3
    # blocks_9_0_relu(layer 291, final feature map): 13 x 13 x 960
    # blocks_8_3_add(layer 288, end of block8): 13 x 13 x 160

    # blocks_7_0_ghost1_concat(layer 203, middle in block7) : 26 x 26 x 672
    # blocks_6_4_add(layer 196, end of block6) : 26 x 26 x 112

    # blocks_5_0_ghost1_concat(layer 101, middle in block5) : 52 x 52 x 240
    # blocks_4_0_add(layer 94, end of block4): 52 x 52 x 40

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1: 13 x 13 x 960
    f1 = ghostnet.layers[291].output
    # f2: 26 x 26 x 672
    f2 = ghostnet.layers[203].output
    # f3: 52 x 52 x 240
    #f3 = ghostnet.layers[101].output

    f1_channel_num = 960
    f2_channel_num = 672
    #f3_channel_num = 240

    y1, y2 = tiny_yolo3lite_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2])


def yolo3_ultralite_ghostnet_body(inputs, num_anchors, num_classes):
    '''Create YOLO_v3 Ultra-Lite GhostNet model CNN body in keras.'''
    ghostnet = GhostNet(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(ghostnet.layers)))

    # input: 416 x 416 x 3
    # blocks_9_0_relu(layer 291, final feature map): 13 x 13 x 960
    # blocks_8_3_add(layer 288, end of block8): 13 x 13 x 160

    # blocks_7_0_ghost1_concat(layer 203, middle in block7) : 26 x 26 x 672
    # blocks_6_4_add(layer 196, end of block6) : 26 x 26 x 112

    # blocks_5_0_ghost1_concat(layer 101, middle in block5) : 52 x 52 x 240
    # blocks_4_0_add(layer 94, end of block4): 52 x 52 x 40

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1: 13 x 13 x 960
    f1 = ghostnet.layers[291].output
    # f2: 26 x 26 x 672
    f2 = ghostnet.layers[203].output
    # f3: 52 x 52 x 240
    f3 = ghostnet.layers[101].output

    f1_channel_num = 960
    f2_channel_num = 672
    f3_channel_num = 240

    y1, y2, y3 = yolo3_ultralite_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def tiny_yolo3_ultralite_ghostnet_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 Ultra-Lite GhostNet model CNN body in keras.'''
    ghostnet = GhostNet(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(ghostnet.layers)))

    # input: 416 x 416 x 3
    # blocks_9_0_relu(layer 291, final feature map): 13 x 13 x 960
    # blocks_8_3_add(layer 288, end of block8): 13 x 13 x 160

    # blocks_7_0_ghost1_concat(layer 203, middle in block7) : 26 x 26 x 672
    # blocks_6_4_add(layer 196, end of block6) : 26 x 26 x 112

    # blocks_5_0_ghost1_concat(layer 101, middle in block5) : 52 x 52 x 240
    # blocks_4_0_add(layer 94, end of block4): 52 x 52 x 40

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1: 13 x 13 x 960
    f1 = ghostnet.layers[291].output
    # f2: 26 x 26 x 672
    f2 = ghostnet.layers[203].output
    # f3: 52 x 52 x 240
    #f3 = ghostnet.layers[101].output

    f1_channel_num = 960
    f2_channel_num = 672
    #f3_channel_num = 240

    y1, y2 = tiny_yolo3_ultralite_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2])
