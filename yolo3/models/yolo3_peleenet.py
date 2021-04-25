#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v3 PeleeNet Model Defined in Keras."""

from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from common.backbones.peleenet import PeleeNet

from yolo3.models.layers import yolo3_predictions, yolo3lite_predictions, tiny_yolo3_predictions, tiny_yolo3lite_predictions
from yolo3.models.ultralite_layers import yolo3_ultralite_predictions, tiny_yolo3_ultralite_predictions


def yolo3_peleenet_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 PeleeNet model CNN body in Keras."""
    peleenet = PeleeNet(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(peleenet.layers)))

    # input: 416 x 416 x 3
    # re_lu_338(layer 365, final feature map): 13 x 13 x 704
    # re_lu_307(layer 265, end of stride 16) : 26 x 26 x 512
    # re_lu_266(layer 133, end of stride 8)  : 52 x 52 x 256

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1: 13 x 13 x 704
    f1 = peleenet.layers[365].output
    # f2: 26 x 26 x 512
    f2 = peleenet.layers[265].output
    # f3: 52 x 52 x 256
    f3 = peleenet.layers[133].output

    f1_channel_num = 704
    f2_channel_num = 512
    f3_channel_num = 256

    y1, y2, y3 = yolo3_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yolo3lite_peleenet_body(inputs, num_anchors, num_classes):
    '''Create YOLO_v3 Lite PeleeNet model CNN body in keras.'''
    peleenet = PeleeNet(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(peleenet.layers)))

    # input: 416 x 416 x 3
    # re_lu_338(layer 365, final feature map): 13 x 13 x 704
    # re_lu_307(layer 265, end of stride 16) : 26 x 26 x 512
    # re_lu_266(layer 133, end of stride 8)  : 52 x 52 x 256

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1: 13 x 13 x 704
    f1 = peleenet.layers[365].output
    # f2: 26 x 26 x 512
    f2 = peleenet.layers[265].output
    # f3: 52 x 52 x 256
    f3 = peleenet.layers[133].output

    f1_channel_num = 704
    f2_channel_num = 512
    f3_channel_num = 256

    y1, y2, y3 = yolo3lite_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs = inputs, outputs=[y1,y2,y3])



def tiny_yolo3_peleenet_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 PeleeNet model CNN body in keras.'''
    peleenet = PeleeNet(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(peleenet.layers)))

    # input: 416 x 416 x 3
    # re_lu_338(layer 365, final feature map): 13 x 13 x 704
    # re_lu_307(layer 265, end of stride 16) : 26 x 26 x 512
    # re_lu_266(layer 133, end of stride 8)  : 52 x 52 x 256

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1: 13 x 13 x 704
    f1 = peleenet.layers[365].output
    # f2: 26 x 26 x 512
    f2 = peleenet.layers[265].output
    # f3: 52 x 52 x 256
    f3 = peleenet.layers[133].output

    f1_channel_num = 704
    f2_channel_num = 512

    y1, y2 = tiny_yolo3_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2])


def tiny_yolo3lite_peleenet_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 Lite PeleeNet model CNN body in keras.'''
    peleenet = PeleeNet(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(peleenet.layers)))

    # input: 416 x 416 x 3
    # re_lu_338(layer 365, final feature map): 13 x 13 x 704
    # re_lu_307(layer 265, end of stride 16) : 26 x 26 x 512
    # re_lu_266(layer 133, end of stride 8)  : 52 x 52 x 256

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1: 13 x 13 x 704
    f1 = peleenet.layers[365].output
    # f2: 26 x 26 x 512
    f2 = peleenet.layers[265].output
    # f3: 52 x 52 x 256
    f3 = peleenet.layers[133].output

    f1_channel_num = 704
    f2_channel_num = 512

    y1, y2 = tiny_yolo3lite_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2])


def yolo3_ultralite_peleenet_body(inputs, num_anchors, num_classes):
    '''Create YOLO_v3 Ultra-Lite PeleeNet model CNN body in keras.'''
    peleenet = PeleeNet(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(peleenet.layers)))

    # input: 416 x 416 x 3
    # re_lu_338(layer 365, final feature map): 13 x 13 x 704
    # re_lu_307(layer 265, end of stride 16) : 26 x 26 x 512
    # re_lu_266(layer 133, end of stride 8)  : 52 x 52 x 256

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1: 13 x 13 x 704
    f1 = peleenet.layers[365].output
    # f2: 26 x 26 x 512
    f2 = peleenet.layers[265].output
    # f3: 52 x 52 x 256
    f3 = peleenet.layers[133].output

    f1_channel_num = 704
    f2_channel_num = 512
    f3_channel_num = 256

    y1, y2, y3 = yolo3_ultralite_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def tiny_yolo3_ultralite_peleenet_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 Ultra-Lite PeleeNet model CNN body in keras.'''
    peleenet = PeleeNet(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(peleenet.layers)))

    # input: 416 x 416 x 3
    # re_lu_338(layer 365, final feature map): 13 x 13 x 704
    # re_lu_307(layer 265, end of stride 16) : 26 x 26 x 512
    # re_lu_266(layer 133, end of stride 8)  : 52 x 52 x 256

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1: 13 x 13 x 704
    f1 = peleenet.layers[365].output
    # f2: 26 x 26 x 512
    f2 = peleenet.layers[265].output

    f1_channel_num = 704
    f2_channel_num = 512

    y1, y2 = tiny_yolo3_ultralite_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2])
