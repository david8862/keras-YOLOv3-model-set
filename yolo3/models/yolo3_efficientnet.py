#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v3 EfficientNet Model Defined in Keras."""

from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from common.backbones.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7

#from yolo3.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, make_last_layers, make_depthwise_separable_last_layers, make_spp_depthwise_separable_last_layers
from yolo3.models.layers import yolo3_predictions, yolo3lite_predictions, tiny_yolo3_predictions, tiny_yolo3lite_predictions


def get_efficientnet_backbone_info(input_tensor, level=0):
    """Parse different level EfficientNet backbone feature map info for YOLOv3 head build."""
    if level == 0:
        # input: 416 x 416 x 3
        # top_activation: 13 x 13 x 1280
        # block6a_expand_activation(middle in block6a): 26 x 26 x 672
        # block5c_add(end of block5c): 26 x 26 x 112
        # block4a_expand_activation(middle in block4a): 52 x 52 x 240
        # block3b_add(end of block3b): 52 x 52 x 40
        efficientnet = EfficientNetB0(input_tensor=input_tensor, weights='imagenet', include_top=False)

        f1_name = 'top_activation'
        f1_channel_num = 1280
        f2_name = 'block6a_expand_activation'
        f2_channel_num = 672
        f3_name = 'block4a_expand_activation'
        f3_channel_num = 240

    elif level == 1:
        # input: 416 x 416 x 3
        # top_activation: 13 x 13 x 1280
        # block6a_expand_activation(middle in block6a): 26 x 26 x 672
        # block5d_add(end of block5d): 26 x 26 x 112
        # block4a_expand_activation(middle in block4a): 52 x 52 x 240
        # block3c_add(end of block3c): 52 x 52 x 40
        efficientnet = EfficientNetB1(input_tensor=input_tensor, weights='imagenet', include_top=False)

        f1_name = 'top_activation'
        f1_channel_num = 1280
        f2_name = 'block6a_expand_activation'
        f2_channel_num = 672
        f3_name = 'block4a_expand_activation'
        f3_channel_num = 240

    elif level == 2:
        # input: 416 x 416 x 3
        # top_activation: 13 x 13 x 1408
        # block6a_expand_activation(middle in block6a): 26 x 26 x 720
        # block5d_add(end of block5d): 26 x 26 x 120
        # block4a_expand_activation(middle in block4a): 52 x 52 x 288
        # block3c_add(end of block3c): 52 x 52 x 48
        efficientnet = EfficientNetB2(input_tensor=input_tensor, weights='imagenet', include_top=False)

        f1_name = 'top_activation'
        f1_channel_num = 1408
        f2_name = 'block6a_expand_activation'
        f2_channel_num = 720
        f3_name = 'block4a_expand_activation'
        f3_channel_num = 288

    elif level == 3:
        # input: 416 x 416 x 3
        # top_activation: 13 x 13 x 1536
        # block6a_expand_activation(middle in block6a): 26 x 26 x 816
        # block5e_add(end of block5e): 26 x 26 x 136
        # block4a_expand_activation(middle in block4a): 52 x 52 x 288
        # block3c_add(end of block3c): 52 x 52 x 48
        efficientnet = EfficientNetB3(input_tensor=input_tensor, weights='imagenet', include_top=False)

        f1_name = 'top_activation'
        f1_channel_num = 1536
        f2_name = 'block6a_expand_activation'
        f2_channel_num = 816
        f3_name = 'block4a_expand_activation'
        f3_channel_num = 288

    elif level == 4:
        # input: 416 x 416 x 3
        # top_activation: 13 x 13 x 1792
        # block6a_expand_activation(middle in block6a): 26 x 26 x 960
        # block5f_add(end of block5f): 26 x 26 x 160
        # block4a_expand_activation(middle in block4a): 52 x 52 x 336
        # block3d_add(end of block3d): 52 x 52 x 56
        efficientnet = EfficientNetB4(input_tensor=input_tensor, weights='imagenet', include_top=False)

        f1_name = 'top_activation'
        f1_channel_num = 1792
        f2_name = 'block6a_expand_activation'
        f2_channel_num = 960
        f3_name = 'block4a_expand_activation'
        f3_channel_num = 336

    elif level == 5:
        # input: 416 x 416 x 3
        # top_activation: 13 x 13 x 2048
        # block6a_expand_activation(middle in block6a): 26 x 26 x 1056
        # block5g_add(end of block5g): 26 x 26 x 176
        # block4a_expand_activation(middle in block4a): 52 x 52 x 384
        # block3e_add(end of block3e): 52 x 52 x 64
        efficientnet = EfficientNetB5(input_tensor=input_tensor, weights='imagenet', include_top=False)

        f1_name = 'top_activation'
        f1_channel_num = 2048
        f2_name = 'block6a_expand_activation'
        f2_channel_num = 1056
        f3_name = 'block4a_expand_activation'
        f3_channel_num = 384

    elif level == 6:
        # input: 416 x 416 x 3
        # top_activation: 13 x 13 x 2304
        # block6a_expand_activation(middle in block6a): 26 x 26 x 1200
        # block5h_add(end of block5h): 26 x 26 x 200
        # block4a_expand_activation(middle in block4a): 52 x 52 x 432
        # block3f_add(end of block3f): 52 x 52 x 72
        efficientnet = EfficientNetB6(input_tensor=input_tensor, weights='imagenet', include_top=False)

        f1_name = 'top_activation'
        f1_channel_num = 2304
        f2_name = 'block6a_expand_activation'
        f2_channel_num = 1200
        f3_name = 'block4a_expand_activation'
        f3_channel_num = 432

    elif level == 7:
        # input: 416 x 416 x 3
        # top_activation: 13 x 13 x 2560
        # block6a_expand_activation(middle in block6a): 26 x 26 x 1344
        # block5j_add(end of block5j): 26 x 26 x 224
        # block4a_expand_activation(middle in block4a): 52 x 52 x 480
        # block3g_add(end of block3g): 52 x 52 x 80
        efficientnet = EfficientNetB7(input_tensor=input_tensor, weights='imagenet', include_top=False)

        f1_name = 'top_activation'
        f1_channel_num = 2560
        f2_name = 'block6a_expand_activation'
        f2_channel_num = 1344
        f3_name = 'block4a_expand_activation'
        f3_channel_num = 480

    else:
        raise ValueError('Invalid efficientnet backbone type')

    # f1 shape : 13 x 13 x f1_channel_num
    # f2 shape : 26 x 26 x f2_channel_num
    # f3 shape : 52 x 52 x f3_channel_num
    feature_map_info = {'f1_name' : f1_name,
                        'f1_channel_num' : f1_channel_num,
                        'f2_name' : f2_name,
                        'f2_channel_num' : f2_channel_num,
                        'f3_name' : f3_name,
                        'f3_channel_num' : f3_channel_num,
                        }

    return efficientnet, feature_map_info


def yolo3_efficientnet_body(inputs, num_anchors, num_classes, level=3):
    '''
    Create YOLO_v3 EfficientNet model CNN body in keras.
    # Arguments
        level: EfficientNet level number.
            by default we use EfficientNetB3 as backbone
    '''
    efficientnet, feature_map_info = get_efficientnet_backbone_info(inputs, level=level)
    print('backbone layers number: {}'.format(len(efficientnet.layers)))

    f1 = efficientnet.get_layer('top_activation').output
    f1_channel_num = feature_map_info['f1_channel_num']

    f2 = efficientnet.get_layer('block6a_expand_activation').output
    f2_channel_num = feature_map_info['f2_channel_num']

    f3 = efficientnet.get_layer('block4a_expand_activation').output
    f3_channel_num = feature_map_info['f3_channel_num']

    y1, y2, y3 = yolo3_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yolo3lite_efficientnet_body(inputs, num_anchors, num_classes, level=3):
    '''
    Create YOLO_v3 Lite EfficientNet model CNN body in keras.
    # Arguments
        level: EfficientNet level number.
            by default we use EfficientNetB3 as backbone
    '''
    efficientnet, feature_map_info = get_efficientnet_backbone_info(inputs, level=level)
    print('backbone layers number: {}'.format(len(efficientnet.layers)))

    f1 = efficientnet.get_layer('top_activation').output
    f1_channel_num = feature_map_info['f1_channel_num']

    f2 = efficientnet.get_layer('block6a_expand_activation').output
    f2_channel_num = feature_map_info['f2_channel_num']

    f3 = efficientnet.get_layer('block4a_expand_activation').output
    f3_channel_num = feature_map_info['f3_channel_num']

    y1, y2, y3 = yolo3lite_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yolo3lite_spp_efficientnet_body(inputs, num_anchors, num_classes, level=3):
    '''
    Create YOLO_v3 Lite SPP EfficientNet model CNN body in keras.
    # Arguments
        level: EfficientNet level number.
            by default we use EfficientNetB3 as backbone
    '''
    efficientnet, feature_map_info = get_efficientnet_backbone_info(inputs, level=level)
    print('backbone layers number: {}'.format(len(efficientnet.layers)))

    f1 = efficientnet.get_layer('top_activation').output
    f1_channel_num = feature_map_info['f1_channel_num']

    f2 = efficientnet.get_layer('block6a_expand_activation').output
    f2_channel_num = feature_map_info['f2_channel_num']

    f3 = efficientnet.get_layer('block4a_expand_activation').output
    f3_channel_num = feature_map_info['f3_channel_num']

    y1, y2, y3 = yolo3lite_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes, use_spp=True)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def tiny_yolo3_efficientnet_body(inputs, num_anchors, num_classes, level=0):
    '''
    Create Tiny YOLO_v3 EfficientNet model CNN body in keras.
    # Arguments
        level: EfficientNet level number.
            by default we use basic EfficientNetB0 as backbone
    '''
    efficientnet, feature_map_info = get_efficientnet_backbone_info(inputs, level=level)
    print('backbone layers number: {}'.format(len(efficientnet.layers)))

    f1 = efficientnet.get_layer('top_activation').output
    f2 = efficientnet.get_layer('block6a_expand_activation').output
    f1_channel_num = feature_map_info['f1_channel_num']
    f2_channel_num = feature_map_info['f2_channel_num']

    y1, y2 = tiny_yolo3_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2])


def tiny_yolo3lite_efficientnet_body(inputs, num_anchors, num_classes, level=0):
    '''
    Create Tiny YOLO_v3 Lite EfficientNet model CNN body in keras.
    # Arguments
        level: EfficientNet level number.
            by default we use basic EfficientNetB0 as backbone
    '''
    efficientnet, feature_map_info = get_efficientnet_backbone_info(inputs, level=level)
    print('backbone layers number: {}'.format(len(efficientnet.layers)))

    f1 = efficientnet.get_layer('top_activation').output
    f2 = efficientnet.get_layer('block6a_expand_activation').output
    f1_channel_num = feature_map_info['f1_channel_num']
    f2_channel_num = feature_map_info['f2_channel_num']

    y1, y2 = tiny_yolo3lite_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2])

