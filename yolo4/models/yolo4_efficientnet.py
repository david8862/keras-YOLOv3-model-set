#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v4 EfficientNet Model Defined in Keras."""

from tensorflow.keras.layers import ZeroPadding2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from common.backbones.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7

from yolo4.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Spp_Conv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, Darknet_Depthwise_Separable_Conv2D_BN_Leaky, make_yolo_head, make_yolo_spp_head, make_yolo_depthwise_separable_head, make_yolo_spp_depthwise_separable_head


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


def yolo4_efficientnet_body(inputs, num_anchors, num_classes, level=1):
    '''
    Create YOLO_v4 EfficientNet model CNN body in keras.
    # Arguments
        level: EfficientNet level number.
            by default we use basic EfficientNetB1 as backbone
    '''
    efficientnet, feature_map_info = get_efficientnet_backbone_info(inputs, level=level)

    f1 = efficientnet.get_layer('top_activation').output
    f1_channel_num = feature_map_info['f1_channel_num']

    f2 = efficientnet.get_layer('block6a_expand_activation').output
    f2_channel_num = feature_map_info['f2_channel_num']

    f3 = efficientnet.get_layer('block4a_expand_activation').output
    f3_channel_num = feature_map_info['f3_channel_num']

    #feature map 1 head (13x13x(f1_channel_num//2) for 416 input)
    x1 = make_yolo_spp_head(f1, f1_channel_num//2)

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x1)

    x2 = DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1))(f2)
    x2 = Concatenate()([x2, x1_upsample])

    #feature map 2 head (26x26x(f2_channel_num//2) for 416 input)
    x2 = make_yolo_head(x2, f2_channel_num//2)

    #upsample fpn merge for feature map 2 & 3
    x2_upsample = compose(
            DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1)),
            UpSampling2D(2))(x2)

    x3 = DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1))(f3)
    x3 = Concatenate()([x3, x2_upsample])

    #feature map 3 head & output (52x52xf3_channel_num for 416 input)
    #x3, y3 = make_last_layers(x3, f3_channel_num//2, num_anchors*(num_classes+5))
    x3 = make_yolo_head(x3, f3_channel_num//2)
    y3 = compose(
            DarknetConv2D_BN_Leaky(f3_channel_num, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x3)

    #downsample fpn merge for feature map 3 & 2
    x3_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (3,3), strides=(2,2)))(x3)

    x2 = Concatenate()([x3_downsample, x2])

    #feature map 2 output (26x26xf2_channel_num for 416 input)
    #x2, y2 = make_last_layers(x2, f2_channel_num//2, num_anchors*(num_classes+5))
    x2 = make_yolo_head(x2, f2_channel_num//2)
    y2 = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            DarknetConv2D_BN_Leaky(f1_channel_num//2, (3,3), strides=(2,2)))(x2)

    x1 = Concatenate()([x2_downsample, x1])

    #feature map 1 output (13x13xf1_channel_num for 416 input)
    #x1, y1 = make_last_layers(x1, f1_channel_num//2, num_anchors*(num_classes+5))
    x1 = make_yolo_head(x1, f1_channel_num//2)
    y1 = compose(
            DarknetConv2D_BN_Leaky(f1_channel_num, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x1)

    return Model(inputs, [y1, y2, y3])



def yolo4lite_efficientnet_body(inputs, num_anchors, num_classes, level=1):
    '''
    Create YOLO_v4 Lite EfficientNet model CNN body in keras.
    # Arguments
        level: EfficientNet level number.
            by default we use basic EfficientNetB1 as backbone
    '''
    efficientnet, feature_map_info = get_efficientnet_backbone_info(inputs, level=level)

    f1 = efficientnet.get_layer('top_activation').output
    f1_channel_num = feature_map_info['f1_channel_num']

    f2 = efficientnet.get_layer('block6a_expand_activation').output
    f2_channel_num = feature_map_info['f2_channel_num']

    f3 = efficientnet.get_layer('block4a_expand_activation').output
    f3_channel_num = feature_map_info['f3_channel_num']

    #feature map 1 head (13x13x(f1_channel_num//2) for 416 input)
    x1 = make_yolo_spp_depthwise_separable_head(f1, f1_channel_num//2, block_id_str='8')

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x1)

    x2 = DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1))(f2)
    x2 = Concatenate()([x2, x1_upsample])

    #feature map 2 head (26x26x(f2_channel_num//2) for 416 input)
    x2 = make_yolo_depthwise_separable_head(x2, f2_channel_num//2, block_id_str='9')

    #upsample fpn merge for feature map 2 & 3
    x2_upsample = compose(
            DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1)),
            UpSampling2D(2))(x2)

    x3 = DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1))(f3)
    x3 = Concatenate()([x3, x2_upsample])

    #feature map 3 head & output (52x52xf3_channel_num for 416 input)
    #x3, y3 = make_depthwise_separable_last_layers(x3, f3_channel_num//2, num_anchors*(num_classes+5), block_id_str='10')
    x3 = make_yolo_depthwise_separable_head(x3, f3_channel_num//2, block_id_str='10')
    y3 = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(f3_channel_num, (3,3), block_id_str='10_3'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x3)

    #downsample fpn merge for feature map 3 & 2
    x3_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            Darknet_Depthwise_Separable_Conv2D_BN_Leaky(f2_channel_num//2, (3,3), strides=(2,2), block_id_str='10_4'))(x3)

    x2 = Concatenate()([x3_downsample, x2])

    #feature map 2 output (26x26xf2_channel_num for 416 input)
    #x2, y2 = make_depthwise_separable_last_layers(x2, f2_channel_num//2, num_anchors*(num_classes+5), block_id_str='11')
    x2 = make_yolo_depthwise_separable_head(x2, f2_channel_num//2, block_id_str='11')
    y2 = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(f2_channel_num, (3,3), block_id_str='11_3'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            Darknet_Depthwise_Separable_Conv2D_BN_Leaky(f1_channel_num//2, (3,3), strides=(2,2), block_id_str='11_4'))(x2)

    x1 = Concatenate()([x2_downsample, x1])

    #feature map 1 output (13x13xf1_channel_num for 416 input)
    #x1, y1 = make_depthwise_separable_last_layers(x1, f1_channel_num//2, num_anchors*(num_classes+5), block_id_str='12')
    x1 = make_yolo_depthwise_separable_head(x1, f1_channel_num//2, block_id_str='12')
    y1 = compose(
            Depthwise_Separable_Conv2D_BN_Leaky(f1_channel_num, (3,3), block_id_str='12_3'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x1)

    return Model(inputs, [y1, y2, y3])



def tiny_yolo4_efficientnet_body(inputs, num_anchors, num_classes, level=0, use_spp=True):
    '''
    Create Tiny YOLO_v4 EfficientNet model CNN body in keras.
    # Arguments
        level: EfficientNet level number.
            by default we use basic EfficientNetB0 as backbone
    '''
    efficientnet, feature_map_info = get_efficientnet_backbone_info(inputs, level=level)

    f1 = efficientnet.get_layer('top_activation').output
    f2 = efficientnet.get_layer('block6a_expand_activation').output
    f1_channel_num = feature_map_info['f1_channel_num']
    f2_channel_num = feature_map_info['f2_channel_num']

    #feature map 1 head (13 x 13 x (f1_channel_num//2) for 416 input)
    x1 = DarknetConv2D_BN_Leaky(f1_channel_num//2, (1,1))(f1)
    if use_spp:
        x1 = Spp_Conv2D_BN_Leaky(x1, f1_channel_num//2)

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x1)
    x2 = compose(
            Concatenate(),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=f2_channel_num, kernel_size=(3, 3), block_id_str='8'),
            DarknetConv2D_BN_Leaky(f2_channel_num, (3,3)))([x1_upsample, f2])

    #feature map 2 output (26 x 26 x f2_channel_num for 416 input)
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            #Darknet_Depthwise_Separable_Conv2D_BN_Leaky(f1_channel_num//2, (3,3), strides=(2,2), block_id_str='9'),
            DarknetConv2D_BN_Leaky(f1_channel_num//2, (3,3), strides=(2,2)))(x2)
    x1 = compose(
            Concatenate(),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=f1_channel_num, kernel_size=(3, 3), block_id_str='10'),
            DarknetConv2D_BN_Leaky(f1_channel_num, (3,3)))([x2_downsample, x1])

    #feature map 1 output (13 x 13 x f1_channel_num for 416 input)
    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(x1)

    return Model(inputs, [y1,y2])



def tiny_yolo4lite_efficientnet_body(inputs, num_anchors, num_classes, level=0, use_spp=True):
    '''
    Create Tiny YOLO_v4 Lite EfficientNet model CNN body in keras.
    # Arguments
        level: EfficientNet level number.
            by default we use basic EfficientNetB0 as backbone
    '''
    efficientnet, feature_map_info = get_efficientnet_backbone_info(inputs, level=level)

    f1 = efficientnet.get_layer('top_activation').output
    f2 = efficientnet.get_layer('block6a_expand_activation').output
    f1_channel_num = feature_map_info['f1_channel_num']
    f2_channel_num = feature_map_info['f2_channel_num']

    #feature map 1 head (13 x 13 x (f1_channel_num//2) for 416 input)
    x1 = DarknetConv2D_BN_Leaky(f1_channel_num//2, (1,1))(f1)
    if use_spp:
        x1 = Spp_Conv2D_BN_Leaky(x1, f1_channel_num//2)

    #upsample fpn merge for feature map 1 & 2
    x1_upsample = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x1)
    x2 = compose(
            Concatenate(),
            #DarknetConv2D_BN_Leaky(f2_channel_num, (3,3)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=f2_channel_num, kernel_size=(3, 3), block_id_str='8'))([x1_upsample, f2])

    #feature map 2 output (26 x 26 x f2_channel_num for 416 input)
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(x2)

    #downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
            ZeroPadding2D(((1,0),(1,0))),
            #DarknetConv2D_BN_Leaky(f1_channel_num//2, (3,3), strides=(2,2)),
            Darknet_Depthwise_Separable_Conv2D_BN_Leaky(f1_channel_num//2, (3,3), strides=(2,2), block_id_str='9'))(x2)
    x1 = compose(
            Concatenate(),
            #DarknetConv2D_BN_Leaky(f1_channel_num, (3,3)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=f1_channel_num, kernel_size=(3, 3), block_id_str='10'))([x2_downsample, x1])

    #feature map 1 output (13 x 13 x f1_channel_num for 416 input)
    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(x1)

    return Model(inputs, [y1,y2])

