"""YOLO_v3 Model Defined in Keras."""

import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import Xception
from yolo3.utils import compose
from yolo3.models.layers import DarknetConv2D, DarknetConv2D_BN_Leaky, Depthwise_Separable_Conv2D, make_last_layers, make_depthwise_separable_last_layers


def yolo_xception_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 Xception model CNN body in Keras."""
    xception = Xception(input_tensor=inputs, weights='imagenet', include_top=False)

    # input: 416 x 416 x 3
    # block14_sepconv2_act: 13 x 13 x 2048
    # block13_sepconv2_bn(middle in block13): 26 x 26 x 1024
    # add_46(end of block12): 26 x 26 x 728
    # block4_sepconv2_bn(middle in block4) : 52 x 52 x 728
    # add_37(end of block3) : 52 x 52 x 256

    f1 = xception.get_layer('block14_sepconv2_act').output
    # f1 :13 x 13 x 2048
    x, y1 = make_last_layers(f1, 1024, num_anchors * (num_classes + 5))

    x = compose(
            DarknetConv2D_BN_Leaky(512, (1,1)),
            UpSampling2D(2))(x)

    f2 = xception.get_layer('block13_sepconv2_bn').output
    # f2: 26 x 26 x 1024
    x = Concatenate()([x,f2])

    x, y2 = make_last_layers(x, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)

    f3 = xception.get_layer('block4_sepconv2_bn').output
    # f3 : 52 x 52 x 728
    x = Concatenate()([x, f3])
    x, y3 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yololite_xception_body(inputs, num_anchors, num_classes):
    '''Create YOLO_v3 Lite MobileNet model CNN body in keras.'''
    xception = Xception(input_tensor=inputs, weights='imagenet', include_top=False)

    # input: 416 x 416 x 3
    # block14_sepconv2_act: 13 x 13 x 2048
    # block13_sepconv2_bn(middle in block13): 26 x 26 x 1024
    # add_46(end of block12): 26 x 26 x 728
    # block4_sepconv2_bn(middle in block4) : 52 x 52 x 728
    # add_37(end of block3) : 52 x 52 x 256

    f1 = xception.get_layer('block14_sepconv2_act').output
    # f1 :13 x 13 x 2048
    x, y1 = make_depthwise_separable_last_layers(f1, 1024, num_anchors * (num_classes + 5), block_id_str='14')

    x = compose(
            DarknetConv2D_BN_Leaky(512, (1,1)),
            UpSampling2D(2))(x)

    f2 = xception.get_layer('block13_sepconv2_bn').output
    # f2: 26 x 26 x 1024
    x = Concatenate()([x,f2])

    x, y2 = make_depthwise_separable_last_layers(x, 512, num_anchors * (num_classes + 5), block_id_str='15')

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)

    f3 = xception.get_layer('block4_sepconv2_bn').output
    # f3 : 52 x 52 x 728
    x = Concatenate()([x, f3])
    x, y3 = make_depthwise_separable_last_layers(x, 256, num_anchors * (num_classes + 5), block_id_str='16')

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def tiny_yolo_xception_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 Xception model CNN body in keras.'''
    xception = Xception(input_tensor=inputs, weights='imagenet', include_top=False)

    # input: 416 x 416 x 3
    # block14_sepconv2_act: 13 x 13 x 2048
    # block13_sepconv2_bn(middle in block13): 26 x 26 x 1024
    # add_46(end of block12): 26 x 26 x 728
    # block4_sepconv2_bn(middle in block4) : 52 x 52 x 728
    # add_37(end of block3) : 52 x 52 x 256

    x1 = xception.get_layer('block13_sepconv2_bn').output
    # x1 :26 x 26 x 1024
    x2 = xception.get_layer('block14_sepconv2_act').output
    # x2 :13 x 13 x 2048
    x2 = DarknetConv2D_BN_Leaky(1024, (1,1))(x2)

    y1 = compose(
            DarknetConv2D_BN_Leaky(2048, (3,3)),
            #Depthwise_Separable_Conv2D(filters=2048, kernel_size=(3, 3), block_id_str='14'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(512, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            #Depthwise_Separable_Conv2D(filters=1024, kernel_size=(3, 3), block_id_str='15'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])


def tiny_yololite_xception_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 Lite Xception model CNN body in keras.'''
    xception = Xception(input_tensor=inputs, weights='imagenet', include_top=False)

    # input: 416 x 416 x 3
    # block14_sepconv2_act: 13 x 13 x 2048
    # block13_sepconv2_bn(middle in block13): 26 x 26 x 1024
    # add_46(end of block12): 26 x 26 x 728
    # block4_sepconv2_bn(middle in block4) : 52 x 52 x 728
    # add_37(end of block3) : 52 x 52 x 256

    x1 = xception.get_layer('block13_sepconv2_bn').output
    # x1 :26 x 26 x 1024
    x2 = xception.get_layer('block14_sepconv2_act').output
    # x2 :13 x 13 x 2048
    x2 = DarknetConv2D_BN_Leaky(1024, (1,1))(x2)

    y1 = compose(
            #DarknetConv2D_BN_Leaky(2048, (3,3)),
            Depthwise_Separable_Conv2D(filters=2048, kernel_size=(3, 3), block_id_str='14'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(512, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            #DarknetConv2D_BN_Leaky(1024, (3,3)),
            Depthwise_Separable_Conv2D(filters=1024, kernel_size=(3, 3), block_id_str='15'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])

